# Copyright 2025 Ant Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Sub-agent tools for multi-turn dialogue and tool execution."""

import asyncio
import json
import logging
import os
import re
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import aiohttp

from slime.utils.http_utils import post

from .sub_agent_server import agent_search, agent_visit


logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

SEMAPHORE = asyncio.Semaphore(256)


class MultiTurnConfig:
    """Multi-turn dialogue configuration."""

    def __init__(self, config_dict: Dict[str, Any]):
        """Initialize multi-turn configuration.

        Args:
            config_dict: Configuration dictionary with optional keys:
                - max_turns: Maximum dialogue turns
                - max_model_len: Maximum model context length
                - custom_params: Custom parameters for tools and prompts
        """
        self.max_turns = config_dict.get("max_turns", 20)
        self.max_model_len = config_dict.get("max_model_len", 32768)
        if config_dict:
            self.custom_params = config_dict
        else:
            self.custom_params = {
                "sub_max_turns": 20,
                "topk": 10,
                "search_timeout": 60,
                "visit_timeout": 300,
                "early_stop_prompt": """You have now reached the maximum context length you can handle.
You should stop making tool calls and, based on all the information above, \
think again and provide what you consider the most likely answer in the following format:
<thinking>your final thinking</thinking>
<result>your answer</result>""",
            }

class MockRequest:
    """Mock request object for managing dialogue state."""

    def __init__(self, initial_question: str, tokenizer: Callable):
        """Initialize mock request with optional initial question.

        Args:
            initial_question: Initial user question to start the dialogue.
            tokenizer: Tokenizer function for encoding text.
        """
        self.messages = []
        self.input_ids = []
        self.tokenizer = tokenizer

        if len(initial_question) > 0:
            file_dir = os.path.dirname(os.path.abspath(__file__))
            file_path = os.path.join(file_dir, '../main_agent_prompt_template/sys_prompt.txt')
            with open(file_path, 'r', encoding='utf-8') as f:
                system_content = f.read()
            current_date_str = datetime.now().strftime("%Y-%m-%d")
            formatted_content = system_content.replace("{current_date}", current_date_str)
            self.add_message("system", formatted_content)

            file_path = os.path.join(file_dir, '../main_agent_prompt_template/user_prefix.txt')
            with open(file_path, 'r', encoding='utf-8') as f:
                user_prefix = f.read()
            self.add_message("user", user_prefix.format(question=initial_question))

    def add_message(self, role: str, content: str):
        """Add a message with proper formatting and tokenization.

        Args:
            role: Message role (system/user/assistant).
            content: Message content.
        """
        message = {"role": role, "content": content}
        self.messages.append(message)

        turn_str = f"<|im_start|>{role}\n{content}<|im_end|>\n"
        tokens = self.tokenizer(turn_str, add_special_tokens=False)["input_ids"]
        self.input_ids.extend(tokens)

    def add_system_message(self, content: str):
        """Add a system message."""
        self.add_message("system", content)

    def add_assistant_message(self, content: str):
        """Add an assistant message."""
        self.add_message("assistant", content)

    def add_user_message(self, content: str):
        """Add a user message."""
        self.add_message("user", content)

def extract_and_wrap_result(text: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract content from <summary> tags and wrap in <tool_response>.

    Args:
        text: Input string potentially containing <summary> tags.

    Returns:
        Wrapped string if <summary> found, None otherwise.
    """
    pattern = r"<summary>(.*?)</summary>"
    match = re.search(pattern, text, re.DOTALL)

    if match:
        extracted_content = match.group(1)
        new_wrapped_string = f"<tool_response>{extracted_content}</tool_response>"
        return new_wrapped_string

    return None


def format_conversation_for_summary(messages: List[Dict[str, Any]]) -> str:
    """Format conversation messages into a summary-ready string.

    Args:
        messages: List of message dictionaries with 'role' and 'content' keys.

    Returns:
        Formatted conversation string.
    """
    conversation_parts = []
    for message in messages:
        role = message.get("role", "unknown").capitalize()
        content = message.get("content", "")
        formatted_part = f"[{role}]:\n{content}"
        conversation_parts.append(formatted_part)

    return "\n\n".join(conversation_parts)

class SubAgentActionProcessor:
    """Process sub-agent actions and tool calls."""

    def __init__(self):
        """Initialize action processor."""
        self.subagent_tool = None
        self.instance_id = None

    @staticmethod
    def get_tool_name_and_params(llm_output: str) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """Parse LLM output to extract tool name and parameters.

        Expected format: <tool_call>{"name": "tool_name", "arguments": {...}}</tool_call>

        Args:
            llm_output: String output from the language model.

        Returns:
            Tuple of (tool_name, tool_params) or (None, None) if parsing fails.
        """
        tool_call_match = re.search(r"<tool_call>(.*?)</tool_call>", llm_output, re.DOTALL)

        if not tool_call_match:
            return None, None

        json_content = tool_call_match.group(1).strip()

        try:
            data = json.loads(json_content)
            tool_name = data.get("name")
            tool_params = data.get("arguments")

            if isinstance(tool_name, str) and isinstance(tool_params, dict):
                return tool_name, tool_params
            else:
                return None, None

        except json.JSONDecodeError:
            return None, None

    async def execute_user_turn(
        self,
        req: MockRequest,
        tool_map: Dict[str, Any],
        config: MultiTurnConfig
    ) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
        """Execute user simulation turn.

        Args:
            req: Mock request object containing dialogue state.
            tool_map: Dictionary mapping tool names to tool instances.
            config: Multi-turn configuration.

        Returns:
            Tuple of (should_continue, reason, subagent_result_dict).
        """
        result_dict = None
        try:
            if not req.messages or req.messages[-1]["role"] != 'assistant':
                return False, "error", None

            subagent_tool = tool_map["subagent"]
            router_name, router_params = SubAgentActionProcessor.get_tool_name_and_params(req.messages[-1]['content'])

            user_resp = ""
            if router_name == 'reasoner':
                user_resp, err_msg = await subagent_tool.get_reasoning(
                    router_params.get('problem', '')
                )
                user_resp = f'<tool_response>\n{user_resp}\n</tool_response>'

            elif router_name == 'browser':
                result_dict = await subagent_tool.execute(
                    req.tokenizer,
                    router_params,
                    config,
                )
                full_context = format_conversation_for_summary(result_dict.get('messages', []))
                browser_summary, err_msg = await subagent_tool.get_summary(full_context)

                wrapped_summary = extract_and_wrap_result(browser_summary)
                if wrapped_summary:
                    user_resp = wrapped_summary
                else:
                    user_resp = f"<tool_response>{browser_summary}</tool_response>"
            else:
                return False, "error", None

            lookahead_user_turn_str = f"<|im_start|>user\n{user_resp}<|im_end|>\n"
            user_tokens = req.tokenizer(lookahead_user_turn_str, add_special_tokens=False)["input_ids"]

            if len(req.input_ids) + len(user_tokens) >= config.max_model_len - 1:
                return False, "length", None

            req.add_user_message(user_resp)

            if len(req.input_ids) >= config.max_model_len - 1:
                return False, "length", None

            return True, None, result_dict

        except Exception as e:
            logger.error(f"User simulation execution failed: {str(e)}")
            return False, "error", None

    async def init_tools(self, sampling_params: Dict[str, Any]) -> Dict[str, Any]:
        """Initialize sub-agent tools.

        Args:
            sampling_params: Sampling parameters for LLM generation.

        Returns:
            Dictionary mapping tool names to tool instances.
        """
        self.subagent_tool = SubAgentTool(sampling_params)

        return {
            "subagent": self.subagent_tool
        }

    async def cleanup_tools(self) -> None:
        """Clean up tools and release resources."""
        if self.instance_id and self.subagent_tool:
            try:
                await self.subagent_tool.release(self.instance_id)
            except Exception as e:
                logger.warning(f"Failed to clean up subagent instance: {str(e)}")


def _results_to_string(results):
    """Convert retrieval results to formatted string."""
    if isinstance(results, dict):
        return results.get('result', '')

    if isinstance(results, str):
        return results

    if isinstance(results, list):
        if all(isinstance(item, str) for item in results):
            return "\n---\n".join(results)

    return "The tool you are trying to use seems unreachable currently"


async def search_tool(params: dict, config: MultiTurnConfig = None) -> str:
    """Search tool using retrieval service."""
    payload = {"query": params.get('query'), "name": "search"}
    timeout = config.custom_params.get("search_timeout", 60) if config else 60
    results, error_msg = await agent_search(payload, timeout=timeout)
    return _results_to_string(results)


async def visit_tool(params: dict, config: MultiTurnConfig = None) -> str:
    """Visit tool to fetch webpage content."""
    payload = {"url": params.get('url'), "goal": params.get('goal'), "name": "visit"}
    timeout = config.custom_params.get("visit_timeout", 300) if config else 300
    results, error_msg = await agent_visit(payload, timeout=timeout)
    return _results_to_string(results)


def postprocess_responses(resp: str) -> str:
    """Post-process response to ensure complete tags."""
    for tag in ["</tool_call>", "</result>"]:
        if tag in resp:
            return resp.split(tag)[0] + tag
    return resp


def postprocess_predictions(prediction: str):
    """Parse prediction to extract action and content.

    Args:
        prediction: Prediction string containing tags.

    Returns:
        Tuple of (action, content) where action is 'tool_call' or 'result'.
        Returns (None, "") if no match found.
    """
    pattern = r"<(tool_call|result)>(.*?)</\1>"
    match = re.search(pattern, prediction, re.DOTALL)

    if match:
        action = match.group(1)
        content = match.group(2).strip()
    else:
        action = None
        content = ""

    return action, content


def get_tool_and_params(content: str) -> Optional[Tuple[str, dict]]:
    """Parse JSON string to extract tool name and arguments.

    Args:
        content: Input JSON string.

    Returns:
        Tuple of (tool_name, arguments) on success, None on failure.
    """
    try:
        content_dict = json.loads(content)
        tool_name = content_dict.get('name')
        arguments = content_dict.get('arguments')

        if tool_name not in ['search', 'visit'] or not arguments:
            return None

        return tool_name, arguments

    except (json.JSONDecodeError, AttributeError):
        return None

async def execute_predictions(prediction: str, config: MultiTurnConfig = None) -> tuple[str, bool]:
    """Execute predicted action.

    Args:
        prediction: Prediction string with action tags.
        config: Multi-turn configuration.

    Returns:
        Tuple of (next_observation, done_flag).
    """
    action, content = postprocess_predictions(prediction)

    if action == "tool_call":
        result = get_tool_and_params(content=content)
        if result:
            tool_name, params = result
            if tool_name == 'search':
                async with SEMAPHORE:
                    search_results = await search_tool(params, config)
                next_obs = f"<tool_response>\n{search_results.strip()}\n</tool_response>"
            elif tool_name == 'visit':
                async with SEMAPHORE:
                    visit_results = await visit_tool(params, config)
                next_obs = f"<tool_response>\n{visit_results.strip()}\n</tool_response>"
            else:
                next_obs = ""
            done = False
        else:
            next_obs = ""
            done = False
    elif action == "result":
        next_obs = ""
        done = True
    else:
        next_obs = ""
        done = False

    return next_obs, done


def _detect_language(text: str) -> str:
    """Detect language of text (Chinese or English)."""
    chinese_chars = len([c for c in text if '\u4e00' <= c <= '\u9fff'])
    total_chars = len([c for c in text if c.isalnum()])

    if total_chars == 0:
        return 'en'

    chinese_ratio = chinese_chars / total_chars
    return 'zh' if chinese_ratio > 0.3 else 'en'


def _get_early_stop_prompt(config: MultiTurnConfig = None) -> str:
    """Get early stop prompt text."""
    default_prompt = """You have now reached the maximum context length you can handle.
You should stop making tool calls and, based on all the information above, \
think again and provide what you consider the most likely answer in the following format:
<thinking>your final thinking</thinking>
<result>your answer</result>"""

    if config and config.custom_params:
        return config.custom_params.get('early_stop_prompt', default_prompt)
    return default_prompt

class SubAgentTool:
    """Sub-agent tool for executing sub-tasks."""

    def __init__(self, sampling_params: Dict[str, Any]):
        """Initialize sub-agent tool.

        Args:
            sampling_params: Sampling parameters for LLM generation.
        """
        self.sampling_params = sampling_params

        file_dir = os.path.dirname(os.path.abspath(__file__))

        file_path = os.path.join(file_dir, '../sub_agent_prompt_template/sys_prompt.txt')
        with open(file_path, 'r', encoding='utf-8') as f:
            system_content = f.read()
        current_date_str = datetime.now().strftime("%Y-%m-%d")
        self.subagent_prompt = system_content.replace("{current_date}", current_date_str)

        file_path = os.path.join(file_dir, '../sub_agent_prompt_template/user_prefix.txt')
        with open(file_path, 'r', encoding='utf-8') as f:
            user_prefix = f.read()
        self.user_prefix = user_prefix

        file_path = os.path.join(file_dir, '../sub_agent_prompt_template/summary_template_en.txt')
        with open(file_path, 'r', encoding='utf-8') as f:
            summary_prompt_en = f.read()
        self.summary_prompt_en = summary_prompt_en

        file_path = os.path.join(file_dir, '../sub_agent_prompt_template/summary_template_zh.txt')
        with open(file_path, 'r', encoding='utf-8') as f:
            summary_prompt_zh = f.read()
        self.summary_prompt_zh = summary_prompt_zh

        file_path = os.path.join(file_dir, '../sub_agent_prompt_template/reasoner_template_zh.txt')
        with open(file_path, 'r', encoding='utf-8') as f:
            reasoner_prompt_zh = f.read()
        self.reasoner_prompt_zh = reasoner_prompt_zh

        self._instance_dict = {}

    async def call_agent_model(self, prompt, max_retry=100):
        """Call agent model with retry logic.

        Args:
            prompt: Input prompt for the model.
            max_retry: Maximum number of retries.

        Returns:
            Model output dictionary.
        """
        payload = {
            "text": prompt,
            "sampling_params": self.sampling_params,
        }
        subagent_ip = os.getenv("SUB_AGENT_IP")
        if not subagent_ip:
            raise RuntimeError("SUB_AGENT_IP environment variable is not set")
        url = f"http://{subagent_ip}:3333/generate"

        while max_retry > 0:
            try:
                output = await post(url, payload)
                return output
            except Exception as e:
                logger.warning(f"Sub-agent call failed: {str(e)}, retries remaining: {max_retry-1}")
                max_retry -= 1
                time.sleep(10)

        logger.error("All sub-agent server calls failed")
        return "Sorry, sub-agent is temporarily unable to answer this question."

    async def execute(
        self,
        tokenizer: Callable,
        parameters: dict,
        config: MultiTurnConfig = None,
        **kwargs
    ) -> Tuple[str, float, dict]:
        """Execute multi-turn dialogue with sub-agent.

        Args:
            tokenizer: Tokenizer function.
            parameters: Parameters containing task and context.
            config: Multi-turn configuration.
            **kwargs: Additional arguments.

        Returns:
            Result dictionary containing tokens, response, and loss mask.
        """
        task_for_sub_agent = parameters.get('task', '')
        context_for_sub_agent = parameters.get('context', '')

        sub_req = MockRequest(initial_question='', tokenizer=tokenizer)
        sub_req.add_system_message(self.subagent_prompt)
        sub_req.add_user_message(self.user_prefix.format(task=task_for_sub_agent, context=context_for_sub_agent))

        prompt_str = ""
        for msg in sub_req.messages:
            prompt_str += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
        prompt_tokens_ids = tokenizer(prompt_str, add_special_tokens=False)["input_ids"]

        response_token_ids = []
        loss_masks = []

        max_turns = config.max_turns if config else 20
        max_model_len = config.max_model_len if config else 32768

        for turn in range(max_turns):
            prompt_parts = []

            for msg in sub_req.messages:
                prompt_parts.append(f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>")

            current_prompt = "\n".join(prompt_parts) + "\n<|im_start|>assistant\n"

            output = await self.call_agent_model(current_prompt)

            cur_response = output["text"]

            if "<|im_end|>" in cur_response:
                cur_response = cur_response.split("<|im_end|>")[0]
            if "<|im_start|>" in cur_response:
                cur_response = cur_response.split("<|im_start|>")[0]
            cur_response = cur_response.strip()

            sub_req.add_assistant_message(cur_response)

            if len(sub_req.input_ids) >= max_model_len - 100:
                break

            if output["meta_info"]["finish_reason"]["type"] == "length":
                break

            if sub_req.messages[-1]["role"] == 'assistant':
                next_obs, done = await execute_predictions(cur_response, config)

                if done:
                    break

                if len(next_obs) > 0:
                    lookahead_user_turn_str = f"<|im_start|>user\n{next_obs}<|im_end|>\n"
                    lookahead_tokens = tokenizer(lookahead_user_turn_str, add_special_tokens=False)["input_ids"]

                    if len(sub_req.input_ids) + len(lookahead_tokens) >= max_model_len - 200:
                        final_obs_content = _get_early_stop_prompt(config)
                    else:
                        final_obs_content = next_obs

                    sub_req.add_user_message(final_obs_content)

                else:
                    break

        response_token_ids = sub_req.input_ids[len(prompt_tokens_ids):]

        for msg in sub_req.messages[2:]:
            turn_str = f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
            turn_tokens = tokenizer(turn_str, add_special_tokens=False)["input_ids"]

            if msg['role'] == 'assistant':
                loss_masks.extend([1] * len(turn_tokens))
            else:
                loss_masks.extend([0] * len(turn_tokens))

        full_response_parts = []
        for msg in sub_req.messages[2:]:
            full_response_parts.append(f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>")
        full_response = "\n".join(full_response_parts)

        assert len(loss_masks) == len(response_token_ids) == (len(sub_req.input_ids) - len(prompt_tokens_ids)), \
            f"Length mismatch: len(loss_masks)={len(loss_masks)}, len(response_token_ids)={len(response_token_ids)}, calculated response length (total-prompt)={len(sub_req.input_ids) - len(prompt_tokens_ids)} [Total len={len(sub_req.input_ids)}, Prompt len={len(prompt_tokens_ids)}]"

        result_dict = {
            "tokens": sub_req.input_ids,
            "response_length": len(response_token_ids),
            "response": full_response,
            "loss_mask": loss_masks,
            "messages": sub_req.messages,
        }

        return result_dict

    async def release(self, instance_id: str, **kwargs) -> None:
        """Release instance resources."""
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]

    async def get_summary(
        self,
        full_context: str,
        **kwargs
    ) -> Tuple[str, float, dict]:
        """Get summary of dialogue context using LLM.

        Args:
            full_context: Full dialogue context string.
            **kwargs: Additional arguments.

        Returns:
            Tuple of (llm_response, error_message).
        """
        language = _detect_language(full_context)
        prompt = self.summary_prompt_zh if language == 'zh' else self.summary_prompt_en
        api_base = os.environ.get('SUMMARY_LLM_API_BASE')
        model = os.environ.get('SUMMARY_LLM_MODEL')

        api_key = os.environ.get('SUMMARY_LLM_API_KEY')
        if not api_key:
            logger.error("SUMMARY_LLM_API_KEY environment variable not found")
            return "Error: Missing required API key", "Missing API key"

        llm_response, error_msg = await self.call_llm_api(
            api_url=api_base,
            api_key=api_key,
            prompt=prompt.replace("{action_log_string}", full_context),
            model=model,
            timeout=30
        )

        return llm_response, error_msg

    async def get_reasoning(
        self,
        full_context: str,
        **kwargs
    ) -> Tuple[str, float, dict]:
        """Get reasoning for a problem using LLM.

        Args:
            full_context: Problem description.
            **kwargs: Additional arguments.

        Returns:
            Tuple of (llm_response, error_message).
        """
        prompt = self.reasoner_prompt_zh
        api_base = os.environ.get('REASONER_LLM_API_BASE')
        model = os.environ.get('REASONER_LLM_MODEL')

        api_key = os.environ.get('REASONER_LLM_API_KEY')

        if not api_key:
            logger.error("REASONER_LLM_API_KEY environment variable not found")
            return "Error: Missing required API key", "Missing API key"

        llm_response, error_msg = await self.call_llm_api(
            api_url=api_base,
            api_key=api_key,
            prompt=prompt.format(problem=full_context),
            model=model,
            timeout=30
        )

        return llm_response, error_msg

    async def call_llm_api(
        self,
        api_url: str,
        api_key: str,
        prompt: str,
        model: str = "Kimi-K2-Instruct",
        timeout: int = 120
    ) -> Tuple[Optional[str], Optional[str]]:
        """Call LLM API asynchronously with retry logic.

        Args:
            api_url: API base URL.
            api_key: API authentication key.
            prompt: Input prompt for the LLM.
            model: Model name. Defaults to "Kimi-K2-Instruct".
            timeout: Request timeout in seconds. Defaults to 120.

        Returns:
            Tuple of (response_content, error_message).
        """
        MAX_RETRIES = 5
        INITIAL_RETRY_DELAY = 3

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 2048,
            "temperature": 0.1
        }

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }

        last_error = None
        timeout_config = aiohttp.ClientTimeout(total=timeout)

        for attempt in range(MAX_RETRIES):
            try:
                async with aiohttp.ClientSession(timeout=timeout_config) as session:
                    async with session.post(
                        f"{api_url}/chat/completions",
                        headers=headers,
                        json=payload
                    ) as response:

                        if response.status in [500, 502, 503, 504]:
                            last_error = f"Server Error ({response.status}) on attempt {attempt + 1}/{MAX_RETRIES}"
                            if attempt < MAX_RETRIES - 1:
                                delay = INITIAL_RETRY_DELAY * (2 ** attempt)
                                await asyncio.sleep(delay)
                            continue

                        response.raise_for_status()
                        response_data = await response.json()
                        return response_data["choices"][0]["message"]["content"], None

            except aiohttp.ClientConnectionError as e:
                last_error = f"Connection Error: {e}"
            except asyncio.TimeoutError as e:
                last_error = f"Timeout Error: {e}"
            except aiohttp.ClientError as e:
                last_error = f"Request Error: {e}"
                break
            except (json.JSONDecodeError, KeyError) as e:
                raw_response_text = await response.text() if 'response' in locals() else "N/A"
                last_error = f"Response Parse Error: {e}, Response: {raw_response_text[:200]}"
                break
            except Exception as e:
                last_error = f"Unexpected Error: {e}"
                break

            if attempt < MAX_RETRIES - 1 and last_error:
                delay = INITIAL_RETRY_DELAY * (2 ** attempt)
                await asyncio.sleep(delay)

        final_error_msg = last_error if last_error else "API Call Failed after all retries"
        return None, final_error_msg
