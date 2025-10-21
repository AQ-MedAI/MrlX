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

"""Main agent multi-turn dialogue implementation."""

import logging
import uuid
from typing import Any, Callable, Dict

from slime.utils.http_utils import post
from slime.utils.types import Sample

from agent_utils.database_utils import commit_subagent_data
from agent_utils.sub_agent_tools import MultiTurnConfig, MockRequest, SubAgentActionProcessor
from main_agent_reward import compute_score_em


logger = logging.getLogger(__name__)


class SGLangClient:
    """SGLang client for LLM generation."""

    async def generate_response(self, prompt: str, sampling_params: Dict[str, Any], args: Any) -> Dict[str, Any]:
        """Generate response using SGLang router.

        Args:
            prompt: Input prompt string.
            sampling_params: Sampling parameters dictionary.
            args: Arguments containing router IP and port.

        Returns:
            Response dictionary from the model.
        """
        url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
        payload = {
            "text": prompt,
            "sampling_params": sampling_params,
        }
        return await post(url, payload)


async def multi_turn_generate(
    args: Any,
    sample: Sample,
    sampling_params: Dict[str, Any],
    config: MultiTurnConfig,
    action_processor: SubAgentActionProcessor,
    llm_client: SGLangClient,
    tokenizer: Callable
) -> Sample:
    """Generic multi-turn generation function.

    Args:
        args: Arguments containing configuration.
        sample: Sample object to process.
        sampling_params: Sampling parameters for generation.
        config: Multi-turn configuration.
        action_processor: Action processor for handling tool calls.
        llm_client: LLM client for generation.
        tokenizer: Tokenizer function.

    Returns:
        Processed sample with response.
    """
    tool_map = await action_processor.init_tools(sampling_params)

    question = sample.prompt
    sample.metadata["question"] = question
    _req = MockRequest(
        initial_question=question if isinstance(question, str) else "",
        tokenizer=tokenizer,
    )

    prompt_str = ""

    for msg in _req.messages:
        prompt_str += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
    prompt_tokens_ids = tokenizer(prompt_str, add_special_tokens=False)["input_ids"]

    if len(prompt_str) > 0:
        sample.prompt = prompt_str

    browser_res_list = []

    for turn in range(config.max_turns):
        prompt_parts = []

        if len(_req.input_ids) >= config.max_model_len - 1536:
            break

        for msg in _req.messages:
            prompt_parts.append(f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>")

        current_prompt = "\n".join(prompt_parts) + "\n<|im_start|>assistant\n"

        output = await llm_client.generate_response(current_prompt, sampling_params, args)

        if output["meta_info"]["finish_reason"]["type"] == "abort":
            sample.status = Sample.Status.ABORTED
            return sample

        cur_response = output["text"]

        if "<|im_end|>" in cur_response:
            cur_response = cur_response.split("<|im_end|>")[0]
        if "<|im_start|>" in cur_response:
            cur_response = cur_response.split("<|im_start|>")[0]
        cur_response = cur_response.strip()

        _req.add_assistant_message(cur_response)

        if len(_req.input_ids) >= config.max_model_len - 100:
            break

        if output["meta_info"]["finish_reason"]["type"] == "length":
            break

        if _req.messages[-1]["role"] == 'assistant':
            should_continue, reason, res_dic = await action_processor.execute_user_turn(_req, tool_map, config)

            if res_dic:
                browser_res_list.append(res_dic)
            if reason:
                finish_reason_type = reason
                break

            if not should_continue:
                break

    response_token_ids = _req.input_ids[len(prompt_tokens_ids):]

    loss_masks = []

    for msg in _req.messages[2:]:
        turn_str = f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
        turn_tokens = tokenizer(turn_str, add_special_tokens=False)["input_ids"]

        if msg['role'] == 'assistant':
            loss_masks.extend([1] * len(turn_tokens))
        else:
            loss_masks.extend([0] * len(turn_tokens))

    full_response_parts = []
    for msg in _req.messages[2:]:
        full_response_parts.append(f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>")
    full_response = "\n".join(full_response_parts)

    assert len(loss_masks) == len(response_token_ids) == (len(_req.input_ids) - len(prompt_tokens_ids)), \
        f"Length mismatch: len(loss_masks)={len(loss_masks)}, len(response_token_ids)={len(response_token_ids)}, calculated response length (total-prompt)={len(_req.input_ids) - len(prompt_tokens_ids)} [Total len={len(_req.input_ids)}, Prompt len={len(prompt_tokens_ids)}]"

    sample.tokens = _req.input_ids
    sample.response_length = len(response_token_ids)
    sample.response = full_response
    sample.loss_mask = loss_masks
    sample.metadata["message"] = _req.messages
    sample.metadata['sub_agent_traj'] = browser_res_list

    sample.status = Sample.Status.COMPLETED

    return sample


async def generate(args, sample: Sample, sampling_params) -> Sample:
    """Main generation function interface.

    Args:
        args: Arguments object containing configuration.
        sample: Sample object to process.
        sampling_params: Sampling parameters for generation.

    Returns:
        Completed sample with response.
    """
    from slime.rollout.sglang_rollout import GenerateState

    config = MultiTurnConfig({
        "max_turns": 20,
        "max_model_len": getattr(args, 'max_model_len', 32768),
    })

    state = GenerateState(args)
    action_processor = SubAgentActionProcessor()
    llm_client = SGLangClient()

    completed_sample = await multi_turn_generate(
        args=args,
        sample=sample,
        sampling_params=sampling_params,
        config=config,
        action_processor=action_processor,
        llm_client=llm_client,
        tokenizer=state.tokenizer
    )

    await action_processor.cleanup_tools()

    return completed_sample


async def reward_func(args, sample, **kwargs):
    """Compute reward for main agent sample.

    Args:
        args: Arguments object.
        sample: Sample object to evaluate.
        **kwargs: Additional arguments.

    Returns:
        Computed reward score.
    """
    if not isinstance(sample, Sample):
        raise TypeError("Sample must be an instance of Sample class.")

    score = await compute_score_em(
        solution_str=sample.prompt + sample.response,
        question=sample.metadata["question"],
        ground_truth=sample.label,
        format_score=0.1,
    )

    for browser_sample in sample.metadata['sub_agent_traj']:
        task_id = str(uuid.uuid4())
        commit_subagent_data(task_id, browser_sample, score)
    del sample.metadata['sub_agent_traj']
    return score
