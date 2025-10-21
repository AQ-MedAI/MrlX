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

"""Main agent reward computation with format validation and exact match checking."""

import json
import logging
import os
import random
import re
import string
import time
import uuid
from typing import Any, Dict

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


logger = logging.getLogger(__name__)


JUDGE_PROMPT = """You are an evaluation assistant. Please determine if the predicted answer is equivalent to the labeled answer.

Question: {question}

Labeled Answer: {correct_answer}

Predicted Answer: {response}

Did the model give an answer **equivalent** to the labeled answer? Please respond with "Correct" if they are equivalent, or "Incorrect" if they are not equivalent. Do not include any other text.
"""

DEFAULT_TIMEOUT = 30
MAX_RETRIES = 3
INITIAL_RETRY_DELAY = 1

def normalize_answer(s):
    """Normalize answer text for comparison."""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def em_check(prediction, golden_answers):
    """Exact match check between prediction and golden answers.

    Args:
        prediction: Predicted answer string.
        golden_answers: Ground truth answer(s), can be string or list.

    Returns:
        Score of 1 if match found, 0 otherwise.
    """
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer == normalized_prediction:
            score = 1
            break
    return score


def _validate_assistant_turn(content: str) -> tuple[bool, str]:
    """Validate single assistant turn format using state machine logic.

    Args:
        content: Assistant turn content string.

    Returns:
        Tuple of (is_valid, final_state).
    """
    tags_to_check = ["thinking", "tool_call", "tool_response", "result"]
    for tag in tags_to_check:
        opening_count = len(re.findall(f"<{tag}>", content))
        closing_count = len(re.findall(f"</{tag}>", content))
        if opening_count != closing_count:
            return False, f"Tag imbalance in assistant turn: {tag}"

    split_pattern = r"(</?(?:thinking|tool_call|tool_response|result)>)"
    parts = re.split(split_pattern, content)
    state = "start"

    for part in parts:
        if not part.strip():
            continue

        is_tag = re.match(r"</?(?:thinking|tool_call|tool_response|result)>", part)
        if is_tag:
            if part == "<thinking>" and state in ["start", "after_tool_call", "after_tool_response"]:
                state = "in_thinking"
            elif part == "</thinking>" and state == "in_thinking":
                state = "after_thinking"
            elif part == "<tool_call>" and state in ["start", "after_thinking"]:
                state = "in_tool_call"
            elif part == "</tool_call>" and state == "in_tool_call":
                state = "after_tool_call"
            elif part == "<tool_response>" and state == "after_tool_call":
                state = "in_tool_response"
            elif part == "</tool_response>" and state == "in_tool_response":
                state = "after_tool_response"
            elif part == "<result>" and state == "after_thinking":
                state = "in_result"
            elif part == "</result>" and state == "in_result":
                state = "end"
            else:
                return False, f"Assistant turn state error: unexpected tag {part} in state {state}"
        else:
             if state not in ["in_thinking", "in_tool_call", "in_tool_response", "in_result"]:
                 if part.strip():
                     return False, f"Assistant turn state error: unexpected content '{part.strip()[:50]}...' in state {state}"

    if state not in ["after_tool_call", "end"]:
        return False, f"Assistant turn ended in unexpected state: {state}"

    return True, state

def is_valid_multi_turn_sequence(text: str) -> tuple[bool, str]:
    """Check if a complete multi-turn dialogue sequence is valid.

    Args:
        text: Full dialogue text.

    Returns:
        Tuple of (is_valid, message).
    """
    turns = text.split('<|im_start|>')[1:]
    if not turns:
        return False, "Text is empty or missing <|im_start|> markers"

    expected_role = "system"
    last_assistant_state = ""

    for i, turn_text in enumerate(turns):
        match = re.match(r"\s*(system|user|assistant)\s*\n(.*)", turn_text, re.DOTALL)
        if not match:
            return False, f"Turn {i+1}: Cannot parse role and content"

        role, content = match.groups()
        content = content.strip()

        if content.endswith('<|im_end|>'):
            content = content[:-len('<|im_end|>')].strip()

        if role != expected_role and not (expected_role == "system" and role == "user"):
             return False, f"Role order error: Turn {i+1} expected '{expected_role}', got '{role}'"

        if role == 'system':
            expected_role = 'user'

        elif role == 'user':
            expected_role = 'assistant'

        elif role == 'assistant':
            is_valid, final_state = _validate_assistant_turn(content)

            if not is_valid:
                return False, f"Turn {i+1} (assistant) format error: {final_state}"

            last_assistant_state = final_state
            if final_state == 'after_tool_call':
                expected_role = 'user'
            elif final_state == 'end':
                expected_role = 'end'

    if last_assistant_state != 'end':
        return False, f"Dialogue did not end correctly with <result>, last assistant state: {last_assistant_state}"

    return True, "Valid multi-turn dialogue format"


def extract_solution(solution_str):
    """Extract answer from solution string.

    Args:
        solution_str: Solution text containing <result> tags.

    Returns:
        Extracted answer or None if not found properly.
    """
    answer_pattern = r"<result>(.*?)</result>"
    match = re.finditer(answer_pattern, solution_str, re.DOTALL)
    matches = list(match)

    if len(matches) <= 1:
        return None

    return matches[-1].group(1).strip()


async def compute_score_em(
    solution_str,
    question,
    ground_truth,
    format_score=0,
    score=1.0,
):
    """Exact match (EM) scoring function with format validation.

    Args:
        solution_str: Solution text.
        question: Question text.
        ground_truth: Ground truth answer.
        format_score: Score for valid format.
        score: Score for correct answer.

    Returns:
        Computed score.
    """
    is_valid_format, _ = is_valid_multi_turn_sequence(solution_str)

    answer = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 64) == 1

    if do_print:
        logger.info("--------------------------------")
        logger.info(f"Ground truth: {ground_truth}")
        logger.info(f"Extracted answer: {answer}")
        logger.info(f"Solution string: {solution_str}")

    if answer is None:
        if is_valid_format:
            return format_score
        else:
            return 0.0
    else:
        if is_valid_format:
            if em_check(answer, ground_truth):
                return score
            else:
                judge_item = {
                    "question": question,
                    "answer": ground_truth,
                    "prediction": answer,
                }
                judgement_result = call_llm_judge(judge_item, JUDGE_PROMPT)
                llm_judgment = judgement_result.get("judgement", "Error")
                if llm_judgment == "Correct":
                    return score
                return format_score
        else:
            return 0.0

def call_llm_api(api_url: str, api_key: str, prompt: str, model: str = "Kimi-K2-Instruct", timeout: int = DEFAULT_TIMEOUT):
    """Call LLM API using requests with retry logic.

    Args:
        api_url: API base URL.
        api_key: API authentication key.
        prompt: Input prompt.
        model: Model name. Defaults to "Kimi-K2-Instruct".
        timeout: Request timeout. Defaults to DEFAULT_TIMEOUT.

    Returns:
        Tuple of (response_content, error_message).
    """
    request_id = str(uuid.uuid4())
    log_prefix = f"[LLM Request ID: {request_id}] "

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

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(
                f"{api_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=timeout
            )

            if response.status_code in [500, 502, 503, 504]:
                last_error = f"{log_prefix}Server Error ({response.status_code}) on attempt {attempt + 1}/{MAX_RETRIES}"
                if attempt < MAX_RETRIES - 1:
                    delay = INITIAL_RETRY_DELAY * (attempt + 1)
                    time.sleep(delay)
                continue

            response.raise_for_status()
            response_data = response.json()
            return response_data["choices"][0]["message"]["content"], None

        except requests.exceptions.ConnectionError as e:
            last_error = f"{log_prefix}Connection Error: {e}"
            if attempt < MAX_RETRIES - 1:
                delay = INITIAL_RETRY_DELAY * (attempt + 1)
                time.sleep(delay)
            continue
        except requests.exceptions.Timeout as e:
            last_error = f"{log_prefix}Timeout Error: {e}"
            if attempt < MAX_RETRIES - 1:
                delay = INITIAL_RETRY_DELAY * (attempt + 1)
                time.sleep(delay)
            continue
        except requests.exceptions.RequestException as e:
            last_error = f"{log_prefix}Request Error: {e}"
            break
        except (json.JSONDecodeError, KeyError) as e:
            raw_response_text = response.text if "response" in locals() else "N/A"
            last_error = f"{log_prefix}Response Parse Error: {e}, Response: {raw_response_text[:200]}"
            break
        except Exception as e:
            last_error = f"{log_prefix}Unexpected Error: {e}"
            break

    return None, last_error.replace(log_prefix, "API Call Failed: ") if last_error else "API Call Failed after retries"




def extract_final_judgment(llm_response):
    """Extract the final judgment from LLM response.

    Args:
        llm_response: LLM response string.

    Returns:
        Judgment result: "Correct", "Incorrect", or "Error".
    """
    if not llm_response:
        return "Error"

    llm_response = llm_response.strip().lower()
    if "correct" in llm_response and "incorrect" not in llm_response:
        return "Correct"
    elif "incorrect" in llm_response:
        return "Incorrect"
    else:
        return "Error"


def call_llm_judge(item: Dict[str, Any], judge_prompt: str) -> Dict[str, Any]:
    """Judge if predicted answer matches ground truth using LLM.

    Args:
        item: Dictionary with question, answer, and prediction.
        judge_prompt: Prompt template for judgment.

    Returns:
        Dictionary containing judgment result.
    """

    if not REQUESTS_AVAILABLE:
        return {
            "question": item.get("question", ""),
            "answer": item.get("answer", ""),
            "prediction": item.get("prediction", "").strip(),
            "prompt": "",
            "judgement": "Error",
            "llm_response_full": "Requests not available",
            "error": "Requests library not available"
        }

    question = item.get("question", "")
    correct_answer = item.get("answer", "")
    response = item.get("prediction", "").strip()

    # 如果response以<thinking>或"Tool loop detected"开头，直接返回Incorrect
    if response.startswith("<thinking>") or response.startswith("Tool loop detected"):
        reason = "Response starts with <thinking>" if response.startswith("<thinking>") else "Response starts with 'Tool loop detected'"
        return {
            "question": question,
            "answer": correct_answer,
            "prediction": response,
            "prompt": "",
            "judgement": "Incorrect",
            "llm_response_full": f"{reason}, marked as Incorrect"
        }

    try:
        prompt = judge_prompt.format(question=question, correct_answer=correct_answer, response=response)

        api_base = os.getenv("JUDGE_LLM_API_BASE")
        api_key = os.getenv("JUDGE_LLM_API_KEY")
        model = os.getenv("JUDGE_LLM_MODEL")

        if not api_key:
            return {
                "question": question,
                "answer": correct_answer,
                "prediction": response,
                "prompt": prompt,
                "judgement": "Error",
                "llm_response_full": "API key not found",
                "error": "JUDGE_LLM_API_KEY environment variable not set"
            }
        # 使用新的call_llm_api函数
        llm_response, error_msg = call_llm_api(
            api_url=api_base,
            api_key=api_key,
            prompt=prompt,
            model=model,
            timeout=30
        )

        if error_msg:
            return {
                "question": question,
                "answer": correct_answer,
                "prediction": response,
                "prompt": prompt,
                "judgement": "Error",
                "llm_response_full": "",
                "error": error_msg
            }

        judgment = extract_final_judgment(llm_response)

        return {
            "question": question,
            "answer": correct_answer,
            "prediction": response,
            "prompt": prompt,
            "judgement": judgment,
            "llm_response_full": llm_response
        }

    except Exception as e:
        return {
            "question": question,
            "answer": correct_answer,
            "prediction": response,
            "prompt": prompt if 'prompt' in locals() else "",
            "judgement": "Error",
            "llm_response_full": "",
            "error": str(e)
        }
