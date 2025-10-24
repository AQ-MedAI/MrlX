"""
Module: doctor_multiturn
------------------------
Handles multi-turn conversation generation for the doctor role in the
doctor-patient simulation, using SGLang-based LLM client.

Functions:
    multi_turn_generate(...) -> Sample
    generate(...) -> Sample
    reward_func(...) -> float
"""

import logging
from math import e
from typing import Any, Dict, Callable
import uuid
from pathlib import Path
import re, json

logger = logging.getLogger(__name__)

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.types import Sample

from config import global_config

# Load grader template from txt file
_template_path = Path(__file__).parent / "grader_template.txt"
with open(_template_path, "r", encoding="utf-8") as f:
    GRADER_TEMPLATE = f.read()


# SGLang Client Implementation
class SGLangClient:
    """SGLang Client Implementation."""

    async def generate_response(
        self, prompt: str, sampling_params: Dict[str, Any], args: Any
    ) -> Dict[str, Any]:
        """
        Send generation request to SGLang server.

        Args:
            prompt (str): Input text prompt.
            sampling_params (dict): Sampling configuration for LLM.
            args: Arguments object containing IP/port information.

        Returns:
            dict: Response JSON from SGLang server.
        """
        url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
        payload = {
            "text": prompt,
            "sampling_params": sampling_params,
        }
        return await post(url, payload)


async def generate(args, sample: Sample, sampling_params) -> Sample:
    """
    Simplified generate function interface.

    Args:
        args: Original arguments object.
        sample: Sample object.
        sampling_params: Sampling parameters.
    """
    from slime.rollout.sglang_rollout import GenerateState

    assert (
        sample.status == Sample.Status.PENDING or sample.status == Sample.Status.ABORTED
    ), f"Sample status is {sample.status}"

    # Create components
    state = GenerateState(args)
    llm_client = SGLangClient()

    messages = sample.prompt  # this is a list

    sample.prompt = state.tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    if not sample.tokens:
        sample.tokens = state.tokenizer(sample.prompt, add_special_tokens=False)[
            "input_ids"
        ]

    output = await llm_client.generate_response(sample.prompt, sampling_params, args)

    assert "output_token_logprobs" in output["meta_info"]

    new_response_tokens = [
        item[1] for item in output["meta_info"]["output_token_logprobs"]
    ]
    new_response_log_probs = [
        item[0] for item in output["meta_info"]["output_token_logprobs"]
    ]

    sample.tokens = sample.tokens + new_response_tokens
    sample.response_length += len(new_response_tokens)
    sample.response += output["text"]

    messages.append({"role": "assistant", "content": output["text"]})
    sample.metadata["messages"] = messages

    if sample.rollout_log_probs is None:
        sample.rollout_log_probs = []
    sample.rollout_log_probs += new_response_log_probs

    if "weight_version" in output["meta_info"]:
        sample.weight_versions.append(output["meta_info"]["weight_version"])

    match output["meta_info"]["finish_reason"]["type"]:
        case "length":
            sample.status = Sample.Status.TRUNCATED
        case "abort":
            sample.status = Sample.Status.ABORTED
        case "stop":
            sample.status = Sample.Status.COMPLETED

    return sample


def parse_json_to_dict(json_string: str) -> dict:
    try:
        # Keep only the markdown-style json block if present
        match = re.search(r"```json\s*([\s\S]*?)\s*```", json_string.strip())
        if match:
            json_string = match.group(1)
        else:
            json_string = json_string.strip()

        # Remove markdown-style ```json``` markers if present
        json_cleaned = re.sub(r"^```json\s*|\s*```$", "", json_string)
    except AttributeError:
        print(f"JSON extraction failed: {json_string}")
        return {}

    try:
        return json.loads(json_cleaned)
    except json.JSONDecodeError as e:
        print(f"JSON decoding failed {e}: {json_cleaned}")
        # Match key fields directly to avoid JSON string quote escaping issues
        result = {}

        # Extract "criteria_met": true/false
        criteria_match = re.search(
            r'"criteria_met":\s*(true|false)', json_cleaned, re.IGNORECASE
        )
        if criteria_match:
            result["criteria_met"] = criteria_match.group(1).lower() == "true"

        # Extract the "explanation" string content
        explanation_match = re.search(
            r'"explanation":\s*"([^"]*(?:\\.[^"]*)*)"', json_cleaned
        )
        if explanation_match:
            # Handle escaped characters
            explanation = (
                explanation_match.group(1).replace('\\"', '"').replace("\\\\", "\\")
            )
            result["explanation"] = explanation

        return result


def extract_grader_output(solution_text: str):
    if "</think>" in solution_text:
        solution_text = solution_text.split("</think>")[-1].strip()
    json_cleaned = parse_json_to_dict(solution_text)
    return json_cleaned.get("criteria_met", None)


async def reward_func(args, sample, **kwargs):
    """
    Compute reward score for doctor simulation.

    Returns:
        float: Total reward score.
    """
    if not isinstance(sample, Sample):
        raise TypeError("Sample must be an instance of Sample class.")

    rubrics = sample.metadata.get("rubrics", [])
    messages = sample.metadata.get("messages", [])
    conversation = ""
    for message in messages:
        conversation += f"{message['role']}: {message['content']}\n"

    llm_client = SGLangClient()

    grading_sampling_params = dict(
        temperature=1.0,
        top_p=0.95,
        top_k=20,
        max_new_tokens=16384,
    )

    # Pre-calculate possible positive score and prepare all prompts
    posible_positive_score = sum(
        rubric["points"] for rubric in rubrics if rubric["points"] > 0
    )

    # Pre-replace conversation once to avoid repeated operations
    base_grader_prompt = GRADER_TEMPLATE.replace("<<conversation>>", conversation)

    # Prepare all grading tasks for parallel execution
    async def grade_rubric(rubric: dict) -> tuple[int, bool]:
        """Grade a single rubric criterion and return (points, met)."""
        grader_prompt = base_grader_prompt.replace("<<rubric_item>>", rubric["criterion"])
        result = await llm_client.generate_response(
            grader_prompt, grading_sampling_params, args
        )
        criterion_met = extract_grader_output(result["text"])
        return rubric["points"], bool(criterion_met)

    # Execute all grading tasks concurrently
    import asyncio
    grading_results = await asyncio.gather(
        *[grade_rubric(rubric) for rubric in rubrics]
    )

    # Calculate final score
    score = sum(points for points, met in grading_results if met)

    return score / posible_positive_score
