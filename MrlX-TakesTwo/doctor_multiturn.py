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

import os
import logging
from typing import Any, Dict, Callable
import uuid

logger = logging.getLogger(__name__)

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.types import Sample

from config import global_config
from utils.database_utils import commit_patient_data
from utils.patient_sample_converter import change_patient_tool_to_sample
from utils.patient_tools import (
    PatientSimulatorActionProcessor,
    MockRequest,
)
from utils.chat_template import TEMPLATE_QWEN3_ASSISTANT_MASK
from doctor_reward import compute_score


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


async def multi_turn_generate(
    args: Any,
    sample: Sample,
    sampling_params: Dict[str, Any],
    action_processor: PatientSimulatorActionProcessor,
    llm_client: SGLangClient,
    tokenizer: Callable,
) -> Sample:
    """
    Generic multi-turn doctor-patient conversation generator.

    Handles:
        - Applying chat template to doctor prompts
        - Multi-turn loop with patient simulation
        - Token length checks, completion rules

    Returns:
        Sample: Updated Sample with generated conversation and metadata.
    """

    # Extract extra_info
    extra_info = sample.metadata.get("extra_info", {})
    solution = extra_info.get("solution", {})
    reward_model = extra_info.get("reward_model", "grm")
    chief_complaint = extra_info.get("chief_complaint", "")
    self_report = extra_info.get("self_report", "")
    messages = [i for i in sample.prompt]  # this is a list

    sample.prompt = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=True,
        keep_thinking=True,
    )

    # Store solution and reward_model in metadata for later use
    if solution:
        sample.metadata["solution"] = solution
    if reward_model:
        sample.metadata["reward_model"] = reward_model
    if chief_complaint:
        sample.metadata["chief_complaint"] = chief_complaint
    if self_report:
        sample.metadata["self_report"] = self_report

    # Initialize tools
    tool_map = await action_processor.init_tools(sample, tokenizer)

    # Parse tools_kwargs
    tools_kwargs = {}
    if isinstance(sample.metadata, dict):
        tools_kwargs = sample.metadata.get("tools_kwargs", {})

    # Create request object
    _req = MockRequest(
        tokenizer=tokenizer,
        tools_kwargs=tools_kwargs,
    )

    _req.add_messages(messages)

    finish_reason_type = None

    # Multi-turn loop
    for _ in range(global_config.MAX_TURNS):
        # Generate response using apply_chat_template
        current_prompt = tokenizer.apply_chat_template(
            _req.messages,
            add_generation_prompt=True,
            tokenize=False,
            keep_thinking=True,
            enable_thinking=True,
        )

        output = await llm_client.generate_response(
            current_prompt, sampling_params, args
        )

        # Check abort condition
        if output["meta_info"]["finish_reason"]["type"] == "abort":
            sample.status = Sample.Status.ABORTED
            sample.reward = global_config.DEFAULT_SCORE
            print("Aborted samples:")
            print(_req.messages)
            return sample

        # Rollout output text
        cur_response = output["text"]

        # Add assistant messages
        if "<|im_end|>" in cur_response:
            cur_response = cur_response.split("<|im_end|>")[0]
        # if "<|im_start|>" in cur_response:
        #     cur_response = cur_response.split("<|im_start|>")[0]
        cur_response = cur_response.strip()

        if not cur_response:
            print("Empty response from doctor model.")
            sample.status = Sample.Status.ABORTED
            sample.reward = global_config.DEFAULT_SCORE
            print("Aborted samples:")
            print(_req.messages)
            return sample

        _req.add_assistant_message(cur_response)

        if len(_req.input_ids) >= global_config.MAX_MODEL_LEN:
            finish_reason_type = "length"
            break

        answer = cur_response.split("</think>")[-1].strip()
        if "Diagnosis:" in answer or "Recommendation:" in answer:
            finish_reason_type = "completed"
            break

        # Check length limit
        if output["meta_info"]["finish_reason"]["type"] == "length":
            finish_reason_type = "length"
            break

        # Execute user simulation
        if _req.messages[-1]["role"] == "assistant":
            should_continue, reason = await action_processor.execute_user_simulation(
                _req, tool_map
            )

            if reason:
                finish_reason_type = reason
                break

            if not should_continue:
                break

    # Generate tokens, response, and loss_mask using apply_chat_template
    tokens = tokenizer.apply_chat_template(
        _req.messages,
        add_generation_prompt=False,
        tokenize=True,
        enable_thinking=True,
        keep_thinking=True,
    )

    tokenized_output = tokenizer.apply_chat_template(
        _req.messages[1:],
        tokenize=True,
        return_assistant_tokens_mask=True,
        return_dict=True,
        enable_thinking=True,
        keep_thinking=True,
    )

    response = tokenizer.apply_chat_template(
        _req.messages[1:],
        add_generation_prompt=False,
        tokenize=False,
        enable_thinking=True,
        keep_thinking=True,
    )

    # Set results
    sample.tokens = tokens
    sample.response_length = len(tokenized_output["input_ids"])
    sample.response = response
    sample.loss_mask = tokenized_output["assistant_masks"]
    sample.metadata["messages"] = _req.messages

    # Set status
    if finish_reason_type == "length":
        sample.status = Sample.Status.TRUNCATED
    elif finish_reason_type in ["abort", "error"]:
        sample.status = Sample.Status.ABORTED
    else:
        sample.status = Sample.Status.COMPLETED
    return sample


async def generate(args, sample: Sample, sampling_params) -> Sample:
    """
    Simplified generate function interface.

    Args:
        args: Original arguments object.
        sample: Sample object.
        sampling_params: Sampling parameters.
    """
    from slime.rollout.sglang_rollout import GenerateState

    # Create components
    state = GenerateState(args)
    action_processor = PatientSimulatorActionProcessor()
    llm_client = SGLangClient()

    state.tokenizer.chat_template = TEMPLATE_QWEN3_ASSISTANT_MASK

    completed_sample = await multi_turn_generate(
        args=args,
        sample=sample,
        sampling_params=sampling_params,
        action_processor=action_processor,
        llm_client=llm_client,
        tokenizer=state.tokenizer,  # doctor tokenizer
    )

    patient_sample = await change_patient_tool_to_sample(
        action_processor, completed_sample
    )

    task_id = str(uuid.uuid4())
    if patient_sample:
        commit_patient_data(task_id, patient_sample)
    await action_processor.cleanup_tools()

    return completed_sample


async def reward_func(args, sample, **kwargs):
    """
    Compute reward score for doctor simulation.

    Returns:
        float: Total reward score.
    """
    if not isinstance(sample, Sample):
        raise TypeError("Sample must be an instance of Sample class.")

    reward_details = await compute_score(sample)
    score = reward_details["score"]

    return score if score else 0.0
