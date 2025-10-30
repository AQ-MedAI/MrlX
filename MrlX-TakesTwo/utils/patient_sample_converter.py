"""
Module: patient_sample_converter
--------------------------------
Converts a doctor conversation sample (Sample) into a patient-side
simulation format suitable for LLM model processing.

Features:
    - Loads patient tokenizer once and caches it
    - Prepares chat messages with system prompt, role reversing
    - Tokenizes and calculates loss mask for assistant tokens
    - Extracts response text and length for later training/inference
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Callable, Set
import asyncio

from slime.utils.types import Sample

from utils.chat_template import TEMPLATE_QWEN3_ASSISTANT_MASK
from utils.patient_tools import PatientSimulatorActionProcessor
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from config import global_config

logger = logging.getLogger(__name__)

# --- Cache area ---
# This variable will store the loaded tokenizer instance
_tokenizer_cache: PreTrainedTokenizerBase | None = None


def get_tokenizer() -> PreTrainedTokenizerBase:
    """
    Returns a singleton tokenizer instance for patient simulation.
    Loads the tokenizer once from global_config.PATIENT_TOKENIZER_PATH
    and injects the Qwen3 assistant chat template.

    Returns:
        PreTrainedTokenizerBase: Loaded tokenizer instance.

    Raises:
        Exception: If tokenizer cannot be loaded.
    """
    global _tokenizer_cache

    # Simplified version of double-checked locking (DCL), thread-safe for most Python implementations
    if _tokenizer_cache is None:
        tokenizer_path = global_config.PATIENT_TOKENIZER_PATH
        logger.info(f"Loading tokenizer for the first time, path: {tokenizer_path}")
        try:
            _tokenizer_cache = AutoTokenizer.from_pretrained(tokenizer_path)
            _tokenizer_cache.chat_template = TEMPLATE_QWEN3_ASSISTANT_MASK
        except Exception as e:
            logger.error(f"Unable to load tokenizer from path '{tokenizer_path}': {e}")
            raise  # Re-raise exception as program cannot continue without tokenizer

    return _tokenizer_cache


async def change_patient_tool_to_sample(
    action_processor: PatientSimulatorActionProcessor, doctor_sample: Sample
) -> dict:
    """
    Convert a doctor conversation Sample into patient simulation format.
    This will:
        1. Build a system prompt using chief complaint and self report.
        2. Reverse roles in conversation history.
        3. Apply tokenizer chat template to generate tokens and loss mask.
        4. Extract and tokenize response segment separately.

    Args:
        action_processor (PatientSimulatorActionProcessor): Patient simulator action processor.
        doctor_sample (Sample): Doctor's conversation sample.

    Returns:
        dict: Processed patient sample containing response/tokens/mask, or {} if invalid.
    """

    # patient tokenizer
    tokenizer = get_tokenizer()

    # instance_id = action_processor.instance_id
    # patient_tool = action_processor.patient_tool
    # instance = patient_tool._instance_dict.get(instance_id)


    system_prompt = f"""You are a patient interacting with a doctor. Instructions for Responding to Medical Questions:
Answer each medical question from the doctor concisely in a single sentence, strictly describing your symptoms and avoiding any mention of diagnoses and recommendations.
If the question is unrelated to your chief complaint, state: "Sorry, I cannot answer this question."
Your chief complaint: {doctor_sample.metadata["chief_complaint"]}
If the question is unrelated to your self-report states: "Sorry, I cannot answer your question."
Your self-report states: {doctor_sample.metadata["self_report"]}"""

    # Return {} if messages don't exist
    if "messages" not in doctor_sample.metadata:
        return {}

    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
    ]

    # Add conversation history, removing doctor's first user message and reversing roles
    for entry in doctor_sample.metadata["messages"][1:]:
        if entry["role"] == "assistant":
            messages.append({"role": "user", "content": entry["content"]})
        elif entry["role"] == "user":
            messages.append({"role": "assistant", "content": entry["content"]})

    if len(messages) <= 2:
        return {}

    tokenized_output = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_assistant_tokens_mask=True,
        return_dict=True,
        enable_thinking=True,
        keep_thinking=True,
    )
    tokens = tokenized_output["input_ids"]
    prompt_length = len(tokenizer.apply_chat_template(messages[:2], tokenize=True))
    loss_mask = tokenized_output["assistant_masks"][prompt_length:]

    response = tokenizer.apply_chat_template(
        messages[2:],
        add_generation_prompt=False,
        tokenize=False,
        enable_thinking=True,
        keep_thinking=True,
    )

    response_length = len(
        tokenizer.apply_chat_template(
            messages[2:],
            add_generation_prompt=False,
            tokenize=True,
            enable_thinking=True,
            keep_thinking=True,
        )
    )

    patient_sample = {
        "messages": messages,
        "tokens": tokens,
        "response_length": response_length,
        "loss_mask": loss_mask,
        "response": response,
        "status": doctor_sample.status,
        "chief_complaint": doctor_sample.metadata["chief_complaint"],
        "diagnosis": doctor_sample.metadata["solution"]["diagnosis"],
        "recommendation": doctor_sample.metadata["solution"]["recommendation"],
        "self_report": doctor_sample.metadata.get("self_report", ""),
    }

    return patient_sample
