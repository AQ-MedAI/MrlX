"""
Module: patient_tools
---------------------
Implements patient-side simulation tools for roleplay between doctor and patient.

Key Classes:
    MockRequest - Container for chat messages and tokenization logic.
    PatientSimulatorActionProcessor - Orchestrates patient tool lifecycle and simulation execution.
    PatientSimulatorTool - Core tool calling patient LLM agent and tracking conversation state.

Features:
    - Loads system prompt from file for patient role
    - Handles conversation history role reversal
    - Calls patient agent LLM endpoint
    - Checks input length constraints
"""

import json
import logging
import os
import random
import time
import asyncio
from typing import Any, Dict, List, Optional, Tuple, Callable
from uuid import uuid4
import re

from openai import OpenAI
from pydantic import BaseModel

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.types import Sample
from config import global_config


logger = logging.getLogger(__name__)


class MockRequest:
    """
    Mock request object storing conversation messages and performing tokenization
    for patient simulation.

    Attributes:
        messages (list[dict]): Chat history list.
        input_ids (list[int]): Tokenized IDs of chat history.
        tokenizer (Callable): Tokenizer function.
        tools_kwargs (dict): Optional tools-related parameters.
    """

    def __init__(
        self,
        tokenizer: Callable,
        tools_kwargs: Optional[Dict] = None,
    ):
        self.messages = []  # standard chat format messages
        self.input_ids = []  # token ids of all messages, with generation token when last role is user
        self.tokenizer = tokenizer
        self.tools_kwargs = tools_kwargs or {}

    def add_system_message(self):
        """Add a system message."""
        file_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(
            file_dir, "../doc_prompt_template/system_prompt_v3.txt"
        )
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        message = {"role": "system", "content": content}
        self.messages.append(message)
        self.update_input_ids()

    def add_assistant_message(self, content: str):
        """Add an assistant message."""
        message = {"role": "assistant", "content": content}
        self.messages.append(message)
        self.update_input_ids()

    def add_user_message(self, content: str):
        """Add a user response message."""
        message = {"role": "user", "content": content}
        self.messages.append(message)
        self.update_input_ids()

    def add_messages(self, messages: List[Dict[str, str]]):
        """Add multiple messages."""
        for message in messages:
            self.messages.append(message)
        self.update_input_ids()

    def update_input_ids(self):
        """Update input_ids using tokenizer's apply_chat_template."""
        add_generation_prompt = False
        if self.messages[-1]["role"] == "user":
            add_generation_prompt = True
        self.input_ids = self.tokenizer.apply_chat_template(
            self.messages,
            add_generation_prompt=add_generation_prompt,
            tokenize=True,
            enable_thinking=True,
            keep_thinking=True,
        )


# Patient Simulator Action Processor
class PatientSimulatorActionProcessor:
    """
    Patient Simulator Action Processor - Simplified Version.
    """

    def __init__(self):
        """Simplified constructor."""
        self.patient_tool = None
        self.instance_id = None

    def postprocess_responses(self, resp: str) -> str:
        """Post-process the doctor's response - remove <think> tags."""
        cleaned = re.sub(r"<think>.*?</think>", "", resp, flags=re.DOTALL)
        return cleaned.strip()

    async def execute_user_simulation(
        self, req: MockRequest, tool_map: Dict[str, Any]
    ) -> Tuple[bool, Optional[str]]:
        """
        Executes one round of simulation: patient responds to doctor.

        Returns:
            tuple: (success: bool, error_type: Optional[str])
        """
        try:
            # Validate message role
            if not req.messages or req.messages[-1]["role"] != "assistant":
                return False, "error"

            # Get the patient tool
            patient_tool = tool_map["patient_simulator"]

            # Extract the doctor's question (remove <think> tags)
            doctor_question = self.postprocess_responses(req.messages[-1]["content"])

            if not doctor_question:
                print("Doctor's question is empty after post-processing.")
                print(req.messages)

            # Build parameters
            parameters = json.dumps({"doctor_question": doctor_question})

            # Execute patient simulation
            user_resp, reward, metrics = await patient_tool.execute(
                req, self.instance_id, parameters
            )

            # 1. Tokenize the patient's response first, without adding it to the main history.
            user_tokens = req.tokenizer(user_resp, add_special_tokens=False)[
                "input_ids"
            ]

            # 2. Predict the total length if these new tokens were to be added.
            if len(req.input_ids) + len(user_tokens) >= global_config.MAX_MODEL_LEN:
                # If it would exceed the limit, do not add the response.
                # Signal that the conversation should stop due to length.
                return False, "length"

            # Add user response message
            req.add_user_message(user_resp)

            # Check length limit
            if len(req.input_ids) >= global_config.MAX_MODEL_LEN:
                return False, "length"

            return True, None

        except Exception as e:
            logger.exception(f"User simulation execution failed: {str(e)}")
            return False, "error"

    async def init_tools(self, sample: Sample, tokenizer: Callable) -> Dict[str, Any]:
        # Create a PatientSimulatorTool instance
        self.patient_tool = PatientSimulatorTool(tokenizer)

        extra_info = sample.metadata

        chief_complaint = extra_info.get("chief_complaint", "")
        self_report = extra_info.get("self_report", "")

        # Create a patient instance
        self.instance_id = await self.patient_tool.create(
            self_report=self_report, chief_complaint=chief_complaint
        )

        return {"patient_simulator": self.patient_tool}

    async def cleanup_tools(self) -> None:
        """Clean up tools."""
        if self.instance_id and self.patient_tool:
            try:
                await self.patient_tool.release(self.instance_id)
            except Exception as e:
                logger.warning(f"Failed to clean up patient instance: {str(e)}")


class PatientSimulatorTool:
    """
    Core patient simulator tool that interacts with patient-side LLM.
    """

    def __init__(self, tokenizer: Callable):
        self.tokenizer = tokenizer
        self._instance_dict = {}

    async def create(
        self,
        instance_id: Optional[str] = None,
        chief_complaint: Optional[str] = None,
        self_report: Optional[str] = None,
        **kwargs,
    ) -> str:
        """Initialize patient conversation instance"""
        instance_id = instance_id or str(uuid4())
        self._instance_dict[instance_id] = {
            "self_report": self_report or "",
            "chief_complaint": chief_complaint or "",
            "conversation_history": [],
            "reward": 0.0,
        }
        return instance_id

    async def call_agent_model(self, prompt, max_retry=30):
        payload = {
            "text": prompt,
            "sampling_params": {
                "temperature": 0.6,
                "top_p": 0.95,
                "max_new_tokens": global_config.PATIENT_MAX_NEW_TOKENS,
            },
        }

        patient_ip = global_config.PATIENT_IP

        if not patient_ip:
            raise RuntimeError("PATIENT_IP environment variable is not set")

        url = f"http://{patient_ip}:3333/generate"

        while max_retry > 0:
            try:
                # Call the model to generate a response
                output = await post(url, payload)
                cur_response = output["text"]
                return cur_response
            except Exception as e:
                logger.warning(
                    f"Patient agent call failed: {str(e)}, remaining retries: {max_retry - 1}"
                )
                max_retry -= 1
                time.sleep(10)

        logger.error("All patient agent server calls failed")
        return "Sorry, I cannot answer this question at the moment."

    async def execute(
        self, req: MockRequest, instance_id: str, parameters: str, **kwargs
    ) -> Tuple[str, float, dict]:
        """Execute a single round of conversation"""
        try:
            params = json.loads(parameters)
            question = params["doctor_question"]
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Parameter parsing failed: {str(e)}")
            return "Parameter error: doctor_question field is required", -0.1, {}

        # Get instance data
        instance = self._instance_dict.get(instance_id)
        if not instance:
            return "Invalid session ID", -0.2, {}

        # Build message list
        system_prompt = f"""You are a patient interacting with a doctor. Instructions for Responding to Medical Questions:
        Answer each medical question from the doctor concisely in a single sentence, strictly describing your symptoms and avoiding any mention of diagnoses and recommendations.
        If the question is unrelated to your self-report states: "Sorry, I cannot answer your question."

        Your self-report states: {instance["self_report"]}
        """

        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
        ]

        # Add conversation history, removing doctor's first user message and reversing roles
        for entry in req.messages[1:]:
            if entry["role"] == "assistant":
                messages.append({"role": "user", "content": entry["content"]})
            elif entry["role"] == "user":
                messages.append({"role": "assistant", "content": entry["content"]})

        # Add current question
        messages.append({"role": "user", "content": question})

        current_prompt = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

        try:
            patient_response = await self.call_agent_model(
                current_prompt, max_retry=100
            )
        except Exception as e:
            logger.exception(f"Patient agent call exception: {str(e)}")
            return "Patient agent service error", 0, {}

        # Update conversation history
        instance["conversation_history"].append({"role": "user", "content": question})
        instance["conversation_history"].append(
            {"role": "assistant", "content": patient_response}
        )

        # Calculate reward for this round
        reward = await self.calc_reward(instance_id)

        instance["reward"] += reward

        return patient_response, reward, {}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        # instance = self._instance_dict.get(instance_id, None)
        # if not instance:
        #     return 0
        # else:
        #     return len(instance["conversation_history"])/2
        return 0

    async def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]
