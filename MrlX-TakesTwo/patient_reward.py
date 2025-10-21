"""
Module: patient_reward
----------------------
Implements patient-side multi-turn reward calculation.

Features:
    - Loads scoring prompt templates
    - Calls R1 model to evaluate responses for information, completeness, conflict
    - Calculates weighted reward sums
"""

import os
import asyncio
import json
from openai import AsyncOpenAI
from typing import List, Dict, Tuple, Any
from datetime import datetime, timedelta

from config import global_config
from utils.reward_utils import (
    extract_text,
    convert_messages_to_dialogue,
    parse_score,
    call_r1_model_async,
)

file_dir = os.path.dirname(os.path.abspath(__file__))
docagent_prompt_file = "patient_prompt_template/docagent.txt"


class MedPatientRewardMultiturn:
    """
    Patient-side reward calculator.
    """

    def __init__(self):
        prompt_files = [
            ("docagent", os.path.join(file_dir, docagent_prompt_file)),
        ]

        self.prompts = {}
        for key, file_path in prompt_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    self.prompts[key] = f.read()
            except FileNotFoundError:
                print(f"Error: The prompt file {file_path} was not found.")
                self.prompts[key] = ""

        self.client_r1 = AsyncOpenAI(
            api_key=global_config.DEEPSEEK_R1_API_KEY,
            base_url=global_config.DEEPSEEK_R1_BASE_URL,
        )

    async def calculate_rewards(
        self,
        sample,
        rewards_weights: Dict[str, float] = {
            "information_reward": 1,
            "completeness_reward": 1,
            "conflict_reward": 1,
        },
    ) -> Dict[str, Any]:
        """
        Calculate weighted patient reward score.
        """
        messages = sample.metadata.get("messages", [])
        self_report = sample.metadata.get("self_report", "")
        diagnosis = sample.metadata.get("diagnosis", "")
        recommendation = sample.metadata.get("recommendation", "")

        raw_scores = await self.calculate_all_scores_async(
            messages=messages,
            self_report=self_report,
            diagnosis=diagnosis,
            recommendation=recommendation,
        )

        if raw_scores is None:
            return {"score": 0.0, "details": {}}
        else:
            print(f"raw_scores: {raw_scores}")

        total_reward = 0.0
        reward_details = {}

        for score_key in rewards_weights.keys():
            reward_details[score_key] = raw_scores[score_key]
            total_reward += raw_scores[score_key] * rewards_weights[score_key]

        reward_details["score"] = round(total_reward, 3)

        return reward_details

    async def calculate_all_scores_async(
        self,
        messages: List[Dict[str, str]],
        self_report: str,
        diagnosis: str,
        recommendation: str,
        try_num: int = 5,
    ) -> Dict[str, float]:
        """
        Calls R1 model to score patient-side conversation.

        Returns:
            dict | None: Scores for information, completeness, and conflict.
        """
        messages_wo_system = (
            messages[1:-1] if messages[-1]["role"] == "user" else messages[1:]
        )
        dialogue_wo_system = convert_messages_to_dialogue(messages_wo_system)

        # patient data in the format of system, user(doctor), assistant(patient), ...
        for _ in range(try_num):
            try:
                tasks = [
                    call_r1_model_async(
                        self.client_r1,
                        messages=[
                            {
                                "role": "user",
                                "content": self.prompts["docagent"]
                                .replace("{self report}", self_report)
                                .replace("{diagnosis}", diagnosis)
                                .replace("{recommendation}", recommendation)
                                .replace("{simulated dialogue}", dialogue_wo_system),
                            }
                        ],
                        temperature=0.6,
                    ),
                ]

                results = await asyncio.gather(*tasks)

                docagent_score = parse_score(results[0])

                return {
                    "information_reward": docagent_score.get(
                        "information_control_rate", 0.0
                    ),
                    "completeness_reward": docagent_score.get(
                        "response_completeness_rate", 0.0
                    ),
                    "conflict_reward": docagent_score.get("factual_conflict_rate", 0.0),
                }

            except (json.JSONDecodeError, TypeError, Exception) as error:
                print(
                    f"An error occurred while calculating scores, retrying... Error: {error}"
                )

        print("Failed to calculate scores after multiple retries.")

        return {}


async def compute_score(sample) -> Tuple[Dict[str, Any], timedelta]:
    """
    Wrapper to compute patient rewards and measure execution time.
    """
    start_time = datetime.now()
    calculator = MedPatientRewardMultiturn()

    # Calculate rewards with equal weights for correct and anthropomorphism scores
    reward_details = await calculator.calculate_rewards(
        sample=sample,
        rewards_weights={
            "completeness_reward": 1,
            "conflict_reward": 1,
            "information_reward": 1,
        },
    )

    end_time = datetime.now()
    time_cost = end_time - start_time

    return reward_details, time_cost
