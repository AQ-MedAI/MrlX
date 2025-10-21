"""
Module: doctor_reward
---------------------
Implements multi-turn reward calculation for doctor-side simulation.

Features:
    - Calculates accuracy, information, and compliance rewards for doctor conversations
    - Uses text similarity (Rouge-like) for diagnostic/recommendation quality
    - Aggregates multiple reward functions with weights
"""

import re
import numpy as np
from typing import Callable, Dict, List, Optional, Tuple
from datetime import datetime
import asyncio
import os
from datetime import datetime

from utils.reward_utils import extract_diagosis_recommendation
from config import global_config


class MedRewardMultiturn:
    """
    Doctor-side reward calculator for multi-turn medical simulations.
    """

    def __init__(self):
        self._print_num = -1
        self.has_reply = True
        self.reward_functions = {
            "accuracy_reward": self.accuracy_reward,
            "information_reward": self.information_reward,
            "compliance_reward": self.compliance_reward,
        }

    async def accuracy_reward(
        self,
        messages: List[Dict[str, str]],
        ground_truth: Optional[Dict[str, str]] = None,
    ) -> float:
        """
        Measures diagnostic/recommendation accuracy using simplified ROUGE-F1.

        Returns:
            float: Accuracy reward score.
        """

        def _get_rouge_score(text1, text2):
            from collections import Counter
            import jieba

            if not text1 or not text2:
                return 0.0

            words1 = jieba.lcut(text1)
            words2 = jieba.lcut(text2)

            count1 = Counter(words1)
            count2 = Counter(words2)

            common = count1 & count2
            num_common = sum(common.values())

            if not words1 or not words2:
                return 0.0

            precision = num_common / len(words1) if len(words1) > 0 else 0.0
            recall = num_common / len(words2) if len(words2) > 0 else 0.0

            if precision + recall == 0:
                f1 = 0.0
            else:
                f1 = 2 * precision * recall / (precision + recall)

            return f1

        reward = 0.0
        diagnosis, recommendation = extract_diagosis_recommendation(messages)
        if diagnosis:
            reward += _get_rouge_score(diagnosis, ground_truth.get("diagnosis", "")) * 5
        if recommendation:
            reward += (
                _get_rouge_score(recommendation, ground_truth.get("recommendation", ""))
                * 5
            )
        return reward if reward else 0.0

    async def information_reward(
        self,
        messages: List[Dict[str, str]],
        ground_truth: Optional[Dict[str, str]] = None,
    ) -> float:
        """
        Rewards information gathering quality.

        Penalizes excessive 'sorry' answers; rewards informative patient queries.
        """
        # TODO: Implement information gathering logic
        # Remove the first user message, cause it is chief complaint
        messages = messages[1:]
        reward = 0.0
        for message in messages:
            if message["role"] == "user":
                if "sorry" in message["content"].lower():
                    reward -= 2.0
                else:
                    reward += 1.0
        return reward

    async def compliance_reward(
        self,
        messages: List[Dict[str, str]],
        ground_truth: Optional[Dict[str, str]] = None,
    ) -> float:
        """
        Rewards protocol adherence.

        Penalizes multiple questions per turn, exceeding MAX_TURNS, or missing diagnosis.
        """
        reward = 0.0
        messages = messages[
            1:
        ]  # Remove the first user message, cause it is chief complaint

        for message in messages:
            if message["role"] == "user":
                if message["content"].count("?") >= 2:
                    reward -= 2.0
        diagnosis, recommendation = extract_diagosis_recommendation(messages)
        if not diagnosis or recommendation:
            reward -= 5.0
            return reward

        turns = len(messages) // 2
        if turns > global_config.MAX_TURNS:
            reward -= 5.0

        return reward

    async def calculate_rewards(
        self,
        messages: List[Dict[str, str]],
        ground_truth: Optional[Dict[str, str]] = None,
        rewards_dict: Dict[str, float] = {
            "accuracy_reward": 1.0,
            "information_reward": 1.0,
            "compliance_reward": 1.0,
        },
    ) -> Dict[str, float]:
        """
        Calculate weighted reward sum from defined reward functions.
        """
        tasks = []
        for name in rewards_dict.keys():
            if name not in self.reward_functions:
                continue
            func = self.reward_functions[name]
            tasks.append((name, func(messages, ground_truth)))

        # Execute all reward calculations in parallel
        results = await asyncio.gather(*(task for _, task in tasks))

        # Parse results and calculate weighted sum
        reward_details = {}
        total_reward = 0.0
        for (name, _), value in zip(tasks, results):
            weighted_value = value * rewards_dict.get(name, 1.0)
            reward_details[name] = weighted_value
            total_reward += weighted_value

        reward_details["score"] = total_reward
        return reward_details


async def compute_score(sample):
    """
    Wrapper to compute all rewards for a doctor Sample and log timing.
    """
    start_time = datetime.now().strftime("%H:%M:%S")
    calculator = MedRewardMultiturn()
    reward_details = await calculator.calculate_rewards(
        messages=sample.metadata.get("messages", []),
        ground_truth=sample.metadata.get("solution", {}),
    )
    end_time = datetime.now().strftime("%H:%M:%S")

    print(
        f"start_time: {start_time}, end_time: {end_time}, reward_details: {reward_details}, turn: {len(sample.metadata.get('messages', [])) // 2}"
    )
    return reward_details
