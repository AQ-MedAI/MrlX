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

"""Sub-agent multi-turn dialogue generation from database queue."""

import asyncio
import logging

from slime.utils.types import Sample

from agent_utils.database_utils import get_subagent_data
from sub_agent_reward import compute_format_score


logger = logging.getLogger(__name__)


async def generate(args, sample: Sample, sampling_params) -> Sample:
    """Fetch sub-agent sample from database queue and process.

    Args:
        args: Arguments object.
        sample: Sample object to populate.
        sampling_params: Sampling parameters (unused in this implementation).

    Returns:
        Populated sample from database queue.
    """
    try_cnt = 1
    while True:
        if try_cnt % 1000 == 0:
            logger.debug(f'Try count: {try_cnt}, taskData is missing or null')
        try_cnt += 1
        sample_to_process = get_subagent_data()

        if sample_to_process:
            tokens = sample_to_process.get('tokens', [])
            response_length = sample_to_process.get('responseLength', 0)
            response = sample_to_process.get('response', "")
            loss_mask = sample_to_process.get('lossMask', [])
            messages = sample_to_process.get('messages', [])
            reward = sample_to_process.get('reward', 0.0)

            sample.tokens = tokens
            sample.response_length = response_length
            sample.response = response
            sample.loss_mask = loss_mask
            sample.metadata["messages"] = messages
            sample.metadata["reward"] = reward

            logger.info(f'Successfully got sample with last message: {messages[-1]}, Reward: {reward}')
            return sample
        else:
            await asyncio.sleep(30)



async def reward_func(args, sample, **kwargs):
    """Compute reward for sub-agent sample.

    Args:
        args: Arguments object.
        sample: Sample object to evaluate.
        **kwargs: Additional arguments.

    Returns:
        Computed reward score.
    """
    if not isinstance(sample, Sample):
        raise TypeError("Sample must be an instance of Sample class.")

    messages = sample.metadata.get("messages", [])

    format_score, reason = compute_format_score(messages)
    if format_score == 0.0:
        return format_score

    acc_score = sample.metadata.get("reward", 0.0)
    if acc_score == 0.0:
        return format_score * 0.1
    elif acc_score == 0.1:
        return acc_score
    else:
        return 1.0
