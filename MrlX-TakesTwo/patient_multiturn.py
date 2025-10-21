"""
Module: patient_multiturn
-------------------------

Handles patient-side multi-turn processing:
    - Retrieves patient simulation data from database
    - Populates Sample object
    - Computes patient reward score
"""

import asyncio

from slime.utils.types import Sample

from patient_reward import compute_score
from utils.database_utils import get_patient_data


async def generate(args, sample: Sample, sampling_params) -> Sample:
    """
    Continuously fetch patient data and fill the provided Sample object.
    Waits if no data is available.

    Returns:
        Sample: Updated with patient data from database.
    """
    while True:
        sample_to_process = get_patient_data()
        # print(f'[Debug the trans is {type(sample_to_process)}] {sample_to_process}' )
        if sample_to_process:
            tokens = sample_to_process.get("tokens", [])
            response_length = sample_to_process.get("responseLength", 0)
            response = sample_to_process.get("response", "")
            loss_mask = sample_to_process.get("lossMask", [])
            messages = sample_to_process.get("messages", [])
            chief_complaint = sample_to_process.get("chiefComplaint", "")
            diagnosis = sample_to_process.get("diagnosis", "")
            recommendation = sample_to_process.get("recommendation", "")
            self_report = sample_to_process.get("selfReport", "")

            sample.tokens = tokens
            sample.response_length = response_length
            sample.response = response
            sample.loss_mask = loss_mask
            sample.metadata["messages"] = messages
            sample.metadata["chief_complaint"] = chief_complaint
            sample.metadata["diagnosis"] = diagnosis
            sample.metadata["recommendation"] = recommendation
            sample.metadata["self_report"] = self_report

            # print(f'[Debug the sample is] {sample}' )
            return sample
        else:
            # No task received, the queue is empty
            print(" No unprocessed samples in database, waiting 5 seconds...")
            await asyncio.sleep(5)  # Wait for some time before retrying


async def reward_func(args, sample, **kwargs):
    """
    Compute patient-side reward score.

    Returns:
        float: Total patient reward score.
    """

    if not isinstance(sample, Sample):
        raise TypeError("Sample must be an instance of Sample class.")

    reward_details, time_cost = await compute_score(sample)
    score = reward_details["score"]

    print(f"[Debug] the patient score is {score}")

    return score
