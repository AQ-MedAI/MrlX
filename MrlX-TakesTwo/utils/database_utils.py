"""
Module: database_utils
----------------------
Provides helper functions to:
    - Build queue list keys with a date and unique suffix.
    - Commit patient simulation data to a remote queue service.
    - Fetch patient simulation data from a remote queue service.

Features:
    - Uses `global_config.DATASET_IP` to determine base URL; falls back to fixed address if unset.
    - Retries network operations with configurable limits.
    - Validates environment variables before building list keys.
"""

import json
import os
import requests
import time
from datetime import datetime
from enum import Enum
from slime.utils.types import Sample
from config import global_config


if global_config.DATABASE_SERVER_IP:
    DEFAULT_BASE_URL = f"http://{global_config.DATABASE_SERVER_IP}:18888"
else:
    raise ValueError("Required configuration 'DATABASE_SERVER_IP' is not set or is empty in global_config.")

def _get_base_url() -> str:
    """
    Returns the active base URL for remote queue service.

    Selection order:
        1. If `global_config.DATASET_IP` is set, use it with port 18888.
        2. Otherwise, use fixed medical environment service URL.

    Returns:
        str: Base URL string.
    """
    return DEFAULT_BASE_URL

def _build_list_key() -> str:
    """
    Builds a unique list key using today's date and the environment variable KEY_SUFFIX.

    Returns:
        str: Generated list key.

    Raises:
        RuntimeError: If KEY_SUFFIX is not set in the environment.
    """
    key_suffix = os.getenv("KEY_SUFFIX")
    if not key_suffix:
        raise RuntimeError("KEY_SUFFIX environment variable is not set")

    date_str = datetime.now().strftime("%Y%m%d")
    return f"listKey_{date_str}_{key_suffix}"


# --------------------------------------------------------------------------- #
#  Main entry functions                                                        #
# --------------------------------------------------------------------------- #
def commit_patient_data(task_id: str, patient_data: dict, max_retries: int = 10):
    """
    Submit processed patient simulation data to a remote service with retry logic.

    Args:
        task_id (str): Unique task identifier.
        patient_data (dict): Dictionary containing response, tokens, and metadata.
        max_retries (int): Maximum retry attempts. Default is 10.

    Returns:
        dict | None: JSON response from server if successful, None otherwise.
    """
    url = f"{_get_base_url()}/taskCommit"
    headers = {"Content-Type": "application/json"}

    # Skip commit if sample status is aborted
    status = patient_data.get("status")
    if hasattr(Sample, "Status") and status == Sample.Status.ABORTED:
        print("Sample status is ABORTED, data will not be saved.")
        return

    # Convert Enum status to string if needed
    status_name = status.name if isinstance(status, Enum) else str(status)

    # Build task payload for server
    task_data = {
        "id": task_id,
        "response": patient_data["response"],
        "responseLength": patient_data["response_length"],
        "status": status_name,
        "tokens": patient_data["tokens"],
        "lossMask": patient_data["loss_mask"],
        "messages": patient_data["messages"],
        "chiefComplaint": patient_data.get("chief_complaint", ""),
        "diagnosis": patient_data.get("diagnosis", ""),
        "recommendation": patient_data.get("recommendation", ""),
        "selfReport": patient_data.get("self_report", ""),
        "createdDate": datetime.now().isoformat(),
    }
    data = {
        "listKey": _build_list_key(),
        "taskData": json.dumps(task_data, ensure_ascii=False),
    }

    # Retry sending data
    for attempt in range(max_retries):
        try:
            response = requests.post(url, json=data, headers=headers, timeout=10)
            response.raise_for_status()
            res_data = response.json()

            if res_data.get("success", False) and res_data.get("data", {}).get(
                "success", False
            ):
                print(
                    f"Task {task_id} submitted successfully after {attempt + 1} attempts."
                )
                return res_data
            else:
                print(
                    f"Task {task_id} submission failed (attempt {attempt + 1}): {res_data}"
                )

        except requests.exceptions.RequestException as e:
            print(f"Task {task_id} submission failed (attempt {attempt + 1}): {e}")

    print(f"Task {task_id} submission failed after {max_retries} attempts.")
    return None


def get_patient_data(max_retries: int = 100, retry_delay: int = 5):
    """
    Retrieve patient simulation data from the remote service with retry logic.

    Args:
        max_retries (int): Maximum retry attempts before giving up.
        retry_delay (int): Delay in seconds between retries.

    Returns:
        dict | None: The parsed 'taskData' dictionary if successful, None otherwise.
    """
    url = f"{_get_base_url()}/taskFetch"
    headers = {"Content-Type": "application/json"}
    data = {"listKey": _build_list_key()}

    for attempt in range(max_retries):
        try:
            response = requests.post(url, json=data, headers=headers)
            response.raise_for_status()
            patient_data = response.json()

            # Outer success check
            if not patient_data["success"]:
                inner_data = patient_data.get("data", {})
                if inner_data and not inner_data.get("success", True):
                    error_msg = inner_data.get("errorMsg", "")
                    if error_msg == "队列为空":
                        print("Queue is empty. Exiting.")
                        return None
                    elif error_msg == "系统异常":
                        print(
                            f"System error encountered. Retrying (attempt {attempt + 1}/{max_retries})."
                        )
                        time.sleep(retry_delay)
                        continue
                    else:
                        print(f"Unexpected error: {error_msg}")
                        return None

            # Successful retrieval
            task_data_str = patient_data["data"].get("taskData")
            if task_data_str:
                try:
                    task_data = json.loads(task_data_str)
                    return task_data
                except json.JSONDecodeError:
                    print("Error decoding taskData JSON.")
                    return None
            else:
                print("taskData is missing or null.")
                return None

        except requests.exceptions.RequestException as e:
            print(f"HTTP request failed: {e}")
            time.sleep(retry_delay)

    print("Max retries reached. Giving up.")
    return None
