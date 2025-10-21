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

"""Database utilities for sub-agent data transfer."""

import json
import logging
import os
import time
from datetime import datetime

import requests


logger = logging.getLogger(__name__)

# Get database server configuration from environment variables
DATABASE_SERVER_IP = os.getenv("DATABASE_SERVER_IP")
if DATABASE_SERVER_IP is None:
    # Fall back to SUB_AGENT_IP if DATABASE_SERVER_IP is not set
    DATABASE_SERVER_IP = os.getenv("SUB_AGENT_IP")
    if DATABASE_SERVER_IP is None:
        raise RuntimeError(
            "Environment variable DATABASE_SERVER_IP or SUB_AGENT_IP is missing - "
            "remote data transfer for sub-agents cannot continue."
        )

DATABASE_SERVER_PORT = os.getenv("DATABASE_SERVER_PORT", "18888")

DEFAULT_TIMEOUT = 10
DEFAULT_RETRY_DELAY = 1


def _get_base_url() -> str:
    """Return the base URL for database server."""
    return f"http://{DATABASE_SERVER_IP}:{DATABASE_SERVER_PORT}"


def _build_list_key() -> str:
    """Build a list key in format: listKey_<YYYYMMDD>_<KEY_SUFFIX>.

    Returns:
        str: The constructed list key.

    Raises:
        RuntimeError: If KEY_SUFFIX environment variable is not set.
    """
    key_suffix = os.getenv("KEY_SUFFIX")
    if not key_suffix:
        raise RuntimeError("Environment variable KEY_SUFFIX is not set.")
    date_str = datetime.now().strftime("%Y%m%d")
    return f"listKey_{date_str}_{key_suffix}"


def commit_subagent_data(
    task_id: str,
    sub_agent_data: dict,
    reward: float = 0.0,
    max_retries: int = 10,
):
    """Push a finished task to the remote service with retry logic.

    Args:
        task_id: Unique identifier for the task.
        sub_agent_data: Dictionary containing response, response_length, tokens,
            loss_mask and messages.
        reward: Reward value to attach to the task. Defaults to 0.0.
        max_retries: Number of POST attempts before giving up. Defaults to 10.

    Returns:
        JSON response from server if successful, None otherwise.
    """
    url = f"{_get_base_url()}/taskCommit"
    headers = {"Content-Type": "application/json"}

    status_name = "completed"

    task_data = {
        "id": task_id,
        "response": sub_agent_data["response"],
        "responseLength": sub_agent_data["response_length"],
        "status": status_name,
        "tokens": sub_agent_data["tokens"],
        "lossMask": sub_agent_data["loss_mask"],
        "messages": sub_agent_data["messages"],
        "reward": reward,
        "createdDate": datetime.now().isoformat(),
    }

    data = {
        "listKey": _build_list_key(),
        "taskData": json.dumps(task_data, ensure_ascii=False),
    }

    for attempt in range(max_retries):
        try:
            response = requests.post(
                url, json=data, headers=headers, timeout=DEFAULT_TIMEOUT
            )
            response.raise_for_status()
            res_data = response.json()

            if res_data.get("success") and res_data.get("data", {}).get("success"):
                logger.info(f"Task {task_id} submitted successfully after {attempt + 1} attempt(s).")
                return res_data

            logger.warning(f"Task {task_id} submission failed (attempt {attempt + 1}): {res_data}")
            if attempt < max_retries - 1:
                time.sleep(DEFAULT_RETRY_DELAY)

        except requests.exceptions.RequestException as exc:
            logger.warning(f"Task {task_id} submission failed (attempt {attempt + 1}): {exc}")
            if attempt < max_retries - 1:
                time.sleep(DEFAULT_RETRY_DELAY)

    logger.error(f"Task {task_id} submission failed after {max_retries} attempts.")
    return None


def get_subagent_data(max_retries: int = 100, retry_delay: int = 5):
    """Poll the remote queue for new work intended for this sub-agent.

    Args:
        max_retries: Number of HTTP attempts before giving up. Defaults to 100.
        retry_delay: Seconds to sleep between retries. Defaults to 5.

    Returns:
        Parsed taskData dictionary on success, None if queue is empty or error occurs.
    """
    url = f"{_get_base_url()}/taskFetch"
    headers = {"Content-Type": "application/json"}
    data = {"listKey": _build_list_key()}

    for attempt in range(max_retries):
        try:
            response = requests.post(url, json=data, headers=headers)
            response.raise_for_status()
            sub_agent_resp = response.json()

            if not sub_agent_resp.get("success", False):
                inner = sub_agent_resp.get("data", {})
                inner_success = inner.get("success", True)
                if not inner_success:
                    error_msg = inner.get("errorMsg", "")
                    if error_msg == "Queue is empty":
                        logger.info("The task queue is empty. Nothing to do.")
                        return None
                    elif error_msg == "System error":
                        logger.warning(f"Remote system error. Retrying (attempt {attempt + 1}/{max_retries}).")
                        time.sleep(retry_delay)
                        continue
                    else:
                        logger.error(f"Unknown upstream error: {error_msg}")
                        return None

            task_data_str = sub_agent_resp["data"].get("taskData")
            if not task_data_str:
                return None

            try:
                return json.loads(task_data_str)
            except json.JSONDecodeError:
                logger.error("Upstream returned malformed JSON in taskData.")
                return None

        except requests.exceptions.RequestException as exc:
            logger.warning(f"HTTP request failed: {exc}")
            time.sleep(retry_delay)

    logger.error("Reached maximum retries while polling for tasks. Giving up.")
    return None
