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

"""Asynchronous sub-agent server for search and visit operations."""

import asyncio
import json
import os
from typing import Any, Optional, Tuple

import aiohttp


RETRIEVAL_SERVICE_URL = os.getenv("RETRIEVAL_SERVICE_URL", 'http://127.0.0.1:50001/retrieve')
MAX_RETRIES = 10
INITIAL_RETRY_DELAY = 3


async def agent_search(
    payload: dict,
    top_k: int = 5,
    timeout: int = 60,
    proxy: Optional[str] = None
) -> Tuple[Optional[Any], Optional[str]]:
    """Perform search using retrieval service with async HTTP."""
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    last_error = None
    timeout_obj = aiohttp.ClientTimeout(total=timeout)

    for attempt in range(MAX_RETRIES):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    RETRIEVAL_SERVICE_URL,
                    headers=headers,
                    json=payload,
                    timeout=timeout_obj,
                    proxy=proxy,
                ) as response:
                    raw_response_text = await response.text()

                    if response.status in [500, 502, 503, 504]:
                        last_error = (
                            f"API Request Error: Server Error ({response.status}) on attempt "
                            f"{attempt + 1}/{MAX_RETRIES}"
                        )
                        if attempt < MAX_RETRIES - 1:
                            delay = INITIAL_RETRY_DELAY * (attempt + 1)
                            await asyncio.sleep(delay)
                        continue

                    response.raise_for_status()
                    return json.loads(raw_response_text), None

        except (aiohttp.ClientConnectionError, asyncio.TimeoutError) as e:
            last_error = f"{type(e).__name__}: {e}"
            if attempt < MAX_RETRIES - 1:
                delay = INITIAL_RETRY_DELAY * (attempt + 1)
                await asyncio.sleep(delay)
            continue
        except aiohttp.ClientResponseError as e:
            last_error = f"API Request Error: {e.status} {e.message}"
            break
        except json.JSONDecodeError as e:
            last_error = f"API Response JSON Decode Error: {e}, Response: {raw_response_text[:200]}"
            break
        except Exception as e:
            last_error = f"Unexpected Error: {e}"
            break

    return None, last_error if last_error else "API Call Failed after retries"


async def agent_visit(
    payload: dict,
    timeout: int = 400
) -> Tuple[Optional[Any], Optional[str]]:
    """Visit a webpage and return content using async HTTP."""
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    last_error = None
    timeout_obj = aiohttp.ClientTimeout(total=timeout)

    for attempt in range(MAX_RETRIES):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    RETRIEVAL_SERVICE_URL,
                    headers=headers,
                    json=payload,
                    timeout=timeout_obj,
                ) as response:
                    raw_response_text = await response.text()

                    if response.status in [500, 502, 503, 504]:
                        last_error = (
                            f"API Request Error: Server Error ({response.status}) on attempt "
                            f"{attempt + 1}/{MAX_RETRIES}"
                        )
                        if attempt < MAX_RETRIES - 1:
                            delay = INITIAL_RETRY_DELAY * (attempt + 1)
                            await asyncio.sleep(delay)
                        continue

                    response.raise_for_status()
                    return json.loads(raw_response_text), None

        except (aiohttp.ClientConnectionError, asyncio.TimeoutError) as e:
            last_error = f"{type(e).__name__}: {e}"
            if attempt < MAX_RETRIES - 1:
                delay = INITIAL_RETRY_DELAY * (attempt + 1)
                await asyncio.sleep(delay)
            continue
        except aiohttp.ClientResponseError as e:
            last_error = f"API Request Error: {e.status} {e.message}"
            break
        except json.JSONDecodeError as e:
            last_error = f"API Response JSON Decode Error: {e}, Response: {raw_response_text[:200]}"
            break
        except Exception as e:
            last_error = f"Unexpected Error: {e}"
            break

    return None, last_error if last_error else "API Call Failed after retries"
