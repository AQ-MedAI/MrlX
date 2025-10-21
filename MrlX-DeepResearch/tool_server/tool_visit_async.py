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

"""Asynchronous webpage visit tool with LLM-based content extraction."""

import asyncio
import json
import os
import re
from typing import Dict, List, Union

import aiohttp

from env_config import config


JINA_API_KEYS = config.get("JINA_API_KEY")
TOOL_SERVER_LLM_API_KEY = config.get("TOOL_SERVER_LLM_API_KEY")
TOOL_SERVER_LLM_BASE_URL = config.get("TOOL_SERVER_LLM_BASE_URL")
TOOL_SERVER_LLM_MODEL = config.get("TOOL_SERVER_LLM_MODEL")
WEBCONTENT_MAXLENGTH = config.get("WEBCONTENT_MAXLENGTH")

EXTRACTOR_PROMPT = """Please analyze the following webpage content and extract useful information based on the user's goal.

Webpage Content:
{webpage_content}

User Goal: {goal}

Please provide a JSON response with the following structure:
{{
    "evidence": {{
        "most_relevant_information": "Extract the most relevant information from the webpage that directly addresses the user's goal. Be specific and detailed."
    }},
    "summary": {{
        "concise_paragraph": "Provide a concise summary of the key points that are relevant to the user's goal."
    }}
}}

Focus on extracting information that is directly relevant to the user's goal. If the webpage doesn't contain relevant information, indicate this clearly in your response."""


class VisitAsync:
    """Asynchronous webpage visit tool with LLM-based extraction."""

    name = "visit"
    description = "Visit webpage(s) asynchronously and return extracted content summary."

    def __init__(self):
        """Initialize async visit tool."""
        self._session = None
        self._semaphore = asyncio.Semaphore(200)

    async def _get_session(self):
        """Get or create aiohttp session.

        Returns:
            Active aiohttp ClientSession.
        """
        if self._session is None or self._session.closed:
            connector = aiohttp.TCPConnector(limit=500, limit_per_host=200, ttl_dns_cache=300, use_dns_cache=True)
            timeout = aiohttp.ClientTimeout(total=60, connect=10, sock_read=30)
            self._session = aiohttp.ClientSession(connector=connector, timeout=timeout)
        return self._session

    async def call(self, params: Union[str, Dict], **kwargs) -> str:
        """Visit webpage(s) and extract relevant content.

        Args:
            params: Dictionary with 'url' and 'goal' keys.
            **kwargs: Additional arguments.

        Returns:
            Extracted and formatted webpage content.
        """
        try:
            url = params["url"]
            goal = params["goal"]
        except (TypeError, KeyError):
            return "Error: Invalid parameters. Expected a JSON object with 'url' and 'goal' keys."

        urls = [url] if isinstance(url, str) else url

        tasks = [self._visit_single(u, goal) for u in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_results = []
        for i, res in enumerate(results):
            if isinstance(res, Exception):
                all_results.append(f"Error visiting '{urls[i]}': {str(res)}")
            else:
                all_results.append(res)

        return "\n\n---\n\n".join(all_results)

    def _create_error_message(self, url: str, goal: str, message: str = "The provided webpage content could not be accessed or processed.") -> str:
        """Generate formatted error message.

        Args:
            url: Webpage URL.
            goal: User's extraction goal.
            message: Error message content.

        Returns:
            Formatted error message string.
        """
        useful_information = f"The useful information in {url} for user goal {goal} as follows: \n\n"
        useful_information += f"Evidence in page: \n{message}\n\n"
        useful_information += f"Summary: \n{message}\n\n"
        return useful_information

    async def _visit_single(self, url: str, goal: str) -> str:
        """Visit single URL and extract content.

        Args:
            url: Webpage URL to visit.
            goal: User's extraction goal.

        Returns:
            Formatted extracted content.
        """
        async with self._semaphore:
            try:
                content = await self._readpage(url)
                if not content or content.startswith("[visit] Failed"):
                    return self._create_error_message(url, goal)

                raw_llm_output = await self._extract_with_llm(content, goal)

                # Clean up <think> tags and other noise
                if "<think>" in raw_llm_output:
                    raw_llm_output = re.split(r'</think>', raw_llm_output, maxsplit=1)[-1].strip()

                # Robustly find and parse the JSON object
                try:
                    json_start = raw_llm_output.find('{')
                    json_end = raw_llm_output.rfind('}')
                    if json_start == -1 or json_end == -1:
                        raise json.JSONDecodeError("JSON object not found", raw_llm_output, 0)

                    json_str = raw_llm_output[json_start : json_end + 1]
                    data = json.loads(json_str)

                    # Extract evidence and summary, handling nested structures
                    evidence = data.get("evidence", "N/A")
                    if isinstance(evidence, dict):
                        evidence = evidence.get("most_relevant_information", str(evidence))

                    summary = data.get("summary", "N/A")
                    if isinstance(summary, dict):
                        summary = summary.get("concise_paragraph", str(summary))

                    # Format the output to match the expected format
                    useful_information = f"The useful information in {url} for user goal {goal} as follows: \n\n"
                    useful_information += f"Evidence in page: \n{str(evidence)}\n\n"
                    useful_information += f"Summary: \n{str(summary)}\n\n"
                    return useful_information

                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Error parsing LLM output for {url}. Error: {e}. Raw output: {raw_llm_output}")
                    return self._create_error_message(url, goal, "The LLM response was in an invalid format.")

            except Exception as e:
                print(f"[visit] Critical failure processing {url}: {str(e)}")
                return self._create_error_message(url, goal)

    async def _readpage(self, url: str) -> str:
        """Read webpage content using Jina Reader.

        Args:
            url: Webpage URL.

        Returns:
            Webpage content as text.
        """
        if JINA_API_KEYS:
            try:
                return await self._jina_readpage(url)
            except Exception as e:
                print(f"Jina failed for {url}: {e}")

        return "[visit] Failed to read page."

    async def _jina_readpage(self, url: str) -> str:
        """Use Jina Reader API to fetch webpage content.

        Args:
            url: Webpage URL to fetch.

        Returns:
            Webpage content as text.

        Raises:
            Exception: If all retries fail.
        """
        session = await self._get_session()

        headers = {
            "Authorization": f"Bearer {JINA_API_KEYS}",
        }

        max_retries = 3
        for attempt in range(max_retries):
            try:
                async with session.get(
                    f"https://r.jina.ai/{url}",
                    headers=headers,
                    timeout=20
                ) as response:
                    if response.status == 200:
                        return await response.text()
                    else:
                        text = await response.text()
                        raise Exception(f"Jina API error: {response.status} - {text}")
            except Exception as e:
                if attempt == max_retries - 1:
                    raise Exception(f"Jina readpage error: {str(e)}")
                await asyncio.sleep(1)

        return "[visit] Failed to read page."

    async def _extract_with_llm(self, content: str, goal: str) -> str:
        """Extract information from content using LLM.

        Args:
            content: Webpage content.
            goal: Extraction goal.

        Returns:
            LLM extraction result as JSON string.

        Raises:
            Exception: If extraction fails.
        """
        try:
            max_length = min(len(content), WEBCONTENT_MAXLENGTH)
            truncated_content = content[:max_length]

            if TOOL_SERVER_LLM_API_KEY:
                return await self._extract_with_llm_api(truncated_content, goal)
            else:
                return "{}"

        except Exception as e:
            print(f"LLM extraction error: {str(e)}")
            raise

    async def _extract_with_llm_api(self, content: str, goal: str) -> str:
        """Call LLM API to extract information.

        Args:
            content: Webpage content to process.
            goal: Extraction goal.

        Returns:
            LLM response content.

        Raises:
            Exception: If API call fails.
        """
        session = await self._get_session()

        messages = [{
            "role": "user",
            "content": EXTRACTOR_PROMPT.format(webpage_content=content, goal=goal)
        }]

        headers = {
            "Authorization": f"Bearer {TOOL_SERVER_LLM_API_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": TOOL_SERVER_LLM_MODEL,
            "messages": messages,
            "temperature": 0.7
        }

        async with session.post(
            f"{TOOL_SERVER_LLM_BASE_URL}/chat/completions",
            headers=headers,
            json=payload
        ) as response:
            if response.status == 200:
                data = await response.json()
                return data['choices'][0]['message']['content']
            else:
                text = await response.text()
                raise Exception(f"LLM API error: {response.status} - {text}")

    async def __aenter__(self):
        await self._get_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._session and not self._session.closed:
            await self._session.close()
