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

"""Asynchronous search tool implementation."""

import asyncio
import json
import os
import re
from typing import List, Union

import aiohttp

from env_config import config


GOOGLE_SEARCH_KEY = config.get("GOOGLE_SEARCH_KEY")

FILTERED_DOMAINS = {
    "huggingface.co/datasets",
}


class SearchAsync:
    """Asynchronous web search tool using Google Search API."""

    name = "search"
    description = "Performs batched web searches asynchronously with formatted results."

    def __init__(self):
        """Initialize async search tool."""
        self._session = None
        self._semaphore = asyncio.Semaphore(500)

    async def _get_session(self):
        """Get or create aiohttp session.

        Returns:
            Active aiohttp ClientSession.
        """
        if self._session is None or self._session.closed:
            connector = aiohttp.TCPConnector(limit=1000, limit_per_host=500, ttl_dns_cache=300, use_dns_cache=True)
            timeout = aiohttp.ClientTimeout(total=60, connect=10, sock_read=30)
            self._session = aiohttp.ClientSession(connector=connector, timeout=timeout)
        return self._session

    def _detect_language(self, text: str) -> str:
        """Detect if text is primarily Chinese.

        Args:
            text: Input text string.

        Returns:
            'zh' for Chinese, 'en' for English.
        """
        if not text or len(text) == 0:
            return 'en'
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        total_chars = len(text)
        if total_chars > 0 and chinese_chars / total_chars > 0.3:
            return 'zh'
        else:
            return 'en'

    async def call(self, query: Union[str, List[str]], **kwargs) -> str:
        """Perform search queries asynchronously.

        Args:
            query: Single query string or list of query strings.
            **kwargs: Additional arguments.

        Returns:
            Formatted search results as string.
        """
        queries = [query] if isinstance(query, str) else query

        max_queries = config.get("SEARCH_MAX_QUERIES")
        queries_to_process = queries[:max_queries]

        tasks = [self._search_single(q) for q in queries_to_process]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                all_results.append(f"Error searching '{queries_to_process[i]}': {str(result)}")
            else:
                all_results.append(result)

        return "\n=======\n".join(all_results)

    async def _search_single(self, query: str) -> str:
        """Execute a single search query.

        Args:
            query: Search query string.

        Returns:
            Formatted search results.

        Raises:
            Exception: If search fails.
        """
        async with self._semaphore:
            try:
                session = await self._get_session()
                headers = {'X-API-KEY': GOOGLE_SEARCH_KEY, 'Content-Type': 'application/json'}
                data = {"q": query, "num": 10}

                if self._detect_language(query) == 'zh':
                    data['gl'] = 'cn'

                async with session.post("https://google.serper.dev/search", headers=headers, json=data) as response:
                    response.raise_for_status()
                    results = await response.json()
                    return self._format_results(query, results)

            except asyncio.TimeoutError:
                raise Exception("Search request timeout")
            except Exception as e:
                raise Exception(f"Search error: {str(e)}")

    def _format_results(self, query: str, results: dict) -> str:
        """Format search results into readable text.

        Args:
            query: Original search query.
            results: Search results dictionary from API.

        Returns:
            Formatted search results string.
        """
        if "organic" not in results:
            return f"No results found for '{query}'. Try with a more general query, or remove the year filter."

        web_snippets = []
        for idx, page in enumerate(results["organic"], 1):
            page_link = page.get('link', '')
            if any(domain in page_link for domain in FILTERED_DOMAINS):
                continue

            date_published = f"\nDate published: {page['date']}" if "date" in page else ""
            source = f"\nSource: {page['source']}" if "source" in page else ""
            snippet = f"\n{page.get('snippet', '')}"

            redacted_version = f"{idx}. [{page.get('title', 'No Title')}]({page_link}){date_published}{source}{snippet}"

            redacted_version = redacted_version.replace("Your browser can't play this video.", "")
            web_snippets.append(redacted_version)

        content = f"A Google search for '{query}' found {len(web_snippets)} results:\n\n## Web Results\n" + "\n\n".join(web_snippets)
        return content

    async def close_session(self):
        """Close the aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()
