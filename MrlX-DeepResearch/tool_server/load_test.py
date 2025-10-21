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

"""Load testing utility for tool server concurrent requests."""

import asyncio
import time

import aiohttp


async def test_concurrent_requests(num_requests=50):
    """Test concurrent requests to tool server.

    Args:
        num_requests: Number of concurrent requests to make. Defaults to 50.
    """
    async with aiohttp.ClientSession() as session:
        tasks = []
        for i in range(num_requests):
            if i % 2 == 0:
                task = session.post(
                    "http://localhost:50001/retrieve",
                    json={"name": "search", "query": f"test {i}"}
                )
            else:
                task = session.post(
                    "http://localhost:50001/retrieve",
                    json={
                        "name": "visit",
                        "url": "https://example.com",
                        "goal": f"test {i}"
                    }
                )
            tasks.append(task)

        start = time.time()
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        duration = time.time() - start

        success = sum(1 for r in responses if not isinstance(r, Exception) and r.status == 200)
        print(f"Completed {num_requests} requests in {duration:.2f} seconds")
        print(f"Success rate: {success/num_requests*100:.1f}%")
        print(f"Average response time: {duration/num_requests:.3f} seconds")


if __name__ == "__main__":
    asyncio.run(test_concurrent_requests(100))
