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

"""FastAPI-based task queue server for agent communication."""

import json
import threading
from typing import Dict, List

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel


app = FastAPI()

storage: Dict[str, List[str]] = {}
lock = threading.Lock()


class TaskFetchRequest(BaseModel):
    """Request model for task fetch endpoint."""

    listKey: str


class TaskCommitRequest(BaseModel):
    """Request model for task commit endpoint."""

    listKey: str
    taskData: str


@app.get("/health")
def health_check():
    """Health check endpoint.

    Returns:
        JSON response indicating service health.
    """
    return {"status": "healthy", "service": "database_server"}


@app.post("/taskFetch")
def task_fetch(payload: TaskFetchRequest):
    """Fetch a task from the queue.

    Args:
        payload: Request payload containing the list key.

    Returns:
        JSON response with task data or error message.
    """
    list_key = payload.listKey

    with lock:
        if list_key not in storage or not storage[list_key]:
            return JSONResponse(
                status_code=200,
                content={"success": False, "data": {"success": False, "errorMsg": "Queue is empty"}},
            )

        try:
            task_data = storage[list_key].pop(0)
            return {"success": True, "data": {"success": True, "taskData": task_data}}
        except Exception:
            return JSONResponse(
                status_code=500,
                content={"success": False, "data": {"success": False, "errorMsg": "System error"}},
            )


@app.post("/taskCommit")
def task_commit(payload: TaskCommitRequest):
    """Commit a task to the queue.

    Args:
        payload: Request payload containing the list key and task data.

    Returns:
        JSON response indicating success or failure.
    """
    list_key = payload.listKey
    task_data = payload.taskData

    with lock:
        if list_key not in storage:
            storage[list_key] = []

        try:
            json.loads(task_data)
            storage[list_key].append(task_data)
            return {"success": True, "data": {"success": True}}
        except json.JSONDecodeError:
            return JSONResponse(
                status_code=400,
                content={"success": False, "data": {"success": False, "errorMsg": "Invalid taskData JSON"}},
            )
        except Exception:
            return JSONResponse(
                status_code=500,
                content={"success": False, "data": {"success": False, "errorMsg": "Server error"}},
            )


if __name__ == "__main__":
    import os
    import uvicorn

    host = os.getenv("DATABASE_SERVER_HOST", "0.0.0.0")
    port = int(os.getenv("DATABASE_SERVER_PORT", "18888"))
    uvicorn.run("database_server:app", host=host, port=port, reload=False)
