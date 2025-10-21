"""
Module: database_server
-----------------------
In-memory task queue service for patient simulation workflow.

Provides:
    - /taskFetch  : Fetch the next task from a queue (FIFO order).
    - /taskCommit : Commit/append a new task to a queue.

Features:
    - Thread-safe operations using a global Lock (`threading.Lock`).
    - Queues identified uniquely by `listKey`.
    - Tasks stored as JSON strings (validated on commit).
    - Ephemeral storage: all queues reset when process restarts.
"""

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import threading
import json
import uvicorn

from typing import Dict, List
from config import global_config

# --------------------------------------------------------------------------- #
#  Application setup & in-memory storage                                      #
# --------------------------------------------------------------------------- #
app = FastAPI()

# Dictionary storing task queues:
#   key   -> listKey (string identifier of queue)
#   value -> list of taskData JSON strings (FIFO order)
storage: Dict[str, List[str]] = {}

# Lock to ensure thread-safe access to `storage`
lock = threading.Lock()


# --------------------------------------------------------------------------- #
#  Request Models                                                              #
# --------------------------------------------------------------------------- #
class TaskFetchRequest(BaseModel):
    """
    Request body for /taskFetch API.

    Attributes:
        listKey (str): Key identifying the queue to fetch from.
    """
    listKey: str


class TaskCommitRequest(BaseModel):
    """
    Request body for /taskCommit API.

    Attributes:
        listKey (str): Key identifying the queue to commit to.
        taskData (str): JSON-formatted task content.
    """
    listKey: str
    taskData: str


# --------------------------------------------------------------------------- #
#  API Endpoints                                                               #
# --------------------------------------------------------------------------- #
@app.post("/taskFetch")
def task_fetch(payload: TaskFetchRequest):
    """
    Fetch and remove the first task in the specified queue (FIFO order).

    Args:
        payload (TaskFetchRequest): Incoming request containing `listKey`.

    Returns:
        JSON object:
            - success=True + taskData if a task exists.
            - success=False + error if the queue is empty.
            - success=False + "System error" if an unexpected exception occurs.
    """
    list_key = payload.listKey

    with lock:
        # Check if queue exists and has tasks
        if list_key not in storage or not storage[list_key]:
            return JSONResponse(
                status_code=200,
                content={"success": False, "data": {"success": False, "errorMsg": "Queue is empty"}},
            )

        try:
            # Pop the first task (FIFO)
            task_data = storage[list_key].pop(0)
            return {"success": True, "data": {"success": True, "taskData": task_data}}
        except Exception:
            return JSONResponse(
                status_code=500,
                content={"success": False, "data": {"success": False, "errorMsg": "System error"}},
            )


@app.post("/taskCommit")
def task_commit(payload: TaskCommitRequest):
    """
    Commit/append a new task to the specified queue.

    Validates that taskData is a proper JSON string before storing.

    Args:
        payload (TaskCommitRequest): Incoming request with `listKey` and `taskData`.

    Returns:
        JSON object:
            - success=True if committed successfully.
            - success=False + "Invalid taskData JSON" if validation fails.
            - success=False + "Server error" if any unexpected exception occurs.
    """
    list_key = payload.listKey
    task_data = payload.taskData

    with lock:
        # Create queue if it does not exist
        if list_key not in storage:
            storage[list_key] = []

        try:
            # Validate taskData JSON format
            json.loads(task_data)

            # Append the task to the queue
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


# --------------------------------------------------------------------------- #
#  Main Entrypoint                                                             #
#  Run this module directly to start the FastAPI server:                       #
#      $ python database_server.py                                             #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    
    DATABASE_SERVER_IP = global_config.DATABASE_SERVER_IP
    uvicorn.run("database_server:app", host=DATABASE_SERVER_IP, port=18888, reload=False)