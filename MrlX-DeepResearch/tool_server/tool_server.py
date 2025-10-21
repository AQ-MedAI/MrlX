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

"""FastAPI-based async tool call service for search and visit operations."""

import asyncio
import json
import json as stdlib_json
import logging
import os
import sys
import threading
import time
import uuid
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler
from typing import Any, Dict, List, Optional, Union

from fastapi import FastAPI, HTTPException
from fastapi import Request as FastAPIRequest
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from env_config import config


current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

try:
    from tool_search_async import SearchAsync
    from tool_visit_async import VisitAsync
    TOOLS_AVAILABLE = True
    print("Successfully imported async tool classes")
except ImportError as e:
    TOOLS_AVAILABLE = False
    print(f"Warning: Could not import tool classes: {e}")
    print("Falling back to error responses for tool calls")

# Environment Variables from centralized config
MAX_CONCURRENT_REQUESTS = config.get("MAX_CONCURRENT_REQUESTS")
REQUEST_TIMEOUT = config.get("REQUEST_TIMEOUT")
VISIT_SEMAPHORE_LIMIT = config.get("VISIT_SEMAPHORE_LIMIT")
SEARCH_SEMAPHORE_LIMIT = config.get("SEARCH_SEMAPHORE_LIMIT")

# API Keys from centralized config
WEBCONTENT_MAXLENGTH = config.get("WEBCONTENT_MAXLENGTH")
JINA_API_KEYS = config.get("JINA_API_KEY")
GOOGLE_SEARCH_KEY = config.get("GOOGLE_SEARCH_KEY")
TOOL_SERVER_LLM_API_KEY = config.get("TOOL_SERVER_LLM_API_KEY")
TOOL_SERVER_LLM_BASE_URL = config.get("TOOL_SERVER_LLM_BASE_URL")
TOOL_SERVER_LLM_MODEL = config.get("TOOL_SERVER_LLM_MODEL")

# Global variables
active_requests = {}

class JSONRequestLogger:
    """JSON request logger for each worker process."""

    def __init__(self, worker_id: str = None):
        """Initialize JSON request logger.

        Args:
            worker_id: Worker identifier. Defaults to worker_{pid}.
        """
        self.worker_id = worker_id or f"worker_{os.getpid()}"
        self.log_dir = os.path.join(os.path.dirname(__file__), "logs", "requests")
        os.makedirs(self.log_dir, exist_ok=True)

        today = datetime.now().strftime("%Y%m%d")
        self.log_file = os.path.join(self.log_dir, f"requests_{self.worker_id}_{today}.jsonl")

        self._lock = threading.Lock()

        print(f"JSON Request Logger initialized for {self.worker_id} -> {self.log_file}")

    def log_request(self, request_data: Dict[str, Any]):
        """Log request data to JSON file.

        Args:
            request_data: Dictionary containing request information.
        """
        try:
            with self._lock:
                with open(self.log_file, 'a', encoding='utf-8') as f:
                    json.dump(request_data, f, ensure_ascii=False)
                    f.write('\n')
        except Exception as e:
            print(f"Error writing to JSON log: {e}")

    def get_log_file_path(self) -> str:
        """Get current log file path.

        Returns:
            Path to current log file.
        """
        return self.log_file

# Global instance
json_logger = JSONRequestLogger()

def setup_logging():
    """Configure logging with console and file handlers.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger("tool_server")
    logger.setLevel(logging.INFO)

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)

    log_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(log_dir, exist_ok=True)

    file_handler = TimedRotatingFileHandler(
        filename=os.path.join(log_dir, "tool_server.log"),
        when="midnight",
        interval=1,
        backupCount=7,
        encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    return logger


logger = setup_logging()


class SearchQueryPayload(BaseModel):
    """Request payload model for tool calls."""

    url: Optional[Union[str, List[str]]] = None
    goal: Optional[str] = None
    query: Optional[Union[str, List[str]]] = None
    name: str


class ToolCallApiResponse(BaseModel):
    """Response model for tool call results."""

    result: str


class UnicodeJSONResponse(JSONResponse):
    """Custom JSON response class with UTF-8 encoding."""

    def render(self, content) -> bytes:
        return stdlib_json.dumps(
            content,
            ensure_ascii=False,
            allow_nan=False,
            indent=None,
            separators=(",", ":"),
        ).encode("utf-8")

    @property
    def media_type(self) -> str:
        return "application/json; charset=utf-8"

# FastAPI App
app = FastAPI(
    title="Async Tool Call Service",
    description="An async tool call service with improved concurrency handling",
    version="3.0.0",
    default_response_class=UnicodeJSONResponse
)

# Semaphores for concurrency control
visit_semaphore = asyncio.Semaphore(VISIT_SEMAPHORE_LIMIT)
search_semaphore = asyncio.Semaphore(SEARCH_SEMAPHORE_LIMIT)

# Initialize tools
search_tool = None
visit_tool = None

if TOOLS_AVAILABLE:
    search_tool = SearchAsync()
    visit_tool = VisitAsync()

# Middleware
@app.middleware("http")
async def log_requests_middleware(request: FastAPIRequest, call_next):
    request_id = str(uuid.uuid4().hex[:8])
    request.state.request_id = request_id
    start_time = time.time()

    client_host = request.client.host if request.client else "unknown"
    logger.info(
        f"Request started  | ID: {request_id} | Client: {client_host} | Path: {request.method} {request.url.path}")

    response = await call_next(request)
    process_time = (time.time() - start_time) * 1000

    logger.info(
        f"Request finished | ID: {request_id} | Status: {response.status_code} | Duration: {process_time:.2f}ms"
    )

    return response

@app.post("/retrieve", response_model=ToolCallApiResponse, tags=["Tools"])
async def tool_results_retrieval(payload: SearchQueryPayload, request: FastAPIRequest):
    """Handle tool execution requests for search and visit operations.

    Args:
        payload: Tool request payload containing tool name and parameters.
        request: FastAPI request object.

    Returns:
        Tool execution results.

    Raises:
        HTTPException: If request fails or times out.
    """
    request_id = request.state.request_id if hasattr(request.state, 'request_id') else str(uuid.uuid4().hex[:8])
    start_time = time.time()
    client_host = request.client.host if request.client else "unknown"

    # Prepare JSON log data
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "request_id": request_id,
        "worker_id": json_logger.worker_id,
        "client_ip": client_host,
        "tool_name": payload.name,
        "request_payload": {
            "url": payload.url,
            "goal": payload.goal,
            "query": payload.query,
            "name": payload.name
        },
        "status": "processing",
        "error": None,
        "result_length": 0,
        "processing_time_ms": 0
    }

    logger.info(f"Processing payload | ID: {request_id} | Tool: {payload.name}")

    try:
        # Check concurrent limit
        if len(active_requests) > MAX_CONCURRENT_REQUESTS:
            error_msg = f"Server busy. Active requests: {len(active_requests)}"
            log_data.update({
                "status": "error",
                "error": error_msg,
                "processing_time_ms": (time.time() - start_time) * 1000
            })
            json_logger.log_request(log_data)
            raise HTTPException(status_code=503, detail=error_msg)

        # Validate tool name
        if payload.name not in ["search", "visit"]:
            error_msg = f"Error: Unknown tool '{payload.name}'. Supported tools: search, visit"
            log_data.update({
                "status": "error",
                "error": error_msg,
                "processing_time_ms": (time.time() - start_time) * 1000
            })
            json_logger.log_request(log_data)
            return ToolCallApiResponse(result=error_msg)

        # Handle search tool
        if payload.name == "search":
            if not TOOLS_AVAILABLE or search_tool is None:
                error_msg = "Error: Search tool not available. Please check dependencies."
                log_data.update({
                    "status": "error",
                    "error": error_msg,
                    "processing_time_ms": (time.time() - start_time) * 1000
                })
                json_logger.log_request(log_data)
                return ToolCallApiResponse(result=error_msg)

            if not payload.query:
                error_msg = "Error: 'query' parameter is required for search tool"
                log_data.update({
                    "status": "error",
                    "error": error_msg,
                    "processing_time_ms": (time.time() - start_time) * 1000
                })
                json_logger.log_request(log_data)
                return ToolCallApiResponse(result=error_msg)

            logger.info(f"Search tool input | ID: {request_id} | Query: {payload.query}")

            # Async call to search tool
            async with search_semaphore:
                result = await asyncio.wait_for(
                    search_tool.call(payload.query),
                    timeout=REQUEST_TIMEOUT
                )

            # Update log data
            processing_time = (time.time() - start_time) * 1000
            log_data.update({
                "status": "success",
                "result_length": len(result),
                "processing_time_ms": processing_time,
                "result_preview": result[:500] + "..." if len(result) > 500 else result
            })

            result_preview = result[:500] + "..." if len(result) > 500 else result
            logger.info(f"Search tool output | ID: {request_id} | Result length: {len(result)} | Preview: {result_preview}")

            json_logger.log_request(log_data)

            return ToolCallApiResponse(result=result)

        # Handle visit tool
        elif payload.name == "visit":
            if not TOOLS_AVAILABLE or visit_tool is None:
                error_msg = "Error: Visit tool not available. Please check dependencies."
                log_data.update({
                    "status": "error",
                    "error": error_msg,
                    "processing_time_ms": (time.time() - start_time) * 1000
                })
                json_logger.log_request(log_data)
                return ToolCallApiResponse(result=error_msg)

            if not payload.url:
                error_msg = "Error: 'url' parameter is required for visit tool"
                log_data.update({
                    "status": "error",
                    "error": error_msg,
                    "processing_time_ms": (time.time() - start_time) * 1000
                })
                json_logger.log_request(log_data)
                return ToolCallApiResponse(result=error_msg)

            logger.info(f"Visit tool input | ID: {request_id} | URL: {payload.url} | Goal: {payload.goal}")

            # Async call to visit tool
            visit_params = {"url": payload.url, "goal": payload.goal}
            async with visit_semaphore:
                result = await asyncio.wait_for(
                    visit_tool.call(visit_params),
                    timeout=REQUEST_TIMEOUT
                )

            processing_time = (time.time() - start_time) * 1000
            log_data.update({
                "status": "success",
                "result_length": len(result),
                "processing_time_ms": processing_time,
                "result_preview": result[:500] + "..." if len(result) > 500 else result
            })

            result_preview = result[:500] + "..." if len(result) > 500 else result
            logger.info(f"Visit tool output | ID: {request_id} | Result length: {len(result)} | Preview: {result_preview}")

            json_logger.log_request(log_data)

            return ToolCallApiResponse(result=result)

    except asyncio.TimeoutError:
        error_msg = f"Request timeout after {REQUEST_TIMEOUT} seconds"
        log_data.update({
            "status": "timeout",
            "error": error_msg,
            "processing_time_ms": REQUEST_TIMEOUT * 1000
        })
        json_logger.log_request(log_data)
        raise HTTPException(status_code=504, detail=error_msg)
    except Exception as e:
        error_msg = str(e)
        processing_time = (time.time() - start_time) * 1000
        log_data.update({
            "status": "error",
            "error": error_msg,
            "processing_time_ms": processing_time
        })
        json_logger.log_request(log_data)
        logger.error(f"Error processing request | ID: {request_id} | Error: {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)

@app.get("/", tags=["Health"])
async def read_root():
    """Get service information and status.

    Returns:
        Service metadata and configuration.
    """
    return {
        "message": "Async Tool Call Service is running with FastAPI.",
        "supported_tools": ["search", "visit"],
        "version": "3.0.0",
        "worker_id": json_logger.worker_id,
        "json_log_file": json_logger.get_log_file_path(),
        "active_requests": len(active_requests),
        "semaphore_limits": {
            "visit": VISIT_SEMAPHORE_LIMIT,
            "search": SEARCH_SEMAPHORE_LIMIT
        }
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint with detailed service status.

    Returns:
        Health status and configuration details.
    """
    return {
        "status": "healthy",
        "worker_id": json_logger.worker_id,
        "tools": ["search", "visit"],
        "api_keys_configured": config.get_api_keys_status(),
        "tools_available": TOOLS_AVAILABLE,
        "json_log_file": json_logger.get_log_file_path(),
        "required_keys_validation": config.validate_required_keys()
    }


@app.post("/cache/clear", tags=["Cache"])
async def clear_cache():
    """Clear search tool cache.

    Returns:
        Cache clear status message.

    Raises:
        HTTPException: If cache clearing fails.
    """
    try:
        if not TOOLS_AVAILABLE or search_tool is None:
            return {"message": "Search tool not available, no cache to clear"}

        if hasattr(search_tool, 'clear_cache'):
            await search_tool.clear_cache()
            return {"message": "Cache cleared successfully"}
        else:
            return {"message": "No cache to clear"}
    except Exception as e:
        logger.error(f"Error clearing cache: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error clearing cache: {str(e)}")

if __name__ == "__main__":
    import uvicorn

    host = config.get("HOST")
    port = config.get("PORT")
    workers = config.get("WORKERS")
    backlog = config.get("BACKLOG")
    limit_concurrency = config.get("LIMIT_CONCURRENCY")
    limit_max_requests = config.get("LIMIT_MAX_REQUESTS")

    logger.info("--------------------------------------------------")
    logger.info(f"Starting Async Tool Call Service on http://{host}:{port}")
    logger.info(f"API documentation: http://{host}:{port}/docs")
    logger.info("Supported tools: search, visit")
    logger.info(f"Concurrency limits - Visit: {VISIT_SEMAPHORE_LIMIT}, Search: {SEARCH_SEMAPHORE_LIMIT}")
    logger.info(f"Request timeout: {REQUEST_TIMEOUT} seconds")
    logger.info(f"Logs will be saved to 'logs/tool_server.log'")
    logger.info(f"JSON request logs will be saved to: {json_logger.get_log_file_path()}")
    logger.info(f"Worker ID: {json_logger.worker_id}")
    logger.info("--------------------------------------------------")

    uvicorn.run(
        "tool_server:app",
        host=host,
        port=port,
        reload=False,
        workers=workers,
        backlog=backlog,
        log_config=None,
        access_log=False,
        loop="uvloop",
        http="httptools",
        limit_concurrency=limit_concurrency,
        limit_max_requests=limit_max_requests,
    )
