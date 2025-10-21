#!/bin/bash

# System optimization
ulimit -n 65535
ulimit -u 32768
echo "File descriptor limit set to: $(ulimit -n)"
echo "Process limit set to: $(ulimit -u)"

# Environment variables
export PYTHONUNBUFFERED=1
export MAX_CONCURRENT_REQUESTS=2000
export REQUEST_TIMEOUT=180
export VISIT_SEMAPHORE_LIMIT=200
export SEARCH_SEMAPHORE_LIMIT=500

# Create necessary directories
mkdir -p logs

# Install dependencies
pip install aiohttp aiofiles uvloop fastapi uvicorn httptools aiodns

# Start server
echo "Starting tool server..."
echo "Workers: 16"
echo "Max concurrent requests: $MAX_CONCURRENT_REQUESTS"
echo "Visit semaphore limit per worker: $VISIT_SEMAPHORE_LIMIT"
echo "Search semaphore limit per worker: $SEARCH_SEMAPHORE_LIMIT"
echo "Using uvloop + httptools for maximum performance"
echo "Press Ctrl+C to stop the server"
python tool_server.py
