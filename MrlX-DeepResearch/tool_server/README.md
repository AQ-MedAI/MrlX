# Tool Server

A high-performance async tool server for web search and content extraction.

## Features

- **Async Search Tool**: Google search integration with language detection
- **Async Visit Tool**: Web content extraction using Jina API
- **Centralized Configuration**: All environment variables managed in one place
- **High Performance**: Optimized for concurrent requests with semaphore controls
- **JSON Logging**: Request logging for monitoring and debugging

## Quick Start

### 1. Set Environment Variables

```bash
# Required API Keys
export GOOGLE_SEARCH_KEY="your_google_search_key"
export JINA_API_KEY="your_jina_key"
export TOOL_SERVER_LLM_API_KEY="your_llm_api_key"
export TOOL_SERVER_LLM_BASE_URL="https://openrouter.ai/api/v1"
export TOOL_SERVER_LLM_MODEL="deepseek/deepseek-chat-v3-0324"

# Optional Configuration
export MAX_CONCURRENT_REQUESTS=2000
export REQUEST_TIMEOUT=180
export VISIT_SEMAPHORE_LIMIT=200
export SEARCH_SEMAPHORE_LIMIT=500
```

### 2. Install Dependencies

```bash
pip install aiohttp aiofiles uvloop fastapi uvicorn httptools aiodns
```

### 3. Start Server

```bash
# Using the start script
./start_server.sh

# Or directly
python tool_server.py
```

## API Endpoints

### POST /retrieve
Main endpoint for tool calls.

**Search Tool:**
```json
{
    "name": "search",
    "query": "your search query"
}
```

**Visit Tool:**
```json
{
    "name": "visit",
    "url": "https://example.com",
    "goal": "extract specific information"
}
```

### GET /health
Health check endpoint with configuration status.

### GET /
Basic server information and status.

### POST /cache/clear
Clear search cache (if available).

## Configuration

All configuration is managed through `env_config.py`. Key settings:

- **Server Settings**: Host, port, workers, concurrency limits
- **API Keys**: Google Search, Jina, LLM Provider
- **Tool Settings**: Semaphore limits, timeouts, content length limits

## Architecture

- **tool_server.py**: Main FastAPI application
- **tool_search_async.py**: Async search implementation
- **tool_visit_async.py**: Async web content extraction
- **env_config.py**: Centralized environment configuration

## Performance

- **Concurrent Requests**: Up to 2000 concurrent requests
- **Workers**: 16 worker processes by default
- **Optimizations**: uvloop, httptools, aiohttp with connection pooling
- **Logging**: JSON request logs for monitoring

## Monitoring

- Request logs saved to `logs/requests/`
- Server logs saved to `logs/tool_server.log`
- Health endpoint provides configuration status
- JSON logging for request analysis
