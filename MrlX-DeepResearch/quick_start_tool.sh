#!/bin/bash

echo "=========================================="
echo "  Tool Server Quick Start"
echo "=========================================="

# Get script directory
QUICKSTART_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Check if .env file exists
if [ ! -f "$QUICKSTART_ROOT/.env" ]; then
    echo "Error: .env file not found"
    echo ""
    echo "Please follow these steps:"
    echo "1. Copy template: cp env.example .env"
    echo "2. Edit .env file and fill in all required variables"
    exit 1
fi

# Initialize Tool Server environment variables
echo ""
echo "Step 1/3: Initializing Tool Server environment variables..."
source "$QUICKSTART_ROOT/init_env/init_tool_server_env.sh"

# Check if tool server is already running
echo ""
echo "Step 2/3: Checking if Tool Server is already running..."
TOOL_PORT=${TOOL_SERVER_PORT:-50001}

if command -v lsof &> /dev/null; then
    if lsof -Pi :$TOOL_PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
        echo "⚠️  Tool Server is already running on port $TOOL_PORT"
        echo "If you want to restart, please stop the existing process first:"
        echo "  lsof -ti:$TOOL_PORT | xargs kill -9"
        exit 0
    fi
elif command -v netstat &> /dev/null; then
    if netstat -tuln 2>/dev/null | grep -q ":$TOOL_PORT "; then
        echo "⚠️  Tool Server is already running on port $TOOL_PORT"
        echo "If you want to restart, please stop the existing process first"
        exit 0
    fi
else
    echo "Warning: Cannot detect if port is in use (lsof/netstat not available)"
fi

# Start Tool Server
echo ""
echo "Step 3/3: Starting Tool Server..."
cd "$QUICKSTART_ROOT/tool_server"

# Create logs directory if not exists
mkdir -p logs

# Start tool server in background
nohup python tool_server.py > logs/tool_server.log 2>&1 &
TOOL_PID=$!
echo "Tool Server started with PID: $TOOL_PID"

# Wait for server to be ready (check health endpoint)
echo "Waiting for Tool Server to be ready..."
MAX_WAIT=30
WAIT_COUNT=0

while [ $WAIT_COUNT -lt $MAX_WAIT ]; do
    if curl -s http://localhost:$TOOL_PORT/health >/dev/null 2>&1; then
        echo "✓ Tool Server is ready!"
        echo "=========================================="
        echo "Tool Server started successfully"
        echo "  - Host: $TOOL_SERVER_HOST"
        echo "  - Port: $TOOL_PORT"
        echo "  - PID: $TOOL_PID"
        echo "  - Logs: $QUICKSTART_ROOT/tool_server/logs/tool_server.log"
        echo "=========================================="
        exit 0
    fi
    sleep 1
    WAIT_COUNT=$((WAIT_COUNT + 1))
    echo -n "."
done

echo ""
echo "⚠️  Tool Server may not be fully ready yet"
echo "Check logs: tail -f $QUICKSTART_ROOT/tool_server/logs/tool_server.log"
