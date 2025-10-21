#!/bin/bash

echo "=========================================="
echo "  Main Agent Quick Start"
echo "=========================================="

# Get script directory
QUICKSTART_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Check if .env file exists
if [ ! -f "$QUICKSTART_ROOT/.env" ]; then
    echo "Error: .env file not found"
    echo ""
    echo "Please follow these steps:"
    echo "1. Get Sub Agent IP: hostname -i (in sub container)"
    echo "2. Copy template: cp env.example .env"
    echo "3. Edit .env file and fill in all required variables"
    echo "4. Copy .env file to both main and sub containers"
    exit 1
fi

# 1. Initialize Main Agent environment variables first
echo ""
echo "Step 1/5: Initializing Main Agent environment variables..."
source init_env/init_main_agent_env.sh

# 2. Install slime dependencies
echo ""
echo "Step 2/5: Installing slime dependencies..."

# Check if SLIME_DIR is set (after loading .env)
if [ -z "$SLIME_DIR" ]; then
    echo "Error: SLIME_DIR environment variable is not set"
    echo ""
    echo "Please set SLIME_DIR in your .env file:"
    echo "1. Clone slime framework in BOTH containers:"
    echo "   git clone https://github.com/THUDM/slime.git"
    echo "   cd slime"
    echo "   echo \"SLIME_DIR=\$(pwd)\""
    echo "2. Copy the output path and set SLIME_DIR in .env file"
    echo "3. Recommended: Use the same path in both containers (same .env file)"
    echo "4. Example: SLIME_DIR=/path/to/your/slime"
    exit 1
fi

# Check if SLIME_DIR exists
if [ ! -d "$SLIME_DIR" ]; then
    echo "Error: SLIME_DIR directory does not exist: $SLIME_DIR"
    echo "Please check your SLIME_DIR setting in .env file"
    exit 1
fi

cd "$SLIME_DIR"
pip install -e .
pip install distro chardet

echo "SLIME_DIR: $SLIME_DIR"

# 3. Check and start Tool Server
echo ""
echo "Step 3/5: Checking Tool Server status..."
TOOL_PORT=${TOOL_SERVER_PORT:-50001}

# Check if tool server is already running
TOOL_RUNNING=false
if command -v lsof &> /dev/null; then
    if lsof -Pi :$TOOL_PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
        TOOL_RUNNING=true
    fi
elif command -v netstat &> /dev/null; then
    if netstat -tuln 2>/dev/null | grep -q ":$TOOL_PORT "; then
        TOOL_RUNNING=true
    fi
fi

if [ "$TOOL_RUNNING" = true ]; then
    echo "✓ Tool Server already running on port $TOOL_PORT, skip startup"
else
    echo "Tool Server not detected, starting now..."
    bash "$QUICKSTART_ROOT/quick_start_tool.sh"
    if [ $? -ne 0 ]; then
        echo "⚠️  Warning: Tool Server startup failed, but continuing..."
        echo "You can manually start it later: bash quick_start_tool.sh"
    fi
fi

# 4. Check Database Server connectivity
echo ""
echo "Step 4/5: Checking Database Server connectivity..."
DATABASE_SERVER_IP=${DATABASE_SERVER_IP:-${SUB_AGENT_IP}}
DATABASE_PORT=${DATABASE_SERVER_PORT:-18888}

echo "Attempting to connect to Database Server at $DATABASE_SERVER_IP:$DATABASE_PORT..."
MAX_RETRY=30
RETRY_COUNT=0
DATABASE_ACCESSIBLE=false

while [ $RETRY_COUNT -lt $MAX_RETRY ]; do
    if command -v curl &> /dev/null; then
        if curl -s --connect-timeout 2 http://$DATABASE_SERVER_IP:$DATABASE_PORT/health >/dev/null 2>&1; then
            DATABASE_ACCESSIBLE=true
            break
        fi
    elif command -v nc &> /dev/null; then
        if nc -z -w 2 $DATABASE_SERVER_IP $DATABASE_PORT >/dev/null 2>&1; then
            DATABASE_ACCESSIBLE=true
            break
        fi
    else
        echo "Warning: Neither curl nor nc available, skipping Database Server connectivity check"
        DATABASE_ACCESSIBLE=true
        break
    fi

    if [ $((RETRY_COUNT % 5)) -eq 0 ]; then
        echo "Waiting for Database Server to be accessible... ($RETRY_COUNT/$MAX_RETRY attempts)"
    fi
    sleep 1
    RETRY_COUNT=$((RETRY_COUNT + 1))
done

if [ "$DATABASE_ACCESSIBLE" = true ]; then
    echo "Database Server is accessible at $DATABASE_SERVER_IP:$DATABASE_PORT"
else
    echo "Warning: Cannot connect to Database Server at $DATABASE_SERVER_IP:$DATABASE_PORT"
    echo "Please ensure:"
    echo "  1. Database Server is running on Sub Agent container"
    echo "  2. DATABASE_SERVER_IP or SUB_AGENT_IP is correctly set in .env"
    echo "  3. Network connectivity between containers is working"
    echo ""
    echo "Continuing anyway, but Main Agent may fail if Database Server is required..."
fi

# 4. Start Main Agent training
echo ""
echo "Step 4/5: Starting Main Agent training..."
echo "=========================================="
echo "Main Agent environment is ready"
echo ""
echo "IMPORTANT: Make sure Sub Agent is already running!"
echo ""
echo "If Sub Agent is not running, start it first in the sub container:"
echo "  bash quick_start_sub.sh"
echo "=========================================="
echo ""

# Copy agent_co_train to slime/examples directory
echo "Copying to $SLIME_DIR/examples/agent_co_train..."
# mkdir -p "$SLIME_DIR/examples"
# rm -rf "$SLIME_DIR/examples/agent_co_train"
cp -r "$QUICKSTART_ROOT/agent_co_train" "$SLIME_DIR/examples"

# Run Main Agent training from slime/examples/agent_co_train
cd "$SLIME_DIR/examples/agent_co_train"
bash run.sh main
