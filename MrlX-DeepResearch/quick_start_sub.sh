#!/bin/bash

echo "=========================================="
echo "  Sub Agent Quick Start"
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

# 1. Initialize Sub Agent environment variables first
echo ""
echo "Step 1/5: Initializing Sub Agent environment variables..."
source init_env/init_sub_agent_env.sh

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

# 3. Start Database Server
echo ""
echo "Step 3/5: Starting Database Server..."
DATABASE_PORT=${DATABASE_SERVER_PORT:-18888}

# Check if database server is already running
DATABASE_RUNNING=false
if command -v lsof &> /dev/null; then
    if lsof -Pi :$DATABASE_PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
        DATABASE_RUNNING=true
    fi
elif command -v netstat &> /dev/null; then
    if netstat -tuln 2>/dev/null | grep -q ":$DATABASE_PORT "; then
        DATABASE_RUNNING=true
    fi
fi

if [ "$DATABASE_RUNNING" = true ]; then
    echo "Database Server already running on port $DATABASE_PORT, skip startup"
else
    echo "Database Server not detected, starting now..."
    # Create logs directory if not exists
    mkdir -p logs
    # Start database server in background
    cd "$QUICKSTART_ROOT/db"
    nohup python database_server.py > "$QUICKSTART_ROOT/logs/database_server.log" 2>&1 &
    DATABASE_PID=$!
    cd "$QUICKSTART_ROOT"
    echo "Database Server started with PID: $DATABASE_PID"
    echo "Waiting for Database Server to be ready..."

    # Wait for database server to be ready (max 5 minutes)
    MAX_WAIT=300  # 5 minutes
    WAIT_COUNT=0
    DATABASE_READY=false

    while [ $WAIT_COUNT -lt $MAX_WAIT ]; do
        # Check if database server is listening on port or health check
        if command -v curl &> /dev/null; then
            if curl -s http://localhost:$DATABASE_PORT/health >/dev/null 2>&1; then
                DATABASE_READY=true
                break
            fi
        elif command -v lsof &> /dev/null; then
            if lsof -Pi :$DATABASE_PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
                DATABASE_READY=true
                break
            fi
        elif command -v netstat &> /dev/null; then
            if netstat -tuln 2>/dev/null | grep -q ":$DATABASE_PORT "; then
                DATABASE_READY=true
                break
            fi
        fi

        if [ $((WAIT_COUNT % 10)) -eq 0 ]; then
            echo "Waiting for Database Server to be ready... ($WAIT_COUNT/$MAX_WAIT seconds)"
        fi
        sleep 1
        WAIT_COUNT=$((WAIT_COUNT + 1))
    done

    if [ "$DATABASE_READY" = true ]; then
        echo "Database Server is ready and listening on port $DATABASE_PORT (took $WAIT_COUNT seconds)"
    else
        echo "Error: Database Server failed to start within $MAX_WAIT seconds"
        echo "Please check logs/database_server.log for details"
        exit 1
    fi
fi

# 4. Start Router
echo ""
echo "Step 4/5: Starting Router..."
ROUTER_PORT=${SUB_AGENT_PORT:-3333}

# Check if router is already running
ROUTER_RUNNING=false
if command -v lsof &> /dev/null; then
    if lsof -Pi :$ROUTER_PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
        ROUTER_RUNNING=true
    fi
elif command -v netstat &> /dev/null; then
    if netstat -tuln 2>/dev/null | grep -q ":$ROUTER_PORT "; then
        ROUTER_RUNNING=true
    fi
fi

if [ "$ROUTER_RUNNING" = true ]; then
    echo "Router already running on port $ROUTER_PORT, skip startup"
else
    echo "Router not detected, starting now..."
    # Create logs directory if not exists
    mkdir -p logs
    # Start router in background
    nohup bash "$QUICKSTART_ROOT/agent_co_train/start_router.sh" > logs/router.log 2>&1 &
    ROUTER_PID=$!
    echo "Router started with PID: $ROUTER_PID"
    echo "Waiting for Router to be ready..."

    # Wait for router to be ready (max 5 minutes)
    MAX_WAIT=300  # 5 minutes
    WAIT_COUNT=0
    ROUTER_READY=false

    while [ $WAIT_COUNT -lt $MAX_WAIT ]; do
        # Check if router is listening on port
        if command -v lsof &> /dev/null; then
            if lsof -Pi :$ROUTER_PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
                ROUTER_READY=true
                break
            fi
        elif command -v netstat &> /dev/null; then
            if netstat -tuln 2>/dev/null | grep -q ":$ROUTER_PORT "; then
                ROUTER_READY=true
                break
            fi
        fi

        if [ $((WAIT_COUNT % 10)) -eq 0 ]; then
            echo "Waiting for router to be ready... ($WAIT_COUNT/$MAX_WAIT seconds)"
        fi
        sleep 1
        WAIT_COUNT=$((WAIT_COUNT + 1))
    done

    if [ "$ROUTER_READY" = true ]; then
        echo "Router is ready and listening on port $ROUTER_PORT (took $WAIT_COUNT seconds)"
    else
        echo "Error: Router failed to start within $MAX_WAIT seconds"
        echo "Please check logs/router.log for details"
        exit 1
    fi
fi

# 5. Start Sub Agent training
echo ""
echo "Step 5/5: Starting Sub Agent training..."
echo "=========================================="
echo "Sub Agent environment is ready"
echo ""
echo "IMPORTANT: Sub Agent is starting..."
echo "After Sub Agent is running, start Main Agent in the main container:"
echo "  bash quick_start_main.sh"
echo "=========================================="
echo ""

# Copy agent_co_train to slime/examples directory
echo "Copying to $SLIME_DIR/examples/agent_co_train..."
# mkdir -p "$SLIME_DIR/examples"
# rm -rf "$SLIME_DIR/examples/agent_co_train"
cp -r "$QUICKSTART_ROOT/agent_co_train" "$SLIME_DIR/examples"

# Run Sub Agent training from slime/examples/agent_co_train
cd "$SLIME_DIR/examples/agent_co_train"
bash run.sh sub
