#!/bin/bash

# Load general environment variables
INIT_ENV_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$INIT_ENV_DIR/init_general_env.sh"

# Database Server specific configuration
export DATABASE_SERVER_HOST=${DATABASE_SERVER_HOST:-0.0.0.0}
export DATABASE_SERVER_PORT=${DATABASE_SERVER_PORT:-18888}

# If DATABASE_SERVER_IP not set, use SUB_AGENT_IP (for local deployment on sub agent container)
if [ -z "$DATABASE_SERVER_IP" ]; then
    SUB_AGENT_IP_AUTO=$(hostname -i | awk '{print $1}')
    export DATABASE_SERVER_IP=${SUB_AGENT_IP:-$SUB_AGENT_IP_AUTO}
fi

echo "Database Server environment variables set"
echo "  - DATABASE_SERVER_HOST: $DATABASE_SERVER_HOST"
echo "  - DATABASE_SERVER_PORT: $DATABASE_SERVER_PORT"
echo "  - DATABASE_SERVER_IP: $DATABASE_SERVER_IP"
