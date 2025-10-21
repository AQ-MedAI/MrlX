#!/bin/bash

# Get script directory
INIT_ENV_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Load environment variable configuration file
ENV_FILE="$INIT_ENV_DIR/../.env"

if [ -f "$ENV_FILE" ]; then
    source "$ENV_FILE"
    echo "Loaded environment configuration: $ENV_FILE"
else
    echo "Error: .env file not found"
    echo "Please copy env.example to .env and fill in the required variables"
    exit 1
fi

# General environment variable configuration
export SUMMARY_LLM_API_BASE=${SUMMARY_LLM_API_BASE}
export SUMMARY_LLM_MODEL=${SUMMARY_LLM_MODEL}
export SUMMARY_LLM_API_KEY=${SUMMARY_LLM_API_KEY}
export KEY_SUFFIX=${KEY_SUFFIX}
export SLIME_DIR=${SLIME_DIR}

echo "General environment variables set"
