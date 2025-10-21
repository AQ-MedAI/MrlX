#!/bin/bash

# Load general environment variables
INIT_ENV_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$INIT_ENV_DIR/init_general_env.sh"

# Sub Agent specific configuration
# Automatically get local IP for confirmation
SUB_AGENT_IP_AUTO=$(hostname -i | awk '{print $1}')
export SUB_AGENT_IP=${SUB_AGENT_IP_AUTO}

# Sub Agent Model Checkpoints (Required)
export SUB_HF_CHECKPOINT=${SUB_HF_CHECKPOINT}
export SUB_REF_LOAD=${SUB_REF_LOAD}

# Sub Agent Paths (Optional)
if [ -n "${SUB_CHECKPOINT_DIR}" ]; then
    export SUB_CHECKPOINT_DIR=${SUB_CHECKPOINT_DIR}
fi

if [ -n "${SUB_PROMPT_DATA}" ]; then
    export SUB_PROMPT_DATA=${SUB_PROMPT_DATA}
fi

echo "Sub Agent environment variables set"
echo "  - SUB_AGENT_IP (local): $SUB_AGENT_IP"
echo "  - SUB_HF_CHECKPOINT: $SUB_HF_CHECKPOINT"
echo "  - SUB_REF_LOAD: $SUB_REF_LOAD"
if [ -n "${SUB_CHECKPOINT_DIR}" ]; then
    echo "  - SUB_CHECKPOINT_DIR: $SUB_CHECKPOINT_DIR"
else
    echo "  - SUB_CHECKPOINT_DIR: (using default)"
fi
if [ -n "${SUB_PROMPT_DATA}" ]; then
    echo "  - SUB_PROMPT_DATA: $SUB_PROMPT_DATA"
else
    echo "  - SUB_PROMPT_DATA: (using default)"
fi
