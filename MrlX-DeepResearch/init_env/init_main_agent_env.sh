#!/bin/bash

# Load general environment variables
INIT_ENV_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$INIT_ENV_DIR/init_general_env.sh"

# Main Agent specific configuration
export JUDGE_LLM_API_BASE=${JUDGE_LLM_API_BASE}
export JUDGE_LLM_MODEL=${JUDGE_LLM_MODEL}
export JUDGE_LLM_API_KEY=${JUDGE_LLM_API_KEY}
export REASONER_LLM_API_BASE=${REASONER_LLM_API_BASE}
export REASONER_LLM_MODEL=${REASONER_LLM_MODEL}
export REASONER_LLM_API_KEY=${REASONER_LLM_API_KEY}
export SUB_AGENT_IP=${SUB_AGENT_IP}

# Main Agent Model Checkpoints (Required)
export MAIN_HF_CHECKPOINT=${MAIN_HF_CHECKPOINT}
export MAIN_REF_LOAD=${MAIN_REF_LOAD}

# Main Agent Paths (Optional)
if [ -n "${MAIN_CHECKPOINT_DIR}" ]; then
    export MAIN_CHECKPOINT_DIR=${MAIN_CHECKPOINT_DIR}
fi

if [ -n "${MAIN_PROMPT_DATA}" ]; then
    export MAIN_PROMPT_DATA=${MAIN_PROMPT_DATA}
fi

if [ -n "${MAIN_RAY_TEMP_DIR}" ]; then
    export MAIN_RAY_TEMP_DIR=${MAIN_RAY_TEMP_DIR}
fi

# Optional: Tool Server URL (only if deployed on separate container)
if [ -n "${RETRIEVAL_SERVICE_URL}" ]; then
    export RETRIEVAL_SERVICE_URL=${RETRIEVAL_SERVICE_URL}
fi

echo "Main Agent environment variables set"
echo "  - JUDGE_LLM_API_BASE: $JUDGE_LLM_API_BASE"
echo "  - JUDGE_LLM_MODEL: $JUDGE_LLM_MODEL"
echo "  - REASONER_LLM_API_BASE: $REASONER_LLM_API_BASE"
echo "  - REASONER_LLM_MODEL: $REASONER_LLM_MODEL"
echo "  - SUB_AGENT_IP: $SUB_AGENT_IP"
echo "  - MAIN_HF_CHECKPOINT: $MAIN_HF_CHECKPOINT"
echo "  - MAIN_REF_LOAD: $MAIN_REF_LOAD"
if [ -n "${MAIN_CHECKPOINT_DIR}" ]; then
    echo "  - MAIN_CHECKPOINT_DIR: $MAIN_CHECKPOINT_DIR"
else
    echo "  - MAIN_CHECKPOINT_DIR: (using default)"
fi
if [ -n "${MAIN_PROMPT_DATA}" ]; then
    echo "  - MAIN_PROMPT_DATA: $MAIN_PROMPT_DATA"
else
    echo "  - MAIN_PROMPT_DATA: (using default)"
fi
if [ -n "${MAIN_RAY_TEMP_DIR}" ]; then
    echo "  - MAIN_RAY_TEMP_DIR: $MAIN_RAY_TEMP_DIR"
else
    echo "  - MAIN_RAY_TEMP_DIR: (using default)"
fi
if [ -n "${RETRIEVAL_SERVICE_URL}" ]; then
    echo "  - RETRIEVAL_SERVICE_URL: $RETRIEVAL_SERVICE_URL"
else
    echo "  - RETRIEVAL_SERVICE_URL: (using default localhost:50001)"
fi
