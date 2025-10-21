
export SUB_AGENT_HOST=${SGLANG_HOST:-$(hostname -i)}
export SUB_AGENT_PORT=${SGLANG_PORT:-3333}

python -m sglang_router.launch_router \
    --host $SUB_AGENT_HOST \
    --port $SUB_AGENT_PORT
