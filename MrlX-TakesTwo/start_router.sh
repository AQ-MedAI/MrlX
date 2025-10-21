#!/bin/bash
set -ex
export PATIENT_HOST=${SGLANG_HOST:-$(hostname -i)}
export PATIENT_PORT=${SGLANG_PORT:-3333}

python -m sglang_router.launch_router \
    --host $PATIENT_HOST \
    --port $PATIENT_PORT
