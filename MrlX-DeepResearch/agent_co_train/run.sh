#!/usr/bin/env bash
set -euo pipefail

# pip install distro
# pip install -e .
# bash examples/agent_co_train/start_router.sh

if [[ -z "${1:-}" ]]; then
  echo "Usage: $0 {sub|main}" >&2
  exit 1
fi

run_type=$1

cd "$SLIME_DIR"

if [[ -z "${SUB_AGENT_IP:-}" ]]; then
  echo "SUB_AGENT_IP environment variable must be set." >&2
  exit 1
fi

if [[ -z "${SUMMARY_LLM_API_KEY:-}" ]]; then
  echo "SUMMARY_LLM_API_KEY environment variable must be set." >&2
  exit 1
fi

export KEY_SUFFIX=${KEY_SUFFIX:-"test"}

mkdir -p "${LOG_DIR:-$SLIME_DIR/logs}"/"$(date +%m%d)"

case "$run_type" in
  sub)
    bash examples/agent_co_train/run_qwen3_30B_sub.sh 2>&1 | tee "${LOG_DIR:-$SLIME_DIR/logs}"/"$(date +%m%d)"/"${KEY_SUFFIX}_sub_$(date +%H%M).log"
    ;;
  main)
    bash examples/agent_co_train/run_qwen3_30B_main.sh 2>&1 | tee "${LOG_DIR:-$SLIME_DIR/logs}"/"$(date +%m%d)"/"${KEY_SUFFIX}_main_$(date +%H%M).log"
    ;;
  *)
    echo "Unknown type: $run_type" >&2
    echo "Usage: $0 {sub|main}" >&2
    exit 1
    ;;
esac
