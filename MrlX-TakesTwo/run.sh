#!/usr/bin/env bash
set -euo pipefail

# pip install distro
# pip install -e .
# bash examples/med_co_train/start_router.sh

if [[ -z "${1:-}" ]]; then
  echo "Usage: $0 {patient|doctor}" >&2
  exit 1
fi

run_type=$1

export SLIME_DIR=${SLIME_DIR:-"/code/slime"}
cd "$SLIME_DIR"

if [[ -z "${PATIENT_IP:-}" ]]; then
  echo "PATIENT_IP environment variable must be set." >&2
  exit 1
fi

if [[ -z "${DEEPSEEK_R1_API_KEY:-}" ]]; then
  echo "DEEPSEEK_R1_API_KEY environment variable must be set." >&2
  exit 1
fi

if [[ -z "${DEEPSEEK_R1_BASE_URL:-}" ]]; then
  echo "DEEPSEEK_R1_BASE_URL environment variable must be set." >&2
  exit 1
fi

if [[ -z "${KEY_SUFFIX:-}" ]]; then
  echo "KEY_SUFFIX environment variable must be set." >&2
  exit 1
fi

# Prepare log directory
LOG_DATE="$(date +%m%d)"
LOG_DIR="/logs/${LOG_DATE}"
mkdir -p "$LOG_DIR"

case "$run_type" in
  patient)

    # Get current host IP (first one)
    HOST_IP=$(hostname -i | awk '{print $1}')
    echo "Detected host IP: $HOST_IP"

    # Start Router service in the background and log output
    cd "$SLIME_DIR/examples/med_co_train"
    bash start_router.sh 2>&1 | tee "$LOG_DIR/${KEY_SUFFIX}_router.log" &

    # Wait until Router is ready (port 3333)
    until nc -z "$HOST_IP" 3333; do
      echo "Waiting for Router service to start..."
      sleep 1
    done

    # Start Database service in the background and log output
    bash start_database_server.sh 2>&1 | tee "$LOG_DIR/${KEY_SUFFIX}_database_server.log" &

    # Wait until Database is ready (port 5432)
    until nc -z "$HOST_IP" 18888; do
      echo "Waiting for Database service to start..."
      sleep 1
    done

    # Return to main project directory
    cd "$SLIME_DIR"

    # Run patient mode training and log both stdout and stderr
    bash examples/MrlX-TakesTwo/run_qwen3_8B_patient.sh 2>&1 | tee "$LOG_DIR/${KEY_SUFFIX}_patient.log"
    ;;

  doctor)
    # Run doctor mode training and log both stdout and stderr
    bash examples/MrlX-TakesTwo/run_qwen3_32B_doc.sh 2>&1 | tee "$LOG_DIR/${KEY_SUFFIX}_doctor.log"
    ;;

  *)
    # Invalid run_type provided
    echo "Unknown type: $run_type" >&2
    echo "Usage: $0 {patient|doctor}" >&2
    exit 1
    ;;
esac