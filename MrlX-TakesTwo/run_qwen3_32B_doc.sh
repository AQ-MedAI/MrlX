#!/bin/bash

set -ex
export PYTHONBUFFERED=16

NVLINK_COUNT=$(nvidia-smi | grep -o "NVLink" | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
   HAS_NVLINK=1
else
   HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

EXPAMPLE_DIR="$SLIME_DIR/examples/MrlX-TakesTwo"
source "$SLIME_DIR/scripts/models/qwen3-32B.sh"

export TRAIN_MODE=${TRAIN_MODE:-"local"}

# Set environment variables based on TRAIN_MODE
if [ "$TRAIN_MODE" = "local" ]; then
   export RAY_ADDRESS="127.0.0.1:8286"
else
   if [ -z "$RAY_ADDRESS" ]; then echo "Error: RAY_ADDRESS environment variable is not set" && exit 1; fi
fi

HF_CKPT_PATH=${HF_CKPT_PATH:-""}
DIST_CKPT_PATH=${DIST_CKPT_PATH:-""}
SAVE_PATH=${SAVE_PATH:-""}
PROMPT_DATA_PATH=${PROMPT_DATA_PATH:-""}

CKPT_ARGS=(
   --hf-checkpoint "$HF_CKPT_PATH"
   --ref-load "$DIST_CKPT_PATH"
   --load "$SAVE_PATH"
   --save "$SAVE_PATH"
   --save-interval 11
)

ROLLOUT_ARGS=(
   --prompt-data "$PROMPT_DATA_PATH"
   --input-key messages
   # --label-key label
   --metadata-key extra_info
   # --apply-chat-template
   --rollout-shuffle

   --rm-type deepscaler

   --num-epoch 1
   --num-rollout 55
   --rollout-batch-size 128
   --n-samples-per-prompt 8
   --rollout-max-response-len 8192
   --rollout-temperature 1.0

   --global-batch-size 1024
   --num-steps-per-rollout 1
   --balance-data
)

PERF_ARGS=(
   --tensor-model-parallel-size 8
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --no-check-for-nan-in-loss-and-grad
   # --micro-batch-size 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 20480
)

GRPO_ARGS=(
   --advantage-estimator grpo
   # --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --kl-coef 0.00
   --entropy-coef 0.001
   --eps-clip 0.2
   --eps-clip-high 0.28
   --calculate-per-token-loss
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.01
   --adam-beta1 0.9
   --adam-beta2 0.95
   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)

WANDB_ARGS=(
   # --use-wandb
   # --wandb-project slime-dev
   # --wandb-group Qwen3-32B-doctor
   # --wandb-key ${WANDB_KEY}
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 8
   --sglang-mem-fraction-static 0.5
   --sglang-cuda-graph-bs 1 2 4 8 $(seq 16 8 256)
   # --sglang-enable-ep-moe
   # --debug-rollout-only
)

MISC_ARGS=(
   # default dropout in megatron is 0.1
   --attention-dropout 0.0
   --hidden-dropout 0.0
   # should be good for model performance
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   # need to comment this when using model with MLA
   --attention-backend flash
)

CUSTOM_ARGS=(
   --custom-generate-function-path doctor_multiturn.generate
   --custom-rm-path doctor_multiturn.reward_func
   # --debug-rollout-only
)

# launch the master node of ray in container
if [ "$TRAIN_MODE" = "local" ]; then
   export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
   ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats
fi

RUNTIME_ENV_JSON="{
  \"working_dir\": \"${SLIME_DIR}\",
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/:${EXPAMPLE_DIR}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"MAX_TURNS\": \"${MAX_TURNS:-10}\",
    \"KEY_SUFFIX\": \"${KEY_SUFFIX}\",
    \"PATIENT_IP\": \"${PATIENT_IP}\",
    \"DEEPSEEK_R1_BASE_URL\": \"${DEEPSEEK_R1_BASE_URL}\",
    \"DEEPSEEK_R1_API_KEY\": \"${DEEPSEEK_R1_API_KEY}\"
  }
}"

ray job submit --address="http://${RAY_ADDRESS}" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 8 \
   --colocate \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${DISTRIBUTED_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${CUSTOM_ARGS[@]}
