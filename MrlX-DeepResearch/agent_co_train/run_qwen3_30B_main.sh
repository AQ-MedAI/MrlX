#!/bin/bash

# for rerun the task
pkill -9 ray
ray stop --force

set -ex
export PYTHONBUFFERED=16

NVLINK_COUNT=$(nvidia-smi | grep -o "NVLink" | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
   HAS_NVLINK=1
else
   HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

CO_TRAIN_DIR=${CO_TRAIN_DIR:-"$SLIME_DIR/examples/agent_co_train"}
source "$SLIME_DIR/scripts/models/qwen3-30B-A3B.sh"

# Set default paths (model checkpoints require user configuration)
MAIN_CHECKPOINT_DIR=${MAIN_CHECKPOINT_DIR:-"$SLIME_DIR/../checkpoints/main"}
MAIN_PROMPT_DATA=${MAIN_PROMPT_DATA:-"$CO_TRAIN_DIR/mock_data_main.jsonl"}
MAIN_RAY_TEMP_DIR=${MAIN_RAY_TEMP_DIR:-"$SLIME_DIR/../ray_temp/main"}

CKPT_ARGS=(
   --hf-checkpoint ${MAIN_HF_CHECKPOINT}
   --ref-load ${MAIN_REF_LOAD}
   --load ${MAIN_CHECKPOINT_DIR}/single-main-$(date +%m%d)
   --save ${MAIN_CHECKPOINT_DIR}/single-main-$(date +%m%d)
   --save-interval 1
)

ROLLOUT_ARGS=(
   --prompt-data ${MAIN_PROMPT_DATA}
   --input-key question
   --label-key answer
#    --metadata-key extra_info
   # --apply-chat-template
   --rollout-shuffle

   --rm-type deepscaler

   --num-rollout 99999
   --rollout-batch-size 2
   --n-samples-per-prompt 8
   --rollout-max-response-len 1024
   --rollout-temperature 1.0

   --global-batch-size 16
   --balance-data
)

# EVAL_ARGS=(
#    --eval-interval 20
#    --eval-prompt-data aime /ossfs/workspace/aime-2024/aime-2024.jsonl
#    --n-samples-per-eval-prompt 1
#    --eval-max-response-len 16384
#    --eval-top-p 0.7
# )

PERF_ARGS=(
   --tensor-model-parallel-size 4
   --sequence-parallel
   --pipeline-model-parallel-size 2
   --context-parallel-size 1
   --expert-model-parallel-size 4
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --no-check-for-nan-in-loss-and-grad
   # --micro-batch-size 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 10240
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --kl-coef 0.00
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98

   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)

WANDB_ARGS=(
   # --use-wandb
   # --wandb-project slime-dev
   # --wandb-group qwen3-4B-test
   # --wandb-key ${WANDB_KEY}
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 8
   --sglang-mem-fraction-static 0.5
   --sglang-enable-ep-moe
   --sglang-cuda-graph-bs 1 2 4 8 $(seq 16 8 256)
   # --sglang-enable-dp-attention
   # --sglang-dp-size 8
   # --debug-rollout-only
)

# 495222
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
   --custom-generate-function-path main_agent_multiturn.generate
   --custom-rm-path main_agent_multiturn.reward_func
   # --debug-rollout-only
   # --colocate
)

# launch the master node of ray in container
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats --temp-dir ${MAIN_RAY_TEMP_DIR}

# export PYTHONPATH=/root/my_libs:$PYTHONPATH
# Build the runtime environment JSON with proper variable substitution
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/:${CO_TRAIN_DIR}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 8 \
   --rollout-num-gpus 8 \
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
