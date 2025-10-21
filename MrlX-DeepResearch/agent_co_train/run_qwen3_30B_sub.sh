#!/bin/bash

# for rerun the task
pkill -9 ray
ray stop --force

set -ex

export SUB_HOST=${SGLANG_HOST:-$(hostname -i)}
export SUB_PORT=${SGLANG_PORT:-3333}

# will prevent ray from buffering stdout/stderr
export PYTHONBUFFERED=16
export RAR_DEBUG=legacy

CO_TRAIN_DIR=${CO_TRAIN_DIR:-"$SLIME_DIR/examples/agent_co_train"}
source "$SLIME_DIR/scripts/models/qwen3-30B-A3B.sh"

# Set default paths (model checkpoints require user configuration)
SUB_CHECKPOINT_DIR=${SUB_CHECKPOINT_DIR:-"$SLIME_DIR/../checkpoints/sub"}
SUB_PROMPT_DATA=${SUB_PROMPT_DATA:-"$CO_TRAIN_DIR/mock_data_sub.jsonl"}

CKPT_ARGS=(
   --hf-checkpoint ${SUB_HF_CHECKPOINT}
   --ref-load ${SUB_REF_LOAD}
   --load ${SUB_CHECKPOINT_DIR}/single-sub-$(date +%m%d)
   --save ${SUB_CHECKPOINT_DIR}/single-sub-$(date +%m%d)
   --save-interval 1
)

ROLLOUT_ARGS=(
   # --disable-rollout-global-dataset
   --prompt-data ${SUB_PROMPT_DATA}
   --input-key question
   --label-key answer
#    --metadata-key extra_info
#    --apply-chat-template
   --rollout-shuffle
   --num-rollout 99999
   --rollout-batch-size 2
   --n-samples-per-prompt 8
   --rollout-max-response-len 1024
   --rollout-temperature 1.0

   --global-batch-size 16
   --balance-data
)

PERF_ARGS=(
   --tensor-model-parallel-size 4
   --sequence-parallel
   --pipeline-model-parallel-size 1
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
   # --wandb-group search-r1_qwen2.5-3B-test
   # --wandb-key ${WANDB_KEY}
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 4
   --sglang-mem-fraction-static 0.4
   --sglang-enable-ep-moe
   --sglang-cuda-graph-bs 1 2 4 8 $(seq 16 8 256)
   # --sglang-enable-dp-attention
   # --sglang-dp-size 8
   --sglang-router-ip $SUB_HOST
   --sglang-router-port $SUB_PORT
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
   --custom-generate-function-path sub_agent_multiturn.generate
   --custom-rm-path sub_agent_multiturn.reward_func
   # --debug-rollout-only
   # --colocate
)

# launch the master node of ray in container
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats

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
   --actor-num-gpus-per-node 4 \
   --rollout-num-gpus 4 \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${DISTRIBUTED_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${CUSTOM_ARGS[@]}
