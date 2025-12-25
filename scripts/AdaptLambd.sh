#!/bin/bash

# for rerun the task
pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python

set -ex

# will prevent ray from buffering stdout/stderr
export PYTHONBUFFERED=16

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"
EXP_NAME="${EXP_NAME:-exp_$(date +%Y%m%d_%H%M%S)}"
# EXP_NAME="verify_lambd_adaptive_0.05"
EXP_NAME="EasyWarmup-AdaptLambd"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/models/qwen3-4B.sh"
SAVE_DIR="/mnt/shared-storage-user/p1-shared/liyizhuo/share/save/Qwen3-4B-Base-${EXP_NAME}/"
CRITIC_SAVE_DIR="${SAVE_DIR}/critic"

CKPT_ARGS=(
   --hf-checkpoint /mnt/shared-storage-user/p1-shared/Qwen/Qwen3-4B-Base
   # --hf-checkpoint /mnt/shared-storage-user/p1-shared/liyizhuo/share/models/Qwen3-4B
   #--hf-checkpoint /root/Qwen3-4B-FP8
   # --ref-load /mnt/shared-storage-user/p1-shared/Qwen/Qwen3-4B-Base
   --ref-load /mnt/shared-storage-user/p1-shared/liyizhuo/share/models/Qwen3-4B-Base-torch_dist
   --load /mnt/shared-storage-user/p1-shared/Qwen/Qwen3-4B-Base
   --save ${SAVE_DIR}

   --critic-hf-checkpoint /mnt/shared-storage-user/p1-shared/Qwen/Qwen3-4B-Base
   --critic-ref-load /mnt/shared-storage-user/p1-shared/liyizhuo/share/save/Qwen3-4B-Base-slime_PPO_CriticPretrain/critic
   --critic-load /mnt/shared-storage-user/p1-shared/liyizhuo/share/save/Qwen3-4B-Base-slime_PPO_CriticPretrain/critic
   --critic-save ${CRITIC_SAVE_DIR}
   
   --save-interval 100
)

ROLLOUT_ARGS=(
   # --prompt-data /mnt/shared-storage-user/p1-shared/liyizhuo/share/data/debug2.jsonl
   --prompt-data /mnt/shared-storage-user/p1-shared/liyizhuo/share/data/DAPO-Math-17k/dapo-math-17k.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type dapo
   --reward-key score
   
   --num-rollout 1000
   --rollout-batch-size 32
   --n-samples-per-prompt 4
   --rollout-max-response-len 16384
   --rollout-temperature 0.8

   --global-batch-size 128
   --balance-data
)

EVAL_ARGS=(
   --eval-interval 10
   # --eval-prompt-data aime24 /mnt/shared-storage-user/p1-shared/liyizhuo/share/data/aime_2024/aime-2024-deepmathformat.jsonl aime25 /mnt/shared-storage-user/p1-shared/liyizhuo/share/data/aime-2025/aime-2025-deepmathformat.jsonl
   --eval-prompt-data aime24 /mnt/shared-storage-user/p1-shared/liyizhuo/share/data/aime_2024/aime-2024-deepmathformat.jsonl
   --n-samples-per-eval-prompt 16
   --eval-max-response-len 16384
   --eval-top-p 0.95
   --eval-temperature 0.6
   --log-passrate
)

PERF_ARGS=(
   --tensor-model-parallel-size 1
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   # --micro-batch-size 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 17408
)

# GRPO_ARGS=(
#    --advantage-estimator ppo
#    --use-kl-loss
#    --kl-loss-coef 0.00
#    --kl-loss-type low_var_kl
#    --entropy-coef 0.00
#    --eps-clip 0.2
#    --eps-clip-high 0.28
# )

PPO_ARGS=(
   --advantage-estimator ppo
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --kl-coef 0.00
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
   --num-critic-only-steps 10
   # --num-critic-only-steps 1
   # --normalize-advantages
   --critic-lr 2e-6
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

export WANDB_DIR="/mnt/shared-storage-user/p1-shared/liyizhuo/code/slime/wandb"
export WANDB_API_KEY="5cb606567741f22337bbbd70cf464c2951631e9a"
export WANDB_MODE="offline"

WANDB_ARGS=(
   --use-wandb
   --wandb-project slime-one
   --wandb-group Qwen3-4B-PPO
   --wandb-key ${WANDB_API_KEY}
   --wandb-dir ${WANDB_DIR}
   --wandb-mode ${WANDB_MODE}
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 2
   --sglang-mem-fraction-static 0.7
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


DEBUG_DIR="/mnt/shared-storage-user/p1-shared/liyizhuo/code/slime/debug_data/${EXP_NAME}"
ROLLOUT_DEBUG_DIR="${DEBUG_DIR}/rollout"
TRAIN_DEBUG_DIR="${DEBUG_DIR}/train"
ANALYSIS_DIR="${DEBUG_DIR}/analysis"
mkdir -p "${ROLLOUT_DEBUG_DIR}" "${TRAIN_DEBUG_DIR}" "${ANALYSIS_DIR}"
echo "Debug data will be saved to: ${DEBUG_DIR}"

DEBUG_ARGS=(
   --save-debug-rollout-data "${ROLLOUT_DEBUG_DIR}/rollout_data_{}.pkl"
   --save-debug-train-data "${TRAIN_DEBUG_DIR}/train_data_{}.pkl"
   --dump-details "${ANALYSIS_DIR}/training_details.json"
)

CUSTOM_ARGS=(
   --use-asyppo
   --use-hf-config-for-megatron
   --log-position-value-stats
   --max-log-positions 500
   # --lambd 1
   --lambd-actor 1
   # --lambd-critic 1
   --lambd-critic 1
   --use-adaptive-lambda
   --alpha 0.05
   --log-position-advantage-stats
)

# launch the master node of ray in container
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=26500

# Build the runtime environment JSON with proper variable substitution
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\"
  }
}"

LOG_DIR="/mnt/shared-storage-user/p1-shared/liyizhuo/code/slime/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/${EXP_NAME}.log"
echo "Log file will be saved to: ${LOG_FILE}"

export RANK=${NODE_RANK:-0}
export NODE_COUNT=${KUBEBRAIN_REPLICA_TOTAL:-1}
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export PROC_PER_NODE=${PROC_PER_NODE:-8}

if [ -z "$RANK" ]; then
  echo "RANK not set. Defaulting to RANK=0 for single node"
  export RANK=0
fi

# ========= Submit Ray Job (rank 0 only) =========
if [ "$RANK" == "0" ]; then
   ray job submit --address="http://127.0.0.1:26500" \
      --runtime-env-json="${RUNTIME_ENV_JSON}" \
      -- python3 train.py \
      --actor-num-nodes 1 \
      --actor-num-gpus-per-node 4 \
      --colocate \
      ${MODEL_ARGS[@]} \
      ${CKPT_ARGS[@]} \
      ${ROLLOUT_ARGS[@]} \
      ${OPTIMIZER_ARGS[@]} \
      ${PPO_ARGS[@]} \
      ${WANDB_ARGS[@]} \
      ${PERF_ARGS[@]} \
      ${EVAL_ARGS[@]} \
      ${SGLANG_ARGS[@]} \
      ${DEBUG_ARGS[@]} \
      ${CUSTOM_ARGS[@]} \
      ${MISC_ARGS[@]} 2>&1 | tee -a "${LOG_FILE}"
fi