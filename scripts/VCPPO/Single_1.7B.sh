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
# EXP_NAME="1_7BCritic+WARMUP+32*4"  # Single critic
EXP_NAME="4bsftv2_onecritic1.7b-16*4-32k-utd1-bceloss"  # Double critic configuration
# EXP_NAME="?????"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# source "${SCRIPT_DIR}/models/qwen3-4B.sh"
SAVE_DIR="/mnt/shared-storage-user/p1-shared/liyizhuo/share/save/P1_PPO/Qwen3-1_7B-Base-${EXP_NAME}/"
CRITIC_SAVE_DIR="${SAVE_DIR}/critic"
CRITIC2_SAVE_DIR="${SAVE_DIR}/critic2"
TP_SIZE=2
PP_SIZE=1
CP_SIZE=1
EP_SIZE=1
ETP_SIZE=1
MAX_LEN=$((1024 * 32))
MAX_TOKENS_PER_GPU=$((($MAX_LEN / $CP_SIZE) + 1024))
ROLLOUT_BATCH_SIZE=16
N_SAMPLES_PER_PROMPT=4
NUM_STEPS_PER_ROLLOUT=1

# Check if resume checkpoint exists
if [ -n "$RESUME_CHECKPOINT_DIR" ] && [ -d "$RESUME_CHECKPOINT_DIR" ]; then
    # Check for latest_checkpointed_iteration.txt or similar checkpoint indicator
    if [ -f "${RESUME_CHECKPOINT_DIR}/latest_checkpointed_iteration.txt" ]; then
        LATEST_ITER=$(cat "${RESUME_CHECKPOINT_DIR}/latest_checkpointed_iteration.txt" | tr -d '[:space:]')
        echo "✅ Found resume checkpoint at: ${RESUME_CHECKPOINT_DIR}"
        echo "   Latest iteration: ${LATEST_ITER}"
        echo "   Will resume from rollout ${RESUME_ROLLOUT_ID}"
        USE_RESUME_CHECKPOINT=true
    elif [ -n "$(find "${RESUME_CHECKPOINT_DIR}" -name "iter_*" -type d | head -1)" ]; then
        # Alternative: check if any iteration directory exists
        echo "✅ Found resume checkpoint at: ${RESUME_CHECKPOINT_DIR}"
        echo "   Will resume from rollout ${RESUME_ROLLOUT_ID}"
        USE_RESUME_CHECKPOINT=true
    else
        echo "⚠️  Resume checkpoint directory exists but no checkpoint found, starting from initial checkpoint"
        USE_RESUME_CHECKPOINT=false
    fi
else
    echo "⚠️  Resume checkpoint not found or not set, starting from initial checkpoint"
    USE_RESUME_CHECKPOINT=false
fi

# Check critic checkpoint if resume is enabled
if [ "$USE_RESUME_CHECKPOINT" = true ] && [ -n "$RESUME_CRITIC_CHECKPOINT_DIR" ] && [ -d "$RESUME_CRITIC_CHECKPOINT_DIR" ]; then
    echo "✅ Found resume critic checkpoint at: ${RESUME_CRITIC_CHECKPOINT_DIR}"
    USE_RESUME_CRITIC_CHECKPOINT=true
else
    USE_RESUME_CRITIC_CHECKPOINT=false
fi

# Check critic2 checkpoint if resume is enabled
if [ "$USE_RESUME_CHECKPOINT" = true ] && [ -n "$RESUME_CRITIC2_CHECKPOINT_DIR" ] && [ -d "$RESUME_CRITIC2_CHECKPOINT_DIR" ]; then
    echo "✅ Found resume critic2 checkpoint at: ${RESUME_CRITIC2_CHECKPOINT_DIR}"
    USE_RESUME_CRITIC2_CHECKPOINT=true
else
    USE_RESUME_CRITIC2_CHECKPOINT=false
fi
# ========================================================================

# Set checkpoint arguments based on resume status
if [ "$USE_RESUME_CHECKPOINT" = true ]; then
    CKPT_ARGS=(
       --hf-checkpoint /mnt/shared-storage-user/p1-shared/Qwen/Qwen3-4B-Base
       --ref-load /mnt/shared-storage-user/p1-shared/liyizhuo/share/models/Qwen3-4B-Base-torch_dist
       --load ${RESUME_CHECKPOINT_DIR}
       --start-rollout-id ${RESUME_ROLLOUT_ID}
       --save ${RESUME_CHECKPOINT_DIR}
       
       --critic-hf-checkpoint /mnt/shared-storage-user/p1-shared/Qwen/Qwen3-1.7B-Base
       --critic-ref-load /mnt/shared-storage-user/p1-shared/liyizhuo/share/save/Qwen3-1_7B-Base-17B_CriticPretrain/critic
       $(if [ "$USE_RESUME_CRITIC_CHECKPOINT" = true ]; then echo "--critic-load ${RESUME_CRITIC_CHECKPOINT_DIR}"; fi)
       --critic-save ${CRITIC_SAVE_DIR}
       
       # Critic2 configuration (second critic model)
       --use-critic2
       --critic2-hf-checkpoint /mnt/shared-storage-user/p1-shared/Qwen/Qwen3-1.7B-Base
       --critic2-ref-load /mnt/shared-storage-user/p1-shared/liyizhuo/share/save/Qwen3-1_7B-Base-17B_CriticPretrain/critic2
       $(if [ "$USE_RESUME_CRITIC2_CHECKPOINT" = true ]; then echo "--critic2-load ${RESUME_CRITIC2_CHECKPOINT_DIR}"; fi)
       --critic2-save ${CRITIC2_SAVE_DIR}
       
       --save-interval 500
    )
else
    CKPT_ARGS=(
       --hf-checkpoint /mnt/shared-storage-user/p1-shared/lichenxi1/Qwen3-4B-Thinking-2507-SFT-v2
       # --hf-checkpoint /mnt/shared-storage-user/p1-shared/liyizhuo/share/models/Qwen3-4B
       #--hf-checkpoint /root/Qwen3-4B-FP8
       # --ref-load /mnt/shared-storage-user/p1-shared/Qwen/Qwen3-4B-Base
       --ref-load /mnt/shared-storage-user/p1-shared/Qwen/Qwen3-4B-Thinking-2507-SFT-v2_torch_dist
       --load /mnt/shared-storage-user/p1-shared/Qwen/Qwen3-4B-Thinking-2507-SFT-v2_torch_dist
       --save ${SAVE_DIR}

       --critic-hf-checkpoint /mnt/shared-storage-user/p1-shared/Qwen/Qwen3-1.7B-Base
       --critic-ref-load /mnt/shared-storage-user/p1-shared/liyizhuo/share/save/Qwen3-1_7B-Base-17B_CriticPretrain/critic
       --critic-load /mnt/shared-storage-user/p1-shared/liyizhuo/share/save/Qwen3-1_7B-Base-17B_CriticPretrain/critic
       --critic-save ${CRITIC_SAVE_DIR}
       
       # Critic2 configuration (second critic model)
      #  --use-critic2
       --critic2-hf-checkpoint /mnt/shared-storage-user/p1-shared/Qwen/Qwen3-1.7B-Base
       --critic2-ref-load /mnt/shared-storage-user/p1-shared/liyizhuo/share/save/Qwen3-1_7B-Base-17B_CriticPretrain/critic
       --critic2-load /mnt/shared-storage-user/p1-shared/liyizhuo/share/save/Qwen3-1_7B-Base-17B_CriticPretrain/critic
       --critic2-save ${CRITIC2_SAVE_DIR}
       
       --save-interval 1000
    )
fi

ROLLOUT_ARGS=(
#    --prompt-data /mnt/shared-storage-user/p1-shared/liyizhuo/share/data/debug2.jsonl
   # --prompt-data /mnt/shared-storage-user/p1-shared/liyizhuo/share/data/DAPO-Math-17k/dapo-math-17k.jsonl
   # --prompt-data proofdata /mnt/shared-storage-user/p1-shared/chenjiacheng/data/imo/train_math_data_with_marking_7pt_valid.jsonl math1205 /mnt/shared-storage-user/p1-shared/chenjiacheng/data/imo/train_math_data_verifiable_1205_processed_v3.jsonl
   --prompt-data /mnt/shared-storage-user/p1-shared/liyizhuo/share/data/P1/merge.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type remote_rm
   --rm-url "http://10.102.203.45:8021/"
   --reward-key score
   
   --num-rollout 6000
   --rollout-batch-size $ROLLOUT_BATCH_SIZE
   --n-samples-per-prompt $N_SAMPLES_PER_PROMPT
   --rollout-max-response-len $MAX_LEN
   --rollout-temperature 1.0

   # --global-batch-size 128
   --num-steps-per-rollout $NUM_STEPS_PER_ROLLOUT
   --use-tis
   --partial-rollout
   --over-sampling-batch-size $((ROLLOUT_BATCH_SIZE * 2))
   --balance-data
   --dynamic-sampling-filter-path slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std
)
N_SAMPLES_PER_EVAL=(4 1 4)
IMO_PATH=/mnt/shared-storage-user/p1-shared/zhaoyufeng/eval_results/test_jsonl
EVAL_ARGS=(
   --eval-interval 4
   --eval-prompt-data amo $IMO_PATH/amobench.jsonl answerbench $IMO_PATH/answerbench.jsonl proofbench /mnt/shared-storage-user/p1-shared/chenjiacheng/data/imo/proofbench.jsonl
   --n-samples-per-eval-prompt ${N_SAMPLES_PER_EVAL[@]}
   --eval-max-response-len 81920
   --eval-top-p 0.95
   --eval-temperature 1.0
   --log-passrate
)

PERF_ARGS=(
   --tensor-model-parallel-size $TP_SIZE
   --sequence-parallel
   --pipeline-model-parallel-size $PP_SIZE
   --context-parallel-size $CP_SIZE
   # --expert-model-parallel-size $EP_SIZE
   # --expert-tensor-parallel-size $ETP_SIZE

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   # --micro-batch-size 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu $MAX_TOKENS_PER_GPU
   --transformer-impl transformer_engine
   --bf16
   --fp8-format e4m3
   --fp8-recipe blockwise
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
#    --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
   --num-critic-only-steps 10
   # --num-critic-only-steps 1
   # --normalize-advantages
   --critic-lr 4e-6
   --critic-num-nodes 1
   --critic-num-gpus-per-node 2


   # Critic2 configuration (same settings as critic1)
   --critic2-num-nodes 1
   --critic2-num-gpus-per-node 2
   --critic2-lr 4e-6
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
   --sglang-cuda-graph-bs 1 2 4 8 $(seq 16 8 256)
   --sglang-speculative-algorithm EAGLE3
   --sglang-speculative-num-steps 3
   --sglang-speculative-eagle-topk 1
   --sglang-speculative-num-draft-tokens 4
   --sglang-speculative-draft-model-path /mnt/shared-storage-user/p1-shared/leihaodi/spec_decode/draft-model/Qwen3-4B_eagle3-AngelSlim
)
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
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
   # --use-adaptive-lambda
   # --alpha 0.05
   --log-position-advantage-stats
   # --use-positive-nll-loss
#    --positive-nll-coef 0.1
#    --positive-reward-threshold 0.0
   --eval-first
   --finetune
   --no-load-optim
   # --use-asytrain-critic
   # --use-advantage-diff-mask
   # --advantage-diff-mask-k 0.1
   # --use-entropy-value-divergence-filter
   # --entropy-divergence-filter-h 0.2
   # --entropy-coef 0.001
#    --critic-num-gpus-per-node 2
#    --critic-num-nodes 1
#    --critic2-num-gpus-per-node 2
#    --critic2-num-nodes 1
   --eval-use-xverify
   # --eval-group
   --train-use-xverify
   # --eval-log-dir ${DEBUG_DIR}/eval
   # --start-rollout-id 0
   # --use-bce-value-loss
)

# launch the master node of ray in container
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
# 使用 8 GPUs: Actor(2) + Critic1(colocate with actor) + Critic2(2) + Rollout(2-4)
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 4 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=26500

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


# ${MODEL_ARGS[@]} \
# ========= Submit Ray Job (rank 0 only) =========
if [ "$RANK" == "0" ]; then
   ray job submit --address="http://127.0.0.1:26500" \
      --runtime-env-json="${RUNTIME_ENV_JSON}" \
      -- python3 train.py \
      --actor-num-nodes 1 \
      --actor-num-gpus-per-node 2 \
      --colocate \
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
