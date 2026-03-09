#!/bin/bash
export PARTITION=${GROUP}
# 修改成你的路径！！！！！！！！！！！
# export CFSCTL=/mnt/shared-storage-user/p1-shared/chenjiacheng/cfs/bin/cfsctl
# export CFG=/mnt/shared-storage-user/p1-shared/chenjiacheng/cfs/cfsd.cfg

# # Replace sources.list with new configuration
# tee /etc/apt/sources.list > /dev/null << 'EOF'
# # ubuntu 22.04
# deb     http://mirrors.i.h.pjlab.org.cn/repository/apt-jammy-proxy/ubuntu/ jammy main restricted universe multiverse
# deb     http://mirrors.i.h.pjlab.org.cn/repository/apt-jammy-proxy/ubuntu/ jammy-security main restricted universe multiverse
# deb     http://mirrors.i.h.pjlab.org.cn/repository/apt-jammy-proxy/ubuntu/ jammy-updates main restricted universe multiverse
# deb     http://mirrors.i.h.pjlab.org.cn/repository/apt-jammy-proxy/ubuntu/ jammy-proposed main restricted universe multiverse
# deb     http://mirrors.i.h.pjlab.org.cn/repository/apt-jammy-proxy/ubuntu/ jammy-backports main restricted universe multiverse
# deb-src http://mirrors.i.h.pjlab.org.cn/repository/apt-jammy-proxy/ubuntu/ jammy main restricted universe multiverse
# deb-src http://mirrors.i.h.pjlab.org.cn/repository/apt-jammy-proxy/ubuntu/ jammy-security main restricted universe multiverse
# deb-src http://mirrors.i.h.pjlab.org.cn/repository/apt-jammy-proxy/ubuntu/ jammy-updates main restricted universe multiverse
# deb-src http://mirrors.i.h.pjlab.org.cn/repository/apt-jammy-proxy/ubuntu/ jammy-proposed main restricted universe multiverse
# deb-src http://mirrors.i.h.pjlab.org.cn/repository/apt-jammy-proxy/ubuntu/ jammy-backports main restricted universe multiverse
# EOF

# # Update package lists
# apt-get update

# DEBIAN_FRONTEND=noninteractive apt-get -y install \
# gcc g++ automake cmake libtool pkgconf \
# libpmemobj-dev libmemkind-dev libtbb-dev rapidjson-dev \
# libjson-c-dev libboost-dev gettext libfuse2 libfuse-dev \
# git sudo vim curl libcurl4-openssl-dev wget pandoc \
# gfortran bzip2 flex libpmix-dev libnl-3-dev libibverbs-dev libssl-dev \
# gdb numactl python3 python3-venv python3-pip binutils-dev

# $CFSCTL -p $PARTITION -n $NODE_COUNT -X $MASTER_ADDR -s $CFG start;   
# [ $? -ne 0 ] && exit 1

cd /mnt/shared-storage-user/p1-shared/chenjiacheng/p1-slime
# pip install -U brainpp
pip install -e . --no-deps --no-index --disable-pip-version-check --no-build-isolation
# pip install tensorboard
# 10.102.223.30 ：只有rule
# 10.102.208.20 ：部署了sft后的xverify
# 10.102.205.32 ：部署了30b-instruct作为reward model
# pip install tensorboard colorama

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
export PIP_INDEX_URL="http://mirrors.h.pjlab.org.cn/pypi/simple/"
export PIP_EXTRA_INDEX_URL="http://pypi.i.h.pjlab.org.cn/brain/dev/+simple"
export PIP_TRUSTED_HOST="mirrors.h.pjlab.org.cn pypi.i.h.pjlab.org.cn"
export PIP_NO_INDEX="false" # 如果要完全禁用公网访问，改为 "true"

pip install math_verify

export WANDB_MODE="offline"
export WANDB_KEY="fd2fa7acc90e8a95676b607f6bfc7d780b16ef65"
export WANDB_DIR="/mnt/shared-storage-user/p1-shared/chenjiacheng/wandb"

# EXP_NAME="thinking-nofilter-128-16-gspo-80k-remote-rollis-3e-3-pass88zero-resume-152-$(date "+%m%d-%H%M%S")"
EXP_NAME="0113-4bsft-v2-marking-verifiable-$(date "+%m%d-%H%M%S")"

# Multi-node environment (defaults for single-node if not provided)
export RANK=${NODE_RANK:-0}
export NODE_COUNT=${KUBEBRAIN_REPLICA_TOTAL:-1}
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export PROC_PER_NODE=${PROC_PER_NODE:-8}

# ========== 角色识别与 MASTER_ADDR 设置 ==========
if [ -z "$RANK" ]; then
  echo "RANK not set. Please set RANK=0 for master, RANK=1,2,... for workers"
  exit 1
fi

SHARED_DIR="/mnt/shared-storage-user/p1-shared/chenjiacheng"
READY_FLAG_FILE="$SHARED_DIR/ray_head_ready_30B"

# will prevent ray from buffering stdout/stderr
export PYTHONBUFFERED=16
# export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,expandable_segments:False"

NVLINK_COUNT=$(nvidia-smi | grep -o "NVLink" | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

HAS_NVLINK=0

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/models/qwen3-4B-Instruct-2507.sh"

TP_SIZE=8
PP_SIZE=1
CP_SIZE=1
EP_SIZE=1
ETP_SIZE=1
MAX_LEN=$((1024 * 64))
MAX_TOKENS_PER_GPU=$((($MAX_LEN / $CP_SIZE) + 1024))
ROLLOUT_BATCH_SIZE=64
N_SAMPLES_PER_PROMPT=8
HIPPO_PATH=/mnt/shared-storage-user/p1-shared/physics_data/HiPhO_test/test_slime
VISUAL_PATH=/mnt/shared-storage-user/p1-shared/physics_data/HiPhO_test/test_slime_visual
MATH_TEST_PATH=/mnt/shared-storage-user/p1-shared/zhaoyufeng/test_jsonl
IMO_PATH=/mnt/shared-storage-user/p1-shared/zhaoyufeng/eval_results/test_jsonl
# RESUME_PATH=thinking-nofilter-128-32-gspo-32k-remote-partial-0825-145437
# RESUME_PATH=cuiganqu/Qwen/Qwen3-30B-A3B-Thinking-2507_slime/thinking-nofilter-128-16-gspo-48k-remote-partial-pass88-0910-052525
# RESUME_PATH=cuiganqu/Qwen/Qwen3-30B-A3B-Thinking-2507_slime/thinking-nofilter-128-16-gspo-48k-remote-partial-pass88-0908-082128
# RESUME_PATH=cuiganqu/Qwen/Qwen3-30B-A3B-Thinking-2507_slime/thinking-nofilter-128-16-gspo-64k-remote-tis-partial-pass88zero-resume-120-0924-162826
# RESUME_PATH=cuiganqu/Qwen/Qwen3-30B-A3B-Thinking-2507_slime/ablate-thinking-128-16-gspo-32k-remote-rollis-1e-3-pass88zero-1030-040121
# RESUME_PATH=cuiganqu/Qwen/Qwen3-30B-A3B-Thinking-2507_slime/fp8-thinking-128-16-gspo-32k-remote-rollis-1e-3-pass88zero-oss-1123-021333
CKPT_ARGS=(
   --hf-checkpoint /mnt/shared-storage-user/p1-shared/lichenxi1/Qwen3-4B-Thinking-2507-SFT-v2
   #--hf-checkpoint /root/Qwen3-4B-FP8
   --ref-load /mnt/shared-storage-user/p1-shared/Qwen/Qwen3-4B-Thinking-2507-SFT-v2_torch_dist
   # --load /mnt/shared-storage-user/${RESUME_PATH}
   --load /mnt/shared-storage-user/cuiganqu/Qwen/Qwen3-4B-Thinking-2507_slime/${EXP_NAME}
   --save /mnt/shared-storage-user/cuiganqu/Qwen/Qwen3-4B-Thinking-2507_slime/${EXP_NAME}
   --save-interval 1280
)

ROLLOUT_ARGS=(
   # --prompt-data /mnt/shared-storage-user/p1-shared/chenjiacheng/data/dapo-math-17k.jsonl
   # --prompt-data /mnt/shared-storage-user/p1-shared/chenjiacheng/data/train_dataset_0722_filtered_pass88_zero_pass@88.jsonl
   # --prompt-data /mnt/shared-storage-user/p1-shared/chenjiacheng/data/train_dataset_0722_filtered_0829.jsonl
   # --prompt-data /mnt/shared-storage-user/p1-shared/chenjiacheng/data/train_dataset_with_visual_1009_filtered.jsonl
   # --prompt-data /mnt/shared-storage-user/p1-shared/chenjiacheng/data/imo/math_phy_data_1124.jsonl
   # --prompt-data /mnt/shared-storage-user/p1-shared/chenjiacheng/data/imo/math_data_verifiable_1124.jsonl
   --prompt-data proofdata /mnt/shared-storage-user/p1-shared/chenjiacheng/data/imo/train_math_data_with_marking_7pt_valid.jsonl math1205 /mnt/shared-storage-user/p1-shared/chenjiacheng/data/imo/train_math_data_verifiable_1205_processed_v3.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type remote_rm
   --rm-url "http://10.102.203.45:8021/"
   # --rm-url "http://10.102.247.32:8001/"
   --reward-key score
   --num-rollout 6000
   --rollout-batch-size $ROLLOUT_BATCH_SIZE
   --n-samples-per-prompt $N_SAMPLES_PER_PROMPT
   --rollout-max-response-len $MAX_LEN
   --rollout-temperature 1.0
   # --global-batch-size 256
   # --use-token-output
   --num-steps-per-rollout 4
   --use-tis
   # --use-rollout-is
   --partial-rollout
   --over-sampling-batch-size $((ROLLOUT_BATCH_SIZE * 2))
   --dynamic-sampling-filter-path slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std
   --balance-data
   --finetune
)

# Number of samples per eval prompt for each dataset (in order of eval-prompt-data)
# Datasets: amo, answerbench, proofbench
N_SAMPLES_PER_EVAL=(4 1 4)
# N_SAMPLES_PER_EVAL=4

EVAL_ARGS=(
   --eval-interval 4
   # --eval-before-train
   --eval-use-xverify
   --eval-group
   --train-use-xverify # /mnt/shared-storage-user/p1-shared/chenjiacheng/data/ipho2024_0717_slime.jsonl
   --eval-prompt-data amo $IMO_PATH/amobench.jsonl answerbench $IMO_PATH/answerbench.jsonl proofbench /mnt/shared-storage-user/p1-shared/chenjiacheng/data/imo/proofbench.jsonl
   # --eval-prompt-data APhO_2025_visual $VISUAL_PATH/APhO_2025_visual.jsonl IPhO_2025_visual $VISUAL_PATH/IPhO_2025_visual.jsonl EuPhO_2025_visual $VISUAL_PATH/EuPhO_2025_visual.jsonl NBPhO_2025_visual $VISUAL_PATH/NBPhO_2025_visual.jsonl PanPhO_2025_visual $VISUAL_PATH/PanPhO_2025_visual.jsonl FMA_2025_visual $VISUAL_PATH/'F=MA_2025_visual.jsonl' PanMechanics_2025_visual $VISUAL_PATH/PanMechanics_2025_visual.jsonl
   --n-samples-per-eval-prompt ${N_SAMPLES_PER_EVAL[@]}
   --eval-max-response-len 81920
   --eval-top-p 0.95
   --eval-temperature 1.0
)

PERF_ARGS=(
   --tensor-model-parallel-size $TP_SIZE
   --sequence-parallel
   --pipeline-model-parallel-size $PP_SIZE
   --context-parallel-size $CP_SIZE
   # --expert-model-parallel-size $EP_SIZE
   # --expert-tensor-parallel-size $ETP_SIZE
   # --overlap-param-gather
   # --overlap-grad-reduce
   # --tp-comm-overlap
   # --moe-grouped-gemm
   # --moe-token-dispatcher-type alltoall
   # --recompute-granularity full
   # --recompute-method uniform
   # --recompute-num-layers 1
   # --micro-batch-size 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu $MAX_TOKENS_PER_GPU

   # use deepep for megatron
   # --moe-enable-deepep
   # --moe-token-dispatcher-type flex

   # fp8
   --transformer-impl transformer_engine
   --bf16
   --fp8-format e4m3
   --fp8-recipe blockwise
)

GRPO_ARGS=(
   --advantage-estimator grpo
   # --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
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
   --use-wandb
   --wandb-project slime-Qwen3-30B-A3B
   --wandb-group ${EXP_NAME}
   --wandb-key ${WANDB_KEY}
   --wandb-dir ${WANDB_DIR}
   --wandb-mode offline
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 2
   --sglang-mem-fraction-static 0.7
   # --sglang-enable-ep-moe
   # --sglang-expert-parallel-size 4
   # --sglang-kv-cache-dtype fp8_e4m3
   # dp attention
   # --sglang-enable-dp-attention
   # --sglang-dp-size 4
   # --sglang-moe-dense-tp-size 1
   # --sglang-enable-dp-lm-head

   # enable deepep for sglang
   # --sglang-enable-deepep-moe
   # --sglang-disable-radix-cache
   # --sglang-moe-a2a-backend deepep
   # --sglang-deepep-mode auto
   --sglang-cuda-graph-bs 1 2 4 8 $(seq 16 8 256)
   # --sglang-attention-backend fa3
   # --use-slime-router
   # --sglang-server-concurrency 256
   # --sglang-log-level INFO
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
#\"PYTORCH_CUDA_ALLOC_CONF\": \"max_split_size_mb:128,expandable_segments:False\",
# Build the runtime environment JSON with proper variable substitution
RUNTIME_ENV_JSON="{
    \"env_vars\": {
      \"PYTHONPATH\": \"/root/Megatron-LM/\",
      \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
      \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
      \"MASTER_ADDR\": \"${MASTER_ADDR}\",
      \"NVTE_FP8_BLOCK_SCALING_FP32_SCALES\": \"1\",
      \"NCCL_TIMEOUT_MS\":\"36000000\"
    }
  }"


# ========= 启动 Ray =========
if [ "$RANK" == "0" ]; then
  if [ -f "$READY_FLAG_FILE" ]; then
    rm -f "$READY_FLAG_FILE"
  fi
  echo "[RANK 0] Starting Ray Head node..."
  ray start --head --port=6379 --node-ip-address=$MASTER_ADDR --num-gpus=8 --disable-usage-stats
  echo "[RANK 0] Ray Head started successfully."
  touch "$READY_FLAG_FILE"
else
  echo "[RANK $RANK] Waiting for Ray Head to be ready..."
  sleep 10

  MAX_WAIT=120
  elapsed=0
  while [ ! -f "$READY_FLAG_FILE" ] && [ $elapsed -lt $MAX_WAIT ]; do
    echo "  ⏳ Still waiting... ($elapsed/$MAX_WAIT)"
    sleep 2
    elapsed=$((elapsed + 2))
  done

  if [ ! -f "$READY_FLAG_FILE" ]; then
    echo "❌ Timed out waiting for Ray Head to be ready."
    exit 1
  fi

  WORKER_IP=$(hostname -I | awk '{print $1}')

  echo "[RANK $RANK] Detected Ray Head at $MASTER_ADDR, starting worker at $WORKER_IP..."
  ray start --address=$MASTER_ADDR:6379 --node-ip-address=$WORKER_IP --num-gpus=8 --disable-usage-stats --block
  echo "[RANK $RANK] Worker started successfully."
fi

wait
# --colocate \ --rollout-num-gpus 48 \
# ========= 仅主节点提交 Ray Job =========
if [ "$RANK" == "0" ]; then
  ray job submit --address="http://127.0.0.1:8265" \
     --runtime-env-json="${RUNTIME_ENV_JSON}" \
     -- python3 train.py \
     --actor-num-nodes ${NODE_COUNT} \
     --actor-num-gpus-per-node 8 \
     --no-load-optim \
     --no-save-optim \
     --colocate \
     --offload \
     ${MODEL_ARGS[@]} \
     ${CKPT_ARGS[@]} \
     ${ROLLOUT_ARGS[@]} \
     ${OPTIMIZER_ARGS[@]} \
     ${GRPO_ARGS[@]} \
     ${WANDB_ARGS[@]} \
     ${PERF_ARGS[@]} \
     ${EVAL_ARGS[@]} \
     ${SGLANG_ARGS[@]} \
     ${MISC_ARGS[@]}
fi