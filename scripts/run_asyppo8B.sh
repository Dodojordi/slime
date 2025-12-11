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

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/models/qwen3-8B.sh"

CKPT_ARGS=(
   # --hf-checkpoint /mnt/shared-storage-user/p1-shared/Qwen/Qwen3-8B-Base
   # --ref-load /mnt/shared-storage-user/p1-shared/liyizhuo/share/models/Qwen3-8B-Base-torch_dist
   # --load /mnt/shared-storage-user/p1-shared/Qwen/Qwen3-8B-Base
   # --save /mnt/shared-storage-user/p1-shared/liyizhuo/share/save/Qwen3-8B-Base-slime_PPO/
   # --hf-checkpoint /mnt/shared-storage-user/p1-shared/Qwen/Qwen3-4B-Base
   # --ref-load /mnt/shared-storage-user/p1-shared/liyizhuo/share/models/Qwen3-4B-Base-torch_dist
   # --load /mnt/shared-storage-user/p1-shared/Qwen/Qwen3-4B-Base
   # --save /mnt/shared-storage-user/p1-shared/liyizhuo/share/save/Qwen3-4B-Base-slime_PPO/
   --hf-checkpoint /mnt/shared-storage-user/p1-shared/Qwen/Qwen3-8B-Base
   --ref-load /mnt/shared-storage-user/p1-shared/liyizhuo/share/models/Qwen3-8B-Base-torch_dist
   --load /mnt/shared-storage-user/p1-shared/Qwen/Qwen3-8B-Base
   --save /mnt/shared-storage-user/p1-shared/liyizhuo/share/save/AsyPPO/Base/Actor_8B

   # --critic-hf-checkpoint /mnt/shared-storage-user/p1-shared/Qwen/Qwen3-1.7B-Base
   # --critic-load /mnt/shared-storage-user/p1-shared/Qwen/Qwen3-1.7B-Base
   --critic-save /mnt/shared-storage-user/p1-shared/liyizhuo/share/save/AsyPPO/Base/Critic_8B

   --save-interval 500
)

ROLLOUT_ARGS=(
   # --prompt-data /mnt/shared-storage-user/p1-shared/liyizhuo/share/data/DAPO-Math-17k/dapo-math-17k.jsonl
   --prompt-data /mnt/shared-storage-user/p1-shared/liyizhuo/share/data/AsyPPO/train.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type dapo
   --reward-key score
   
   --num-rollout 500
   --rollout-batch-size 32
   --n-samples-per-prompt 8
   --rollout-max-response-len 10240
   --rollout-temperature 0.8

   --global-batch-size 128
   --balance-data
)

EVAL_ARGS=(
   --eval-interval 10
   # --eval-prompt-data aime /mnt/shared-storage-user/p1-shared/liyizhuo/share/data/aime_2024/aime-2024.jsonl
   # --eval-prompt-data aime /mnt/shared-storage-user/p1-shared/liyizhuo/share/data/aime_2024/aime-2024-deepmathformat.jsonl
   --eval-prompt-data aime2425 /mnt/shared-storage-user/p1-shared/liyizhuo/share/data/AsyPPO/test.jsonl
   --n-samples-per-eval-prompt 16
   --eval-max-response-len 10240
   --eval-top-p 0.7
   --eval-temperature 0.8
   --log-passrate
)

PERF_ARGS=(
   --tensor-model-parallel-size 2
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
   --max-tokens-per-gpu 9216
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
   # --normalize-advantages
   --critic-lr 1e-6
   --num-steps-per-rollout 2
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
   --wandb-project AsyPPO
   --wandb-group critic_size
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
   # --use-hf-config-for-megatron
)

# CUSTOM_ARGS=(
#    --log-position-value-stats
#    --max-log-positions 500
#    --lambd-actor 0.95
#    --lambd-critic 0.95
# )

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

ray job submit --address="http://127.0.0.1:26500" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 4 \
   --colocate \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${PPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MODEL_ARGS[@]} \
   ${CUSTOM_ARGS[@]} \
   ${MISC_ARGS[@]}