#!/bin/bash

set -ex

SCRIPT=$1
NODE_COUNT=$2
export KUBEBRAIN_CLUSTER_ENTRY="https://h.pjlab.org.cn"
export KUBEBRAIN_NAMESPACE="ailab-p1"
export BRAINPP_ACCESS_KEY_ID=78af6a9bcd173db6da99c57525b437cf
export BRAINPP_SECRET_ACCESS_KEY=53eb947566828a644a753c5a5ccd103c
# --positive-tags node/gpu-lg-cmc-h-h200-2508.host.h.pjlab.org.cn \

rjob submit --name=rm-cgq-8001 \
  --gpu=2 --memory=400000 --cpu=32 \
  --charged-group=p1_gpu \
  --private-machine=group \
  --mount=gpfs://gpfs1/p1-shared:/mnt/shared-storage-user/p1-shared \
  --image=registry.h.pjlab.org.cn/ailab-p1-p1_gpu/slime:20251111-v1 \
  --positive-tags node/gpu-lg-cmc-h-h200-2511.host.h.pjlab.org.cn \
  -P 1 --host-network=true \
  -e DISTRIBUTED_JOB=true \
  -e NCCL_PXN_DISABLE=0 \
  -- bash /mnt/shared-storage-user/p1-shared/chenjiacheng/p1-slime/rm-cgq/run_rm.sh -m 34886 -r 8005