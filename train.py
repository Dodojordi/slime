import ray
from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_WEIGHTS

try:
    from sglang.srt.constants import GPU_MEMORY_TYPE_CUDA_GRAPH
except ImportError:
    GPU_MEMORY_TYPE_CUDA_GRAPH = None

from slime.ray.placement_group import create_placement_groups, create_rollout_manager, create_training_models
from slime.utils.arguments import parse_args
from slime.utils.logging_utils import configure_logger
from slime.utils.misc import should_run_periodic_action
from slime.utils.tracking_utils import init_tracking
from slime.utils.add_my_custom_arguments import add_my_custom_arguments

import logging
logger = logging.getLogger(__name__)

def train(args):
    configure_logger()
    # allocate the GPUs
    pgs = create_placement_groups(args)
    init_tracking(args)

    # create the rollout manager, with sglang engines inside.
    # need to initialize rollout manager first to calculate num_rollout
    rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, pgs["rollout"])

    # 修改：返回三个模型（添加 critic_model2）
    actor_model, critic_model, critic_model2 = create_training_models(args, pgs, rollout_manager)
    
    if args.offload_rollout:
        ray.get(rollout_manager.onload.remote(tags=[GPU_MEMORY_TYPE_WEIGHTS]))

    # always update weight first so that sglang has the loaded weights from training.
    actor_model.update_weights()

    if args.check_weight_update_equal:
        ray.get(rollout_manager.check_weights.remote(action="compare"))

    if args.offload_rollout:
        if GPU_MEMORY_TYPE_CUDA_GRAPH is not None:
            ray.get(rollout_manager.onload.remote(tags=[GPU_MEMORY_TYPE_CUDA_GRAPH]))
        ray.get(rollout_manager.onload.remote(tags=[GPU_MEMORY_TYPE_KV_CACHE]))

    # special case for eval-only
    if args.num_rollout == 0 and args.eval_interval is not None:
        ray.get(rollout_manager.eval.remote(rollout_id=0))

    def offload_train():
        if args.offload_train:
            if args.use_critic:
                critic_model.offload()
                # 添加：offload 第二个 critic
                if args.use_critic2:
                    critic_model2.offload()
                if rollout_id >= args.num_critic_only_steps:
                    actor_model.offload()
            else:
                actor_model.offload()
        else:
            actor_model.clear_memory()

    def onload_rollout():
        if args.offload_rollout:
            ray.get(rollout_manager.onload.remote(tags=[GPU_MEMORY_TYPE_WEIGHTS]))

    # train loop.
    # note that for async training, one can change the position of the sync operation(ray.get).
    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        if args.eval_interval is not None and rollout_id == 0 and args.eval_first:
            ray.get(rollout_manager.eval.remote(rollout_id))

        rollout_data_ref = ray.get(rollout_manager.generate.remote(rollout_id))

        if args.offload_rollout:
            ray.get(rollout_manager.offload.remote())

        # 三阶段训练：Critic forward → Critic send values → Actor compute → Critic complete
        if args.use_critic:
            logger.info(f"[train.py] Rollout {rollout_id}: Stage 1 - Critics forward pass")
            # Stage 1: Critics 前向传播（保存状态并立即返回）
            critic_train_handle = critic_model.async_train(rollout_id, rollout_data_ref)
            if args.use_critic2:
                critic_train_handle2 = critic_model2.async_train(rollout_id, rollout_data_ref)
            
            logger.info(f"[train.py] Rollout {rollout_id}: Waiting for critic1 forward pass")
            ray.get(critic_train_handle)
            logger.info(f"[train.py] Rollout {rollout_id}: Critic1 forward pass completed")
            
            if args.use_critic2:
                logger.info(f"[train.py] Rollout {rollout_id}: Waiting for critic2 forward pass")
                ray.get(critic_train_handle2)
                logger.info(f"[train.py] Rollout {rollout_id}: Critic2 forward pass completed")
            
            # Stage 2: Critics 发送 values 给 Actor（异步启动，进入等待状态）
            if rollout_id >= args.num_critic_only_steps:
                logger.info(f"[train.py] Rollout {rollout_id}: Stage 2 - Starting async_train_critic_with_returns")
                critic_finish_handles = critic_model.async_train_critic_with_returns()
                if args.use_critic2:
                    critic_finish_handles2 = critic_model2.async_train_critic_with_returns()
                logger.info(f"[train.py] Rollout {rollout_id}: async_train_critic_with_returns started")
                
                # Stage 3: Actor 接收 values，计算 returns，广播给 Critics
                logger.info(f"[train.py] Rollout {rollout_id}: Stage 3 - Starting actor training")
                ray.get(actor_model.async_train(rollout_id, rollout_data_ref))
                logger.info(f"[train.py] Rollout {rollout_id}: Actor training completed")
                
                # Stage 4: 等待 Critics 接收 returns 并完成训练
                logger.info(f"[train.py] Rollout {rollout_id}: Stage 4 - Waiting for critic1 to complete")
                ray.get(critic_finish_handles)
                logger.info(f"[train.py] Rollout {rollout_id}: Critic1 training completed")
                
                if args.use_critic2:
                    logger.info(f"[train.py] Rollout {rollout_id}: Waiting for critic2 to complete")
                    ray.get(critic_finish_handles2)
                    logger.info(f"[train.py] Rollout {rollout_id}: Critic2 training completed")
            else:
                # Critic-only 步骤：直接完成（不需要和 Actor 交互）
                logger.info(f"[train.py] Rollout {rollout_id}: Critic-only step, no actor training")
                pass
        else:
            ray.get(actor_model.async_train(rollout_id, rollout_data_ref))

        if should_run_periodic_action(rollout_id, args.save_interval, num_rollout_per_epoch):
            if (not args.use_critic) or (rollout_id >= args.num_critic_only_steps):
                actor_model.save_model(rollout_id)
            if args.use_critic:
                critic_model.save_model(rollout_id)
            # 添加：保存第二个 critic
            if args.use_critic2:
                critic_model2.save_model(rollout_id)
            if args.rollout_global_dataset:
                ray.get(rollout_manager.save.remote(rollout_id))

        offload_train()
        onload_rollout()
        actor_model.update_weights()

        if args.offload_rollout:
            if GPU_MEMORY_TYPE_CUDA_GRAPH is not None:
                ray.get(rollout_manager.onload.remote(tags=[GPU_MEMORY_TYPE_CUDA_GRAPH]))
            ray.get(rollout_manager.onload.remote(tags=[GPU_MEMORY_TYPE_KV_CACHE]))

        if should_run_periodic_action(rollout_id, args.eval_interval, num_rollout_per_epoch):
            ray.get(rollout_manager.eval.remote(rollout_id))

    ray.get(rollout_manager.dispose.remote())


if __name__ == "__main__":
    args = parse_args(add_custom_arguments=add_my_custom_arguments)
    train(args)
