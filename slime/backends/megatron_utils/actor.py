import logging
import os
import socket
from argparse import Namespace
from contextlib import nullcontext

import ray
import torch
import torch.distributed as dist
from megatron.core import mpu
from ray.actor import ActorHandle
from torch_memory_saver import torch_memory_saver
from transformers import AutoConfig, AutoTokenizer

from slime.ray.train_actor import TrainRayActor
from slime.utils import train_dump_utils
from slime.utils.context_utils import with_defer
from slime.utils.data import process_rollout_data
from slime.utils.distributed_utils import get_gloo_group, init_process_group
from slime.utils.memory_utils import clear_memory, print_memory
from slime.utils.ray_utils import Box
from slime.utils.reloadable_process_group import destroy_process_groups, monkey_patch_torch_dist, reload_process_groups
from slime.utils.routing_replay import RoutingReplay
from slime.utils.timer import Timer, inverse_timer, timer
from slime.utils.tracking_utils import init_tracking
from slime.utils.types import RolloutBatch

from ...utils.profile_utils import TrainProfiler
from ...utils.tensor_backper import TensorBackuper
from .checkpoint import load_checkpoint
from .cp_utils import slice_log_prob_with_cp, slice_with_cp
from .data import DataIterator, get_data_iterator, log_perf_data, log_rollout_data, sync_actor_critic_data
from .initialize import init, is_megatron_main_rank
from .loss import compute_advantages_and_returns, get_log_probs_and_entropy, get_values
from .model import forward_only, initialize_model_and_optimizer, save, train
from .update_weight.common import named_params_and_buffers
from .update_weight.update_weight_from_distributed import UpdateWeightFromDistributed
from .update_weight.update_weight_from_tensor import UpdateWeightFromTensor

logging.getLogger("megatron").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


class MegatronTrainRayActor(TrainRayActor):
    @with_defer(lambda: Timer().start("train_wait"))
    def init(
        self,
        args: Namespace,
        role: str,
        with_ref: bool = False,
    ) -> int | None:
        monkey_patch_torch_dist()

        super().init(args, role, with_ref)
        
        # === 新增：如果是 critic 且使用不同模型，重新应用 HF config ===
        if (
            role == "critic"
            and hasattr(args, 'critic_hf_checkpoint')
            and args.critic_hf_checkpoint != args.hf_checkpoint
            and hasattr(args, "use_hf_config_for_megatron")
            and args.use_hf_config_for_megatron
        ):
            logger.info(f"Reapplying HF config for critic from {args.critic_hf_checkpoint} to {args.hf_checkpoint}")
            from slime.backends.megatron_utils.config_mapping import get_mapper
            
            critic_hf_config = AutoConfig.from_pretrained(args.critic_hf_checkpoint, trust_remote_code=True)
            if args.use_hf_config_for_megatron:
                megatron_config_from_hf = get_mapper(critic_hf_config.model_type)(critic_hf_config)
                # 覆盖模型架构参数
                for key, value in megatron_config_from_hf.transformer_config.items():
                    setattr(args, key, value)
                for key, value in megatron_config_from_hf.gpt_model_args.items():
                    setattr(args, key, value)
            # 更新 hf_checkpoint 指向 critic 的
            args.hf_checkpoint = args.critic_hf_checkpoint
        # === 新增结束 ===

        init(args)

        if is_megatron_main_rank():
            init_tracking(args, primary=False)

        self.prof = TrainProfiler(args)

        # read config and tokenizer serialized to prevent concurrent writing bug.
        for i in range(dist.get_world_size()):
            if i == dist.get_rank():
                self.hf_config = AutoConfig.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
                self.tokenizer = AutoTokenizer.from_pretrained(self.args.hf_checkpoint, trust_remote_code=True)
            dist.barrier(group=get_gloo_group())

        self.train_parallel_config = {
            "dp_size": mpu.get_data_parallel_world_size(with_context_parallel=False),
        }

        if self.args.debug_rollout_only:
            return 0

        if role == "critic":
            self.args.load = self.args.critic_ref_load
            self.args.save = self.args.critic_save
            self.args.lr = self.args.critic_lr
            self.args.lr_warmup_iters = self.args.critic_lr_warmup_iters
            if hasattr(args, 'use_asyppo') and args.use_asyppo:
                # 检查 critic_load 是否为有效的 Megatron checkpoint
                critic_load_valid = (
                    self.args.critic_load is not None
                    and os.path.exists(self.args.critic_load)
                    and os.path.exists(os.path.join(self.args.critic_load, "latest_checkpointed_iteration.txt"))
                )
                
                if not critic_load_valid:
                    # critic_load 无效，尝试回退
                    if hasattr(args, 'critic_ref_load') and args.critic_ref_load:
                        logger.info(
                            f"critic_load '{self.args.critic_load}' is not a valid Megatron checkpoint. "
                            f"Falling back to critic_ref_load: {args.critic_ref_load}"
                        )
                        self.args.load = args.critic_ref_load
                        self.args.no_load_optim = True
                        self.args.no_load_rng = True
                        self.args.finetune = True
                    else:
                        # 如果没有 critic_ref_load，回退到普通的 ref_load
                        logger.warning(
                            f"critic_load '{self.args.critic_load}' is not valid and no critic_ref_load specified. "
                            f"Falling back to ref_load: {args.ref_load}"
                        )
                        self.args.load = args.ref_load
                        self.args.no_load_optim = True
                        self.args.no_load_rng = True
                        self.args.finetune = True
                else:
                    # critic_load 有效，直接使用
                    logger.info(f"Loading critic from valid checkpoint: {self.args.critic_load}")
                    self.args.load = self.args.critic_load

        (self.model, self.optimizer, self.opt_param_scheduler, loaded_rollout_id) = initialize_model_and_optimizer(
            args, role
        )

        if role == "critic":
            if self.args.offload_train:
                self.sleep()
            return

        start_rollout_id = loaded_rollout_id + 1

        self.weights_backuper = TensorBackuper.create(
            source_getter=lambda: named_params_and_buffers(
                self.args,
                self.model,
                convert_to_global_name=args.megatron_to_hf_mode == "raw",
                translate_gpu_to_cpu=not self.args.enable_weights_backuper,
            ),
            single_tag=None if args.enable_weights_backuper else "actor",
        )
        self._active_model_tag: str | None = "actor"
        self.weights_backuper.backup("actor")

        if with_ref:
            self.load_other_checkpoint("ref", args.ref_load)

        if self.args.keep_old_actor:
            # Load old_actor checkpoint
            self.load_other_checkpoint("old_actor", args.load)
            # Create rollout_actor as a copy of current actor
            if args.update_weights_interval == 1:
                self.weights_backuper.backup("rollout_actor")

        if self.args.vocab_size is None:
            self.args.vocab_size = self.tokenizer.vocab_size

        update_weight_cls = UpdateWeightFromTensor if self.args.colocate else UpdateWeightFromDistributed
        self.weight_updater = update_weight_cls(
            self.args,
            self.model,
            weights_getter=lambda: self.weights_backuper.get("actor"),
            model_name=type(self.hf_config).__name__.lower() if self.args.model_name is None else self.args.model_name,
            quantization_config=getattr(self.hf_config, "quantization_config", None),
        )

        # empty cache after initialization
        clear_memory()

        if self.args.offload_train:
            # recover to actor in the end.
            self._switch_model("actor")
            self.sleep()

        self.rollout_engines = None

        self.rollout_data_postprocess = None
        if self.args.rollout_data_postprocess_path is not None:
            from slime.utils.misc import load_function

            self.rollout_data_postprocess = load_function(self.args.rollout_data_postprocess_path)

        self.prof.on_init_end()

        return start_rollout_id

    @timer
    def sleep(self) -> None:
        assert self.args.offload_train

        clear_memory(clear_host_memory=True)
        print_memory("before offload model")
        destroy_process_groups()

        torch_memory_saver.pause()

        print_memory("after offload model")

    @timer
    def wake_up(self) -> None:
        assert self.args.offload_train
        print_memory("before wake_up model")

        torch_memory_saver.resume()

        clear_memory()
        reload_process_groups()
        print_memory("after wake_up model")

    def _get_rollout_data(self, rollout_data_ref: Box) -> RolloutBatch:
        # Fetch data through ray on CPU, not sure if this will be performance bottleneck.
        # Both first pp stage and the last pp stage will receive the data.
        rollout_data = process_rollout_data(
            self.args,
            rollout_data_ref,
            mpu.get_data_parallel_rank(with_context_parallel=False),
            mpu.get_data_parallel_world_size(with_context_parallel=False),
        )
        # TODO: this is ugly, move to somewhere else?
        # move tokens to GPU in advance
        rollout_data["tokens"] = [
            torch.tensor(t, dtype=torch.long, device=torch.cuda.current_device()) for t in rollout_data["tokens"]
        ]
        rollout_data["loss_masks"] = [
            torch.tensor(t, dtype=torch.int, device=torch.cuda.current_device()) for t in rollout_data["loss_masks"]
        ]
        if "rollout_log_probs" in rollout_data:
            rollout_data["rollout_log_probs"] = [
                torch.tensor(
                    slice_log_prob_with_cp(log_prob, total_length, response_length),
                    device=torch.cuda.current_device(),
                    dtype=torch.float32,
                )
                for log_prob, total_length, response_length in zip(
                    rollout_data["rollout_log_probs"],
                    rollout_data["total_lengths"],
                    rollout_data["response_lengths"],
                    strict=False,
                )
            ]
        if "rollout_routed_experts" in rollout_data:
            rollout_data["rollout_routed_experts"] = [
                torch.from_numpy(r) for r in rollout_data["rollout_routed_experts"]
            ]
        return rollout_data

    def _switch_model(self, target_tag: str) -> None:
        if target_tag not in self.weights_backuper.backup_tags:
            raise ValueError(f"Cannot switch to unknown model tag: {target_tag}")
        self.weights_backuper.restore(target_tag)
        self._active_model_tag = target_tag

    def fill_routing_replay(self, data_iterator, num_microbatches, rollout_data):
        if "rollout_routed_experts" not in rollout_data:
            raise ValueError(
                "rollout_routed_experts is required in rollout_data when use_rollout_routing_replay is set."
            )

        from megatron.core.transformer.transformer_block import get_num_layers_to_build
        from megatron.core.transformer.transformer_layer import get_transformer_layer_offset

        from slime.utils.routing_replay import RoutingReplay

        for iterator in data_iterator:
            iterator.reset()

        tp_rank = mpu.get_tensor_model_parallel_rank()
        tp_size = mpu.get_tensor_model_parallel_world_size()

        def pad_func(experts, pad):
            _, num_layers, topk = experts.shape
            pad = (
                torch.arange(
                    pad * num_layers * topk,
                    device=experts.device,
                    dtype=experts.dtype,
                ).reshape((pad, num_layers, topk))
                % self.args.num_experts
            )
            return torch.cat([experts, pad], dim=0)

        for _ in range(sum(num_microbatches)):
            batch = data_iterator[0].get_next(["rollout_routed_experts", "tokens"])
            rollout_routed_experts = batch["rollout_routed_experts"]
            tokens = batch["tokens"]
            assert len(rollout_routed_experts) == len(tokens)
            for a, b in zip(rollout_routed_experts, tokens, strict=False):
                assert a.shape[0] == b.shape[0] - 1, f"{a.shape}, {b.shape}"

            # We need to pad the experts to the last token. We won't calculate loss on this token so this should be fine.
            # TODO: fuse this padding with the following slice_with_cp to reduce memory copy.
            rollout_routed_experts = [pad_func(r, 1) for r in rollout_routed_experts]
            # TODO: maybe extract a common process function for here and get_batch?
            rollout_routed_experts = [slice_with_cp(r, pad_func) for r in rollout_routed_experts]
            rollout_routed_experts = torch.cat(rollout_routed_experts, dim=0)
            pad_size = mpu.get_tensor_model_parallel_world_size() * self.args.data_pad_size_multiplier
            pad = (pad_size - rollout_routed_experts.size(0) % pad_size) % pad_size
            if pad != 0:
                rollout_routed_experts = pad_func(rollout_routed_experts, pad)

            if self.args.sequence_parallel:
                seqlen = rollout_routed_experts.size(0)
                assert seqlen % tp_size == 0
                start, end = seqlen // tp_size * tp_rank, seqlen // tp_size * (tp_rank + 1)
                rollout_routed_experts = rollout_routed_experts[start:end]

            routing_replay_offset = 0
            for vp_stage, model in enumerate(self.model):
                config = model.module.config
                num_layers_to_build = get_num_layers_to_build(config, vp_stage=vp_stage)
                offset = get_transformer_layer_offset(config, vp_stage=vp_stage)
                for layer_id in range(offset, offset + num_layers_to_build):
                    # skip dense layer
                    if isinstance(config.moe_layer_freq, int):
                        if layer_id % config.moe_layer_freq != 0:
                            continue
                    elif isinstance(config.moe_layer_freq, list):
                        assert len(config.moe_layer_freq) == config.num_layers
                        if config.moe_layer_freq[layer_id] == 0:
                            continue
                    layer_routed_experts = rollout_routed_experts[:, layer_id]
                    RoutingReplay.all_routing_replays[routing_replay_offset].record(layer_routed_experts)
                    routing_replay_offset += 1
            assert routing_replay_offset == len(RoutingReplay.all_routing_replays)

        del rollout_data["rollout_routed_experts"]

        for iterator in data_iterator:
            iterator.reset()

    def compute_log_prob(
        self,
        data_iterator: list[DataIterator],
        num_microbatches: list[int],
        store_prefix: str = "",
    ) -> dict[str, list[torch.Tensor]]:

        with timer(f"{store_prefix}log_probs"):
            return forward_only(
                get_log_probs_and_entropy,
                self.args,
                self.model,
                data_iterator,
                num_microbatches,
                store_prefix=store_prefix,
            )

    def train(self, rollout_id: int, rollout_data_ref: Box) -> None:
        if self.args.offload_train:
            self.wake_up()

        with timer("data_preprocess"):
            rollout_data = self._get_rollout_data(rollout_data_ref)
            if self.args.debug_rollout_only:
                log_rollout_data(rollout_id, self.args, rollout_data)
                return

        if self.role == "critic":
            return self.train_critic(rollout_id, rollout_data)
        else:
            return self.train_actor(rollout_id, rollout_data)

    def train_critic(self, rollout_id: int, rollout_data: RolloutBatch) -> None:
        # Create data iterator for log_probs and train.
        data_iterator, num_microbatches = get_data_iterator(self.args, self.model, rollout_data)
        rollout_data.update(
            forward_only(
                get_values,
                self.args,
                self.model,
                data_iterator,
                num_microbatches,
            )
        )
        
        # ============ 新增：统计position-value关系 ============
        # if getattr(self.args, 'log_position_value_stats', False) and "values" in rollout_data:
        #     # 只在最后一个pipeline stage记录（与model.py中的logging保持一致）
        #     if (
        #         mpu.get_data_parallel_rank(with_context_parallel=True) == 0
        #         and mpu.get_tensor_model_parallel_rank() == 0
        #         and mpu.get_pipeline_model_parallel_rank() == mpu.get_pipeline_model_parallel_world_size() - 1
        #     ):
        #         import numpy as np
        #         from slime.utils import tracking_utils
                
        #         # 计算训练步数（与model.py中保持一致）
        #         num_steps_per_rollout = len(num_microbatches)
        #         # 这里用rollout的起始step_id，即rollout_id * num_steps_per_rollout
        #         accumulated_step_id = rollout_id * num_steps_per_rollout
                
        #         # 统计每个position的value
        #         position_value_stats = {}  # {position: list of values}
                
        #         for values_tensor in rollout_data["values"]:
        #             # values_tensor 的形状是 [response_length]
        #             for pos, val in enumerate(values_tensor.tolist()):
        #                 if pos not in position_value_stats:
        #                     position_value_stats[pos] = []
        #                 position_value_stats[pos].append(val)
                
        #         # 计算每个position的平均value并记录
        #         log_dict = {}
        #         max_position = max(position_value_stats.keys()) if position_value_stats else 0
                
        #         max_log_positions = getattr(self.args, 'max_log_positions', 50)
        #         for pos in sorted(position_value_stats.keys())[:max_log_positions]:
        #             vals = np.array(position_value_stats[pos])
        #             # 只记录平均值
        #             log_dict[f"critic/value_pos_{pos}_mean"] = float(vals.mean())
                    
        #             # 可选：如果需要更多统计信息，可以取消下面的注释
        #             # log_dict[f"critic/value_pos_{pos}_std"] = float(vals.std())
        #             # log_dict[f"critic/value_pos_{pos}_count"] = len(vals)
                
        #         # 额外记录一些元信息
        #         log_dict["critic/max_position"] = max_position
        #         log_dict["critic/num_samples"] = len(rollout_data["values"])
        #         log_dict["train/step"] = accumulated_step_id
                
        #         # 使用tracking_utils记录
        #         tracking_utils.log(self.args, log_dict, step_key="train/step")
                
        #         # 格式化输出position-value统计信息
        #         position_stats_str = ", ".join([
        #             f"pos_{pos}={log_dict[f'critic/value_pos_{pos}_mean']:.4f}" 
        #             for pos in sorted(position_value_stats.keys())[:max_log_positions]
        #         ])
        #         logger.info(f"Position-Value Stats at step {accumulated_step_id}: "
        #                 f"max_pos={max_position}, num_samples={len(rollout_data['values'])}")
        #         logger.info(f"Position-wise mean values: {position_stats_str}")
        
        if getattr(self.args, 'log_position_value_stats', False) and "values" in rollout_data:
            # 只在最后一个pipeline stage记录（与model.py中的logging保持一致）
            if (
                mpu.get_data_parallel_rank(with_context_parallel=True) == 0
                and mpu.get_tensor_model_parallel_rank() == 0
                and mpu.get_pipeline_model_parallel_rank() == mpu.get_pipeline_model_parallel_world_size() - 1
            ):
                import numpy as np
                from slime.utils import tracking_utils
                from torch.nn.utils.rnn import pad_sequence
                
                # 计算训练步数（与model.py中保持一致）
                num_steps_per_rollout = len(num_microbatches)
                # 这里用rollout的起始step_id，即rollout_id * num_steps_per_rollout
                accumulated_step_id = rollout_id * num_steps_per_rollout
                
                # 使用 tensor 操作批量计算每个位置的均值
                # 1. Pad 到最大长度以便批量计算
                padded_values = pad_sequence(rollout_data["values"], batch_first=True, padding_value=float('nan'))  # [B, MaxLen]
                
                # 2. 计算最大位置（用于元信息记录）
                max_position = padded_values.shape[1] - 1 if padded_values.shape[1] > 0 else 0
                
                # 3. 只处理前 max_log_positions 个位置
                max_log_positions = getattr(self.args, 'max_log_positions', 50)
                target_len = min(padded_values.shape[1], max_log_positions)
                sliced_values = padded_values[:, :target_len]  # [B, target_len]
                
                # 4. 批量计算每个位置的均值（忽略 padding 的 nan）
                means = torch.nanmean(sliced_values, dim=0)  # [target_len]
                
                # 5. 构建 log_dict（保持原有格式）
                log_dict = {}
                # 只记录有有效数据的位置（非 nan 的位置）
                valid_positions = []
                for pos in range(target_len):
                    mean_val = float(means[pos].item())
                    if not torch.isnan(means[pos]):
                        log_dict[f"critic/value_pos_{pos}_mean"] = mean_val
                        valid_positions.append(pos)
                
                # 额外记录一些元信息（保持原有格式）
                log_dict["critic/max_position"] = max_position
                log_dict["critic/num_samples"] = len(rollout_data["values"])
                log_dict["train/step"] = accumulated_step_id
                
                # 使用tracking_utils记录
                tracking_utils.log(self.args, log_dict, step_key="train/step")
                
                # 格式化输出position-value统计信息（保持原有格式）
                position_stats_str = ", ".join([
                    f"pos_{pos}={log_dict[f'critic/value_pos_{pos}_mean']:.4f}" 
                    for pos in valid_positions
                ])
                logger.info(f"Position-Value Stats at step {accumulated_step_id}: "
                        f"max_pos={max_position}, num_samples={len(rollout_data['values'])}")
                logger.info(f"Position-wise mean values: {position_stats_str}")
        
        # ============ 结束：统计position-value关系 ============
        if rollout_id >= self.args.num_critic_only_steps:
            sync_actor_critic_data(self.args, rollout_data, self._actor_critic_groups)
        # sync_actor_critic_data(self.args, rollout_data, self._actor_critic_groups)
        compute_advantages_and_returns(self.args, rollout_data)

        self.args.loss_type = "value_loss"
        # train(
        #     rollout_id,
        #     self.model,
        #     self.optimizer,
        #     self.opt_param_scheduler,
        #     data_iterator,
        #     num_microbatches,
        # )
        with timer("critic_train"):
            train(
                rollout_id,
                self.model,
                self.optimizer,
                self.opt_param_scheduler,
                data_iterator,
                num_microbatches,
            )
        # --- end 计时critic训练 ---

        # 也做profiler记录
        self.prof.step(rollout_id=rollout_id)

        train_dump_utils.save_debug_train_data(self.args, rollout_id=rollout_id, rollout_data=rollout_data)

        log_perf_data(rollout_id, self.args)

    def train_actor(self, rollout_id: int, rollout_data: RolloutBatch) -> None:
        # Create data iterator for log_probs and train.
        data_iterator, num_microbatches = get_data_iterator(self.args, self.model, rollout_data)

        if self.args.use_rollout_routing_replay:
            self.fill_routing_replay(data_iterator, num_microbatches, rollout_data)

        with inverse_timer("train_wait"), timer("train"):
            if self.args.compute_advantages_and_returns:
                if "ref" in self.weights_backuper.backup_tags:
                    if self.args.use_routing_replay:
                        os.environ["ROUTING_REPLAY_STAGE"] = "fallthrough"
                    self._switch_model("ref")
                    rollout_data.update(
                        self.compute_log_prob(
                            data_iterator,
                            num_microbatches,
                            store_prefix="ref_",
                        )
                    )
                self._switch_model("old_actor" if self.args.keep_old_actor else "actor")
                if not self.args.use_rollout_logprobs or self.args.get_mismatch_metrics:
                    if self.args.use_routing_replay:
                        if self.args.use_rollout_routing_replay:
                            os.environ["ROUTING_REPLAY_STAGE"] = "replay_forward"
                        else:
                            os.environ["ROUTING_REPLAY_STAGE"] = "record"
                    rollout_data.update(
                        self.compute_log_prob(
                            data_iterator,
                            num_microbatches,
                            store_prefix="",
                        )
                    )
                    if self.args.use_rollout_routing_replay:
                        RoutingReplay.clear_all_forward()

                if self.args.use_critic:
                    sync_actor_critic_data(
                        self.args,
                        rollout_data,
                        self._actor_critic_groups,
                    )
                if self._active_model_tag != "actor":
                    self._switch_model("actor")

                # Calculate adv and returns. Need to performed before training (instead of on the fly),
                # because we may need normalize the whole rollout.
                compute_advantages_and_returns(self.args, rollout_data)

                # ============ 新增：统计position-advantage关系 ============
                if getattr(self.args, 'log_position_value_stats', False) and "advantages" in rollout_data:
                    # 只在最后一个pipeline stage记录（与value统计保持一致）
                    if (
                        mpu.get_data_parallel_rank(with_context_parallel=True) == 0
                        and mpu.get_tensor_model_parallel_rank() == 0
                        and mpu.get_pipeline_model_parallel_rank() == mpu.get_pipeline_model_parallel_world_size() - 1
                    ):
                        import numpy as np
                        from slime.utils import tracking_utils
                        from torch.nn.utils.rnn import pad_sequence

                        # 计算训练步数（与value统计保持一致）
                        num_steps_per_rollout = len(num_microbatches)
                        accumulated_step_id = rollout_id * num_steps_per_rollout

                        # 使用 tensor 操作批量计算每个位置的均值
                        # 1. Pad 到最大长度以便批量计算
                        padded_advantages = pad_sequence(rollout_data["advantages"], batch_first=True, padding_value=float('nan'))  # [B, MaxLen]

                        # 2. 计算最大位置（用于元信息记录）
                        max_position = padded_advantages.shape[1] - 1 if padded_advantages.shape[1] > 0 else 0

                        # 3. 只处理前 max_log_positions 个位置
                        max_log_positions = getattr(self.args, 'max_log_positions', 50)
                        target_len = min(padded_advantages.shape[1], max_log_positions)
                        sliced_advantages = padded_advantages[:, :target_len]  # [B, target_len]

                        # 4. 批量计算每个位置的均值、标准差等统计信息（忽略 padding 的 nan）
                        means = torch.from_numpy(np.nanmean(sliced_advantages.cpu().numpy(), axis=0))  # [target_len]
                        stds = torch.from_numpy(np.nanstd(sliced_advantages.cpu().numpy(), axis=0))    # [target_len]

                        # 5. 构建 log_dict
                        log_dict = {}
                        valid_positions = []
                        for pos in range(target_len):
                            mean_adv = float(means[pos].item())
                            std_adv = float(stds[pos].item())
                            if not np.isnan(means[pos].item()):
                                log_dict[f"actor/advantage_pos_{pos}_mean"] = mean_adv
                                log_dict[f"actor/advantage_pos_{pos}_std"] = std_adv
                                valid_positions.append(pos)

                        # 额外记录一些元信息
                        log_dict["actor/advantage_max_position"] = max_position
                        log_dict["actor/advantage_num_samples"] = len(rollout_data["advantages"])
                        log_dict["train/step"] = accumulated_step_id

                        # 使用tracking_utils记录
                        tracking_utils.log(self.args, log_dict, step_key="train/step")

                        # 格式化输出position-advantage统计信息
                        position_stats_str = ", ".join([
                            f"pos_{pos}=μ{log_dict[f'actor/advantage_pos_{pos}_mean']:.4f}±σ{log_dict[f'actor/advantage_pos_{pos}_std']:.4f}"
                            for pos in valid_positions
                        ])
                        logger.info(f"Position-Advantage Stats at step {accumulated_step_id}: "
                                    f"max_pos={max_position}, num_samples={len(rollout_data['advantages'])}")
                        logger.info(f"Position-wise mean±std advantages: {position_stats_str}")
                # ============ 结束：统计position-advantage关系 ============

            if self.rollout_data_postprocess is not None:
                self.rollout_data_postprocess(self.args)

            log_rollout_data(rollout_id, self.args, rollout_data)

            # Train
            if self.args.use_routing_replay:
                os.environ["ROUTING_REPLAY_STAGE"] = "replay_backward"
            with timer("actor_train"):
                train(
                    rollout_id,
                    self.model,
                    self.optimizer,
                    self.opt_param_scheduler,
                    data_iterator,
                    num_microbatches,
                )

            self.prof.step(rollout_id=rollout_id)

        train_dump_utils.save_debug_train_data(self.args, rollout_id=rollout_id, rollout_data=rollout_data)

        if self.args.use_routing_replay:
            RoutingReplay.clear_all()

        # update the cpu actor weight to the latest model
        self.weights_backuper.backup("actor")

        # Update ref model if needed
        if (
            self.args.ref_update_interval is not None
            and (rollout_id + 1) % self.args.ref_update_interval == 0
            and "ref" in self.weights_backuper.backup_tags
        ):
            with timer("ref_model_update"):
                if is_megatron_main_rank():
                    logger.info(f"Updating ref model at rollout_id {rollout_id}")
                self.weights_backuper.backup("ref")

        log_perf_data(rollout_id, self.args)

    @timer
    def save_model(self, iteration: int) -> None:
        if self.args.debug_rollout_only:
            return

        save(iteration, self.model, self.optimizer, self.opt_param_scheduler)

    @timer
    def update_weights(self) -> None:
        if self.args.debug_train_only or self.args.debug_rollout_only:
            return

        if self.args.offload_train:
            reload_process_groups()

        rollout_engines, rollout_engine_lock, num_new_engines = ray.get(
            self.rollout_manager.get_rollout_engines_and_lock.remote()
        )
        if num_new_engines > 0:
            self.weight_updater.connect_rollout_engines(rollout_engines, rollout_engine_lock)
            dist.barrier(group=get_gloo_group())

        with torch_memory_saver.disable() if self.args.offload_train else nullcontext():
            print_memory("before update_weights")
            self.weight_updater.update_weights()
            print_memory("after update_weights")

            if getattr(self.args, "keep_old_actor", False):
                if self.args.update_weights_interval == 1:
                    logger.info("updating model queue: rollout_actor -> old_actor, actor -> rollout_actor")
                    # Queue-style update: rollout_actor params -> old_actor, actor params -> rollout_actor
                    # First copy rollout_actor to old_actor
                    self.weights_backuper.copy(src_tag="rollout_actor", dst_tag="old_actor")
                    # Then copy current actor to rollout_actor
                    self.weights_backuper.backup("rollout_actor")
                else:
                    self.weights_backuper.backup("old_actor")

        if self.args.offload_train:
            destroy_process_groups()

    def load_other_checkpoint(self, model_tag: str, path: str) -> None:
        old_args = self.args.load, self.args.no_load_optim, self.args.no_load_rng, self.args.finetune
        self.args.load = path
        self.args.no_load_optim = True
        self.args.no_load_rng = True
        self.args.finetune = True

        if model_tag == "ref" and self.args.ref_ckpt_step is not None:
            old_ckpt_step = self.args.ckpt_step
            self.args.ckpt_step = self.args.ref_ckpt_step

        _, _ = load_checkpoint(
            self.model,
            None,
            None,
            checkpointing_context={},
            skip_load_to_model_and_opt=False,
        )
        self.args.load, self.args.no_load_optim, self.args.no_load_rng, self.args.finetune = old_args

        if model_tag == "ref" and self.args.ref_ckpt_step is not None:
            self.args.ckpt_step = old_ckpt_step

        self.weights_backuper.backup(model_tag)
        self._active_model_tag = model_tag

    def connect_actor_critic(
        self,
        actor_handle: ActorHandle | None = None,
        master_address: str | None = None,
        master_port: int | None = None,
    ) -> None:
        if self.role == "actor":
            master_address = ray.util.get_node_ip_address()
            with socket.socket() as sock:
                sock.bind(("", 0))
                master_port = sock.getsockname()[1]
            actor_handle.connect_actor_critic.remote(master_address=master_address, master_port=master_port)

        group_name = "actor_critic"
        world_size = 2
        self._actor_critic_groups = init_process_group(
            backend="nccl",
            init_method=f"tcp://{master_address}:{master_port}",
            world_size=world_size,
            rank=0 if self.role == "actor" else 1,
            group_name=group_name,
        )
