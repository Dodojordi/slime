import json
import logging
import os
import random
import re

import numpy as np
import pandas as pd
import ray

from slime.utils.types import MultimodalTypes, Sample

from .timer import Timer

__all__ = ["Dataset"]

logger = logging.getLogger(__name__)


# TODO: don't read the whole file into memory.
def read_file(path):
    path, row_slice = _parse_generalized_path(path)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Prompt dataset path '{path}' does not exist.")

    if path.endswith(".jsonl"):
        df = pd.read_json(path, lines=True, dtype={"label": str})
    elif path.endswith(".parquet"):
        df = pd.read_parquet(path, dtype_backend="pyarrow")
    else:
        raise ValueError(f"Unsupported file format: {path}. Supported formats are .jsonl and .parquet.")

    if row_slice is not None:
        logger.info(f"read_file path={path} slice {len(df)=} rows into {row_slice=}")
        df = df.iloc[row_slice]

    for _, row in df.iterrows():
        yield row.to_dict()


def _parse_generalized_path(s: str):
    if (m := re.match(r"^(?P<real_path>.*)@\[(?P<start>-?\d*):(?P<end>-?\d*)\]$", s)) is not None:
        path = m.group("real_path")
        start = int(x) if (x := m.group("start")) != "" else None
        end = int(x) if (x := m.group("end")) != "" else None
        return path, slice(start, end)

    return s, None


def _should_skip_prompt(prompt, tokenizer, processor, max_length, apply_chat_template_kwargs):
    if max_length is None:
        return False

    from slime.utils.processing_utils import prepare_model_inputs

    input_ids, _ = prepare_model_inputs(prompt, tokenizer, processor, None, apply_chat_template_kwargs)
    return len(input_ids) > max_length


def _build_messages(data: dict, prompt_key: str, multimodal_keys: dict = None):
    messages = data.get(prompt_key)

    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]

    if multimodal_keys:
        # Build mapping: placeholder -> (MultimodalType, content_list)
        multimodals = {}
        for type_name, data_key in multimodal_keys.items():
            mt = MultimodalTypes.get(type_name)
            if mt:
                multimodals[mt.placeholder] = (mt, list(data.get(data_key)))

        pattern = "(" + "|".join(re.escape(p) for p in multimodals.keys()) + ")"

        for message in messages:
            if isinstance(message["content"], str):
                content_list = []
                for segment in re.split(pattern, message["content"]):
                    if not segment:
                        continue
                    if segment in multimodals:
                        mt, content = multimodals[segment]
                        content_list.append({"type": mt.name, mt.name: content.pop(0)})
                    else:
                        content_list.append({"type": "text", "text": segment})
                message["content"] = content_list

            elif isinstance(message["content"], list):
                # TODO: handle more general cases. where message['content'] is a dict and contains multiple types of content.
                # e.g.
                #  "content": [
                #     {
                #         "type": "image",
                #         "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
                #     },
                #     {"type": "text", "text": "Describe this image."},
                # ],
                logger.warning("message['content'] is a list of dicts, no processing will be done.")
                continue
            else:
                raise ValueError(
                    f"Unsupported content type: {type(message['content'])}, expected str or list of dicts"
                )

    return messages


class Dataset:
    def __init__(
        self,
        path,
        tokenizer,
        processor,
        max_length,
        *,
        prompt_key="text",
        multimodal_keys=None,
        label_key=None,
        tool_key=None,
        metadata_key="metadata",
        seed=42,
        apply_chat_template=False,
        apply_chat_template_kwargs=None,
    ):
        self.origin_samples = []
        for data in read_file(path):
            prompt = _build_messages(data, prompt_key, multimodal_keys)

            metadata = data.get(metadata_key) or {}
            if tool_key is not None and tool_key in data:
                tools = data[tool_key]
                if isinstance(tools, str):
                    tools = json.loads(tools)
                elif isinstance(tools, np.ndarray):
                    tools = tools.tolist()
                assert isinstance(tools, list), f"tools must be a list, got {type(tools)} instead"
                metadata["tools"] = tools

            # TODO: this is slow.
            if _should_skip_prompt(prompt, tokenizer, processor, max_length, apply_chat_template_kwargs):
                continue

            self.origin_samples.append(
                Sample(
                    prompt=prompt,
                    label=data[label_key] if label_key is not None else None,
                    metadata=metadata,
                )
            )

        self.epoch_id = -1
        self.seed = seed
        self.samples = self.origin_samples

    def shuffle(self, new_epoch_id):
        if self.epoch_id == new_epoch_id:
            return

        random.seed(self.seed + new_epoch_id)
        permutation = list(range(len(self.samples)))
        random.shuffle(permutation)
        self.samples = [self.origin_samples[i] for i in permutation]
        self.epoch_id = new_epoch_id

    def __getitem__(self, idx):
        return self.samples[idx]

    def __len__(self):
        return len(self.samples)


def get_minimum_num_micro_batch_size(total_lengths, max_tokens_per_gpu):
    # use first fit to get the number of micro batches
    batches = []
    for length in total_lengths:
        for i in range(len(batches)):
            if batches[i] + length <= max_tokens_per_gpu:
                batches[i] += length
                break
        else:
            batches.append(length)

    return len(batches)


def process_rollout_data(args, rollout_data_ref, dp_rank, dp_size):
    assert len(rollout_data_ref) == dp_size
    rollout_data = ray.get(rollout_data_ref[dp_rank].inner)

    partition = rollout_data.pop("partition")
    total_lengths = rollout_data["total_lengths"]

    # save the seqlen of the whole rollout batch
    Timer().seq_lens = total_lengths
    rollout_data["total_lengths"] = [total_lengths[i] for i in partition]

    return rollout_data
# def process_rollout_data(args, rollout_data_ref, dp_rank, dp_size):
#     """
#     处理 rollout 数据，支持当 dp_size 小于数据分片数时合并多个分片。
    
#     Args:
#         args: 训练参数
#         rollout_data_ref: Ray object references 列表，包含分片后的数据
#         dp_rank: 当前进程的数据并行 rank
#         dp_size: 当前模型的数据并行大小
    
#     Returns:
#         合并后的 rollout_data 字典
#     """
#     num_data_shards = len(rollout_data_ref)
    
#     # 情况 1: dp_size 等于数据分片数（标准情况，Actor）
#     if dp_size == num_data_shards:
#         rollout_data = ray.get(rollout_data_ref[dp_rank].inner)
#         partition = rollout_data.pop("partition")
#         total_lengths = rollout_data["total_lengths"]
        
#         # save the seqlen of the whole rollout batch
#         Timer().seq_lens = total_lengths
#         rollout_data["total_lengths"] = [total_lengths[i] for i in partition]
        
#         return rollout_data
    
#     # 情况 2: dp_size 小于数据分片数（Critic 需要合并多个分片）
#     elif dp_size < num_data_shards:
#         # 验证数据分片数必须能被 dp_size 整除
#         if num_data_shards % dp_size != 0:
#             raise ValueError(
#                 f"数据分片数量 ({num_data_shards}) 必须能被 dp_size ({dp_size}) 整除。"
#                 f"当前 Actor 的数据并行度是 {num_data_shards}，"
#                 f"Critic 的数据并行度是 {dp_size}。"
#                 f"请确保 Actor GPU 数是 Critic GPU 数的整数倍。"
#             )
        
#         # 计算每个 rank 应该处理多少个数据分片
#         shards_per_rank = num_data_shards // dp_size
        
#         # 计算当前 rank 应该处理哪些数据分片
#         start_shard_idx = dp_rank * shards_per_rank
#         end_shard_idx = start_shard_idx + shards_per_rank
        
#         if dp_rank == 0:
#             logger.info(
#                 f"合并数据分片：dp_size={dp_size}, 数据分片数={num_data_shards}, "
#                 f"每个 rank 处理 {shards_per_rank} 个分片"
#             )
        
#         # 合并多个数据分片
#         merged_rollout_data = None
#         all_partitions = []
        
#         for shard_idx in range(start_shard_idx, end_shard_idx):
#             rollout_data = ray.get(rollout_data_ref[shard_idx].inner)
#             partition = rollout_data.pop("partition")
#             all_partitions.extend(partition)
            
#             if merged_rollout_data is None:
#                 # 第一个分片，直接使用
#                 merged_rollout_data = rollout_data
#             else:
#                 # 合并后续分片的列表类型数据
#                 for key in [
#                     "tokens",
#                     "multimodal_inputs",
#                     "response_lengths",
#                     "rewards",
#                     "truncated",
#                     "loss_masks",
#                     "round_number",
#                     "sample_indices",
#                     "rollout_log_probs",
#                     "rollout_routed_experts",
#                     "prompt",
#                     "teacher_log_probs",
#                     "positive_nll_mask",
#                 ]:
#                     if key in rollout_data and key in merged_rollout_data:
#                         merged_rollout_data[key].extend(rollout_data[key])
#                     elif key in rollout_data:
#                         # 第一个分片没有这个 key，但后续分片有
#                         merged_rollout_data[key] = rollout_data[key]
                
#                 # 对于非列表类型的数据（如 raw_reward, total_lengths），保持不变
#                 # 这些数据在所有分片中是相同的全局数据
        
#         total_lengths = merged_rollout_data["total_lengths"]
        
#         # save the seqlen of the whole rollout batch
#         Timer().seq_lens = total_lengths
#         merged_rollout_data["total_lengths"] = [total_lengths[i] for i in all_partitions]
        
#         return merged_rollout_data
    
#     # 情况 3: dp_size 大于数据分片数（不支持）
#     else:
#         raise ValueError(
#             f"不支持 dp_size ({dp_size}) 大于数据分片数 ({num_data_shards})。"
#             f"这意味着 Critic 的 GPU 数多于 Actor 的 GPU 数，"
#             f"当前框架不支持这种配置。请确保 Actor GPU 数 >= Critic GPU 数。"
#         )