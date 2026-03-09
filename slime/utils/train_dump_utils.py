import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def save_debug_train_data(args, *, rollout_id, rollout_data, role=None):
    """
    Save debug training data to disk.
    
    Args:
        args: Arguments containing save_debug_train_data path template
        rollout_id: Current rollout ID
        rollout_data: Training data to save
        role: Optional role name (e.g., "actor", "critic", "critic2") to distinguish saved files
    """
    if (path_template := args.save_debug_train_data) is not None:
        rank = torch.distributed.get_rank()
        
        # 如果提供了role，将其添加到文件名中
        if role is not None:
            # 修改path_template，在rank后面添加role
            # 原格式: {rollout_id}_{rank}.pt
            # 新格式: {rollout_id}_{rank}_{role}.pt
            path = Path(path_template.format(rollout_id=rollout_id, rank=rank, role=role))
        else:
            # 向后兼容：如果没有提供role，使用原格式
            path = Path(path_template.format(rollout_id=rollout_id, rank=rank))
        
        logger.info(f"Save debug train data to {path} (role={role})")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            dict(
                rollout_id=rollout_id,
                rank=rank,
                role=role,  # 将role也保存到文件中
                rollout_data=rollout_data,
            ),
            path,
        )
