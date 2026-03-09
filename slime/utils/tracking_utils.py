import logging
import wandb
from slime.utils.tensorboard_utils import _TensorboardAdapter

from . import wandb_utils

logger = logging.getLogger(__name__)


def init_tracking(args, primary: bool = True, **kwargs):
    if primary:
        wandb_utils.init_wandb_primary(args, **kwargs)
    else:
        wandb_utils.init_wandb_secondary(args, **kwargs)


# TODO further refactor, e.g. put TensorBoard init to the "init" part
def log(args, metrics, step_key: str):
    if args.use_wandb:
        # 检查 wandb 是否已初始化
        if wandb.run is not None:
            wandb.log(metrics)
        else:
            # wandb 未初始化，记录 debug 信息但不崩溃
            logger.debug(f"wandb.run is None, skipping wandb.log() for metrics: {list(metrics.keys())}")

    if args.use_tensorboard:
        metrics_except_step = {k: v for k, v in metrics.items() if k != step_key}
        _TensorboardAdapter(args).log(data=metrics_except_step, step=metrics[step_key])
