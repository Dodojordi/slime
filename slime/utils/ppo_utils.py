# Adapt from https://github.com/OpenRLHF/OpenRLHF/blob/10c733694ed9fbb78a0a2ff6a05efc7401584d46/openrlhf/models/utils.py
# and https://github.com/OpenRLHF/OpenRLHF/blob/10c733694ed9fbb78a0a2ff6a05efc7401584d46/openrlhf/trainer/ppo_utils/experience_maker.py

from argparse import Namespace
from typing import Callable, Optional
import torch
import torch.distributed as dist
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)
@torch.compile(dynamic=True)
def compute_approx_kl(
    log_probs: torch.Tensor,
    log_probs_base: torch.Tensor,
    kl_loss_type: str,
    importance_ratio: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Compute the approximate KL divergence between two distributions.
    Schulman blog: http://joschu.net/blog/kl-approx.html

    Args:
        log_probs: Log probabilities of the new distribution.
        log_probs_base: Log probabilities of the base distribution.
        kl_loss_type: Type of KL estimator (k1, k2, k3, low_var_kl).
        importance_ratio: Optional IS ratio (π_θ/π_old) for unbiased KL estimation.
    """
    log_ratio = log_probs.float() - log_probs_base.float()

    if kl_loss_type == "k1":
        kl = log_ratio
    elif kl_loss_type == "k2":
        kl = log_ratio**2 / 2.0
    elif kl_loss_type in ["k3", "low_var_kl"]:
        # The non negative kl approximation in
        # http://joschu.net/blog/kl-approx.html
        # Besides non negative, it is also unbiased and have lower variance.
        log_ratio = -log_ratio
        kl = log_ratio.exp() - 1 - log_ratio
    else:
        raise ValueError(f"Unknown kl_loss_type: {kl_loss_type}")

    # Apply IS ratio for unbiased KL estimation (DeepSeek-V3.2)
    if importance_ratio is not None:
        kl = importance_ratio * kl

    # Clamp only for low_var_kl for numerical stability
    if kl_loss_type == "low_var_kl":
        kl = torch.clamp(kl, min=-10, max=10)

    return kl


def compute_opsm_mask(
    args: Namespace,
    full_log_probs: list[torch.Tensor],
    full_old_log_probs: list[torch.Tensor],
    advantages: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute Off-Policy Sequence Masking (OPSM) mask.

    Args:
        args: Configuration containing `opsm_delta` threshold.
        full_log_probs: Current policy log-probs per sample.
        full_old_log_probs: Old policy log-probs per sample.
        advantages: Advantage values per sample.
        loss_masks: Loss masks per sample.

    Returns:
        Tuple of `(opsm_mask, opsm_clipfrac)` where `opsm_mask` is a
        concatenated tensor of per-token masks and
        `opsm_clipfrac` is the count of masked sequences.
    """
    opsm_mask_list = []
    device = advantages[0].device
    opsm_clipfrac = torch.tensor(0.0, device=device)

    for full_log_prob, full_old_log_prob, advantage, loss_mask in zip(
        full_log_probs, full_old_log_probs, advantages, loss_masks, strict=False
    ):
        # Calculate sequence-level KL
        seq_kl = ((full_old_log_prob - full_log_prob) * loss_mask).sum() / torch.clamp_min(loss_mask.sum(), 1)

        # Create mask: 0 if (advantage < 0 and seq_kl > delta), else 1
        mask = ((advantage < 0) & (seq_kl > args.opsm_delta)).float()
        opsm_clipfrac += mask.sum() / torch.clamp_min(loss_mask.sum(), 1)

        opsm_mask_list.append(1 - mask)

    opsm_mask = torch.cat(opsm_mask_list, dim=0)
    return opsm_mask, opsm_clipfrac


def compute_gspo_kl(
    full_log_probs: list[torch.Tensor],
    full_old_log_probs: list[torch.Tensor],
    local_log_probs: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
) -> torch.Tensor:
    """Compute GSPO-style per-sequence KL divergence.

    Args:
        full_log_probs: Current policy log-probs per sample (full or CP-local).
        full_old_log_probs: Old policy log-probs per sample (full or CP-local).
        local_log_probs: Local (CP-local) log-probs for expansion shape reference.
        loss_masks: Loss masks per sample.

    Returns:
        Concatenated tensor of per-token KL values where each token in a
        sequence has the same KL value (the sequence-level KL).
    """
    # Compute sequence-level KL and expand to per-token
    ppo_kl = [
        ((old_logprob - log_prob) * loss_mask).sum() / torch.clamp_min(loss_mask.sum(), 1)
        for log_prob, old_logprob, loss_mask in zip(full_log_probs, full_old_log_probs, loss_masks, strict=False)
    ]
    ppo_kl = [kl.expand_as(log_prob) for kl, log_prob in zip(ppo_kl, local_log_probs, strict=False)]
    ppo_kl = torch.cat(ppo_kl, dim=0)

    return ppo_kl


@torch.compile(dynamic=True)
def compute_policy_loss(
    ppo_kl: torch.Tensor,
    advantages: torch.Tensor,
    eps_clip: float,
    eps_clip_high: float,
    eps_clip_c: float | None = None,
):
    ratio = (-ppo_kl).exp()
    pg_losses1 = -ratio * advantages
    pg_losses2 = -ratio.clamp(1 - eps_clip, 1 + eps_clip_high) * advantages
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)
    clipfrac = torch.gt(pg_losses2, pg_losses1).float()

    if eps_clip_c is not None:
        assert (
            eps_clip_c > 1.0
        ), f"The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0, but get the value: {eps_clip_c}."
        pg_losses3 = -eps_clip_c * advantages
        clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
        pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
    else:
        pg_losses = clip_pg_losses1

    return pg_losses, clipfrac


def compute_log_probs(logits: torch.Tensor, tokens: torch.Tensor, process_group: dist.ProcessGroup | None):
    from megatron.core.fusions.fused_cross_entropy import fused_vocab_parallel_cross_entropy

    # convert to [seq_len, batch_size, vocab_size] as expected by fused_vocab_parallel_cross_entropy
    logits = logits.unsqueeze(1)
    tokens = tokens.unsqueeze(1)
    return -fused_vocab_parallel_cross_entropy(logits, tokens, process_group)


# from https://github.com/volcengine/verl/blob/0bdf7f469854815177e73dcfe9e420836c952e6e/verl/utils/megatron/tensor_parallel.py#L99
class _VocabParallelEntropy(torch.autograd.Function):

    @staticmethod
    def forward(ctx, vocab_parallel_logits: torch.Tensor, process_group: dist.ProcessGroup) -> torch.Tensor:

        @torch.compile(dynamic=True)
        def mul_reduce(a, b):
            return (a * b).sum(dim=-1, keepdim=True)

        logits_max = vocab_parallel_logits.max(dim=-1, keepdim=True).values
        dist.all_reduce(logits_max, op=dist.ReduceOp.MAX, group=process_group)
        normalized_vocab_parallel_logits = vocab_parallel_logits - logits_max
        normalized_exp_logits = normalized_vocab_parallel_logits.exp_()
        normalized_sum_exp_logits = normalized_exp_logits.sum(dim=-1, keepdim=True)
        dist.all_reduce(normalized_sum_exp_logits, group=process_group)
        softmax_logits = normalized_exp_logits.div_(normalized_sum_exp_logits)
        sum_softmax_times_logits = mul_reduce(softmax_logits, vocab_parallel_logits)
        dist.all_reduce(sum_softmax_times_logits, group=process_group)
        entropy = logits_max + normalized_sum_exp_logits.log() - sum_softmax_times_logits
        ctx.save_for_backward(vocab_parallel_logits, softmax_logits, sum_softmax_times_logits)
        return entropy.squeeze(dim=-1)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        vocab_parallel_logits, softmax_logits, sum_softmax_times_logits = ctx.saved_tensors
        # reuse softmax_logits as grad
        vocab_parallel_logits.sub_(sum_softmax_times_logits)
        softmax_logits.mul_(vocab_parallel_logits)
        softmax_logits.mul_(grad_output.unsqueeze(dim=-1))
        # recover vocab_parallel_logits
        vocab_parallel_logits.add_(sum_softmax_times_logits)
        softmax_logits.mul_(-1)
        return softmax_logits, None


def _compute_entropy_no_autograd(
    vocab_parallel_logits: torch.Tensor, 
    process_group: dist.ProcessGroup
) -> torch.Tensor:
    """
    纯 forward 的 entropy 计算，不使用 autograd.Function。
    
    不构建计算图，不保存中间张量，仅用于监控。
    当 entropy_coef=0 时调用此函数，可节省约 29 GB 显存（64k 序列）。
    """
    # 和 _VocabParallelEntropy.forward 相同的数学逻辑，但不调用 ctx.save_for_backward
    logits_max = vocab_parallel_logits.max(dim=-1, keepdim=True).values
    dist.all_reduce(logits_max, op=dist.ReduceOp.MAX, group=process_group)
    
    normalized = vocab_parallel_logits - logits_max
    exp_logits = normalized.exp()
    
    sum_exp = exp_logits.sum(dim=-1, keepdim=True)
    dist.all_reduce(sum_exp, group=process_group)
    
    softmax_logits = exp_logits / sum_exp
    sum_softmax_logits = (softmax_logits * vocab_parallel_logits).sum(dim=-1, keepdim=True)
    dist.all_reduce(sum_softmax_logits, group=process_group)
    
    entropy = logits_max + sum_exp.log() - sum_softmax_logits
    return entropy.squeeze(dim=-1)


def compute_entropy_from_logits(logits: torch.Tensor, process_group) -> torch.Tensor:
    return _VocabParallelEntropy.apply(logits, process_group)


def get_grpo_returns(
    rewards: torch.Tensor,
    kl: list[torch.Tensor],
):
    returns = []
    for i in range(len(rewards)):
        returns.append(torch.ones_like(kl[i]) * rewards[i])
    return returns


def get_reinforce_plus_plus_returns(
    rewards: torch.Tensor,
    kl: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
    response_lengths: list[int],
    total_lengths: list[int],
    kl_coef: float,
    gamma: float,
) -> list[torch.Tensor]:
    """
    Calculates discounted returns for REINFORCE++ (https://arxiv.org/pdf/2501.03262)

    Args:
        rewards (Tensor): A tensor of scalar rewards for each sequence.
        kl (List[Tensor]): List of per-token KL divergence tensors for sequence chunks.
        loss_masks (List[Tensor]): List of response-only loss masks for each full sequence.
        response_lengths (List[int]): The full length of each response sequence.
        total_lengths (List[int]): The full length of each sequence (prompt + response).
        kl_coef (float): Coefficient for the KL penalty.
        gamma (float): The discount factor.

    Returns:
        List[torch.Tensor]: A list of return (G_t) tensors for the
                            local sequence chunks owned by the current GPU rank.
    """
    from megatron.core import mpu

    cp_size = mpu.get_context_parallel_world_size()

    final_returns_chunks = []
    for i in range(len(rewards)):
        local_kl_chunk = kl[i]
        total_len, response_len = total_lengths[i], response_lengths[i]

        if cp_size > 1:
            # Step 1,2:Gather all chunks and token_offsets from all ranks and reconstruct the full response tensor by splitting and placing each part
            from slime.backends.megatron_utils.cp_utils import all_gather_with_cp

            full_kl_response = all_gather_with_cp(local_kl_chunk, total_len, response_len)
        else:
            full_kl_response = local_kl_chunk

        # Step 3: Compute returns on full response kl tensor.
        token_level_rewards = -kl_coef * full_kl_response
        full_mask = loss_masks[i]
        assert full_mask.sum().item() > 0, f"Sequence at index {i} is fully masked."
        last_idx = full_mask.nonzero(as_tuple=True)[0][-1]
        token_level_rewards[last_idx] += rewards[i]

        returns_for_seq = torch.zeros_like(token_level_rewards)
        running_return = 0.0
        for t in reversed(range(token_level_rewards.size(0))):
            # G_t = r_t + gamma * G_{t+1}
            running_return = token_level_rewards[t] + gamma * running_return
            returns_for_seq[t] = running_return

        # Step 4: Pick up the results corresponding to our local chunk's parts.
        if cp_size > 1:
            from slime.backends.megatron_utils.cp_utils import slice_log_prob_with_cp

            local_returns_chunk = slice_log_prob_with_cp(returns_for_seq, total_len, response_len)
        else:
            local_returns_chunk = returns_for_seq

        final_returns_chunks.append(local_returns_chunk)

    return final_returns_chunks


def get_reinforce_plus_plus_baseline_advantages(
    rewards: torch.Tensor,
    kl: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
    kl_coef: float,
) -> list[torch.Tensor]:
    """
    Calculates the unwhitened advantages for the REINFORCE++-baseline algorithm.
    Broadcasting the scalar (reward - group_baseline) to each token.

    Args:
        rewards (Tensor): A tensor of scalar rewards, where the group-wise
                                baseline has already been subtracted.
        kl (list[Tensor]): A list of per-token KL divergence tensors. Used to
                                 get the shape for broadcasting.
        loss_masks (list[Tensor]): A list of per-token loss masks.
        kl_coef (float): Coefficient for the KL penalty.

    Returns:
        list[Tensor]: A list of tensors containing the unwhitened advantages.
    """
    # Broadcast to get unwhitened advantages
    unwhitened_advantages = [
        torch.ones_like(kl_tensor) * reward_val - kl_coef * kl_tensor
        for kl_tensor, reward_val in zip(kl, rewards, strict=False)
    ]

    return unwhitened_advantages


def get_advantages_and_returns(
    total_len: int,
    response_len: int,
    values: torch.Tensor,
    rewards: torch.Tensor,
    gamma: float,
    # lambd: float,
    lambd_actor: float,
    lambd_critic: float,
    alpha: float = 0.05,
    use_adaptive_lambda: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Function that computes advantages and returns from rewards and values.
    Calculated as in the original PPO paper: https://arxiv.org/abs/1707.06347
    Note that rewards may include a KL divergence loss term.

    Advantages looks like this:
    Adv1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
            - V1 + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

    Returns looks like this:
    Ret1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
                + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

    Input:
    - values: Tensor of shape (response_size,)
    - rewards: Tensor of shape (response_size,)

    Output:
    - advantages: Tensor of shape (response_size,)
    - returns: Tensor of shape (response_size,)
    """
    
    if use_adaptive_lambda:
        lambd_actor = compute_adaptive_lambda_policy(response_len, alpha)
        
    from megatron.core import mpu

    cp_size = mpu.get_context_parallel_world_size()
    if cp_size > 1:
        from slime.backends.megatron_utils.cp_utils import all_gather_with_cp

        full_rewards = all_gather_with_cp(rewards, total_len, response_len)
        full_values = all_gather_with_cp(values, total_len, response_len)
    else:
        full_rewards = rewards
        full_values = values

# lastgaelam = 0
    # 分别计算actor和critic的GAE
    lastgaelam_actor = 0
    lastgaelam_critic = 0
    advantages_reversed = []
    returns_reversed = []
    

    for t in reversed(range(response_len)):
        nextvalues = full_values[t + 1] if t < response_len - 1 else 0.0
        delta = full_rewards[t] + gamma * nextvalues - full_values[t]
        # lastgaelam = delta + gamma * lambd * lastgaelam
        # advantages_reversed.append(lastgaelam)
        # Actor的advantage计算
        lastgaelam_actor = delta + gamma * lambd_actor * lastgaelam_actor
        advantages_reversed.append(lastgaelam_actor)
        # Critic的advantage计算
        lastgaelam_critic = delta + gamma * lambd_critic * lastgaelam_critic
        returns_reversed.append(lastgaelam_critic+full_values[t])
        
        
    full_advantages = torch.tensor(advantages_reversed[::-1], dtype=full_values.dtype, device=full_values.device)
    # full_returns = full_advantages + full_values
    full_returns = torch.tensor(returns_reversed[::-1], dtype=full_values.dtype, device=full_values.device)

    if cp_size > 1:
        from slime.backends.megatron_utils.cp_utils import slice_log_prob_with_cp

        advantages = slice_log_prob_with_cp(full_advantages, total_len, response_len)
        returns = slice_log_prob_with_cp(full_returns, total_len, response_len)
    else:
        advantages = full_advantages
        returns = full_returns

    return advantages.detach(), returns

def compute_adaptive_lambda_policy(
    response_lengths: int | list[int] | torch.Tensor,
    alpha: float,
) -> float | list[float] | torch.Tensor:
    """
    计算长度自适应的lambda_policy值。
    
    公式: λ_policy = 1 - 1/(alpha * l)
    其中 l 是response的长度，alpha是常数参数。
    
    Args:
        response_lengths: Response序列的长度，可以是单个int、int列表或Tensor
        alpha: 常数参数，必须大于0
    
    Returns:
        计算得到的lambda_policy值，类型与输入response_lengths相同
        
    Examples:
        >>> compute_adaptive_lambda_policy(100, alpha=2.0)
        0.995  # 1 - 1/(2.0 * 100) = 1 - 0.005 = 0.995
        
        >>> compute_adaptive_lambda_policy([50, 100, 200], alpha=1.0)
        [0.98, 0.99, 0.995]
    """
    eps = 1e-6  # 增加一个小的epsilon防止除0
    if alpha <= 0:
        raise ValueError(f"alpha must be greater than 0, got {alpha}")
    
    def clamp_lambda(lam):
        # 保证最终的lambda在[0.95,1]范围内
        return max(0.95, min(1.0, lam))
    
    if isinstance(response_lengths, int):
        # 单个长度
        if response_lengths <= 0:
            raise ValueError(f"response_length must be greater than 0, got {response_lengths}")
        lam = 1.0 - 1.0 / (alpha * max(response_lengths, eps))
        return clamp_lambda(lam)
    
    elif isinstance(response_lengths, list):
        # 列表
        result = []
        for l in response_lengths:
            if l > 0:
                lam = 1.0 - 1.0 / (alpha * max(l, eps))
                lam = clamp_lambda(lam)
            else:
                lam = 0.0
            result.append(lam)
        return result
    
    elif isinstance(response_lengths, torch.Tensor):
        # Tensor
        if response_lengths.dtype != torch.int64 and response_lengths.dtype != torch.int32:
            response_lengths = response_lengths.to(torch.int64)
        length_f = response_lengths.float().clamp(min=eps)
        result = 1.0 - 1.0 / (alpha * length_f)
        result = torch.where(response_lengths > 0, result, torch.zeros_like(result))
        # 保证范围在[0.95,1]
        result = torch.clamp(result, min=0.95, max=1.0)
        return result
    
    else:
        raise TypeError(f"Unsupported type for response_lengths: {type(response_lengths)}")

def get_advantages_and_returns_batch(
    total_lengths,
    response_lengths,
    values_list,
    rewards_list,
    gamma,
    lambd_actor,
    lambd_critic,
    use_adaptive_lambda: bool = False,
    alpha: float = 0.05,
    chunked: bool = True,
):
    """
    Batched GAE with CP support.
    Input:
        total_lengths:     list[int], each sample's total_len
        response_lengths:  list[int], each sample's response_len
        values_list:       list[Tensor], each shape = [resp_len_i]
        rewards_list:      list[Tensor], same shape
        use_adaptive_lambda: If True, compute adaptive lambda for each sample
        alpha: Constant parameter for adaptive lambda
    Output:
        advantages_list:   list[Tensor], each shape = [resp_len_i]
        returns_list:      list[Tensor], same shape
    """

    from megatron.core import mpu

    with torch.no_grad():
        B = len(response_lengths)
        assert B == len(values_list)
        assert B == len(rewards_list)

        # 如果启用自适应lambda，为每个样本计算lambda值
        if use_adaptive_lambda:
            lambd_actor_list = compute_adaptive_lambda_policy(response_lengths, alpha)
            # Fix: Make lambd_critic_list a Python list (matching lambd_actor_list's structure)
            lambd_critic_list = [lambd_critic] * len(lambd_actor_list)
            logger.info(f"Adaptive lambda (lambd_actor) computed: {lambd_actor_list}")
            logger.info(f"Adaptive lambda (lambd_critic) computed: {lambd_critic_list}")
            
            # 检查是否所有样本的lambda都相同
            if len(set(lambd_actor_list)) > 1 or len(set(lambd_critic_list)) > 1:
                # 每个样本的lambda不同，需要逐个处理，使用 get_advantages_and_returns
                advantages_list = []
                returns_list = []
                
                for i in range(B):
                    total_len = total_lengths[i]
                    resp_len = response_lengths[i]
                    v = values_list[i]
                    r = rewards_list[i]
                    lambd_a = lambd_actor_list[i]
                    lambd_c = lambd_critic_list[i]
                    
                    # 调用单个样本版本
                    adv, ret = get_advantages_and_returns(
                        total_len=total_len,
                        response_len=resp_len,
                        values=v,
                        rewards=r,
                        gamma=gamma,
                        lambd_actor=lambd_a,
                        lambd_critic=lambd_c,
                        use_adaptive_lambda=False,  # 已经计算过了
                        alpha=alpha,  # 虽然不会用到，但保持接口一致
                    )
                    advantages_list.append(adv)
                    returns_list.append(ret)
                
                return advantages_list, returns_list
            else:
                # 所有样本的lambda相同，可以使用批量处理
                lambd_actor = lambd_actor_list[0]
                lambd_critic = lambd_critic_list[0]
                logger.info(f"Adaptive lambda (lambd_actor) computed: {lambd_actor}")
                logger.info(f"Adaptive lambda (lambd_critic) computed: {lambd_critic}")

        cp_size = mpu.get_context_parallel_world_size()
        logger.info(f"cp_size: {cp_size}")
        # logger.info(f"values_list: {values_list}")
        # logger.info(f"rewards_list: {rewards_list}")
        # logger.info(f"total_lengths: {total_lengths}")
        # logger.info(f"response_lengths: {response_lengths}")
        # logger.info(f"chunked: {chunked}")
        device = values_list[0].device
        dtype = values_list[0].dtype

        if cp_size > 1:
            from slime.backends.megatron_utils.cp_utils import all_gather_with_cp

            full_values_list = []
            full_rewards_list = []

            for total_len, resp_len, v, r in zip(
                total_lengths, response_lengths, values_list, rewards_list, strict=False
            ):
                full_v = all_gather_with_cp(v, total_len, resp_len)
                full_r = all_gather_with_cp(r, total_len, resp_len)
                full_values_list.append(full_v)
                full_rewards_list.append(full_r)

        else:
            full_values_list = values_list
            full_rewards_list = rewards_list

        # pad to max_len for batched GAE
        max_len = max(response_lengths)

        full_values = torch.zeros(B, max_len, device=device, dtype=dtype)
        full_rewards = torch.zeros(B, max_len, device=device, dtype=dtype)

        for i in range(B):
            L = response_lengths[i]
            full_values[i, :L] = full_values_list[i][:L]
            full_rewards[i, :L] = full_rewards_list[i][:L]

        if not chunked:
            full_advantages, full_returns = vanilla_gae(
                rewards=full_rewards,
                values=full_values,
                gamma=gamma,
                lambd_actor=lambd_actor,
                lambd_critic=lambd_critic,
            )
        else:
            full_advantages, full_returns = chunked_gae_dual_lambda(
                rewards=full_rewards,
                values=full_values,
                gamma=gamma,
                lambd_actor=lambd_actor,
                lambd_critic=lambd_critic,
            )

        advantages_list = []
        returns_list = []

        if cp_size > 1:
            from slime.backends.megatron_utils.cp_utils import slice_log_prob_with_cp

            for total_len, resp_len, adv_row, ret_row in zip(
                total_lengths,
                response_lengths,
                full_advantages,
                full_returns,
                strict=False,
            ):
                adv_full = adv_row
                ret_full = ret_row

                adv_sliced = slice_log_prob_with_cp(adv_full[:resp_len], total_len, resp_len)
                ret_sliced = slice_log_prob_with_cp(ret_full[:resp_len], total_len, resp_len)

                advantages_list.append(adv_sliced)
                returns_list.append(ret_sliced)

        else:
            for i in range(B):
                L = response_lengths[i]
                advantages_list.append(full_advantages[i, :L])
                returns_list.append(full_returns[i, :L])

    return advantages_list, returns_list


def vanilla_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    gamma: float,
    # lambd: float,
    lambd_actor: float,
    lambd_critic: float,
):
    B, T = rewards.shape
    device = rewards.device
    dtype = rewards.dtype

    # lastgaelam = torch.zeros(B, device=device, dtype=dtype)
    lastgaelam_actor = torch.zeros(B, device=device, dtype=dtype)
    lastgaelam_critic = torch.zeros(B, device=device, dtype=dtype)
    adv_rev = []
    ret_rev = []

    for t in reversed(range(T)):
        next_value = values[:, t + 1] if t < T - 1 else 0.0
        delta = rewards[:, t] + gamma * next_value - values[:, t]
        # lastgaelam = delta + gamma * lambd * lastgaelam
        # Actor advantages
        lastgaelam_actor = delta + gamma * lambd_actor * lastgaelam_actor
        adv_rev.append(lastgaelam_actor)
        # Critic returns
        lastgaelam_critic = delta + gamma * lambd_critic * lastgaelam_critic
        ret_rev.append(lastgaelam_critic+values[:, t])

    full_advantages = torch.stack(adv_rev[::-1], dim=1)  # [B, max_len]
    # full_returns = full_advantages + values  # [B, max_len]
    full_returns = torch.stack(ret_rev[::-1], dim=1)  # [B, max_len]
    return full_advantages, full_returns


def chunked_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    gamma: float,
    lambd: float,
    chunk_size: int = 128,
):
    """
    Compute Generalized Advantage Estimation (GAE) using a FlashLinearAttention-
    inspired algorithm: parallel prefix scan within chunks and recurrent state
    propagation across chunks.

    This reduces the sequential dependency length from O(T) to O(T / chunk_size),
    while keeping chunk computations fully parallelizable (O(C^2) per chunk).

    Args:
        rewards (Tensor): [B, T] reward sequence.
        values (Tensor):  [B, T] value predictions. The next-value of the final
                          step is assumed to be zero (standard PPO convention).
        gamma (float): discount factor.
        lam (float): GAE lambda.
        chunk_size (int): sequence chunk length for parallel scan.

    Returns:
        advantages (Tensor): [B, T] computed advantages.
        returns (Tensor):    [B, T] advantages + values.
    """

    # -------------------------------------------------------------------------
    # Validate inputs
    # -------------------------------------------------------------------------
    assert rewards.ndim == 2 and values.ndim == 2
    B, T = rewards.shape
    assert values.shape == (B, T)

    device = rewards.device
    dtype = rewards.dtype

    # -------------------------------------------------------------------------
    # Build δ_t = r_t + γ * V_{t+1} - V_t   with V_{T} = 0
    # -------------------------------------------------------------------------
    next_values = torch.cat(
        [values[:, 1:], torch.zeros(B, 1, device=device, dtype=dtype)],
        dim=1,
    )
    deltas = rewards + gamma * next_values - values

    # Reformulate backward GAE as a forward scan on the reversed sequence:
    #   S[i] = Δ[i] + w * S[i - 1],   w = γλ
    w = gamma * lambd
    deltas_rev = torch.flip(deltas, dims=[1])  # [B, T]

    # -------------------------------------------------------------------------
    # Pad to a multiple of chunk_size
    # -------------------------------------------------------------------------
    if T % chunk_size != 0:
        pad = chunk_size - (T % chunk_size)
        deltas_rev = F.pad(deltas_rev, (0, pad))
    else:
        pad = 0

    B, T_pad = deltas_rev.shape
    n_chunks = T_pad // chunk_size

    deltas_chunks = deltas_rev.view(B, n_chunks, chunk_size)

    # -------------------------------------------------------------------------
    # Construct the intra-chunk parallel scan kernel M
    #
    # For a chunk Δ[0..C-1], we want:
    #   S_local[t] = sum_{k=0..t} w^(t-k) * Δ[k]
    #
    # This is implemented as:
    #   S_local = Δ @ M
    #
    # where:
    #   M[i, j] = w^(j - i)    if j >= i
    #             0            otherwise
    # -------------------------------------------------------------------------
    idx = torch.arange(chunk_size, device=device)
    row = idx[:, None]
    col = idx[None, :]
    diff = col - row

    M = torch.zeros(chunk_size, chunk_size, device=device, dtype=dtype)
    mask = diff >= 0

    if w == 0.0:
        M[mask & (diff == 0)] = 1.0
    else:
        M[mask] = w ** diff[mask].to(dtype)

    # pow_vec[t] = w^(t+1), used to inject the recurrent state s_prev
    if w == 0.0:
        pow_vec = torch.zeros(chunk_size, device=device, dtype=dtype)
    else:
        pow_vec = w ** torch.arange(1, chunk_size + 1, device=device, dtype=dtype)

    # -------------------------------------------------------------------------
    # Parallel compute local chunk results (assuming initial state = 0)
    # -------------------------------------------------------------------------
    deltas_flat = deltas_chunks.reshape(B * n_chunks, chunk_size)
    S_local_flat = deltas_flat @ M
    S_local_chunks = S_local_flat.view(B, n_chunks, chunk_size)

    # Effective length of each chunk (the last chunk may be padded)
    lengths = [chunk_size] * n_chunks
    if pad > 0:
        lengths[-1] = chunk_size - pad

    # -------------------------------------------------------------------------
    # Recurrent propagation between chunks
    #
    # Each chunk contributes:
    #   S_global[t] = S_local[t] + w^(t+1) * s_prev
    #
    # And updates:
    #   s_prev = S_global[last_t]
    # -------------------------------------------------------------------------
    S_rev = deltas_rev.new_zeros(B, T_pad)
    s_prev = torch.zeros(B, device=device, dtype=dtype)

    for c in range(n_chunks):
        Lc = lengths[c]
        start = c * chunk_size
        end = start + Lc

        S_local = S_local_chunks[:, c, :Lc]
        S_global = S_local + s_prev.unsqueeze(1) * pow_vec[:Lc]

        S_rev[:, start:end] = S_global
        s_prev = S_global[:, -1]  # state for next chunk

    # Remove padding and flip back to original time order
    if pad > 0:
        S_rev = S_rev[:, :T]

    advantages = torch.flip(S_rev, dims=[1])
    returns = advantages + values

    return advantages, returns

def chunked_gae_dual_lambda(
    rewards: torch.Tensor,
    values: torch.Tensor,
    gamma: float,
    lambd_actor: float,
    lambd_critic: float,
    chunk_size: int = 128,
):
    """
    使用分块并行计算的GAE，支持actor和critic使用不同的lambda
    """
    # 分别对actor和critic执行chunked_gae
    advantages = chunked_gae_single_lambda(
        rewards, values, gamma, lambd_actor, chunk_size
    )
    
    returns_gae = chunked_gae_single_lambda(
        rewards, values, gamma, lambd_critic, chunk_size
    )
    returns = returns_gae + values
    
    return advantages, returns


def chunked_gae_single_lambda(
    rewards: torch.Tensor,
    values: torch.Tensor,
    gamma: float,
    lambd: float,
    chunk_size: int = 128,
):
    """
    单个lambda的chunked GAE计算（原chunked_gae的核心逻辑）
    返回advantages（不包含values）
    """
    # 原来的chunked_gae实现，但只返回advantages部分
    # ... 保持原有的chunk计算逻辑 ...
    
    assert rewards.ndim == 2 and values.ndim == 2
    B, T = rewards.shape
    device = rewards.device
    dtype = rewards.dtype

    next_values = torch.cat(
        [values[:, 1:], torch.zeros(B, 1, device=device, dtype=dtype)],
        dim=1,
    )
    deltas = rewards + gamma * next_values - values
    w = gamma * lambd
    deltas_rev = torch.flip(deltas, dims=[1])

    # ... 原有的padding和chunk逻辑 ...
    if T % chunk_size != 0:
        pad = chunk_size - (T % chunk_size)
        deltas_rev = F.pad(deltas_rev, (0, pad))
    else:
        pad = 0

    B, T_pad = deltas_rev.shape
    n_chunks = T_pad // chunk_size
    deltas_chunks = deltas_rev.view(B, n_chunks, chunk_size)

    idx = torch.arange(chunk_size, device=device)
    row = idx[:, None]
    col = idx[None, :]
    diff = col - row

    M = torch.zeros(chunk_size, chunk_size, device=device, dtype=dtype)
    mask = diff >= 0

    if w == 0.0:
        M[mask & (diff == 0)] = 1.0
    else:
        M[mask] = w ** diff[mask].to(dtype)

    if w == 0.0:
        pow_vec = torch.zeros(chunk_size, device=device, dtype=dtype)
    else:
        pow_vec = w ** torch.arange(1, chunk_size + 1, device=device, dtype=dtype)

    deltas_flat = deltas_chunks.reshape(B * n_chunks, chunk_size)
    S_local_flat = deltas_flat @ M
    S_local_chunks = S_local_flat.view(B, n_chunks, chunk_size)

    lengths = [chunk_size] * n_chunks
    if pad > 0:
        lengths[-1] = chunk_size - pad

    S_rev = deltas_rev.new_zeros(B, T_pad)
    s_prev = torch.zeros(B, device=device, dtype=dtype)

    for c in range(n_chunks):
        Lc = lengths[c]
        start = c * chunk_size
        end = start + Lc

        S_local = S_local_chunks[:, c, :Lc]
        S_global = S_local + s_prev.unsqueeze(1) * pow_vec[:Lc]

        S_rev[:, start:end] = S_global
        s_prev = S_global[:, -1]

    if pad > 0:
        S_rev = S_rev[:, :T]

    advantages = torch.flip(S_rev, dims=[1])
    
    return advantages

# def calculate_log_probs_and_entropy(logits, tokens, tp_group, with_entropy: bool = False):
#     logits = logits.contiguous()
#     # TODO: not sure why we need to clone the logits here.
#     # Without the clone, the backward will trigger inplace edit error.
#     # It seems that the function with tp will modify the logits inplace.
#     if logits.size(0) != 0:
#         log_prob = compute_log_probs(logits.clone(), tokens, tp_group)
#     else:
#         log_prob = logits.new_zeros((0,))

#     if with_entropy:
#         if logits.size(0) != 0:
#             entropy = compute_entropy_from_logits(logits.clone(), tp_group)
#         else:
#             entropy = logits.new_zeros((0,))
#     else:
#         entropy = None
#     return log_prob, entropy
def calculate_log_probs_and_entropy(
    logits, 
    tokens, 
    tp_group, 
    with_entropy: bool = False, 
    chunk_size: int = -1,
    requires_entropy_grad: bool = True  # 新增参数
):
    """计算 log probs 和 entropy，支持条件梯度追踪。
    
    Args:
        logits: 模型输出的 logits
        tokens: 目标 tokens
        tp_group: Tensor Parallel 通信组
        with_entropy: 是否计算 entropy
        chunk_size: 分块大小，-1 表示不分块
        requires_entropy_grad: entropy 是否需要梯度（当 entropy_coef=0 时设为 False）
    
    Returns:
        log_prob: Log probabilities
        entropy: Entropy 值（如果 with_entropy=True）
    """
    import contextlib  # 确保导入
    
    logits = logits.contiguous()
    entropy = None
    
    if logits.size(0) != 0:
        # === 计算 log_probs（总是需要梯度）===
        if chunk_size > 0:
            # 分块计算 log_probs
            num_chunks = (logits.size(0) - 1) // chunk_size + 1
            tokens_chunks = tokens.chunk(num_chunks, dim=0)
            logits_chunks = logits.chunk(num_chunks, dim=0)
            
            log_probs = []
            for tokens_chunk, logits_chunk in zip(tokens_chunks, logits_chunks, strict=True):
                log_prob = compute_log_probs(logits_chunk.clone(), tokens_chunk, tp_group)
                log_probs.append(log_prob)
            log_prob = torch.cat(log_probs, dim=0)
        else:
            # 不分块
            log_prob = compute_log_probs(logits.clone(), tokens, tp_group)
        
        # === 计算 entropy（条件梯度追踪）===
        # if with_entropy:
        #     # 🔑 关键修改：根据 requires_entropy_grad 选择上下文
        #     entropy_context = torch.no_grad() if not requires_entropy_grad else contextlib.nullcontext()
            
        #     with entropy_context:
        #         if chunk_size > 0:
        #             # 分块计算 entropy
        #             entropys = []
        #             for _, logits_chunk in zip(tokens_chunks, logits_chunks, strict=True):
        #                 e = compute_entropy_from_logits(logits_chunk.clone(), tp_group)
        #                 entropys.append(e)
        #             entropy = torch.cat(entropys, dim=0)
        #         else:
        #             # 不分块
        #             entropy = compute_entropy_from_logits(logits.clone(), tp_group)
        if with_entropy:
            if chunk_size > 0:
                # 分块计算 entropy
                entropys = []
                for _, logits_chunk in zip(tokens_chunks, logits_chunks, strict=True):
                    if requires_entropy_grad:
                        # 需要梯度：使用 autograd.Function
                        e = compute_entropy_from_logits(logits_chunk.clone(), tp_group)
                    else:
                        # 不需要梯度：使用纯 forward 函数，不保存中间张量
                        with torch.no_grad():
                            e = _compute_entropy_no_autograd(logits_chunk.clone(), tp_group)
                    entropys.append(e)
                entropy = torch.cat(entropys, dim=0)
            else:
                # 不分块
                if requires_entropy_grad:
                    entropy = compute_entropy_from_logits(logits.clone(), tp_group)
                else:
                    with torch.no_grad():
                        entropy = _compute_entropy_no_autograd(logits.clone(), tp_group)
    else:
        # 空序列处理
        log_prob = logits.new_zeros((0,))
        if with_entropy:
            entropy = logits.new_zeros((0,))
    
    return log_prob, entropy


def compute_positive_nll_loss(
    log_probs: torch.Tensor,
    positive_nll_mask: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
    response_lengths: list[int],
    positive_reward_threshold: float = 0.0,  # 保留参数以保持兼容性，但不再使用
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute Negative Log-Likelihood (NLL) loss for positive samples (correct answers).
    
    Formula: LNLL(θ) = -1/P * Σ(oi∈T) Σ(t=1 to |oi|) log πθ(at|st)
    where T is the set of correct answers, P is the total number of tokens in positive samples.
    
    Args:
        log_probs: Concatenated log probabilities of all samples, shape [total_tokens]
        positive_nll_mask: List of masks indicating which tokens should be used for NLL loss computation.
                          Each mask is a tensor of shape [response_length] with 1 for positive tokens, 0 otherwise.
        loss_masks: List of loss masks for each sample (used for validation, should match positive_nll_mask shape)
        response_lengths: List of response lengths for each sample
        positive_reward_threshold: Threshold to determine positive samples (deprecated, kept for compatibility)
        sum_of_sample_mean: Function to compute mean over samples (unused, kept for API consistency)
        
    Returns:
        Scalar tensor representing the NLL loss for positive samples
    """
    device = log_probs.device
    dtype = log_probs.dtype
    
    # Split log_probs back into per-sample tensors
    log_probs_per_sample = []
    start_idx = 0
    for resp_len in response_lengths:
        end_idx = start_idx + resp_len
        log_probs_per_sample.append(log_probs[start_idx:end_idx])
        start_idx = end_idx
    
    # Collect log_probs for tokens marked as positive in positive_nll_mask
    positive_log_probs_list = []
    total_positive_tokens = 0
    
    for log_prob_sample, nll_mask in zip(log_probs_per_sample, positive_nll_mask, strict=False):
        # Apply positive_nll_mask to get valid positive tokens
        # positive_nll_mask already contains the information about which tokens are positive
        masked_log_probs = log_prob_sample * nll_mask.float()
        positive_log_probs_list.append(masked_log_probs)
        
        # Count valid positive tokens for normalization
        num_valid_tokens = nll_mask.sum().item()
        total_positive_tokens += num_valid_tokens
    
    # If no positive samples, return zero loss
    if total_positive_tokens == 0 or len(positive_log_probs_list) == 0:
        return torch.tensor(0.0, device=device, dtype=dtype, requires_grad=True)
    
    # Concatenate all positive sample log_probs
    all_positive_log_probs = torch.cat(positive_log_probs_list, dim=0)
    
    # Compute NLL loss: -1/P * Σ log πθ(at|st)
    # Formula: LNLL(θ) = -mean(log_probs) = -sum(log_probs) / P
    nll_loss = -all_positive_log_probs.sum() / total_positive_tokens
    
    return nll_loss

def compute_advantage_diff_mask(
    values1: list[torch.Tensor],
    values2: list[torch.Tensor],
    top_k_percent: float = 0.1,
    response_lengths: list[int] | None = None,
) -> list[torch.Tensor]:
    """
    计算基于两个critic值差异的advantage mask（token级别）。
    
    根据公式(4)，计算两个critic输出在每个token位置的标准差，
    然后在所有token中找出标准差最小（agreement最高）的top_k_percent，
    mask掉这些低信息token的advantage。
    
    Args:
        values1: 第一个critic的value估计列表，每个元素shape=[response_length]
        values2: 第二个critic的value估计列表，每个元素shape=[response_length]
        top_k_percent: mask掉标准差最小的top k%的token，默认0.1表示mask掉10%的token
        response_lengths: 每个样本的response长度列表（可选，用于验证）
    
    Returns:
        mask_list: 每个样本的mask tensor列表，1表示保留，0表示mask掉
    
    公式解释：
    - σ_t = std(V1_t, V2_t) 计算每个token位置两个critic的标准差
    - 在所有token中找出标准差最小的top k%: Low_k(σ)
    - 这些token的 I^A_t = 0（被mask），其余 I^A_t = 1（保留）
    """
    assert len(values1) == len(values2), f"values1和values2长度不匹配: {len(values1)} vs {len(values2)}"
    
    device = values1[0].device
    dtype = values1[0].dtype
    
    # Step 1: 计算每个token位置的标准差
    token_std_list = []
    for v1, v2 in zip(values1, values2, strict=True):
        # 检查形状是否一致
        assert v1.shape == v2.shape, f"values形状不匹配: {v1.shape} vs {v2.shape}"
        
        # 计算每个token位置的标准差: σ_t = std(V1_t, V2_t)
        # 将两个值堆叠成[2, response_length]，然后沿dim=0计算std
        stacked = torch.stack([v1, v2], dim=0)  # [2, response_length]
        token_std = torch.std(stacked, dim=0, unbiased=False)  # [response_length]
        
        token_std_list.append(token_std)
    
    # Step 2: 将所有token的标准差拼接成一个大tensor
    all_token_std = torch.cat(token_std_list, dim=0)  # [total_tokens]
    total_tokens = all_token_std.numel()
    
    # Step 3: 精确选出 std 最小的 top_k 个 token 的“全局索引”
    num_to_mask = max(1, int(total_tokens * top_k_percent))
    num_to_mask = min(num_to_mask, total_tokens)

    # 取最小的 num_to_mask 个位置（精确到索引，而不是阈值）
    _, flat_mask_indices = torch.topk(
        all_token_std,
        k=num_to_mask,
        largest=False,   # 最小的
        sorted=False,
    )

    # Step 4: 根据索引构造全局 mask，再切回每个样本
    global_mask = torch.ones(total_tokens, device=all_token_std.device, dtype=all_token_std.dtype)
    global_mask[flat_mask_indices] = 0.0

    mask_list = []
    offset = 0
    for token_std in token_std_list:
        n = token_std.numel()
        mask_list.append(global_mask[offset : offset + n].view_as(token_std))
        offset += n
    assert offset == total_tokens
    
    # Step 5: 统计信息
    total_masked = sum([(mask == 0).sum().item() for mask in mask_list])
    std_min = all_token_std.min().item()
    std_max = all_token_std.max().item()
    std_mean = all_token_std.mean().item()
    
    logger.info(
        f"Advantage diff mask (token-level): masked {total_masked}/{total_tokens} tokens "
        f"(top {top_k_percent*100:.1f}% lowest std by exact indices). "
        f"Std - min: {std_min:.4f}, max: {std_max:.4f}, mean: {std_mean:.4f}"
    )
    
    return mask_list


def compute_entropy_mask_by_value_divergence(
    values1: list[torch.Tensor],
    values2: list[torch.Tensor],
    top_h_percent: float = 0.2,
    response_lengths: list[int] | None = None,
) -> list[torch.Tensor]:
    """
    计算基于两个critic值差异的熵正则化过滤mask（token级别）。
    
    根据论文公式(5)，计算两个critic输出在每个token位置的标准差，
    然后在所有token中找出标准差最大（分歧最大）的top_h_percent，
    将这些高分歧token的熵正则化项过滤掉（mask=0），其余保留（mask=1）。
    
    Args:
        values1: 第一个critic的value估计列表，每个元素shape=[response_length]
        values2: 第二个critic的value估计列表，每个元素shape=[response_length]
        top_h_percent: 过滤掉标准差最大的top h%的token，默认0.1表示过滤掉10%的token
        response_lengths: 每个样本的response长度列表（可选，用于验证）
    
    Returns:
        mask_list: 每个样本的mask tensor列表，1表示保留熵正则化，0表示过滤掉
    
    公式解释：
    - σ_t = std(V1_t, V2_t) 计算每个token位置两个critic的标准差
    - 在所有token中找出标准差最大的top h%: top_h(σ)
    - 对于这些token: I^H_t = 0（过滤掉熵正则化）
    - 对于其余token: I^H_t = 1（保留熵正则化）
    """
    assert len(values1) == len(values2), f"values1和values2长度不匹配: {len(values1)} vs {len(values2)}"
    
    device = values1[0].device
    dtype = values1[0].dtype
    
    # Step 1: 计算每个token位置的标准差
    token_std_list = []
    for v1, v2 in zip(values1, values2, strict=True):
        # 检查形状是否一致
        assert v1.shape == v2.shape, f"values形状不匹配: {v1.shape} vs {v2.shape}"
        
        # 计算每个token位置的标准差: σ_t = std(V1_t, V2_t)
        # 将两个值堆叠成[2, response_length]，然后沿dim=0计算std
        stacked = torch.stack([v1, v2], dim=0)  # [2, response_length]
        token_std = torch.std(stacked, dim=0, unbiased=False)  # [response_length]
        
        token_std_list.append(token_std)
    
    # Step 2: 将所有token的标准差拼接成一个大tensor
    all_token_std = torch.cat(token_std_list, dim=0)  # [total_tokens]
    total_tokens = all_token_std.numel()
    
    # Step 3: 找出标准差最大的top_h个token（分歧最大的token）
    num_to_mask = max(1, int(total_tokens * top_h_percent))
    
    # 获取标准差最大的top_h个token的阈值
    # torch.kthvalue 返回第k小的值
    # 我们需要第(total_tokens - num_to_mask + 1)小的值作为阈值
    k_value = max(1, total_tokens - num_to_mask + 1)
    threshold_std, _ = torch.kthvalue(all_token_std, k_value)
    
    # Step 4: 创建mask：标准差大于等于阈值的token被过滤掉（mask=0）
    mask_list = []
    for token_std in token_std_list:
        # 对于标准差 >= threshold 的token（高分歧），mask设为0；其余为1
        mask = (token_std < threshold_std).float()
        mask_list.append(mask)
    
    # Step 5: 统计信息
    total_masked = sum([(mask == 0).sum().item() for mask in mask_list])
    std_min = all_token_std.min().item()
    std_max = all_token_std.max().item()
    std_mean = all_token_std.mean().item()
    
    logger.info(f"Entropy mask (value divergence): masked {total_masked}/{total_tokens} tokens "
                f"(top {top_h_percent*100:.1f}% highest std). "
                f"Std - min: {std_min:.4f}, max: {std_max:.4f}, mean: {std_mean:.4f}, "
                f"threshold: {threshold_std.item():.4f}")
    
    return mask_list