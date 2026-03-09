# Critic2 参数总结

本文档总结了所有 critic2 的参数及其回退逻辑。

## 参数列表

### 1. 资源配置参数
- `--use-critic2`: 是否启用第二个 critic 模型 (action="store_true", default=False)
- `--critic2-num-nodes`: Critic2 的节点数 (default: 回退到 critic_num_nodes)
- `--critic2-num-gpus-per-node`: Critic2 每个节点的 GPU 数 (default: 回退到 critic_num_gpus_per_node)

### 2. Checkpoint 路径参数
- `--critic2-hf-checkpoint`: Critic2 的 HuggingFace checkpoint 路径 (default: 回退到 critic_hf_checkpoint)
- `--critic2-ref-load`: Critic2 的参考模型加载路径 (default: 回退到 critic_ref_load)
- `--critic2-load`: Critic2 的完整 checkpoint 加载路径 (default: 回退到 critic_load)
- `--critic2-save`: Critic2 的保存路径 (default: 回退到 critic_save)

### 3. 训练参数
- `--critic2-lr`: Critic2 的学习率 (default: 回退到 critic_lr)
- `--critic2-lr-warmup-iters`: Critic2 的学习率 warmup 迭代次数 (default: 回退到 lr_warmup_iters)

## 回退逻辑实现位置

### 1. placement_group.py (第 74-86 行)
```python
# 为 critic2 参数设置回退逻辑
if args.use_critic2:
    if not hasattr(args, 'critic2_num_nodes') or args.critic2_num_nodes is None:
        args.critic2_num_nodes = args.critic_num_nodes if args.use_critic else 1
    
    if not hasattr(args, 'critic2_num_gpus_per_node') or args.critic2_num_gpus_per_node is None:
        args.critic2_num_gpus_per_node = args.critic_num_gpus_per_node if args.use_critic else args.actor_num_gpus_per_node
```

### 2. actor.py - HF Checkpoint 配置 (第 79-103 行)
```python
if role == "critic2":
    # 如果未指定 critic2_hf_checkpoint，回退使用 critic_hf_checkpoint
    critic2_hf_checkpoint_to_use = (
        args.critic2_hf_checkpoint if (hasattr(args, 'critic2_hf_checkpoint') and args.critic2_hf_checkpoint is not None)
        else (args.critic_hf_checkpoint if hasattr(args, 'critic_hf_checkpoint') else args.hf_checkpoint)
    )
    # ... 应用配置
```

### 3. actor.py - 训练参数配置 (第 161-180 行)
```python
elif role == "critic2":
    # Checkpoint 路径参数
    self.args.load = (
        self.args.critic2_ref_load if self.args.critic2_ref_load is not None 
        else self.args.critic_ref_load
    )
    self.args.save = (
        self.args.critic2_save if self.args.critic2_save is not None 
        else self.args.critic_save
    )
    
    # 学习率参数
    self.args.lr = (
        self.args.critic2_lr if self.args.critic2_lr is not None 
        else self.args.critic_lr
    )
    self.args.lr_warmup_iters = (
        self.args.critic2_lr_warmup_iters if self.args.critic2_lr_warmup_iters is not None 
        else self.args.lr_warmup_iters
    )
```

### 4. actor.py - AsyPPO Checkpoint 加载逻辑 (第 181-218 行)
```python
if hasattr(args, 'use_asyppo') and args.use_asyppo:
    # 获取 critic2_load，如果未指定则使用 critic_load
    critic2_load_to_check = (
        self.args.critic2_load if self.args.critic2_load is not None 
        else self.args.critic_load
    )
    
    # 检查有效性并进行回退
    if not critic_load_valid:
        ref_load_to_use = (
            args.critic2_ref_load if (hasattr(args, 'critic2_ref_load') and args.critic2_ref_load)
            else args.critic_ref_load
        )
        # ... 使用回退的 checkpoint
```

## 使用示例

### 示例 1: 最小配置（完全使用 critic1 的配置）
```bash
--use-critic2
```
所有参数都会自动回退到 critic1 的值。

### 示例 2: 只指定不同的模型
```bash
--use-critic2 \
--critic2-hf-checkpoint /path/to/different/model \
--critic2-ref-load /path/to/different/checkpoint \
--critic2-save /path/to/save/critic2
```
其他参数（学习率、资源配置等）会使用 critic1 的值。

### 示例 3: 完全自定义配置
```bash
--use-critic2 \
--critic2-num-nodes 1 \
--critic2-num-gpus-per-node 2 \
--critic2-hf-checkpoint /path/to/model \
--critic2-ref-load /path/to/checkpoint \
--critic2-save /path/to/save \
--critic2-lr 2e-6 \
--critic2-lr-warmup-iters 100
```

## 注意事项

1. **参数回退顺序**：
   - critic2 参数 → critic 参数 → 默认值

2. **必须设置的参数**：
   - `--use-critic2`: 启用第二个 critic（必需）
   - 其他参数都有回退逻辑，可以不设置

3. **推荐配置**：
   - 如果两个 critics 使用相同的模型，只需设置 `--use-critic2` 和保存路径 `--critic2-save`
   - 如果使用不同的模型，还需要设置 `--critic2-hf-checkpoint` 和相关 checkpoint 路径

4. **资源分配**：
   - 确保总 GPU 数足够：`actor_gpus + critic1_gpus + critic2_gpus + rollout_gpus`
   - 使用 `ray start --num-gpus` 时要考虑所有组件的 GPU 需求

