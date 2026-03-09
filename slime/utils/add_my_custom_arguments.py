def add_my_custom_arguments(parser):
    # 添加你的自定义参数
    parser.add_argument("--critic-ref-load", type=str, default=None, help="The reference checkpoint for critic model (Megatron format). Used when critic-load is invalid.")
    
    parser.add_argument(
    "--critic-hf-checkpoint", 
    type=str, 
    default=None, 
    help="The checkpoint for critic model's HF config. If not set, will use the same as the critic model.")
    parser.add_argument("--use-asyppo", action="store_true", default=False, help="Whether to use AsyPPO,critic model must have the same tokenizer as the actor model.")
    
    parser.add_argument("--lambd-actor", type=float, default=0.95, help="PPO GAE lambd for actor")
    
    parser.add_argument("--lambd-critic", type=float, default=0.95, help="PPO GAE lambd for critic")
    
    parser.add_argument(
                "--log-position-value-stats",
                action="store_true",
                default=False,
                help="Whether to log the position-value stats.",
            )
    
    parser.add_argument(
                "--max-log-positions",
                type=int,
                default=500,
                help="The maximum number of positions to log the value stats.",
            )
    parser.add_argument("--use-adaptive-lambda", action="store_true", default=False, help="Whether to use adaptive lambda for PPO GAE.")
    parser.add_argument("--alpha", type=float, default=0.05, help="The alpha for adaptive lambda for PPO GAE.")
    parser.add_argument("--rm-timeout", type=float, default=600.0, help="The timeout for RM.")
    parser.add_argument("--train-use-xverify", action="store_true", default=False, help="Whether to use XVerify for training.")
    parser.add_argument("--eval-use-xverify", action="store_true", default=False, help="Whether to use XVerify for evaluation.")
    parser.add_argument("--eval-group", action="store_true", default=False, help="Whether to evaluate the model in group.")
    parser.add_argument("--eval-log-dir", type=str, default=".", help="The directory to save the evaluation results.")
    parser.add_argument("--eval-first", action="store_true", default=False, help="Whether to only evaluate the model.")
    parser.add_argument("--log-position-advantage-stats", action="store_true", default=False, help="Whether to log the position-advantage stats.")
    parser.add_argument("--max-log-positions-advantage", type=int, default=500, help="The maximum number of positions to log the advantage stats.")
    parser.add_argument("--use-positive-nll-loss", action="store_true", default=False, help="Whether to use positive NLL loss.")
    parser.add_argument("--positive-nll-coef", type=float, default=0.1, help="The coefficient for positive NLL loss.")
    parser.add_argument("--positive-reward-threshold", type=float, default=0.0, help="The threshold for positive reward.")
    
        # ========== 添加第二个 critic 的参数 ==========
    parser.add_argument("--use-critic2", action="store_true", default=False, 
                       help="Whether to use a second critic model.")
    parser.add_argument("--critic2-num-nodes", type=int, default=None, 
                       help="Number of nodes for the second critic model.")
    parser.add_argument("--critic2-num-gpus-per-node", type=int, default=None, 
                       help="Number of GPUs per node for the second critic model.")
    parser.add_argument("--critic2-lr", type=float, default=None, 
                       help="Learning rate for the second critic model.")
    parser.add_argument("--critic2-hf-checkpoint", type=str, default=None, 
                       help="The checkpoint for critic2 model's HF config.")
    parser.add_argument("--critic2-ref-load", type=str, default=None, 
                       help="The reference checkpoint for critic2 model (Megatron format).")
    parser.add_argument("--critic2-save", type=str, default=None, 
                       help="The save path for critic2 model.")
    parser.add_argument("--critic2-lr-warmup-iters", type=int, default=None, 
                       help="Learning rate warmup iterations for critic2.")
    parser.add_argument("--critic2-load", type=str, default=None, 
                       help="The load path for critic2 model.")

    # ========== 第二个 critic 参数结束 ==========
    
    
    #asyppo训练相关
    parser.add_argument("--use-asytrain-critic", action="store_true", default=False,
                   help="Whether to use asynchronous critic training where different critics train on different samples.")
    parser.add_argument(
    '--use-advantage-diff-mask',
    action='store_true',
    help='Enable advantage masking based on critic value agreement (标准差最小的样本被mask)'
    )
    parser.add_argument(
        '--advantage-diff-mask-k',
        type=float,
        default=0.1,
        help='Percentage of samples to mask (with lowest std/highest agreement). Default: 0.1 (10%)'
    )
    parser.add_argument(
        '--use-entropy-value-divergence-filter',
        action='store_true',
        help='Enable entropy value divergence filter.'
    )
    parser.add_argument(
        '--entropy-divergence-filter-h',
        type=float,
        default=0.2,
        help='Percentage of samples to filter (with highest entropy value divergence). Default: 0.1 (10%)'
    )
    
    parser.add_argument(
        '--use-bce-value-loss',
        action='store_true',
        help='Enable BCE loss for value head.'
    )
    
    parser.add_argument(
    "--log-probs-chunk-size",
    type=int,
    default=-1,
    help="Chunk size for log_probs/entropy. -1 disables. Use 8192 for long sequences.",
)
    return parser 