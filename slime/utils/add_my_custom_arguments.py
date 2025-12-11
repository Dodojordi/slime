def add_my_custom_arguments(parser):
    # 添加你的自定义参数
    parser.add_argument("--critic-ref-load", type=str, default=None, help="The reference checkpoint for critic model (Megatron format). Used when critic-load is invalid.")
    
    parser.add_argument(
    "--critic-hf-checkpoint", 
    type=str, 
    default=None, 
    help="The checkpoint for critic model's HF config. If not set, will use the same as the critic model.")
    
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

    return parser 