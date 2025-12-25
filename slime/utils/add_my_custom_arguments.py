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
    parser.add_argument("--eval-first", action="store_true", default=False, help="Whether to only evaluate the model.")
    parser.add_argument("--log-position-advantage-stats", action="store_true", default=False, help="Whether to log the position-advantage stats.")
    parser.add_argument("--max-log-positions-advantage", type=int, default=500, help="The maximum number of positions to log the advantage stats.")
    parser.add_argument("--use-positive-nll-loss", action="store_true", default=False, help="Whether to use positive NLL loss.")
    parser.add_argument("--positive-nll-coef", type=float, default=0.1, help="The coefficient for positive NLL loss.")
    parser.add_argument("--positive-reward-threshold", type=float, default=0.0, help="The threshold for positive reward.")
    return parser 