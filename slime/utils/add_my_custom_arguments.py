def add_my_custom_arguments(parser):
    # 添加你的自定义参数
    parser.add_argument(
        "--my-custom-arg",
        type=str,
        default="default_value",
        help="这是我的自定义参数",
    )
    parser.add_argument(
        "--another-arg",
        type=int,
        default=100,
        help="另一个自定义参数",
    )
        
    return parser 