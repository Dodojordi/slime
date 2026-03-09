#!/bin/bash

# 默认端口设置
DEFAULT_MODEL_PORT=34882
DEFAULT_REWARD_PORT=8001

# 解析命令行参数
MODEL_PORT=$DEFAULT_MODEL_PORT
REWARD_PORT=$DEFAULT_REWARD_PORT

# 显示帮助信息
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -m, --model-port PORT    指定30B模型服务器端口 (默认: $DEFAULT_MODEL_PORT)"
    echo "  -r, --reward-port PORT   指定reward model服务器端口 (默认: $DEFAULT_REWARD_PORT)"
    echo "  -h, --help               显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  $0                                    # 使用默认端口"
    echo "  $0 -m 34883 -r 8002                  # 指定自定义端口"
    echo "  $0 --model-port 34883 --reward-port 8002"
    echo ""
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model-port)
            MODEL_PORT="$2"
            shift 2
            ;;
        -r|--reward-port)
            REWARD_PORT="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            show_help
            exit 1
            ;;
    esac
done

# 验证端口参数
if ! [[ "$MODEL_PORT" =~ ^[0-9]+$ ]] || [ "$MODEL_PORT" -lt 1024 ] || [ "$MODEL_PORT" -gt 65535 ]; then
    echo "错误: 模型端口必须是1024-65535之间的数字"
    exit 1
fi

if ! [[ "$REWARD_PORT" =~ ^[0-9]+$ ]] || [ "$REWARD_PORT" -lt 1024 ] || [ "$REWARD_PORT" -gt 65535 ]; then
    echo "错误: Reward端口必须是1024-65535之间的数字"
    exit 1
fi

if [ "$MODEL_PORT" -eq "$REWARD_PORT" ]; then
    echo "错误: 模型端口和reward端口不能相同"
    exit 1
fi

echo "配置信息:"
echo "  30B模型服务器端口: $MODEL_PORT"
echo "  Reward模型服务器端口: $REWARD_PORT"
echo ""

cd /mnt/shared-storage-user/p1-shared/liyizhuo/code/slime/rm-cgq

# 设置环境变量
export PIP_INDEX_URL="http://mirrors.h.pjlab.org.cn/pypi/simple/"
export PIP_EXTRA_INDEX_URL="http://pypi.i.h.pjlab.org.cn/brain/dev/+simple"
export PIP_TRUSTED_HOST="mirrors.h.pjlab.org.cn pypi.i.h.pjlab.org.cn"
export PIP_NO_INDEX="false"  # 如果要完全禁用公网访问，改为 "true"

# 安装依赖
echo "Installing dependencies..."
pip install math_verify
pip install fastapi uvicorn torch

# 启动30B模型服务器（支持多卡）
echo "Starting 30B model server with multi-GPU support on port $MODEL_PORT..."
python launch_30b.py --port $MODEL_PORT &
MODEL_SERVER_PID=$!

# 等待模型服务器启动
sleep 10

# 启动reward model server的函数
start_reward_server() {
    echo "Starting reward model server on port $REWARD_PORT..."
    echo "Using model server port: $MODEL_PORT"
    MODEL_PORT=$MODEL_PORT uvicorn reward_model_server:app --host 0.0.0.0 --port $REWARD_PORT --timeout-keep-alive 30 --log-level info --workers 32
}

# 监控函数
monitor_server() {
    while true; do
        # 检查reward server是否响应
        if ! curl -s http://localhost:$REWARD_PORT/health > /dev/null 2>&1; then
            echo "Reward server is not responding, restarting..."
            pkill -f "uvicorn reward_model_server"
            sleep 2
            start_reward_server &
            REWARD_SERVER_PID=$!
        else
            # 服务器正常运行，只记录状态，不重启
            echo "Reward server is healthy"
        fi
        
        # 检查模型服务器是否还在运行
        if ! kill -0 $MODEL_SERVER_PID 2>/dev/null; then
            echo "Model server died, restarting..."
            python launch_30b.py --port $MODEL_PORT &
            MODEL_SERVER_PID=$!
            sleep 10
        fi
        
        sleep 30  # 每30秒检查一次
    done
}

# 启动reward server
start_reward_server &
REWARD_SERVER_PID=$!

# 启动监控
monitor_server &
MONITOR_PID=$!

# 等待信号
trap 'echo "Shutting down servers..."; kill $REWARD_SERVER_PID $MODEL_SERVER_PID $MONITOR_PID 2>/dev/null; exit' SIGTERM SIGINT

# 等待所有进程
wait
