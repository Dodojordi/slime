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

cd /mnt/shared-storage-user/p1-shared/chenjiacheng/p1-slime/rm-gptoss

# 设置环境变量
export PIP_INDEX_URL="http://mirrors.h.pjlab.org.cn/pypi/simple/"
export PIP_EXTRA_INDEX_URL="http://pypi.i.h.pjlab.org.cn/brain/dev/+simple"
export PIP_TRUSTED_HOST="mirrors.h.pjlab.org.cn pypi.i.h.pjlab.org.cn"
export PIP_NO_INDEX="false"  # 如果要完全禁用公网访问，改为 "true"

# 安装依赖
echo "Installing dependencies..."
pip install math_verify
pip install fastapi uvicorn torch

# 启动gpt-oss-120b模型服务器的函数
start_model_server() {
    echo "Starting gpt-oss-120b model server with multi-GPU support on port $MODEL_PORT..."
    python launch_model.py --port $MODEL_PORT --model-path /mnt/shared-storage-user/p1-shared/chengqianjia/models/gpt-oss-120b --host 0.0.0.0 --mem-fraction-static 0.8 --tensor-parallel-size 2 --max-concurrent-requests 100 --gpu-memory-utilization 0.9 &
    MODEL_SERVER_PID=$!
}

# 启动reward model server的函数
start_reward_server() {
    echo "Starting reward model server on port $REWARD_PORT..."
    echo "Using model server port: $MODEL_PORT"
    MODEL_PORT=$MODEL_PORT uvicorn reward_model_server:app --host 0.0.0.0 --port $REWARD_PORT --timeout-keep-alive 30 --log-level info --workers 32
}

# 检查端口是否在监听
check_port_listening() {
    local port=$1
    # 使用nc或ss命令检查端口是否在监听
    if command -v nc >/dev/null 2>&1; then
        nc -z localhost $port >/dev/null 2>&1
    elif command -v ss >/dev/null 2>&1; then
        ss -lnt | grep -q ":$port "
    else
        # 如果nc和ss都不可用，检查进程是否存在
        pgrep -f "sglang.launch_server.*--port.*$port" >/dev/null 2>&1
    fi
}

# 等待端口监听函数
wait_for_model_port() {
    local port=$1
    local max_attempts=120  # 等待最多20分钟
    local attempt=0
    
    echo "Waiting for model server to start listening on port $port..."
    while [ $attempt -lt $max_attempts ]; do
        if check_port_listening $port; then
            echo "Model server is listening on port $port"
            return 0
        fi
        sleep 10
        attempt=$((attempt + 1))
        if [ $((attempt % 6)) -eq 0 ]; then
            echo "Still waiting for model server... ($attempt/$max_attempts)"
        fi
    done
    
    echo "Timeout waiting for model server to start"
    return 1
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
        
        # 检查模型服务器端口是否在监听（而不是检查进程PID）
        if ! check_port_listening $MODEL_PORT; then
            echo "Model server port $MODEL_PORT is not listening, restarting..."
            # 确保之前的进程已停止
            pkill -f "launch_model.py.*--port.*$MODEL_PORT" 2>/dev/null
            pkill -f "sglang.launch_server.*--port.*$MODEL_PORT" 2>/dev/null
            sleep 5
            # 使用正确的启动命令重启
            start_model_server
            # 等待启动完成，避免下次循环再次重启
            wait_for_model_port $MODEL_PORT
        else
            echo "Model server port $MODEL_PORT is listening"
        fi
        
        sleep 30  # 每30秒检查一次
    done
}

# 启动gpt-oss-120b模型服务器
start_model_server

# 等待模型服务器启动
wait_for_model_port $MODEL_PORT


# 启动reward server
start_reward_server &
REWARD_SERVER_PID=$!

# 等待reward server启动，避免monitor立即杀掉它
echo "Waiting for reward server to initialize..."
sleep 10

# 启动监控
monitor_server &
MONITOR_PID=$!

# 等待信号
trap 'echo "Shutting down servers..."; kill $REWARD_SERVER_PID $MODEL_SERVER_PID $MONITOR_PID 2>/dev/null; exit' SIGTERM SIGINT

# 等待所有进程
wait
