# reward_model_server.py
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from typing import Optional
import uvicorn
from typing import Union, List, Optional
import asyncio
import signal
import sys
import gc
import time
import argparse
import os
from p1 import compute_score_p1

app = FastAPI()

# 全局变量用于优雅关闭
shutdown_event = asyncio.Event()

# 全局变量存储模型端口
MODEL_PORT = int(os.environ.get('MODEL_PORT', 34882))

class RewardRequest(BaseModel):
    response: str
    label: Optional[Union[str, List[str]]] = None  # 可以是 None
    points: Optional[List[float]] = None
    question: Optional[str] = None
    use_xverify: bool = False

@app.post("/")
async def evaluate_reward(req: RewardRequest):
    print("Received Request:", req)
    try:
        # 直接同步调用，避免signal问题
        result = compute_score_p1(
            model_output=req.response, 
            label=req.label, 
            points=req.points, 
            question=req.question, 
            use_xverify=req.use_xverify,
            model_port=MODEL_PORT
        )
        return result
    except Exception as e:
        print(f"Error processing request: {e}")
        # 强制垃圾回收以清理可能的资源泄漏
        gc.collect()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": time.time()}

@app.on_event("startup")
async def startup_event():
    print("Reward model server started successfully!")
    # 设置信号处理器
    def signal_handler(signum, frame):
        print(f"Received signal {signum}, shutting down gracefully...")
        shutdown_event.set()
    
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

@app.on_event("shutdown")
async def shutdown_event_handler():
    print("Shutting down reward model server...")
    # 强制垃圾回收
    gc.collect()
    # 等待一小段时间确保资源释放
    await asyncio.sleep(1)

# 定期清理资源的后台任务
@app.middleware("http")
async def cleanup_middleware(request: Request, call_next):
    try:
        response = await call_next(request)
        # 每10个请求后强制垃圾回收
        if hasattr(request.app.state, 'request_count'):
            request.app.state.request_count += 1
        else:
            request.app.state.request_count = 1
        
        if request.app.state.request_count % 10 == 0:
            gc.collect()
            print(f"Cleaned up resources after {request.app.state.request_count} requests")
        
        return response
    except Exception as e:
        print(f"Error in middleware: {e}")
        # 强制垃圾回收
        gc.collect()
        raise

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Reward Model Server")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8001, help="Server port")
    parser.add_argument("--timeout-keep-alive", type=int, default=30, help="Keep alive timeout")
    parser.add_argument("--log-level", default="info", help="Log level")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    print(f"Starting reward model server on {args.host}:{args.port}")
    uvicorn.run(
        "reward_model_server:app",
        host=args.host,
        port=args.port,
        timeout_keep_alive=args.timeout_keep_alive,
        log_level=args.log_level
    )

