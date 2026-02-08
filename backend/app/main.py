from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import uvicorn
import os
import uuid
import shutil
from typing import Optional

from app.api.endpoints import router as api_router
from app.core.config import settings
from app.services.task_manager import TaskManager

# 创建lifespan上下文管理器
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时初始化
    os.makedirs(settings.TEMP_DIR, exist_ok=True)
    os.makedirs(settings.OUTPUT_DIR, exist_ok=True)
    print(f"临时目录: {settings.TEMP_DIR}")
    print(f"输出目录: {settings.OUTPUT_DIR}")
    
    yield  # 应用运行期间
    
    # 关闭时清理（可选）
    pass

# 创建FastAPI应用
app = FastAPI(
    title="基于语义对齐的自动视频生成API",
    description="通过语义对齐将PPTX、MP3和SRT合成为自动切换页面的视频",
    version="1.0.0",
    lifespan=lifespan
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境中应限制来源
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 包含API路由
app.include_router(api_router, prefix="/api")

# 挂载静态文件（前端）
frontend_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "frontend")
if os.path.exists(frontend_path):
    app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")

# 初始化任务管理器
task_manager = TaskManager()



@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {"status": "healthy", "service": "video-generation-api"}

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )