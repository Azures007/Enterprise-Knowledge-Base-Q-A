#!/usr/bin/env python3
"""
=============================================================================
企业知识库问答系统 - FastAPI 应用入口

启动命令:
    # 开发模式
    uvicorn app:app --reload --host 0.0.0.0 --port 8000

    # 生产模式
    uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4

    # 或通过 Python 直接运行
    python app.py

API 文档:
    - Swagger UI: http://localhost:8000/docs
    - ReDoc:      http://localhost:8000/redoc
=============================================================================
"""

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# 加载环境变量（必须在导入其他模块之前）
load_dotenv()

from src.api.routes import router
from src.utils.logger import setup_logger
from config.settings import settings

logger = setup_logger(__name__)


# ==============================================================================
# FastAPI 应用初始化
# ==============================================================================

app = FastAPI(
    title="企业知识库问答系统",
    description="""
    基于 RAG（检索增强生成）技术的企业知识库智能问答系统。

    功能特性:
    - 📄 支持 PDF、Word、TXT、Markdown 等多种文档格式
    - 🔍 语义化向量检索，精准定位相关知识
    - 💬 基于通义千问大模型生成专业回答
    - 📡 流式 SSE 输出，实时显示生成过程
    - 📊 知识库管理，查看统计信息
    """,
    version="1.0.0",
    contact={
        "name": "Enterprise KB Team",
    },
)


# ==============================================================================
# 中间件配置
# ==============================================================================

# CORS 跨域配置 - 允许前端跨域访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境请替换为具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 分级限流中间件
from src.ratelimit import RateLimitMiddleware
app.add_middleware(RateLimitMiddleware)


# ==============================================================================
# 全局异常处理
# ==============================================================================


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """全局未捕获异常处理"""
    logger.error(f"全局异常: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "code": -1,
            "message": f"服务器内部错误: {str(exc)}",
            "data": None,
        },
    )


# ==============================================================================
# 注册路由
# ==============================================================================

app.include_router(router)

# 注册前端静态文件（生产模式）
FRONTEND_DIST = settings.PROJECT_ROOT / "frontend" / "dist"
if FRONTEND_DIST.exists():
    app.mount(
        "/assets",
        StaticFiles(directory=str(FRONTEND_DIST / "assets")),
        name="frontend_assets",
    )

    @app.get("/")
    async def serve_frontend():
        """提供前端页面"""
        return FileResponse(str(FRONTEND_DIST / "index.html"))

    logger.info(f"前端静态文件已加载: {FRONTEND_DIST}")
else:
    logger.warning(
        f"前端静态文件未找到 ({FRONTEND_DIST})。"
        f"请先执行: cd frontend && npm install && npm run build"
    )


# ==============================================================================
# 应用事件
# ==============================================================================


@app.on_event("startup")
async def startup_event():
    """应用启动时的初始化操作"""
    logger.info("=" * 60)
    logger.info("企业知识库问答系统启动中...")

    if FRONTEND_DIST.exists():
        logger.info(f"前端界面: http://localhost:{settings.PORT}/")
    logger.info(f"API 文档: http://localhost:{settings.PORT}/docs")
    logger.info(f"健康检查: http://localhost:{settings.PORT}/api/health")
    logger.info("=" * 60)


@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭时的清理操作"""
    logger.info("企业知识库问答系统已关闭")


# ==============================================================================
# 入口
# ==============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True,
        log_level="info",
    )
