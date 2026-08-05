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
from contextlib import asynccontextmanager
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
# 应用生命周期
# ==============================================================================
# 使用 lifespan 在启动时创建共享单例（RAG 管线、文档加载器、对话管理器），
# 存放到 app.state，供路由通过依赖注入访问；退出时统一关闭资源
# （asyncpg 连接池等），替代已废弃的 on_event 钩子。

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ---- 启动阶段 ----
    logger.info("=" * 60)
    logger.info("企业知识库问答系统启动中...")

    from src.document_loader import DocumentLoader
    from src.rag import RAGPipeline
    from src.text_processor import TextChunker
    from src.conversations import PGConversationManager
    from src.users import UserManager

    # 创建 RAG 管线（内部根据 VECTOR_STORE_TYPE 选择后端）
    rag_pipeline = RAGPipeline()
    app.state.rag_pipeline = rag_pipeline

    # 文档加载器与分块器
    app.state.document_loader = DocumentLoader()
    app.state.text_chunker = TextChunker()

    # 对话管理器：复用 RAG 管线的 PGVectorStore（若为 PG 后端）
    if hasattr(rag_pipeline.vector_store, "aadd_message"):
        app.state.conversation_mgr = PGConversationManager(vector_store=rag_pipeline.vector_store)
    else:
        app.state.conversation_mgr = PGConversationManager()

    # 用户管理器：复用 PGVectorStore，确保 admin 账号存在
    app.state.user_manager = UserManager(vector_store=rag_pipeline.vector_store)
    try:
        await app.state.user_manager.ensure_admin()
    except Exception as e:
        logger.warning(f"admin 账号初始化失败（不影响启动）: {e}")

    # 异步导入任务管理器（大文件后台导入）
    from src.ingest_tasks import IngestTaskManager
    app.state.ingest_task_mgr = IngestTaskManager()

    logger.info("共享组件初始化完成")

    if FRONTEND_DIST.exists():
        logger.info(f"前端界面: http://localhost:{settings.PORT}/")
    logger.info(f"API 文档: http://localhost:{settings.PORT}/docs")
    logger.info(f"健康检查: http://localhost:{settings.PORT}/api/health")
    logger.info("=" * 60)

    yield

    # ---- 关闭阶段 ----
    logger.info("企业知识库问答系统正在关闭...")
    try:
        # 关闭 asyncpg 连接池
        store = getattr(rag_pipeline, "vector_store", None)
        if store is not None and hasattr(store, "aclose"):
            await store.aclose()
    except Exception as e:
        logger.warning(f"关闭资源时出错: {e}")
    logger.info("企业知识库问答系统已关闭")


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
    - 🔐 JWT/API Key 认证（可通过 AUTH_ENABLED 开关）
    - 💬 多轮对话，自动关联历史
    - 📊 知识库管理，查看统计信息
    """,
    version="1.0.0",
    contact={
        "name": "Enterprise KB Team",
    },
    lifespan=lifespan,
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
    # Windows 上 Python 的 mimetypes 可能从注册表误识别 .js 为 text/plain，
    # 导致浏览器严格 MIME 检查拒绝加载前端模块脚本。这里显式纠正。
    # 注意：不要在 add_type 之后调用 mimetypes.init()，那会重新加载注册表覆盖本次修正。
    import mimetypes

    if mimetypes.guess_type("app.js")[0] != "application/javascript":
        mimetypes.add_type("application/javascript", ".js", strict=True)
        logger.info("已纠正 .js 文件的 MIME 类型为 application/javascript")

    class NoCacheStaticFiles(StaticFiles):
        """静态文件挂载：附加 no-cache 头，避免浏览器缓存旧的构建产物。"""

        def file_response(
            self, full_path, stat_result=None, scope=None, status_code=200
        ):
            response = super().file_response(
                full_path, stat_result=stat_result, scope=scope, status_code=status_code
            )
            response.headers.setdefault("Cache-Control", "no-cache")
            return response

    app.mount(
        "/assets",
        NoCacheStaticFiles(directory=str(FRONTEND_DIST / "assets")),
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
# 入口
# ==============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=False,
        log_level="info",
    )
