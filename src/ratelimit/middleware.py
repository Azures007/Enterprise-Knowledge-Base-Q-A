"""
=============================================================================
FastAPI 限流中间件

对所有 API 请求进行分级限流：
    1. IP 限流（滑动窗口）
    2. 全局限流（滑动窗口）
    3. 百炼配额保护（令牌桶）
    4. 降级管理

请求通过后，在响应头中携带限流信息。
=============================================================================
"""

import time
from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from src.ratelimit.degradation import degrade_manager
from src.ratelimit.sliding_window import check_global_limit, check_ip_limit
from src.ratelimit.token_bucket import get_token_bucket
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# 不需要限流的路由
WHITE_LIST = {
    "/api/health",
    "/docs",
    "/redoc",
    "/openapi.json",
    "/",
}

# 需要百炼配额的路由（消耗令牌桶）
BAILIAN_ROUTES = {
    "/api/query",
    "/api/query/stream",
}

# 核心请求路由（降级时优先保护）
CORE_ROUTES = {
    "/api/query",
    "/api/query/stream",
}


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    分级限流中间件。

    执行顺序：
        1. 白名单跳过
        2. IP 限流（10 QPS）
        3. 全局限流（50 QPS）
        4. 降级检查
        5. 百炼配额保护（8 QPS）
        6. 处理请求
        7. 添加限流响应头
    """

    def __init__(self, app: ASGIApp):
        super().__init__(app)
        # 百炼 API 令牌桶：每秒 8 个，桶容量 16（预留安全余量）
        self.bailian_bucket = get_token_bucket("bailian_api", rate=8, capacity=16)

    async def dispatch(self, request: Request, call_next: Any) -> Any:
        path = request.url.path

        # ── 1. 白名单跳过 ──
        if path in WHITE_LIST or path.startswith("/assets") or path.startswith("/static"):
            return await call_next(request)

        # 获取客户端 IP
        ip = request.client.host if request.client else "unknown"

        # ── 2. IP 限流 ──
        ip_allowed, ip_count, ip_limit = check_ip_limit(ip)
        if not ip_allowed:
            logger.warning(f"IP 限流: {ip} ({ip_count}/{ip_limit})")
            return JSONResponse(
                status_code=429,
                content={
                    "code": -1,
                    "message": "请求过于频繁，请稍后再试",
                    "data": {
                        "error_type": "rate_limit_ip",
                        "retry_after": 1,
                    },
                },
                headers={
                    "Retry-After": "1",
                    "X-RateLimit-Limit": str(ip_limit),
                    "X-RateLimit-Remaining": "0",
                },
            )

        # ── 3. 全局限流 ──
        global_allowed, global_count, global_limit = check_global_limit()
        if not global_allowed:
            logger.warning(f"全局限流: {global_count}/{global_limit}")
            return JSONResponse(
                status_code=429,
                content={
                    "code": -1,
                    "message": "系统当前访问量较大，请稍后再试",
                    "data": {
                        "error_type": "rate_limit_global",
                        "retry_after": 2,
                    },
                },
                headers={
                    "Retry-After": "2",
                    "X-RateLimit-Limit": str(global_limit),
                    "X-RateLimit-Remaining": "0",
                },
            )

        # ── 4. 降级检查 ──
        is_core = path in CORE_ROUTES
        allowed, degrade_msg = degrade_manager.is_allowed(is_core_request=is_core)
        if not allowed:
            logger.warning(f"降级拒绝: {path} (level={degrade_manager.level_name})")
            return JSONResponse(
                status_code=429,
                content={
                    "code": -1,
                    "message": degrade_msg,
                    "data": {
                        "error_type": "degraded",
                        "degrade_level": degrade_manager.level_name,
                    },
                },
                headers={
                    "Retry-After": "5",
                    "X-Degrade-Level": degrade_manager.level_name,
                },
            )

        # ── 5. 百炼配额保护 ──
        if path in BAILIAN_ROUTES:
            if not self.bailian_bucket.consume():
                logger.warning(f"百炼配额限流: {path}")
                return JSONResponse(
                    status_code=429,
                    content={
                        "code": -1,
                        "message": "AI 模型服务当前负载较高，请稍后再试",
                        "data": {
                            "error_type": "rate_limit_llm",
                            "retry_after": 3,
                        },
                    },
                    headers={
                        "Retry-After": "3",
                        "X-RateLimit-Limit": str(self.bailian_bucket.capacity),
                        "X-RateLimit-Remaining": "0",
                    },
                )

        # ── 6. 处理请求 ──
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time

        # ── 7. 添加限流响应头 ──
        remaining = max(0, int(self.bailian_bucket.available))
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(int(time.time() + 1))
        response.headers["X-Process-Time"] = f"{process_time:.3f}s"

        return response