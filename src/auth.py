"""
=============================================================================
API 认证模块

提供两套认证机制，可通过配置灵活开关：
    1. JWT 令牌：POST /api/auth/login 获取，请求头 `Authorization: Bearer <token>`
    2. 静态 API Key：请求头 `X-API-Key: <key>`，密钥列表在 .env 的 API_KEYS 中配置

启用方式：.env 中设置 AUTH_ENABLED=true。
未启用时 require_auth 直接放行，不影响开发环境使用。

使用方法:
    from src.auth import require_auth, create_token, login

    @router.post("/api/xxx")
    async def xxx(credentials: AuthCredentials = Depends(require_auth)):
        ...
=============================================================================
"""

from datetime import datetime, timedelta, timezone
from typing import Optional

import jwt
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from config.settings import settings
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# HTTPBearer 自动解析 `Authorization: Bearer <token>` 头
_bearer_scheme = HTTPBearer(auto_error=False)


class AuthError(Exception):
    """认证失败异常"""
    pass


# ==============================================================================
# Token 签发与校验
# ==============================================================================

def create_token(
    username: str,
    expires_minutes: int | None = None,
) -> str:
    """
    签发 JWT 令牌。

    Args:
        username:      用户名（放入 token 的 sub 声明）
        expires_minutes: 有效期（分钟），默认取配置 JWT_EXPIRE_MINUTES

    Returns:
        JWT 字符串
    """
    expire_minutes = expires_minutes or settings.JWT_EXPIRE_MINUTES
    now = datetime.now(timezone.utc)
    payload = {
        "sub": username,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=expire_minutes)).timestamp()),
    }
    token = jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)
    return token


def verify_token(token: str) -> dict:
    """
    校验 JWT 令牌并返回其 payload。

    Raises:
        AuthError: token 无效或已过期
    """
    try:
        payload = jwt.decode(
            token,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM],
        )
        return payload
    except jwt.ExpiredSignatureError:
        raise AuthError("令牌已过期")
    except jwt.InvalidTokenError as e:
        raise AuthError(f"无效的令牌: {e}")


def _api_keys() -> list[str]:
    """解析配置中的 API Key 列表"""
    return [k.strip() for k in settings.API_KEYS.split(",") if k.strip()]


def authenticate_request(
    bearer_cred: Optional[HTTPAuthorizationCredentials],
    api_key: Optional[str],
) -> dict:
    """
    统一认证入口：优先 Bearer token，其次 X-API-Key。

    Returns:
        认证主体信息 dict（含 username 或 api_key）

    Raises:
        HTTPException(401): 认证失败
    """
    # ---- 方式一：JWT Bearer Token ----
    if bearer_cred and bearer_cred.scheme.lower() == "bearer":
        try:
            payload = verify_token(bearer_cred.credentials)
            return {"auth_type": "jwt", "username": payload.get("sub", "unknown")}
        except AuthError as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"JWT 认证失败: {e}",
            )

    # ---- 方式二：静态 API Key ----
    if api_key and api_key in _api_keys():
        return {"auth_type": "api_key", "api_key": api_key}

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="未认证：请在请求头中携带 Authorization: Bearer <token> 或 X-API-Key: <key>",
        headers={"WWW-Authenticate": "Bearer"},
    )


# ==============================================================================
# FastAPI 依赖
# ==============================================================================

async def require_auth(
    request: Request,
    bearer_cred: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_scheme),
) -> dict:
    """
    FastAPI 依赖：要求请求通过认证。

    AUTH_ENABLED=false 时直接放行（开发环境）。
    启用后所有依赖此参数的接口都会要求合法 token 或 API Key。

    返回的 dict 含 username 与 is_admin（从数据库实时查询）。

    用法:
        @router.get("/api/stats")
        async def stats(auth: dict = Depends(require_auth)):
            ...
    """
    if not settings.AUTH_ENABLED:
        return {"auth_type": "disabled", "username": "anonymous", "is_admin": False}

    api_key = request.headers.get("X-API-Key")
    auth = authenticate_request(bearer_cred, api_key)

    # 附加管理员标记：从数据库实时查（不轻信 token，防篡改）
    username = auth.get("username")
    if username:
        try:
            user_manager = request.app.state.user_manager
            user = await user_manager._store.aget_user_by_username(username)
            auth["is_admin"] = bool(user and user.get("is_admin"))
        except Exception:
            auth["is_admin"] = False
    else:
        auth["is_admin"] = False
    return auth
