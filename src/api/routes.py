"""
=============================================================================
API 路由模块

提供 RESTful API 接口，供前端或其他服务调用知识库问答功能。

接口列表:
    POST   /api/auth/login       - 登录获取 JWT 令牌
    POST   /api/query            - 知识库问答（支持多轮对话）
    POST   /api/query/stream     - 知识库问答（流式 SSE，支持多轮对话）
    POST   /api/ingest           - 上传并导入文档
    GET    /api/collections      - 查看知识库集合列表
    GET    /api/stats            - 查看知识库统计信息
    DELETE /api/collections/{name} - 删除指定集合
    GET    /api/health           - 健康检查

认证: AUTH_ENABLED=true 时，除 /api/auth/login 与 /api/health 外，
      所有接口需携带 Authorization: Bearer <token> 或 X-API-Key: <key>
=============================================================================

使用方法（启动服务）:
    uvicorn app:app --host 0.0.0.0 --port 8000
"""

import json
import os
import time
from pathlib import Path
from typing import Any

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
)
from fastapi.responses import StreamingResponse

from src.document_loader import DocumentLoader
from src.rag import RAGPipeline
from src.text_processor import TextChunker
from src.utils.logger import setup_logger
from config.settings import settings

from src.conversations import ConversationManager, PGConversationManager
from src.auth import create_token, require_auth
from src.user_scope import user_personal_collection, is_admin_user, visible_collections

from .models import (
    AddMessageRequest,
    APIResponse,
    BatchDeleteConversationsRequest,
    ChangePasswordRequest,
    CreateCollectionRequest,
    CreateUserRequest,
    IngestConfirmRequest,
    LoginRequest,
    LoginResponseData,
    MessageFeedbackRequest,
    QueryRequest,
    QueryResponseData,
    QueryStats,
    RenameCollectionRequest,
    ResetPasswordRequest,
    SourceInfo,
    StreamQueryRequest,
    UpdateConversationTitleRequest,
    UpdateMessageRequest,
)

logger = setup_logger(__name__)

# ==============================================================================
# 依赖注入
# ==============================================================================
# 单例通过 request.app.state 持有（FastAPI lifespan 中创建），
# 避免模块级全局变量在多 worker 下的竞态问题。

def get_rag_pipeline(request: Request) -> RAGPipeline:
    """获取 RAG 管线单例（来自 app.state）"""
    return request.app.state.rag_pipeline


def get_document_loader(request: Request) -> DocumentLoader:
    """获取文档加载器单例（来自 app.state）"""
    return request.app.state.document_loader


def get_text_chunker(request: Request) -> TextChunker:
    """获取文本分块器单例（来自 app.state）"""
    return request.app.state.text_chunker


def get_conversation_mgr(request: Request) -> PGConversationManager:
    """获取对话管理器单例（来自 app.state）"""
    return request.app.state.conversation_mgr


async def _resolve_user_id(mgr: PGConversationManager, auth: dict) -> int | None:
    """
    从认证信息解析用户 ID，用于对话归属过滤。

    - 认证开启且为 JWT 用户 → 返回数据库用户 ID
    - 认证关闭（anonymous）或 API Key → 返回 None（不过滤，保持兼容）
    """
    username = auth.get("username")
    if not username or username == "anonymous":
        return None
    try:
        return await mgr.aget_user_id(username)
    except Exception:
        return None


# 多轮对话历史 token 预算（字符估算，约为 token 数 × 2）
HISTORY_TOKEN_BUDGET = 4000
# 每个历史消息的 token 估算（字符数 → token，中文约 1 字符 ≈ 0.6 token）
_CHARS_PER_TOKEN = 1.6


def _trim_history_by_budget(history: list[dict], budget: int = HISTORY_TOKEN_BUDGET) -> list[dict]:
    """
    按 token 预算动态截断历史消息，从旧到新丢弃，确保总长度不超过预算。

    保留尽量多的最近消息（多轮上下文），超出预算时优先丢弃最早的。
    """
    if not history:
        return history
    total_chars = sum(len(m.get("content", "")) for m in history)
    if total_chars / _CHARS_PER_TOKEN <= budget:
        return history
    # 从旧到新逐条丢弃，直到不超预算
    trimmed = list(history)
    for _ in range(len(history)):
        total_chars = sum(len(m.get("content", "")) for m in trimmed)
        if total_chars / _CHARS_PER_TOKEN <= budget:
            break
        trimmed.pop(0)  # 丢弃最早的一条
    return trimmed


async def _load_conversation_history(mgr, conversation_id, user_id) -> list[dict]:
    """
    加载对话历史并应用 token 预算截断。

    Returns:
        格式化的历史消息列表 [{"role": "user"|"assistant", "content": "..."}]
    """
    if conversation_id is None:
        return []
    try:
        messages = await mgr.aget_messages(conversation_id, user_id=user_id)
        history = [
            {"role": "assistant" if m["role"] == "ai" else "user", "content": m["content"]}
            for m in messages
        ]
        return _trim_history_by_budget(history)
    except Exception as e:
        logger.warning(f"加载对话历史失败: {e}")
        return []


async def _resolve_user_id_for_collections(rag, auth: dict) -> int | None:
    """从认证信息解析用户 ID，用于集合归属判断。"""
    username = auth.get("username")
    if not username or username == "anonymous":
        return None
    try:
        user = await rag.vector_store.aget_user_by_username(username)
        return user["id"] if user else None
    except Exception:
        return None


async def _collection_owner_map(rag) -> dict[str, int | None]:
    """构建 集合名 → owner_id 的映射。"""
    rows = await rag.vector_store.alist_collections_with_owner()
    return {r["name"]: r["owner_id"] for r in rows}


async def _visible_collections_for(rag, auth: dict) -> list[str]:
    """获取当前用户可见的集合名列表（含归属过滤）。"""
    all_raw = await rag.vector_store.alist_collections()
    owner_map = await _collection_owner_map(rag)
    user_id = await _resolve_user_id_for_collections(rag, auth)
    return visible_collections(auth, all_raw, user_id=user_id, owner_map=owner_map)


async def _multi_collection_search(
    rag,
    question: str,
    collections: list[str],
    k: int,
    filter_criteria: dict | None = None,
) -> list[dict]:
    """
    多集合并行检索：对每个可见集合分别检索，合并结果取全局最优候选。

    这是方案 B 的核心——不依赖 LLM 路由选集合，而是所有集合都检索，
    从全局取最相关的候选块（可跨集合），彻底避免"路由选错集合"。

    Returns:
        合并后的候选块列表（按分数降序），每项含 content, metadata, score, collection
    """
    candidate_k = max(k * 2, settings.RETRIEVAL_CANDIDATE_K)  # 每集合候选放宽
    all_candidates = []
    for coll in collections:
        try:
            await rag.vector_store.aswitch_collection(coll)
            search_method = getattr(rag.vector_store, "ahybrid_search", None)
            if search_method is not None and getattr(settings, "HYBRID_SEARCH_ENABLED", True):
                docs = await rag.vector_store.ahybrid_search(
                    query=question, k=candidate_k, filter=filter_criteria,
                )
            else:
                docs = await rag.vector_store.asimilarity_search(
                    query=question, k=candidate_k, filter=filter_criteria,
                )
            for d in docs:
                d.setdefault("metadata", {})
                d["metadata"]["_collection"] = coll  # 标记来源集合
            all_candidates.extend(docs)
            logger.info(f"集合 '{coll}' 检索到 {len(docs)} 个候选")
        except Exception as e:
            logger.warning(f"集合 '{coll}' 检索失败: {e}")

    # 按分数降序，取全局 top-k
    all_candidates.sort(key=lambda d: d.get("score", 0), reverse=True)
    return all_candidates[:k]


async def _resolve_upload_collection(rag, auth: dict, requested: str | None) -> str:
    """
    解析上传文档的目标集合。

    - 管理员：可指定任意集合；未指定 → 个人集合（知识库）
    - 普通用户：指定的集合必须是自己可见的（个人集合或自建集合），否则回退个人集合
    - 未指定：回退个人集合
    """
    username = auth.get("username", "anonymous")
    personal = user_personal_collection(username)

    if not requested:
        return personal

    if is_admin_user(auth):
        return requested

    # 普通用户：仅当请求的集合自己可见时才使用，否则回退个人集合
    visible = await _visible_collections_for(rag, auth)
    if requested in visible:
        return requested
    logger.info(f"用户 {username} 尝试上传到无权集合 '{requested}'，回退个人集合 '{personal}'")
    return personal


async def _log_query_audit(
    rag,
    *,
    username: str,
    question: str,
    answer: str = "",
    answer_type: str = "",
    collection: str | None = None,
    sources: list | None = None,
    conversation_id: int | None = None,
    k: int | None = None,
    concise: bool = False,
    from_cache: bool = False,
    latency_ms: int | None = None,
    usage: dict | None = None,
    status: str = "success",
    error_msg: str | None = None,
):
    """
    记录一次问答的审计日志。

    审计写入失败不影响主流程（helper 内部捕获异常）。
    """
    if not settings.AUDIT_ENABLED:
        return
    usage = usage or {}
    try:
        await rag.vector_store.aadd_query_audit(
            username=username,
            conversation_id=conversation_id,
            question=question,
            answer=answer,
            answer_type=answer_type,
            collection=collection,
            sources=sources or [],
            k=k,
            concise=concise,
            from_cache=from_cache,
            latency_ms=latency_ms,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            total_tokens=usage.get("total_tokens"),
            model=settings.LLM_MODEL_NAME,
            status=status,
            error_msg=error_msg,
        )
    except Exception as e:
        logger.warning(f"查询审计写入失败: {e}")


# ==============================================================================
# 路由定义
# ==============================================================================

router = APIRouter(prefix="/api", tags=["知识库问答"])


# ==============================================================================
# 认证
# ==============================================================================

@router.post("/auth/login", summary="登录获取 JWT 令牌", response_model=APIResponse)
async def login(
    body: LoginRequest,
    request: Request,
):
    """
    使用用户名密码登录，获取 JWT 令牌。

    令牌放入请求头 `Authorization: Bearer <token>`。
    admin 账号在服务启动时自动创建（用户名/密码来自 AUTH_ADMIN_* 配置）。
    """
    user_manager = request.app.state.user_manager
    user = await user_manager.authenticate(body.username, body.password)
    if user is None:
        raise HTTPException(
            status_code=401,
            detail="用户名或密码错误",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = create_token(user["username"])
    return APIResponse(
        code=0,
        message="success",
        data=LoginResponseData(
            token=token,
            token_type="bearer",
            expires_in=settings.JWT_EXPIRE_MINUTES * 60,
            username=user["username"],
            is_admin=bool(user.get("is_admin", False)),
        ),
    )


# ---------------------------------------------------------------
# 用户管理（仅管理员）
# ---------------------------------------------------------------

@router.get("/users", summary="获取用户列表（管理员）")
async def list_users(
    request: Request,
    auth: dict = Depends(require_auth),
):
    """获取所有用户列表（仅管理员）。"""
    if not settings.AUTH_ENABLED:
        pass  # 认证关闭：不做限制
    else:
        username = auth.get("username")
        if not username or username == "anonymous":
            raise HTTPException(status_code=401, detail="需要登录")
        um = request.app.state.user_manager
        user = await um._store.aget_user_by_username(username)
        if user is None or not user["is_admin"]:
            raise HTTPException(status_code=403, detail="仅管理员可执行此操作")

    um = request.app.state.user_manager
    users = await um.list_users()
    return {"code": 0, "message": "success", "data": users}


@router.post("/users", summary="创建用户（管理员）")
async def create_user(
    body: CreateUserRequest,
    request: Request,
    auth: dict = Depends(require_auth),
):
    """创建新用户（仅管理员）。"""
    if settings.AUTH_ENABLED:
        username = auth.get("username")
        if not username or username == "anonymous":
            raise HTTPException(status_code=401, detail="需要登录")
        um = request.app.state.user_manager
        admin = await um._store.aget_user_by_username(username)
        if admin is None or not admin["is_admin"]:
            raise HTTPException(status_code=403, detail="仅管理员可执行此操作")

    um = request.app.state.user_manager
    try:
        user = await um.create_user(
            username=body.username,
            password=body.password,
            display_name=body.display_name,
            is_admin=body.is_admin,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"创建用户失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"创建用户失败: {e}")

    return {"code": 0, "message": "success", "data": user}


@router.delete("/users/{user_id}", summary="删除用户（管理员）")
async def delete_user(
    user_id: int,
    request: Request,
    auth: dict = Depends(require_auth),
):
    """删除用户及其所有对话（仅管理员）。"""
    if settings.AUTH_ENABLED:
        username = auth.get("username")
        if not username or username == "anonymous":
            raise HTTPException(status_code=401, detail="需要登录")
        um = request.app.state.user_manager
        admin = await um._store.aget_user_by_username(username)
        if admin is None or not admin["is_admin"]:
            raise HTTPException(status_code=403, detail="仅管理员可执行此操作")
        # 防止删除自己
        if admin["id"] == user_id:
            raise HTTPException(status_code=400, detail="不能删除当前登录的管理员账号")

    um = request.app.state.user_manager
    ok = await um.delete_user(user_id)
    if not ok:
        raise HTTPException(status_code=404, detail="用户不存在")
    return {"code": 0, "message": "用户已删除", "data": {}}


# ---------------------------------------------------------------
# 修改密码
# ---------------------------------------------------------------

@router.post("/auth/change-password", summary="修改自己的密码")
async def change_password(
    body: ChangePasswordRequest,
    request: Request,
    auth: dict = Depends(require_auth),
):
    """当前登录用户修改自己的密码。"""
    username = auth.get("username")
    if not username or username == "anonymous":
        raise HTTPException(status_code=401, detail="需要登录")

    um = request.app.state.user_manager
    try:
        ok = await um.change_password(
            username=username,
            old_password=body.old_password,
            new_password=body.new_password,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if not ok:
        raise HTTPException(status_code=400, detail="原密码错误")
    return {"code": 0, "message": "密码已修改", "data": {}}


@router.post("/users/{user_id}/reset-password", summary="管理员重置用户密码")
async def reset_password(
    user_id: int,
    body: ResetPasswordRequest,
    request: Request,
    auth: dict = Depends(require_auth),
):
    """管理员重置指定用户的密码。"""
    if not settings.AUTH_ENABLED:
        pass
    else:
        username = auth.get("username")
        if not username or username == "anonymous":
            raise HTTPException(status_code=401, detail="需要登录")
        um = request.app.state.user_manager
        admin = await um._store.aget_user_by_username(username)
        if admin is None or not admin["is_admin"]:
            raise HTTPException(status_code=403, detail="仅管理员可执行此操作")

    um = request.app.state.user_manager
    try:
        ok = await um.reset_password(
            admin_username=auth.get("username", "admin"),
            target_user_id=user_id,
            new_password=body.new_password,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if not ok:
        raise HTTPException(status_code=404, detail="用户不存在")
    return {"code": 0, "message": "密码已重置", "data": {}}


# ---------------------------------------------------------------
# 自动路由：根据问题判断最相关的集合
# ---------------------------------------------------------------

async def _auto_route_collection(question: str, collections: list[str], rag) -> str | None:
    """
    根据用户问题自动选择最相关的知识库集合。

    如果无法确定，返回 None。
    """
    if len(collections) == 1:
        return collections[0]

    # 构造 prompt 让 LLM 判断
    collections_desc = "\n".join(f"- {c}" for c in collections)
    prompt = f"""根据用户的问题，从以下知识库集合中选择最相关的一个。
如果问题与任何一个集合都不相关，或者无法确定，请只返回「无法确定」四个字。
只返回集合名称或「无法确定」，不要其他文字。

可选的集合：
{collections_desc}

用户问题：{question}

最相关的集合名称："""

    try:
        answer = await rag.llm.agenerate(
            prompt=prompt,
            system_prompt="你是一个知识库路由助手。分析用户问题，选择最匹配的集合名称，只返回名称本身。如果无法确定，返回「无法确定」。",
            max_tokens=50,
            temperature=0.1,
        )
        answer = answer.strip().strip('"').strip("'").strip()
        # 验证返回的集合名是否有效
        if answer in collections:
            logger.info(f"自动路由: '{question[:50]}...' → '{answer}'")
            return answer
    except Exception as e:
        logger.warning(f"自动路由异常: {e}")

    logger.info(f"自动路由无法确定: '{question[:50]}...'")
    return None


# ---------------------------------------------------------------
# 健康检查
# ---------------------------------------------------------------

@router.get("/health", summary="健康检查")
async def health_check():
    """
    服务健康检查接口，返回服务运行状态。
    """
    return {
        "status": "ok",
        "service": "Enterprise Knowledge Base Q&A",
        "version": "1.0.0",
    }


# ---------------------------------------------------------------
# 知识库问答
# ---------------------------------------------------------------

@router.post("/query", summary="知识库问答", response_model=APIResponse)
async def query_knowledge_base(
    body: QueryRequest,
    rag: RAGPipeline = Depends(get_rag_pipeline),
    mgr: PGConversationManager = Depends(get_conversation_mgr),
    auth: dict = Depends(require_auth),
):
    """
    向知识库提问并获取回答。

    - collection: 可选，指定查询的集合名称。不传则自动路由到最相关的集合。
    - conversation_id: 可选，传入对话 ID 后使用该对话历史进行多轮问答，
      用户与 AI 的消息会自动写入该对话。
    """
    question = body.question.strip()
    k = body.k
    concise = body.concise
    filter_criteria = body.filter
    collection_name = body.collection
    conversation_id = body.conversation_id
    user_id = await _resolve_user_id(mgr, auth)
    start_time = time.time()  # 查询审计：记录起始时刻

    # ---- 加载多轮对话历史（按 token 预算动态截断） ----
    history = await _load_conversation_history(mgr, conversation_id, user_id)

    # ---- 自动路由：确定查询哪个集合（仅限当前用户可见且有内容的集合） ----
    all_collections = await _visible_collections_for(rag, auth)
    # 过滤掉空集合（chunk_count=0），避免自动路由到无内容集合导致误判"知识库为空"
    if len(all_collections) > 1:
        try:
            coll_stats = await rag.vector_store.aget_collection_stats(all_collections)
            non_empty = [s["name"] for s in coll_stats if s["chunk_count"] > 0]
            if non_empty:
                all_collections = non_empty
        except Exception as e:
            logger.warning(f"过滤空集合失败，使用全部可见集合: {e}")

    # 知识库为空时，直接走 RAG 管线（其内部会使用通用知识回答并提示上传）
    if not all_collections:
        try:
            result = await rag.aquery(
                question=question,
                k=k,
                stream=False,
                concise=concise,
                filter_criteria=filter_criteria,
                history=history or None,
            )
            answer = result["answer"]
            sources = [SourceInfo(**s) for s in result["sources"]]

            # ---- 持久化对话（如传入 conversation_id） ----
            if conversation_id is not None:
                try:
                    # 仅在用户有权限（对话归属当前用户或匿名共享）时写入
                    owned_msgs = await mgr.aget_messages(conversation_id, user_id=user_id)
                    if owned_msgs or user_id is None:
                        await mgr.aadd_message(conversation_id, "user", question)
                        await mgr.aadd_message(
                            conversation_id, "ai", answer,
                            sources=[s.model_dump() for s in sources],
                            answer_type=result.get("answer_type", "general"),
                        )
                except Exception as e:
                    logger.warning(f"持久化对话失败: {e}")

            # ---- 查询审计（空知识库分支） ----
            try:
                await _log_query_audit(
                    rag,
                    username=auth.get("username", "anonymous"),
                    question=question,
                    answer=answer,
                    answer_type=result.get("answer_type", "general"),
                    collection=None,
                    sources=[s.model_dump() for s in sources],
                    conversation_id=conversation_id,
                    k=k,
                    concise=concise,
                    from_cache=result.get("from_cache", False),
                    latency_ms=int((time.time() - start_time) * 1000),
                    usage=result.get("usage") or {},
                )
            except Exception:
                pass

            return APIResponse(
                code=0,
                message="success",
                data=QueryResponseData(
                    question=question,
                    answer=answer,
                    sources=sources,
                    answer_type=result.get("answer_type", "general"),
                    collection=None,
                    stats=QueryStats(
                        retrieved_chunks=len(result["context"]),
                        unique_sources=len(result["sources"]),
                    ),
                ),
            )
        except Exception as e:
            logger.error(f"问答接口异常: {e}", exc_info=True)
            # 失败也记审计
            try:
                await _log_query_audit(
                    rag,
                    username=auth.get("username", "anonymous"),
                    question=question,
                    conversation_id=conversation_id,
                    k=k,
                    concise=concise,
                    latency_ms=int((time.time() - start_time) * 1000),
                    status="failed",
                    error_msg=str(e)[:500],
                )
            except Exception:
                pass
            raise HTTPException(status_code=500, detail=f"问答服务异常: {e}")

    if collection_name and collection_name in all_collections:
        # 用户指定了集合 → 单集合检索
        target_collection = collection_name
        await rag.vector_store.aswitch_collection(target_collection)
        prefetched = None
    elif len(all_collections) == 1:
        # 只有一个集合，直接使用
        target_collection = all_collections[0]
        await rag.vector_store.aswitch_collection(target_collection)
        prefetched = None
    else:
        # 多集合 → 方案 B：所有集合并行检索，取全局最优（不依赖 LLM 路由）
        target_collection = None  # 跨集合，无单一目标集合
        candidate_k = (k or settings.RETRIEVAL_TOP_K) * 2
        prefetched = await _multi_collection_search(
            rag, question, all_collections, candidate_k, filter_criteria
        )
        if not prefetched:
            # 所有集合都检索不到，回退通用知识
            target_collection = None

    try:
        # 工具模式启用时，不传 prefetched_docs，让管线内部走 Agentic 检索
        # （工具模式闸门条件是 not prefetched_docs）
        tool_mode = (
            getattr(settings, "TOOL_CALLING_ENABLED", False)
            and getattr(rag, "tool_executor", None) is not None
        )
        result = await rag.aquery(
            question=question,
            k=k,
            stream=False,
            concise=concise,
            filter_criteria=filter_criteria,
            history=history or None,
            prefetched_docs=None if tool_mode else prefetched,
        )
        answer = result["answer"]
        sources = [SourceInfo(**s) for s in result["sources"]]

        # ---- 持久化对话（如传入 conversation_id） ----
        if conversation_id is not None:
            try:
                await mgr.aadd_message(conversation_id, "user", question)
                await mgr.aadd_message(
                    conversation_id, "ai", answer,
                    sources=[s.model_dump() for s in sources],
                    answer_type=result.get("answer_type", "general"),
                )
            except Exception as e:
                logger.warning(f"持久化对话失败: {e}")

        # ---- 查询审计（正常分支） ----
        try:
            await _log_query_audit(
                rag,
                username=auth.get("username", "anonymous"),
                question=question,
                answer=answer,
                answer_type=result.get("answer_type", "general"),
                collection=target_collection,
                sources=[s.model_dump() for s in sources],
                conversation_id=conversation_id,
                k=k,
                concise=concise,
                from_cache=result.get("from_cache", False),
                latency_ms=int((time.time() - start_time) * 1000),
                usage=result.get("usage") or {},
            )
        except Exception:
            pass

        return APIResponse(
            code=0,
            message="success",
            data=QueryResponseData(
                question=question,
                answer=answer,
                sources=sources,
                answer_type=result.get("answer_type", "general"),
                collection=target_collection,
                stats=QueryStats(
                    retrieved_chunks=len(result["context"]),
                    unique_sources=len(result["sources"]),
                ),
                related_questions=result.get("related_questions", []),
            ),
        )
    except Exception as e:
        logger.error(f"问答接口异常: {e}", exc_info=True)
        # 失败也记审计
        try:
            await _log_query_audit(
                rag,
                username=auth.get("username", "anonymous"),
                question=question,
                conversation_id=conversation_id,
                k=k,
                concise=concise,
                latency_ms=int((time.time() - start_time) * 1000),
                status="failed",
                error_msg=str(e)[:500],
            )
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=f"问答服务异常: {e}")


# ---------------------------------------------------------------
# 流式问答 (SSE)
# ---------------------------------------------------------------

@router.post("/query/stream", summary="流式知识库问答（SSE）")
async def query_knowledge_base_stream(
    body: StreamQueryRequest,
    rag: RAGPipeline = Depends(get_rag_pipeline),
    mgr: PGConversationManager = Depends(get_conversation_mgr),
    auth: dict = Depends(require_auth),
):
    """
    流式知识库问答，使用 Server-Sent Events (SSE) 协议。

    - conversation_id: 可选，传入对话 ID 后使用该对话历史进行多轮问答。
    """
    question = body.question.strip()
    k = body.k
    concise = body.concise
    collection_name = body.collection
    conversation_id = body.conversation_id
    user_id = await _resolve_user_id(mgr, auth)
    start_time = time.time()  # 查询审计：记录起始时刻
    # 流式请求体没有 filter 字段，多集合检索的 filter_criteria 固定为 None
    filter_criteria = None

    # ---- 加载多轮对话历史（按 token 预算动态截断） ----
    history = await _load_conversation_history(mgr, conversation_id, user_id)

    # ---- 自动路由：确定查询哪个集合（仅限当前用户可见且有内容的集合） ----
    all_collections = await _visible_collections_for(rag, auth)
    # 过滤掉空集合（chunk_count=0），避免自动路由到无内容集合导致误判"知识库为空"
    if len(all_collections) > 1:
        try:
            coll_stats = await rag.vector_store.aget_collection_stats(all_collections)
            non_empty = [s["name"] for s in coll_stats if s["chunk_count"] > 0]
            if non_empty:
                all_collections = non_empty
        except Exception as e:
            logger.warning(f"过滤空集合失败，使用全部可见集合: {e}")

    # 知识库为空时，直接走 RAG 管线（其内部会使用通用知识回答并提示上传）
    if not all_collections:
        async def empty_kb_generator():
            """知识库为空时的 SSE 生成器：LLM 通用知识回答 + 上传提示"""
            try:
                result = await rag.astream_query(
                    question=question,
                    k=k,
                    concise=concise,
                    history=history or None,
                )
                sources = result["sources"]
                answer_type = result.get("answer_type", "general")
                answer_generator = result["answer"]

                yield f"data: {json.dumps({'type': 'meta', 'sources': sources, 'answer_type': answer_type}, ensure_ascii=False)}\n\n"

                full_answer = ""
                async for text_chunk in answer_generator:
                    if text_chunk:
                        full_answer += text_chunk
                        yield f"data: {json.dumps({'type': 'chunk', 'data': text_chunk}, ensure_ascii=False)}\n\n"

                yield f"data: {json.dumps({'type': 'done', 'data': full_answer}, ensure_ascii=False)}\n\n"

                # 持久化对话（如传入 conversation_id，仅在有权访问时写入）
                if conversation_id is not None:
                    try:
                        owned_msgs = await mgr.aget_messages(conversation_id, user_id=user_id)
                        if owned_msgs or user_id is None:
                            await mgr.aadd_message(conversation_id, "user", question)
                            await mgr.aadd_message(
                                conversation_id, "ai", full_answer,
                                sources=sources,
                                answer_type=answer_type,
                            )
                    except Exception as e:
                        logger.warning(f"持久化对话失败: {e}")

                # ---- 查询审计（流式空知识库分支） ----
                try:
                    await _log_query_audit(
                        rag,
                        username=auth.get("username", "anonymous"),
                        question=question,
                        answer=full_answer,
                        answer_type=answer_type,
                        collection=None,
                        sources=sources or [],
                        conversation_id=conversation_id,
                        k=k,
                        concise=concise,
                        from_cache=result.get("from_cache", False),
                        latency_ms=int((time.time() - start_time) * 1000),
                        usage=result.get("usage") or {},
                    )
                except Exception:
                    pass

            except Exception as e:
                logger.error(f"流式问答异常: {e}", exc_info=True)
                # 失败也记审计
                try:
                    await _log_query_audit(
                        rag,
                        username=auth.get("username", "anonymous"),
                        question=question,
                        conversation_id=conversation_id,
                        k=k,
                        concise=concise,
                        latency_ms=int((time.time() - start_time) * 1000),
                        status="failed",
                        error_msg=str(e)[:500],
                    )
                except Exception:
                    pass
                yield f"data: {json.dumps({'type': 'error', 'data': str(e)}, ensure_ascii=False)}\n\n"

        return StreamingResponse(empty_kb_generator(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"})

    prefetched = None
    if collection_name and collection_name in all_collections:
        # 用户指定了集合 → 单集合检索
        target_collection = collection_name
        await rag.vector_store.aswitch_collection(target_collection)
    elif len(all_collections) == 1:
        # 只有一个集合，直接使用
        target_collection = all_collections[0]
        await rag.vector_store.aswitch_collection(target_collection)
    else:
        # 多集合 → 方案 B：所有集合并行检索（不依赖 LLM 路由）
        target_collection = None
        candidate_k = (k or settings.RETRIEVAL_TOP_K) * 2
        prefetched = await _multi_collection_search(
            rag, question, all_collections, candidate_k, filter_criteria
        )

    async def event_generator():
        """SSE 事件生成器"""
        try:
            # 工具模式启用时，不传 prefetched_docs，让管线内部走 Agentic 检索
            tool_mode = (
                getattr(settings, "TOOL_CALLING_ENABLED", False)
                and getattr(rag, "tool_executor", None) is not None
            )
            result = await rag.astream_query(
                question=question,
                k=k,
                concise=concise,
                history=history or None,
                prefetched_docs=None if tool_mode else prefetched,
            )
            sources = result["sources"]
            answer_type = result.get("answer_type", "general")
            answer_generator = result["answer"]

            # 发送来源信息和回答模式
            yield f"data: {json.dumps({'type': 'meta', 'sources': sources, 'answer_type': answer_type}, ensure_ascii=False)}\n\n"

            # 流式发送回答片段
            full_answer = ""
            async for text_chunk in answer_generator:
                if text_chunk:
                    full_answer += text_chunk
                    yield f"data: {json.dumps({'type': 'chunk', 'data': text_chunk}, ensure_ascii=False)}\n\n"

            # 发送完成信号
            yield f"data: {json.dumps({'type': 'done', 'data': full_answer}, ensure_ascii=False)}\n\n"

            # ---- 持久化对话（如传入 conversation_id，仅在有权访问时写入） ----
            if conversation_id is not None:
                try:
                    owned_msgs = await mgr.aget_messages(conversation_id, user_id=user_id)
                    if owned_msgs or user_id is None:
                        await mgr.aadd_message(conversation_id, "user", question)
                        await mgr.aadd_message(
                            conversation_id, "ai", full_answer,
                            sources=sources,
                            answer_type=answer_type,
                        )
                except Exception as e:
                    logger.warning(f"持久化对话失败: {e}")

            # ---- 查询审计（流式正常分支） ----
            try:
                await _log_query_audit(
                    rag,
                    username=auth.get("username", "anonymous"),
                    question=question,
                    answer=full_answer,
                    answer_type=answer_type,
                    collection=target_collection,
                    sources=sources or [],
                    conversation_id=conversation_id,
                    k=k,
                    concise=concise,
                    from_cache=result.get("from_cache", False),
                    latency_ms=int((time.time() - start_time) * 1000),
                    usage=result.get("usage") or {},
                )
            except Exception:
                pass

        except Exception as e:
            logger.error(f"流式问答异常: {e}", exc_info=True)
            # 失败也记审计
            try:
                await _log_query_audit(
                    rag,
                    username=auth.get("username", "anonymous"),
                    question=question,
                    conversation_id=conversation_id,
                    k=k,
                    concise=concise,
                    latency_ms=int((time.time() - start_time) * 1000),
                    status="failed",
                    error_msg=str(e)[:500],
                )
            except Exception:
                pass
            yield f"data: {json.dumps({'type': 'error', 'data': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------
# 文档导入
# ---------------------------------------------------------------

@router.get("/upload/token", summary="获取 OSS 直传签名 URL")
async def get_upload_token(
    filename: str = Query(..., description="文件名"),
    rag: RAGPipeline = Depends(get_rag_pipeline),
    auth: dict = Depends(require_auth),
):
    """
    获取 OSS 直传的预签名 URL，前端直接上传到 OSS 后可获得真实进度。

    需要登录认证；签名仅对当前登录用户有效。

    使用流程：
        1. 前端调用此接口获取 upload_url
        2. 前端用 PUT 方法直接上传文件到 upload_url（可监听 XHR progress）
        3. 上传完成后，调用 POST /api/ingest/confirm 确认并解析
    """
    from src.storage import get_storage, OSSStorage
    storage = get_storage()

    if not isinstance(storage, OSSStorage):
        raise HTTPException(status_code=400, detail={
            "error_type": "oss_not_configured",
            "message": "当前存储后端不是 OSS，不支持直传",
            "suggestion": "请将 STORAGE_BACKEND 配置为 oss",
        })

    try:
        upload_info = storage.generate_upload_url(filename)
        return {
            "code": 0,
            "message": "success",
            "data": upload_info,
        }
    except Exception as e:
        logger.error(f"生成上传 token 失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"生成上传 token 失败: {e}")


@router.post("/ingest/confirm", summary="确认 OSS 直传并导入文档")
async def confirm_upload(
    body: IngestConfirmRequest,
    rag: RAGPipeline = Depends(get_rag_pipeline),
    loader: DocumentLoader = Depends(get_document_loader),
    chunker: TextChunker = Depends(get_text_chunker),
    auth: dict = Depends(require_auth),
):
    """
    确认 OSS 直传完成，从 OSS 下载文件并导入知识库。

    文档归入当前用户的个人集合；管理员可指定任意集合。
    """
    object_key = body.object_key.strip()
    filename = (body.filename or "").strip() or object_key.split("/")[-1]
    collection_name = body.collection

    # 切换到目标集合：管理员可指定任意集合，普通用户仅能传到自己可见的集合
    target_collection = await _resolve_upload_collection(rag, auth, collection_name)
    await rag.vector_store.aswitch_collection(target_collection)

    # 从 OSS 下载文件
    from src.storage import get_storage, OSSStorage
    import tempfile

    storage = get_storage()
    if not isinstance(storage, OSSStorage):
        raise HTTPException(status_code=400, detail="当前存储后端不是 OSS")

    try:
        content = storage.get_object(object_key)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"从 OSS 下载文件失败: {e}")

    file_size = len(content)
    file_ext = Path(filename).suffix.lower()
    logger.info(f"OSS 文件已下载: {object_key} ({file_size} bytes)")

    # 保存到临时文件并解析
    tmp_file = tempfile.NamedTemporaryFile(suffix=file_ext, delete=False)
    tmp_path = tmp_file.name
    tmp_file.write(content)
    tmp_file.close()

    try:
        # 计算哈希
        import hashlib
        content_hash = hashlib.sha256(content).hexdigest()

        # 查重（内容相同）
        existing = await rag.vector_store.afind_document_by_hash(content_hash)
        if existing is not None:
            try:
                existing_chunks = await rag.vector_store.acount_document_chunks(existing["id"])
            except AttributeError:
                existing_chunks = 1
                logger.warning("acount_document_chunks 不可用，按正常重复处理")

            if existing_chunks > 0:
                os.unlink(tmp_path)
                raise HTTPException(status_code=409, detail={
                    "error_type": "duplicate_document",
                    "message": f"文件 '{filename}' 与已存在的文档内容重复",
                    "suggestion": f"该文档已在 {existing['created_at'][:10]} 导入过",
                })
            else:
                # 僵尸记录：无分块内容，清理后继续本次导入
                logger.warning(
                    f"检测到僵尸文档记录 id={existing['id']}（{existing['filename']}）"
                    f"无分块内容，自动清理后重新导入"
                )
                try:
                    await rag.vector_store.adelete_document(existing["id"], delete_storage=True)
                except Exception as e:
                    logger.warning(f"清理僵尸记录失败（继续导入）: {e}")

        # 查重（文件名相同）
        existing_name = await rag.vector_store.afind_document_by_filename(filename)
        if existing_name is not None:
            os.unlink(tmp_path)
            raise HTTPException(status_code=409, detail={
                "error_type": "duplicate_filename",
                "message": f"文件名 '{filename}' 已存在",
                "suggestion": "请使用不同的文件名上传",
            })

        # 解析文档
        raw_docs = loader.load_file(tmp_path)
        if not raw_docs:
            os.unlink(tmp_path)
            raise HTTPException(status_code=400, detail={
                "error_type": "empty_content",
                "message": f"文件 '{filename}' 解析后未提取到任何文本",
                "suggestion": "可能是扫描件，请安装 OCR 引擎",
            })

        # 修正 metadata 文件名
        for doc in raw_docs:
            if "metadata" in doc:
                doc["metadata"]["filename"] = filename
                doc["metadata"]["source"] = object_key

        # 记录文件元数据
        try:
            doc_id = await rag.vector_store.aadd_document_record(
                filename=filename,
                file_type=file_ext.lstrip("."),
                file_size=file_size,
                content_hash=content_hash,
                storage_path=object_key,
                storage_backend="oss",
            )
        except Exception as e:
            logger.warning(f"记录文件元数据失败: {e}")
            doc_id = None

        # 分块和导入
        chunked_docs = chunker.split_documents(raw_docs)
        count = await rag.vector_store.aadd_documents(chunked_docs, document_id=doc_id)

        return {
            "code": 0,
            "message": "success",
            "data": {
                "filename": filename,
                "chunks_added": count,
                "file_size": file_size,
                "storage": "oss",
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"文档导入异常: {e}", exc_info=True)
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=f"文档导入失败: {e}")


@router.post("/ingest", summary="上传并导入文档到知识库（传统方式）")
async def ingest_document(
    file: UploadFile = File(...),
    filename: str = Form(None),
    collection: str = Form(None),
    rag: RAGPipeline = Depends(get_rag_pipeline),
    loader: DocumentLoader = Depends(get_document_loader),
    chunker: TextChunker = Depends(get_text_chunker),
    auth: dict = Depends(require_auth),
):
    """
    上传文档文件并将其导入知识库。

    文档归入当前用户的个人集合；管理员归入"知识库"集合。
    collection 参数仅允许指定当前用户可见的集合，否则忽略并归入个人集合。

    支持的格式: pdf, docx, doc, docm, pptx, xlsx, xls,
                 jpg, jpeg, png, bmp, tiff, txt, md, json 等
    """
    # ================================================================
    # 前置校验
    # ================================================================

    # 1. 检查是否有文件
    if not file.filename:
        raise HTTPException(status_code=400, detail={
            "error_type": "no_file",
            "message": "没有选择文件",
            "suggestion": "请选择一个文件后再上传",
        })

    # 2. 检查文件扩展名
    allowed_extensions = set(loader.SUPPORTED_EXTENSIONS.keys())
    file_ext = Path(file.filename).suffix.lower() if file.filename else ""
    if not file_ext:
        raise HTTPException(status_code=400, detail={
            "error_type": "invalid_extension",
            "message": f"文件 '{file.filename}' 没有扩展名，无法识别文件类型",
            "suggestion": "请确保文件有正确的扩展名（如 .pdf, .docx, .txt）",
        })
    if file_ext not in allowed_extensions:
        # 按类型分组展示支持的格式
        type_groups = {
            "文档": [".pdf", ".docx", ".doc", ".docm", ".pptx"],
            "表格": [".xlsx", ".xls"],
            "图片（需 OCR）": [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"],
            "文本/代码": [".txt", ".md", ".py", ".yaml", ".yml", ".html", ".htm", ".xml", ".csv", ".json"],
            "WPS": [".wps", ".et"],
        }
        support_list = "\n".join(f"  {k}: {', '.join(v)}" for k, v in type_groups.items())
        raise HTTPException(status_code=400, detail={
            "error_type": "unsupported_format",
            "message": f"不支持 '{file_ext}' 格式",
            "suggestion": f"支持的格式:\n{support_list}\n\n请将文件转换为支持的格式后重试",
        })

    # 3. 检查文件大小
    HUGE_FILE_LIMIT = 100 * 1024 * 1024
    LARGE_FILE_LIMIT = 50 * 1024 * 1024
    content = await file.read()
    file_size = len(content)

    if file_size == 0:
        raise HTTPException(status_code=400, detail={
            "error_type": "empty_file",
            "message": f"文件 '{file.filename}' 为空",
            "suggestion": "请检查文件是否损坏，或重新导出后上传",
        })

    if file_size > HUGE_FILE_LIMIT:
        raise HTTPException(status_code=400, detail={
            "error_type": "file_too_large",
            "message": f"文件过大 ({file_size / 1024 / 1024:.1f}MB)",
            "suggestion": f"最大支持 {HUGE_FILE_LIMIT // 1024 // 1024}MB 的文件，当前文件 {file_size / 1024 / 1024:.1f}MB。请压缩或拆分后上传",
        })

    # ================================================================
    # 保存并解析
    # ================================================================

    import tempfile

    # 保存到临时文件用于解析
    tmp_file = tempfile.NamedTemporaryFile(suffix=file_ext, delete=False)
    tmp_path = tmp_file.name
    tmp_file.write(content)
    tmp_file.close()

    try:
        logger.info(f"文件已接收: {file.filename} ({file_size} bytes)")

        # ---- 计算内容哈希，检查是否重复 ----
        import hashlib
        content_hash = hashlib.sha256(content).hexdigest()
        logger.info(f"内容哈希: {content_hash[:16]}...")

        try:
            existing = await rag.vector_store.afind_document_by_hash(content_hash)
            logger.info(f"查重结果: {existing}")
        except AttributeError as e:
            logger.warning(f"查重方法不可用: {e}")
            existing = None

        if existing is not None:
            # 检查该文档是否真的完成了分块入库
            try:
                existing_chunks = await rag.vector_store.acount_document_chunks(existing["id"])
            except AttributeError:
                existing_chunks = 1  # 无法判断时视为正常记录，走原有 409 逻辑
                logger.warning("acount_document_chunks 不可用，按正常重复处理")

            if existing_chunks > 0:
                # 正常重复：文档已完整入库
                os.unlink(tmp_path)
                raise HTTPException(status_code=409, detail={
                    "error_type": "duplicate_document",
                    "message": f"文件 '{file.filename}' 与已存在的文档内容重复",
                    "suggestion": (
                        f"该文档已在 {existing['created_at'][:10]} 导入过（"
                        f"原文件名: {existing['filename']}，{existing['file_size']} bytes）。\n"
                        f"如需重新导入，请先删除集合后重试。"
                    ),
                })
            else:
                # 僵尸记录：只有元数据、无分块（上次导入中断），清理后继续本次导入
                logger.warning(
                    f"检测到僵尸文档记录 id={existing['id']}（{existing['filename']}）"
                    f"无分块内容，自动清理后重新导入"
                )
                try:
                    await rag.vector_store.adelete_document(existing["id"], delete_storage=True)
                except Exception as e:
                    logger.warning(f"清理僵尸记录失败（继续导入）: {e}")

        # ---- 检查文件名是否重复 ----
        # 确定最终使用的文件名（优先使用前端传的自定义文件名）
        final_filename = (filename or file.filename).strip()
        if not final_filename:
            final_filename = file.filename

        # 切换到目标集合：管理员可指定任意集合，普通用户仅能传到自己可见的集合
        target_collection = await _resolve_upload_collection(rag, auth, collection)
        await rag.vector_store.aswitch_collection(target_collection)

        try:
            existing_name = await rag.vector_store.afind_document_by_filename(final_filename)
        except AttributeError:
            existing_name = None

        if existing_name is not None:
            logger.info(f"文件名 '{final_filename}' 已存在，执行覆盖更新 (旧文档 id={existing_name['id']})")
            old_doc_id = existing_name["id"]
            try:
                await rag.vector_store.adelete_document(old_doc_id, delete_storage=True)
                logger.info(f"旧文档已删除 (id={old_doc_id})")
            except Exception as e:
                logger.warning(f"删除旧文档失败（继续导入）: {e}")
            from src.cache import qa_cache
            qa_cache.invalidate_by_tags([f"doc:{final_filename}"])
            logger.info(f"已清除旧文档缓存: {final_filename}")

        # 解析文档
        raw_docs = loader.load_file(tmp_path)

        # 修正 metadata 中的文件名为原始文件名（解析时用的是临时文件名）
        for doc in raw_docs:
            if "metadata" in doc:
                doc["metadata"]["filename"] = final_filename
                doc["metadata"]["source"] = final_filename

        # 检查解析结果是否为空
        if not raw_docs:
            os.unlink(tmp_path)
            raise HTTPException(status_code=400, detail={
                "error_type": "empty_content",
                "message": f"文件 '{file.filename}' 解析后未提取到任何文本",
                "suggestion": (
                    "可能原因：\n"
                    "  - 扫描版 PDF/图片（需安装 OCR 引擎：pip install paddleocr）\n"
                    "  - 文档内容为纯图片/SmartArt\n"
                    "  - 文件已损坏"
                ),
            })

        # 上传到存储后端（OSS 或本地）
        from src.storage import get_storage
        storage = get_storage()
        storage_path = storage.save(file.filename, content)

        # 记录文件元数据到 PostgreSQL
        try:
            doc_id = await rag.vector_store.aadd_document_record(
                filename=final_filename,
                file_type=file_ext.lstrip("."),
                file_size=file_size,
                content_hash=content_hash,
                storage_path=storage_path,
                storage_backend=settings.STORAGE_BACKEND,
            )
        except Exception as e:
            logger.warning(f"记录文件元数据失败（不影响解析）: {e}")
            doc_id = None

        # 检查是否有 warnings（如嵌入对象、SmartArt 等）
        warnings = []
        for doc in raw_docs:
            meta_warnings = doc.get("metadata", {}).get("warnings")
            if meta_warnings:
                warnings.extend(meta_warnings)

        # 文本分块
        chunked_docs = chunker.split_documents(raw_docs)

        # 导入知识库（关联 OSS 文件 ID）
        count = await rag.vector_store.aadd_documents(
            chunked_docs,
            document_id=doc_id,
        )

        # 文档更新后，标记引用该文件的旧回答为已过期
        try:
            await rag.vector_store.amark_stale_by_filenames([final_filename])
        except Exception as e:
            logger.warning(f"标记历史回答过期失败: {e}")

        response_data = {
            "filename": file.filename,
            "chunks_added": count,
            "file_size": file_size,
            "storage": settings.STORAGE_BACKEND,
        }
        if warnings:
            response_data["warnings"] = list(set(warnings))

        return {
            "code": 0,
            "message": "success",
            "data": response_data,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"文档导入异常: {e}", exc_info=True)
        # 清理临时文件
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

        error_msg = str(e)
        # 根据错误信息判断类型
        if "OCR" in error_msg and "安装" in error_msg:
            detail = {
                "error_type": "ocr_not_installed",
                "message": "需要 OCR 引擎来识别图片/扫描件中的文字",
                "suggestion": "请安装 OCR 引擎：pip install paddlepaddle paddleocr",
            }
        elif "LibreOffice" in error_msg:
            detail = {
                "error_type": "libreoffice_not_found",
                "message": f"文件 '{file.filename}' 需要 LibreOffice 转换",
                "suggestion": "请安装 LibreOffice (https://www.libreoffice.org/)，或将文件另存为支持的格式后重试",
            }
        elif "密码" in error_msg or "加密" in error_msg:
            detail = {
                "error_type": "encrypted",
                "message": f"文件 '{file.filename}' 已加密",
                "suggestion": "请先解密文件，或移除此密码保护后重新上传",
            }
        else:
            detail = {
                "error_type": "parse_error",
                "message": f"文件解析失败: {error_msg[:200]}",
                "suggestion": "请检查文件是否损坏、格式是否正确，或转换为其他格式后重试",
            }

        raise HTTPException(status_code=422, detail=detail)


# ---------------------------------------------------------------
# 异步文档导入（大文件任务队列）
# ---------------------------------------------------------------

@router.post("/ingest/async", summary="异步上传并导入大文档")
async def ingest_document_async(
    request: Request,
    file: UploadFile = File(...),
    filename: str = Form(None),
    collection: str = Form(None),
    loader: DocumentLoader = Depends(get_document_loader),
    chunker: TextChunker = Depends(get_text_chunker),
    auth: dict = Depends(require_auth),
):
    """
    异步上传大文档（>10MB），立即返回任务 ID，后台解析入库。

    文档归入当前用户的个人集合；管理员可指定任意集合。

    前端轮询 GET /api/ingest/tasks/{task_id} 获取进度。
    小文件仍用同步的 POST /api/ingest。
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail={"error_type": "no_file", "message": "没有选择文件"})

    # 检查扩展名
    file_ext = Path(file.filename).suffix.lower()
    if not file_ext:
        raise HTTPException(status_code=400, detail={"error_type": "invalid_extension", "message": "文件没有扩展名"})

    content = await file.read()
    file_size = len(content)

    if file_size == 0:
        raise HTTPException(status_code=400, detail={"error_type": "empty_file", "message": "文件为空"})

    final_filename = (filename or file.filename).strip() or file.filename

    # 目标集合：管理员可指定任意集合，普通用户仅能传到自己可见的集合
    rag = get_rag_pipeline(request)
    target_collection = await _resolve_upload_collection(rag, auth, collection)

    task_mgr = request.app.state.ingest_task_mgr

    # 提交异步任务
    result = await task_mgr.submit(
        filename=final_filename,
        content=content,
        collection=target_collection,
        rag=rag,
        loader=loader,
        chunker=chunker,
        store=rag.vector_store,
    )

    return {
        "code": 0,
        "message": "success",
        "data": result,
    }


@router.get("/ingest/tasks/{task_id}", summary="查询异步导入任务状态")
async def get_ingest_task(
    task_id: str,
    request: Request,
    _: dict = Depends(require_auth),
):
    """
    查询异步导入任务状态，供前端轮询。

    状态: pending（排队）→ processing（处理中）→ success / failed
    progress: 0~100
    """
    task_mgr = request.app.state.ingest_task_mgr
    rag = get_rag_pipeline(request)

    task = await task_mgr.get_status(task_id, rag.vector_store)
    if task is None:
        raise HTTPException(status_code=404, detail="任务不存在")

    return {
        "code": 0,
        "message": "success",
        "data": task,
    }


@router.get("/ingest/tasks", summary="列出进行中的异步导入任务")
async def list_ingest_tasks(
    request: Request,
    _: dict = Depends(require_auth),
):
    """列出所有进行中的异步导入任务。"""
    task_mgr = request.app.state.ingest_task_mgr
    rag = get_rag_pipeline(request)

    tasks = await task_mgr.list_active(rag.vector_store)
    return {
        "code": 0,
        "message": "success",
        "data": tasks,
    }


# ---------------------------------------------------------------
# 文档管理（删除）
# ---------------------------------------------------------------

@router.get("/collections/{collection_name}/documents", summary="获取集合中的文档列表")
async def list_documents(
    collection_name: str,
    rag: RAGPipeline = Depends(get_rag_pipeline),
    auth: dict = Depends(require_auth),
):
    """获取指定集合中的所有文档列表（仅限当前用户可见的集合）"""
    try:
        if collection_name not in await _visible_collections_for(rag, auth):
            raise HTTPException(status_code=403, detail="无权访问该集合")
        await rag.vector_store.aswitch_collection(collection_name)
        docs = await rag.vector_store.alist_documents()
        return {"code": 0, "message": "success", "data": docs}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取文档列表异常: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"获取文档列表失败: {e}")


@router.delete("/documents/{doc_id}", summary="删除文档（含 OSS 原文件）")
async def delete_document(
    doc_id: int,
    rag: RAGPipeline = Depends(get_rag_pipeline),
    auth: dict = Depends(require_auth),
):
    """
    删除指定文档及其所有分块，同时删除 OSS/本地存储中的原始文件。

    仅能删除当前用户可见集合内的文档（管理员可删全部）。

    删除内容：
        - PostgreSQL chunks 表中的关联分块
        - PostgreSQL documents 表中的记录
        - OSS/本地存储中的原始文件
    """
    try:
        # 先校验文档归属的集合是否当前用户可见
        doc = await rag.vector_store.afind_document_by_id(doc_id)
        if doc is None:
            raise HTTPException(status_code=404, detail="文档不存在")
        visible = await _visible_collections_for(rag, auth)
        # 找到该文档所在集合
        doc_collection = doc.get("collection_name")
        if not is_admin_user(auth) and doc_collection not in visible:
            raise HTTPException(status_code=403, detail="无权删除该文档")

        result = await rag.vector_store.adelete_document(doc_id, delete_storage=True)
        if result is None:
            raise HTTPException(status_code=404, detail="文档不存在")

        # 按被删除文档的文件名精确失效相关缓存
        from src.cache import qa_cache
        if result.get("filename"):
            qa_cache.invalidate_by_tags([f"doc:{result['filename']}"])
            # 标记引用该文档的历史回答为已过期
            try:
                await rag.vector_store.amark_stale_by_filenames([result["filename"]])
            except Exception as e:
                logger.warning(f"标记历史回答过期失败: {e}")
        else:
            qa_cache.clear()

        return {
            "code": 0,
            "message": "success",
            "data": {
                "filename": result["filename"],
                "deleted_chunks": result["deleted_chunks"],
                "file_size": result["file_size"],
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除文档异常: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"删除文档失败: {e}")


# ---------------------------------------------------------------
# 知识库管理
# ---------------------------------------------------------------

@router.get("/stats", summary="查看知识库统计信息")
async def get_stats(
    rag: RAGPipeline = Depends(get_rag_pipeline),
    auth: dict = Depends(require_auth),
):
    """获取当前用户可见范围内的知识库统计信息（管理员为全局）"""
    # 一次 SQL 聚合用户可见集合的统计，避免逐集合循环查询
    visible = await _visible_collections_for(rag, auth)
    stats_rows = await rag.vector_store.aget_collection_stats(visible)
    total_chunks = sum(r["chunk_count"] for r in stats_rows)

    stats = await rag.aget_knowledge_base_stats()
    stats["total_chunks"] = total_chunks
    stats["collections"] = visible
    return {
        "code": 0,
        "message": "success",
        "data": stats,
    }


@router.get("/collections", summary="查看知识库集合列表")
async def list_collections(
    rag: RAGPipeline = Depends(get_rag_pipeline),
    auth: dict = Depends(require_auth),
):
    """列出当前用户可见的知识库集合及其文档数量（管理员可见全部）"""
    names = await _visible_collections_for(rag, auth)
    # 一次 SQL 聚合所有可见集合的统计
    stats_rows = await rag.vector_store.aget_collection_stats(names)
    detailed = [
        {
            "name": r["name"],
            "chunk_count": r["chunk_count"],
            "document_count": r["document_count"],
        }
        for r in stats_rows
    ]
    return {
        "code": 0,
        "message": "success",
        "data": {
            "collections": detailed,
            "total": len(detailed),
        },
    }


@router.post("/collections", summary="创建新集合")
async def create_collection(
    body: CreateCollectionRequest,
    rag: RAGPipeline = Depends(get_rag_pipeline),
    auth: dict = Depends(require_auth),
):
    """创建新的知识库集合（所有用户可创建，归属当前用户）"""
    name = body.name.strip()

    if name in await rag.vector_store.alist_collections():
        raise HTTPException(status_code=409, detail=f"集合 '{name}' 已存在")

    # 归属：当前用户的 user_id（匿名/认证关闭时为 None，视为系统集合）
    owner_id = await _resolve_user_id_for_collections(rag, auth)
    # 创建集合（切换过去即自动创建），保持当前集合为目标集合
    # 注意：不要在此处切回 DEFAULT_COLLECTION，否则会因 aswitch_collection
    # 的"切换即按需建集合"行为，自动创建一个默认集合（如 knowledge_base）
    await rag.vector_store.aswitch_collection(name)
    # 若集合是新建的，补充 owner_id
    await rag.vector_store.aset_collection_owner(name, owner_id)

    return {
        "code": 0,
        "message": "success",
        "data": {"name": name},
    }


@router.put("/collections/{collection_name}", summary="重命名集合")
async def rename_collection(
    collection_name: str,
    body: RenameCollectionRequest,
    rag: RAGPipeline = Depends(get_rag_pipeline),
    auth: dict = Depends(require_auth),
):
    """重命名指定集合（集合归属者或管理员）"""
    owner_id = await _resolve_user_id_for_collections(rag, auth)
    coll_owner = await rag.vector_store.aget_collection_owner(collection_name)
    if not is_admin_user(auth) and coll_owner != owner_id:
        raise HTTPException(status_code=403, detail="无权重命名该集合")
    new_name = body.name.strip()

    if new_name in await rag.vector_store.alist_collections():
        raise HTTPException(status_code=409, detail=f"集合 '{new_name}' 已存在")

    ok = await rag.vector_store.arenmae_collection(collection_name, new_name)
    if not ok:
        raise HTTPException(status_code=404, detail=f"集合 '{collection_name}' 不存在")

    return {"code": 0, "message": "success", "data": {"old_name": collection_name, "new_name": new_name}}


@router.delete("/collections/{collection_name}", summary="删除指定集合（含 OSS 文件）")
async def delete_collection(
    collection_name: str,
    rag: RAGPipeline = Depends(get_rag_pipeline),
    auth: dict = Depends(require_auth),
):
    """删除指定的知识库集合（含所有文档块和 OSS 原文件，集合归属者或管理员）"""
    owner_id = await _resolve_user_id_for_collections(rag, auth)
    coll_owner = await rag.vector_store.aget_collection_owner(collection_name)
    if not is_admin_user(auth) and coll_owner != owner_id:
        raise HTTPException(status_code=403, detail="无权删除该集合")
    try:
        await rag.vector_store.aswitch_collection(collection_name)
        await rag.vector_store.adelete_collection()
        # 清空问答缓存
        from src.cache import qa_cache
        qa_cache.clear()
        # 切回一个存在的集合，防止 _ensure_collection 重建已删除的集合
        remaining = await rag.vector_store.alist_collections()
        if remaining:
            await rag.vector_store.aswitch_collection(remaining[0])
        return {
            "code": 0,
            "message": f"集合 '{collection_name}' 已删除",
            "data": {},
        }
    except Exception as e:
        logger.error(f"删除集合失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"删除集合失败: {e}")


@router.get("/collections/{collection_name}/chunks", summary="查看集合中的文档块列表")
async def get_collection_chunks(
    collection_name: str,
    limit: int = 500,
    offset: int = 0,
    rag: RAGPipeline = Depends(get_rag_pipeline),
    auth: dict = Depends(require_auth),
):
    """
    获取指定集合中所有文档块的列表，包含文本内容和元数据。

    参数:
        - collection_name: 集合名称
        - limit:  最大返回条数（默认 500）
        - offset: 分页偏移

    返回:
        每个文档块包含 id, content, metadata
    """
    try:
        # 校验集合可见性
        if collection_name not in await _visible_collections_for(rag, auth):
            raise HTTPException(status_code=403, detail="无权访问该集合")
        # 切换到目标集合再读取
        await rag.vector_store.aswitch_collection(collection_name)
        chunks = await rag.aget_collection_chunks(limit=limit, offset=offset)

        # 按源文件名分组
        grouped: dict[str, list] = {}
        for chunk in chunks:
            filename = chunk.get("metadata", {}).get("filename", "未知来源")
            if filename not in grouped:
                grouped[filename] = []
            grouped[filename].append(chunk)

        return {
            "code": 0,
            "message": "success",
            "data": {
                "total": len(chunks),
                "collection": collection_name,
                "chunks": chunks,
                "grouped": grouped,
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取文档块列表异常: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"获取文档块列表失败: {e}")


# ---------------------------------------------------------------
# 对话管理
# ---------------------------------------------------------------

@router.get("/conversations", summary="获取所有对话列表")
async def list_conversations(
    mgr: ConversationManager = Depends(get_conversation_mgr),
    auth: dict = Depends(require_auth),
):
    """获取当前用户的历史对话列表（按更新时间倒序）"""
    try:
        user_id = await _resolve_user_id(mgr, auth)
        convs = await mgr.alist_conversations(user_id=user_id)
        return {"code": 0, "message": "success", "data": convs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取对话列表失败: {e}")


@router.post("/conversations", summary="创建新对话")
async def create_conversation(
    mgr: ConversationManager = Depends(get_conversation_mgr),
    auth: dict = Depends(require_auth),
):
    """创建一个新的空对话（归属当前用户）"""
    try:
        user_id = await _resolve_user_id(mgr, auth)
        conv = await mgr.acreate_conversation(user_id=user_id)
        return {"code": 0, "message": "success", "data": conv}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"创建对话失败: {e}")


@router.delete("/conversations", summary="批量删除对话")
async def batch_delete_conversations(
    body: BatchDeleteConversationsRequest,
    mgr: ConversationManager = Depends(get_conversation_mgr),
    auth: dict = Depends(require_auth),
):
    """批量删除指定对话（仅能删除当前用户的对话）"""
    try:
        user_id = await _resolve_user_id(mgr, auth)
        deleted = 0
        for conv_id in body.ids:
            if await mgr.adelete_conversation(conv_id, user_id=user_id):
                deleted += 1

        return {"code": 0, "message": f"成功删除 {deleted} 个对话", "data": {"deleted": deleted}}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"批量删除失败: {e}")


@router.delete("/conversations/{conv_id}", summary="删除对话")
async def delete_conversation(
    conv_id: int,
    mgr: ConversationManager = Depends(get_conversation_mgr),
    auth: dict = Depends(require_auth),
):
    """删除指定对话及其所有消息（仅能删除当前用户的对话）"""
    try:
        user_id = await _resolve_user_id(mgr, auth)
        ok = await mgr.adelete_conversation(conv_id, user_id=user_id)
        if not ok:
            raise HTTPException(status_code=404, detail="对话不存在或无权操作")
        return {"code": 0, "message": "对话已删除", "data": {}}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除对话失败: {e}")


@router.put("/conversations/{conv_id}/title", summary="修改对话标题")
async def update_conversation_title(
    conv_id: int,
    body: UpdateConversationTitleRequest,
    mgr: ConversationManager = Depends(get_conversation_mgr),
    auth: dict = Depends(require_auth),
):
    """修改对话标题（仅能修改当前用户的对话）"""
    try:
        user_id = await _resolve_user_id(mgr, auth)
        ok = await mgr.aupdate_title(conv_id, body.title.strip(), user_id=user_id)
        if not ok:
            raise HTTPException(status_code=404, detail="对话不存在或无权操作")
        return {"code": 0, "message": "success", "data": {}}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"修改标题失败: {e}")


@router.get("/conversations/{conv_id}/messages", summary="获取对话消息列表")
async def get_messages(
    conv_id: int,
    mgr: ConversationManager = Depends(get_conversation_mgr),
    auth: dict = Depends(require_auth),
):
    """获取指定对话的所有消息（仅能读取当前用户的对话）"""
    try:
        user_id = await _resolve_user_id(mgr, auth)
        messages = await mgr.aget_messages(conv_id, user_id=user_id)
        return {"code": 0, "message": "success", "data": messages}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取消息失败: {e}")


@router.post("/conversations/{conv_id}/messages", summary="添加消息到对话")
async def add_message(
    conv_id: int,
    body: AddMessageRequest,
    mgr: ConversationManager = Depends(get_conversation_mgr),
    auth: dict = Depends(require_auth),
):
    """向对话中添加一条消息（仅能操作当前用户的对话）"""
    try:
        user_id = await _resolve_user_id(mgr, auth)
        msg_id = await mgr.aadd_message(
            conv_id,
            body.role,
            body.content,
            body.sources,
            body.answer_type,
        )
        return {"code": 0, "message": "success", "data": {"id": msg_id}}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"添加消息失败: {e}")


@router.put("/conversations/{conv_id}/messages/{msg_id}", summary="更新消息内容")
async def update_message(
    conv_id: int,
    msg_id: int,
    body: UpdateMessageRequest,
    mgr: ConversationManager = Depends(get_conversation_mgr),
    auth: dict = Depends(require_auth),
):
    """更新消息内容（用于流式完成后补充完整内容和来源）"""
    try:
        user_id = await _resolve_user_id(mgr, auth)
        # 校验对话归属
        msgs = await mgr.aget_messages(conv_id, user_id=user_id)
        if not any(m["id"] == msg_id for m in msgs):
            raise HTTPException(status_code=404, detail="消息不存在或无权操作")
        ok = await mgr.aupdate_message_content(msg_id, body.content, body.sources)
        if not ok:
            raise HTTPException(status_code=404, detail="消息不存在")
        return {"code": 0, "message": "success", "data": {}}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"更新消息失败: {e}")


@router.post("/conversations/{conv_id}/messages/{msg_id}/feedback", summary="消息反馈（点赞/点踩）")
async def set_message_feedback(
    conv_id: int,
    msg_id: int,
    body: MessageFeedbackRequest,
    rag: RAGPipeline = Depends(get_rag_pipeline),
    mgr: ConversationManager = Depends(get_conversation_mgr),
    auth: dict = Depends(require_auth),
):
    """
    对一条 AI 回答设置用户反馈：1=赞, -1=踩, 0=清除。

    仅能操作当前用户的对话消息。
    """
    if body.feedback not in (-1, 0, 1):
        raise HTTPException(status_code=400, detail="feedback 必须为 -1（踩）、0（清除）或 1（赞）")
    try:
        user_id = await _resolve_user_id(mgr, auth)
        # 校验对话归属（消息须属于当前用户或匿名共享）
        msgs = await mgr.aget_messages(conv_id, user_id=user_id)
        if not any(m["id"] == msg_id for m in msgs):
            raise HTTPException(status_code=404, detail="消息不存在或无权操作")
        ok = await rag.vector_store.aset_message_feedback(
            msg_id, body.feedback, body.comment
        )
        if not ok:
            raise HTTPException(status_code=404, detail="消息不存在")
        return {"code": 0, "message": "success", "data": {"feedback": body.feedback}}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"设置反馈失败: {e}")


# ---------------------------------------------------------------
# 查询审计（仅管理员）
# ---------------------------------------------------------------

async def _require_admin(request: Request, auth: dict):
    """审计查询接口的管理员校验，返回用户管理器实例。"""
    if not settings.AUTH_ENABLED:
        return request.app.state.user_manager
    username = auth.get("username")
    if not username or username == "anonymous":
        raise HTTPException(status_code=401, detail="需要登录")
    um = request.app.state.user_manager
    user = await um._store.aget_user_by_username(username)
    if user is None or not user["is_admin"]:
        raise HTTPException(status_code=403, detail="仅管理员可执行此操作")
    return um


@router.get("/audit/queries", summary="查询审计列表（管理员）")
async def list_query_audit(
    request: Request,
    limit: int = 50,
    offset: int = 0,
    username: str | None = None,
    rag: RAGPipeline = Depends(get_rag_pipeline),
    auth: dict = Depends(require_auth),
):
    """分页查询问答审计记录（按时间倒序），可选按用户名过滤。"""
    await _require_admin(request, auth)
    try:
        rows = await rag.vector_store.alist_query_audit(
            limit=min(limit, 200), offset=max(offset, 0), username=username,
        )
        return {"code": 0, "message": "success", "data": rows}
    except Exception as e:
        logger.error(f"查询审计失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"查询审计失败: {e}")


@router.get("/audit/summary", summary="查询审计汇总（管理员）")
async def get_query_audit_summary(
    request: Request,
    rag: RAGPipeline = Depends(get_rag_pipeline),
    auth: dict = Depends(require_auth),
):
    """查询问答审计汇总统计（总查询数、缓存命中率、平均延迟、Token 用量、热门问题等）。"""
    await _require_admin(request, auth)
    try:
        summary = await rag.vector_store.aget_query_audit_summary()
        return {"code": 0, "message": "success", "data": summary}
    except Exception as e:
        logger.error(f"查询审计汇总失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"查询审计汇总失败: {e}")
