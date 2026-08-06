"""
=============================================================================
内置工具实现

每个工具 = JSON Schema 定义 + 异步执行函数。

执行函数通过注入的 context（rag / vector_store / reranker）访问系统能力，
不直接依赖全局单例，便于测试与并发。

工具约定：
    - 执行函数签名: async fn(ctx, **kwargs) -> dict
    - 返回值必须是 JSON 可序列化的 dict（LLM 会作为 tool 消息读回）
    - 所有向量存储/检索调用使用 a* 异步方法（与 FastAPI 事件循环兼容）
=============================================================================
"""

import datetime
import json
from typing import Any, Callable

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


# ==============================================================================
# 通用工具
# ==============================================================================

async def _tool_get_current_time(ctx: dict, **kwargs) -> dict:
    """获取当前时间"""
    now = datetime.datetime.now()
    return {
        "datetime": now.strftime("%Y-%m-%d %H:%M:%S"),
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
    }


async def _tool_calculate(ctx: dict, **kwargs) -> dict:
    """基础四则运算（安全解析，仅支持 + - * / 和括号）"""
    expr = str(kwargs.get("expression", "")).strip()
    if not expr:
        return {"error": "表达式为空"}
    # 仅允许数字、运算符、括号、小数点、空格
    allowed = set("0123456789+-*/(). ")
    if any(c not in allowed for c in expr):
        return {"error": f"表达式包含非法字符: {expr}"}
    try:
        result = eval(expr, {"__builtins__": {}}, {})
        return {"expression": expr, "result": result}
    except Exception as e:
        return {"error": f"计算失败: {e}"}


# ==============================================================================
# 知识库工具
# ==============================================================================

async def _tool_list_collections(ctx: dict, **kwargs) -> dict:
    """列出知识库集合及文档块数量"""
    rag = ctx.get("rag")
    vector_store = rag.vector_store if rag else ctx.get("vector_store")
    if vector_store is None:
        return {"error": "向量存储不可用", "collections": []}
    try:
        names = await vector_store.alist_collections()
        stats = await vector_store.aget_collection_stats(names) if names else []
        return {
            "collections": [
                {
                    "name": s["name"],
                    "chunk_count": s["chunk_count"],
                    "document_count": s["document_count"],
                }
                for s in stats
            ]
        }
    except Exception as e:
        logger.warning(f"list_collections 失败: {e}")
        return {"error": str(e), "collections": []}


async def _tool_search_knowledge_base(ctx: dict, **kwargs) -> dict:
    """在知识库中检索与问题相关的文档片段"""
    question = str(kwargs.get("question", "")).strip()
    if not question:
        return {"error": "检索问题不能为空"}
    k = int(kwargs.get("k") or 5)
    k = max(1, min(k, 20))
    collection = kwargs.get("collection")

    rag = ctx.get("rag")
    vector_store = rag.vector_store if rag else ctx.get("vector_store")
    if vector_store is None:
        return {"error": "向量存储不可用", "results": []}

    try:
        # 未指定集合时，跨所有有内容的集合检索（避免命中空集合导致 0 结果）
        if not collection:
            coll_names = await vector_store.alist_collections()
            coll_stats = await vector_store.aget_collection_stats(coll_names)
            non_empty = [s["name"] for s in coll_stats if s["chunk_count"] > 0]
        else:
            non_empty = [collection]
            await vector_store.aswitch_collection(collection)

        # 每个集合粗召回候选，合并后取全局最优
        all_candidates = []
        for coll in non_empty:
            await vector_store.aswitch_collection(coll)
            search_method = getattr(vector_store, "ahybrid_search", None)
            try:
                if search_method is not None:
                    candidates = await search_method(query=question, k=k * 3)
                else:
                    candidates = await vector_store.asimilarity_search(query=question, k=k * 3)
            except Exception:
                continue
            for c in candidates:
                c.setdefault("metadata", {})
                c["metadata"]["_collection"] = coll
                all_candidates.append(c)

        # 重排精排（复用 RAG 管线重排器），取最终 top-k
        reranker = ctx.get("reranker")
        if reranker is not None and all_candidates:
            all_candidates = reranker.rerank(query=question, candidates=all_candidates, top_k=k)
        else:
            all_candidates.sort(key=lambda d: d.get("score", 0), reverse=True)
            all_candidates = all_candidates[:k]

        results = []
        for c in all_candidates:
            meta = c.get("metadata", {})
            results.append({
                "content": c.get("content", ""),
                "score": round(float(c.get("score", 0)), 4),
                "filename": meta.get("filename", ""),
                "source": meta.get("source", ""),
                "page": meta.get("page"),
                "collection": meta.get("_collection") or collection,
            })
        # 精简来源摘要：放在 JSON 最前面，避免主体被截断后丢失来源信息
        source_summary = []
        for r in results:
            key = r.get("source") or r.get("filename") or r.get("content", "")[:30]
            source_summary.append({
                "filename": r.get("filename", ""),
                "source": key,
                "score": r.get("score", 0),
                "page": r.get("page"),
                "collection": r.get("collection"),
            })
        return {
            "sources": source_summary,
            "question": question,
            "k": k,
            "results": results,
        }
    except Exception as e:
        logger.warning(f"search_knowledge_base 失败: {e}")
        return {"error": str(e), "results": []}


async def _tool_get_knowledge_base_stats(ctx: dict, **kwargs) -> dict:
    """获取知识库统计信息"""
    rag = ctx.get("rag")
    if rag is None:
        return {"error": "RAG 管线不可用"}
    try:
        return await rag.aget_knowledge_base_stats()
    except Exception as e:
        logger.warning(f"get_knowledge_base_stats 失败: {e}")
        return {"error": str(e)}


async def _tool_query_audit_summary(ctx: dict, **kwargs) -> dict:
    """获取查询审计汇总统计（管理员工具）"""
    rag = ctx.get("rag")
    vector_store = rag.vector_store if rag else ctx.get("vector_store")
    if vector_store is None:
        return {"error": "向量存储不可用"}
    try:
        return await vector_store.aget_query_audit_summary()
    except Exception as e:
        logger.warning(f"query_audit_summary 失败: {e}")
        return {"error": str(e)}


# ==============================================================================
# 工具注册表
# ==============================================================================

TOOL_DEFINITIONS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "获取当前的日期与时间。当用户询问现在几点、今天几号等时间问题时使用。",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "执行基础四则运算，例如 12*5+3。当用户需要精确计算时使用。",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string", "description": "数学表达式，如 12*5+3"}},
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_collections",
            "description": "列出企业知识库中所有集合及其文档块数量。当用户询问知识库有哪些内容、想了解知识库结构时使用。",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": "在企业知识库中检索与问题最相关的文档片段，返回内容、来源文件与相关度分数。"
                           "当问题涉及知识库文档内容（制度、规范、说明等）时使用此工具获取事实依据。",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "要检索的问题或关键词"},
                    "collection": {"type": "string", "description": "可选，指定检索的集合名称；不指定则检索当前集合"},
                    "k": {"type": "integer", "description": "返回的文档片段数量，默认 5"},
                },
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_knowledge_base_stats",
            "description": "获取知识库统计信息（文档块总数、集合列表、查询次数等）。当用户询问知识库规模、数据量时使用。",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_audit_summary",
            "description": "获取查询审计汇总统计（总查询数、缓存命中率、平均延迟、Token 用量、热门问题）。管理员可用。",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]

# 工具名 → 异步执行函数映射
TOOL_HANDLERS: dict[str, Callable] = {
    "get_current_time": _tool_get_current_time,
    "calculate": _tool_calculate,
    "list_collections": _tool_list_collections,
    "search_knowledge_base": _tool_search_knowledge_base,
    "get_knowledge_base_stats": _tool_get_knowledge_base_stats,
    "query_audit_summary": _tool_query_audit_summary,
}


async def run_tools(ctx: dict, name: str, arguments: dict) -> dict:
    """
    执行单个工具并返回结果。

    Args:
        ctx:       注入上下文 {rag, vector_store, reranker}
        name:      工具名
        arguments: 工具参数（LLM 返回的 JSON 字符串或 dict）

    Returns:
        工具执行结果 dict
    """
    handler = TOOL_HANDLERS.get(name)
    if handler is None:
        return {"error": f"未知工具: {name}"}
    try:
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except (ValueError, TypeError):
                arguments = {"value": arguments}
        if not isinstance(arguments, dict):
            arguments = {"value": arguments}
        result = await handler(ctx, **arguments)
        if not isinstance(result, dict):
            result = {"result": result}
        return result
    except Exception as e:
        logger.warning(f"工具 {name} 执行异常: {e}", exc_info=True)
        return {"error": f"工具 {name} 执行失败: {e}"}
