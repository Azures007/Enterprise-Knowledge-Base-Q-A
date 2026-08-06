"""
=============================================================================
工具注册表与执行器

ToolRegistry:
    维护 工具名 → (JSON Schema, 执行函数) 映射，提供 OpenAI 兼容的工具定义列表。

ToolExecutor:
    驱动完整工具循环（Tool Loop）：
        round 1: messages + tools → LLM 返回 tool_calls
        round 2+: 执行工具 → 追加 assistant(tool_calls) + tool(result) 消息 → 再问 LLM
    带配置：最大轮数、工具结果截断、管理员工具鉴权。
=============================================================================
"""

import json
from typing import Any

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class ToolRegistry:
    """工具注册表：管理工具定义与执行函数。"""

    def __init__(self, definitions: list[dict] | None = None, handlers: dict | None = None):
        self._definitions = list(definitions or [])
        self._handlers = dict(handlers or {})

    def register(self, name: str, definition: dict, handler):
        """注册一个工具。"""
        self._definitions.append(definition)
        self._handlers[name] = handler

    def get_definition(self, name: str) -> dict | None:
        """按名称获取工具定义（OpenAI 兼容格式）。"""
        for d in self._definitions:
            if d.get("function", {}).get("name") == name:
                return d
        return None

    @property
    def definitions(self) -> list[dict]:
        """全部工具定义（OpenAI 兼容格式）。"""
        return list(self._definitions)

    @property
    def names(self) -> list[str]:
        return [d.get("function", {}).get("name") for d in self._definitions]

    def has(self, name: str) -> bool:
        return name in self._handlers


def build_tool_definitions(registry: ToolRegistry | None = None) -> list[dict]:
    """构建要传给 LLM 的工具定义列表。"""
    if registry is not None:
        return registry.definitions
    from src.tools.tools import TOOL_DEFINITIONS

    return list(TOOL_DEFINITIONS)


class ToolExecutor:
    """
    工具调用执行器：驱动完整工具循环。

    用法（异步环境）:
        executor = ToolExecutor(llm, registry, max_rounds=4)
        result = await executor.run(messages, tools, ctx)
        # result: {"content": 最终回答, "tool_calls": [...], "messages": [...], "rounds": n}

    用法（同步环境，如 CLI）:
        result = executor.run_sync(messages, tools, ctx)
    """

    def __init__(
        self,
        llm: Any,
        registry: ToolRegistry | None = None,
        max_rounds: int = 4,
        result_limit: int = 2000,
        admin_tools: tuple[str, ...] = ("query_audit_summary",),
    ):
        """
        Args:
            llm:          LLM 实例（需支持 achat_with_tools）
            registry:     工具注册表（默认使用内置工具）
            max_rounds:   最大工具循环轮数，防止 LLM 反复调工具
            result_limit: 工具结果截断字符数
            admin_tools:  仅管理员可用的工具名
        """
        self.llm = llm
        self.registry = registry
        self.max_rounds = max_rounds
        self.result_limit = result_limit
        self.admin_tools = set(admin_tools)

    async def run(
        self,
        messages: list[dict[str, str]],
        ctx: dict,
        tools: list[dict] | None = None,
        is_admin: bool = False,
    ) -> dict[str, Any]:
        """
        执行工具循环，返回最终回答。

        Args:
            messages: 对话消息（不含工具结果，执行器会追加）
            ctx:      工具执行上下文 {rag, vector_store, reranker}
            tools:    工具定义列表（默认注册表全部工具）
            is_admin: 是否管理员（决定 admin_tools 是否可用）

        Returns:
            dict:
                - content:     最终回答文本
                - tool_calls:  本轮调用过的工具列表（name + 结果摘要）
                - messages:    完整消息序列（含 tool 消息，可用于继续对话）
                - rounds:      实际工具轮数
                - usage:       各轮 token 用量合并
        """
        if self.registry is not None:
            tools = tools or self.registry.definitions
        else:
            tools = tools or build_tool_definitions()

        # 非管理员过滤掉管理工具
        available_tools = [
            t for t in tools
            if is_admin or t.get("function", {}).get("name") not in self.admin_tools
        ]

        from src.tools.tools import run_tools

        working = list(messages)
        total_usage: dict = {}
        tool_log: list[dict] = []

        for round_no in range(self.max_rounds):
            resp = await self.llm.achat_with_tools(
                working, available_tools,
                usage_cb=lambda u: _merge_usage(total_usage, u),
            )
            working.append(resp["message"])

            tool_calls = resp.get("tool_calls") or []
            if not tool_calls:
                # 无工具调用 → 最终回答
                return {
                    "content": resp.get("content", ""),
                    "tool_calls": tool_log,
                    "messages": working,
                    "rounds": round_no,
                    "usage": total_usage,
                }

            # 执行工具
            for tc in tool_calls:
                fn = tc.get("function", {})
                name = fn.get("name", "")
                args = fn.get("arguments", "{}")
                if not self._is_tool_available(name, available_tools):
                    result = {"error": f"工具 {name} 不可用（无权限或未注册）"}
                else:
                    result = await run_tools(ctx, name, args)
                    logger.info(f"工具执行: {name}({str(args)[:80]})")

                # 截断工具结果，防止上下文爆炸
                result_json = json.dumps(result, ensure_ascii=False)
                if len(result_json) > self.result_limit:
                    result_json = result_json[: self.result_limit] + '..."(已截断)"'
                tool_log.append({"name": name, "arguments": args, "result": result_json[:300]})

                working.append({
                    "role": "tool",
                    "tool_call_id": tc.get("id", ""),
                    "content": result_json,
                })

        # 达到最大轮数仍未结束：用当前消息做最后一次普通生成
        logger.warning(f"工具循环达到最大轮数 {self.max_rounds}，走兜底生成")
        final = await self.llm.achat_with_tools(working, available_tools)
        return {
            "content": final.get("content", ""),
            "tool_calls": tool_log,
            "messages": working,
            "rounds": self.max_rounds,
            "usage": total_usage,
        }

    def _is_tool_available(self, name: str, tools: list[dict]) -> bool:
        return any(t.get("function", {}).get("name") == name for t in tools)

    def run_sync(
        self,
        messages: list[dict[str, str]],
        ctx: dict,
        tools: list[dict] | None = None,
        is_admin: bool = False,
    ) -> dict[str, Any]:
        """同步版本的 run（CLI 场景）。"""
        import asyncio

        return asyncio.run(self.run(messages, ctx, tools, is_admin))


def _merge_usage(total: dict, usage: dict) -> None:
    """合并多轮 token 用量。"""
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        total[key] = total.get(key, 0) + (usage.get(key) or 0)
