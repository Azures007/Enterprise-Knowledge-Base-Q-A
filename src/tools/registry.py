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

import asyncio
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
        max_tool_calls: int = 8,
        max_tokens_budget: int = 16000,
        duplicate_threshold: int = 2,
        result_limit: int = 2000,
        confirm_expires: int = 300,
        admin_tools: tuple[str, ...] = ("query_audit_summary",),
        mutation_tools: tuple[str, ...] = (),
    ):
        """
        Args:
            llm:                 LLM 实例（需支持 achat_with_tools）
            registry:            工具注册表（默认使用内置工具）
            max_rounds:          最大工具循环轮数
            max_tool_calls:      整个循环累计工具调用次数上限（含被重复/确认拦截的）
            max_tokens_budget:   累计 token 预算上限，超过则强制收敛
            duplicate_threshold: 同一工具+参数连续/累计调用多少次后判定重复
            result_limit:        工具结果截断字符数
            confirm_expires:     写操作确认请求有效期（秒）
            admin_tools:         仅管理员可用的工具名
            mutation_tools:      写操作工具名（需确认后才执行）
        """
        self.llm = llm
        self.registry = registry
        self.max_rounds = max_rounds
        self.max_tool_calls = max_tool_calls
        self.max_tokens_budget = max_tokens_budget
        self.duplicate_threshold = duplicate_threshold
        self.result_limit = result_limit
        self.confirm_expires = confirm_expires
        self.admin_tools = set(admin_tools)
        self.mutation_tools = set(mutation_tools)
        # 写操作确认请求存储：confirm_id -> {tool, args, created_at, username}
        self._pending_confirms: dict[str, dict] = {}

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
        # 重复调用检测：键 = 工具名 + 规范化参数 → 累计次数
        call_counts: dict[str, int] = {}

        for round_no in range(self.max_rounds):
            resp = await self.llm.achat_with_tools(
                working, available_tools,
                usage_cb=lambda u: _merge_usage(total_usage, u),
            )
            working.append(resp["message"])

            tool_calls = resp.get("tool_calls") or []
            if not tool_calls:
                # 无工具调用 → 最终回答
                return self._build_result(
                    resp.get("content", ""), tool_log, working, round_no, total_usage
                )

            # ---- 重复检测 + 组装待执行列表 ----
            pending: list[tuple] = []  # (tc, name, args, flag)
            for tc in tool_calls:
                fn = tc.get("function", {})
                name = fn.get("name", "")
                args = fn.get("arguments", "{}")
                if not self._is_tool_available(name, available_tools):
                    pending.append((tc, name, args, "unavailable"))
                    continue
                # 写操作拦截：不执行，返回确认请求（配置集合 或 工具定义 x_meta 标记）
                if self._is_mutation(name, available_tools):
                    pending.append((tc, name, args, "confirm"))
                    continue
                # 重复检测
                key = self._call_key(name, args)
                call_counts[key] = call_counts.get(key, 0) + 1
                if call_counts[key] > self.duplicate_threshold:
                    pending.append((tc, name, args, "duplicate"))
                else:
                    pending.append((tc, name, args, "ok"))

            # ---- 上限检查：累计调用次数 / token 预算 ----
            if len(tool_log) + len([p for p in pending if p[3] != "unavailable"]) >= self.max_tool_calls:
                logger.warning(f"工具累计调用次数达到上限 {self.max_tool_calls}，强制收敛")
                final = await self.llm.achat_with_tools(working, available_tools)
                return self._build_result(
                    final.get("content", ""), tool_log, working, round_no, total_usage
                )
            total_tok = total_usage.get("total_tokens", 0)
            if total_tok >= self.max_tokens_budget:
                logger.warning(f"token 预算已达上限 {self.max_tokens_budget}，强制收敛")
                final = await self.llm.achat_with_tools(working, available_tools)
                return self._build_result(
                    final.get("content", ""), tool_log, working, round_no, total_usage
                )

            # ---- 执行工具 ----
            # 无依赖（depends_on 未声明）的工具并行执行；串行兜底
            results = await self._execute_tool_calls(
                pending, ctx, run_tools, available_tools, tool_log
            )
            for (tc, name, args, _), result in zip(pending, results):
                result_json = json.dumps(result, ensure_ascii=False)
                if len(result_json) > self.result_limit:
                    result_json = result_json[: self.result_limit] + '..."(已截断)"'
                working.append({
                    "role": "tool",
                    "tool_call_id": tc.get("id", ""),
                    "content": result_json,
                })

        # 达到最大轮数仍未结束：用当前消息做最后一次普通生成
        logger.warning(f"工具循环达到最大轮数 {self.max_rounds}，走兜底生成")
        final = await self.llm.achat_with_tools(working, available_tools)
        return self._build_result(
            final.get("content", ""), tool_log, working, self.max_rounds, total_usage
        )

    # ================================================================
    # 工具执行（并行 + 拦截）
    # ================================================================

    async def _execute_tool_calls(
        self,
        pending: list[tuple],
        ctx: dict,
        run_tools,
        available_tools: list[dict],
        tool_log: list[dict],
    ) -> list[dict]:
        """
        执行一批工具调用。

        非确认/非重复的普通工具并行执行（asyncio.gather）；
        被拦截的（写操作确认、重复、不可用）直接生成对应返回，不执行。
        """
        from src.tools.tools import run_tools as _rt

        results: list[dict] = []

        async def run_one(tc, name, args, flag):
            if flag == "confirm":
                confirm_id = self._create_confirm(name, args, ctx)
                return {
                    "confirm_required": True,
                    "tool": name,
                    "args": self._safe_args_display(args),
                    "confirm_id": confirm_id,
                    "message": f"操作「{name}」需要用户确认后才会执行",
                }
            if flag == "duplicate":
                logger.warning(f"重复调用检测: {name} 相同参数已多次调用，拒绝执行")
                return {
                    "error": f"工具 {name} 的相同调用已重复多次，请基于已有结果继续回答，"
                             f"或更换参数/换一种方式，不要重复调用同一工具同一参数"
                }
            if flag == "unavailable":
                return {"error": f"工具 {name} 不可用（无权限或未注册）"}
            # 正常执行
            try:
                result = await _rt(ctx, name, args)
                logger.info(f"工具执行: {name}({str(args)[:80]})")
                return result
            except Exception as e:
                logger.warning(f"工具 {name} 执行异常: {e}", exc_info=True)
                return {"error": f"工具 {name} 执行失败: {e}"}

        # 并行执行（无依赖判断：本实现默认并行，依赖留给后续按 depends_on 串行）
        results = await asyncio.gather(*[
            run_one(tc, name, args, flag) for (tc, name, args, flag) in pending
        ])

        # 记录日志（含拦截项）
        for (tc, name, args, flag), result in zip(pending, results):
            tool_log.append({
                "name": name,
                "arguments": args,
                "result": json.dumps(result, ensure_ascii=False)[:300],
                "flag": flag,
            })
        return results

    # ================================================================
    # 重复检测辅助
    # ================================================================

    @staticmethod
    def _call_key(name: str, args: Any) -> str:
        """生成"工具名 + 规范化参数"的键，用于重复调用检测。"""
        try:
            if isinstance(args, str):
                args = json.loads(args)
            if not isinstance(args, dict):
                args = {"value": str(args)}
            # 规范化：key 排序，保证参数顺序不影响判重
            norm = json.dumps(args, sort_keys=True, ensure_ascii=False)
        except (ValueError, TypeError):
            norm = str(args)
        return f"{name}:{norm}"

    @staticmethod
    def _safe_args_display(args: Any) -> str:
        """确认请求里展示参数（截断，避免暴露过长内容）。"""
        try:
            if isinstance(args, str):
                args = json.loads(args)
        except (ValueError, TypeError):
            pass
        return json.dumps(args, ensure_ascii=False)[:300]

    # ================================================================
    # 写操作确认
    # ================================================================

    def _create_confirm(self, name: str, args: Any, ctx: dict) -> str:
        """为写操作创建确认请求，返回 confirm_id。"""
        import secrets
        import time

        confirm_id = "cf_" + secrets.token_hex(8)
        self._pending_confirms[confirm_id] = {
            "tool": name,
            "args": self._safe_args_display(args),
            "created_at": time.time(),
            "username": (ctx.get("auth") or {}).get("username") or "anonymous",
            "expires_at": time.time() + self.confirm_expires,
        }
        # 清理过期确认请求
        now = time.time()
        for cid in list(self._pending_confirms):
            if self._pending_confirms[cid].get("expires_at", 0) < now:
                del self._pending_confirms[cid]
        return confirm_id

    def confirm_mutation(self, confirm_id: str, username: str) -> tuple[bool, str, dict | None]:
        """
        确认并执行一个待确认的写操作。

        Returns:
            (ok, message, args): ok=False 时 message 为失败原因
        """
        import time

        pending = self._pending_confirms.pop(confirm_id, None)
        if pending is None:
            return False, "确认请求不存在或已过期", None
        if pending.get("expires_at", 0) < time.time():
            return False, "确认请求已过期", None
        # 权限：仅创建者可确认（或匿名）
        if pending.get("username") not in (username, "anonymous", None):
            return False, "该确认请求不属于当前用户", None
        return True, "ok", pending

    @staticmethod
    def _build_result(content, tool_log, working, rounds, usage) -> dict:
        """组装返回 dict。"""
        return {
            "content": content,
            "tool_calls": tool_log,
            "messages": working,
            "rounds": rounds,
            "usage": usage,
        }

    def _is_tool_available(self, name: str, tools: list[dict]) -> bool:
        return any(t.get("function", {}).get("name") == name for t in tools)

    def _is_mutation(self, name: str, tools: list[dict]) -> bool:
        """判断工具是否为写操作（需确认）。

        优先读工具定义里的 x_meta.mutation；配置集合 TOOL_MUTATION_TOOLS 兜底。
        """
        if name in self.mutation_tools:
            return True
        for t in tools:
            if t.get("function", {}).get("name") == name:
                return bool(t.get("x_meta", {}).get("mutation", False))
        return False

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
