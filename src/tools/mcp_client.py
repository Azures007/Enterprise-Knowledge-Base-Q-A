"""
=============================================================================
MCP (Model Context Protocol) 客户端适配层

让项目作为 MCP Client 接入现成的 MCP Server（如社区现成的天气 server），
把 MCP 工具纳入现有的 Function Calling 工具循环。

设计：
    - 启动时用 ClientSession 连接 MCP Server（stdio 或 HTTP 两种方式），
      拉取工具列表（list_tools），把 MCP 工具转成 OpenAI function calling 格式。
    - 工具执行时路由到 MCP 的 call_tool，返回结果供 LLM 读取。
    - 依赖注入：MCP 工具定义与执行函数与内置工具一样，作为
      {name: (definition, handler)} 注入现有 ToolRegistry。

配置（.env）:
    MCP_SERVERS_JSON = '[{"name":"weather","type":"stdio",\
        "command":"npx","args":["-y","yiketianqi-weather-mcp"]},...]'
    或 HTTP 方式: {"name":"weather","type":"http","url":"http://localhost:9000/mcp"}

注意：连接在应用启动时建立（lifespan），进程退出时统一 aclose 关闭。
=============================================================================
"""

import json
from typing import Any

from src.monitoring import record_mcp_connect_failed, record_mcp_tool_failed
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def _extract_text(content) -> str:
    """从 CallToolResult.content 提取纯文本（兼容 text/TextContent/JSON 字符串）。"""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(item.get("text", ""))
                else:
                    parts.append(json.dumps(item, ensure_ascii=False))
            elif hasattr(item, "text"):
                # TextContent 等 pydantic 对象
                parts.append(item.text if item.type == "text" else json.dumps(
                    item.model_dump(), ensure_ascii=False))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)


class MCPToolClient:
    """
    单个 MCP Server 的客户端封装。

    用法:
        client = MCPToolClient(cfg)
        await client.connect()            # 建立连接 + 拉取工具
        definitions = client.definitions  # OpenAI 兼容工具定义
        result = await client.call(name, args)  # 执行工具
        await client.aclose()             # 关闭连接
    """

    def __init__(self, cfg: dict):
        self.name = cfg.get("name") or "mcp"
        self.type = (cfg.get("type") or "stdio").lower()
        self.command = cfg.get("command")
        self.args = cfg.get("args") or []
        self.env = cfg.get("env") or {}
        self.url = cfg.get("url")
        self._session = None
        self._context = None  # 传输上下文管理器（stdio/http client）
        self._streams = None
        self._tools: list[dict] = []  # OpenAI 格式工具定义
        self._tool_map: dict[str, Any] = {}  # name -> MCP Tool

    # ------------------------------------------------------------------
    # 连接与工具拉取
    # ------------------------------------------------------------------

    async def connect(self) -> bool:
        """建立连接并拉取工具列表。失败返回 False（不抛出，避免阻断启动）。"""
        try:
            if self.type in ("stdio", "command"):
                if not self.command:
                    logger.warning(f"MCP[{self.name}] 未配置 command，跳过")
                    return False
                import asyncio
                from mcp import ClientSession
                from mcp.client.stdio import StdioServerParameters, stdio_client

                params = StdioServerParameters(
                    command=self.command,
                    args=self.args,
                    env=self.env or None,
                )
                self._context = stdio_client(params)
                streams = await self._context.__aenter__()
            elif self.type in ("http", "sse", "streamable_http"):
                if not self.url:
                    logger.warning(f"MCP[{self.name}] 未配置 url，跳过")
                    return False
                from mcp import ClientSession
                from mcp.client.streamable_http import streamable_http_client

                self._context = streamable_http_client(self.url)
                streams = await self._context.__aenter__()
            else:
                logger.warning(f"MCP[{self.name}] 未知传输类型: {self.type}")
                return False

            self._streams = streams
            self._session = ClientSession(
                streams[0], streams[1],
                client_info={"name": "enterprise-kb", "version": "1.0.0"},
            )
            await self._session.__aenter__()
            await self._session.initialize()

            # 拉取工具列表
            result = await self._session.list_tools()
            tools = getattr(result, "tools", None) or []
            for tool in tools:
                name = getattr(tool, "name", "")
                desc = getattr(tool, "description", "") or ""
                schema = getattr(tool, "input_schema", None) or {"type": "object", "properties": {}}
                self._tool_map[name] = tool
                self._tools.append({
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": desc,
                        "parameters": schema,
                    },
                    # 与内置工具一致：MCP 工具默认只读；写操作需在 x_meta 标记
                    "x_meta": {"mutation": False, "depends_on": []},
                })
            logger.info(
                f"MCP[{self.name}] 已连接，加载 {len(self._tools)} 个工具: "
                f"{[t['function']['name'] for t in self._tools]}"
            )
            return True
        except Exception as e:
            logger.warning(f"MCP[{self.name}] 连接失败: {e}")
            record_mcp_connect_failed(f"{self.name}: {e}")
            await self.aclose()
            return False

    @property
    def definitions(self) -> list[dict]:
        """OpenAI 兼容的工具定义列表（供 ToolRegistry 使用）。"""
        return list(self._tools)

    @property
    def tool_names(self) -> list[str]:
        return [t["function"]["name"] for t in self._tools]

    # ------------------------------------------------------------------
    # 工具执行
    # ------------------------------------------------------------------

    async def call(self, name: str, arguments: dict | None = None) -> dict:
        """调用 MCP 工具，返回 JSON 可序列化的结果 dict（LLM 作为 tool 消息读回）。"""
        if self._session is None:
            return {"error": f"MCP[{self.name}] 未连接"}
        if name not in self._tool_map:
            return {"error": f"MCP[{self.name}] 未注册工具: {name}"}
        try:
            args = arguments or {}
            result = await self._session.call_tool(name, args)
            if getattr(result, "is_error", False):
                text = _extract_text(getattr(result, "content", None))
                return {"error": text or f"工具 {name} 执行失败"}
            text = _extract_text(getattr(result, "content", None))
            # 尝试解析成结构化 JSON；失败则作为纯文本返回
            try:
                parsed = json.loads(text)
                if isinstance(parsed, dict):
                    return parsed
                return {"result": parsed}
            except (ValueError, TypeError):
                return {"result": text}
        except Exception as e:
            logger.warning(f"MCP[{self.name}] 工具 {name} 执行异常: {e}")
            record_mcp_tool_failed(f"{self.name}.{name}: {e}")
            return {"error": f"MCP 工具 {name} 执行失败: {e}"}

    async def aclose(self):
        """关闭连接与传输。"""
        try:
            if self._session is not None:
                await self._session.__aexit__(None, None, None)
        except Exception:
            pass
        try:
            if self._context is not None:
                await self._context.__aexit__(None, None, None)
        except Exception:
            pass
        self._session = None
        self._context = None
        self._streams = None


class MCPManager:
    """
    管理所有配置的 MCP Server，汇总成一份工具定义与执行函数，
    注入现有的 ToolRegistry。
    """

    def __init__(self, servers_config: str = ""):
        self._clients: list[MCPToolClient] = []
        self._handler_map: dict[str, Any] = {}
        self._config = self._parse_config(servers_config)

    @staticmethod
    def _parse_config(raw: str) -> list[dict]:
        """解析 MCP_SERVERS_JSON 配置。"""
        if not raw or not raw.strip():
            return []
        try:
            data = json.loads(raw)
            if isinstance(data, list):
                return [c for c in data if isinstance(c, dict)]
            if isinstance(data, dict):
                # 兼容 {"mcpServers": {...}} 结构
                servers = data.get("mcpServers", data)
                out = []
                for name, cfg in servers.items():
                    if isinstance(cfg, dict):
                        item = dict(cfg)
                        item.setdefault("name", name)
                        out.append(item)
                return out
        except Exception as e:
            logger.warning(f"MCP_SERVERS_JSON 解析失败: {e}")
        return []

    async def connect_all(self) -> None:
        """连接所有配置的 MCP Server，构建工具名 → 执行函数映射。"""
        for cfg in self._config:
            client = MCPToolClient(cfg)
            ok = await client.connect()
            if ok:
                self._clients.append(client)
                for name in client.tool_names:
                    self._handler_map[name] = self._make_handler(client, name)

    @staticmethod
    def _make_handler(client: MCPToolClient, name: str):
        """为某 MCP 工具构造异步执行函数（与内置 handler 签名一致）。"""
        async def handler(ctx: dict, **kwargs) -> dict:
            return await client.call(name, kwargs)
        handler.__name__ = f"mcp_{name}"
        return handler

    @property
    def is_enabled(self) -> bool:
        return len(self._clients) > 0

    def merge_into(self, definitions: list[dict], handlers: dict) -> tuple[list[dict], dict]:
        """
        把 MCP 工具合并进内置工具定义与执行函数。
        - MCP 工具优先：与内置重名时，MCP 版本覆盖内置（用户显式配置了 MCP server）。
          示例：内置 get_weather（Open-Meteo）被 MCP 的 get_weather 覆盖。
        - 返回 (definitions, handlers)，两者传入 ToolRegistry。
        """
        merged_defs = [d for d in definitions
                       if d.get("function", {}).get("name") not in self._handler_map]
        merged_handlers = {k: v for k, v in handlers.items()
                           if k not in self._handler_map}
        for client in self._clients:
            for tool in client.definitions:
                name = tool["function"]["name"]
                merged_defs.append(tool)
                merged_handlers[name] = self._handler_map[name]
        return merged_defs, merged_handlers

    async def aclose_all(self):
        for client in self._clients:
            await client.aclose()
        self._clients = []
        self._handler_map = {}
