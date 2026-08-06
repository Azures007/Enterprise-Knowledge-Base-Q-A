"""
=============================================================================
工具调用模块（Function Calling）

提供 LLM 可调用的工具注册表与执行器：

    - ToolRegistry:   注册工具（函数 + JSON Schema 定义），按名字分发执行
    - ToolExecutor:   驱动完整工具循环（LLM 请求 → tool_calls → 执行 → 喂回），
                      带最大轮数、结果截断、参数校验

内置工具（tools.py）：
    - get_current_time      获取当前时间
    - calculate             基础四则运算
    - list_collections      列出知识库集合
    - search_knowledge_base 在知识库中检索相关文档片段
    - get_knowledge_base_stats 获取知识库统计
    - query_audit_summary   查询审计汇总（管理员）

使用方法:
    from src.tools import ToolExecutor, build_tool_definitions, run_tools

    # 只执行一次工具调用（不涉及 LLM）
    result = run_tools("search_knowledge_base", {"question": "考勤制度"})
=============================================================================
"""

from src.tools.registry import ToolExecutor, ToolRegistry, build_tool_definitions
from src.tools.tools import (
    TOOL_DEFINITIONS,
    run_tools,
)

__all__ = [
    "ToolExecutor",
    "ToolRegistry",
    "build_tool_definitions",
    "TOOL_DEFINITIONS",
    "run_tools",
]
