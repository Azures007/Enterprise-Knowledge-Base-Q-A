"""
=============================================================================
优雅降级可观测性指标模块

集中统计系统各降级点与失败情况，通过 /api/metrics 暴露 JSON 指标，
便于运维监控与告警。均为进程内线程安全计数器。

统计项:
    - rerank_degraded      重排降级次数（交叉编码器不可用/加载失败 → 轻量重排）
    - redis_degraded       Redis 降级次数（连接失败 → 内存模式）
    - mcp_connect_failed   MCP Server 连接失败次数
    - mcp_tool_failed      MCP 工具调用失败次数
    - rate_limit_degraded  限流降级次数（Redis 不可用 → 内存限流）
    - ingest_zombie_cleanup 僵尸文档记录清理次数
    - llm_call_failed      LLM 调用失败次数
    - embedding_call_failed 嵌入调用失败次数

每个计数项同时记录「总次数」与「最近失败时间」，便于判断是否在持续恶化。
=============================================================================
"""

import threading
import time
from typing import Any

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class MetricsRegistry:
    """线程安全的分级指标注册表。"""

    def __init__(self):
        self._lock = threading.Lock()
        self._started_at = time.time()
        # name -> {"count": int, "last_at": float|None, "last_msg": str|None}
        self._counters: dict[str, dict[str, Any]] = {}
        # name -> {"value": Any, "updated_at": float}
        self._gauges: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # 计数器
    # ------------------------------------------------------------------

    def increment(self, name: str, by: int = 1, msg: str | None = None) -> None:
        """递增一个计数器（线程安全）。"""
        with self._lock:
            entry = self._counters.setdefault(name, {
                "count": 0,
                "last_at": None,
                "last_msg": None,
            })
            entry["count"] += by
            entry["last_at"] = time.time()
            if msg:
                entry["last_msg"] = msg[:200]

    def get_counter(self, name: str) -> dict[str, Any] | None:
        with self._lock:
            entry = self._counters.get(name)
            if entry is None:
                return None
            return dict(entry)

    # ------------------------------------------------------------------
    # 仪表（瞬时值，如当前模式）
    # ------------------------------------------------------------------

    def set_gauge(self, name: str, value: Any) -> None:
        """设置一个瞬时值（如当前重排模式、当前缓存后端）。"""
        with self._lock:
            self._gauges[name] = {"value": value, "updated_at": time.time()}

    # ------------------------------------------------------------------
    # 汇总导出
    # ------------------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        """导出全量指标快照（供 /api/metrics）。"""
        with self._lock:
            counters = {}
            for name, entry in self._counters.items():
                item = {
                    "count": entry["count"],
                    "last_at": entry["last_at"],
                }
                if entry["last_msg"]:
                    item["last_msg"] = entry["last_msg"]
                counters[name] = item
            gauges = {
                name: {"value": entry["value"], "updated_at": entry["updated_at"]}
                for name, entry in self._gauges.items()
            }
            return {
                "counters": counters,
                "gauges": gauges,
                "started_at": self._started_at,
            }


# 全局单例
_metrics = MetricsRegistry()


def get_metrics() -> MetricsRegistry:
    """获取全局指标注册表单例。"""
    return _metrics


# ==============================================================================
# 便捷函数：各降级点调用（避免处处 import MetricsRegistry 的冗长）
# ==============================================================================

def record_rerank_degraded(msg: str | None = None):
    """记录一次重排降级（交叉编码器不可用/失败 → 轻量重排）。"""
    _metrics.increment("rerank_degraded", msg=msg)
    _metrics.set_gauge("rerank_mode", None)  # 由 Reranker 主动设置真实模式


def record_redis_degraded(msg: str | None = None):
    """记录一次 Redis 降级（连接失败 → 内存模式）。"""
    _metrics.increment("redis_degraded", msg=msg)


def record_mcp_connect_failed(msg: str | None = None):
    """记录一次 MCP Server 连接失败。"""
    _metrics.increment("mcp_connect_failed", msg=msg)


def record_mcp_tool_failed(msg: str | None = None):
    """记录一次 MCP 工具调用失败。"""
    _metrics.increment("mcp_tool_failed", msg=msg)


def record_rate_limit_degraded(msg: str | None = None):
    """记录一次限流降级（Redis 不可用 → 内存限流）。"""
    _metrics.increment("rate_limit_degraded", msg=msg)


def record_llm_failed(msg: str | None = None):
    """记录一次 LLM 调用失败。"""
    _metrics.increment("llm_call_failed", msg=msg)


def record_embedding_failed(msg: str | None = None):
    """记录一次嵌入调用失败。"""
    _metrics.increment("embedding_call_failed", msg=msg)


def record_hybrid_keyword_channel(kw_hits: int):
    """记录一次混合检索关键词通道召回数量（用于观测关键词通道活跃度）。"""
    _metrics.increment("hybrid_keyword_channel_hits", by=max(0, int(kw_hits)))
