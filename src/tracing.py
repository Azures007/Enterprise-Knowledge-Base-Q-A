"""
=============================================================================
RAG 查询链路追踪模块

记录一次问答从输入到输出的每一层输入/输出，便于开发者排查
「召回 / 重排 / 生成」是哪一层出了问题。

设计:
    - RAGTracer 实例对应一次问答，持有 trace_id 与 stage 列表。
    - 管线各环节通过 trace_stage(...) 记录该层的输入与输出。
    - 通过 contextvars.ContextVar 传递「当前查询的 tracer」，
      管线内部埋点无需改动任何调用签名。
    - 查询结束后 finalize() 将完整链路序列化落盘到 logs/traces/，
      同时保留在进程内存中供 /api/traces 实时查询。

安全:
    - trace 内容可能含完整文档片段与用户问题，仅管理员可见。
    - /api/traces 接口通过 _require_admin 做权限校验。
=============================================================================
"""

import contextvars
import json
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# 日志根目录下的 traces 子目录
TRACES_DIR = Path(__file__).resolve().parent.parent / "logs" / "traces"

# 进程内存中保留的最近 trace 数上限（实时查询窗口）
MAX_IN_MEMORY = 200

# contextvar：当前正在处理的查询对应的 tracer
_current_tracer: contextvars.ContextVar["RAGTracer | None"] = contextvars.ContextVar(
    "current_rag_tracer", default=None
)


@dataclass
class TraceStage:
    """单个环节的输入/输出记录。"""

    stage: str          # 环节名，如 retrieval、rerank、generation
    input: Any = None   # 该环节的输入（问题 / 候选 / prompt 等）
    output: Any = None  # 该环节的输出（召回结果 / 重排结果 / 回答等）
    ts: float = field(default_factory=time.time)


class RAGTracer:
    """单次查询的链路追踪器。"""

    def __init__(self, question: str):
        self.trace_id = uuid.uuid4().hex[:12]
        self.question = question
        self.started_at = time.time()
        self.finished_at: float | None = None
        self.answer_type: str | None = None
        self.stages: list[TraceStage] = []
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # 记录
    # ------------------------------------------------------------------

    def log(self, stage: str, input: Any = None, output: Any = None) -> None:
        """追加一个环节记录（线程安全）。"""
        with self._lock:
            self.stages.append(TraceStage(stage=stage, input=input, output=output))

    def set_answer_type(self, answer_type: str) -> None:
        with self._lock:
            self.answer_type = answer_type

    # ------------------------------------------------------------------
    # 序列化 / 落盘
    # ------------------------------------------------------------------

    def finalize(self) -> dict[str, Any]:
        """结束追踪：记录结束时间，返回完整 trace 字典（并落盘 + 入内存）。"""
        self.finished_at = time.time()
        data = self.to_dict()
        try:
            TRACES_DIR.mkdir(parents=True, exist_ok=True)
            path = TRACES_DIR / f"{self.trace_id}.json"
            path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            logger.warning(f"trace 落盘失败: {e}")
        register_trace(data)
        return data

    def to_dict(self) -> dict[str, Any]:
        """转为可序列化字典（供 API / 落盘）。"""
        with self._lock:
            stages = [asdict(s) for s in self.stages]
        duration_ms = None
        if self.finished_at is not None:
            duration_ms = round((self.finished_at - self.started_at) * 1000)
        return {
            "trace_id": self.trace_id,
            "question": self.question,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_ms": duration_ms,
            "answer_type": self.answer_type,
            "stage_count": len(stages),
            "stages": stages,
        }


# ==============================================================================
# 模块级状态：当前 tracer + 内存 trace 列表
# ==============================================================================

_memory_lock = threading.Lock()
_memory_traces: list[dict[str, Any]] = []


def register_trace(data: dict[str, Any]) -> None:
    """把已完成的 trace 放入内存列表（供 /api/traces 实时查询）。"""
    global _memory_traces
    with _memory_lock:
        _memory_traces.append(data)
        if len(_memory_traces) > MAX_IN_MEMORY:
            _memory_traces = _memory_traces[-MAX_IN_MEMORY:]


def list_traces(limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
    """按时间倒序返回最近的 trace 摘要列表。"""
    with _memory_lock:
        items = list(_memory_traces)
    items.sort(key=lambda d: d.get("started_at", 0), reverse=True)
    return items[offset: offset + limit]


def get_trace(trace_id: str) -> dict[str, Any] | None:
    """按 trace_id 查内存中的完整 trace。"""
    with _memory_lock:
        for d in _memory_traces:
            if d.get("trace_id") == trace_id:
                return d
    return None


# ==============================================================================
# 上下文访问入口
# ==============================================================================


def get_tracer() -> RAGTracer | None:
    """返回当前查询的 tracer（无则返回 None）。"""
    return _current_tracer.get()


def begin_trace(question: str) -> RAGTracer:
    """创建 tracer 并设为当前查询的上下文（在 query 入口调用）。"""
    tracer = RAGTracer(question)
    _current_tracer.set(tracer)
    return tracer


def end_trace() -> dict[str, Any] | None:
    """结束当前查询的追踪并清理上下文。返回落盘的 trace 数据。"""
    tracer = _current_tracer.get()
    if tracer is None:
        return None
    _current_tracer.set(None)
    return tracer.finalize()
