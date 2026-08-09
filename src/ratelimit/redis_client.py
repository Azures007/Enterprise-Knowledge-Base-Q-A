"""
=============================================================================
Redis 连接管理（带内存降级）

如果 Redis 不可用，自动降级为内存模式，不影响系统运行。
=============================================================================
"""

import json
import time
from collections import defaultdict, deque
from threading import Lock
from typing import Any, Optional

from src.monitoring import record_redis_degraded, record_rate_limit_degraded
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

_redis_client = None
_redis_available = False


class MemoryQueue:
    """内存队列（Redis 降级时使用）"""

    def __init__(self):
        self._data: dict[str, deque] = defaultdict(deque)
        self._lock = Lock()

    def add(self, key: str, value: Any, maxlen: int = 1000):
        with self._lock:
            q = self._data[key]
            q.append((time.time(), value))
            # 清理过期数据
            cutoff = time.time() - 3600
            while q and q[0][0] < cutoff:
                q.popleft()
            if len(q) > maxlen:
                while len(q) > maxlen:
                    q.popleft()

    def count_since(self, key: str, since: float) -> int:
        with self._lock:
            q = self._data.get(key, deque())
            # 清理过期
            while q and q[0][0] < since:
                q.popleft()
            return len(q)

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            q = self._data.get(key, deque())
            return q[-1][1] if q else default

    def set(self, key: str, value: Any):
        with self._lock:
            self._data[key].clear()
            self._data[key].append((time.time(), value))

    def increment(self, key: str, amount: int = 1) -> int:
        with self._lock:
            now = time.time()
            q = self._data[key]
            # 清理 1 秒前的数据
            while q and q[0][0] < now - 1:
                q.popleft()
            # 加一个计数
            q.append((now, amount))
            return sum(v for _, v in q)

    def clear(self):
        with self._lock:
            self._data.clear()


def get_redis():
    """获取 Redis 连接"""
    global _redis_client, _redis_available
    if _redis_client is not None:
        return _redis_client
    if _redis_available is False:
        return None

    try:
        import redis as redis_module
        from config.settings import settings

        _redis_client = redis_module.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            db=settings.REDIS_DB,
            socket_connect_timeout=2,
            socket_timeout=2,
            decode_responses=True,
        )
        _redis_client.ping()
        _redis_available = True
        logger.info("Redis 连接成功")
        return _redis_client
    except Exception as e:
        _redis_available = False
        _redis_client = None
        record_redis_degraded(f"Redis 连接失败: {e}")
        record_rate_limit_degraded("Redis 不可用，限流降级为内存模式")
        logger.warning(f"Redis 不可用，使用内存限流: {e}")
        return None


# 全局内存队列实例（Redis 降级时使用）
_memory_queue = MemoryQueue()


def get_queue() -> MemoryQueue:
    """获取队列（Redis 或内存）"""
    return _memory_queue