"""
=============================================================================
滑动窗口限流器

支持 IP 限流和全局限流，使用 Redis 有序集合实现（不可用时降级为内存）。
=============================================================================
"""

import time
from typing import Optional

from src.ratelimit.redis_client import get_redis, get_queue, MemoryQueue
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class SlidingWindowLimiter:
    """
    滑动窗口限流器。

    统计指定时间窗口内的请求数，超过阈值则拒绝。
    """

    def __init__(self, window: int = 1, max_requests: int = 10, name: str = "default"):
        """
        Args:
            window:       时间窗口（秒）
            max_requests: 窗口内最大请求数
            name:         限流器名称
        """
        self.window = window
        self.max_requests = max_requests
        self.name = name

    def _redis_key(self, key: str) -> str:
        return f"ratelimit:{self.name}:{key}"

    def allow(self, key: str) -> tuple[bool, int, int]:
        """
        检查是否允许请求。

        Args:
            key: 限流键（如 IP 地址、用户 ID）

        Returns:
            (allowed, count, limit)
            - allowed: 是否允许请求
            - count:   当前窗口内的请求数
            - limit:   窗口限制数
        """
        now = time.time()
        window_start = now - self.window

        # 尝试 Redis
        redis = get_redis()
        if redis is not None:
            try:
                redis_key = self._redis_key(key)
                pipeline = redis.pipeline()
                pipeline.zremrangebyscore(redis_key, 0, window_start)
                pipeline.zcard(redis_key)
                pipeline.expire(redis_key, self.window + 1)
                results = pipeline.execute()
                count = results[1]

                if count >= self.max_requests:
                    return False, count, self.max_requests

                pipeline = redis.pipeline()
                pipeline.zadd(redis_key, {str(now): now})
                pipeline.expire(redis_key, self.window + 1)
                pipeline.execute()
                return True, count + 1, self.max_requests
            except Exception as e:
                logger.warning(f"Redis 限流异常，降级到内存: {e}")

        # 降级到内存
        queue = get_queue()
        queue.add(self._redis_key(key), now, maxlen=self.max_requests * 10)
        count = queue.count_since(self._redis_key(key), window_start)

        if count >= self.max_requests:
            return False, count, self.max_requests

        return True, count + 1, self.max_requests


# 全局限流器实例
_ip_limiter = SlidingWindowLimiter(window=1, max_requests=10, name="ip")  # 每 IP 10 QPS
_global_limiter = SlidingWindowLimiter(window=1, max_requests=50, name="global")  # 全局 50 QPS


def check_ip_limit(ip: str) -> tuple[bool, int, int]:
    """检查 IP 限流"""
    return _ip_limiter.allow(ip)


def check_global_limit() -> tuple[bool, int, int]:
    """检查全局限流"""
    return _global_limiter.allow("global")