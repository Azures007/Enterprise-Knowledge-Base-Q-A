"""
=============================================================================
令牌桶算法

用于保护百炼 API 配额，以固定速率补充令牌，防止突发流量打爆 API。
=============================================================================

使用方法:
    from src.ratelimit.token_bucket import TokenBucket

    bucket = TokenBucket(rate=10, capacity=20)  # 每秒 10 个，最多累积 20 个
    if bucket.consume():
        # 调用 API
    else:
        # 限流
"""

import time
from threading import Lock


class TokenBucket:
    """
    令牌桶限流器。

    以固定速率向桶中添加令牌，桶满则停止添加。
    请求到来时消耗一个令牌，如果没有令牌则拒绝请求。
    """

    def __init__(self, rate: float = 10, capacity: int = 20, name: str = "default"):
        """
        Args:
            rate:     令牌补充速率（个/秒）
            capacity: 桶容量（最大累积令牌数）
            name:     桶名称（用于日志）
        """
        self.rate = rate
        self.capacity = capacity
        self.name = name
        self._tokens = float(capacity)
        self._last_refill = time.time()
        self._lock = Lock()

    def _refill(self):
        """补充令牌"""
        now = time.time()
        elapsed = now - self._last_refill
        self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)
        self._last_refill = now

    def consume(self, tokens: int = 1) -> bool:
        """
        消耗令牌。

        Args:
            tokens: 消耗的令牌数

        Returns:
            True 表示成功消耗，False 表示令牌不足
        """
        with self._lock:
            self._refill()
            if self._tokens >= tokens:
                self._tokens -= tokens
                return True
            return False

    @property
    def available(self) -> float:
        """当前可用令牌数"""
        with self._lock:
            self._refill()
            return self._tokens

    @property
    def usage_ratio(self) -> float:
        """当前使用率（0~1）"""
        return 1.0 - (self.available / self.capacity)


# 全局令牌桶
_buckets: dict[str, TokenBucket] = {}
_buckets_lock = Lock()


def get_token_bucket(name: str, rate: float = 10, capacity: int = 20) -> TokenBucket:
    """获取或创建令牌桶"""
    global _buckets
    with _buckets_lock:
        if name not in _buckets:
            _buckets[name] = TokenBucket(rate, capacity, name)
        return _buckets[name]


def get_all_buckets() -> dict[str, TokenBucket]:
    """获取所有令牌桶（用于查看使用率）"""
    return dict(_buckets)