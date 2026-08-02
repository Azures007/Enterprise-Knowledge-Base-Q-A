"""
=============================================================================
降级管理器

根据当前系统负载（百炼 API 使用率）自动切换降级级别。
=============================================================================
"""

from enum import IntEnum
from typing import Any

from src.ratelimit.token_bucket import get_token_bucket, get_all_buckets
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class DegradeLevel(IntEnum):
    """降级级别"""
    NORMAL = 0      # 正常
    LIGHT = 1       # 轻度降级：非核心功能走本地
    MEDIUM = 2      # 中度降级：所有请求返回缓存或提示
    HEAVY = 3       # 重度降级：直接返回 429


class DegradeManager:
    """
    降级管理器。

    根据百炼 API 令牌桶的使用率，自动切换降级级别。
    """

    def __init__(self):
        self._current_level = DegradeLevel.NORMAL
        self._check_interval = 5  # 每 5 秒检查一次
        self._last_check = 0

    def _evaluate(self) -> DegradeLevel:
        """评估当前应该使用的降级级别"""
        import time

        now = time.time()
        if now - self._last_check < self._check_interval:
            return self._current_level
        self._last_check = now

        # 获取百炼 API 令牌桶的使用率
        buckets = get_all_buckets()
        if not buckets:
            return DegradeLevel.NORMAL

        # 取最高使用率
        max_usage = max(b.usage_ratio for b in buckets.values())

        if max_usage >= 0.95:
            new_level = DegradeLevel.HEAVY
        elif max_usage >= 0.85:
            new_level = DegradeLevel.MEDIUM
        elif max_usage >= 0.70:
            new_level = DegradeLevel.LIGHT
        else:
            new_level = DegradeLevel.NORMAL

        if new_level != self._current_level:
            logger.info(
                f"降级级别变更: {self._current_level.name} → {new_level.name} "
                f"(使用率: {max_usage:.0%})"
            )
            self._current_level = new_level

        return self._current_level

    @property
    def level(self) -> DegradeLevel:
        """当前降级级别"""
        return self._evaluate()

    @property
    def level_name(self) -> str:
        """当前降级级别名称"""
        return self.level.name

    def is_allowed(self, is_core_request: bool = True) -> tuple[bool, str]:
        """
        检查当前降级级别是否允许请求。

        Args:
            is_core_request: 是否是核心请求（问答）/ 非核心（标题生成等）

        Returns:
            (allowed, message)
        """
        level = self.level

        if level == DegradeLevel.HEAVY:
            return False, "系统负载过高，请稍后再试"

        if level == DegradeLevel.MEDIUM:
            return False, "当前访问量较大，请稍后再试"

        if level == DegradeLevel.LIGHT and not is_core_request:
            return False, "系统负载较高，非核心功能暂时关闭"

        return True, ""


# 全局降级管理器实例
degrade_manager = DegradeManager()