"""
=============================================================================
问答缓存模块

缓存高频问题的回答，避免重复向量检索和 LLM 调用。
存储后端：Redis（不可用时降级为内存字典）。
=============================================================================

使用方法:
    from src.cache import QACache

    cache = QACache()
    result = cache.get("公司的考勤制度是什么？")
    if result:
        # 使用缓存结果
    else:
        # 正常 RAG 流程
        cache.set("公司的考勤制度是什么？", answer_data)
"""

import hashlib
import json
import time
from threading import Lock
from typing import Any, Optional

from config.settings import settings
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class MemoryCache:
    """内存缓存（Redis 降级时使用）"""

    def __init__(self):
        self._data: dict[str, tuple[float, str]] = {}
        self._lock = Lock()

    def get(self, key: str) -> Optional[str]:
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                return None
            expires_at, value = entry
            if time.time() > expires_at:
                del self._data[key]
                return None
            return value

    def set(self, key: str, value: str, ttl: int):
        with self._lock:
            self._data[key] = (time.time() + ttl, value)

    def clear(self):
        with self._lock:
            self._data.clear()

    def clear_by_prefix(self, prefix: str):
        with self._lock:
            keys = [k for k in self._data if k.startswith(prefix)]
            for k in keys:
                del self._data[k]

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._data)


class QACache:
    """
    问答缓存。

    缓存键：问题的 SHA256 哈希
    缓存值：JSON 序列化的回答结果
    """

    def __init__(self, ttl: int = 3600, prefix: str = "qa_cache:"):
        """
        Args:
            ttl:    缓存过期时间（秒），默认 1 小时
            prefix: Redis 键前缀
        """
        self.ttl = ttl
        self.prefix = prefix
        self._redis = None
        self._redis_available = False
        self._memory = MemoryCache()
        self._init_redis()

    def _init_redis(self):
        """尝试连接 Redis"""
        try:
            import redis as redis_module

            self._redis = redis_module.Redis(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                db=settings.REDIS_DB + 1,  # 使用不同数据库，避免与限流冲突
                socket_connect_timeout=2,
                socket_timeout=2,
                decode_responses=True,
            )
            self._redis.ping()
            self._redis_available = True
            logger.info("问答缓存: Redis 连接成功")
        except Exception as e:
            self._redis_available = False
            self._redis = None
            logger.info(f"问答缓存: Redis 不可用，使用内存缓存 ({e})")

    def _make_key(self, question: str) -> str:
        """生成缓存键（先归一化，再哈希）"""
        import re
        # 归一化处理
        text = question.lower().strip()
        # 去掉常见前缀
        text = re.sub(
            r'^(请问|我想问一下|我想知道|能不能告诉我|可以告诉我|麻烦告诉我|请告诉我|问一下|想问下|咨询一下|请教一下|谁能告诉我)',
            '', text
        ).strip()
        # 去掉标点符号
        text = re.sub(r'[，。！？；：、,.!?;:()（）【】\[\]""''「」『』]', '', text).strip()
        # 去掉多余空格
        text = re.sub(r'\s+', ' ', text).strip()
        h = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return f"{self.prefix}{h}"

    def get(self, question: str) -> Optional[dict[str, Any]]:
        """
        获取缓存。

        Args:
            question: 用户问题

        Returns:
            缓存的回答数据，或 None
        """
        key = self._make_key(question)

        if self._redis_available and self._redis:
            try:
                data = self._redis.get(key)
                if data:
                    logger.info(f"缓存命中: {question[:50]}...")
                    return json.loads(data)
            except Exception:
                pass

        # 降级到内存
        data = self._memory.get(key)
        if data:
            logger.info(f"缓存命中(内存): {question[:50]}...")
            return json.loads(data)

        return None

    def set(self, question: str, result: dict[str, Any]):
        """
        设置缓存。

        Args:
            question: 用户问题
            result:   回答数据（含 answer, sources, answer_type）
        """
        key = self._make_key(question)
        value = json.dumps(result, ensure_ascii=False)

        if self._redis_available and self._redis:
            try:
                self._redis.setex(key, self.ttl, value)
                logger.info(f"缓存已设置: {question[:50]}... ({self.ttl}s)")
                return
            except Exception:
                pass

        # 降级到内存
        self._memory.set(key, value, self.ttl)
        logger.info(f"缓存已设置(内存): {question[:50]}... ({self.ttl}s)")

    def clear(self):
        """清空所有缓存"""
        if self._redis_available and self._redis:
            try:
                keys = self._redis.keys(f"{self.prefix}*")
                if keys:
                    self._redis.delete(*keys)
                logger.info(f"缓存已清空 (Redis, {len(keys)} 条)")
            except Exception:
                pass

        self._memory.clear()
        logger.info("缓存已清空 (内存)")

    @property
    def size(self) -> int:
        """缓存条目数"""
        if self._redis_available and self._redis:
            try:
                return len(self._redis.keys(f"{self.prefix}*"))
            except Exception:
                pass
        return self._memory.size


# 全局缓存实例
qa_cache = QACache()