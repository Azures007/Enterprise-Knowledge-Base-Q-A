import hashlib
import json
import time
from collections import OrderedDict
from threading import Lock
from typing import Any, Optional

from config.settings import settings
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class MemoryCache:
    """
    带 LRU 淘汰策略的线程安全内存缓存。

    使用 OrderedDict 维护插入/访问顺序：每次 get/set 将键移到末尾，
    当条目数超过 max_size 时从头部（最久未使用）淘汰。
    """

    def __init__(self, max_size: int | None = None):
        self._max_size = max_size or getattr(settings, "CACHE_MAX_ENTRIES", 10000)
        self._data: OrderedDict[str, tuple[float, str]] = OrderedDict()
        self._tag_index: dict[str, set[str]] = {}
        self._lock = Lock()

    def get(self, key: str) -> Optional[str]:
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                return None
            expires_at, value = entry
            if time.time() > expires_at:
                del self._data[key]
                # 清理 tag 索引
                self._cleanup_key_tags(key)
                return None
            # LRU：命中后移到末尾（最近使用）
            self._data.move_to_end(key)
            return value

    def set(self, key: str, value: str, ttl: int, tags: list[str] | None = None):
        with self._lock:
            self._data[key] = (time.time() + ttl, value)
            # 新写入视为最近使用
            self._data.move_to_end(key)
            if tags:
                for tag in tags:
                    if tag not in self._tag_index:
                        self._tag_index[tag] = set()
                    self._tag_index[tag].add(key)
            self._evict_if_needed()

    def _evict_if_needed(self):
        """超过最大容量时，从头部淘汰最久未使用的条目"""
        overflow = len(self._data) - self._max_size
        if overflow <= 0:
            return
        evicted = 0
        while len(self._data) > self._max_size:
            oldest_key, _ = self._data.popitem(last=False)
            self._cleanup_key_tags(oldest_key)
            evicted += 1
        if evicted:
            logger.debug(f"缓存已淘汰 {evicted} 条 LRU 条目（容量 {self._max_size}）")

    def clear(self):
        with self._lock:
            self._data.clear()
            self._tag_index.clear()

    def clear_by_tags(self, tags: list[str]):
        with self._lock:
            keys_to_delete = set()
            for tag in tags:
                if tag in self._tag_index:
                    keys_to_delete.update(self._tag_index[tag])
                    del self._tag_index[tag]
            for k in keys_to_delete:
                self._data.pop(k, None)
            logger.debug(f"按标签清除了 {len(keys_to_delete)} 条缓存")

    def _cleanup_key_tags(self, key: str):
        tags_to_remove = [t for t, keys in self._tag_index.items() if key in keys]
        for tag in tags_to_remove:
            self._tag_index[tag].discard(key)
            if not self._tag_index[tag]:
                del self._tag_index[tag]

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._data)

    @property
    def max_size(self) -> int:
        return self._max_size


class QACache:
    def __init__(self, ttl: int | None = None, prefix: str = "qa_cache:"):
        self.ttl = ttl or getattr(settings, "CACHE_QA_TTL", 3600)
        self.prefix = prefix
        self._tag_index_prefix = "qa_cache:tag_index:"
        self._redis = None
        self._redis_available = False
        self._memory = MemoryCache(max_size=getattr(settings, "CACHE_MAX_ENTRIES", 10000))
        self._init_redis()

    def _init_redis(self):
        try:
            import redis as redis_module
            self._redis = redis_module.Redis(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                db=settings.REDIS_DB + 1,
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
        import re
        text = question.lower().strip()
        text = re.sub(
            r'^(请问|我想问一下|我想知道|能不能告诉我|可以告诉我|麻烦告诉我|请告诉我|问一下|想问下|咨询一下|请教一下|谁能告诉我)',
            '', text
        ).strip()
        text = re.sub(r'[，。！？；：、,.!?;:()（）【】\[\]""''「」『』]', '', text).strip()
        text = re.sub(r'\s+', ' ', text).strip()
        h = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return f"{self.prefix}{h}"

    def _get_tag_index_key(self, tag: str) -> str:
        return f"{self._tag_index_prefix}{hashlib.sha256(tag.encode()).hexdigest()}"

    def get(self, question: str) -> Optional[dict[str, Any]]:
        key = self._make_key(question)
        if self._redis_available and self._redis:
            try:
                data = self._redis.get(key)
                if data:
                    logger.info(f"缓存命中: {question[:50]}...")
                    return json.loads(data)
            except Exception:
                pass
        data = self._memory.get(key)
        if data:
            logger.info(f"缓存命中(内存): {question[:50]}...")
            return json.loads(data)
        return None

    def set(self, question: str, result: dict[str, Any], tags: list[str] | None = None):
        key = self._make_key(question)
        value = json.dumps(result, ensure_ascii=False)
        if self._redis_available and self._redis:
            try:
                self._redis.setex(key, self.ttl, value)
                if tags:
                    for tag in tags:
                        idx_key = self._get_tag_index_key(tag)
                        self._redis.hset(idx_key, key, "1")
                        self._redis.expire(idx_key, self.ttl)
                logger.info(f"缓存已设置: {question[:50]}... ({self.ttl}s)")
                return
            except Exception:
                pass
        self._memory.set(key, value, self.ttl, tags)
        logger.info(f"缓存已设置(内存): {question[:50]}... ({self.ttl}s)")

    def clear(self):
        if self._redis_available and self._redis:
            try:
                keys = self._redis.keys(f"{self.prefix}*")
                if keys:
                    self._redis.delete(*keys)
                tag_keys = self._redis.keys(f"{self._tag_index_prefix}*")
                if tag_keys:
                    self._redis.delete(*tag_keys)
                logger.info(f"缓存已清空 (Redis, {len(keys)} 条)")
            except Exception:
                pass
        self._memory.clear()
        logger.info("缓存已清空 (内存)")

    def invalidate_by_tags(self, tags: list[str]):
        """按标签批量删除缓存条目"""
        if not tags:
            return
        if self._redis_available and self._redis:
            try:
                all_keys = set()
                for tag in tags:
                    idx_key = self._get_tag_index_key(tag)
                    key_map = self._redis.hgetall(idx_key)
                    all_keys.update(key_map.keys())
                    self._redis.delete(idx_key)
                if all_keys:
                    self._redis.delete(*all_keys)
                    logger.info(f"按标签清除了 {len(all_keys)} 条缓存 (Redis)")
            except Exception as e:
                logger.warning(f"Redis 按标签清除失败: {e}")
                return
        self._memory.clear_by_tags(tags)

    @property
    def size(self) -> int:
        if self._redis_available and self._redis:
            try:
                return len(self._redis.keys(f"{self.prefix}*"))
            except Exception:
                pass
        return self._memory.size


qa_cache = QACache()