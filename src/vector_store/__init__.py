"""
向量存储模块

导出两个向量存储实现：
    - VectorStoreManager: 基于 ChromaDB（嵌入式，无需外部服务）
    - PGVectorStore:      基于 PostgreSQL + pgvector（生产推荐）

PGVectorStore 为惰性导入，仅在使用时才会加载 psycopg2，
避免在只使用 ChromaDB 时强制依赖 PostgreSQL 驱动。
"""

from src.vector_store.manager import VectorStoreManager


def PGVectorStore(*args, **kwargs):
    """惰性导入 PGVectorStore，避免 ChromaDB 模式下依赖 psycopg2。"""
    from src.vector_store.pg_manager import PGVectorStore as _PGVectorStore
    return _PGVectorStore(*args, **kwargs)


__all__ = ["VectorStoreManager", "PGVectorStore"]
