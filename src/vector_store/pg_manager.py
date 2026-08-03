"""
=============================================================================
PostgreSQL + pgvector 向量存储模块

替代 ChromaDB，使用 PostgreSQL 统一存储所有数据：
    - collections    → 知识库集合
    - documents      → 原始文件元数据
    - chunks         → 分块文本 + 向量 + 元数据
    - conversations  → 对话
    - messages       → 消息

依赖: psycopg2-binary, pgvector（PostgreSQL 扩展）
=============================================================================

使用方法:
    from src.vector_store import PGVectorStore
    from src.embeddings import BailianEmbeddings

    embedder = BailianEmbeddings()
    store = PGVectorStore(embedder)
    store.add_documents(docs)
    results = store.similarity_search("公司考勤", k=5)
"""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import psycopg2
import psycopg2.extras
import psycopg2.pool

from config.settings import settings
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class PGVectorStoreError(Exception):
    """PostgreSQL 向量存储异常"""
    pass


class _PooledConnection:
    """
    连接池连接包装器。

    在退出 with 代码块时自动将连接归还连接池，
    保持原有 `with self._get_conn() as conn:` 调用方式不变。
    """

    def __init__(self, pool, conn):
        self._pool = pool
        self._conn = conn

    def __getattr__(self, name):
        return getattr(self._conn, name)

    def __enter__(self):
        return self._conn

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._pool.putconn(self._conn)
        return False


class PGVectorStore:
    """
    基于 PostgreSQL + pgvector 的向量存储管理器。

    与 ChromaDB 的 VectorStoreManager 接口兼容，可直接替换。
    """

    def __init__(
        self,
        embedder: Any,
        collection_name: str | None = None,
        connection_string: str | None = None,
    ):
        """
        Args:
            embedder:        嵌入模型实例
            collection_name: 默认集合名称
            connection_string: PostgreSQL 连接字符串
        """
        self.embedder = embedder
        self.collection_name = collection_name or settings.DEFAULT_COLLECTION
        self.conn_string = connection_string or settings.DATABASE_URL

        # 初始化数据库表
        self._init_db()
        # 不在这里确保集合存在，避免删除后自动重建
        # 集合会在 add_documents / switch_collection 时按需创建

        self._dimension = settings.VECTOR_DIMENSION

        logger.info(
            f"PGVectorStore 初始化完成: collection={collection_name}"
        )

    # ================================================================
    # 数据库初始化
    # ================================================================

    def _get_conn(self):
        """
        从连接池获取数据库连接。

        返回的 _PooledConnection 支持 with 语法，退出时自动归还连接池。
        """
        if not hasattr(self, "_pool"):
            self._pool = psycopg2.pool.ThreadedConnectionPool(
                minconn=1,
                maxconn=20,
                dsn=self.conn_string,
            )
            logger.info(f"数据库连接池已创建 (min=1, max=20)")
        try:
            conn = self._pool.getconn()
            conn.autocommit = True
            return _PooledConnection(self._pool, conn)
        except psycopg2.pool.PoolError as e:
            raise PGVectorStoreError(f"获取数据库连接失败（连接池已耗尽）: {e}") from e

    def _init_db(self):
        """创建所有必要的表和扩展"""
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                # 启用 pgvector 扩展
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")

                # 集合表
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS collections (
                        id          SERIAL PRIMARY KEY,
                        name        VARCHAR(255) UNIQUE NOT NULL,
                        description TEXT,
                        created_at  TIMESTAMP DEFAULT NOW()
                    )
                """)

                # 文档记录表（原始文件元数据）
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS documents (
                        id              SERIAL PRIMARY KEY,
                        collection_id   INTEGER REFERENCES collections(id) ON DELETE CASCADE,
                        filename        VARCHAR(255) NOT NULL,
                        file_type       VARCHAR(20),
                        file_size       INTEGER,
                        content_hash    VARCHAR(64),
                        storage_backend VARCHAR(20) DEFAULT 'local',
                        storage_path    TEXT,
                        created_at      TIMESTAMP DEFAULT NOW()
                    )
                """)

                # 文本块 + 向量表
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS chunks (
                        id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        collection_id   INTEGER NOT NULL REFERENCES collections(id) ON DELETE CASCADE,
                        document_id     INTEGER REFERENCES documents(id) ON DELETE SET NULL,
                        content         TEXT NOT NULL,
                        metadata        JSONB DEFAULT '{{}}'::jsonb,
                        embedding       VECTOR({settings.VECTOR_DIMENSION}),
                        created_at      TIMESTAMP DEFAULT NOW()
                    )
                """)

                # 对话表
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS conversations (
                        id          SERIAL PRIMARY KEY,
                        title       VARCHAR(255) DEFAULT '新对话',
                        created_at  TIMESTAMP DEFAULT NOW(),
                        updated_at  TIMESTAMP DEFAULT NOW()
                    )
                """)

                # 消息表
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS messages (
                        id              SERIAL PRIMARY KEY,
                        conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                        role            VARCHAR(10) CHECK (role IN ('user','ai','system')),
                        content         TEXT NOT NULL DEFAULT '',
                        sources         JSONB DEFAULT '[]'::jsonb,
                        answer_type     VARCHAR(20),
                        created_at      TIMESTAMP DEFAULT NOW()
                    )
                """)

                # 向量索引（HNSW）
                cur.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw
                    ON chunks
                    USING hnsw (embedding vector_cosine_ops)
                    WITH (m = 16, ef_construction = 200)
                """)

                # 普通索引
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_chunks_collection
                    ON chunks(collection_id)
                """)
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_messages_conversation
                    ON messages(conversation_id, created_at)
                """)

                # 导入审计日志表
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS ingest_audit_log (
                        id          SERIAL PRIMARY KEY,
                        collection  VARCHAR(255),
                        filename    VARCHAR(255),
                        action      VARCHAR(20),
                        doc_id      INTEGER,
                        old_doc_id  INTEGER,
                        old_content_hash VARCHAR(64),
                        chunks_added INTEGER,
                        file_size   INTEGER,
                        status      VARCHAR(20),
                        error_msg   TEXT,
                        created_at  TIMESTAMP DEFAULT NOW()
                    )
                """)

        logger.info("PostgreSQL 数据库表初始化完成")

    def _ensure_collection(self, name: str) -> int:
        """确保集合存在，不存在则创建，返回集合 ID"""
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id FROM collections WHERE name = %s", (name,)
                )
                row = cur.fetchone()
                if row:
                    return row[0]

                cur.execute(
                    "INSERT INTO collections (name) VALUES (%s) RETURNING id",
                    (name,),
                )
                coll_id = cur.fetchone()[0]
                logger.info(f"创建集合 '{name}' (id={coll_id})")
                return coll_id

    def _get_collection_id(self, name: str) -> int | None:
        """获取集合 ID，不存在则返回 None（不会自动创建）"""
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id FROM collections WHERE name = %s", (name,)
                )
                row = cur.fetchone()
                return row[0] if row else None

    # ================================================================
    # 文档管理
    # ================================================================

    def add_documents(
        self,
        documents: list[dict[str, Any]],
        batch_size: int = 64,
        document_id: int | None = None,
    ) -> int:
        """
        向向量库中添加文档块。

        Args:
            documents: 文档块列表，每项包含 'page_content' 和 'metadata'
            batch_size: 每批处理的文档数
            document_id: 关联的文档记录 ID（与 OSS 原始文件对应）

        Returns:
            成功添加的文档块数量
        """
        if not documents:
            return 0

        texts = [doc["page_content"] for doc in documents]
        metadatas = [doc.get("metadata", {}) for doc in documents]

        # 计算嵌入向量
        logger.info(f"正在计算 {len(texts)} 个文档块的嵌入向量...")
        try:
            embeddings = self.embedder.embed_documents(texts)
        except Exception as e:
            raise PGVectorStoreError(f"嵌入计算失败: {e}") from e

        coll_id = self._ensure_collection(self.collection_name)

        # 批量写入 PostgreSQL
        added_count = 0
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                for i in range(0, len(texts), batch_size):
                    batch_end = min(i + batch_size, len(texts))

                    values = []
                    for j in range(i, batch_end):
                        meta_json = json.dumps(metadatas[j], ensure_ascii=False)
                        embedding_str = "[" + ",".join(str(v) for v in embeddings[j]) + "]"
                        values.append((
                            coll_id,
                            document_id,
                            texts[j],
                            meta_json,
                            embedding_str,
                        ))

                    psycopg2.extras.execute_values(
                        cur,
                        """
                        INSERT INTO chunks (collection_id, document_id, content, metadata, embedding)
                        VALUES %s
                        """,
                        values,
                        template="(%s, %s, %s, %s::jsonb, %s::vector)",
                    )

                    added_count += len(values)
                    logger.debug(f"已添加 {added_count}/{len(texts)} 个文档块")

        self.log_audit(
            action="ingest",
            filename=metadatas[0].get("filename", "unknown") if metadatas else "unknown",
            status="success",
            doc_id=document_id,
            chunks_added=added_count,
        )
        logger.info(f"文档入库完成: 共 {added_count} 个文档块")
        return added_count

    def add_document_record(
        self,
        filename: str,
        file_type: str,
        file_size: int,
        content_hash: str = "",
        storage_path: str = "",
        storage_backend: str = "local",
    ) -> int:
        """记录原始文件元数据到 documents 表，返回文档 ID"""
        coll_id = self._ensure_collection(self.collection_name)

        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO documents (collection_id, filename, file_type, file_size, content_hash, storage_backend, storage_path)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                    """,
                    (coll_id, filename, file_type, file_size, content_hash, storage_backend, storage_path),
                )
                doc_id = cur.fetchone()[0]
        logger.info(f"文档记录已创建: {filename} (id={doc_id})")
        return doc_id

    def log_audit(
        self,
        action: str,
        filename: str,
        status: str = "success",
        error_msg: str | None = None,
        **kwargs: Any,
    ):
        """
        记录导入/删除操作的审计日志。

        Args:
            action:    "ingest" / "delete"
            filename:  文件名
            status:    "success" / "error" / "overwrite"
            error_msg: 错误信息（可选）
            **kwargs:  额外字段 (doc_id, old_doc_id, old_content_hash, chunks_added, file_size)
        """
        try:
            with self._get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO ingest_audit_log
                        (collection, filename, action, doc_id, old_doc_id,
                         old_content_hash, chunks_added, file_size, status, error_msg)
                        VALUES
                        (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            self.collection_name,
                            filename,
                            action,
                            kwargs.get("doc_id"),
                            kwargs.get("old_doc_id"),
                            kwargs.get("old_content_hash"),
                            kwargs.get("chunks_added"),
                            kwargs.get("file_size"),
                            status,
                            error_msg,
                        ),
                    )
            logger.debug(f"审计日志已记录: {action} {filename} [{status}]")
        except Exception as e:
            logger.warning(f"审计日志记录失败: {e}")

    def find_document_by_hash(self, content_hash: str) -> dict | None:
        """
        根据内容哈希查找是否已存在相同文档。

        Returns:
            如果找到，返回文档信息 dict；否则返回 None
        """
        coll_id = self._get_collection_id(self.collection_name)
        if coll_id is None:
            return None
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, filename, file_size, created_at
                    FROM documents
                    WHERE collection_id = %s AND content_hash = %s
                    ORDER BY created_at DESC
                    LIMIT 1
                    """,
                    (coll_id, content_hash),
                )
                row = cur.fetchone()

        if row is None:
            return None

        return {
            "id": row[0],
            "filename": row[1],
            "file_size": row[2],
            "created_at": row[3].isoformat() if row[3] else None,
        }

    def find_document_by_filename(self, filename: str) -> dict | None:
        """
        根据文件名查找是否已存在同名文档（同一集合内）。

        Returns:
            如果找到，返回文档信息 dict；否则返回 None
        """
        coll_id = self._get_collection_id(self.collection_name)
        if coll_id is None:
            return None
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, filename, file_size, created_at
                    FROM documents
                    WHERE collection_id = %s AND filename = %s
                    ORDER BY created_at DESC
                    LIMIT 1
                    """,
                    (coll_id, filename),
                )
                row = cur.fetchone()

        if row is None:
            return None

        return {
            "id": row[0],
            "filename": row[1],
            "file_size": row[2],
            "created_at": row[3].isoformat() if row[3] else None,
        }

    def list_documents(self) -> list[dict]:
        """
        获取当前集合中的所有文档列表（含 chunk 数量）。

        Returns:
            list[dict]: 每项包含 id, filename, file_type, file_size, storage_backend, storage_path, created_at, chunk_count
        """
        coll_id = self._get_collection_id(self.collection_name)
        if coll_id is None:
            return None
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT d.id, d.filename, d.file_type, d.file_size,
                           d.storage_backend, d.storage_path, d.created_at,
                           (
                             SELECT COUNT(*) FROM chunks c
                             WHERE c.document_id = d.id
                                OR (c.document_id IS NULL AND c.metadata->>'filename' = d.filename)
                           ) AS chunk_count
                    FROM documents d
                    WHERE d.collection_id = %s
                    ORDER BY d.created_at DESC
                    """,
                    (coll_id,),
                )
                rows = cur.fetchall()

        return [
            {
                "id": r[0],
                "filename": r[1],
                "file_type": r[2],
                "file_size": r[3],
                "storage_backend": r[4],
                "storage_path": r[5],
                "created_at": r[6].isoformat() if r[6] else None,
                "chunk_count": r[7],
            }
            for r in rows
        ]

    def delete_document(self, doc_id: int, delete_storage: bool = True) -> dict | None:
        """
        删除文档及其所有分块，可选择同时删除存储文件。

        Args:
            doc_id:         文档 ID
            delete_storage: 是否删除存储后端中的原始文件

        Returns:
            被删除文档的信息 dict；如果不存在则返回 None
        """
        # 先获取文档信息
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT filename, file_type, file_size, storage_backend, storage_path, content_hash FROM documents WHERE id = %s",
                    (doc_id,),
                )
                row = cur.fetchone()

        if row is None:
            return None

        doc_info = {
            "id": doc_id,
            "filename": row[0],
            "file_type": row[1],
            "file_size": row[2],
            "storage_backend": row[3],
            "storage_path": row[4],
            "content_hash": row[5],
        }

        # 删除存储后端中的原始文件
        if delete_storage and doc_info["storage_path"]:
            try:
                from src.storage import get_storage
                storage = get_storage()
                storage.delete(doc_info["storage_path"])
                logger.info(f"存储文件已删除: {doc_info['storage_path']}")
            except Exception as e:
                logger.warning(f"删除存储文件失败（不影响数据库删除）: {e}")

        # 删除数据库中的 chunks 和 document 记录
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                # 删除通过 document_id 关联的 chunks
                cur.execute("DELETE FROM chunks WHERE document_id = %s", (doc_id,))
                deleted_by_id = cur.rowcount
                # 删除旧数据中未关联 document_id 但文件名匹配的 chunks（兼容历史数据）
                cur.execute(
                    "DELETE FROM chunks WHERE document_id IS NULL AND metadata->>'filename' = %s",
                    (doc_info["filename"],),
                )
                deleted_by_name = cur.rowcount
                deleted_chunks = deleted_by_id + deleted_by_name
                cur.execute("DELETE FROM documents WHERE id = %s", (doc_id,))

        self.log_audit(
            action="delete",
            filename=doc_info["filename"],
            status="success",
            doc_id=doc_id,
            old_content_hash=doc_info.get("content_hash"),
            file_size=doc_info.get("file_size"),
        )
        logger.info(
            f"文档已删除: {doc_info['filename']} (id={doc_id}, "
            f"chunks={deleted_chunks})"
        )
        doc_info["deleted_chunks"] = deleted_chunks
        return doc_info

    # ================================================================
    # 相似度检索
    # ================================================================

    def similarity_search(
        self,
        query: str,
        k: int | None = None,
        filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        语义相似度检索（余弦相似度）。

        Args:
            query:  查询文本
            k:      返回结果数量
            filter: 元数据过滤条件（JSONB 字段过滤）

        Returns:
            list[dict]: 每项包含 content, metadata, score, distance
        """
        k = k or settings.RETRIEVAL_TOP_K

        # 计算查询向量
        try:
            query_embedding = self.embedder.embed_query(query)
        except Exception as e:
            raise PGVectorStoreError(f"查询嵌入计算失败: {e}") from e

        embedding_str = "[" + ",".join(str(v) for v in query_embedding) + "]"

        coll_id = self._ensure_collection(self.collection_name)

        with self._get_conn() as conn:
            with conn.cursor() as cur:
                # 用 cosine 距离检索
                cur.execute(
                    """
                    SELECT
                        content,
                        metadata,
                        1 - (embedding <=> %s::vector) AS cosine_similarity
                    FROM chunks
                    WHERE collection_id = %s
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (embedding_str, coll_id, embedding_str, k),
                )

                rows = cur.fetchall()

        results = []
        for row in rows:
            cosine_sim = float(row[2])
            score = max(0, cosine_sim)
            results.append({
                "content": row[0],
                "metadata": row[1] if isinstance(row[1], dict) else json.loads(row[1]),
                "score": round(score, 4),
                "distance": round(1.0 - score, 4),
            })

        logger.debug(
            f"向量检索完成: query='{query[:50]}...', "
            f"k={k}, 结果数={len(results)}"
        )
        return results

    # ================================================================
    # 集合与文档管理
    # ================================================================

    def count(self) -> int:
        """返回当前集合中的文档块总数"""
        coll_id = self._get_collection_id(self.collection_name)
        if coll_id is None:
            return 0
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT COUNT(*) FROM chunks WHERE collection_id = %s",
                    (coll_id,),
                )
                return cur.fetchone()[0]

    def delete_collection(self):
        """删除当前集合及其所有文档块（含 OSS 原文件）"""
        coll_id = self._get_collection_id(self.collection_name)
        if coll_id is None:
            logger.info(f"集合 '{self.collection_name}' 不存在，无需删除")
            return

        # 先获取所有文档的存储路径，用于删除 OSS 文件
        oss_paths = []
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT storage_path, storage_backend FROM documents WHERE collection_id = %s",
                    (coll_id,),
                )
                oss_paths = [(r[0], r[1]) for r in cur.fetchall() if r[0]]

        if oss_paths:
            from src.storage import get_storage
            storage = get_storage()
            for path, backend in oss_paths:
                try:
                    storage.delete(path)
                except Exception as e:
                    logger.warning(f"删除存储文件失败: {path} - {e}")

        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM chunks WHERE collection_id = %s", (coll_id,))
                deleted_chunks = cur.rowcount
                cur.execute("DELETE FROM documents WHERE collection_id = %s", (coll_id,))
                cur.execute(
                    "DELETE FROM collections WHERE name = %s",
                    (self.collection_name,),
                )
        logger.info(
            f"集合 '{self.collection_name}' 已删除"
            f"（清理 {deleted_chunks} 个文档块，{len(oss_paths)} 个存储文件）"
        )

    def switch_collection(self, collection_name: str):
        """切换到指定集合"""
        self.collection_name = collection_name
        self._ensure_collection(self.collection_name)
        logger.info(f"已切换到集合 '{collection_name}'")

    def rename_collection(self, old_name: str, new_name: str) -> bool:
        """重命名集合"""
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE collections SET name = %s WHERE name = %s",
                    (new_name, old_name),
                )
                renamed = cur.rowcount > 0
        if renamed:
            self.collection_name = new_name
            logger.info(f"集合已重命名: '{old_name}' → '{new_name}'")
        return renamed

    def list_collections(self) -> list[str]:
        """列出所有集合名称"""
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT name FROM collections ORDER BY name")
                return [row[0] for row in cur.fetchall()]

    def get_all_chunks(
        self,
        limit: int = 200,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """获取当前集合的所有文档块"""
        coll_id = self._get_collection_id(self.collection_name)
        if coll_id is None:
            return None
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, content, metadata
                    FROM chunks
                    WHERE collection_id = %s
                    ORDER BY created_at DESC
                    LIMIT %s OFFSET %s
                    """,
                    (coll_id, limit, offset),
                )
                rows = cur.fetchall()

        return [
            {
                "id": str(row[0]),
                "content": row[1],
                "metadata": row[2] if isinstance(row[2], dict) else json.loads(row[2]),
            }
            for row in rows
        ]

    # ================================================================
    # 对话管理（直接操作同一数据库）
    # ================================================================

    def list_conversations(self) -> list[dict[str, Any]]:
        """获取所有对话列表"""
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT c.id, c.title, c.created_at, c.updated_at,
                           (SELECT COUNT(*) FROM messages m WHERE m.conversation_id = c.id) AS msg_count
                    FROM conversations c
                    ORDER BY c.updated_at DESC
                """)
                rows = cur.fetchall()

        return [
            {
                "id": row[0],
                "title": row[1],
                "created_at": row[2].isoformat() if row[2] else None,
                "updated_at": row[3].isoformat() if row[3] else None,
                "message_count": row[4],
            }
            for row in rows
        ]

    def create_conversation(self, title: str | None = None) -> dict[str, Any]:
        """创建新对话"""
        now = datetime.now()
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO conversations (title, created_at, updated_at) VALUES (%s, %s, %s) RETURNING id",
                    (title or "新对话", now, now),
                )
                conv_id = cur.fetchone()[0]

        return {
            "id": conv_id,
            "title": title or "新对话",
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
            "message_count": 0,
        }

    def delete_conversation(self, conv_id: int) -> bool:
        """删除对话"""
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM conversations WHERE id = %s", (conv_id,))
                return cur.rowcount > 0

    def update_conversation_title(self, conv_id: int, title: str) -> bool:
        """修改对话标题"""
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE conversations SET title = %s, updated_at = NOW() WHERE id = %s",
                    (title, conv_id),
                )
                return cur.rowcount > 0

    def get_conversation_messages(self, conv_id: int) -> list[dict[str, Any]]:
        """获取对话的消息列表"""
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, role, content, sources, answer_type, created_at
                    FROM messages
                    WHERE conversation_id = %s
                    ORDER BY created_at ASC, id ASC
                    """,
                    (conv_id,),
                )
                rows = cur.fetchall()

        return [
            {
                "id": row[0],
                "role": row[1],
                "content": row[2],
                "sources": row[3] if isinstance(row[3], list) else json.loads(row[3]) if row[3] else [],
                "answer_type": row[4],
                "created_at": row[5].isoformat() if row[5] else None,
            }
            for row in rows
        ]

    def add_message(
        self, conv_id: int, role: str, content: str,
        sources: list | None = None, answer_type: str | None = None,
    ) -> int:
        """添加消息到对话，如果是第一条用户消息则自动生成对话标题"""
        sources_json = json.dumps(sources or [], ensure_ascii=False)
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                # 检查是否是第一条消息
                cur.execute(
                    "SELECT COUNT(*) FROM messages WHERE conversation_id = %s",
                    (conv_id,),
                )
                msg_count = cur.fetchone()[0]

                # 插入消息
                cur.execute(
                    """
                    INSERT INTO messages (conversation_id, role, content, sources, answer_type)
                    VALUES (%s, %s, %s, %s::jsonb, %s)
                    RETURNING id
                    """,
                    (conv_id, role, content, sources_json, answer_type),
                )
                msg_id = cur.fetchone()[0]

                # 如果是第一条用户消息，自动生成标题
                if msg_count == 0 and role == "user":
                    title = content.strip()[:25]
                    # 去掉首尾的特殊字符
                    title = title.strip("，。！？,.!?\n\r\t ")
                    if len(content) > 25:
                        title += "..."
                    if not title:
                        title = "新对话"
                    cur.execute(
                        "UPDATE conversations SET title = %s, updated_at = NOW() WHERE id = %s",
                        (title, conv_id),
                    )
                else:
                    cur.execute(
                        "UPDATE conversations SET updated_at = NOW() WHERE id = %s",
                        (conv_id,),
                    )
        return msg_id

    def update_message_content(
        self, msg_id: int, content: str, sources: list | None = None,
    ) -> bool:
        """更新消息内容"""
        sources_json = json.dumps(sources or [], ensure_ascii=False)
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE messages SET content = %s, sources = %s::jsonb WHERE id = %s",
                    (content, sources_json, msg_id),
                )
                return cur.rowcount > 0
