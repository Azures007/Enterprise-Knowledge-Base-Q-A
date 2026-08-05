"""
=============================================================================
PostgreSQL + pgvector 向量存储模块

替代 ChromaDB，使用 PostgreSQL 统一存储所有数据：
    - collections    → 知识库集合
    - documents      → 原始文件元数据
    - chunks         → 分块文本 + 向量 + 元数据
    - conversations  → 对话
    - messages       → 消息

架构：
    - 异步核心：使用 asyncpg 连接池，供 FastAPI 异步事件循环内调用（不阻塞其他请求）
    - 同步 shim：CLI 工具（ingest.py / query.py）通过 asyncio.run 桥接异步核心，
      只需维护一套 SQL，保证接口签名与同步调用兼容

依赖: asyncpg, pgvector（PostgreSQL 扩展）
=============================================================================

使用方法:
    # 异步（API 服务）
    store = PGVectorStore(embedder)
    await store.ainit_db()
    await store.aadd_documents(docs)
    results = await store.asimilarity_search("公司考勤", k=5)
    await store.aclose()

    # 同步（CLI 工具）
    store = PGVectorStore(embedder)
    store.add_documents(docs)
    results = store.similarity_search("公司考勤", k=5)
"""

import asyncio
import json
import sys
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import asyncpg

from config.settings import settings
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# Windows 兼容 + CLI 同步桥接
# ---------------------------------------------------------------------------
# asyncpg 不支持 Windows 默认的 ProactorEventLoop，此处统一切换为 Selector
# 策略。CLI 同步 shim 使用一个常驻的后台线程事件循环 + 单例连接池，避免
# 每次调用重复创建/销毁池在 Windows 上的竞态问题。
if sys.platform == "win32" and not isinstance(
    asyncio.get_event_loop_policy(),
    asyncio.WindowsSelectorEventLoopPolicy,
):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

_CLI_LOOP: asyncio.AbstractEventLoop | None = None
_CLI_LOOP_THREAD: threading.Thread | None = None
_CLI_LOCK = threading.Lock()


def _get_cli_loop() -> asyncio.AbstractEventLoop:
    """获取（或创建）常驻的 CLI 后台事件循环（Selector）"""
    global _CLI_LOOP, _CLI_LOOP_THREAD
    with _CLI_LOCK:
        if _CLI_LOOP is None or _CLI_LOOP.is_closed():
            _CLI_LOOP = asyncio.new_event_loop()
            _CLI_LOOP_THREAD = threading.Thread(
                target=_CLI_LOOP.run_forever,
                daemon=True,
                name="pg-cli-loop",
            )
            _CLI_LOOP_THREAD.start()
        return _CLI_LOOP


class PGVectorStoreError(Exception):
    """PostgreSQL 向量存储异常"""
    pass


def _parse_rowcount(status: str) -> int:
    """解析 asyncpg execute() 返回的状态字符串，如 'DELETE 5' → 5"""
    try:
        return int(status.split()[-1])
    except (ValueError, IndexError):
        return 0


def _json_or_dict(value) -> Any:
    """asyncpg 的 jsonb 默认返回 str，统一转为 dict/对象"""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (ValueError, TypeError):
            return value
    return value


class PGVectorStore:
    """
    基于 PostgreSQL + pgvector 的向量存储管理器。

    同时提供异步（a* 前缀）与同步方法，接口与 ChromaDB 的 VectorStoreManager 兼容。
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

        self._apool: asyncpg.Pool | None = None
        self._cli_pool: asyncpg.Pool | None = None
        self._dimension = settings.VECTOR_DIMENSION

        # 初始化数据库表（同步，兼容 CLI 场景）
        self.init_db()

        logger.info(
            f"PGVectorStore 初始化完成: collection={collection_name}"
        )

    # ================================================================
    # 连接管理
    # ================================================================

    async def _aconn(self) -> asyncpg.Pool:
        """懒创建 asyncpg 连接池并返回"""
        if self._apool is None:
            self._apool = await asyncpg.create_pool(
                dsn=self.conn_string,
                min_size=1,
                max_size=10,
                command_timeout=60,
            )
            logger.info("asyncpg 连接池已创建 (min=1, max=10)")
        return self._apool

    async def aclose(self):
        """关闭连接池（FastAPI lifespan 退出时调用）"""
        if self._apool is not None:
            await self._apool.close()
            self._apool = None
            logger.info("asyncpg 连接池已关闭")

    def _sync(self, coro_factory):
        """
        同步 shim：在常驻后台事件循环中运行异步方法。

        通过 run_coroutine_threadsafe 提交到 _get_cli_loop() 的常驻事件循环。
        实例级连接池 self._cli_pool 首次调用时创建，后续复用，
        避免每次调用重复创建/销毁连接池在 Windows 上的竞态。
        仅适用于 CLI 等无运行中事件循环的场景。
        """
        if self._apool is not None:
            raise PGVectorStoreError(
                "连接池已由异步上下文创建，请使用异步方法（a* 前缀）"
            )

        async def runner():
            if self._cli_pool is None:
                self._cli_pool = await asyncpg.create_pool(
                    dsn=self.conn_string,
                    min_size=1,
                    max_size=5,
                    command_timeout=60,
                )
            self._apool = self._cli_pool
            try:
                return await coro_factory()
            finally:
                self._apool = None

        loop = _get_cli_loop()
        future = asyncio.run_coroutine_threadsafe(runner(), loop)
        return future.result()

    # ================================================================
    # 数据库初始化
    # ================================================================

    async def _ainit_db(self):
        """创建所有必要的表和扩展（异步）"""
        async with (await self._aconn()).acquire() as conn:
            # 启用 pgvector 扩展
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

            # 集合表
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS collections (
                    id          SERIAL PRIMARY KEY,
                    name        VARCHAR(255) UNIQUE NOT NULL,
                    description TEXT,
                    owner_id    INTEGER REFERENCES users(id) ON DELETE CASCADE,
                    created_at  TIMESTAMP DEFAULT NOW()
                )
            """)
            # 兼容旧表：无 owner_id 列时补充（历史集合 owner_id 为 NULL，视为系统共享）
            try:
                await conn.execute(
                    "ALTER TABLE collections ADD COLUMN IF NOT EXISTS owner_id INTEGER REFERENCES users(id) ON DELETE CASCADE"
                )
            except Exception:
                pass
            try:
                await conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_collections_owner ON collections(owner_id)"
                )
            except Exception:
                pass

            # 文档记录表（原始文件元数据）
            await conn.execute("""
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
            await conn.execute(f"""
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
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id          SERIAL PRIMARY KEY,
                    title       VARCHAR(255) DEFAULT '新对话',
                    user_id     INTEGER REFERENCES users(id) ON DELETE CASCADE,
                    created_at  TIMESTAMP DEFAULT NOW(),
                    updated_at  TIMESTAMP DEFAULT NOW()
                )
            """)
            # 兼容旧表：若无 user_id 列则补充（旧数据 user_id 为 NULL，视为共享）
            try:
                await conn.execute(
                    "ALTER TABLE conversations ADD COLUMN IF NOT EXISTS user_id INTEGER REFERENCES users(id) ON DELETE CASCADE"
                )
            except Exception:
                pass
            try:
                await conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_conversations_user ON conversations(user_id)"
                )
            except Exception:
                pass

            # 消息表
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id              SERIAL PRIMARY KEY,
                    conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                    role            VARCHAR(10) CHECK (role IN ('user','ai','system')),
                    content         TEXT NOT NULL DEFAULT '',
                    sources         JSONB DEFAULT '[]'::jsonb,
                    answer_type     VARCHAR(20),
                    is_stale        BOOLEAN DEFAULT false,
                    created_at      TIMESTAMP DEFAULT NOW()
                )
            """)
            # 兼容旧表：无 is_stale 列时补充
            try:
                await conn.execute(
                    "ALTER TABLE messages ADD COLUMN IF NOT EXISTS is_stale BOOLEAN DEFAULT false"
                )
            except Exception:
                pass

            # 向量索引（HNSW）
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw
                ON chunks
                USING hnsw (embedding vector_cosine_ops)
                WITH (m = 16, ef_construction = 200)
            """)

            # 普通索引
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_chunks_collection
                ON chunks(collection_id)
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_conversation
                ON messages(conversation_id, created_at)
            """)

            # 导入审计日志表
            await conn.execute("""
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
            # 兼容扩展：异步导入任务字段（旧表无此列时自动补上）
            for col_sql in (
                'ALTER TABLE ingest_audit_log ADD COLUMN IF NOT EXISTS task_id VARCHAR(64)',
                'ALTER TABLE ingest_audit_log ADD COLUMN IF NOT EXISTS progress INTEGER DEFAULT 0',
                'ALTER TABLE ingest_audit_log ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP DEFAULT NOW()',
            ):
                try:
                    await conn.execute(col_sql)
                except Exception:
                    pass
            try:
                await conn.execute(
                    'CREATE INDEX IF NOT EXISTS idx_ingest_task_id ON ingest_audit_log(task_id)'
                )
            except Exception:
                pass

            # 用户表（登录认证）
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id              SERIAL PRIMARY KEY,
                    username        VARCHAR(255) UNIQUE NOT NULL,
                    password_hash   VARCHAR(255) NOT NULL,
                    display_name    VARCHAR(255),
                    is_admin        BOOLEAN DEFAULT false,
                    created_at      TIMESTAMP DEFAULT NOW(),
                    last_login_at   TIMESTAMP
                )
            """)

        logger.info("PostgreSQL 数据库表初始化完成")

    def init_db(self):
        """创建所有必要的表和扩展（同步，CLI 使用）"""
        return self._sync(self._ainit_db)

    # ================================================================
    # 集合管理（异步核心）
    # ================================================================

    async def _aensure_collection(self, name: str, conn=None, owner_id: int | None = None) -> int:
        """确保集合存在，不存在则创建，返回集合 ID。新建集合时可指定归属用户。"""
        if conn is not None:
            row = await conn.fetchrow(
                "SELECT id FROM collections WHERE name = $1", name
            )
            if row:
                return row["id"]
            if owner_id is not None:
                coll_id = await conn.fetchval(
                    "INSERT INTO collections (name, owner_id) VALUES ($1, $2) RETURNING id",
                    name, owner_id,
                )
            else:
                coll_id = await conn.fetchval(
                    "INSERT INTO collections (name) VALUES ($1) RETURNING id",
                    name,
                )
            logger.info(f"创建集合 '{name}' (id={coll_id}, owner={owner_id})")
            return coll_id
        async with (await self._aconn()).acquire() as conn:
            row = await conn.fetchrow(
                "SELECT id FROM collections WHERE name = $1", name
            )
            if row:
                return row["id"]
            if owner_id is not None:
                coll_id = await conn.fetchval(
                    "INSERT INTO collections (name, owner_id) VALUES ($1, $2) RETURNING id",
                    name, owner_id,
                )
            else:
                coll_id = await conn.fetchval(
                    "INSERT INTO collections (name) VALUES ($1) RETURNING id",
                    name,
                )
            logger.info(f"创建集合 '{name}' (id={coll_id}, owner={owner_id})")
            return coll_id

    async def _aget_collection_id(self, name: str) -> int | None:
        """获取集合 ID，不存在则返回 None（不会自动创建）"""
        async with (await self._aconn()).acquire() as conn:
            row = await conn.fetchrow(
                "SELECT id FROM collections WHERE name = $1", name
            )
            return row["id"] if row else None

    async def alist_collections(self) -> list[str]:
        """列出所有集合名称"""
        async with (await self._aconn()).acquire() as conn:
            rows = await conn.fetch("SELECT name FROM collections ORDER BY name")
            return [r["name"] for r in rows]

    async def alist_collections_with_owner(self) -> list[dict]:
        """列出所有集合（含归属用户 ID 与名称）。"""
        async with (await self._aconn()).acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT c.id, c.name, c.owner_id, u.username AS owner_username
                FROM collections c
                LEFT JOIN users u ON c.owner_id = u.id
                ORDER BY c.name
                """
            )
        return [
            {
                "id": r["id"],
                "name": r["name"],
                "owner_id": r["owner_id"],
                "owner_username": r["owner_username"],
            }
            for r in rows
        ]

    async def aget_collection_owner(self, name: str) -> int | None:
        """查询集合的归属用户 ID（历史/系统集合返回 None）。"""
        async with (await self._aconn()).acquire() as conn:
            return await conn.fetchval(
                "SELECT owner_id FROM collections WHERE name = $1", name
            )

    async def aset_collection_owner(self, name: str, owner_id: int | None) -> bool:
        """设置集合的归属用户（owner_id 为 None 表示系统/共享集合）。"""
        async with (await self._aconn()).acquire() as conn:
            return _parse_rowcount(
                await conn.execute(
                    "UPDATE collections SET owner_id = $1 WHERE name = $2",
                    owner_id, name,
                )
            ) > 0

    async def aswitch_collection(self, collection_name: str):
        """切换到指定集合"""
        self.collection_name = collection_name
        await self._aensure_collection(self.collection_name)
        logger.info(f"已切换到集合 '{collection_name}'")

    async def arenmae_collection(self, old_name: str, new_name: str) -> bool:
        """重命名集合"""
        async with (await self._aconn()).acquire() as conn:
            result = await conn.execute(
                "UPDATE collections SET name = $1 WHERE name = $2",
                new_name, old_name,
            )
            renamed = _parse_rowcount(result) > 0
        if renamed:
            if self.collection_name == old_name:
                self.collection_name = new_name
            logger.info(f"集合已重命名: '{old_name}' → '{new_name}'")
        return renamed

    async def acount(self) -> int:
        """返回当前集合中的文档块总数"""
        coll_id = await self._aget_collection_id(self.collection_name)
        if coll_id is None:
            return 0
        async with (await self._aconn()).acquire() as conn:
            return await conn.fetchval(
                "SELECT COUNT(*) FROM chunks WHERE collection_id = $1", coll_id
            )

    async def adelete_collection(self):
        """删除当前集合及其所有文档块（含 OSS 原文件）"""
        coll_id = await self._aget_collection_id(self.collection_name)
        if coll_id is None:
            logger.info(f"集合 '{self.collection_name}' 不存在，无需删除")
            return

        # 先获取所有文档的存储路径，用于删除 OSS 文件
        oss_paths = []
        async with (await self._aconn()).acquire() as conn:
            rows = await conn.fetch(
                "SELECT storage_path FROM documents WHERE collection_id = $1",
                coll_id,
            )
            oss_paths = [r["storage_path"] for r in rows if r["storage_path"]]

        if oss_paths:
            from src.storage import get_storage
            storage = get_storage()
            for path in oss_paths:
                try:
                    storage.delete(path)
                except Exception as e:
                    logger.warning(f"删除存储文件失败: {path} - {e}")

        async with (await self._aconn()).acquire() as conn:
            await conn.execute("DELETE FROM chunks WHERE collection_id = $1", coll_id)
            await conn.execute("DELETE FROM documents WHERE collection_id = $1", coll_id)
            await conn.execute(
                "DELETE FROM collections WHERE name = $1", self.collection_name
            )
        logger.info(f"集合 '{self.collection_name}' 已删除")

    # ================================================================
    # 文档管理（异步核心）
    # ================================================================

    async def _aembed_texts(self, texts: list[str]):
        """嵌入计算：优先使用嵌入器的异步方法"""
        if hasattr(self.embedder, "aembed_documents"):
            return await self.embedder.aembed_documents(texts)
        return self.embedder.embed_documents(texts)

    async def _aembed_query(self, query: str):
        if hasattr(self.embedder, "aembed_query"):
            return await self.embedder.aembed_query(query)
        return self.embedder.embed_query(query)

    async def aadd_documents(
        self,
        documents: list[dict[str, Any]],
        batch_size: int = 64,
        document_id: int | None = None,
    ) -> int:
        """
        向向量库中添加文档块（异步）。

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

        logger.info(f"正在计算 {len(texts)} 个文档块的嵌入向量...")
        try:
            embeddings = await self._aembed_texts(texts)
        except Exception as e:
            raise PGVectorStoreError(f"嵌入计算失败: {e}") from e

        coll_id = await self._aensure_collection(self.collection_name)

        added_count = 0
        async with (await self._aconn()).acquire() as conn:
            for i in range(0, len(texts), batch_size):
                batch_end = min(i + batch_size, len(texts))
                rows = []
                for j in range(i, batch_end):
                    meta_json = json.dumps(metadatas[j], ensure_ascii=False)
                    embedding_str = "[" + ",".join(str(v) for v in embeddings[j]) + "]"
                    rows.append((
                        coll_id,
                        document_id,
                        texts[j],
                        meta_json,
                        embedding_str,
                    ))
                await conn.executemany(
                    """
                    INSERT INTO chunks (collection_id, document_id, content, metadata, embedding)
                    VALUES ($1, $2, $3, $4::jsonb, $5::vector)
                    """,
                    rows,
                )
                added_count += len(rows)
                logger.debug(f"已添加 {added_count}/{len(texts)} 个文档块")

        await self._alog_audit(
            action="ingest",
            filename=metadatas[0].get("filename", "unknown") if metadatas else "unknown",
            status="success",
            doc_id=document_id,
            chunks_added=added_count,
        )
        logger.info(f"文档入库完成: 共 {added_count} 个文档块")
        return added_count

    async def aadd_document_record(
        self,
        filename: str,
        file_type: str,
        file_size: int,
        content_hash: str = "",
        storage_path: str = "",
        storage_backend: str = "local",
    ) -> int:
        """记录原始文件元数据到 documents 表，返回文档 ID"""
        coll_id = await self._aensure_collection(self.collection_name)
        async with (await self._aconn()).acquire() as conn:
            doc_id = await conn.fetchval(
                """
                INSERT INTO documents (collection_id, filename, file_type, file_size, content_hash, storage_backend, storage_path)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                RETURNING id
                """,
                coll_id, filename, file_type, file_size, content_hash, storage_backend, storage_path,
            )
        logger.info(f"文档记录已创建: {filename} (id={doc_id})")
        return doc_id

    async def _alog_audit(
        self,
        action: str,
        filename: str,
        status: str = "success",
        error_msg: str | None = None,
        **kwargs: Any,
    ):
        """记录导入/删除操作的审计日志"""
        try:
            async with (await self._aconn()).acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO ingest_audit_log
                    (collection, filename, action, doc_id, old_doc_id,
                     old_content_hash, chunks_added, file_size, status, error_msg)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                    """,
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
                )
            logger.debug(f"审计日志已记录: {action} {filename} [{status}]")
        except Exception as e:
            logger.warning(f"审计日志记录失败: {e}")

    async def afind_document_by_hash(self, content_hash: str) -> dict | None:
        """根据内容哈希查找是否已存在相同文档。"""
        coll_id = await self._aget_collection_id(self.collection_name)
        if coll_id is None:
            return None
        async with (await self._aconn()).acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT id, filename, file_size, created_at
                FROM documents
                WHERE collection_id = $1 AND content_hash = $2
                ORDER BY created_at DESC
                LIMIT 1
                """,
                coll_id, content_hash,
            )
        if row is None:
            return None
        return {
            "id": row["id"],
            "filename": row["filename"],
            "file_size": row["file_size"],
            "created_at": row["created_at"].isoformat() if row["created_at"] else None,
        }

    async def afind_document_by_filename(self, filename: str) -> dict | None:
        """根据文件名查找是否已存在同名文档（同一集合内）。"""
        coll_id = await self._aget_collection_id(self.collection_name)
        if coll_id is None:
            return None
        async with (await self._aconn()).acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT id, filename, file_size, created_at
                FROM documents
                WHERE collection_id = $1 AND filename = $2
                ORDER BY created_at DESC
                LIMIT 1
                """,
                coll_id, filename,
            )
        if row is None:
            return None
        return {
            "id": row["id"],
            "filename": row["filename"],
            "file_size": row["file_size"],
            "created_at": row["created_at"].isoformat() if row["created_at"] else None,
        }

    async def afind_document_by_id(self, doc_id: int) -> dict | None:
        """按 ID 查询文档，返回含所属集合名的信息。"""
        async with (await self._aconn()).acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT d.id, d.filename, d.collection_id, c.name AS collection_name
                FROM documents d
                LEFT JOIN collections c ON d.collection_id = c.id
                WHERE d.id = $1
                """,
                doc_id,
            )
        if row is None:
            return None
        return {
            "id": row["id"],
            "filename": row["filename"],
            "collection_id": row["collection_id"],
            "collection_name": row["collection_name"],
        }

    async def acount_document_chunks(self, doc_id: int) -> int:
        """
        统计指定文档关联的 chunk 数量（含旧数据中按文件名关联的 chunk）。

        用于判断文档是否真的完成了分块入库，避免"僵尸记录"（只有元数据、
        无分块内容）被误判为正常重复。
        """
        async with (await self._aconn()).acquire() as conn:
            # 先查文档文件名
            row = await conn.fetchrow(
                "SELECT filename FROM documents WHERE id = $1", doc_id
            )
            if row is None:
                return 0
            filename = row["filename"]
            return await conn.fetchval(
                """
                SELECT COUNT(*) FROM chunks
                WHERE document_id = $1
                   OR (document_id IS NULL AND metadata->>'filename' = $2)
                """,
                doc_id, filename,
            )

    async def alist_documents(self) -> list[dict]:
        """获取当前集合中的所有文档列表（含 chunk 数量）。"""
        coll_id = await self._aget_collection_id(self.collection_name)
        if coll_id is None:
            return []
        async with (await self._aconn()).acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT d.id, d.filename, d.file_type, d.file_size,
                       d.storage_backend, d.storage_path, d.created_at,
                       (
                         SELECT COUNT(*) FROM chunks c
                         WHERE c.document_id = d.id
                            OR (c.document_id IS NULL AND c.metadata->>'filename' = d.filename)
                       ) AS chunk_count
                FROM documents d
                WHERE d.collection_id = $1
                ORDER BY d.created_at DESC
                """,
                coll_id,
            )
        return [
            {
                "id": r["id"],
                "filename": r["filename"],
                "file_type": r["file_type"],
                "file_size": r["file_size"],
                "storage_backend": r["storage_backend"],
                "storage_path": r["storage_path"],
                "created_at": r["created_at"].isoformat() if r["created_at"] else None,
                "chunk_count": r["chunk_count"],
            }
            for r in rows
        ]

    async def adelete_document(self, doc_id: int, delete_storage: bool = True) -> dict | None:
        """删除文档及其所有分块，可选择同时删除存储文件。"""
        async with (await self._aconn()).acquire() as conn:
            row = await conn.fetchrow(
                "SELECT filename, file_type, file_size, storage_backend, storage_path, content_hash FROM documents WHERE id = $1",
                doc_id,
            )
            if row is None:
                return None

            doc_info = {
                "id": doc_id,
                "filename": row["filename"],
                "file_type": row["file_type"],
                "file_size": row["file_size"],
                "storage_backend": row["storage_backend"],
                "storage_path": row["storage_path"],
                "content_hash": row["content_hash"],
            }

            # 删除通过 document_id 关联的 chunks
            deleted_by_id = _parse_rowcount(
                await conn.execute("DELETE FROM chunks WHERE document_id = $1", doc_id)
            )
            # 兼容历史数据（未关联 document_id 但文件名匹配）
            deleted_by_name = _parse_rowcount(
                await conn.execute(
                    "DELETE FROM chunks WHERE document_id IS NULL AND metadata->>'filename' = $1",
                    doc_info["filename"],
                )
            )
            deleted_chunks = deleted_by_id + deleted_by_name
            await conn.execute("DELETE FROM documents WHERE id = $1", doc_id)

        # 删除存储后端中的原始文件
        if delete_storage and doc_info["storage_path"]:
            try:
                from src.storage import get_storage
                storage = get_storage()
                storage.delete(doc_info["storage_path"])
                logger.info(f"存储文件已删除: {doc_info['storage_path']}")
            except Exception as e:
                logger.warning(f"删除存储文件失败（不影响数据库删除）: {e}")

        await self._alog_audit(
            action="delete",
            filename=doc_info["filename"],
            status="success",
            doc_id=doc_id,
            old_content_hash=doc_info.get("content_hash"),
            file_size=doc_info.get("file_size"),
        )
        logger.info(
            f"文档已删除: {doc_info['filename']} (id={doc_id}, chunks={deleted_chunks})"
        )
        doc_info["deleted_chunks"] = deleted_chunks
        return doc_info

    # ================================================================
    # 相似度检索（异步核心）
    # ================================================================

    async def asimilarity_search(
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

        try:
            query_embedding = await self._aembed_query(query)
        except Exception as e:
            raise PGVectorStoreError(f"查询嵌入计算失败: {e}") from e

        embedding_str = "[" + ",".join(str(v) for v in query_embedding) + "]"
        coll_id = await self._aensure_collection(self.collection_name)

        # 元数据过滤：追加 WHERE 条件
        # 参数顺序: $1=查询向量, $2=集合ID, $3=limit, 其后为 metadata 过滤键值对
        filter_sql = ""
        params: list = [embedding_str, coll_id, k]
        if filter:
            # 支持形如 {"filename": "xxx"} 的简单等值过滤
            conds = []
            for key, val in filter.items():
                params.append(key)
                params.append(str(val))
                n = len(params)
                conds.append(f"metadata->>${n - 1} = ${n}")
            filter_sql = " AND " + " AND ".join(conds)

        async with (await self._aconn()).acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT
                    content,
                    metadata,
                    1 - (embedding <=> $1::vector) AS cosine_similarity
                FROM chunks
                WHERE collection_id = $2{filter_sql}
                ORDER BY embedding <=> $1::vector
                LIMIT $3
                """,
                *params,
            )

        results = []
        for row in rows:
            cosine_sim = float(row["cosine_similarity"])
            score = max(0, cosine_sim)
            results.append({
                "content": row["content"],
                "metadata": _json_or_dict(row["metadata"]),
                "score": round(score, 4),
                "distance": round(1.0 - score, 4),
            })

        logger.debug(
            f"向量检索完成: query='{query[:50]}...', "
            f"k={k}, 结果数={len(results)}"
        )
        return results

    async def aget_all_chunks(
        self,
        limit: int = 200,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """获取当前集合的所有文档块。"""
        coll_id = await self._aget_collection_id(self.collection_name)
        if coll_id is None:
            return []
        async with (await self._aconn()).acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, content, metadata
                FROM chunks
                WHERE collection_id = $1
                ORDER BY created_at DESC
                LIMIT $2 OFFSET $3
                """,
                coll_id, limit, offset,
            )
        return [
            {
                "id": str(r["id"]),
                "content": r["content"],
                "metadata": _json_or_dict(r["metadata"]),
            }
            for r in rows
        ]

    # ================================================================
    # 对话管理（异步核心，直接操作同一数据库）
    # ================================================================

    async def alist_conversations(self, user_id: int | None = None) -> list[dict[str, Any]]:
        """
        获取对话列表。

        Args:
            user_id: 若提供，仅返回该用户的对话；None 返回全部（兼容）
        """
        async with (await self._aconn()).acquire() as conn:
            if user_id is not None:
                rows = await conn.fetch(
                    """
                    SELECT c.id, c.title, c.created_at, c.updated_at,
                           (SELECT COUNT(*) FROM messages m WHERE m.conversation_id = c.id) AS msg_count
                    FROM conversations c
                    WHERE c.user_id = $1
                    ORDER BY c.updated_at DESC
                    """,
                    user_id,
                )
            else:
                rows = await conn.fetch("""
                    SELECT c.id, c.title, c.created_at, c.updated_at,
                           (SELECT COUNT(*) FROM messages m WHERE m.conversation_id = c.id) AS msg_count
                    FROM conversations c
                    ORDER BY c.updated_at DESC
                """)
        return [
            {
                "id": r["id"],
                "title": r["title"],
                "created_at": r["created_at"].isoformat() if r["created_at"] else None,
                "updated_at": r["updated_at"].isoformat() if r["updated_at"] else None,
                "message_count": r["msg_count"],
            }
            for r in rows
        ]

    async def acreate_conversation(self, title: str | None = None, user_id: int | None = None) -> dict[str, Any]:
        """创建新对话（可指定归属用户）。"""
        now = datetime.now()
        async with (await self._aconn()).acquire() as conn:
            if user_id is not None:
                conv_id = await conn.fetchval(
                    "INSERT INTO conversations (title, user_id, created_at, updated_at) VALUES ($1, $2, $3, $4) RETURNING id",
                    title or "新对话", user_id, now, now,
                )
            else:
                conv_id = await conn.fetchval(
                    "INSERT INTO conversations (title, created_at, updated_at) VALUES ($1, $2, $3) RETURNING id",
                    title or "新对话", now, now,
                )
        return {
            "id": conv_id,
            "title": title or "新对话",
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
            "message_count": 0,
        }

    async def adelete_conversation(self, conv_id: int, user_id: int | None = None) -> bool:
        """删除对话（指定 user_id 时仅能删自己的对话）。"""
        async with (await self._aconn()).acquire() as conn:
            if user_id is not None:
                return _parse_rowcount(
                    await conn.execute(
                        "DELETE FROM conversations WHERE id = $1 AND user_id = $2",
                        conv_id, user_id,
                    )
                ) > 0
            return _parse_rowcount(
                await conn.execute("DELETE FROM conversations WHERE id = $1", conv_id)
            ) > 0

    async def aupdate_conversation_title(self, conv_id: int, title: str, user_id: int | None = None) -> bool:
        """修改对话标题（指定 user_id 时仅能改自己的对话）。"""
        async with (await self._aconn()).acquire() as conn:
            if user_id is not None:
                return _parse_rowcount(
                    await conn.execute(
                        "UPDATE conversations SET title = $1, updated_at = NOW() WHERE id = $2 AND user_id = $3",
                        title, conv_id, user_id,
                    )
                ) > 0
            return _parse_rowcount(
                await conn.execute(
                    "UPDATE conversations SET title = $1, updated_at = NOW() WHERE id = $2",
                    title, conv_id,
                )
            ) > 0

    async def aget_conversation_messages(self, conv_id: int, user_id: int | None = None) -> list[dict[str, Any]]:
        """获取对话的消息列表（指定 user_id 时仅能读自己的对话）。"""
        async with (await self._aconn()).acquire() as conn:
            # 校验对话归属（可选）
            if user_id is not None:
                owner = await conn.fetchval(
                    "SELECT user_id FROM conversations WHERE id = $1", conv_id
                )
                if owner is None or owner != user_id:
                    return []
            rows = await conn.fetch(
                """
                SELECT id, role, content, sources, answer_type, is_stale, created_at
                FROM messages
                WHERE conversation_id = $1
                ORDER BY created_at ASC, id ASC
                """,
                conv_id,
            )
        return [
            {
                "id": r["id"],
                "role": r["role"],
                "content": r["content"],
                "sources": _json_or_dict(r["sources"]) if r["sources"] else [],
                "answer_type": r["answer_type"],
                "is_stale": bool(r["is_stale"]),
                "created_at": r["created_at"].isoformat() if r["created_at"] else None,
            }
            for r in rows
        ]

    async def aadd_message(
        self, conv_id: int, role: str, content: str,
        sources: list | None = None, answer_type: str | None = None,
    ) -> int:
        """添加消息到对话，如果是第一条用户消息则自动生成对话标题。"""
        sources_json = json.dumps(sources or [], ensure_ascii=False)
        async with (await self._aconn()).acquire() as conn:
            msg_count = await conn.fetchval(
                "SELECT COUNT(*) FROM messages WHERE conversation_id = $1", conv_id
            )
            msg_id = await conn.fetchval(
                """
                INSERT INTO messages (conversation_id, role, content, sources, answer_type)
                VALUES ($1, $2, $3, $4::jsonb, $5)
                RETURNING id
                """,
                conv_id, role, content, sources_json, answer_type,
            )

            # 第一条用户消息自动生成标题
            if msg_count == 0 and role == "user":
                title = content.strip()[:25]
                title = title.strip("，。！？,.!?\n\r\t ")
                if len(content) > 25:
                    title += "..."
                if not title:
                    title = "新对话"
                await conn.execute(
                    "UPDATE conversations SET title = $1, updated_at = NOW() WHERE id = $2",
                    title, conv_id,
                )
            else:
                await conn.execute(
                    "UPDATE conversations SET updated_at = NOW() WHERE id = $1",
                    conv_id,
                )
        return msg_id

    async def aupdate_message_content(
        self, msg_id: int, content: str, sources: list | None = None,
    ) -> bool:
        """更新消息内容。"""
        sources_json = json.dumps(sources or [], ensure_ascii=False)
        async with (await self._aconn()).acquire() as conn:
            return _parse_rowcount(
                await conn.execute(
                    "UPDATE messages SET content = $1, sources = $2::jsonb WHERE id = $3",
                    content, sources_json, msg_id,
                )
            ) > 0

    async def amark_stale_by_filenames(self, filenames: list[str]) -> int:
        """
        将引用指定文件的 AI 回答标记为已过期（is_stale=true）。

        用于文档新增/删除后，历史对话中基于旧数据的回答失效提示。
        sources 是 JSONB 数组，形如 [{"filename": "xxx", ...}, ...]。
        """
        if not filenames:
            return 0
        marked = 0
        async with (await self._aconn()).acquire() as conn:
            for fname in set(filenames):
                result = await conn.execute(
                    """
                    UPDATE messages
                    SET is_stale = true
                    WHERE role = 'ai'
                      AND sources::text LIKE '%' || $1 || '%'
                      AND is_stale = false
                    """,
                    fname,
                )
                marked += _parse_rowcount(result)
        return marked

    # ================================================================
    # 异步导入任务管理（ingest_audit_log 扩展为任务表）
    # ================================================================

    async def acreate_ingest_task(
        self,
        task_id: str,
        filename: str,
        collection: str,
        file_size: int,
    ) -> int:
        """创建异步导入任务记录，返回任务行 id。"""
        async with (await self._aconn()).acquire() as conn:
            row_id = await conn.fetchval(
                """
                INSERT INTO ingest_audit_log
                (task_id, collection, filename, action, file_size, status, progress)
                VALUES ($1, $2, $3, 'ingest_async', $4, 'pending', 0)
                RETURNING id
                """,
                task_id, collection, filename, file_size,
            )
        return row_id

    async def aget_ingest_task(self, task_id: str) -> dict | None:
        """查询异步导入任务状态。"""
        async with (await self._aconn()).acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT id, task_id, collection, filename, file_size,
                       chunks_added, status, progress, error_msg, created_at, updated_at
                FROM ingest_audit_log
                WHERE task_id = $1
                ORDER BY id DESC
                LIMIT 1
                """,
                task_id,
            )
        if row is None:
            return None
        return {
            "id": row["id"],
            "task_id": row["task_id"],
            "collection": row["collection"],
            "filename": row["filename"],
            "file_size": row["file_size"],
            "chunks_added": row["chunks_added"],
            "status": row["status"],
            "progress": row["progress"],
            "error_msg": row["error_msg"],
            "created_at": row["created_at"].isoformat() if row["created_at"] else None,
            "updated_at": row["updated_at"].isoformat() if row["updated_at"] else None,
        }

    async def aupdate_ingest_task(
        self,
        task_id: str,
        status: str | None = None,
        progress: int | None = None,
        chunks_added: int | None = None,
        error_msg: str | None = None,
        doc_id: int | None = None,
    ) -> None:
        """更新异步导入任务状态。"""
        sets = ["updated_at = NOW()"]
        params: list = []
        if status is not None:
            sets.append(f"status = ${len(params) + 1}")
            params.append(status)
        if progress is not None:
            sets.append(f"progress = ${len(params) + 1}")
            params.append(progress)
        if chunks_added is not None:
            sets.append(f"chunks_added = ${len(params) + 1}")
            params.append(chunks_added)
        if error_msg is not None:
            sets.append(f"error_msg = ${len(params) + 1}")
            params.append(error_msg)
        if doc_id is not None:
            sets.append(f"doc_id = ${len(params) + 1}")
            params.append(doc_id)
        params.append(task_id)
        sql = f"UPDATE ingest_audit_log SET {', '.join(sets)} WHERE task_id = ${len(params)}"
        async with (await self._aconn()).acquire() as conn:
            await conn.execute(sql, *params)

    async def alist_active_ingest_tasks(self) -> list[dict]:
        """列出进行中的异步导入任务（pending / processing）。"""
        async with (await self._aconn()).acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT task_id, filename, collection, file_size, status, progress,
                       error_msg, created_at, updated_at
                FROM ingest_audit_log
                WHERE task_id IS NOT NULL AND status IN ('pending', 'processing')
                ORDER BY created_at DESC
                LIMIT 50
                """
            )
        return [
            {
                "task_id": r["task_id"],
                "filename": r["filename"],
                "collection": r["collection"],
                "file_size": r["file_size"],
                "status": r["status"],
                "progress": r["progress"],
                "error_msg": r["error_msg"],
                "created_at": r["created_at"].isoformat() if r["created_at"] else None,
                "updated_at": r["updated_at"].isoformat() if r["updated_at"] else None,
            }
            for r in rows
        ]

    # ================================================================
    # 用户管理（异步核心）
    # ================================================================

    async def acreate_user(
        self,
        username: str,
        password_hash: str,
        display_name: str | None = None,
        is_admin: bool = False,
    ) -> int:
        """创建用户，返回用户 ID（用户名已存在时抛唯一约束错误）。"""
        async with (await self._aconn()).acquire() as conn:
            try:
                user_id = await conn.fetchval(
                    """
                    INSERT INTO users (username, password_hash, display_name, is_admin)
                    VALUES ($1, $2, $3, $4)
                    RETURNING id
                    """,
                    username, password_hash, display_name, is_admin,
                )
            except asyncpg.exceptions.UniqueViolationError:
                raise PGVectorStoreError(f"用户名 '{username}' 已存在")
            return user_id

    async def aget_user_by_username(self, username: str) -> dict | None:
        """按用户名查询用户。"""
        async with (await self._aconn()).acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT id, username, password_hash, display_name, is_admin, created_at, last_login_at
                FROM users
                WHERE username = $1
                """,
                username,
            )
        if row is None:
            return None
        return {
            "id": row["id"],
            "username": row["username"],
            "password_hash": row["password_hash"],
            "display_name": row["display_name"],
            "is_admin": row["is_admin"],
            "created_at": row["created_at"].isoformat() if row["created_at"] else None,
            "last_login_at": row["last_login_at"].isoformat() if row["last_login_at"] else None,
        }

    async def aupdate_user_login(self, user_id: int):
        """更新用户最近登录时间。"""
        async with (await self._aconn()).acquire() as conn:
            await conn.execute(
                "UPDATE users SET last_login_at = NOW() WHERE id = $1", user_id
            )

    async def aupdate_user_password(self, user_id: int, password_hash: str) -> bool:
        """更新用户密码。"""
        async with (await self._aconn()).acquire() as conn:
            return _parse_rowcount(
                await conn.execute(
                    "UPDATE users SET password_hash = $1 WHERE id = $2",
                    password_hash, user_id,
                )
            ) > 0

    async def alist_users(self) -> list[dict]:
        """列出所有用户。"""
        async with (await self._aconn()).acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, username, display_name, is_admin, created_at, last_login_at
                FROM users
                ORDER BY id ASC
                """
            )
        return [
            {
                "id": r["id"],
                "username": r["username"],
                "display_name": r["display_name"],
                "is_admin": r["is_admin"],
                "created_at": r["created_at"].isoformat() if r["created_at"] else None,
                "last_login_at": r["last_login_at"].isoformat() if r["last_login_at"] else None,
            }
            for r in rows
        ]

    async def adelete_user(self, user_id: int) -> bool:
        """删除用户（级联删除其对话）。"""
        async with (await self._aconn()).acquire() as conn:
            return _parse_rowcount(
                await conn.execute("DELETE FROM users WHERE id = $1", user_id)
            ) > 0

    # ================================================================
    # 同步 shim（CLI 工具兼容，通过 asyncio.run 桥接）
    # ================================================================

    def add_documents(self, documents, batch_size=64, document_id=None) -> int:
        return self._sync(lambda: self.aadd_documents(documents, batch_size, document_id))

    def add_document_record(self, filename, file_type, file_size, content_hash="", storage_path="", storage_backend="local") -> int:
        return self._sync(lambda: self.aadd_document_record(filename, file_type, file_size, content_hash, storage_path, storage_backend))

    def find_document_by_hash(self, content_hash: str) -> dict | None:
        return self._sync(lambda: self.afind_document_by_hash(content_hash))

    def find_document_by_filename(self, filename: str) -> dict | None:
        return self._sync(lambda: self.afind_document_by_filename(filename))

    def count_document_chunks(self, doc_id: int) -> int:
        return self._sync(lambda: self.acount_document_chunks(doc_id))

    def list_documents(self) -> list[dict]:
        return self._sync(lambda: self.alist_documents())

    def delete_document(self, doc_id: int, delete_storage: bool = True) -> dict | None:
        return self._sync(lambda: self.adelete_document(doc_id, delete_storage))

    def similarity_search(self, query, k=None, filter=None) -> list[dict]:
        return self._sync(lambda: self.asimilarity_search(query, k, filter))

    def count(self) -> int:
        return self._sync(lambda: self.acount())

    def delete_collection(self):
        return self._sync(lambda: self.adelete_collection())

    def switch_collection(self, collection_name: str):
        return self._sync(lambda: self.aswitch_collection(collection_name))

    def rename_collection(self, old_name: str, new_name: str) -> bool:
        return self._sync(lambda: self.arenmae_collection(old_name, new_name))

    def list_collections(self) -> list[str]:
        return self._sync(lambda: self.alist_collections())

    def get_all_chunks(self, limit=200, offset=0) -> list[dict]:
        return self._sync(lambda: self.aget_all_chunks(limit, offset))

    # 对话管理同步 shim
    def list_conversations(self) -> list[dict]:
        return self._sync(lambda: self.alist_conversations())

    def create_conversation(self, title=None) -> dict:
        return self._sync(lambda: self.acreate_conversation(title))

    def delete_conversation(self, conv_id: int) -> bool:
        return self._sync(lambda: self.adelete_conversation(conv_id))

    def update_conversation_title(self, conv_id: int, title: str) -> bool:
        return self._sync(lambda: self.aupdate_conversation_title(conv_id, title))

    def get_conversation_messages(self, conv_id: int) -> list[dict]:
        return self._sync(lambda: self.aget_conversation_messages(conv_id))

    def add_message(self, conv_id, role, content, sources=None, answer_type=None) -> int:
        return self._sync(lambda: self.aadd_message(conv_id, role, content, sources, answer_type))

    def update_message_content(self, msg_id, content, sources=None) -> bool:
        return self._sync(lambda: self.aupdate_message_content(msg_id, content, sources))

    # 用户管理同步 shim
    def create_user(self, username, password_hash, display_name=None, is_admin=False) -> int:
        return self._sync(lambda: self.acreate_user(username, password_hash, display_name, is_admin))

    def get_user_by_username(self, username: str) -> dict | None:
        return self._sync(lambda: self.aget_user_by_username(username))

    def update_user_login(self, user_id: int):
        return self._sync(lambda: self.aupdate_user_login(user_id))

    def list_users(self) -> list[dict]:
        return self._sync(lambda: self.alist_users())
