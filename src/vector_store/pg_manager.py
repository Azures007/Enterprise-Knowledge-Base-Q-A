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
from src.monitoring import get_metrics
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
            # 版本化迁移：为 documents 表补充版本列
            #   version      当前版本号（同名文档的每次更新 +1，从 1 开始）
            #   is_latest    是否当前生效版本（旧版本保留记录但不参与检索）
            #   prev_doc_id  指向上一个版本的文档 ID（构成版本链）
            for col_sql in (
                'ALTER TABLE documents ADD COLUMN IF NOT EXISTS version INTEGER DEFAULT 1',
                'ALTER TABLE documents ADD COLUMN IF NOT EXISTS is_latest BOOLEAN DEFAULT true',
                'ALTER TABLE documents ADD COLUMN IF NOT EXISTS prev_doc_id INTEGER',
            ):
                try:
                    await conn.execute(col_sql)
                except Exception as e:
                    logger.warning(f"documents 版本列迁移失败: {e}")

            # 文本块 + 向量表
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS chunks (
                    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    collection_id   INTEGER NOT NULL REFERENCES collections(id) ON DELETE CASCADE,
                    document_id     INTEGER REFERENCES documents(id) ON DELETE SET NULL,
                    content         TEXT NOT NULL,
                    metadata        JSONB DEFAULT '{{}}'::jsonb,
                    embedding       VECTOR({settings.VECTOR_DIMENSION}),
                    search_vector   tsvector GENERATED ALWAYS AS
                                    (to_tsvector('simple', coalesce(content, ''))) STORED,
                    created_at      TIMESTAMP DEFAULT NOW()
                )
            """)
            # 兼容旧表：无 search_vector 列时补充生成列 + GIN 索引
            try:
                await conn.execute("""
                    ALTER TABLE chunks ADD COLUMN IF NOT EXISTS search_vector tsvector
                    GENERATED ALWAYS AS (to_tsvector('simple', coalesce(content, ''))) STORED
                """)
            except Exception:
                pass
            try:
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_chunks_search_vector
                    ON chunks USING GIN (search_vector)
                """)
            except Exception:
                pass

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
            # 兼容旧表：无反馈列时补充（用户点赞/点踩，1=赞 -1=踩 NULL=无）
            try:
                await conn.execute(
                    "ALTER TABLE messages ADD COLUMN IF NOT EXISTS feedback SMALLINT"
                )
            except Exception:
                pass
            try:
                await conn.execute(
                    "ALTER TABLE messages ADD COLUMN IF NOT EXISTS feedback_comment TEXT"
                )
            except Exception:
                pass
            try:
                await conn.execute(
                    "ALTER TABLE messages ADD COLUMN IF NOT EXISTS feedback_at TIMESTAMP"
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

            # 查询审计日志表（每次问答一条）
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS query_audit_log (
                    id                SERIAL PRIMARY KEY,
                    username          VARCHAR(255),
                    user_id           INTEGER,
                    conversation_id   INTEGER,
                    question          TEXT,
                    answer            TEXT,
                    answer_type       VARCHAR(20),
                    collection        VARCHAR(255),
                    sources           JSONB DEFAULT '[]'::jsonb,
                    k                 INTEGER,
                    concise           BOOLEAN DEFAULT false,
                    from_cache        BOOLEAN DEFAULT false,
                    latency_ms        INTEGER,
                    prompt_tokens     INTEGER,
                    completion_tokens INTEGER,
                    total_tokens      INTEGER,
                    model             VARCHAR(64),
                    status            VARCHAR(20) DEFAULT 'success',
                    error_msg         TEXT,
                    created_at        TIMESTAMP DEFAULT NOW()
                )
            """)
            # 兼容旧表：缺列时自动补充（迁移安全）
            for col_sql in (
                'ALTER TABLE query_audit_log ADD COLUMN IF NOT EXISTS user_id INTEGER',
                'ALTER TABLE query_audit_log ADD COLUMN IF NOT EXISTS conversation_id INTEGER',
                'ALTER TABLE query_audit_log ADD COLUMN IF NOT EXISTS answer TEXT',
                'ALTER TABLE query_audit_log ADD COLUMN IF NOT EXISTS collection VARCHAR(255)',
                'ALTER TABLE query_audit_log ADD COLUMN IF NOT EXISTS sources JSONB DEFAULT \'[]\'::jsonb',
                'ALTER TABLE query_audit_log ADD COLUMN IF NOT EXISTS k INTEGER',
                'ALTER TABLE query_audit_log ADD COLUMN IF NOT EXISTS concise BOOLEAN DEFAULT false',
                'ALTER TABLE query_audit_log ADD COLUMN IF NOT EXISTS from_cache BOOLEAN DEFAULT false',
                'ALTER TABLE query_audit_log ADD COLUMN IF NOT EXISTS latency_ms INTEGER',
                'ALTER TABLE query_audit_log ADD COLUMN IF NOT EXISTS prompt_tokens INTEGER',
                'ALTER TABLE query_audit_log ADD COLUMN IF NOT EXISTS completion_tokens INTEGER',
                'ALTER TABLE query_audit_log ADD COLUMN IF NOT EXISTS total_tokens INTEGER',
                'ALTER TABLE query_audit_log ADD COLUMN IF NOT EXISTS model VARCHAR(64)',
                'ALTER TABLE query_audit_log ADD COLUMN IF NOT EXISTS status VARCHAR(20) DEFAULT \'success\'',
                'ALTER TABLE query_audit_log ADD COLUMN IF NOT EXISTS error_msg TEXT',
            ):
                try:
                    await conn.execute(col_sql)
                except Exception:
                    pass
            try:
                await conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_query_audit_created ON query_audit_log(created_at)"
                )
            except Exception:
                pass
            try:
                await conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_query_audit_user ON query_audit_log(username)"
                )
            except Exception:
                pass

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

    async def aget_collection_stats(self, names: list[str] | None = None) -> list[dict]:
        """
        一次 SQL 聚合多个集合的统计信息（chunk 数、文档数）。

        Args:
            names: 需要统计的集合名列表；None 表示全部集合

        Returns:
            list[dict]: 每项含 name, chunk_count, document_count
        """
        async with (await self._aconn()).acquire() as conn:
            if names:
                rows = await conn.fetch(
                    """
                    SELECT
                        c.name,
                        (SELECT COUNT(*) FROM chunks ch WHERE ch.collection_id = c.id) AS chunk_count,
                        (SELECT COUNT(*) FROM documents d WHERE d.collection_id = c.id) AS document_count
                    FROM collections c
                    WHERE c.name = ANY($1)
                    ORDER BY c.name
                    """,
                    names,
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT
                        c.name,
                        (SELECT COUNT(*) FROM chunks ch WHERE ch.collection_id = c.id) AS chunk_count,
                        (SELECT COUNT(*) FROM documents d WHERE d.collection_id = c.id) AS document_count
                    FROM collections c
                    ORDER BY c.name
                    """
                )
        return [
            {
                "name": r["name"],
                "chunk_count": r["chunk_count"],
                "document_count": r["document_count"],
            }
            for r in rows
        ]

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

    async def _amark_document_superseded(self, doc_id: int) -> None:
        """将文档标记为已失效（被新版本覆盖），其 chunks 不再参与检索。"""
        async with (await self._aconn()).acquire() as conn:
            await conn.execute(
                "UPDATE documents SET is_latest = false WHERE id = $1",
                doc_id,
            )

    async def _aget_document_version(self, doc_id: int) -> int:
        """获取文档当前版本号。"""
        async with (await self._aconn()).acquire() as conn:
            row = await conn.fetchrow(
                "SELECT version FROM documents WHERE id = $1",
                doc_id,
            )
        return (row["version"] if row and row["version"] else 1) or 1

    async def _amark_document_version(self, doc_id: int, version: int, prev_doc_id: int | None) -> None:
        """设置文档的版本号与前置版本链。"""
        async with (await self._aconn()).acquire() as conn:
            await conn.execute(
                "UPDATE documents SET version = $1, prev_doc_id = $2 WHERE id = $3",
                version, prev_doc_id, doc_id,
            )

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
                WHERE collection_id = $1 AND content_hash = $2 AND is_latest = true
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
                WHERE collection_id = $1 AND filename = $2 AND is_latest = true
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
        """获取当前集合中的所有文档列表（含 chunk 数量，仅最新版本）。"""
        coll_id = await self._aget_collection_id(self.collection_name)
        if coll_id is None:
            return []
        async with (await self._aconn()).acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT d.id, d.filename, d.file_type, d.file_size,
                       d.storage_backend, d.storage_path, d.created_at,
                       d.version, d.is_latest, d.prev_doc_id,
                       (
                         SELECT COUNT(*) FROM chunks c
                         WHERE c.document_id = d.id
                            OR (c.document_id IS NULL AND c.metadata->>'filename' = d.filename)
                       ) AS chunk_count
                FROM documents d
                WHERE d.collection_id = $1 AND d.is_latest = true
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
                "version": r["version"],
                "is_latest": r["is_latest"],
                "prev_doc_id": r["prev_doc_id"],
            }
            for r in rows
        ]

    # ================================================================
    # 文档版本化（重传同名文件时保留历史版本，支持回滚）
    # ================================================================

    async def areplace_document_for_new_version(self, old_doc_id: int) -> None:
        """重传覆盖时，将旧文档标记为已失效（is_latest=false），保留其记录与 chunks 供回滚。"""
        await self._amark_document_superseded(old_doc_id)
        logger.info(f"文档 {old_doc_id} 已被新版本覆盖（is_latest=false，保留历史版本）")

    async def alist_document_versions(self, doc_id: int) -> list[dict[str, Any]]:
        """
        获取文档的完整版本链（含当前版本，按版本号降序）。

        Returns:
            [{"id", "version", "is_latest", "filename", "file_size",
              "content_hash", "created_at", "chunk_count"}, ...]
        """
        async with (await self._aconn()).acquire() as conn:
            # 先定位文档所在集合与文件族：从当前文档沿 prev_doc_id 回溯，
            # 或从该文档向上找最新的版本族根
            row = await conn.fetchrow(
                "SELECT collection_id, filename FROM documents WHERE id = $1",
                doc_id,
            )
            if row is None:
                return []
            coll_id, filename = row["collection_id"], row["filename"]
            # 同一文件名（同一集合）的所有版本
            rows = await conn.fetch(
                """
                SELECT d.id, d.version, d.is_latest, d.filename, d.file_size,
                       d.content_hash, d.created_at, d.prev_doc_id,
                       (
                         SELECT COUNT(*) FROM chunks c
                         WHERE c.document_id = d.id
                            OR (c.document_id IS NULL AND c.metadata->>'filename' = d.filename AND d.version = 1)
                       ) AS chunk_count
                FROM documents d
                WHERE d.collection_id = $1 AND d.filename = $2
                ORDER BY d.version DESC NULLS LAST, d.id DESC
                """,
                coll_id, filename,
            )
        return [
            {
                "id": r["id"],
                "version": r["version"],
                "is_latest": r["is_latest"],
                "filename": r["filename"],
                "file_size": r["file_size"],
                "content_hash": r["content_hash"],
                "prev_doc_id": r["prev_doc_id"],
                "created_at": r["created_at"].isoformat() if r["created_at"] else None,
                "chunk_count": r["chunk_count"],
            }
            for r in rows
        ]

    async def arollback_document(self, doc_id: int) -> dict[str, Any] | None:
        """
        回滚到指定版本的文档。

        将该版本标记为最新（is_latest=true），当前最新版本标记为失效。
        若指定版本已是当前版本，直接返回其信息不动作。

        Returns:
            回滚后的文档信息 dict（含 filename, version），或 None（文档不存在）
        """
        async with (await self._aconn()).acquire() as conn:
            target = await conn.fetchrow(
                "SELECT id, collection_id, filename, version, is_latest FROM documents WHERE id = $1",
                doc_id,
            )
            if target is None:
                return None
            # 若目标已是当前版本，无需回滚
            if target["is_latest"]:
                return {
                    "id": target["id"],
                    "filename": target["filename"],
                    "version": target["version"],
                    "already_latest": True,
                }

            # 标记当前最新版本失效
            await conn.execute(
                """
                UPDATE documents SET is_latest = false
                WHERE collection_id = $1 AND filename = $2 AND is_latest = true
                """,
                target["collection_id"], target["filename"],
            )
            # 目标版本设为最新
            await conn.execute(
                "UPDATE documents SET is_latest = true WHERE id = $1",
                doc_id,
            )

        return {
            "id": target["id"],
            "filename": target["filename"],
            "version": target["version"],
            "already_latest": False,
        }

    async def adelete_document(self, doc_id: int, delete_storage: bool = True) -> dict | None:
        """
        删除文档及其所有分块（含历史版本），可选择同时删除存储文件。

        版本化场景下，同名文档的所有历史版本一并删除，避免孤儿记录。
        """
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

            # 找到该文档所在的整个版本链（同集合 + 同文件名的所有版本）
            family_rows = await conn.fetch(
                """
                SELECT id, storage_path, storage_backend
                FROM documents
                WHERE collection_id = (SELECT collection_id FROM documents WHERE id = $1)
                  AND filename = (SELECT filename FROM documents WHERE id = $1)
                """,
                doc_id,
            )
            family_ids = [r["id"] for r in family_rows]
            # 记录所有版本的存储文件路径（供删除存储后端）
            storage_paths = [
                (r["storage_path"], r["storage_backend"])
                for r in family_rows if r["storage_path"]
            ]

            # 删除版本链所有文档关联的 chunks（document_id 关联）
            deleted_chunks = 0
            for fid in family_ids:
                deleted_chunks += _parse_rowcount(
                    await conn.execute("DELETE FROM chunks WHERE document_id = $1", fid)
                )
            # 兼容历史数据（未关联 document_id 但文件名匹配，仅删除最新家族对应记录）
            deleted_by_name = _parse_rowcount(
                await conn.execute(
                    "DELETE FROM chunks WHERE document_id IS NULL AND metadata->>'filename' = $1",
                    doc_info["filename"],
                )
            )
            deleted_chunks += deleted_by_name
            # 删除版本链所有文档记录
            if family_ids:
                await conn.execute(
                    "DELETE FROM documents WHERE id = ANY($1::int[])",
                    family_ids,
                )

        # 删除存储后端中的原始文件（所有版本）
        if delete_storage and storage_paths:
            from src.storage import get_storage
            try:
                storage = get_storage()
            except Exception:
                storage = None
            for spath, _sbackend in storage_paths:
                if storage is not None:
                    try:
                        storage.delete(spath)
                        logger.info(f"存储文件已删除: {spath}")
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
                conds.append(f"c.metadata->>${n - 1} = ${n}")
            filter_sql = " AND " + " AND ".join(conds)

        async with (await self._aconn()).acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT
                    c.content AS content,
                    c.metadata AS metadata,
                    1 - (c.embedding <=> $1::vector) AS cosine_similarity
                FROM chunks c
                LEFT JOIN documents d ON c.document_id = d.id
                WHERE c.collection_id = $2{filter_sql}
                  AND (c.document_id IS NULL OR d.is_latest = true)
                ORDER BY c.embedding <=> $1::vector
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

    async def ahybrid_search(
        self,
        query: str,
        k: int | None = None,
        filter: dict[str, Any] | None = None,
        vector_weight: float = 0.7,
        keyword_weight: float = 0.3,
    ) -> list[dict[str, Any]]:
        """
        混合检索：向量语义召回 + 关键词全文召回 双通道融合。

        与纯向量检索（asimilarity_search）的区别：
        - 向量通道：HNSW 索引的余弦相似度召回，捕捉语义相近但用词不同的块
        - 关键词通道：PostgreSQL 全文检索（search_vector tsvector + GIN 索引），
          精确匹配专有名词/编号/代码片段（向量分往往不高，纯向量易漏检）
        - 融合策略由 HYBRID_FUSION_MODE 决定：
            - "rrf"（默认）：Reciprocal Rank Fusion，对两通道的排序取倒数加权，
              不依赖分数绝对值可比性，避免向量分/关键词分量纲差异互相压轧
            - "weighted"：保留经典的 向量 0.7 + 关键词 0.3 线性加权（兼容旧行为）

        解决纯向量检索对专有名词/编号/代码片段召回差的问题。
        """
        k = k or settings.RETRIEVAL_TOP_K
        # 按配置选择融合策略：rrf 走双通道融合，weighted 走线性加权（旧逻辑）
        fusion_mode = getattr(settings, "HYBRID_FUSION_MODE", "rrf")
        if fusion_mode == "rrf":
            return await self._ahybrid_search_rrf(
                query, k=k, filter=filter,
                vector_weight=vector_weight, keyword_weight=keyword_weight,
            )
        return await self._ahybrid_search_weighted(
            query, k=k, filter=filter,
            vector_weight=vector_weight, keyword_weight=keyword_weight,
        )

    # ------------------------------------------------------------------
    # 融合模式 2：线性加权（兼容旧行为）
    # ------------------------------------------------------------------

    async def _ahybrid_search_weighted(
        self,
        query: str,
        k: int | None = None,
        filter: dict[str, Any] | None = None,
        vector_weight: float = 0.7,
        keyword_weight: float = 0.3,
    ) -> list[dict[str, Any]]:
        """线性加权混合检索：向量分 + n-gram 关键词命中率加权。

        - 向量分：余弦相似度（0~1）
        - 关键词分：n-gram 关键词命中率（0~1）。中文按 2~4 字切分，
          英文按整词，统计命中率。不依赖 PostgreSQL 分词器（对中文有效）。
        - 综合分 = vector_weight * 向量分 + keyword_weight * 关键词分
        """
        k = k or settings.RETRIEVAL_TOP_K

        # 计算查询向量
        try:
            query_embedding = await self._aembed_query(query)
        except Exception as e:
            raise PGVectorStoreError(f"查询嵌入计算失败: {e}") from e
        embedding_str = "[" + ",".join(str(v) for v in query_embedding) + "]"

        coll_id = await self._aensure_collection(self.collection_name)

        # 提取查询关键词（英文整词 + 中文 2~4 字 n-gram）
        keywords = self._extract_query_keywords(query)

        # 无有效关键词时退化为纯向量检索
        if not keywords:
            return await self.asimilarity_search(query, k=k, filter=filter)

        # 元数据过滤
        filter_sql = ""
        params: list = [embedding_str, coll_id, k]
        if filter:
            conds = []
            for key, val in filter.items():
                params.append(key)
                params.append(str(val))
                n = len(params)
                conds.append(f"c.metadata->>${n - 1} = ${n}")
            filter_sql = " AND " + " AND ".join(conds)

        # 关键词命中计数：每命中一个关键词 content LIKE %kw% 计 1 分（boolean→int）
        kw_conditions = []
        for i, kw in enumerate(keywords):
            params.append(f"%{kw}%")
            kw_conditions.append(f"(CASE WHEN c.content LIKE ${len(params)} THEN 1 ELSE 0 END)")
        kw_score_sql = " + ".join(kw_conditions) if kw_conditions else "0"

        async with (await self._aconn()).acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT
                    c.content AS content,
                    c.metadata AS metadata,
                    1 - (c.embedding <=> $1::vector) AS cosine_similarity,
                    -- 关键词命中数（0 ~ len(keywords)），归一化到 0~1
                    LEAST(1.0, ({kw_score_sql})::float / {max(len(keywords), 1)}) AS keyword_score
                FROM chunks c
                LEFT JOIN documents d ON c.document_id = d.id
                WHERE c.collection_id = $2{filter_sql}
                  AND (c.document_id IS NULL OR d.is_latest = true)
                ORDER BY ({vector_weight} * (1 - (c.embedding <=> $1::vector))
                         + {keyword_weight} * LEAST(1.0, ({kw_score_sql})::float / {max(len(keywords), 1)})) DESC
                LIMIT $3
                """,
                *params,
            )

        results = []
        for row in rows:
            cosine_sim = float(row["cosine_similarity"])
            keyword_score = float(row["keyword_score"])
            vec_score = max(0, cosine_sim)
            combined = vector_weight * vec_score + keyword_weight * keyword_score
            results.append({
                "content": row["content"],
                "metadata": _json_or_dict(row["metadata"]),
                "score": round(combined, 4),
                "vector_score": round(vec_score, 4),
                "keyword_score": round(keyword_score, 4),
                "distance": round(1.0 - vec_score, 4),
            })

        logger.debug(
            f"混合检索完成: query='{query[:50]}...', "
            f"k={k}, 结果数={len(results)}"
        )
        return results

    # ------------------------------------------------------------------
    # 融合模式 1：RRF 双通道召回（向量 + 全文，默认）
    # ------------------------------------------------------------------

    async def _ahybrid_search_rrf(
        self,
        query: str,
        k: int | None = None,
        filter: dict[str, Any] | None = None,
        vector_weight: float = 0.7,
        keyword_weight: float = 0.3,
    ) -> list[dict[str, Any]]:
        """
        RRF 双通道混合检索：向量语义召回 + PostgreSQL 全文召回分别取 top-candidate，
        再用 Reciprocal Rank Fusion 融合排序。

        设计要点：
        1. 关键词通道走 chunks.search_vector（tsvector 生成列 + GIN 索引），
           用 plainto_tsquery('simple', ...) 做词元匹配。相比旧实现的
           LIKE '%kw%' 全表扫描，能用上索引，且不依赖外部中文分词器。
        2. RRF 分数 = Σ w_channel / (60 + rank_channel)，对通道内的排名倒数加权，
           不要求两个通道的分数在同一量纲 —— 避免向量分与关键词分量级差异
           互相压轧（旧的线性加权在候选数大时，向量分几乎主导）。
        3. 两个通道都限定在候选池内取 top candidate_k（默认 20），再取并集，
           最终 LIMIT k。向量通道用 HNSW 索引，全文通道用 GIN 索引，均非全表扫描。
        """
        k = k or settings.RETRIEVAL_TOP_K
        candidate_k = getattr(settings, "HYBRID_CANDIDATE_K", 20)
        fusion_const = getattr(settings, "HYBRID_RRF_K", 60)

        # ---- 通道 1：向量语义召回（HNSW） ----
        vec_results = await self.asimilarity_search(query, k=candidate_k, filter=filter)

        # ---- 通道 2：关键词全文召回（tsvector + GIN） ----
        kw_results: list[dict[str, Any]] = []
        kw_terms = self._extract_query_keywords(query)
        # plainto_tsquery('simple', '词1 词2'): 词元全部命中才匹配。
        # 中文按空格间隔传入，等价于 AND 语义；无有效词元时跳过关键词通道。
        if kw_terms:
            kw_results = await self._akeyword_search(
                query, k=candidate_k, filter=filter, terms=kw_terms,
            )

        # ---- RRF 融合 ----
        rank_scores: dict[str, float] = {}
        channel_flags: dict[str, dict] = {}

        # 向量通道贡献
        for rank, doc in enumerate(vec_results):
            key = doc["content"]
            rank_scores[key] = rank_scores.get(key, 0.0) + vector_weight / (fusion_const + rank)
            channel_flags.setdefault(key, {})["vector"] = doc
        # 全文通道贡献
        for rank, doc in enumerate(kw_results):
            key = doc["content"]
            rank_scores[key] = rank_scores.get(key, 0.0) + keyword_weight / (fusion_const + rank)
            entry = channel_flags.setdefault(key, {})
            entry["keyword"] = doc

        # 合并通道信息，按 RRF 分数降序取前 k
        merged: list[dict[str, Any]] = []
        for key, score in rank_scores.items():
            channels = channel_flags.get(key, {})
            vec_doc = channels.get("vector")
            kw_doc = channels.get("keyword")
            # 综合分取两通道分数的加权平均（与 RRF 排名并行展示，供重排参考）
            vec_score = float(vec_doc.get("score", 0.0)) if vec_doc else 0.0
            kw_score = float(kw_doc.get("score", 0.0)) if kw_doc else 0.0
            if vec_doc and kw_doc:
                combined = vector_weight * vec_score + keyword_weight * kw_score
            else:
                combined = vec_score if vec_doc else kw_score

            item = {
                "content": key,
                "metadata": (vec_doc or kw_doc).get("metadata", {}),
                "score": round(combined, 4),
                "rrf_score": round(score, 6),
                "vector_score": round(vec_score, 4) if vec_doc else None,
                "keyword_score": round(kw_score, 4) if kw_doc else None,
                "distance": round(1.0 - vec_score, 4) if vec_doc else None,
            }
            merged.append(item)

        merged.sort(key=lambda d: d["rrf_score"], reverse=True)

        # 保留 rrf 分数最高的前 k；分数并列时优先双通道都命中的块
        merged.sort(
            key=lambda d: (
                d["rrf_score"],
                1 if d["vector_score"] is not None and d["keyword_score"] is not None else 0,
            ),
            reverse=True,
        )
        merged = merged[:k]

        logger.debug(
            f"RRF 混合检索完成: query='{query[:50]}...', "
            f"向量通道={len(vec_results)}, 全文通道={len(kw_results)}, "
            f"融合后={len(merged)} (模式={fusion_const})"
        )
        get_metrics().set_gauge(
            "hybrid_fusion_mode", getattr(settings, "HYBRID_FUSION_MODE", "rrf")
        )
        # 关键词通道召回数 → 观测指标（纯向量查询可能不触发关键词通道）
        if kw_results:
            from src.monitoring import record_hybrid_keyword_channel
            record_hybrid_keyword_channel(len(kw_results))
        return merged

    async def _akeyword_search(
        self,
        query: str,
        k: int | None = None,
        filter: dict[str, Any] | None = None,
        terms: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """
        关键词全文检索：利用 chunks.search_vector（tsvector 生成列 + GIN 索引）。

        用 plainto_tsquery('simple', terms_joined) 构造查询，匹配 content 的
        word token（'simple' 分词器按空白/标点切词，中文整句视为一个 token，
        因此仅对查询中已有的英文/数字词元有效）。score 用 ts_rank 归一化。

        查询串以空格连接词元，避免 plainto_tsquery 将多个中文 n-gram 当 OR
        处理导致召回膨胀。
        """
        k = k or settings.RETRIEVAL_TOP_K
        coll_id = await self._aensure_collection(self.collection_name)
        if coll_id is None:
            return []
        if not terms:
            return []

        # 词元统一小写（'simple' 分词器对英文大小写不敏感）
        cleaned = [t.lower() for t in terms if t and t.strip()]
        if not cleaned:
            return []
        ts_query = " ".join(cleaned)

        filter_sql = ""
        params: list = [coll_id, k, ts_query]
        if filter:
            conds = []
            for key, val in filter.items():
                params.append(key)
                params.append(str(val))
                n = len(params)
                conds.append(f"c.metadata->>${n - 1} = ${n}")
            filter_sql = " AND " + " AND ".join(conds)

        async with (await self._aconn()).acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT
                    c.content AS content,
                    c.metadata AS metadata,
                    ts_rank(c.search_vector, plainto_tsquery('simple', $3)) AS rank_score
                FROM chunks c
                LEFT JOIN documents d ON c.document_id = d.id
                WHERE c.collection_id = $1{filter_sql}
                  AND (c.document_id IS NULL OR d.is_latest = true)
                  AND c.search_vector @@ plainto_tsquery('simple', $3)
                ORDER BY rank_score DESC
                LIMIT $2
                """,
                *params,
            )

        results = []
        for row in rows:
            rank_score = float(row["rank_score"])
            # ts_rank 归一化到 0~1（对 'simple' 分词器，score 通常很小）
            norm_score = min(1.0, rank_score)
            results.append({
                "content": row["content"],
                "metadata": _json_or_dict(row["metadata"]),
                "score": round(norm_score, 4),
            })

        logger.debug(
            f"关键词全文检索完成: query='{query[:50]}...', "
            f"terms={len(cleaned)}, 结果数={len(results)}"
        )
        return results

    @staticmethod
    def _extract_query_keywords(query: str) -> list[str]:
        """
        从查询文本提取关键词列表（中英文兼容）。

        - 英文/数字：按整词保留（如 ERP、192.168、postgresql）
        - 中文：按 2~4 字 n-gram 切分（如 "考勤制度" → 考勤、勤制、制度、考勤制...）
        - 过滤过短/无意义的词，去重
        """
        import re

        keywords = []
        # 英文单词 + 数字 + 代码片段
        en_words = re.findall(r"[a-zA-Z0-9][a-zA-Z0-9_.\-]{1,}", query)
        keywords.extend(w.lower() for w in en_words)

        # 中文连续片段（2字及以上）
        cn_segments = re.findall(r"[一-鿿]{2,}", query)
        for seg in cn_segments:
            # 2-gram 和 3-gram 作为关键词（覆盖中文词边界）
            for n in (2, 3):
                for i in range(len(seg) - n + 1):
                    gram = seg[i:i + n]
                    if gram not in keywords:
                        keywords.append(gram)

        # 去重并限制数量（避免 SQL 过长）
        seen = set()
        result = []
        for kw in keywords:
            if kw not in seen:
                seen.add(kw)
                result.append(kw)
            if len(result) >= 15:
                break
        return result

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

    async def aget_conversation_owner(self, conv_id: int) -> int | None:
        """获取对话归属用户 ID（None 表示匿名/不存在）。"""
        async with (await self._aconn()).acquire() as conn:
            return await conn.fetchval(
                "SELECT user_id FROM conversations WHERE id = $1", conv_id
            )

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
                SELECT id, role, content, sources, answer_type, is_stale,
                       feedback, feedback_comment, feedback_at, created_at
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
                "feedback": r["feedback"],
                "feedback_comment": r["feedback_comment"],
                "feedback_at": r["feedback_at"].isoformat() if r["feedback_at"] else None,
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

    async def aset_message_feedback(
        self, msg_id: int, feedback: int, comment: str | None = None,
    ) -> bool:
        """
        设置消息的用户反馈（点赞/点踩）。

        Args:
            msg_id:   消息 ID
            feedback: 1=赞, -1=踩, 0=清除反馈
            comment:  点踩原因（可选）

        Returns:
            是否成功更新
        """
        async with (await self._aconn()).acquire() as conn:
            if feedback == 0:
                return _parse_rowcount(
                    await conn.execute(
                        """
                        UPDATE messages
                        SET feedback = NULL, feedback_comment = NULL, feedback_at = NULL
                        WHERE id = $1
                        """,
                        msg_id,
                    )
                ) > 0
            return _parse_rowcount(
                await conn.execute(
                    """
                    UPDATE messages
                    SET feedback = $1, feedback_comment = $2, feedback_at = NOW()
                    WHERE id = $3
                    """,
                    feedback, comment, msg_id,
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
    # 查询审计（异步核心）
    # ================================================================

    async def aadd_query_audit(
        self,
        *,
        username: str | None = None,
        user_id: int | None = None,
        conversation_id: int | None = None,
        question: str = "",
        answer: str = "",
        answer_type: str = "",
        collection: str | None = None,
        sources: list | None = None,
        k: int | None = None,
        concise: bool = False,
        from_cache: bool = False,
        latency_ms: int | None = None,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        total_tokens: int | None = None,
        model: str | None = None,
        status: str = "success",
        error_msg: str | None = None,
    ) -> None:
        """
        记录一次问答的审计日志。

        写入失败仅记 warning，不抛出（审计不应影响主流程）。
        """
        sources_json = json.dumps(sources or [], ensure_ascii=False)
        try:
            async with (await self._aconn()).acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO query_audit_log
                    (username, user_id, conversation_id, question, answer,
                     answer_type, collection, sources, k, concise, from_cache,
                     latency_ms, prompt_tokens, completion_tokens, total_tokens,
                     model, status, error_msg)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8::jsonb, $9, $10, $11,
                            $12, $13, $14, $15, $16, $17, $18)
                    """,
                    username, user_id, conversation_id, question, answer,
                    answer_type, collection, sources_json, k, concise, from_cache,
                    latency_ms, prompt_tokens, completion_tokens, total_tokens,
                    model, status, error_msg,
                )
            logger.debug(f"查询审计已记录: user={username}, q='{question[:40]}'")
        except Exception as e:
            logger.warning(f"查询审计记录失败: {e}")

    async def alist_query_audit(
        self,
        limit: int = 50,
        offset: int = 0,
        username: str | None = None,
    ) -> list[dict[str, Any]]:
        """分页查询审计记录（按时间倒序），可选按用户名过滤。"""
        async with (await self._aconn()).acquire() as conn:
            if username:
                rows = await conn.fetch(
                    """
                    SELECT id, username, user_id, conversation_id, question, answer,
                           answer_type, collection, sources, k, concise, from_cache,
                           latency_ms, prompt_tokens, completion_tokens, total_tokens,
                           model, status, error_msg, created_at
                    FROM query_audit_log
                    WHERE username = $1
                    ORDER BY created_at DESC, id DESC
                    LIMIT $2 OFFSET $3
                    """,
                    username, limit, offset,
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT id, username, user_id, conversation_id, question, answer,
                           answer_type, collection, sources, k, concise, from_cache,
                           latency_ms, prompt_tokens, completion_tokens, total_tokens,
                           model, status, error_msg, created_at
                    FROM query_audit_log
                    ORDER BY created_at DESC, id DESC
                    LIMIT $1 OFFSET $2
                    """,
                    limit, offset,
                )
        return [
            {
                "id": r["id"],
                "username": r["username"],
                "user_id": r["user_id"],
                "conversation_id": r["conversation_id"],
                "question": r["question"],
                "answer": r["answer"],
                "answer_type": r["answer_type"],
                "collection": r["collection"],
                "sources": _json_or_dict(r["sources"]) if r["sources"] else [],
                "k": r["k"],
                "concise": bool(r["concise"]),
                "from_cache": bool(r["from_cache"]),
                "latency_ms": r["latency_ms"],
                "prompt_tokens": r["prompt_tokens"],
                "completion_tokens": r["completion_tokens"],
                "total_tokens": r["total_tokens"],
                "model": r["model"],
                "status": r["status"],
                "error_msg": r["error_msg"],
                "created_at": r["created_at"].isoformat() if r["created_at"] else None,
            }
            for r in rows
        ]

    async def aget_query_audit_summary(self) -> dict[str, Any]:
        """
        查询审计汇总统计（一次 SQL 聚合）。

        Returns:
            dict 含 total_queries, cache_hit_count, cache_hit_rate,
                 avg_latency_ms, total_tokens, answer_type_dist, top_questions
        """
        async with (await self._aconn()).acquire() as conn:
            # 整体统计（仅成功记录计入延迟/缓存率）
            row = await conn.fetchrow(
                """
                SELECT
                    COUNT(*) AS total_queries,
                    COUNT(*) FILTER (WHERE from_cache) AS cache_hit_count,
                    ROUND(AVG(latency_ms) FILTER (WHERE status = 'success')) AS avg_latency_ms,
                    COALESCE(SUM(total_tokens), 0) AS total_tokens,
                    COALESCE(SUM(prompt_tokens), 0) AS total_prompt_tokens,
                    COALESCE(SUM(completion_tokens), 0) AS total_completion_tokens
                FROM query_audit_log
                """
            )
            # 回答类型分布
            type_rows = await conn.fetch(
                """
                SELECT answer_type, COUNT(*) AS cnt
                FROM query_audit_log
                GROUP BY answer_type
                ORDER BY cnt DESC
                """
            )
            # 热门问题 TopN
            q_rows = await conn.fetch(
                """
                SELECT question, COUNT(*) AS cnt
                FROM query_audit_log
                WHERE question <> ''
                GROUP BY question
                ORDER BY cnt DESC
                LIMIT 10
                """
            )

        total = row["total_queries"] or 0
        cache_hits = row["cache_hit_count"] or 0
        return {
            "total_queries": total,
            "cache_hit_count": cache_hits,
            "cache_hit_rate": round(cache_hits / total, 4) if total else 0.0,
            "avg_latency_ms": row["avg_latency_ms"],
            "total_tokens": row["total_tokens"],
            "total_prompt_tokens": row["total_prompt_tokens"],
            "total_completion_tokens": row["total_completion_tokens"],
            "answer_type_dist": [
                {"answer_type": r["answer_type"], "count": r["cnt"]} for r in type_rows
            ],
            "top_questions": [
                {"question": r["question"], "count": r["cnt"]} for r in q_rows
            ],
        }

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

    def hybrid_search(self, query, k=None, filter=None, vector_weight=0.7, keyword_weight=0.3) -> list[dict]:
        return self._sync(lambda: self.ahybrid_search(query, k, filter, vector_weight, keyword_weight))

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

    def set_message_feedback(self, msg_id, feedback, comment=None) -> bool:
        return self._sync(lambda: self.aset_message_feedback(msg_id, feedback, comment))

    def add_query_audit(self, **fields) -> None:
        return self._sync(lambda: self.aadd_query_audit(**fields))

    def list_query_audit(self, limit=50, offset=0, username=None) -> list[dict]:
        return self._sync(lambda: self.alist_query_audit(limit, offset, username))

    def get_query_audit_summary(self) -> dict:
        return self._sync(lambda: self.aget_query_audit_summary())

    # 用户管理同步 shim
    def create_user(self, username, password_hash, display_name=None, is_admin=False) -> int:
        return self._sync(lambda: self.acreate_user(username, password_hash, display_name, is_admin))

    def get_user_by_username(self, username: str) -> dict | None:
        return self._sync(lambda: self.aget_user_by_username(username))

    def update_user_login(self, user_id: int):
        return self._sync(lambda: self.aupdate_user_login(user_id))

    def list_users(self) -> list[dict]:
        return self._sync(lambda: self.alist_users())
