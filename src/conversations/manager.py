"""
=============================================================================
对话管理模块

基于 SQLite 实现对话的持久化存储。每个对话包含多条消息记录，
支持对话的增删改查和消息的历史回溯。

表结构:
    conversations: id, title, created_at, updated_at
    messages:      id, conversation_id, role, content, sources, answer_type, created_at
=============================================================================

使用方法:
    from src.conversations.manager import ConversationManager

    mgr = ConversationManager("data/conversations.db")
    conv = mgr.create_conversation()
    mgr.add_message(conv["id"], "user", "法国首都是什么？")
    mgr.add_message(conv["id"], "ai", "巴黎", sources=[], answer_type="general")
    messages = mgr.get_messages(conv["id"])
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class ConversationManager:
    """对话管理器，基于 SQLite 实现持久化。"""

    def __init__(self, db_path: str | Path = "data/conversations.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        logger.info(f"对话管理器初始化完成: {self.db_path}")

    # ================================================================
    # 数据库初始化
    # ================================================================

    def _init_db(self):
        """创建对话和消息表（如不存在）"""
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    title       TEXT    NOT NULL DEFAULT '新对话',
                    created_at  TEXT    NOT NULL,
                    updated_at  TEXT    NOT NULL
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id INTEGER NOT NULL,
                    role            TEXT    NOT NULL CHECK(role IN ('user','ai','system')),
                    content         TEXT    NOT NULL DEFAULT '',
                    sources         TEXT    DEFAULT '[]',
                    answer_type     TEXT    DEFAULT NULL,
                    created_at      TEXT    NOT NULL,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_messages_conv
                    ON messages(conversation_id, created_at);
            """)

    # ================================================================
    # 对话 CRUD
    # ================================================================

    def list_conversations(self) -> list[dict[str, Any]]:
        """获取所有对话列表（按更新时间倒序）"""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT c.id, c.title, c.created_at, c.updated_at,
                       (SELECT COUNT(*) FROM messages m WHERE m.conversation_id = c.id) AS msg_count
                FROM conversations c
                ORDER BY c.updated_at DESC
                """
            ).fetchall()

        return [
            {
                "id": row[0],
                "title": row[1],
                "created_at": row[2],
                "updated_at": row[3],
                "message_count": row[4],
            }
            for row in rows
        ]

    def create_conversation(self, title: str | None = None) -> dict[str, Any]:
        """创建新对话，返回对话信息"""
        now = datetime.now().isoformat()
        final_title = title or "新对话"

        with self._connect() as conn:
            cursor = conn.execute(
                "INSERT INTO conversations (title, created_at, updated_at) VALUES (?, ?, ?)",
                (final_title, now, now),
            )
            conv_id = cursor.lastrowid

        logger.info(f"创建对话 [{conv_id}]: {final_title}")
        return {
            "id": conv_id,
            "title": final_title,
            "created_at": now,
            "updated_at": now,
            "message_count": 0,
        }

    def delete_conversation(self, conv_id: int) -> bool:
        """删除对话及其所有消息"""
        with self._connect() as conn:
            conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conv_id,))
            cursor = conn.execute(
                "DELETE FROM conversations WHERE id = ?", (conv_id,)
            )
            deleted = cursor.rowcount > 0

        if deleted:
            logger.info(f"删除对话 [{conv_id}]")
        return deleted

    def update_title(self, conv_id: int, title: str) -> bool:
        """修改对话标题"""
        now = datetime.now().isoformat()
        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
                (title, now, conv_id),
            )
            return cursor.rowcount > 0

    def get_conversation(self, conv_id: int) -> dict[str, Any] | None:
        """获取单个对话信息"""
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT c.id, c.title, c.created_at, c.updated_at,
                       (SELECT COUNT(*) FROM messages m WHERE m.conversation_id = c.id) AS msg_count
                FROM conversations c WHERE c.id = ?
                """,
                (conv_id,),
            ).fetchone()

        if not row:
            return None

        return {
            "id": row[0],
            "title": row[1],
            "created_at": row[2],
            "updated_at": row[3],
            "message_count": row[4],
        }

    # ================================================================
    # 消息管理
    # ================================================================

    def get_messages(self, conv_id: int) -> list[dict[str, Any]]:
        """获取指定对话的全部消息（按时间正序）"""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, role, content, sources, answer_type, created_at
                FROM messages
                WHERE conversation_id = ?
                ORDER BY created_at ASC, id ASC
                """,
                (conv_id,),
            ).fetchall()

        return [
            {
                "id": row[0],
                "role": row[1],
                "content": row[2],
                "sources": json.loads(row[3]) if row[3] else [],
                "answer_type": row[4],
                "created_at": row[5],
            }
            for row in rows
        ]

    def add_message(
        self,
        conv_id: int,
        role: str,
        content: str,
        sources: list | None = None,
        answer_type: str | None = None,
    ) -> int:
        """
        向对话中添加一条消息。

        Returns:
            新消息的 ID
        """
        now = datetime.now().isoformat()
        sources_json = json.dumps(sources or [], ensure_ascii=False)

        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO messages (conversation_id, role, content, sources, answer_type, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (conv_id, role, content, sources_json, answer_type, now),
            )
            msg_id = cursor.lastrowid

            # 更新对话的 updated_at
            conn.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (now, conv_id),
            )

        return msg_id

    def update_message_content(
        self, msg_id: int, content: str, sources: list | None = None
    ) -> bool:
        """更新消息内容（用于流式完成后补充完整内容）"""
        sources_json = json.dumps(sources or [], ensure_ascii=False)
        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE messages SET content = ?, sources = ? WHERE id = ?",
                (content, sources_json, msg_id),
            )
            return cursor.rowcount > 0

    # ================================================================
    # 内部工具
    # ================================================================

    def _connect(self) -> sqlite3.Connection:
        """获取数据库连接"""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn


class PGConversationManager:
    """
    基于 PostgreSQL 的对话管理器。

    与 ConversationManager 接口兼容，内部委托给 PGVectorStore。
    提供同步与异步（a* 前缀）两套方法。
    """

    def __init__(self, vector_store=None):
        """
        Args:
            vector_store: PGVectorStore 实例（如为 None 则自动创建）
        """
        if vector_store is None:
            from src.embeddings import BailianEmbeddings
            from src.vector_store import PGVectorStore
            embedder = BailianEmbeddings()
            vector_store = PGVectorStore(embedder)
        self._store = vector_store
        logger.info("PG 对话管理器初始化完成")

    def list_conversations(self) -> list[dict]:
        return self._store.list_conversations()

    def create_conversation(self, title=None) -> dict:
        return self._store.create_conversation(title)

    def delete_conversation(self, conv_id: int) -> bool:
        return self._store.delete_conversation(conv_id)

    def update_title(self, conv_id: int, title: str) -> bool:
        return self._store.update_conversation_title(conv_id, title)

    def get_conversation(self, conv_id: int) -> dict | None:
        convs = self._store.list_conversations()
        for c in convs:
            if c["id"] == conv_id:
                return c
        return None

    def get_messages(self, conv_id: int) -> list[dict]:
        return self._store.get_conversation_messages(conv_id)

    def add_message(self, conv_id: int, role: str, content: str,
                    sources=None, answer_type=None) -> int:
        return self._store.add_message(conv_id, role, content, sources, answer_type)

    def update_message_content(self, msg_id: int, content: str,
                                sources=None) -> bool:
        return self._store.update_message_content(msg_id, content, sources)

    # ================================================================
    # 异步方法（供 FastAPI 事件循环调用）
    # ================================================================

    async def alist_conversations(self, user_id: int | None = None) -> list[dict]:
        return await self._store.alist_conversations(user_id=user_id)

    async def acreate_conversation(self, title=None, user_id: int | None = None) -> dict:
        return await self._store.acreate_conversation(title, user_id=user_id)

    async def adelete_conversation(self, conv_id: int, user_id: int | None = None) -> bool:
        return await self._store.adelete_conversation(conv_id, user_id=user_id)

    async def aupdate_title(self, conv_id: int, title: str, user_id: int | None = None) -> bool:
        return await self._store.aupdate_conversation_title(conv_id, title, user_id=user_id)

    async def aget_messages(self, conv_id: int, user_id: int | None = None) -> list[dict]:
        return await self._store.aget_conversation_messages(conv_id, user_id=user_id)

    async def aadd_message(self, conv_id: int, role: str, content: str,
                           sources=None, answer_type=None) -> int:
        return await self._store.aadd_message(conv_id, role, content, sources, answer_type)

    async def aupdate_message_content(self, msg_id: int, content: str,
                                      sources=None) -> bool:
        return await self._store.aupdate_message_content(msg_id, content, sources)

    async def aget_user_id(self, username: str) -> int | None:
        """按用户名查询用户 ID（用于对话归属校验）。"""
        try:
            user = await self._store.aget_user_by_username(username)
            return user["id"] if user else None
        except Exception:
            return None
