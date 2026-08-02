#!/usr/bin/env python3
"""
=============================================================================
数据迁移脚本：从 ChromaDB + SQLite → PostgreSQL + pgvector

使用方法：
    # 先确保 PostgreSQL 已运行且 knowledge_base 数据库已创建
    # 然后执行：
    python scripts/migrate_to_pg.py

    # 迁移指定集合（默认迁移所有）
    python scripts/migrate_to_pg.py --collection knowledge_base
=============================================================================
"""

import argparse
import sys
from pathlib import Path

# 将项目根目录加入路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.embeddings import BailianEmbeddings
from src.vector_store import VectorStoreManager, PGVectorStore
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def migrate_collection(collection_name: str, pg_store: PGVectorStore, chroma_store: VectorStoreManager):
    """将单个集合从 ChromaDB 迁移到 PostgreSQL"""
    print(f"\n{'='*60}")
    print(f"迁移集合: {collection_name}")
    print(f"{'='*60}")

    # 切换到 ChromaDB 中对应的集合
    try:
        chroma_store.switch_collection(collection_name)
    except Exception as e:
        print(f"  ⚠️  无法切换到 ChromaDB 集合 '{collection_name}': {e}")
        return 0

    # 获取所有 chunks
    chunks = chroma_store.get_all_chunks(limit=10000)
    if not chunks:
        print(f"  ℹ️  集合 '{collection_name}' 中没有数据")
        return 0

    # ChromaDB 的 get_all_chunks 返回的是 content 字段，需要映射为 page_content
    normalized = []
    for c in chunks:
        normalized.append({
            "page_content": c.get("content") or c.get("page_content", ""),
            "metadata": c.get("metadata", {}),
        })
    print(f"  📄 ChromaDB 中找到 {len(normalized)} 个文档块")

    # 切换到 PostgreSQL 中对应的集合
    pg_store.switch_collection(collection_name)

    # 导入到 PostgreSQL
    count = pg_store.add_documents(normalized)

    print(f"  ✅ 迁移完成: {count} 个文档块 → PostgreSQL")
    return count


def migrate_conversations(pg_store: PGVectorStore):
    """迁移对话数据"""
    from src.conversations import ConversationManager
    from config.settings import settings

    sqlite_db = settings.PROJECT_ROOT / "data" / "conversations.db"
    if not sqlite_db.exists():
        print("  ℹ️  无对话数据需要迁移")
        return 0

    sqlite_mgr = ConversationManager(str(sqlite_db))
    convs = sqlite_mgr.list_conversations()
    if not convs:
        print("  ℹ️  SQLite 中无对话数据")
        return 0

    print(f"\n{'='*60}")
    print(f"迁移对话数据")
    print(f"{'='*60}")
    print(f"  📄 SQLite 中找到 {len(convs)} 个对话")

    for conv in convs:
        # 创建对话
        new_conv = pg_store.create_conversation(conv["title"])
        pg_conv_id = new_conv["id"]

        # 获取消息
        messages = sqlite_mgr.get_messages(conv["id"])
        for msg in messages:
            pg_store.add_message(
                pg_conv_id, msg["role"], msg["content"],
                msg.get("sources"), msg.get("answer_type"),
            )

        print(f"  ✅ 对话 '{conv['title']}' ({len(messages)} 条消息)")

    return len(convs)


def main():
    parser = argparse.ArgumentParser(description="从 ChromaDB/SQLite 迁移到 PostgreSQL")
    parser.add_argument(
        "--collection", "-c",
        help="指定要迁移的集合名称（默认迁移所有）",
    )
    parser.add_argument(
        "--skip-conversations",
        action="store_true",
        help="跳过对话数据迁移",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  数据迁移工具: ChromaDB + SQLite → PostgreSQL + pgvector")
    print("=" * 60)

    # 初始化
    print("\n📦 初始化组件...")
    embedder = BailianEmbeddings()
    pg_store = PGVectorStore(embedder)
    chroma_store = VectorStoreManager(embedder)

    # 迁移集合
    total_chunks = 0
    if args.collection:
        total_chunks += migrate_collection(args.collection, pg_store, chroma_store)
    else:
        # 迁移所有集合
        collections = chroma_store.list_collections()
        if not collections:
            print("  ℹ️  ChromaDB 中无集合")
        for coll in collections:
            total_chunks += migrate_collection(coll, pg_store, chroma_store)

    # 迁移对话
    total_convs = 0
    if not args.skip_conversations:
        total_convs = migrate_conversations(pg_store)

    print(f"\n{'='*60}")
    print(f"  迁移完成!")
    print(f"  📄 文档块: {total_chunks}")
    print(f"  💬 对话:   {total_convs}")
    print(f"{'='*60}")
    print(f"\n现在可以切换配置使用 PostgreSQL:")
    print(f"  1. 在 .env 中配置 DATABASE_URL")
    print(f"  2. 修改 src/api/routes.py 中的依赖为 PGVectorStore")
    print(f"  3. 重启服务")


if __name__ == "__main__":
    main()
