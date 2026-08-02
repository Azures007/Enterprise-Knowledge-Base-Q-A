"""
=============================================================================
向量数据库管理模块

基于 ChromaDB 实现向量存储与相似度检索。提供文档向量的批量入库、
持久化存储和近似最近邻搜索功能。

ChromaDB 是一个轻量级、嵌入式的向量数据库，支持：
    - 持久化存储到本地文件系统
    - 多种距离度量（余弦相似度、L2 等）
    - 元数据过滤
    - 集合（Collection）管理
=============================================================================

使用方法:
    from src.vector_store import VectorStoreManager
    from src.embeddings import BailianEmbeddings

    embedder = BailianEmbeddings()
    store = VectorStoreManager(embedder)

    # 添加文档
    store.add_documents(docs)

    # 检索
    results = store.similarity_search("公司考勤制度", k=5)
"""

import json
import uuid
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.errors import NotFoundError as ChromaNotFoundError

from config.settings import settings
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class VectorStoreError(Exception):
    """向量数据库操作异常"""
    pass


class VectorStoreManager:
    """
    向量数据库管理器，封装 ChromaDB 的常见操作。
    """

    def __init__(
        self,
        embedder: Any,
        collection_name: str | None = None,
        persist_dir: str | Path | None = None,
    ):
        """
        Args:
            embedder:        嵌入模型实例（必须实现 embed_documents 和 embed_query 方法）
            collection_name: 集合名称
            persist_dir:     ChromaDB 持久化目录（默认从全局配置读取）
        """
        self.embedder = embedder
        self.collection_name = collection_name or settings.DEFAULT_COLLECTION
        self.persist_dir = Path(persist_dir or settings.VECTOR_DB_PATH)

        # 确保持久化目录存在
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        # 初始化 ChromaDB 客户端
        self._client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=False,
            ),
        )

        # 获取或创建集合
        self._collection = self._get_or_create_collection()

        logger.info(
            f"向量数据库初始化完成: collection={collection_name}, "
            f"persist_dir={self.persist_dir}"
        )

    # ================================================================
    # 文档管理
    # ================================================================

    def add_documents(
        self,
        documents: list[dict[str, Any]],
        batch_size: int = 64,
    ) -> int:
        """
        向向量库中添加文档块。

        Args:
            documents: 文档块列表，每项包含 'page_content' 和 'metadata'
            batch_size: 每批处理的文档数

        Returns:
            成功添加的文档块数量
        """
        if not documents:
            return 0

        texts = [doc["page_content"] for doc in documents]
        metadatas = [doc.get("metadata", {}) for doc in documents]

        # 生成唯一 ID
        ids = [str(uuid.uuid4()) for _ in documents]

        # 计算嵌入向量
        logger.info(f"正在计算 {len(texts)} 个文档块的嵌入向量...")
        try:
            embeddings = self.embedder.embed_documents(texts)
        except Exception as e:
            raise VectorStoreError(f"嵌入计算失败: {e}") from e

        if len(embeddings) != len(texts):
            raise VectorStoreError(
                f"嵌入向量数量不匹配: 期望 {len(texts)}，实际 {len(embeddings)}"
            )

        # 分批写入 ChromaDB
        added_count = 0
        for i in range(0, len(texts), batch_size):
            batch_end = min(i + batch_size, len(texts))

            batch_texts = texts[i:batch_end]
            batch_metadatas = metadatas[i:batch_end]
            batch_embeddings = embeddings[i:batch_end]
            batch_ids = ids[i:batch_end]

            # 清理元数据中 ChromaDB 不支持的字段
            cleaned_metadatas = []
            for meta in batch_metadatas:
                cleaned = {}
                for k, v in meta.items():
                    # ChromaDB 元数据只支持 str, int, float, bool
                    if isinstance(v, (str, int, float, bool)):
                        cleaned[k] = v
                    elif isinstance(v, list):
                        cleaned[k] = json.dumps(v, ensure_ascii=False)[:200]
                    else:
                        cleaned[k] = str(v)[:200]
                cleaned_metadatas.append(cleaned)

            self._collection.add(
                ids=batch_ids,
                embeddings=batch_embeddings,
                documents=batch_texts,
                metadatas=cleaned_metadatas,
            )

            added_count += len(batch_texts)
            logger.debug(f"已添加 {added_count}/{len(texts)} 个文档块")

        logger.info(f"文档入库完成: 共 {added_count} 个文档块")
        return added_count

    def similarity_search(
        self,
        query: str,
        k: int | None = None,
        filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        语义相似度检索。

        Args:
            query:  查询文本
            k:      返回结果数量（默认从全局配置读取）
            filter: 元数据过滤条件（ChromaDB where 语法）

        Returns:
            检索结果列表，每项包含:
                - content:  文档内容
                - metadata: 文档元数据
                - score:    相似度分数
        """
        k = k or settings.RETRIEVAL_TOP_K

        try:
            # 计算查询向量
            query_embedding = self.embedder.embed_query(query)
        except Exception as e:
            raise VectorStoreError(f"查询嵌入计算失败: {e}") from e

        # 执行检索
        kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": k,
            "include": ["documents", "metadatas", "distances"],
        }
        if filter:
            kwargs["where"] = filter

        try:
            results = self._collection.query(**kwargs)
        except Exception as e:
            raise VectorStoreError(f"向量检索失败: {e}") from e

        # 结果解析
        formatted_results: list[dict[str, Any]] = []
        if not results["ids"][0]:
            return formatted_results

        for idx in range(len(results["ids"][0])):
            distance = results["distances"][0][idx] if results["distances"] else 0.0
            # ChromaDB 使用余弦距离 (1 - cosine_similarity)，范围 [0, 2]
            # 转换为直观的相似度分数 (0~1)，1 表示最相似
            cosine_sim = max(0, 1.0 - distance)
            score = cosine_sim

            formatted_results.append({
                "content": results["documents"][0][idx],
                "metadata": results["metadatas"][0][idx] if results["metadatas"] else {},
                "score": round(score, 4),
                "distance": round(distance, 4),
            })

        logger.debug(
            f"向量检索完成: query='{query[:50]}...', "
            f"k={k}, 结果数={len(formatted_results)}"
        )
        return formatted_results

    def get_all_chunks(
        self,
        limit: int = 200,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """
        获取集合中所有文档块及其内容（不需要向量检索）。

        用于前端展示知识库内容。

        Args:
            limit:  最大返回条数
            offset: 跳过前 offset 条

        Returns:
            list[dict]: 每项包含 id, content, metadata
        """
        try:
            results = self._collection.get(
                limit=limit,
                offset=offset,
                include=["documents", "metadatas"],
            )
        except Exception as e:
            raise VectorStoreError(f"获取文档块失败: {e}") from e

        formatted: list[dict[str, Any]] = []
        if not results["ids"]:
            return formatted

        for idx in range(len(results["ids"])):
            formatted.append({
                "id": results["ids"][idx],
                "content": results["documents"][idx] if results["documents"] else "",
                "metadata": results["metadatas"][idx] if results["metadatas"] else {},
            })

        logger.debug(f"获取文档块: {len(formatted)} 条 (limit={limit}, offset={offset})")
        return formatted

    def count(self) -> int:
        """返回向量库中文档块总数"""
        return self._collection.count()

    def delete_collection(self):
        """删除当前集合"""
        try:
            self._client.delete_collection(self.collection_name)
            self._collection = self._get_or_create_collection()
            logger.info(f"集合 '{self.collection_name}' 已重置")
        except Exception as e:
            raise VectorStoreError(f"删除集合失败: {e}") from e

    def list_collections(self) -> list[str]:
        """列出所有集合名称"""
        return [c.name for c in self._client.list_collections()]

    def switch_collection(self, collection_name: str) -> None:
        """
        切换到指定集合，后续操作在新集合上执行。

        Args:
            collection_name: 目标集合名称
        """
        self.collection_name = collection_name
        self._collection = self._get_or_create_collection()
        logger.info(f"已切换到集合 '{collection_name}' (文档数: {self._collection.count()})")

    # ================================================================
    # 内部方法
    # ================================================================

    def _get_or_create_collection(self) -> Any:
        """获取已有集合或创建新集合"""
        try:
            collection = self._client.get_collection(self.collection_name)
            logger.info(
                f"获取已有集合 '{self.collection_name}', "
                f"当前文档数: {collection.count()}"
            )
            return collection
        except (ValueError, ChromaNotFoundError):
            # 集合不存在，创建
            collection = self._client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(f"创建新集合 '{self.collection_name}'")
            return collection
