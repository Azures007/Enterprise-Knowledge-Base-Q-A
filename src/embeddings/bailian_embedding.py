"""
=============================================================================
阿里云百炼文本嵌入模块

调用阿里云百炼的文本嵌入模型（text-embedding-v3）将文本转换为向量，
支持批量处理、重试机制和错误处理。

API 文档:
    https://help.aliyun.com/zh/model-studio/developer-reference/text-embedding
=============================================================================

使用方法:
    from src.embeddings import BailianEmbeddings

    embedder = BailianEmbeddings()
    vectors = embedder.embed_documents(["文本1", "文本2", "文本3"])
    vector = embedder.embed_query("查询文本")
"""

import time
from typing import Any

import httpx

from config.settings import settings
from src.monitoring import record_embedding_failed
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# 单次请求最大文本条数（阿里云百炼 text-embedding-v3 限制 batch ≤ 10）
EMBEDDING_BATCH_SIZE = 10


class BailianEmbeddingsError(Exception):
    """嵌入计算过程中的异常"""
    pass


class BailianEmbeddings:
    """
    阿里云百炼文本嵌入模型封装。

    使用 OpenAI 兼容接口调用 text-embedding-v3 模型。
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """
        Args:
            api_key:     百炼 API 密钥（默认从全局配置读取）
            model:       嵌入模型名称（默认从全局配置读取）
            base_url:    API 基础地址（默认从全局配置读取）
            max_retries: 失败重试次数
            retry_delay: 重试间隔（秒）
        """
        self.api_key = api_key or settings.BAILIAN_API_KEY
        self.model = model or settings.EMBEDDING_MODEL_NAME
        self.base_url = base_url or settings.BAILIAN_API_BASE
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        self._dimension: int | None = None

        # 构造嵌入 API 端点
        self._embed_url = f"{self.base_url.rstrip('/')}/embeddings"

        logger.info(
            f"嵌入模型初始化: model={self.model}, base_url={self.base_url}"
        )

    # ================================================================
    # 公开接口
    # ================================================================

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        批量计算文本嵌入向量。

        Args:
            texts: 文本列表

        Returns:
            向量列表，每个向量为 float 列表

        Raises:
            BailianEmbeddingsError: API 调用失败
        """
        if not texts:
            return []

        # 去重空白文本
        valid_texts = [t for t in texts if t and t.strip()]
        if not valid_texts:
            return [[] for _ in texts]

        # 批量处理（百炼 API 限制每次最多 10 条）
        batch_size = EMBEDDING_BATCH_SIZE
        all_embeddings: list[list[float]] = []

        for i in range(0, len(valid_texts), batch_size):
            batch = valid_texts[i : i + batch_size]
            logger.debug(f"嵌入批处理: {i + 1}~{i + len(batch)}/{len(valid_texts)}")

            embeddings = self._call_api(batch)
            all_embeddings.extend(embeddings)

        return all_embeddings

    def embed_query(self, text: str) -> list[float]:
        """
        计算单条查询文本的嵌入向量。

        Args:
            text: 查询文本

        Returns:
            向量（float 列表）
        """
        if not text or not text.strip():
            raise BailianEmbeddingsError("查询文本不能为空")

        result = self._call_api([text])
        if not result:
            raise BailianEmbeddingsError("嵌入计算返回空结果")

        return result[0]

    # ================================================================
    # 异步接口（用于 FastAPI 事件循环内调用，不阻塞其他请求）
    # ================================================================

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """异步批量计算文本嵌入向量（接口语义同 embed_documents）。"""
        if not texts:
            return []

        valid_texts = [t for t in texts if t and t.strip()]
        if not valid_texts:
            return [[] for _ in texts]

        batch_size = EMBEDDING_BATCH_SIZE
        all_embeddings: list[list[float]] = []

        for i in range(0, len(valid_texts), batch_size):
            batch = valid_texts[i : i + batch_size]
            logger.debug(f"嵌入批处理(异步): {i + 1}~{i + len(batch)}/{len(valid_texts)}")
            embeddings = await self._acall_api(batch)
            all_embeddings.extend(embeddings)

        return all_embeddings

    async def aembed_query(self, text: str) -> list[float]:
        """异步计算单条查询文本的嵌入向量（接口语义同 embed_query）。"""
        if not text or not text.strip():
            raise BailianEmbeddingsError("查询文本不能为空")

        result = await self._acall_api([text])
        if not result:
            raise BailianEmbeddingsError("嵌入计算返回空结果")

        return result[0]

    @property
    def dimension(self) -> int:
        """返回 embedding 向量的维度"""
        if self._dimension is None:
            # 调用一次获取维度信息
            test_vector = self.embed_query("测试")
            self._dimension = len(test_vector)
        return self._dimension

    # ================================================================
    # 内部方法
    # ================================================================

    def _call_api(self, texts: list[str]) -> list[list[float]]:
        """
        调用百炼嵌入 API。

        Args:
            texts: 待嵌入的文本列表

        Returns:
            嵌入向量列表
        """
        last_error: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                payload = {
                    "model": self.model,
                    "input": texts,
                    "encoding_format": "float",
                }

                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                }

                response = httpx.post(
                    self._embed_url,
                    json=payload,
                    headers=headers,
                    timeout=60.0,
                )
                response.raise_for_status()
                result = response.json()

                # 解析返回数据
                data = result.get("data", [])
                # 按 index 排序确保顺序一致
                data.sort(key=lambda x: x.get("index", 0))
                embeddings = [item["embedding"] for item in data]

                if self._dimension is None and embeddings:
                    self._dimension = len(embeddings[0])

                logger.debug(
                    f"嵌入 API 调用成功: {len(texts)} 条文本, "
                    f"维度={self._dimension}"
                )
                return embeddings

            except httpx.HTTPStatusError as e:
                last_error = e
                status = e.response.status_code
                body = e.response.text
                logger.warning(
                    f"嵌入 API HTTP 错误 (尝试 {attempt}/{self.max_retries}): "
                    f"status={status}, body={body[:200]}"
                )
                if status in (400, 401, 403):
                    # 客户端错误，无需重试
                    raise BailianEmbeddingsError(
                        f"API 请求被拒绝: status={status}, {body[:200]}"
                    ) from e

            except httpx.TimeoutException as e:
                last_error = e
                logger.warning(
                    f"嵌入 API 超时 (尝试 {attempt}/{self.max_retries})"
                )

            except Exception as e:
                last_error = e
                logger.warning(
                    f"嵌入 API 未知错误 (尝试 {attempt}/{self.max_retries}): {e}"
                )

            # 重试前等待
            if attempt < self.max_retries:
                time.sleep(self.retry_delay * attempt)

        record_embedding_failed(f"已达最大重试次数: {last_error}")
        raise BailianEmbeddingsError(
            f"嵌入 API 调用失败（已达最大重试次数 {self.max_retries}）: {last_error}"
        )

    async def _acall_api(self, texts: list[str]) -> list[list[float]]:
        """
        异步调用百炼嵌入 API（与 _call_api 相同语义，使用 httpx.AsyncClient）。

        Args:
            texts: 待嵌入的文本列表

        Returns:
            嵌入向量列表
        """
        last_error: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                payload = {
                    "model": self.model,
                    "input": texts,
                    "encoding_format": "float",
                }

                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                }

                async with httpx.AsyncClient(timeout=60.0) as client:
                    response = await client.post(
                        self._embed_url,
                        json=payload,
                        headers=headers,
                    )
                response.raise_for_status()
                result = response.json()

                data = result.get("data", [])
                data.sort(key=lambda x: x.get("index", 0))
                embeddings = [item["embedding"] for item in data]

                if self._dimension is None and embeddings:
                    self._dimension = len(embeddings[0])

                logger.debug(
                    f"嵌入 API 异步调用成功: {len(texts)} 条文本, "
                    f"维度={self._dimension}"
                )
                return embeddings

            except httpx.HTTPStatusError as e:
                last_error = e
                status = e.response.status_code
                body = e.response.text
                logger.warning(
                    f"嵌入 API HTTP 错误(异步) (尝试 {attempt}/{self.max_retries}): "
                    f"status={status}, body={body[:200]}"
                )
                if status in (400, 401, 403):
                    raise BailianEmbeddingsError(
                        f"API 请求被拒绝: status={status}, {body[:200]}"
                    ) from e

            except httpx.TimeoutException as e:
                last_error = e
                logger.warning(
                    f"嵌入 API 超时(异步) (尝试 {attempt}/{self.max_retries})"
                )

            except Exception as e:
                last_error = e
                logger.warning(
                    f"嵌入 API 未知错误(异步) (尝试 {attempt}/{self.max_retries}): {e}"
                )

            # 重试前等待
            if attempt < self.max_retries:
                await self._asleep(self.retry_delay * attempt)

        record_embedding_failed(f"异步已达最大重试次数: {last_error}")
        raise BailianEmbeddingsError(
            f"嵌入 API 异步调用失败（已达最大重试次数 {self.max_retries}）: {last_error}"
        )

    @staticmethod
    async def _asleep(seconds: float):
        """异步等待（供异步重试逻辑使用）"""
        import asyncio
        await asyncio.sleep(seconds)
