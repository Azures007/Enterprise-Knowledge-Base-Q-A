"""
=============================================================================
文本分块处理模块

将大段文本按指定大小切割成相互重叠的小块，确保在检索时能够准确定位
到相关信息。支持按字符数和 Token 数两种切割方式。

分块策略：
    1. 优先在段落边界（\\n\\n）处切割
    2. 其次在句子边界（。！？）处切割
    3. 最后按字符数硬切割
    4. 相邻块之间保持重叠以维持上下文连贯
=============================================================================

使用方法:
    from src.text_processor import TextChunker

    chunker = TextChunker(chunk_size=500, chunk_overlap=100)
    chunks = chunker.split_documents(documents)
"""

import re
from typing import Any

from config.settings import settings
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class TextChunker:
    """
    文本分块器，将文档列表中的每个文档按配置切割为若干小块。
    """

    # 中英文句子结束符
    SENTENCE_END_PATTERN = re.compile(r"(?<=[。！？.!?\n])")

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ):
        """
        Args:
            chunk_size:    每个块的目标字符数（默认从全局配置读取）
            chunk_overlap: 相邻块重叠字符数（默认从全局配置读取）
        """
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(f"重叠大小({self.chunk_overlap}) 不能大于等于 块大小({self.chunk_size})")

        logger.info(
            f"文本分块器初始化: chunk_size={self.chunk_size}, "
            f"chunk_overlap={self.chunk_overlap}"
        )

    def split_documents(
        self,
        documents: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        将文档列表切分为小块。

        Args:
            documents: DocumentLoader 输出的文档列表

        Returns:
            切分后的文档块列表，每个块包含 'page_content' 和 'metadata'
            （metadata 中会添加 chunk_index 和 chunk_total 信息）
        """
        all_chunks: list[dict[str, Any]] = []

        for doc in documents:
            text = doc["page_content"]
            metadata = dict(doc["metadata"])  # 拷贝元数据

            chunks = self._split_text(text)

            for i, chunk_text in enumerate(chunks):
                chunk_meta = dict(metadata)
                chunk_meta["chunk_index"] = i
                chunk_meta["chunk_total"] = len(chunks)
                chunk_meta["chunk_size"] = len(chunk_text)

                all_chunks.append({
                    "page_content": chunk_text,
                    "metadata": chunk_meta,
                })

        logger.info(
            f"文本分块完成: {len(documents)} 个文档 → {len(all_chunks)} 个块"
        )
        return all_chunks

    def split_text(self, text: str) -> list[str]:
        """
        将单段文本切分为小块（公开接口）。

        Args:
            text: 原始文本

        Returns:
            文本块列表
        """
        return self._split_text(text)

    # ================================================================
    # 内部实现
    # ================================================================

    def _split_text(self, text: str) -> list[str]:
        """
        核心分块逻辑。

        策略:
            1. 先按段落分割
            2. 再组合成不超过 chunk_size 的块
            3. 相邻块之间保留 overlap 字符的重叠
        """
        # 去除首尾空白
        text = text.strip()
        if not text:
            return []

        # 按段落拆分（连续换行分隔）
        paragraphs = re.split(r"\n\s*\n", text)
        # 过滤空段落
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        chunks: list[str] = []
        current_chunk = ""

        for para in paragraphs:
            # 如果单个段落超过 chunk_size，需要进一步切分
            if len(para) > self.chunk_size:
                # 先把当前积累的块加入结果
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""

                # 将长段落进一步切分
                sub_chunks = self._split_long_text(para)
                chunks.extend(sub_chunks)
                continue

            # 如果加上当前段落不超过限制，则追加
            if not current_chunk:
                current_chunk = para
            elif len(current_chunk) + 1 + len(para) <= self.chunk_size:
                current_chunk += "\n\n" + para
            else:
                # 保存当前块，开始新块（带上重叠部分）
                chunks.append(current_chunk.strip())
                current_chunk = self._get_overlap(current_chunk) + "\n\n" + para

        # 处理最后一个块
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        # 过滤过短的块（可能只剩重叠部分）
        chunks = [c for c in chunks if len(c) > 10]

        return chunks

    def _split_long_text(self, text: str) -> list[str]:
        """
        将超长文本按句子边界切分为多个小块。
        """
        # 按句子分割
        sentences = self.SENTENCE_END_PATTERN.split(text)
        sentences = [s.strip() for s in sentences if s.strip()]

        chunks: list[str] = []
        current = ""

        for sentence in sentences:
            if not current:
                current = sentence
            elif len(current) + len(sentence) <= self.chunk_size:
                current += sentence
            else:
                chunks.append(current.strip())
                current = self._get_overlap(current) + sentence

        if current.strip():
            chunks.append(current.strip())

        return chunks

    def _get_overlap(self, text: str) -> str:
        """
        从文本尾部截取重叠部分。

        Args:
            text: 原始文本

        Returns:
            尾部 overlap_size 字符的文本（如有）
        """
        if len(text) <= self.chunk_overlap:
            return text
        # 尽量从句子边界开始截取
        overlap_text = text[-self.chunk_overlap:]
        # 查找重叠部分中第一个句子边界
        match = re.search(r"[。！？.!?\n]", overlap_text)
        if match:
            return overlap_text[match.end():]
        return overlap_text
