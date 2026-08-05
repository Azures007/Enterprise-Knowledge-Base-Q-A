"""
=============================================================================
文本分块处理模块

将大段文本按指定大小切割成相互重叠的小块，确保在检索时能够准确定位
到相关信息。

分块策略：
    1. Markdown 文档 → 标题感知的语义分块：
       - 解析 ATX 标题（# ~ ######），按标题层级构建"章节单元"
       - 每个块以章节标题链为前缀（如「章节：第二章 薪酬结构 > 2.1 基本工资」）
       - 块内保持主题完整、主题之间隔离，检索命中时自带章节上下文
       - Markdown 表格作为不可分割的原子单元：整表进、整表出，
         不拆散行，避免丢失表头列含义
    2. 普通文本 / 无标题 Markdown → 字符分块：
       - 优先在段落边界（\\n\\n）处切割
       - 其次在句子边界（。！？）处切割
       - 最后按字符数硬切割
       - 相邻块之间保持重叠以维持上下文连贯
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

    自动识别 Markdown 文档并启用标题感知分块；其余文档使用字符分块。
    """

    # 中英文句子结束符
    SENTENCE_END_PATTERN = re.compile(r"(?<=[。！？.!?\n])")

    # Markdown ATX 标题（# ~ ######）
    MARKDOWN_HEADING_RE = re.compile(r"^\s*(#{1,6})\s+(.*)$")

    # Markdown 表格行（GFM，以 | 开头结尾）
    MARKDOWN_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")

    # 表格内部行连接占位符（替换换行，避免表格被外部切分逻辑拆开）
    # 用 ASCII 分隔符而非换行：使得表格行在 _split_text 的段落/句子切分中
    # 被视为一个整体；最终输出前再还原为换行
    _TABLE_SEP = "\x01"

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

        Markdown 文档自动启用标题感知分块，其余文档使用字符分块。

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

            if self._is_markdown_doc(metadata):
                chunks = self._split_markdown(text)
            else:
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
        将单段文本切分为小块（公开接口，纯字符分块）。

        Args:
            text: 原始文本

        Returns:
            文本块列表
        """
        return self._split_text(text)

    # ================================================================
    # Markdown 语义分块
    # ================================================================

    def _is_markdown_doc(self, metadata: dict) -> bool:
        """根据文档元数据判断是否为 Markdown 文档。"""
        file_type = (metadata.get("file_type") or "").lower()
        filename = (metadata.get("filename") or "").lower()
        return file_type == "md" or filename.endswith(".md")

    def _split_markdown(self, text: str) -> list[str]:
        """
        Markdown 标题感知分块。

        流程:
            1. 解析 ATX 标题，按层级把正文归入最近的章节单元
            2. 每个单元带完整的标题链前缀（如「章节：A > B」）
            3. 单元 ≤ chunk_size → 与相邻单元合并成块
            4. 单元 > chunk_size → 保留标题链前缀，内部按段落/句子细分
        """
        sections = self._parse_markdown_sections(text)
        chunks = self._build_section_chunks(sections)
        return [c for c in chunks if len(c) > 10]

    def _parse_markdown_sections(self, text: str) -> list[tuple[str, str]]:
        """
        解析 Markdown 标题并构建章节单元。

        Markdown 表格的连续行会被合并为一个原子单元（行间用占位符连接），
        使其在后续分块中不被拆开。

        Returns:
            list[(title_chain, body)]:
                - title_chain: 从根到该章节的标题路径，如 "第二章 薪酬结构 > 2.1 基本工资"
                - body:        该章节下的正文（不含标题）
            文档开头无标题的内容作为导言（title_chain 为空）。
        """
        stack: list[tuple[int, str]] = []  # 每层标题 (层级, 文本)
        sections: list[tuple[str, str]] = []
        current_body: list[str] = []
        # 表格块缓冲：连续表格行先合并，避免被当作独立行切分
        table_buffer: list[str] = []

        def flush_table():
            """将缓冲的表格行合并为一个原子单元写入正文"""
            if table_buffer:
                current_body.append(self._TABLE_SEP.join(table_buffer))
                table_buffer.clear()

        def flush():
            """将当前章节写入 sections（若有内容）"""
            flush_table()
            chain = " > ".join(t for _, t in stack)
            body = "\n".join(current_body).strip()
            if chain or body:
                sections.append((chain, body))
            current_body.clear()

        for line in text.split("\n"):
            m = self.MARKDOWN_HEADING_RE.match(line)
            if m:
                flush()  # 标题出现，收尾当前章节
                level = len(m.group(1))
                title = m.group(2).strip()
                # 弹出层级不低于当前标题的祖先（同级或更深 → 新章节）
                while stack and stack[-1][0] >= level:
                    stack.pop()
                stack.append((level, title))
            elif self.MARKDOWN_TABLE_ROW_RE.match(line):
                # 表格行：加入表格缓冲（连续行聚合）
                table_buffer.append(line.strip())
            else:
                # 普通行：先冲刷表格缓冲，再积累
                flush_table()
                current_body.append(line)

        flush()  # 收尾最后一个章节

        return sections

    def _build_section_chunks(self, sections: list[tuple[str, str]]) -> list[str]:
        """
        将章节单元组装成分块。

        - 小章节与相邻章节合并（不超过 chunk_size）
        - 超长章节保留标题链前缀，内部再细分
        """
        chunks: list[str] = []
        current = ""

        for chain, body in sections:
            section_text = self._format_section(chain, body)
            if not section_text:
                continue

            if len(section_text) <= self.chunk_size:
                # 章节不超长：尝试并入当前块
                if not current:
                    current = section_text
                elif len(current) + 1 + len(section_text) <= self.chunk_size:
                    current += "\n\n" + section_text
                else:
                    chunks.append(current.strip())
                    current = self._get_overlap(current) + "\n\n" + section_text
            else:
                # 超长章节：单独处理，内部细分并保留标题前缀
                if current:
                    chunks.append(current.strip())
                    current = ""
                chunks.extend(self._split_long_section(chain, body))

        if current.strip():
            chunks.append(current.strip())

        return chunks

    @staticmethod
    def _format_section(chain: str, body: str) -> str:
        """
        生成带标题链前缀的章节文本，并将表格占位符还原为换行。

        例: chain="第二章 薪酬结构 > 2.1 基本工资" body="..." →
            "章节：第二章 薪酬结构 > 2.1 基本工资\n\n..."
        """
        # 还原表格行换行（占位符 → \n）
        body = body.replace(TextChunker._TABLE_SEP, "\n")
        if chain:
            return f"章节：{chain}\n\n{body}" if body else f"章节：{chain}"
        return body

    def _split_markdown_body(self, body: str) -> list[str]:
        """
        将章节正文拆分为段落级单元，表格作为不可分割的原子单元。

        表格行在 _parse_markdown_sections 中已用占位符连成一行，
        这里按普通段落切分时，表格整体是一个"段落"，不会被拆行。
        """
        # 去除首尾空白
        body = body.strip()
        if not body:
            return []
        # 按段落拆分（连续换行分隔）。表格占位符已把行连成整体，
        # 因此表格不会被 \n\n 从中间切开
        return [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]

    def _split_long_section(self, chain: str, body: str) -> list[str]:
        """
        超长章节：复用字符分块切分正文，每块追加标题链前缀。

        表格单元已由占位符连成整体，超长切分时表格行不会被拆开。

        Args:
            chain: 标题链（不含「章节：」前缀）
            body:  章节正文
        """
        prefix = f"章节：{chain}\n\n" if chain else ""
        # 按段落级单元切分（表格整体视为一段）
        paragraphs = self._split_markdown_body(body)
        if not paragraphs:
            return [prefix] if prefix else []

        # 复用 _split_text 的核心组装逻辑，但确保段落不跨表格拆分
        sub_chunks: list[str] = []
        current = ""
        for para in paragraphs:
            if len(para) > self.chunk_size:
                if current:
                    sub_chunks.append(current.strip())
                    current = ""
                sub_chunks.extend(self._split_long_text(para))
                continue
            if not current:
                current = para
            elif len(current) + 1 + len(para) <= self.chunk_size:
                current += "\n\n" + para
            else:
                sub_chunks.append(current.strip())
                current = self._get_overlap(current) + "\n\n" + para
        if current.strip():
            sub_chunks.append(current.strip())

        # 给每个子块加标题前缀并还原表格换行
        result = []
        for s in sub_chunks:
            s = s.replace(self._TABLE_SEP, "\n")
            result.append((prefix + s) if prefix else s)
        return result

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
