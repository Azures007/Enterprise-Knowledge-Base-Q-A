"""
=============================================================================
RAG 检索增强生成管线（混合模式）

采用「通用知识优先，知识库补充」的混合问答策略：

    流程:
    1. 接收用户问题
    2. 检查知识库状态，如有内容则尝试向量检索
    3. 根据检索结果质量判断回答策略：
       - 有高质量匹配 → 基于知识库内容增强回答（RAG）
       - 无匹配或低分匹配 → 使用 LLM 通用知识回答
       - 知识库为空 → 直接使用 LLM 回答
    4. 返回回答并标注来源类型
=============================================================================

使用方法:
    from src.rag import RAGPipeline

    rag = RAGPipeline()
    result = rag.query("法国的首都是什么？")      # 用通用知识
    result = rag.query("公司的考勤制度是什么？")   # 先检索知识库，必要时用通用知识
    print(result["answer"])
    print(result["answer_type"])  # "kb" | "general" | "hybrid"
"""

from typing import Any

from config.settings import settings
from src.embeddings import BailianEmbeddings
from src.llm import BailianLLM
from src.utils.logger import setup_logger
from src.vector_store import VectorStoreManager

logger = setup_logger(__name__)


# 检索分数阈值（余弦相似度 0~1）：低于此值认为检索结果不相关，回退到通用知识
SCORE_THRESHOLD = 0.35

# 问答缓存
from src.cache import qa_cache


class RAGPipelineError(Exception):
    """RAG 管线运行异常"""
    pass


class RAGPipeline:
    """
    RAG 检索增强生成管线（混合模式）。

    支持两种回答模式：
    1. 知识库模式（kb）—— 基于检索到的文档内容回答
    2. 通用知识模式（general）—— 基于 LLM 自身知识回答
    3. 混合模式（hybrid）—— 结合知识库与 LLM 知识
    """

    # ---- 完整模式提示词（默认） ----
    DEFAULT_SYSTEM_PROMPT = """你是一个专业的企业知识库智能助手，同时也具备丰富的通用知识。

## 核心原则
根据问题类型和检索结果，灵活选择回答方式：

### 1️⃣ 如果检索到了相关的知识库文档
优先基于文档内容回答，但**不局限于此**——你可以综合自己的知识给出更完整的回答。
- 引用知识库内容时标注 📚 来源（文件名）
- 用自己的知识补充时标注 💡 补充说明

### 2️⃣ 如果没有检索到相关文档，或文档不相关
完全使用你的通用知识回答，但请以「💡 通用知识」开头。

### 3️⃣ 混合场景
如果知识库有部分相关信息，结合文档和你自己的知识给出完整、全面的回答。

## 规则
- 中文回答
- 专业、清晰、全面
- 不要编造知识库中不存在但声称来自知识库的信息
- 不确定时可以说明

---
📚 参考文档内容：
{context}

❓ 用户问题：{question}"""

    # ---- 简洁模式提示词 ----
    CONCISE_SYSTEM_PROMPT = """你是一个同时拥有企业知识库和通用知识的智能助手。

回答策略：
1. 如果参考文档与问题相关 → 基于文档简洁回答，标注 📚 来源
2. 如果参考文档为空或不相关 → 用你的通用知识回答，标注 💡 通用知识
3. 混合 → 结合两者

参考文档：
{context}

问题：{question}"""

    # ---- 纯通用知识模式（知识库为空时使用） ----
    GENERAL_SYSTEM_PROMPT = """你是一个知识渊博的智能助手。请用中文专业地回答用户的问题。

如果你不确定答案，请如实告知，不要编造信息。

用户问题：{question}"""

    def __init__(
        self,
        embedder: Any | None = None,
        llm: Any | None = None,
        vector_store: Any | None = None,
    ):
        """
        Args:
            embedder:      嵌入模型实例（默认新建）
            llm:           LLM 实例（默认新建）
            vector_store:  向量存储实例（默认新建）
        """
        self.embedder = embedder or BailianEmbeddings()
        self.llm = llm or BailianLLM()
        self.vector_store = vector_store or VectorStoreManager(self.embedder)

        self._total_queries = 0
        self._total_tokens_estimate = 0

        logger.info("RAG 混合管线初始化完成")

    # ================================================================
    # 核心问答接口
    # ================================================================

    def query(
        self,
        question: str,
        k: int | None = None,
        stream: bool = False,
        concise: bool = False,
        filter_criteria: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        执行一次混合问答。

        自动判断使用「知识库检索增强」还是「通用知识」。

        Args:
            question:        用户问题
            k:               检索文档块数量
            stream:          是否使用流式输出
            concise:         是否使用简洁模式
            filter_criteria: 元数据过滤条件

        Returns:
            dict:
                - answer:      LLM 生成的回答
                - sources:     参考来源列表（可能为空）
                - context:     检索到的原始文档块（可能为空）
                - answer_type: "general" | "hybrid" | "kb"
        """
        if not question or not question.strip():
            raise RAGPipelineError("问题不能为空")

        self._total_queries += 1
        logger.info(f"收到问题: {question[:120]}")

        # ---- 第 0 步：检查缓存 ----
        if not stream:
            cached = qa_cache.get(question)
            if cached is not None:
                logger.info(f"缓存命中: {question[:80]}...")
                return {
                    "answer": cached["answer"],
                    "sources": cached.get("sources", []),
                    "context": [],
                    "answer_type": cached.get("answer_type", "general"),
                    "stream": False,
                    "from_cache": True,
                }

        # ---- 第 1 步：检查知识库，尝试检索 ----
        kb_chunk_count = self.vector_store.count()
        retrieved_docs = []
        has_relevant_docs = False

        if kb_chunk_count > 0:
            logger.info(f"知识库中有 {kb_chunk_count} 个文档块，尝试向量检索...")
            try:
                retrieved_docs = self.vector_store.similarity_search(
                    query=question,
                    k=k or settings.RETRIEVAL_TOP_K,
                    filter=filter_criteria,
                )
            except Exception as e:
                logger.warning(f"检索失败，将使用通用知识: {e}")

            if retrieved_docs:
                # 检查检索质量：最高分是否超过阈值
                max_score = max(doc.get("score", 0) for doc in retrieved_docs)
                if max_score >= SCORE_THRESHOLD:
                    has_relevant_docs = True
                    logger.info(
                        f"检索到 {len(retrieved_docs)} 个相关文档块"
                        f"（最高分: {max_score:.3f}）→ 使用知识库增强模式"
                    )
                else:
                    logger.info(
                        f"检索结果分数偏低（最高分: {max_score:.3f}）"
                        f"→ 将结合通用知识回答"
                    )
            else:
                logger.info("检索结果为空，将使用通用知识回答")
        else:
            logger.info("知识库为空，将使用通用知识回答")

        # ---- 第 2 步：组装提示 ----
        sources = []
        context_text = ""

        if has_relevant_docs and retrieved_docs:
            # 有相关文档 → RAG 模式（可结合通用知识）
            context_text = self._format_context(retrieved_docs)
            sources = self._extract_sources(retrieved_docs)
            answer_type = "kb" if len(retrieved_docs) >= 2 else "hybrid"

            prompt_template = (
                self.CONCISE_SYSTEM_PROMPT if concise else self.DEFAULT_SYSTEM_PROMPT
            )
            system_prompt = prompt_template.format(
                context=context_text,
                question=question,
            )
        else:
            # 无相关文档 → 通用知识模式
            answer_type = "general"
            if kb_chunk_count > 0:
                # 有文档但无匹配：提供检索到的文档作为参考（低分但也可能有用）
                if retrieved_docs:
                    context_text = self._format_context(retrieved_docs)
                    sources = self._extract_sources(retrieved_docs)
                    prompt_template = (
                        self.CONCISE_SYSTEM_PROMPT
                        if concise
                        else self.DEFAULT_SYSTEM_PROMPT
                    )
                else:
                    context_text = "（知识库中无相关文档）"
                    prompt_template = (
                        self.CONCISE_SYSTEM_PROMPT
                        if concise
                        else self.DEFAULT_SYSTEM_PROMPT
                    )
                system_prompt = prompt_template.format(
                    context=context_text,
                    question=question,
                )
            else:
                # 知识库完全为空 → 纯通用知识
                system_prompt = self.GENERAL_SYSTEM_PROMPT.format(question=question)

        # 估计 token 消耗
        estimated_tokens = len(system_prompt) + len(question) * 2
        self._total_tokens_estimate += estimated_tokens

        # ---- 第 3 步：LLM 生成 ----
        logger.info(f"正在生成回答（模式: {answer_type}）...")
        try:
            if stream:
                answer_generator = self.llm.stream(
                    prompt=question,
                    system_prompt=system_prompt,
                )
                return {
                    "answer": answer_generator,
                    "sources": sources,
                    "context": retrieved_docs or [],
                    "answer_type": answer_type,
                    "stream": True,
                }
            else:
                answer = self.llm.generate(
                    prompt=question,
                    system_prompt=system_prompt,
                )
                logger.info(
                    f"回答生成完成（模式: {answer_type}, 长度={len(answer)}字）"
                )
                # 保存到缓存（非流式）
                qa_cache.set(question, {
                    "answer": answer,
                    "sources": sources,
                    "answer_type": answer_type,
                })
                return {
                    "answer": answer,
                    "sources": sources,
                    "context": retrieved_docs or [],
                    "answer_type": answer_type,
                    "stream": False,
                    "from_cache": False,
                }

        except Exception as e:
            raise RAGPipelineError(f"回答生成失败: {e}") from e

    def stream_query(
        self,
        question: str,
        k: int | None = None,
        concise: bool = False,
        filter_criteria: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        流式混合问答的快捷方式。
        """
        return self.query(
            question=question,
            k=k,
            stream=True,
            concise=concise,
            filter_criteria=filter_criteria,
        )

    # ================================================================
    # 知识库管理
    # ================================================================

    def add_documents(self, documents: list[dict[str, Any]], document_id: int | None = None) -> int:
        """向知识库中添加文档（自动清除问答缓存）。"""
        result = self.vector_store.add_documents(documents, document_id=document_id)
        if result > 0:
            qa_cache.clear()
            logger.info("知识库已更新，问答缓存已清空")
        return result

    def get_knowledge_base_stats(self) -> dict[str, Any]:
        """获取知识库统计信息"""
        return {
            "total_chunks": self.vector_store.count(),
            "total_queries": self._total_queries,
            "estimated_tokens_used": self._total_tokens_estimate,
            "collections": self.vector_store.list_collections(),
        }

    def get_collection_chunks(
        self,
        limit: int = 500,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """
        获取当前集合中的所有文档块及其内容。

        Args:
            limit:  最大返回条数
            offset: 分页偏移

        Returns:
            list[dict]: 每项包含 id, content, metadata
        """
        return self.vector_store.get_all_chunks(limit=limit, offset=offset)

    # ================================================================
    # 内部方法
    # ================================================================

    @staticmethod
    def _format_context(documents: list[dict[str, Any]]) -> str:
        """将检索到的文档块格式化为提示上下文"""
        sections = []
        for i, doc in enumerate(documents, 1):
            source = doc.get("metadata", {}).get("filename", "未知来源")
            content = doc.get("content", "")
            sections.append(f"[文档 {i}] (来源: {source})\n{content}\n")
        return "\n---\n".join(sections)

    @staticmethod
    def _extract_sources(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """提取并去重来源信息"""
        seen = set()
        sources = []
        for doc in documents:
            meta = doc.get("metadata", {})
            source_key = meta.get("source", "")
            if source_key and source_key not in seen:
                seen.add(source_key)
                sources.append({
                    "filename": meta.get("filename", "未知"),
                    "source": source_key,
                    "page": meta.get("page"),
                    "slide": meta.get("slide"),
                    "score": doc.get("score", 0),
                })
        return sources
