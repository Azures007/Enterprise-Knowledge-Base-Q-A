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
- 引用知识库内容时，在句子末尾标注来源编号 [N]
  （N 对应「参考文档」的序号，如 [1]、[2]，可多个如 [1][2]）
- 用自己的知识补充时标注 💡 补充说明

### 2️⃣ 如果没有检索到相关文档，或文档不相关
完全使用你的通用知识回答，但请以「💡 通用知识」开头。

### 3️⃣ 混合场景
如果知识库有部分相关信息，结合文档和你自己的知识给出完整、全面的回答。

## 引用规则（重要）
- 回答中每引用一个文档的某句话/数据，就在对应句子末尾加上 [N]
- N 必须是「参考文档内容」列表中实际存在的序号，不要编造不存在的序号
- 来源编号 [N] 紧跟在引用的句子后面，格式如：根据公司规定，迟到超过30分钟记旷工半天[1]。
- 如果回答没有引用任何文档，则不要添加任何 [N]

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

    # ---- 知识库为空时的通用知识模式（带上传引导） ----
    GENERAL_EMPTY_KB_PROMPT = """你是一个知识渊博的智能助手。请用中文专业地回答用户的问题。

注意：当前企业的知识库中还没有上传任何文档。请你：
1. 首先基于你自己的通用知识，专业、完整地回答用户的问题；
2. 在回答的末尾，自然地提醒用户：当前知识库为空，如需获得更准确、针对性的回答，
   可以将相关文档上传到知识库后再提问。

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
            vector_store:  向量存储实例（默认新建，根据 settings.VECTOR_STORE_TYPE 自动选择后端）
        """
        self.embedder = embedder or BailianEmbeddings()
        self.llm = llm or BailianLLM()
        self.vector_store = vector_store or self._create_default_vector_store()

        # 重排器（用于检索后精排）
        from src.reranker import Reranker
        self.reranker = Reranker()

        self._total_queries = 0
        self._total_tokens_estimate = 0

        store_type = getattr(settings, 'VECTOR_STORE_TYPE', 'pg')
        logger.info(
            f"RAG 混合管线初始化完成（向量后端: {store_type}, "
            f"重排模式: {self.reranker.mode}）"
        )

    @staticmethod
    def _create_default_vector_store():
        """根据 settings.VECTOR_STORE_TYPE 创建默认向量存储后端"""
        store_type = getattr(settings, 'VECTOR_STORE_TYPE', 'pg').lower()

        if store_type == 'pg':
            from src.vector_store import PGVectorStore
            logger.info("自动选择向量存储后端: PostgreSQL + pgvector")
            try:
                return PGVectorStore(BailianEmbeddings())
            except ImportError as e:
                raise RAGPipelineError(
                    f"使用 pg 向量存储需要安装 PostgreSQL 驱动: {e}\n"
                    f"请执行: pip install psycopg2-binary\n"
                    f"或者在 .env 中设置 VECTOR_STORE_TYPE=chroma 以使用 ChromaDB"
                ) from e
        else:
            logger.info("自动选择向量存储后端: ChromaDB")
            return VectorStoreManager(BailianEmbeddings())

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
        history: list[dict[str, str]] | None = None,
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
            history:         多轮对话历史，格式: [{"role": "user"|"assistant", "content": "..."}]

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

        # ---- 第 0 步：检查缓存（仅无对话历史时使用缓存） ----
        if not stream and not history:
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

        # ---- 第 0.5 步：多轮问题重写（有历史时） ----
        rewritten_question = question
        if history and getattr(settings, "QUERY_REWRITE_ENABLED", True):
            rewritten_question = self._rewrite_question(question, history)
            if rewritten_question and rewritten_question != question:
                logger.info(f"问题重写: '{question[:50]}' → '{rewritten_question[:60]}'")

        # ---- 第 1~2 步：检索 + 组装提示 ----
        prepared = self._prepare_query(
            question=rewritten_question,
            k=k,
            concise=concise,
            filter_criteria=filter_criteria,
        )

        # ---- 第 3 步：LLM 生成 ----
        logger.info(f"正在生成回答（模式: {prepared['answer_type']}）...")
        try:
            if stream:
                answer_generator = self.llm.stream(
                    prompt=question,
                    system_prompt=prepared["system_prompt"],
                    history=history,
                )
                return {
                    "answer": answer_generator,
                    "sources": prepared["sources"],
                    "context": prepared["retrieved_docs"],
                    "answer_type": prepared["answer_type"],
                    "stream": True,
                }
            else:
                answer = self.llm.generate(
                    prompt=question,
                    system_prompt=prepared["system_prompt"],
                    history=history,
                )
                logger.info(
                    f"回答生成完成（模式: {prepared['answer_type']}, 长度={len(answer)}字）"
                )
                # 生成相关问题推荐（基于检索到的知识库文档）
                related = []
                if prepared["retrieved_docs"] and getattr(settings, "RELATED_QUESTIONS_ENABLED", True):
                    related = self._relate_questions(question, answer)
                # 保存到缓存（非流式、无历史），按来源文件打标签以便精准失效
                if not history:
                    cache_tags = [
                        f"doc:{s.get('filename', '')}"
                        for s in prepared["sources"] if s.get("filename")
                    ]
                    qa_cache.set(question, {
                        "answer": answer,
                        "sources": prepared["sources"],
                        "answer_type": prepared["answer_type"],
                    }, tags=cache_tags)
                return {
                    "answer": answer,
                    "sources": prepared["sources"],
                    "context": prepared["retrieved_docs"],
                    "answer_type": prepared["answer_type"],
                    "stream": False,
                    "from_cache": False,
                    "related_questions": related,
                }

        except Exception as e:
            raise RAGPipelineError(f"回答生成失败: {e}") from e

    async def aquery(
        self,
        question: str,
        k: int | None = None,
        stream: bool = False,
        concise: bool = False,
        filter_criteria: dict[str, Any] | None = None,
        history: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        """
        异步执行一次混合问答（用于 FastAPI 事件循环，不阻塞其他请求）。

        接口语义与 query() 完全一致。
        """
        if not question or not question.strip():
            raise RAGPipelineError("问题不能为空")

        self._total_queries += 1
        logger.info(f"收到问题(异步): {question[:120]}")

        # ---- 第 0 步：检查缓存（仅无对话历史时使用缓存） ----
        if not stream and not history:
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

        # ---- 第 0.5 步：多轮问题重写（有历史时，结合上下文把问题改写成独立完整问题） ----
        rewritten_question = question
        if history and getattr(settings, "QUERY_REWRITE_ENABLED", True):
            rewritten_question = await self._arewrite_question(question, history)
            if rewritten_question and rewritten_question != question:
                logger.info(f"问题重写: '{question[:50]}' → '{rewritten_question[:60]}'")

        # ---- 第 1~2 步：检索 + 组装提示（异步） ----
        prepared = await self._aprepare_query(
            question=rewritten_question,
            k=k,
            concise=concise,
            filter_criteria=filter_criteria,
        )

        # ---- 第 3 步：LLM 生成（异步） ----
        logger.info(f"正在生成回答(异步)（模式: {prepared['answer_type']}）...")
        try:
            if stream:
                answer_generator = self.llm.astream(
                    prompt=question,
                    system_prompt=prepared["system_prompt"],
                    history=history,
                )
                return {
                    "answer": answer_generator,
                    "sources": prepared["sources"],
                    "context": prepared["retrieved_docs"],
                    "answer_type": prepared["answer_type"],
                    "stream": True,
                }
            else:
                answer = await self.llm.agenerate(
                    prompt=question,
                    system_prompt=prepared["system_prompt"],
                    history=history,
                )
                logger.info(
                    f"回答生成完成(异步)（模式: {prepared['answer_type']}, 长度={len(answer)}字）"
                )
                # 生成相关问题推荐（基于检索到的知识库文档）
                related = []
                if prepared["retrieved_docs"] and getattr(settings, "RELATED_QUESTIONS_ENABLED", True):
                    related = await self._arelate_questions(question, answer)
                if not history:
                    cache_tags = [
                        f"doc:{s.get('filename', '')}"
                        for s in prepared["sources"] if s.get("filename")
                    ]
                    qa_cache.set(question, {
                        "answer": answer,
                        "sources": prepared["sources"],
                        "answer_type": prepared["answer_type"],
                    }, tags=cache_tags)
                return {
                    "answer": answer,
                    "sources": prepared["sources"],
                    "context": prepared["retrieved_docs"],
                    "answer_type": prepared["answer_type"],
                    "stream": False,
                    "from_cache": False,
                    "related_questions": related,
                }

        except Exception as e:
            raise RAGPipelineError(f"回答生成失败(异步): {e}") from e

    def stream_query(
        self,
        question: str,
        k: int | None = None,
        concise: bool = False,
        filter_criteria: dict[str, Any] | None = None,
        history: list[dict[str, str]] | None = None,
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
            history=history,
        )

    async def astream_query(
        self,
        question: str,
        k: int | None = None,
        concise: bool = False,
        filter_criteria: dict[str, Any] | None = None,
        history: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        """
        异步流式混合问答的快捷方式。
        """
        return await self.aquery(
            question=question,
            k=k,
            stream=True,
            concise=concise,
            filter_criteria=filter_criteria,
            history=history,
        )

    # ================================================================
    # 检索与提示词构建（同步/异步共享逻辑）
    # ================================================================

    def _prepare_query(
        self,
        question: str,
        k: int | None = None,
        concise: bool = False,
        filter_criteria: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        执行向量检索并组装系统提示词（同步路径）。

        Returns:
            dict: 包含 system_prompt, sources, retrieved_docs, answer_type
        """
        kb_chunk_count = self.vector_store.count()
        retrieved_docs = []
        has_relevant_docs = False

        if kb_chunk_count > 0:
            logger.info(f"知识库中有 {kb_chunk_count} 个文档块，尝试向量检索...")
            try:
                # 阶段 1：粗召回（候选放宽）。启用混合检索时用 向量+关键词 融合
                candidate_k = k or settings.RETRIEVAL_CANDIDATE_K
                search_method = getattr(self.vector_store, "hybrid_search", None)
                if search_method is not None and getattr(settings, "HYBRID_SEARCH_ENABLED", True):
                    candidates = self.vector_store.hybrid_search(
                        query=question,
                        k=candidate_k,
                        filter=filter_criteria,
                    )
                else:
                    candidates = self.vector_store.similarity_search(
                        query=question,
                        k=candidate_k,
                        filter=filter_criteria,
                    )
                # 阶段 2：重排精排，取最终 top-k
                final_k = min(k or settings.RETRIEVAL_TOP_K, len(candidates)) if candidates else 0
                retrieved_docs = self.reranker.rerank(
                    query=question,
                    candidates=candidates,
                    top_k=final_k,
                )
            except Exception as e:
                logger.warning(f"检索失败，将使用通用知识: {e}")

            # 阶段 3：过滤低分块与无关文档（方案 1 + 2）
            retrieved_docs, has_relevant_docs = self._filter_relevant_docs(retrieved_docs)

        system_prompt, answer_type, sources, context = self._build_prompt(
            question=question,
            concise=concise,
            retrieved_docs=retrieved_docs,
            has_relevant_docs=has_relevant_docs,
            kb_chunk_count=kb_chunk_count,
        )

        estimated_tokens = len(system_prompt) + len(question) * 2
        self._total_tokens_estimate += estimated_tokens

        return {
            "system_prompt": system_prompt,
            "sources": sources,
            "retrieved_docs": retrieved_docs,
            "answer_type": answer_type,
            "context_text": context,
        }

    async def _aprepare_query(
        self,
        question: str,
        k: int | None = None,
        concise: bool = False,
        filter_criteria: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """执行向量检索并组装系统提示词（异步路径）。"""
        kb_chunk_count = await self.vector_store.acount()
        retrieved_docs = []
        has_relevant_docs = False

        if kb_chunk_count > 0:
            logger.info(f"知识库中有 {kb_chunk_count} 个文档块，尝试向量检索(异步)...")
            try:
                # 阶段 1：粗召回（候选放宽）。启用混合检索时用 向量+关键词 融合
                candidate_k = k or settings.RETRIEVAL_CANDIDATE_K
                search_method = getattr(self.vector_store, "ahybrid_search", None)
                if search_method is not None and getattr(settings, "HYBRID_SEARCH_ENABLED", True):
                    candidates = await self.vector_store.ahybrid_search(
                        query=question,
                        k=candidate_k,
                        filter=filter_criteria,
                    )
                else:
                    candidates = await self.vector_store.asimilarity_search(
                        query=question,
                        k=candidate_k,
                        filter=filter_criteria,
                    )
                # 阶段 2：重排精排，取最终 top-k
                final_k = min(k or settings.RETRIEVAL_TOP_K, len(candidates)) if candidates else 0
                retrieved_docs = self.reranker.rerank(
                    query=question,
                    candidates=candidates,
                    top_k=final_k,
                )
            except Exception as e:
                logger.warning(f"检索失败(异步)，将使用通用知识: {e}")

            # 阶段 3：过滤低分块与无关文档（方案 1 + 2）
            retrieved_docs, has_relevant_docs = self._filter_relevant_docs(retrieved_docs)

        system_prompt, answer_type, sources, context = self._build_prompt(
            question=question,
            concise=concise,
            retrieved_docs=retrieved_docs,
            has_relevant_docs=has_relevant_docs,
            kb_chunk_count=kb_chunk_count,
        )

        estimated_tokens = len(system_prompt) + len(question) * 2
        self._total_tokens_estimate += estimated_tokens

        return {
            "system_prompt": system_prompt,
            "sources": sources,
            "retrieved_docs": retrieved_docs,
            "answer_type": answer_type,
            "context_text": context,
        }

    def _filter_relevant_docs(
        self,
        retrieved_docs: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], bool]:
        """
        过滤低相关文档块（方案 1 + 2）。

        方案 1：单块过滤 —— 剔除 score < SCORE_THRESHOLD 的低分块。
        方案 2：文档级过滤 —— 按文档聚合，仅保留"文档最高分 ≥ 文档级阈值"的
                文档，其余文档整篇剔除（其所有块均不进入 context / sources）。

        Args:
            retrieved_docs: 向量检索返回的候选块列表

        Returns:
            (filtered_docs, has_relevant_docs):
                - filtered_docs:     过滤后的相关块列表
                - has_relevant_docs: 是否存在相关文档（决定是否走 RAG 模式）
        """
        if not retrieved_docs:
            return [], False

        # 方案 1：按块分数过滤，仅保留高分块
        relevant_chunks = [
            doc for doc in retrieved_docs
            if doc.get("score", 0) >= SCORE_THRESHOLD
        ]
        if not relevant_chunks:
            logger.info(
                f"检索到 {len(retrieved_docs)} 个候选块，但均低于阈值 "
                f"{SCORE_THRESHOLD} → 视为无相关文档"
            )
            return [], False

        # 方案 2：按文档聚合，只保留"文档最高分 ≥ 文档级阈值"的文档
        # 单块阈值较低（0.35），弱相关块也可能通过；文档级阈值取更高值，
        # 只有当某文档至少有一个强相关块时才整篇保留，避免弱相关文档混入来源
        doc_key = lambda d: d.get("metadata", {}).get("source") or d.get("metadata", {}).get("filename", "")

        # 计算每个文档的最高分
        doc_max_score: dict[str, float] = {}
        for d in retrieved_docs:
            key = doc_key(d)
            doc_max_score[key] = max(doc_max_score.get(key, 0), d.get("score", 0))

        # 文档级阈值：比单块阈值更严格，过滤弱相关文档
        doc_threshold = min(SCORE_THRESHOLD + 0.1, 0.55)
        relevant_docs = {
            key for key, score in doc_max_score.items()
            if score >= doc_threshold
        }

        if not relevant_docs:
            logger.info(
                f"检索到 {len(relevant_chunks)} 个高分块，但所有文档最高分 "
                f"均低于文档级阈值 {doc_threshold:.2f} → 视为无相关文档"
            )
            return [], False

        # 保留相关文档的所有检索块（按原顺序）
        filtered = [
            d for d in retrieved_docs if doc_key(d) in relevant_docs
        ]

        # 按分数降序排列，保证 context 里最相关的块靠前
        filtered.sort(key=lambda d: d.get("score", 0), reverse=True)

        logger.info(
            f"检索过滤: {len(retrieved_docs)} 个候选块 → "
            f"{len(filtered)} 个块，来自 {len(relevant_docs)} 个相关文档"
            f"（文档级阈值: {doc_threshold:.2f}, 最高分: {filtered[0].get('score', 0):.3f}）"
        )
        return filtered, True

    # ================================================================
    # 多轮问题重写
    # ================================================================

    REWRITE_SYSTEM_PROMPT = """你是一个对话上下文理解助手。根据用户的多轮对话历史，
把最新的用户问题改写成一个【独立的、完整的、无指代】的问题，使其不依赖上下文也能被搜索引擎/知识库理解。

规则：
- 保留原问题的意图与信息
- 把指代词（它、这、那、这个、那里、怎么、如何等）替换成历史中明确提到的具体对象
- 如果问题本身已经完整独立，原样返回
- 只返回改写后的问题本身，不要任何解释、标点包裹或多余文字"""

    async def _arewrite_question(self, question: str, history: list[dict[str, str]]) -> str:
        """用 LLM 把多轮问题重写成独立完整问题（异步）。"""
        try:
            # 构造最近的对话摘要（最多取最近 6 条）
            recent = history[-6:]
            history_text = "\n".join(
                f"{'用户' if m.get('role') == 'user' else '助手'}: {m.get('content', '')}"
                for m in recent
            )
            prompt = (
                f"对话历史：\n{history_text}\n\n"
                f"当前问题：{question}\n\n"
                f"改写后的独立问题："
            )
            rewritten = await self.llm.agenerate(
                prompt=prompt,
                system_prompt=self.REWRITE_SYSTEM_PROMPT,
                max_tokens=100,
                temperature=0.0,
            )
            rewritten = rewritten.strip().strip('"').strip("'").strip()
            return rewritten if rewritten else question
        except Exception as e:
            logger.warning(f"问题重写失败，使用原问题: {e}")
            return question

    def _rewrite_question(self, question: str, history: list[dict[str, str]]) -> str:
        """用 LLM 把多轮问题重写成独立完整问题（同步）。"""
        try:
            recent = history[-6:]
            history_text = "\n".join(
                f"{'用户' if m.get('role') == 'user' else '助手'}: {m.get('content', '')}"
                for m in recent
            )
            prompt = (
                f"对话历史：\n{history_text}\n\n"
                f"当前问题：{question}\n\n"
                f"改写后的独立问题："
            )
            rewritten = self.llm.generate(
                prompt=prompt,
                system_prompt=self.REWRITE_SYSTEM_PROMPT,
                max_tokens=100,
                temperature=0.0,
            )
            rewritten = rewritten.strip().strip('"').strip("'").strip()
            return rewritten if rewritten else question
        except Exception as e:
            logger.warning(f"问题重写失败，使用原问题: {e}")
            return question

    # ================================================================
    # 相关问题推荐
    # ================================================================

    RELATED_SYSTEM_PROMPT = """你是知识库问答助手。根据用户的当前问题和回答内容，
生成 2~3 个相关的后续问题，帮助用户深入探索。

规则：
- 只生成与当前话题强相关、且知识库/文档可能覆盖的问题
- 每个问题独立成行，用数字开头（如：1. xxx）
- 不要生成与当前问题重复的问题
- 只返回问题列表，不要其他文字"""

    async def _arelate_questions(self, question: str, answer: str) -> list[str]:
        """基于当前问题和回答生成相关问题（异步）。"""
        try:
            prompt = (
                f"用户问题：{question}\n\n"
                f"回答内容：{answer[:1500]}\n\n"
                f"相关后续问题："
            )
            resp = await self.llm.agenerate(
                prompt=prompt,
                system_prompt=self.RELATED_SYSTEM_PROMPT,
                max_tokens=200,
                temperature=0.6,
            )
            return self._parse_related(resp)
        except Exception as e:
            logger.warning(f"生成相关问题失败: {e}")
            return []

    def _relate_questions(self, question: str, answer: str) -> list[str]:
        """基于当前问题和回答生成相关问题（同步）。"""
        try:
            prompt = (
                f"用户问题：{question}\n\n"
                f"回答内容：{answer[:1500]}\n\n"
                f"相关后续问题："
            )
            resp = self.llm.generate(
                prompt=prompt,
                system_prompt=self.RELATED_SYSTEM_PROMPT,
                max_tokens=200,
                temperature=0.6,
            )
            return self._parse_related(resp)
        except Exception as e:
            logger.warning(f"生成相关问题失败: {e}")
            return []

    @staticmethod
    def _parse_related(resp: str) -> list[str]:
        """解析 LLM 返回的相关问题列表。"""
        questions = []
        for line in resp.split("\n"):
            line = line.strip()
            if not line:
                continue
            # 去掉 "1. " / "1、" / "1." 前缀
            import re
            cleaned = re.sub(r"^\d+[\.、]\s*", "", line).strip()
            if cleaned and cleaned not in questions:
                questions.append(cleaned)
        return questions[:3]

    def _build_prompt(
        self,
        question: str,
        concise: bool,
        retrieved_docs: list[dict[str, Any]],
        has_relevant_docs: bool,
        kb_chunk_count: int,
    ) -> tuple[str, str, list[dict[str, Any]], str]:
        """
        根据检索结果组装系统提示词。

        Returns:
            (system_prompt, answer_type, sources, context_text)
        """
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
                # 带上传引导的提示词，让 LLM 回答后用自然语言提醒可上传文档
                system_prompt = self.GENERAL_EMPTY_KB_PROMPT.format(question=question)

        return system_prompt, answer_type, sources, context_text

    # ================================================================
    # 知识库管理
    # ================================================================

    def add_documents(self, documents: list[dict[str, Any]], document_id: int | None = None) -> int:
        """
        向知识库中添加文档。

        仅使引用了这些文档来源的问答缓存失效，而非清空全部缓存。
        """
        result = self.vector_store.add_documents(documents, document_id=document_id)
        if result > 0:
            # 按来源文件名精确失效缓存
            affected_tags = []
            for doc in documents:
                filename = doc.get("metadata", {}).get("filename")
                if filename:
                    affected_tags.append(f"doc:{filename}")
            if affected_tags:
                qa_cache.invalidate_by_tags(affected_tags)
                logger.info(f"知识库已更新，已按来源文件失效 {len(affected_tags)} 个标签的缓存")
            else:
                # 文档没有文件名信息时，无法精确定位，只能全量清空（兜底）
                qa_cache.clear()
                logger.info("知识库已更新，问答缓存已清空（无文件名信息）")
        return result

    async def aadd_documents(self, documents: list[dict[str, Any]], document_id: int | None = None) -> int:
        """
        向知识库中添加文档（异步版本）。

        仅使引用了这些文档来源的问答缓存失效，而非清空全部缓存。
        """
        result = await self.vector_store.aadd_documents(documents, document_id=document_id)
        if result > 0:
            affected_tags = []
            for doc in documents:
                filename = doc.get("metadata", {}).get("filename")
                if filename:
                    affected_tags.append(f"doc:{filename}")
            if affected_tags:
                qa_cache.invalidate_by_tags(affected_tags)
                logger.info(f"知识库已更新，已按来源文件失效 {len(affected_tags)} 个标签的缓存")
            else:
                qa_cache.clear()
                logger.info("知识库已更新，问答缓存已清空（无文件名信息）")
        return result

    def get_knowledge_base_stats(self) -> dict[str, Any]:
        """获取知识库统计信息（CLI 使用）"""
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
        获取当前集合中的所有文档块及其内容（CLI 使用）。
        """
        return self.vector_store.get_all_chunks(limit=limit, offset=offset)

    async def aget_knowledge_base_stats(self) -> dict[str, Any]:
        """获取知识库统计信息（异步版本）"""
        return {
            "total_chunks": await self.vector_store.acount(),
            "total_queries": self._total_queries,
            "estimated_tokens_used": self._total_tokens_estimate,
            "collections": await self.vector_store.alist_collections(),
        }

    async def aget_collection_chunks(
        self,
        limit: int = 500,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """
        获取当前集合中的所有文档块及其内容（异步版本）。
        """
        return await self.vector_store.aget_all_chunks(limit=limit, offset=offset)

    # ================================================================
    # 内部方法
    # ================================================================

    @staticmethod
    def _format_context(documents: list[dict[str, Any]]) -> str:
        """将检索到的文档块格式化为提示上下文，带 [N] 编号供回答引用。"""
        sections = []
        for i, doc in enumerate(documents, 1):
            source = doc.get("metadata", {}).get("filename", "未知来源")
            content = doc.get("content", "")
            sections.append(f"[{i}] (来源: {source})\n{content}\n")
        return "\n---\n".join(sections)

    @staticmethod
    def _extract_sources(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        提取并去重来源信息，并按文档聚合被检索到的分块。

        同一文档的多个命中块会合并为一个来源，chunks 中按相似度分数降序。
        返回的每项含:
            filename, source, page, slide, score,
            chunks: [{content, score, chunk_index}, ...]
        """
        # 按 source 分组收集每个文档的命中块
        doc_chunks: dict[str, dict] = {}
        order: list[str] = []

        for doc in documents:
            meta = doc.get("metadata", {})
            source_key = meta.get("source", "")
            if not source_key:
                continue

            if source_key not in doc_chunks:
                doc_chunks[source_key] = {
                    "filename": meta.get("filename", "未知"),
                    "source": source_key,
                    "page": meta.get("page"),
                    "slide": meta.get("slide"),
                    "score": 0.0,
                    "chunks": [],
                }
                order.append(source_key)

            entry = doc_chunks[source_key]
            # 该文档的最高分作为来源分
            entry["score"] = max(entry["score"], doc.get("score", 0))
            # 追加命中块
            entry["chunks"].append({
                "content": doc.get("content", ""),
                "score": doc.get("score", 0),
                "chunk_index": meta.get("chunk_index"),
            })

        # 按文档最高分降序排列来源，并给每个来源分配 [N] 索引（对应回答中的引用标注）
        sources = [doc_chunks[k] for k in order]
        sources.sort(key=lambda s: s["score"], reverse=True)

        # 每个文档内部的块按分数降序
        for idx, s in enumerate(sources, 1):
            s["index"] = idx
            s["chunks"].sort(key=lambda c: c["score"], reverse=True)

        return sources
