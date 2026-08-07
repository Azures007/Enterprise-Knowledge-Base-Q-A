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


def _safe_json_loads(content: str):
    """
    容错解析工具结果 JSON。

    工具结果可能被截断（result_limit），导致 json.loads 失败。
    策略（从易到难）：
        1. 直接 json.loads
        2. 去掉尾部截断标记
        3. 在截断边界处做括号配平补全（补闭合的 ]/}/字符串）
        4. json.JSONDecoder.raw_decode 取头部完整 JSON
    """
    import json

    if not content:
        return None
    # 1. 直接解析
    try:
        return json.loads(content)
    except (ValueError, TypeError):
        pass

    # 去掉截断标记与尾部逗号
    cleaned = content
    for marker in ('..."(已截断)"', '"(已截断)"', '..."', '...'):
        if cleaned.endswith(marker):
            cleaned = cleaned[: -len(marker)].rstrip().rstrip(",")
            break

    # 2. 括号配平补全：从截断点往回，逐字符移除不完整的 token，
    #    直到剩余内容括号配平且可解析
    for cut in range(len(cleaned), 0, -1):
        candidate = cleaned[:cut]
        # 括号配平检查（忽略字符串内的括号，粗略即可）
        stack = []
        in_str = False
        escape = False
        for ch in candidate:
            if in_str:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch in "[{":
                stack.append(ch)
            elif ch in "]}":
                if stack:
                    stack.pop()
        # 栈里只剩未闭合的 [ 和 {，补全闭合即可
        if stack and all(c in "[{" for c in stack):
            suffix = ""
            for c in reversed(stack):
                suffix += "]" if c == "[" else "}"
            try:
                return json.loads(candidate + suffix)
            except (ValueError, TypeError):
                continue
        if not stack:
            # 已配平但仍解析失败（可能截断在值中间），继续往前找
            try:
                return json.loads(candidate)
            except (ValueError, TypeError):
                continue

    # 3. raw_decode 取头部完整 JSON 对象
    try:
        obj, _ = json.JSONDecoder().raw_decode(content.lstrip())
        if isinstance(obj, dict):
            return obj
    except (ValueError, TypeError):
        pass
    return None


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

        # 工具调用执行器（Agentic 检索，开关启用）
        self.tool_executor = None
        if getattr(settings, "TOOL_CALLING_ENABLED", False):
            try:
                from src.tools import ToolExecutor, ToolRegistry
                from src.tools.tools import TOOL_DEFINITIONS, TOOL_HANDLERS

                self.tool_registry = ToolRegistry(TOOL_DEFINITIONS, TOOL_HANDLERS)
                self.tool_executor = ToolExecutor(
                    self.llm,
                    self.tool_registry,
                    max_rounds=getattr(settings, "TOOL_CALLING_MAX_ROUNDS", 4),
                    max_tool_calls=getattr(settings, "TOOL_CALLING_MAX_CALLS", 8),
                    max_tokens_budget=getattr(settings, "TOOL_CALLING_TOKEN_BUDGET", 16000),
                    duplicate_threshold=getattr(settings, "TOOL_CALLING_DUP_THRESHOLD", 2),
                    result_limit=getattr(settings, "TOOL_CALLING_RESULT_LIMIT", 2000),
                    confirm_expires=getattr(settings, "TOOL_CONFIRM_EXPIRES", 300),
                    mutation_tools=tuple(getattr(settings, "TOOL_MUTATION_TOOLS", "").split(",")),
                )
                logger.info("工具调用已启用（Agentic 检索）")
            except Exception as e:
                logger.warning(f"工具调用初始化失败，退回标准检索: {e}")

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
                    "usage": {},  # 缓存命中无 LLM 调用
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
        usage_holder: dict = {}
        try:
            if stream:
                answer_generator = self.llm.stream(
                    prompt=question,
                    system_prompt=prepared["system_prompt"],
                    history=history,
                    usage_cb=lambda u: usage_holder.update(u),
                )
                return {
                    "answer": answer_generator,
                    "sources": prepared["sources"],
                    "context": prepared["retrieved_docs"],
                    "answer_type": prepared["answer_type"],
                    "stream": True,
                    "usage": usage_holder,
                }
            else:
                answer = self.llm.generate(
                    prompt=question,
                    system_prompt=prepared["system_prompt"],
                    history=history,
                    usage_cb=lambda u: usage_holder.update(u),
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
                    "usage": usage_holder,
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
        prefetched_docs: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """
        异步执行一次混合问答（用于 FastAPI 事件循环，不阻塞其他请求）。

        接口语义与 query() 完全一致。

        Args:
            prefetched_docs: 可选的预检索候选块（跨集合检索时由路由层传入，
                管线跳过内部检索直接走重排+过滤）。
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
                    "usage": {},  # 缓存命中无 LLM 调用
                }

        # ---- 第 0.5 步：多轮问题重写（有历史时，结合上下文把问题改写成独立完整问题） ----
        rewritten_question = question
        if history and getattr(settings, "QUERY_REWRITE_ENABLED", True):
            rewritten_question = await self._arewrite_question(question, history)
            if rewritten_question and rewritten_question != question:
                logger.info(f"问题重写: '{question[:50]}' → '{rewritten_question[:60]}'")

        # ---- 第 0.75 步：工具调用模式（Agentic 检索） ----
        # 开关启用且有执行器时走工具模式；跨集合预检索(prefetched_docs)时不走，
        # 避免与路由层的方案 B 检索冲突
        if (
            not stream
            and self.tool_executor is not None
            and not prefetched_docs
            and getattr(settings, "TOOL_CALLING_ENABLED", False)
        ):
            logger.info(f"走工具调用模式: {question[:80]}")
            try:
                tool_result = await self._atool_call_query(
                    question=rewritten_question,
                    k=k,
                    concise=concise,
                    history=history,
                    is_admin=False,
                )
                # 缓存（非流式、无历史）
                if not history:
                    cache_tags = [
                        f"doc:{s.get('filename', '')}"
                        for s in tool_result["sources"] if s.get("filename")
                    ]
                    qa_cache.set(question, {
                        "answer": tool_result["answer"],
                        "sources": tool_result["sources"],
                        "answer_type": tool_result["answer_type"],
                    }, tags=cache_tags)
                return tool_result
            except Exception as e:
                logger.warning(f"工具调用模式失败，退回标准检索: {e}")

        # ---- 第 1~2 步：检索 + 组装提示（异步） ----
        prepared = await self._aprepare_query(
            question=rewritten_question,
            k=k,
            concise=concise,
            filter_criteria=filter_criteria,
            prefetched_docs=prefetched_docs,
        )

        # ---- 第 3 步：LLM 生成（异步） ----
        logger.info(f"正在生成回答(异步)（模式: {prepared['answer_type']}）...")
        usage_holder: dict = {}
        try:
            if stream:
                answer_generator = self.llm.astream(
                    prompt=question,
                    system_prompt=prepared["system_prompt"],
                    history=history,
                    usage_cb=lambda u: usage_holder.update(u),
                )
                return {
                    "answer": answer_generator,
                    "sources": prepared["sources"],
                    "context": prepared["retrieved_docs"],
                    "answer_type": prepared["answer_type"],
                    "stream": True,
                    "usage": usage_holder,
                }
            else:
                answer = await self.llm.agenerate(
                    prompt=question,
                    system_prompt=prepared["system_prompt"],
                    history=history,
                    usage_cb=lambda u: usage_holder.update(u),
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
                    "usage": usage_holder,
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
        prefetched_docs: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """
        异步流式混合问答的快捷方式。

        工具调用模式开启时，工具循环在内部非流式完成，最终回答以流式片段返回。
        """
        # 工具调用模式：流式路径也在内部完成工具循环，再以生成器形式返回回答
        if (
            self.tool_executor is not None
            and not prefetched_docs
            and getattr(settings, "TOOL_CALLING_ENABLED", False)
        ):
            try:
                tool_result = await self._atool_call_query(
                    question=question,
                    k=k,
                    concise=concise,
                    history=history,
                    is_admin=False,
                )
                content = tool_result["answer"]

                async def tool_stream():
                    # 分段产出最终回答（模拟流式；工具循环本身已非流式完成）
                    chunk_size = 20
                    for i in range(0, len(content), chunk_size):
                        yield content[i : i + chunk_size]

                return {
                    "answer": tool_stream(),
                    "sources": tool_result["sources"],
                    "context": [],
                    "answer_type": tool_result["answer_type"],
                    "stream": True,
                    "usage": tool_result.get("usage", {}),
                    "tool_calls": tool_result.get("tool_calls", []),
                }
            except Exception as e:
                logger.warning(f"工具调用模式失败(流式)，退回标准检索: {e}")

        return await self.aquery(
            question=question,
            k=k,
            stream=True,
            concise=concise,
            filter_criteria=filter_criteria,
            history=history,
            prefetched_docs=prefetched_docs,
        )

    # ================================================================
    # Agentic 检索（工具调用模式）
    # ================================================================

    async def _atool_call_query(
        self,
        question: str,
        k: int | None = None,
        concise: bool = False,
        history: list[dict[str, str]] | None = None,
        is_admin: bool = False,
    ) -> dict[str, Any]:
        """
        工具调用模式的问答：让 LLM 自主决定调用哪些工具（列集合、检索知识库、查统计），
        工具结果作为上下文生成带引用的回答。

        Args:
            question:  用户问题
            k:         检索块数量（传给 search_knowledge_base 工具）
            concise:   是否简洁回答
            history:   多轮对话历史
            is_admin:  是否管理员（决定管理工具是否可用）

        Returns:
            dict（与 aquery 返回结构一致）:
                - answer / sources / context / answer_type / usage / tool_calls
        """
        if self.tool_executor is None:
            raise RAGPipelineError("工具调用未启用")

        # 构造初始消息：系统提示 + 历史 + 当前问题
        system_prompt = (
            self.CONCISE_SYSTEM_PROMPT if concise else self.DEFAULT_SYSTEM_PROMPT
        )
        # 工具模式下 context 由工具注入，初始提示不含文档上下文
        tool_system_prompt = (
            "你是一个企业知识库智能助手。你可以调用工具来获取信息："
            "需要查阅知识库文档时调用 search_knowledge_base（并先 list_collections 了解有哪些集合）；"
            "询问时间/计算/知识库统计时调用对应工具。"
            "引用规则：只有当工具返回的内容包含知识库文档（如 search_knowledge_base 的"
            " results 里带 filename/source）时，回答中才需要标注来源编号 [N]，"
            "且 N 必须对应工具返回中实际存在的来源；"
            "如果回答完全基于工具数据（如天气、时间、计算）而未引用任何知识库文档，"
            "则绝对不要在回答中添加任何 [N] 标注。"
        )
        messages: list[dict[str, str]] = [{"role": "system", "content": tool_system_prompt}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": question})

        # 工具上下文：注入当前 RAG 管线与检索参数
        ctx = {
            "rag": self,
            "vector_store": self.vector_store,
            "reranker": self.reranker,
            "default_k": k or settings.RETRIEVAL_TOP_K,
        }

        result = await self.tool_executor.run(
            messages, ctx, tools=None, is_admin=is_admin
        )

        # 汇总工具检索到的来源（从 tool 消息中提取 search_knowledge_base 的结果）
        sources = self._extract_tool_sources(result.get("messages", []))

        answer = result.get("content", "")
        # 兜底清洗：sources 为空（如纯天气/时间回答）时，剥掉回答里的 [N] 引用标记，
        # 避免 LLM 幻觉出不存在的来源编号
        if not sources:
            answer = self._strip_citations(answer)

        return {
            "answer": answer,
            "sources": sources,
            "context": [],  # 工具模式上下文由工具结果承载
            "answer_type": "tool" if sources else "general",
            "stream": False,
            "from_cache": False,
            "usage": result.get("usage", {}),
            "related_questions": [],
            "tool_calls": result.get("tool_calls", []),
        }

    @staticmethod
    def _strip_citations(text: str) -> str:
        """
        剥除回答中的 [N] 引用标记（无来源时兜底）。

        只匹配"独立的、单个数字的引用标记"，即 [1]、[2] 或连续多个 [1][2]。
        不匹配数组（[1,2,3]）、小数（[5.0]）、负号（[-3]）等数值内容，
        避免误删正文里的数学表达。
        """
        import re

        if not text:
            return text
        # 仅匹配括号内为纯数字（无逗号/点/负号）的引用标记，连续多个也一并删除
        return re.sub(r"\[\d+\]", "", text)

    @staticmethod
    def _extract_tool_sources(messages: list[dict]) -> list[dict]:
        """
        从工具循环的完整消息序列中提取检索来源。

        优先读取工具结果里的精简 sources 摘要字段（不受截断影响）；
        摘要缺失时回退到解析完整 results。
        """
        sources: dict[str, dict] = {}
        order: list[str] = []

        for msg in messages:
            if msg.get("role") != "tool":
                continue
            content = msg.get("content", "")
            data = _safe_json_loads(content)
            if not isinstance(data, dict):
                continue
            # 优先使用精简摘要（search_knowledge_base 返回）
            src_list = data.get("sources") if isinstance(data.get("sources"), list) else None
            # 回退：从 results 构建
            if not src_list and isinstance(data.get("results"), list):
                src_list = []
                for r in data["results"]:
                    src_list.append({
                        "filename": r.get("filename", ""),
                        "source": r.get("source") or r.get("filename", ""),
                        "score": r.get("score", 0),
                        "page": r.get("page"),
                        "collection": r.get("collection"),
                    })
            if not src_list:
                continue
            for r in src_list:
                key = r.get("source") or r.get("filename") or ""
                if not key:
                    continue
                if key not in sources:
                    sources[key] = {
                        "filename": r.get("filename", ""),
                        "source": r.get("source", ""),
                        "score": r.get("score", 0),
                        "page": r.get("page"),
                        "collection": r.get("collection"),
                        "chunks": [],
                    }
                    order.append(key)
                entry = sources[key]
                entry["score"] = max(entry["score"], r.get("score", 0))

        # 按分数降序，分配 [N] 引用编号
        result = [sources[k] for k in order]
        result.sort(key=lambda s: s["score"], reverse=True)
        for idx, s in enumerate(result, 1):
            s["index"] = idx
        return result

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
        prefetched_docs: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """执行检索并组装系统提示词（异步路径）。

        Args:
            prefetched_docs: 预检索的候选块（跨集合检索时由路由层传入）。
                传入时跳过内部检索，直接重排+过滤。
        """
        kb_chunk_count = await self.vector_store.acount()
        retrieved_docs = []
        has_relevant_docs = False

        if prefetched_docs:
            # 预检索模式：跳过内部检索，直接重排精排
            logger.info(f"使用预检索结果: {len(prefetched_docs)} 个候选块(异步)")
            try:
                final_k = min(k or settings.RETRIEVAL_TOP_K, len(prefetched_docs))
                retrieved_docs = self.reranker.rerank(
                    query=question,
                    candidates=prefetched_docs,
                    top_k=final_k,
                )
            except Exception as e:
                logger.warning(f"预检索结果重排失败(异步)，使用原始候选: {e}")
                retrieved_docs = prefetched_docs[:k or settings.RETRIEVAL_TOP_K]
        elif kb_chunk_count > 0:
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
