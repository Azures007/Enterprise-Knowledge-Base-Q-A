"""
=============================================================================
检索重排（Rerank）模块

向量检索（双塔相似度）只能粗召回，无法细粒度判断"分数高是否真的相关"。
重排器对"问题 vs 候选块"逐对计算相关性，从中选出真正相关的块。

两级实现：
    1. 交叉编码器重排（首选）：bge-reranker 系列，把问题与文档拼接后整体
       编码，语义相关性判断准确度比向量相似度高一个量级。
       依赖 torch + sentence-transformers（可选安装）。
    2. 轻量重排（降级）：无外部依赖。融合向量分数与关键词重合度，
       在纯向量排序基础上进一步修正，无需下载模型也能用。

使用方法:
    from src.reranker import Reranker

    reranker = Reranker()
    docs = reranker.rerank(query="公司考勤制度是什么？", candidates=[...])
=============================================================================
"""

import re
from typing import Any

from config.settings import settings
from src.monitoring import get_metrics, record_rerank_degraded
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class Reranker:
    """
    检索重排器。根据配置优先使用交叉编码器，不可用时自动降级为轻量重排。
    """

    def __init__(self, enabled: bool | None = None):
        """
        Args:
            enabled: 是否启用重排（默认读取 settings.RERANK_ENABLED）
        """
        self.enabled = enabled if enabled is not None else settings.RERANK_ENABLED
        self._model = None
        self._model_available = False
        self._mode = "off"

        if self.enabled:
            self._try_load_cross_encoder()
            # 若交叉编码器不可用，标记为轻量模式（rerank 时实际生效）
            if self._mode == "off":
                self._mode = "lightweight"
        # 暴露当前模式到指标（供 /api/metrics 查看）
        get_metrics().set_gauge("rerank_mode", self._mode)

    def _try_load_cross_encoder(self):
        """尝试加载本地交叉编码器模型（bge-reranker-base）。

        模型加载可能触发 Hugging Face 联网下载，在无外网/慢网环境下会
        长时间阻塞。这里用带超时的后台线程加载：超时即降级为轻量重排，
        保证应用启动不被模型下载阻塞。
        """
        import threading

        model_name = getattr(settings, "RERANKER_MODEL", None) or "BAAI/bge-reranker-base"
        result = {}

        def _load():
            try:
                from sentence_transformers import CrossEncoder
                result["model"] = CrossEncoder(model_name)
                result["ok"] = True
            except Exception as e:
                result["error"] = e

        # 模型下载/加载最多等待 RERANKER_TIMEOUT 秒（默认 30s）
        timeout = getattr(settings, "RERANKER_TIMEOUT", 30)
        t = threading.Thread(target=_load, daemon=True)
        t.start()
        t.join(timeout=timeout)

        if result.get("ok"):
            self._model = result["model"]
            self._model_available = True
            self._mode = "cross_encoder"
            logger.info(f"重排器: 交叉编码器已加载 ({model_name})")
        elif t.is_alive():
            # 超时：模型仍在后台下载，本次放弃，降级轻量重排
            self._model = None
            self._model_available = False
            record_rerank_degraded(f"交叉编码器加载超时（>{timeout}s）")
            logger.warning(
                f"交叉编码器加载超时（>{timeout}s，可能需从 Hugging Face 下载模型），"
                f"已降级为轻量重排。可设置 RERANKER_TIMEOUT 增大等待时间"
            )
        else:
            self._model = None
            self._model_available = False
            record_rerank_degraded(f"交叉编码器加载失败: {result.get('error')}")
            logger.warning(
                f"交叉编码器加载失败，降级为轻量重排: {result.get('error')}\n"
                f"如需使用语义重排，请安装: pip install sentence-transformers"
            )

    # ================================================================
    # 公开接口
    # ================================================================

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        对候选块重排，返回分数降序的前 top_k 个。

        Args:
            query:      用户问题
            candidates: 向量检索返回的候选块列表（含 content, score, metadata）
            top_k:      返回数量（默认 settings.RETRIEVAL_TOP_K）

        Returns:
            重排后的块列表（按相关性降序），每项含原字段 + 新增 rerank_score
        """
        if not self.enabled or not candidates:
            return candidates

        top_k = top_k or settings.RETRIEVAL_TOP_K

        if self._mode == "cross_encoder":
            return self._rerank_cross_encoder(query, candidates, top_k)
        else:
            return self._rerank_lightweight(query, candidates, top_k)

    # ================================================================
    # 交叉编码器重排
    # ================================================================

    def _rerank_cross_encoder(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        """使用交叉编码器逐对计算相关性分数"""
        try:
            pairs = [
                (query, doc.get("content", ""))
                for doc in candidates
            ]
            scores = self._model.predict(pairs)

            # 合并分数并按交叉编码器分数降序
            ranked = list(zip(candidates, scores))
            ranked.sort(key=lambda x: x[1], reverse=True)

            result = []
            for doc, score in ranked[:top_k]:
                item = dict(doc)
                item["rerank_score"] = round(float(score), 4)
                result.append(item)

            logger.debug(
                f"交叉编码器重排: {len(candidates)} 候选 → {len(result)} 结果"
            )
            return result
        except Exception as e:
            record_rerank_degraded(f"交叉编码器重排运行时失败: {e}")
            logger.warning(f"交叉编码器重排失败，降级为轻量重排: {e}")
            return self._rerank_lightweight(query, candidates, top_k)

    # ================================================================
    # 轻量重排（无依赖降级）
    # ================================================================

    @staticmethod
    def _keyword_overlap(query: str, content: str) -> float:
        """
        计算问题与文档块的关键词重合度。

        提取中文分词式的关键片段（2~4 字 n-gram）与英文单词，
        统计双方交集占问题关键词的比例，作为关键词相关性分数（0~1）。
        """
        # 英文单词 + 中文连续片段（粗略关键信息）
        en_words = re.findall(r"[a-zA-Z0-9_]+", query.lower())
        cn_segments = re.findall(r"[一-鿿]{2,}", query)

        query_keys = set(en_words)
        for seg in cn_segments:
            # 中文取 2-gram 与 3-gram 作为关键词
            for n in (2, 3):
                for i in range(len(seg) - n + 1):
                    query_keys.add(seg[i:i + n])

        if not query_keys:
            return 0.0

        content_lower = content.lower()
        hits = sum(1 for k in query_keys if k in content_lower)
        return hits / len(query_keys)

    def _rerank_lightweight(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        """
        轻量重排：向量分数 + 关键词重合度双层判定。

        关键设计：向量相似度对"恰好提到同一词"的块很敏感（单块高分陷阱），
        因此不能简单在向量分上叠加关键词分——那样向量分仍占主导，重排失效。

        这里采用两层机制：
            1. 关键词否决：内容中完全没有问题关键信息的块（关键词重合度过低），
               即使向量分再高也降级处理，避免"提到同一词但不相关"的块混入。
            2. 加权排序：通过否决的块，按 0.5*向量 + 0.5*关键词 排序，
               让关键词维度与向量维度同等权重，而非被向量分主导。
        """
        self._mode = "lightweight"

        # 关键词重合度阈值：低于此值视为"内容不含问题关键信息"
        kw_threshold = 0.05

        # 第一层：计算每个候选的综合分（含否决标记）
        scored = []
        for doc in candidates:
            vec_score = doc.get("score", 0)
            kw_score = self._keyword_overlap(query, doc.get("content", ""))

            if kw_score < kw_threshold:
                # 关键词否决：内容不含问题关键信息 → 大幅降权
                # 但保留其作为候选（向量分高可能仍有价值），仅将其排到所有
                # 含关键词的块之后
                combined = 0.1 * vec_score + 0.0 * kw_score
                blocked = True
            else:
                # 含关键词：向量与关键词各占 50% 权重
                combined = 0.5 * vec_score + 0.5 * kw_score
                blocked = False

            scored.append((doc, vec_score, kw_score, combined, blocked))

        # 排序：含关键词的块在前（按综合分），被否决的块在后（按向量分）
        scored.sort(
            key=lambda x: (not x[4], x[3]),
            reverse=True,
        )

        result = []
        for doc, vec_score, kw_score, combined, blocked in scored[:top_k]:
            item = dict(doc)
            item["rerank_score"] = round(combined, 4)
            item["keyword_score"] = round(kw_score, 4)
            item["keyword_blocked"] = blocked
            result.append(item)

        logger.debug(
            f"轻量重排: {len(candidates)} 候选 → {len(result)} 结果"
            f"（{sum(1 for x in scored[:len(result)] if x[4])} 个被关键词否决）"
        )
        return result

    # ================================================================
    # 状态
    # ================================================================

    @property
    def mode(self) -> str:
        """当前重排模式: off | lightweight | cross_encoder"""
        return self._mode

    @property
    def is_active(self) -> bool:
        """是否启用了重排"""
        return self.enabled and self._mode != "off"
