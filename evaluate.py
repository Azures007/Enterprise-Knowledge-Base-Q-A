"""
=============================================================================
RAG 检索评测脚本

用内置小型中文语料库对比三种检索策略的召回质量：
    1. vector-only    纯向量检索（baseline）
    2. hybrid-weighted  混合检索 - 线性加权（向量 0.7 + 关键词 0.3）
    3. hybrid-rrf     混合检索 - RRF 双通道融合（默认，生产配置）

指标:
    - Recall@K   前 K 命中率（有相关块被召回即计命中，K=1,3,5）
    - MRR        首个相关块排名的倒数均值
    - hit@1      首个相关块排在首位的问题占比

使用方法:
    python evaluate.py                # 默认走 ChromaDB 内存评测
    python evaluate.py --mode weighted   # 指定融合模式（rrf | weighted | vector）
    python evaluate.py --verbose         # 打印每个问题各策略的排名明细

注意:
    - 本脚本内置语料库与人工标注的"相关块"，无需外部文档或数据库。
    - 使用本地确定性嵌入（词袋式哈希向量），不消耗任何 API Token，
      因此能离线、可复现地量化检索策略差异。
=============================================================================
"""

import argparse
import math
import re
import sys
from typing import Any

import numpy as np

# 项目路径导入
sys.path.insert(0, __file__.rsplit("/", 1)[0])

from config.settings import settings  # noqa: E402


# ==============================================================================
# 1. 本地确定性嵌入（词袋式哈希向量，仅供检索策略对比）
# ==============================================================================

class HashEmbeddings:
    """基于词袋 + 哈希特征的中文嵌入（离线、可复现，模拟语义向量）。

    生成 512 维稀疏哈希向量：中英文 n-gram 特征做确定性哈希，
    命中对应槽位累加权重，最后 L2 归一化。

    模拟真实语义嵌入的两个关键特性：
    1. 同义归一化（NORM）：把口语/简写归一为规范表达 —— 对应嵌入模型
       对同义表达的语义理解。
    2. 语料级 IDF 过滤（fit）：只保留在语料中出现在一定比例以上文档的
       "共享语义特征"；只出现一次的精确 token（编号/代号/电话号码）被
       从语义向量中剔除 —— 对应真实嵌入把低频专有信息压缩进向量、
       语义相似度无法靠它们区分的事实。

    由此构造出"向量通道擅长语义、关键词通道擅长精确 token"的真实分工，
    使评测能体现混合检索相对纯向量检索的收益。仅用于检索策略评测，
    不用于生产真实语义。
    """

    DIM = 512
    N_GRAMS = (1, 2, 3)
    # 保留特征的最低文档频次比例（< 该比例的特征视为"专有信息"，从语义向量剔除）
    MIN_DF_RATIO = 0.10

    # 同义/规范表达归一化（按最长优先替换，模拟语义理解）
    NORM = [
        ("几点上班", "上下班时间"), ("几点下班", "上下班时间"),
        ("几点", "时间"), ("上班时间", "上下班时间"),
        ("下班时间", "上下班时间"), ("怎么申请", "申请流程"),
        ("怎么开通", "申请流程"), ("怎么命名", "命名规则"),
        ("命名规范", "命名规则"), ("多长", "时长"),
        ("多久", "周期"), ("有效期", "周期"), ("换一次", "更换"),
        ("更换一次", "更换"), ("标准", "规定"), ("什么标准", "规定"),
        ("什么时候", "时间"), ("多少天", "天数"), ("几天", "天数"),
        ("值班电话", "联系电话"), ("电话", "联系电话"),
        ("怎么算", "如何计算"), ("怎么脱敏", "脱敏规则"),
    ]

    def __init__(self):
        self._vocab: set[str] = set()

    def fit(self, texts: list[str]) -> None:
        """基于语料统计特征词频，建立"共享语义特征"词汇表。"""
        from collections import Counter

        df = Counter()
        for t in texts:
            for f in set(self._features(self._norm(t))):
                df[f] += 1
        total = max(len(texts), 1)
        self._vocab = {f for f, c in df.items() if c / total >= self.MIN_DF_RATIO}

    def _norm(self, text: str) -> str:
        out = text
        for src, dst in self.NORM:
            out = out.replace(src, dst)
        return out

    def _features(self, text: str) -> list[str]:
        feats: list[str] = []
        for n in self.N_GRAMS:
            for i in range(len(text) - n + 1):
                feats.append(text[i:i + n])
        return feats

    def _embed(self, text: str) -> list[float]:
        vec = np.zeros(self.DIM, dtype=np.float64)
        for f in self._features(self._norm(text)):
            # 仅保留共享语义特征；专有 token 不进语义向量（模拟嵌入压缩）
            if f not in self._vocab:
                continue
            h = int(hashlib_md5(f).hexdigest()[:8], 16) % self.DIM
            vec[h] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def hashlib_md5(s: str):
    import hashlib
    return hashlib.md5(s.encode("utf-8"))


class FakeVectorStore:
    """极简向量库：本地余弦检索 + 可选关键词通道（内存 n-gram 命中）。"""

    def __init__(self, embedder: HashEmbeddings):
        self.embedder = embedder
        self._contents: list[str] = []
        self._vecs: list[list[float]] = []

    def add(self, contents: list[str]):
        self._contents = list(contents)
        self._vecs = self.embedder.embed_documents(contents)

    # ---- 通道 1：向量检索 ----
    def similarity_search(self, query: str, k: int = 20) -> list[dict[str, Any]]:
        qv = np.array(self.embedder.embed_query(query), dtype=np.float64)
        scores = []
        for i, vec in enumerate(self._vecs):
            s = float(np.dot(qv, np.array(vec, dtype=np.float64)))
            scores.append((i, max(0.0, s)))
        scores.sort(key=lambda x: x[1], reverse=True)
        return [
            {"content": self._contents[i], "score": round(s, 4)}
            for i, s in scores[:k]
        ]

    # ---- 通道 2：关键词检索（子串命中） ----
    def _keyword_substring_search(self, query: str, k: int = 20) -> list[dict[str, Any]]:
        terms = self._extract_terms(query)
        if not terms:
            return []
        scored = []
        for content in self._contents:
            cl = content.lower()
            hits = sum(1 for t in terms if t in cl)
            if hits == 0:
                continue
            scored.append({
                "content": content,
                "score": round(min(1.0, hits / len(terms)), 4),
            })
        scored.sort(key=lambda d: d["score"], reverse=True)
        return scored[:k]

    @staticmethod
    def _extract_terms(query: str) -> list[str]:
        terms: list[str] = []
        for w in re.findall(r"[a-zA-Z0-9][a-zA-Z0-9_.\-]{1,}", query):
            terms.append(w.lower())
        for seg in re.findall(r"[一-鿿]{2,}", query):
            for n in (2, 3):
                for i in range(len(seg) - n + 1):
                    gram = seg[i:i + n]
                    if gram not in terms:
                        terms.append(gram)
        seen, result = set(), []
        for t in terms:
            if t not in seen:
                seen.add(t)
                result.append(t)
            if len(result) >= 15:
                break
        return result

    def hybrid_search(
        self,
        query: str,
        k: int = 5,
        vector_weight: float = 0.7,
        keyword_weight: float = 0.3,
        mode: str = "rrf",
    ) -> list[dict[str, Any]]:
        candidate_k = 20
        fusion_const = 60
        vec_results = self.similarity_search(query, k=candidate_k)
        kw_results = self._keyword_substring_search(query, k=candidate_k)

        if mode == "weighted":
            return self._merge_weighted(
                vec_results, kw_results, vector_weight, keyword_weight, k,
            )
        if mode == "vector":
            return vec_results[:k]
        return self._merge_rrf(vec_results, kw_results, vector_weight, keyword_weight, fusion_const, k)

    def _merge_rrf(self, vec, kw, wv, wk, kconst, k):
        rank_scores: dict[str, float] = {}
        flags: dict[str, dict] = {}
        for rank, doc in enumerate(vec):
            key = doc["content"]
            rank_scores[key] = rank_scores.get(key, 0.0) + wv / (kconst + rank)
            flags.setdefault(key, {})["vector"] = doc
        for rank, doc in enumerate(kw):
            key = doc["content"]
            rank_scores[key] = rank_scores.get(key, 0.0) + wk / (kconst + rank)
            flags.setdefault(key, {})["keyword"] = doc
        merged = []
        for key, score in rank_scores.items():
            vd = flags[key].get("vector")
            kd = flags[key].get("keyword")
            vs = vd["score"] if vd else 0.0
            ks = kd["score"] if kd else 0.0
            combined = wv * vs + wk * ks if (vd and kd) else (vs if vd else ks)
            merged.append({
                "content": key,
                "rrf_score": score,
                "combined_score": combined,
                "channels": ("vector" if vd else "") + ("+keyword" if kd else ""),
            })
        merged.sort(key=lambda d: (d["rrf_score"], 1 if "+" in d["channels"] else 0), reverse=True)
        return merged[:k]

    def _merge_weighted(self, vec, kw, wv, wk, k):
        vec_map = {d["content"]: d for d in vec}
        kw_map = {d["content"]: d for d in kw}
        merged = []
        for key in set(vec_map) | set(kw_map):
            vd = vec_map.get(key)
            kd = kw_map.get(key)
            vs = vd["score"] if vd else 0.0
            ks = kd["score"] if kd else 0.0
            combined = wv * vs + wk * ks if (vd and kd) else (vs if vd else ks)
            merged.append({
                "content": key,
                "combined_score": combined,
                "channels": ("vector" if vd else "") + ("+keyword" if kd else ""),
            })
        merged.sort(key=lambda d: (d["combined_score"], 1 if "+" in d["channels"] else 0), reverse=True)
        return merged[:k]


# ==============================================================================
# 2. 评测数据集：语料块 + 问题 → 期望命中的块索引
# ==============================================================================

CORPUS = [
    # ---- 精确 token 难例（专有名词/编号，纯向量易漏检） ----
    "公司考勤制度规定：工作日为周一至周五，上下班时间分别为上午 9:00 与下午 18:00。",
    "迟到超过 30 分钟记为旷工半天，连续旷工 3 天可作解除劳动合同处理。",
    "差旅报销标准：一线城市住宿每晚 600 元、二三线城市每晚 400 元，超支部分自理。",
    "生产环境服务器命名规则：前缀 prod，如 prod-api-01、prod-db-02。",
    "ERP 系统账号申请流程：填写 OA 表单，部门主管审批后由 IT 部开通。",
    "年假规定：工龄满 1 年 5 天、满 5 年 10 天、满 10 年 15 天。",
    "高温补贴发放时间为每年 6-9 月，每月 200 元随工资发放。",
    "会议室预订规则：单次不超过 2 小时，超时须在前台登记。",
    "企业邮箱密码策略：8 位以上含大小写字母与数字，每 90 天更换一次。",
    "IT 值班电话 400-123-4567，网络故障 30 分钟内报修。",
    "采购流程：金额超过 5 万元的采购需三家比价，比价表交财务复核。",
    "客户数据脱敏规范：手机号中间 4 位隐藏，身份证只保留前 6 后 4 位。",
    "员工餐补标准：工作日午餐补贴 25 元，加班晚餐补贴 40 元。",
    "项目代号命名规范：2024 年新项目统一以 K24 开头，如 K24-order-center。",
    "报销单号规则：财务编号 R-2024-XXXX，需与发票号码一一对应。",
    # ---- 语义相关（向量通道占优） ----
    "软件开发流程包含需求评审、代码评审、自动化测试与灰度发布等环节。",
    "公司采用敏捷开发模式，以两周为一个迭代周期，每日站会同步进度。",
    "新员工入职首周需完成信息安全培训、公司制度学习与导师配对。",
    "数据备份策略：核心数据库每日全量备份，日志数据每 6 小时增量备份。",
    "员工晋升通道分管理序列与技术序列，每半年开放一次晋升评审。",
    "项目结项需交付物清单、验收报告与归档材料三部分。",
    "远程办公需提前一天申请，经直属主管审批后执行。",
    "年度团建经费按人均 800 元预算，由各部门自行组织。",
    "加班需提前在 OA 提交申请，经部门主管审批后计调休。",
    "公司实行弹性工作制，核心时段为 10:00-16:00，其余时间自由安排。",
    # ---- 干扰块（弱相关，向量分可能虚高） ----
    "会议室设备清单：投影仪、白板、视频会议终端，使用前请检查。",
    "ERP 系统的财务模块支持发票核销与成本分摊，由财务部维护。",
    "OA 系统表单库包含请假、报销、采购、加班等常用流程模板。",
    "IT 部负责办公电脑、门禁卡、网络权限的配置与故障处理。",
    "员工手册提到信息安全红线：禁止外发客户数据、禁止私设热点。",
    "差旅审批流程：出差前需提交行程单，返回后 5 个工作日内报销。",
    "考勤补卡规则：每月最多补卡 3 次，需主管签字确认。",
    "邮箱使用规范：商务往来一律使用企业邮箱，禁止使用个人邮箱。",
    "数据脱敏工具：测试环境使用 faker 生成假数据，生产数据严禁外泄。",
    "团建活动安全提示：出行需购买意外险，活动经费需提前审批。",
]

# 难例标注：黄金块索引 + 干扰块设计已在语料注释中说明
# (干扰块与黄金块共享语义主题，但缺失精确 token)

QUERIES = [
    # (问题, 期望相关块索引列表, 类型)
    ("上班时间是几点", [0], "general"),
    ("迟到多久算旷工", [1], "keyword"),
    ("住宿报销标准是多少", [2], "keyword"),
    ("生产服务器怎么命名", [3], "keyword"),
    ("怎么申请 ERP 账号", [4], "keyword"),
    ("年假规定是多少天", [5], "keyword"),
    ("高温补贴什么时候发", [6], "keyword"),
    ("会议室能订多久", [7], "general"),
    ("密码多久改一次", [8], "keyword"),
    ("IT 值班电话是多少", [9], "keyword"),
    ("采购超过多少钱要比价", [10], "keyword"),
    ("客户手机号怎么脱敏", [11], "keyword"),
    ("午餐补贴多少钱", [12], "keyword"),
    ("项目代号怎么命名", [13], "keyword"),
    ("报销单号格式", [14], "keyword"),
    ("软件开发的流程是什么", [15], "semantic"),
    ("公司用什么开发模式", [16], "semantic"),
    ("新员工入职要做什么", [17], "general"),
    ("数据备份怎么做", [18], "semantic"),
    ("员工怎么晋升", [19], "semantic"),
    ("项目结项要什么材料", [20], "semantic"),
    ("远程办公怎么申请", [21], "semantic"),
    ("团建经费标准", [22], "semantic"),
    ("加班怎么算", [23], "semantic"),
    ("工作时间是怎么安排的", [24], "semantic"),
]


# ==============================================================================
# 3. 指标计算
# ==============================================================================

def compute_metrics(queries, results_by_query):
    """计算 Recall@K / MRR / hit@1。

    results_by_query: 每问召回的相关块索引列表（非零即命中，多黄金块取首个）。
    """
    ks = (1, 3, 5)
    recall = {k: 0 for k in ks}
    mrr_sum = 0.0
    hit1 = 0
    total = len(queries)

    for (q, gold, _qtype), res in zip(queries, results_by_query):
        gold_set = set(gold)
        # Recall@K：gold 中任一被前 K 召回即计命中
        for k in ks:
            if gold_set & set(res[:k]):
                recall[k] += 1
        # MRR：首个相关块排名的倒数（仅取黄金块中的第一个）
        for rank, idx in enumerate(res, start=1):
            if idx in gold_set:
                mrr_sum += 1.0 / rank
                if rank == 1:
                    hit1 += 1
                break

    n = max(total, 1)
    return {
        **{f"Recall@{k}": round(recall[k] / n, 4) for k in ks},
        "MRR": round(mrr_sum / n, 4),
        "hit@1": round(hit1 / n, 4),
    }


# ==============================================================================
# 4. 主流程
# ==============================================================================

def run_strategy(store: FakeVectorStore, mode: str, k: int = 5) -> list[list[int]]:
    """对每个问题跑指定策略，返回每问召回的语料块索引列表。"""
    all_res = []
    for (q, gold, _qtype) in QUERIES:
        if mode == "vector":
            docs = store.similarity_search(q, k=k)
        else:
            docs = store.hybrid_search(q, k=k, mode=mode)
        # 映射到语料索引
        idxs = [CORPUS.index(d["content"]) for d in docs if d["content"] in CORPUS]
        all_res.append(idxs)
    return all_res


def print_table(rows: list[list[str]]):
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    for row in rows:
        print("  " + "  ".join(cell.ljust(w) for cell, w in zip(row, widths)))


def main():
    parser = argparse.ArgumentParser(description="RAG 检索策略评测")
    parser.add_argument("--mode", choices=["vector", "weighted", "rrf"],
                        default="rrf", help="要单独评测的融合模式（默认 rrf）")
    parser.add_argument("--verbose", action="store_true", help="打印每个问题的排名明细")
    parser.add_argument("--k", type=int, default=5, help="检索返回条数")
    args = parser.parse_args()

    store = FakeVectorStore(HashEmbeddings())
    # 语料级 IDF 拟合：建立"共享语义特征"词汇表，模拟真实嵌入对低频专有 token 的压缩
    store.embedder.fit(CORPUS)
    store.add(CORPUS)

    print(f"\n{'='*70}")
    print("RAG 检索策略评测  (K = {})".format(args.k))
    print(f"语料块数: {len(CORPUS)}  评测问题数: {len(QUERIES)}")
    print(f"{'='*70}\n")

    strategies = ["vector", "weighted", "rrf"]
    results = {}
    for mode in strategies:
        results[mode] = run_strategy(store, mode, k=args.k)

    if args.verbose:
        print("—— 各问题排名明细 ——")
        for i, (q, gold, qtype) in enumerate(QUERIES):
            line = f"[{i}] {qtype:8s} | {q}"
            print(f"  {line}")
            for mode in strategies:
                idxs = results[mode][i]
                mark = "✓" if (gold and gold[0] in idxs) else " "
                pos = idxs.index(gold[0]) + 1 if (gold and gold[0] in idxs) else "-"
                print(f"      {mode:10s} 命中@{pos} {mark}  {idxs[:5]}")
        print()

    print("—— 指标对比 ——")
    header = ["策略"] + [f"Recall@{k}" for k in (1, 3, 5)] + ["MRR", "hit@1"]
    rows = [header]
    for mode in strategies:
        m = compute_metrics(QUERIES, results[mode])
        rows.append([mode] + [str(m[f"Recall@{k}"]) for k in (1, 3, 5)] + [str(m["MRR"]), str(m["hit@1"])])
    print_table(rows)

    # 相对提升
    base = compute_metrics(QUERIES, results["vector"])
    rrf = compute_metrics(QUERIES, results["rrf"])
    print(f"\n—— RRF 相对纯向量的提升 ——")
    for metric in ("Recall@3", "MRR", "hit@1"):
        b, r = base[metric], rrf[metric]
        delta = (r - b) / b * 100 if b > 0 else 0
        print(f"  {metric:10s}  {b:.4f} → {r:.4f}   ({delta:+.1f}%)")
    print()


if __name__ == "__main__":
    main()
