r"""
第 15 章 · 检索增强生成（RAG）的配套代码（llm/ch15-rag.md）。只依赖 numpy：python3 ch15_rag.py

用一个小语料库把检索的几种做法从头实现一遍，并用带标注的查询评测：
1. 稀疏检索：BM25（精确匹配关键词、编号）。
2. "稠密"检索：TF-IDF 矩阵做截断 SVD（LSA，第 3 章的低秩近似），在低维空间里算余弦相似度。
   它代替真实的神经网络 embedding，原理相同：把文本映射成向量，用余弦相似度找近邻。
3. 混合检索：倒数排名融合（RRF）。
4. 上下文化分块：给每个块加上它出自哪里，检索更准。
5. 评测：recall@k 和 MRR。
"""
import math, re
from collections import Counter
import numpy as np

# ---------------------------------------------------------------------------
# 语料：每条是一个"块"。有几条故意缺少上下文（"它"指什么不清楚）
# ---------------------------------------------------------------------------
DOCS = {
    "lr":      "梯度下降的学习率也叫步长。学习率太大会发散，必须小于 2/L。",
    "lr2":     "步长选得好，梯度下降收敛快。条件数大时，学习率只能取得很小，收敛很慢。",
    "momentum":"动量法累加过去的梯度，在一致的方向上加速，在来回震荡的方向上相互抵消。",
    "adam1":   "Adam 优化器为每个参数维护一阶矩和二阶矩，用二阶矩归一化步长。",
    "adam2":   "它的默认值是 0.9，二阶矩的衰减系数默认是 0.999。",          # 缺上下文：β1
    "warmup":  "训练开始时学习率从 0 线性升高到峰值，这叫 warmup，可以防止早期发散。",
    "decay":   "学习率衰减：训练后期逐渐降低学习率，让损失更稳定地下降到更低的位置。",
    "softmax": "softmax 把 logits 变成概率分布，实现时先减去最大值，防止指数溢出。",
    "sqrtd":   "注意力打分除以根号 d，因为 q 和 k 的内积方差是 d，不缩放 softmax 会接近 one-hot。",
    "kv":      "生成时前面位置的 key 和 value 不变，可以缓存复用，这就是 KV cache。",
    "lora":    "LoRA 用两个低秩矩阵的乘积表示权重的增量，参数量大幅减少。",
    "svd":     "奇异值分解给出最优的低秩近似，误差等于丢掉的奇异值平方和。",
    "bm25":    "BM25 是基于词频和逆文档频率的排序函数，擅长精确匹配关键词和编号。",
    "mcp":     "MCP 服务器通过 stdio 或 Streamable HTTP 提供工具、资源和提示。",
    "err1":    "错误码 E-4027 表示 MCP 请求的协议版本不受支持，客户端应该先调用 server/discover。",
    "err2":    "错误码 E-4031 表示工具参数校验失败，应该作为工具执行错误返回给模型。",
    "skills":  "Skill 是带 SKILL.md 的文件夹，按渐进式披露的方式加载。",
    "cache":   "提示缓存让相同前缀的请求复用已经算好的结果，命中部分价格更低。",
    "compact": "上下文接近上限时把历史总结成摘要，这叫压缩，可以让 agent 一直运行下去。",
    "evals":   "评测由任务、多次试验和打分器组成，用来发现回归。",
    "inject":  "提示注入是把恶意指令藏在网页、邮件或工具结果里，诱导模型执行。",
}
IDS = list(DOCS)
# 上下文化：给缺少上下文的块加一句"它出自哪里"（真实系统里由模型为每个块生成）
CONTEXT = {"adam2": "（出自：Adam 优化器的超参数，一阶矩衰减系数 β1）"}

QUERIES = [   # (查询, 相关的块)
    ("学习率太大会怎样", {"lr"}),
    ("后期把步长调小有什么用", {"decay"}),          # 查询说"步长"，相关块只说"学习率"
    ("Adam 的 β1 取多少", {"adam2"}),             # 相关块本身没提 Adam 和 β1
    ("E-4027 是什么错误", {"err1"}),               # 编号：要精确匹配
    ("E-4031 怎么处理", {"err2"}),
    ("为什么注意力要除以根号 d", {"sqrtd"}),
    ("怎样减少微调的参数量", {"lora"}),
    ("训练早期为什么会发散", {"warmup", "lr"}),
    ("恶意网页让模型做坏事", {"inject"}),
    ("生成时怎样复用计算", {"kv", "cache"}),
    ("怎样让 agent 长时间运行", {"compact"}),
]


def tokenize(text):
    """中文用相邻两字（bigram），英文和编号整体保留。真实系统用模型自带的分词器。"""
    text = text.lower()
    toks = re.findall(r"[a-z0-9][a-z0-9\-\./]*", text)
    for seg in re.findall(r"[一-鿿]+", text):
        toks += [seg[i:i + 2] for i in range(len(seg) - 1)] or [seg]
    return toks


# ---------------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------------
class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.toks = [tokenize(d) for d in docs]
        self.N = len(docs)
        self.avgdl = sum(map(len, self.toks)) / self.N
        df = Counter(t for ts in self.toks for t in set(ts))
        self.idf = {t: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for t, n in df.items()}
        self.tf = [Counter(ts) for ts in self.toks]
        self.k1, self.b = k1, b

    def scores(self, query):
        q = tokenize(query)
        out = np.zeros(self.N)
        for i, (tf, ts) in enumerate(zip(self.tf, self.toks)):
            norm = self.k1 * (1 - self.b + self.b * len(ts) / self.avgdl)
            out[i] = sum(self.idf.get(t, 0) * tf[t] * (self.k1 + 1) / (tf[t] + norm) for t in q if t in tf)
        return out


# ---------------------------------------------------------------------------
# "稠密"检索：TF-IDF + 截断 SVD（LSA）
# ---------------------------------------------------------------------------
class LSA:
    def __init__(self, docs, k=6):
        self.vocab = sorted({t for d in docs for t in tokenize(d)})
        self.index = {t: j for j, t in enumerate(self.vocab)}
        X = np.array([self.vec(d) for d in docs])                 # 文档 × 词
        df = (X > 0).sum(0)
        self.idf = np.log((1 + len(docs)) / (1 + df)) + 1
        X = X * self.idf
        U, S, Vt = np.linalg.svd(X, full_matrices=False)            # 第 3 章的 SVD
        self.Vk = Vt[:k].T                                          # 词 → k 维"语义"空间
        D = X @ self.Vk
        self.D = D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-12)
        self.energy = (S[:k] ** 2).sum() / (S ** 2).sum()

    def vec(self, text):
        v = np.zeros(len(self.vocab) if hasattr(self, "vocab") else 0)
        for t in tokenize(text):
            if t in self.index: v[self.index[t]] += 1
        return v

    def scores(self, query):
        q = (self.vec(query) * self.idf) @ self.Vk
        q = q / (np.linalg.norm(q) + 1e-12)
        return self.D @ q                                           # 余弦相似度


def rrf(*score_lists, k=60):
    """倒数排名融合：每个检索器贡献 1/(k + 排名)，只看排名，不看分数的尺度。"""
    fused = np.zeros(len(score_lists[0]))
    for s in score_lists:
        ranks = np.empty(len(s)); ranks[np.argsort(-s)] = np.arange(1, len(s) + 1)
        fused += 1 / (k + ranks)
    return fused


def evaluate(score_fn, ks=(1, 3)):
    rec = {k: [] for k in ks}; rr = []
    for q, rel in QUERIES:
        order = [IDS[i] for i in np.argsort(-score_fn(q))]
        for k in ks: rec[k].append(len(rel & set(order[:k])) / len(rel))
        rr.append(next(1 / (r + 1) for r, d in enumerate(order) if d in rel))
    return {f"recall@{k}": np.mean(v) for k, v in rec.items()} | {"MRR": np.mean(rr)}


plain = [DOCS[i] for i in IDS]
ctx = [CONTEXT.get(i, "") + DOCS[i] for i in IDS]

print(f"=== 检索方法对比（{len(QUERIES)} 个带标注的查询） ===")
print(f"  {'方法':<26}{'recall@1':>10}{'recall@3':>10}{'MRR':>8}")
results = {}
for label, docs in [("原始分块", plain), ("上下文化分块", ctx)]:
    bm, ls = BM25(docs), LSA(docs)
    for name, fn in [("BM25", bm.scores), ("LSA 稠密", ls.scores),
                     ("混合 RRF", lambda q, bm=bm, ls=ls: rrf(bm.scores(q), ls.scores(q)))]:
        r = evaluate(fn); results[(label, name)] = r
        print(f"  {label + ' + ' + name:<24}{r['recall@1']:>10.2f}{r['recall@3']:>10.2f}{r['MRR']:>8.2f}")
print(f"  （LSA 用了 6 维，保留了 TF-IDF 矩阵 {LSA(plain).energy:.0%} 的能量）")

print("\n=== 逐条看排名第一的结果（原始分块） ===")
bm, ls = BM25(plain), LSA(plain)
for q, rel in QUERIES:
    top = lambda s: IDS[int(np.argmax(s))]
    b, l, h = top(bm.scores(q)), top(ls.scores(q)), top(rrf(bm.scores(q), ls.scores(q)))
    mark = lambda d: d + ("✓" if d in rel else "✗")
    print(f"  {q:<18} BM25 {mark(b):<10} LSA {mark(l):<10} 混合 {mark(h)}")

print("\n=== 上下文化对单个查询的影响 ===")
q = "Adam 的 β1 取多少"
for label, docs in [("原始", plain), ("上下文化", ctx)]:
    s = BM25(docs).scores(q)
    order = [IDS[i] for i in np.argsort(-s)]
    print(f"  {label}：adam2 排第 {order.index('adam2') + 1}，前三 {order[:3]}")
