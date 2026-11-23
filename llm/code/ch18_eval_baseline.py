r"""
第 18 章 · 实战项目的第一个实验（llm/ch18-project-design.md）。python3 ch18_eval_baseline.py

项目要做一个"读这本书、陪你练面试"的助手。它的检索部分需要一个带标注的评测集。
附录 C 恰好就是：每道标了 ✅ 的面试题，后面链接着能回答它的节。这份脚本：

1. 从附录 C 里取出 ✅ 题和它们链接的节，当作"问题 → 正确的节"的标注；
2. 把全书（第 1–17 章和附录 A）切成块，每块记住自己属于哪一节；
3. 用几种检索方法找块，再把块的排名合并成节的排名；
4. 算 recall@k 和 MRR，看哪种方法好、在哪些题上失败。

只依赖 numpy；装了 jieba（pip install jieba）会多跑两种按词切分的方法。
不需要 API key，也不需要 GPU。
"""
import math, re, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

BOOK = Path(__file__).resolve().parent.parent          # llm/
# 只用第 1–17 章和附录 A。第 18 章起是项目本身，里面引用了附录 C 的题和本脚本的结果，
# 放进语料就等于把答案混进了考卷（评测集污染）。
CHAPTERS = [p for p in sorted(BOOK.glob("ch*.md")) if int(p.name[2:4]) <= 17] \
    + [BOOK / "appendix-a-math-quick-reference.md"]
APPENDIX_C = BOOK / "appendix-c-interview-questions.md"

# ---------------------------------------------------------------------------
# 1. 评测集：附录 C 中标 ✅ 的题
# ---------------------------------------------------------------------------
LINK = re.compile(r"\]\(([\w-]+\.md)#([0-9A-Z]+(?:\.\d+)+)\)")
CHAP_REF = re.compile(r"第 ([\d、]+) 章")  # "第 15 章""第 10、11 章"：整章都算对


def section_of(num):
    """17.12.1 → 17.12：评测按"节"（二级标题）算，小节归到它所在的节。"""
    return ".".join(num.split(".")[:2])


def load_eval_set():
    items = []
    for line in APPENDIX_C.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^\d+\.\s+(★\s*)?(.+?)\s+——\s+✅(.*)$", line)
        if not m:
            continue
        star, question, refs = bool(m.group(1)), m.group(2), m.group(3)
        gold = {section_of(n) for _, n in LINK.findall(refs)}
        chapters = {int(x) for m2 in CHAP_REF.finditer(refs) for x in m2.group(1).split("、")}
        question = re.sub(r"[（(](Anthropic|OpenAI|Sierra|Scale AI|xAI|Google|Databricks)[^）)]*[）)]", "", question)
        if gold or chapters:
            items.append({"q": question.strip(), "gold": gold, "chapters": chapters, "star": star})
    return items


# ---------------------------------------------------------------------------
# 2. 语料：按节切块
# ---------------------------------------------------------------------------
def strip_md(text):
    text = re.sub(r"```.*?```", " ", text, flags=re.S)          # 代码块不进索引
    text = re.sub(r"\$\$.*?\$\$", " ", text, flags=re.S)         # 公式块
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)         # 链接只留文字
    text = re.sub(r"[*`>|#]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_chunks(max_chars=400):
    """每个二级标题是一节；节内按空行分段，再把段落拼成不超过 max_chars 的块。"""
    chunks = []
    for path in CHAPTERS:
        raw = path.read_text(encoding="utf-8")
        chap_title = re.search(r"^# (.+)$", raw, re.M).group(1)
        parts = re.split(r"^## ", raw, flags=re.M)[1:]
        for part in parts:
            head, _, body = part.partition("\n")
            m = re.match(r"([0-9A]+\.\d+)\s+(.*)", head)
            if not m:                                            # 导读、参考资料等不编号的节
                continue
            sec, title = m.group(1), re.sub(r"\*\*\[.*?\]\*\*", "", m.group(2)).strip()
            buf = ""
            for para in re.split(r"\n\s*\n", body):
                para = strip_md(para)
                if not para:
                    continue
                if buf and len(buf) + len(para) > max_chars:
                    chunks.append({"sec": sec, "text": buf, "ctx": f"{chap_title} {sec} {title}"})
                    buf = ""
                buf = (buf + " " + para).strip()
            if buf:
                chunks.append({"sec": sec, "text": buf, "ctx": f"{chap_title} {sec} {title}"})
    return chunks


# ---------------------------------------------------------------------------
# 3. 检索方法
# ---------------------------------------------------------------------------
STOP = set("的 了 是 在 和 与 有 什么 怎么 怎样 为什么 如何 哪些 分别 一个 这个 吗 呢 你 我 它 时 会 要 能 用 把 被 对 中 里 上 下 就 都 也 还 而 或 及".split())


def tok_bigram(text):
    """不依赖分词器：英文按词，中文按相邻两个字。"""
    text = text.lower()
    out = re.findall(r"[a-z0-9]+(?:[.\-][a-z0-9]+)*", text)
    for run in re.findall(r"[一-鿿]+", text):
        out += [run[i:i + 2] for i in range(len(run) - 1)] or [run]
    return out


def tok_jieba(text):
    import jieba
    return [w for w in (t.strip().lower() for t in jieba.cut(text)) if w and w not in STOP and not re.fullmatch(r"\W+", w)]


class BM25:
    def __init__(self, docs, tokenize, k1=1.5, b=0.75):
        self.tok, self.k1, self.b = tokenize, k1, b
        self.tf = [Counter(tokenize(d)) for d in docs]
        self.len = np.array([sum(c.values()) for c in self.tf], float)
        self.avg = self.len.mean()
        df = Counter(w for c in self.tf for w in c)
        n = len(docs)
        self.idf = {w: math.log(1 + (n - f + 0.5) / (f + 0.5)) for w, f in df.items()}

    def scores(self, query):
        s = np.zeros(len(self.tf))
        for w in set(self.tok(query)):
            if w not in self.idf:
                continue
            f = np.array([c.get(w, 0) for c in self.tf], float)
            s += self.idf[w] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.len / self.avg))
        return s


class LSA:
    """第 15 章用过的"穷人版 embedding"：TF-IDF 矩阵做截断 SVD，在低维空间算余弦。"""

    def __init__(self, docs, tokenize, k=128):
        self.tok = tokenize
        tfs = [Counter(tokenize(d)) for d in docs]
        self.vocab = {w: i for i, w in enumerate(sorted({w for c in tfs for w in c}))}
        df = Counter(w for c in tfs for w in c)
        self.idf = np.array([math.log(len(docs) / df[w]) for w in sorted(self.vocab)])
        X = np.zeros((len(docs), len(self.vocab)))
        for i, c in enumerate(tfs):
            for w, f in c.items():
                X[i, self.vocab[w]] = (1 + math.log(f)) * self.idf[self.vocab[w]]
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        self.V = Vt[:k].T                                       # 词空间 → k 维
        D = X @ self.V
        self.D = D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-9)

    def scores(self, query):
        x = np.zeros(len(self.vocab))
        for w, f in Counter(self.tok(query)).items():
            if w in self.vocab:
                x[self.vocab[w]] = (1 + math.log(f)) * self.idf[self.vocab[w]]
        q = x @ self.V
        return self.D @ (q / (np.linalg.norm(q) + 1e-9))


def rrf(*score_lists, k=60):
    """倒数排名融合：只用名次，不管两路分数的量纲。"""
    total = np.zeros(len(score_lists[0]))
    for s in score_lists:
        rank = np.empty(len(s), int)
        rank[np.argsort(-s)] = np.arange(1, len(s) + 1)
        total += 1 / (k + rank)
    return total


# ---------------------------------------------------------------------------
# 4. 评测：块的排名 → 节的排名 → recall@k、MRR
# ---------------------------------------------------------------------------
def section_ranking(scores, chunks):
    seen, out = set(), []
    for i in np.argsort(-scores):
        if chunks[i]["sec"] not in seen:
            seen.add(chunks[i]["sec"]); out.append(chunks[i]["sec"])
    return out


def is_hit(sec, item):
    if sec in item["gold"]:
        return True
    head = sec.split(".")[0]
    return head.isdigit() and int(head) in item["chapters"]


def evaluate(score_fn, items, chunks, ks=(1, 3, 5)):
    rec, rr, ranks = {k: [] for k in ks}, [], []
    for it in items:
        secs = section_ranking(score_fn(it["q"]), chunks)
        r = next((i + 1 for i, s in enumerate(secs) if is_hit(s, it)), None)
        ranks.append(r)
        for k in ks:
            rec[k].append(r is not None and r <= k)
        rr.append(1 / r if r else 0)
    return {f"R@{k}": np.mean(v) for k, v in rec.items()} | {"MRR": np.mean(rr)}, ranks


def pad(text, width):
    """按显示宽度补空格：汉字和全角符号占两格，否则表格对不齐。"""
    import unicodedata
    w = sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in text)
    return text + " " * max(width - w, 0)


def bootstrap_ci(xs, n=2000, seed=0):
    """第 2 章的 bootstrap：评测集只有一百来道题，分数要带区间看。"""
    rng = np.random.default_rng(seed)
    xs = np.asarray(xs, float)
    means = [xs[rng.integers(0, len(xs), len(xs))].mean() for _ in range(n)]
    return np.percentile(means, [2.5, 97.5])


if __name__ == "__main__":
    items = load_eval_set()
    chunks = load_chunks()
    print(f"评测集：{len(items)} 道 ✅ 题（其中 ★ {sum(i['star'] for i in items)} 道）；"
          f"语料：{len(chunks)} 个块，来自 {len({c['sec'] for c in chunks})} 个节")
    print(f"每题的正确节数：平均 {np.mean([len(i['gold']) or 1 for i in items]):.2f}")

    plain = [c["text"] for c in chunks]
    ctx = [c["ctx"] + "。" + c["text"] for c in chunks]          # 上下文化：块前加"出自哪一章哪一节"

    methods = {}
    bm_bi = BM25(plain, tok_bigram); methods["BM25 · 字 bigram"] = bm_bi.scores
    bm_bi_c = BM25(ctx, tok_bigram); methods["BM25 · 字 bigram · 加标题"] = bm_bi_c.scores
    try:
        import jieba, logging
        jieba.setLogLevel(logging.WARNING)
        bm_j = BM25(plain, tok_jieba); methods["BM25 · jieba"] = bm_j.scores
        bm_jc = BM25(ctx, tok_jieba); methods["BM25 · jieba · 加标题"] = bm_jc.scores
        best_tok = tok_jieba
    except ImportError:
        print("（没装 jieba，跳过按词切分的方法）")
        best_tok, bm_jc = tok_bigram, bm_bi_c
    lsa = LSA(ctx, best_tok); methods["LSA（假 embedding）· 加标题"] = lsa.scores
    methods["混合 RRF：BM25 + LSA · 加标题"] = lambda q: rrf(bm_jc.scores(q), lsa.scores(q))

    print(f"\n{pad('方法', 36)}{'R@1':>7}{'R@3':>7}{'R@5':>7}{'MRR':>7}")
    all_ranks = {}
    for name, fn in methods.items():
        m, ranks = evaluate(fn, items, chunks)
        all_ranks[name] = ranks
        print(pad(name, 36) + "".join(f"{m[k]:>7.2f}" for k in ("R@1", "R@3", "R@5", "MRR")))

    best = max(all_ranks, key=lambda n: np.mean([1 / r if r else 0 for r in all_ranks[n]]))
    hits5 = [r is not None and r <= 5 for r in all_ranks[best]]
    lo, hi = bootstrap_ci(hits5)
    print(f"\n最好的方法：{best}；R@5 = {np.mean(hits5):.2f}，95% 区间 [{lo:.2f}, {hi:.2f}]")

    star = [h for h, it in zip(hits5, items) if it["star"]]
    print(f"只看 ★ 题：R@5 = {np.mean(star):.2f}（{len(star)} 道）")

    print("\n最好的方法在前 5 名里也没找到正确节的题（正确节 / 实际排第几）：")
    fails = [(it, r) for it, r in zip(items, all_ranks[best]) if not (r and r <= 5)]
    for it, r in fails[:12]:
        gold = "、".join(sorted(it["gold"])) or "第" + "、".join(map(str, sorted(it["chapters"]))) + "章"
        print(f"  [{gold} | 第 {r} 名] {it['q'][:46]}")
    print(f"  ……共 {len(fails)} 道")
