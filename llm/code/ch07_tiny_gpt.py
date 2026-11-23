r"""
第 7 章 · 从零实现并训练一个小 GPT（llm/ch07-tiny-gpt.md）。只依赖 numpy：python3 ch07_tiny_gpt.py

全部手写，不用任何深度学习框架：
1. Byte-level BPE：在本书的正文上训练 tokenizer，词表 512。
2. 模型：token embedding → L 个 Transformer 块（RMSNorm、带 RoPE 和 GQA 的 causal self-attention、SwiGLU FFN、
   residual connection）→ 最后的 RMSNorm → 和 embedding 共享权重的输出层。结构和 Llama 一样，只是小得多。
3. 反向传播：每一层的梯度都手推手写，再用有限差分做 gradient check。
4. 训练：AdamW、warmup + cosine 学习率、梯度裁剪。
5. 生成：temperature + top-k 采样；KV cache 和"每步重算全部"的结果一致、速度对比。
6. 去掉 causal mask 会怎样。
7. 参数量与计算量；MoE 的路由与负载均衡；RoPE 在长上下文上的外推。

CPU 上约一到两分钟。
"""
import math, re, time
from pathlib import Path
import numpy as np

BOOK = Path(__file__).resolve().parent.parent
rng = np.random.default_rng(0)


# ===========================================================================
# 1. Byte-level BPE（7.2 节）
# ===========================================================================
SNAPSHOT = Path(__file__).resolve().parent / "data" / "ch07_corpus.txt"


def load_corpus(max_bytes=400_000, rebuild=False):
    """语料：本书第 1–6、8–17 章的正文（去掉代码块、公式和 Markdown 符号）。
    书会更新，为了让结果可复现，默认读冻结的快照 data/ch07_corpus.txt；加 --rebuild-corpus 才从书重新生成。"""
    if SNAPSHOT.exists() and not rebuild:
        return SNAPSHOT.read_bytes()[:max_bytes]
    parts = []
    for p in sorted(BOOK.glob("ch*.md")):
        n = int(p.name[2:4])
        if n in (7, 18):
            continue
        t = p.read_text(encoding="utf-8")
        t = re.sub(r"```.*?```", "", t, flags=re.S)
        t = re.sub(r"\$\$.*?\$\$", "", t, flags=re.S)
        t = re.sub(r"\$[^$\n]+\$", "", t)
        t = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", t)
        t = re.sub(r"[*#>|`]", "", t)
        t = re.sub(r"\n\s*\n+", "\n", t)
        parts.append(t)
    data = "\n".join(parts).encode("utf-8")[:max_bytes]
    SNAPSHOT.parent.mkdir(exist_ok=True)
    SNAPSHOT.write_bytes(data)
    return data


def merge(ids, pair, new_id):
    """把序列里所有不重叠的相邻 pair 换成 new_id（从左到右贪心）。"""
    a, b = pair
    pos = np.nonzero((ids[:-1] == a) & (ids[1:] == b))[0]
    if len(pos) == 0:
        return ids
    if a == b:                                  # "aaa" 只能合并成 [aa, a]：去掉和前一个重叠的位置
        keep, last = [], -2
        for p in pos:
            if p != last + 1:
                keep.append(p); last = p
        pos = np.array(keep)
    out = ids.copy()
    out[pos] = new_id
    drop = np.zeros(len(ids), bool); drop[pos + 1] = True
    return out[~drop]


class BPE:
    def __init__(self, data: bytes, vocab_size=512):
        ids = np.frombuffer(data, dtype=np.uint8).astype(np.int32)
        self.merges = []                                     # [(a, b)]，第 k 个合并产生 id 256+k
        self.vocab = {i: bytes([i]) for i in range(256)}
        for new_id in range(256, vocab_size):
            codes = ids[:-1].astype(np.int64) * vocab_size + ids[1:]
            vals, counts = np.unique(codes, return_counts=True)
            best = vals[np.argmax(counts)]
            pair = (int(best // vocab_size), int(best % vocab_size))
            ids = merge(ids, pair, new_id)
            self.merges.append(pair)
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
        self.train_ids = ids

    def encode(self, text: str):
        ids = np.frombuffer(text.encode("utf-8"), dtype=np.uint8).astype(np.int32)
        for k, pair in enumerate(self.merges):               # 按训练时的顺序依次合并
            ids = merge(ids, pair, 256 + k)
        return ids

    def decode(self, ids):
        return b"".join(self.vocab[int(i)] for i in ids).decode("utf-8", errors="replace")


# ===========================================================================
# 2. 模型（7.3 节）与反向传播（7.4 节）
# ===========================================================================
class Config:
    def __init__(self, V=512, d=64, L=3, H=4, Hkv=2, f=176, T=128, base=10000.0, eps=1e-5):
        self.V, self.d, self.L, self.H, self.Hkv, self.f, self.T, self.base, self.eps = V, d, L, H, Hkv, f, T, base, eps
        self.hd = d // H
        self.G = H // Hkv                                     # 每组几个 query head 共用一组 K、V


def init_params(c, dtype=np.float32, seed=0):
    r = np.random.default_rng(seed)
    s, so = 0.02, 0.02 / math.sqrt(2 * c.L)                  # 写回残差流的矩阵初始化小一些
    p = {"wte": r.normal(0, s, (c.V, c.d))}
    for l in range(c.L):
        p[f"{l}.g1"] = np.ones(c.d)
        p[f"{l}.Wq"] = r.normal(0, s, (c.d, c.H * c.hd))
        p[f"{l}.Wk"] = r.normal(0, s, (c.d, c.Hkv * c.hd))
        p[f"{l}.Wv"] = r.normal(0, s, (c.d, c.Hkv * c.hd))
        p[f"{l}.Wo"] = r.normal(0, so, (c.H * c.hd, c.d))
        p[f"{l}.g2"] = np.ones(c.d)
        p[f"{l}.Wg"] = r.normal(0, s, (c.d, c.f))
        p[f"{l}.Wu"] = r.normal(0, s, (c.d, c.f))
        p[f"{l}.Wd"] = r.normal(0, so, (c.f, c.d))
    p["gf"] = np.ones(c.d)
    return {k: v.astype(dtype) for k, v in p.items()}


def rope_tables(c, n_pos, dtype):
    half = c.hd // 2
    theta = c.base ** (-np.arange(half) / half)              # θ_i = base^(-2i/hd)
    ang = np.arange(n_pos)[:, None] * theta[None, :]         # (n_pos, hd/2)
    return np.cos(ang).astype(dtype), np.sin(ang).astype(dtype)


def rope(x, cos, sin):
    """x: (..., T, hd)。前一半和后一半配成一对，旋转 pos·θ_i。"""
    h = x.shape[-1] // 2
    x1, x2 = x[..., :h], x[..., h:]
    return np.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)


def rope_back(dy, cos, sin):
    """旋转矩阵是正交的，反向就是转回去：乘 R^T。"""
    h = dy.shape[-1] // 2
    d1, d2 = dy[..., :h], dy[..., h:]
    return np.concatenate([d1 * cos + d2 * sin, -d1 * sin + d2 * cos], axis=-1)


def rmsnorm(x, g, eps):
    r = 1.0 / np.sqrt((x * x).mean(-1, keepdims=True) + eps)
    return x * r * g, (x, r, g)


def rmsnorm_back(dy, cache):
    x, r, g = cache
    u = dy * g
    dg = (dy * x * r).reshape(-1, x.shape[-1]).sum(0)
    dx = u * r - x * r ** 3 * (u * x).mean(-1, keepdims=True)
    return dx, dg


def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def forward(p, c, idx, targets=None, causal=True):
    """idx: (B, T) 的 token id。返回 logits，以及（给了 targets 时）损失和反向要用的缓存。"""
    B, T = idx.shape
    dt = p["wte"].dtype
    cos, sin = rope_tables(c, T, dt)
    mask = np.triu(np.full((T, T), -1e9, dtype=dt), 1) if causal else np.zeros((T, T), dt)
    x = p["wte"][idx]                                         # (B, T, d)
    caches = []
    for l in range(c.L):
        # ---- attention 子层 ----
        n1, cn1 = rmsnorm(x, p[f"{l}.g1"], c.eps)
        q = (n1 @ p[f"{l}.Wq"]).reshape(B, T, c.H, c.hd).transpose(0, 2, 1, 3)
        k = (n1 @ p[f"{l}.Wk"]).reshape(B, T, c.Hkv, c.hd).transpose(0, 2, 1, 3)
        v = (n1 @ p[f"{l}.Wv"]).reshape(B, T, c.Hkv, c.hd).transpose(0, 2, 1, 3)
        q, k = rope(q, cos, sin), rope(k, cos, sin)
        kr, vr = np.repeat(k, c.G, axis=1), np.repeat(v, c.G, axis=1)   # GQA：把 K、V 复制给组里每个 head
        s = q @ kr.transpose(0, 1, 3, 2) / math.sqrt(c.hd) + mask        # (B, H, T, T)
        P = softmax(s)
        o = (P @ vr).transpose(0, 2, 1, 3).reshape(B, T, c.H * c.hd)
        x = x + o @ p[f"{l}.Wo"]
        # ---- SwiGLU FFN 子层 ----
        n2, cn2 = rmsnorm(x, p[f"{l}.g2"], c.eps)
        gt, ut = n2 @ p[f"{l}.Wg"], n2 @ p[f"{l}.Wu"]
        sg = 1 / (1 + np.exp(-gt))
        silu = gt * sg
        h = silu * ut
        x = x + h @ p[f"{l}.Wd"]
        caches.append((cn1, n1, q, kr, vr, P, o, cn2, n2, gt, ut, sg, silu, h))
    xf, cf = rmsnorm(x, p["gf"], c.eps)
    logits = xf @ p["wte"].T                                  # 输出层和 embedding 共享权重
    if targets is None:
        return logits
    probs = softmax(logits)
    N = B * T
    loss = -np.log(probs.reshape(N, -1)[np.arange(N), targets.reshape(-1)] + 1e-12).mean()
    return logits, loss, (idx, targets, probs, xf, cf, caches, cos, sin)


def backward(p, c, cache):
    idx, targets, probs, xf, cf, caches, cos, sin = cache
    B, T = idx.shape
    N = B * T
    g = {k: np.zeros_like(v) for k, v in p.items()}
    # 交叉熵 + softmax：dlogits = (probs − onehot) / N
    dlog = probs.copy().reshape(N, -1)
    dlog[np.arange(N), targets.reshape(-1)] -= 1
    dlog = (dlog / N).reshape(B, T, -1)
    g["wte"] += dlog.reshape(N, -1).T @ xf.reshape(N, -1)     # 输出层那一份梯度
    dxf = dlog @ p["wte"]
    dx, g["gf"] = rmsnorm_back(dxf, cf)
    for l in reversed(range(c.L)):
        cn1, n1, q, kr, vr, P, o, cn2, n2, gt, ut, sg, silu, h = caches[l]
        # ---- SwiGLU FFN 子层：x = x + h @ Wd ----
        dm = dx
        g[f"{l}.Wd"] += h.reshape(N, -1).T @ dm.reshape(N, -1)
        dh = dm @ p[f"{l}.Wd"].T
        dsilu, dut = dh * ut, dh * silu
        dgt = dsilu * sg * (1 + gt * (1 - sg))               # SiLU'(z) = σ(z)(1 + z(1 − σ(z)))
        g[f"{l}.Wg"] += n2.reshape(N, -1).T @ dgt.reshape(N, -1)
        g[f"{l}.Wu"] += n2.reshape(N, -1).T @ dut.reshape(N, -1)
        dn2 = dgt @ p[f"{l}.Wg"].T + dut @ p[f"{l}.Wu"].T
        dxn, g[f"{l}.g2"] = rmsnorm_back(dn2, cn2)
        dx = dx + dxn                                         # residual：梯度直接穿过，再加上子层的
        # ---- attention 子层：x = x + o @ Wo ----
        da = dx
        g[f"{l}.Wo"] += o.reshape(N, -1).T @ da.reshape(N, -1)
        do = (da @ p[f"{l}.Wo"].T).reshape(B, T, c.H, c.hd).transpose(0, 2, 1, 3)
        dP = do @ vr.transpose(0, 1, 3, 2)
        dvr = P.transpose(0, 1, 3, 2) @ do
        ds = P * (dP - (dP * P).sum(-1, keepdims=True)) / math.sqrt(c.hd)   # softmax 的反向
        dq = ds @ kr
        dkr = ds.transpose(0, 1, 3, 2) @ q
        dk = dkr.reshape(B, c.Hkv, c.G, T, c.hd).sum(2)      # GQA：组里各 head 的梯度加回同一组 K、V
        dv = dvr.reshape(B, c.Hkv, c.G, T, c.hd).sum(2)
        dq, dk = rope_back(dq, cos, sin), rope_back(dk, cos, sin)
        dq = dq.transpose(0, 2, 1, 3).reshape(N, -1)
        dk = dk.transpose(0, 2, 1, 3).reshape(N, -1)
        dv = dv.transpose(0, 2, 1, 3).reshape(N, -1)
        n1f = n1.reshape(N, -1)
        g[f"{l}.Wq"] += n1f.T @ dq
        g[f"{l}.Wk"] += n1f.T @ dk
        g[f"{l}.Wv"] += n1f.T @ dv
        dn1 = (dq @ p[f"{l}.Wq"].T + dk @ p[f"{l}.Wk"].T + dv @ p[f"{l}.Wv"].T).reshape(B, T, -1)
        dxn, g[f"{l}.g1"] = rmsnorm_back(dn1, cn1)
        dx = dx + dxn
    np.add.at(g["wte"], idx, dx)                              # embedding：按 token id 把梯度加回对应的行
    return g


def gradient_check():
    print("=" * 72)
    print("gradient check：手写的反向 vs 有限差分（float64，小模型，检查每一个参数）")
    print("=" * 72)
    r = np.random.default_rng(1)
    c = Config(V=20, d=16, L=2, H=4, Hkv=2, f=24, T=6)          # Hkv=2、每组 2 个 head：GQA 的分组求和写错会被抓到
    p = init_params(c, np.float64, seed=1)
    for k in p:                                               # 加大一点的随机扰动，避免梯度太小
        p[k] = p[k] + r.normal(0, 0.3, p[k].shape)
    idx = r.integers(0, c.V, (2, c.T)); tgt = r.integers(0, c.V, (2, c.T))
    _, _, cache = forward(p, c, idx, tgt)
    g = backward(p, c, cache)
    worst, n = 0.0, 0
    for name in p:
        flat = p[name].reshape(-1)
        for j in range(flat.size):
            old = flat[j]
            flat[j] = old + 1e-5; lp = forward(p, c, idx, tgt)[1]
            flat[j] = old - 1e-5; lm = forward(p, c, idx, tgt)[1]
            flat[j] = old
            num = (lp - lm) / 2e-5
            ana = g[name].reshape(-1)[j]
            worst = max(worst, abs(num - ana) / max(1e-8, abs(num) + abs(ana)))
            n += 1
    print(f"  检查了 {len(p)} 个参数张量、全部 {n} 个元素；最大相对误差 {worst:.2e}")


# ===========================================================================
# 3. 训练（7.6 节）
# ===========================================================================
def count_params(c, tied=True):
    per_layer = c.d * c.H * c.hd + 2 * c.d * c.Hkv * c.hd + c.H * c.hd * c.d + 3 * c.d * c.f + 2 * c.d
    return c.V * c.d * (1 if tied else 2) + c.L * per_layer + c.d


def get_batch(data, c, B):
    ix = rng.integers(0, len(data) - c.T - 1, B)
    x = np.stack([data[i:i + c.T] for i in ix])
    y = np.stack([data[i + 1:i + c.T + 1] for i in ix])
    return x, y


def lr_at(step, total, peak, warmup):
    if step < warmup:
        return peak * (step + 1) / warmup
    t = (step - warmup) / max(1, total - warmup)
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t)))     # cosine 降到峰值的 10%


def train(p, c, train_data, val_data, steps, B=16, peak_lr=3e-3, warmup=100, wd=0.1, clip=1.0,
          causal=True, log_every=None, label=""):
    m = {k: np.zeros_like(v) for k, v in p.items()}
    v = {k: np.zeros_like(v) for k, v in p.items()}
    b1, b2, eps = 0.9, 0.95, 1e-8
    t0 = time.time()
    for step in range(steps):
        x, y = get_batch(train_data, c, B)
        _, loss, cache = forward(p, c, x, y, causal=causal)
        g = backward(p, c, cache)
        norm = math.sqrt(sum(float((gi * gi).sum()) for gi in g.values()))
        scale = min(1.0, clip / (norm + 1e-6))                # 梯度裁剪：按全局范数缩放
        lr = lr_at(step, steps, peak_lr, warmup)
        for k in p:
            gk = g[k] * scale
            m[k] = b1 * m[k] + (1 - b1) * gk
            v[k] = b2 * v[k] + (1 - b2) * gk * gk
            mh, vh = m[k] / (1 - b1 ** (step + 1)), v[k] / (1 - b2 ** (step + 1))
            if p[k].ndim == 2:
                p[k] -= lr * wd * p[k]                        # AdamW：权重衰减和梯度更新分开
            p[k] -= (lr * mh / (np.sqrt(vh) + eps)).astype(p[k].dtype)
        if log_every and (step % log_every == 0 or step == steps - 1):
            print(f"  {label}step {step:5d}  lr {lr:.4f}  train loss {loss:.3f}  "
                  f"val loss {evaluate(p, c, val_data):.3f}  |grad| {norm:6.2f}  ({time.time() - t0:5.1f}s)")
    return p


def evaluate(p, c, data, n=8, B=16, causal=True):
    r = np.random.default_rng(123)
    losses = []
    for _ in range(n):
        ix = r.integers(0, len(data) - c.T - 1, B)
        x = np.stack([data[i:i + c.T] for i in ix]); y = np.stack([data[i + 1:i + c.T + 1] for i in ix])
        losses.append(forward(p, c, x, y, causal=causal)[1])
    return float(np.mean(losses))


# ===========================================================================
# 4. 生成与 KV cache（7.7 节）
# ===========================================================================
def sample_next(logits, temperature=0.8, top_k=40, r=None):
    z = logits / temperature
    if top_k:
        kth = np.sort(z)[-top_k]
        z = np.where(z < kth, -np.inf, z)
    pr = softmax(z)
    return int((r or rng).choice(len(pr), p=pr))


class KVCache:
    def __init__(self, c, dtype):
        self.k = [np.zeros((1, c.Hkv, 0, c.hd), dtype) for _ in range(c.L)]
        self.v = [np.zeros((1, c.Hkv, 0, c.hd), dtype) for _ in range(c.L)]


def step_with_cache(p, c, token, pos, kv, cos, sin):
    """只算新 token 这一个位置：它的 q 去看缓存里所有的 K、V；把它自己的 K、V 追加进缓存。"""
    x = p["wte"][[[token]]]                                   # (1, 1, d)
    cs, sn = cos[pos:pos + 1], sin[pos:pos + 1]
    for l in range(c.L):
        n1, _ = rmsnorm(x, p[f"{l}.g1"], c.eps)
        q = (n1 @ p[f"{l}.Wq"]).reshape(1, 1, c.H, c.hd).transpose(0, 2, 1, 3)
        k = (n1 @ p[f"{l}.Wk"]).reshape(1, 1, c.Hkv, c.hd).transpose(0, 2, 1, 3)
        v = (n1 @ p[f"{l}.Wv"]).reshape(1, 1, c.Hkv, c.hd).transpose(0, 2, 1, 3)
        q, k = rope(q, cs, sn), rope(k, cs, sn)
        kv.k[l] = np.concatenate([kv.k[l], k], axis=2)
        kv.v[l] = np.concatenate([kv.v[l], v], axis=2)
        kr, vr = np.repeat(kv.k[l], c.G, axis=1), np.repeat(kv.v[l], c.G, axis=1)
        P = softmax(q @ kr.transpose(0, 1, 3, 2) / math.sqrt(c.hd))   # 不需要 mask：缓存里只有过去
        o = (P @ vr).transpose(0, 2, 1, 3).reshape(1, 1, -1)
        x = x + o @ p[f"{l}.Wo"]
        n2, _ = rmsnorm(x, p[f"{l}.g2"], c.eps)
        gt = n2 @ p[f"{l}.Wg"]
        x = x + (gt / (1 + np.exp(-gt)) * (n2 @ p[f"{l}.Wu"])) @ p[f"{l}.Wd"]
    xf, _ = rmsnorm(x, p["gf"], c.eps)
    return (xf @ p["wte"].T)[0, 0]


def generate(p, c, prompt_ids, n_new, use_cache=True, seed=7):
    r = np.random.default_rng(seed)
    ids = list(prompt_ids)
    if use_cache:
        cos, sin = rope_tables(c, c.T, p["wte"].dtype)
        kv = KVCache(c, p["wte"].dtype)
        for pos, t in enumerate(ids):                         # 先把提示逐个喂进去（真实系统里是一次 prefill）
            logits = step_with_cache(p, c, t, pos, kv, cos, sin)
        for _ in range(n_new):
            if len(ids) >= c.T:
                break
            nxt = sample_next(logits, r=r); ids.append(nxt)
            logits = step_with_cache(p, c, nxt, len(ids) - 1, kv, cos, sin)
    else:
        for _ in range(n_new):
            if len(ids) >= c.T:
                break
            logits = forward(p, c, np.array([ids]))[0, -1]   # 每一步把整段重新算一遍
            ids.append(sample_next(logits, r=r))
    return ids


# ===========================================================================
# 5. MoE 与长上下文（7.8 节）
# ===========================================================================
def moe_routing_demo():
    print("\n" + "=" * 72)
    print("MoE 路由：8 个 expert，每个 token 选 top-2，capacity factor 1.25")
    print("=" * 72)
    N, E, k, d = 8192, 8, 2, 64
    r = np.random.default_rng(30)
    X = r.normal(0, 1, (N, d))
    W = r.normal(0, 0.1, (d, E))
    for name, bias in [("均衡的 router", np.zeros(E)), ("偏向前两个 expert 的 router", np.array([1.5, 1.0, 0, 0, 0, 0, 0, 0]))]:
        probs = softmax(X @ W + bias)
        top = np.argsort(-probs, axis=1)[:, :k]
        counts = np.bincount(top.reshape(-1), minlength=E)
        f = counts / (N * k)                                   # 每个 expert 分到的 token 比例
        Pm = probs.mean(0)                                     # router 给每个 expert 的平均概率
        aux = E * (f * Pm).sum()                               # Switch Transformer 的负载均衡损失，均衡时 = 1
        cap = int(1.25 * N * k / E)
        dropped = np.maximum(0, counts - cap).sum() / (N * k)
        print(f"  {name:<20} 各 expert 的负载 {np.round(f * E, 2)}  aux loss {aux:.2f}  超出容量被丢弃 {dropped:.1%}")
    print('  （负载以"平均值 = 1"为单位）')


def rope_extrapolation_demo():
    print("\n" + "=" * 72)
    print("RoPE 外推：head 维度 128，base 10000，训练长度 4096")
    print("=" * 72)
    hd, base, Ltr = 128, 10000.0, 4096
    i = np.arange(hd // 2)
    theta = base ** (-2 * i / hd)
    seen = Ltr * theta                                         # 训练时每一对见过的角度范围（位置 0 到 Ltr）
    full = seen >= 2 * np.pi                                   # 训练中转满过一圈：所有角度都见过

    def unseen_frac(th, Lnew):
        return np.mean(((Lnew - 1) * th > seen) & ~full)

    def overshoot(th, Lnew):                                   # 没转满一圈的维度里，新的角度范围最多是训练时的几倍
        return ((Lnew - 1) * th / seen)[~full].max()

    print(f"  训练中转满过一圈的维度对：{full.mean():.0%}；其余 {1 - full.mean():.0%} 只见过一段角度")
    print(f"  {'方法':<22}{'扩到':>6}{'含没见过角度的维度对':>18}{'最多超出几倍':>12}{'最高频一对的相邻角度差':>20}")
    for s in (2, 4, 8):
        Lnew = s * Ltr
        rows = [("直接外推", theta),
                ("Position Interpolation", theta / s)]
        base_ntk = base * s ** (hd / (hd - 2))                 # NTK-aware：改 base
        rows.append(("NTK-aware", base_ntk ** (-2 * i / hd)))
        rot = Ltr * theta / (2 * np.pi)                        # YaRN：训练中转了几圈
        ramp = np.clip((rot - 1) / (32 - 1), 0, 1)             # <1 圈：完全插值；>32 圈：不动；之间线性过渡
        rows.append(("YaRN", theta * (ramp + (1 - ramp) / s)))
        for name, th in rows:
            print(f"  {name:<22}{s:>5}x{unseen_frac(th, Lnew):>16.0%}{overshoot(th, Lnew):>14.2f}{th[0]:>18.3f} rad")


# ===========================================================================
if __name__ == "__main__":
    import sys
    t_start = time.time()
    gradient_check()

    print("\n" + "=" * 72)
    print("Byte-level BPE：在本书正文上训练，词表 512")
    print("=" * 72)
    data = load_corpus(rebuild="--rebuild-corpus" in sys.argv)
    t0 = time.time()
    tok = BPE(data, 512)
    ids = tok.train_ids
    print(f"  语料 {len(data):,} 字节 → {len(ids):,} 个 token，平均每个 token {len(data) / len(ids):.2f} 字节（{time.time() - t0:.1f}s）")
    def show(b):                                              # 不是完整 UTF-8 字符的片段，用十六进制显示
        try:
            return b.decode("utf-8").replace("\n", "⏎")
        except UnicodeDecodeError:
            return "<" + b.hex() + ">"
    longest = sorted(tok.vocab.values(), key=len, reverse=True)[:10]
    print("  最长的几个 token：" + " | ".join(show(b) for b in longest))
    for s in ["梯度下降的学习率", "KV cache 的显存", "Transformer"]:
        e = tok.encode(s)
        print(f"  {s!r:<22} → {len(s.encode('utf-8'))} 字节、{len(e)} 个 token：" +
              " | ".join(show(tok.vocab[int(i)]) for i in e))
    assert tok.decode(tok.encode("测试 round trip：αβγ")) == "测试 round trip：αβγ"

    rng = np.random.default_rng(10)                           # 每个实验用自己的随机种子，互不影响
    n = int(0.9 * len(ids))
    train_data, val_data = ids[:n], ids[n:]
    c = Config()
    p = init_params(c)
    n_params = sum(v.size for v in p.values())
    assert n_params == count_params(c)
    print("\n" + "=" * 72)
    print(f"模型：V={c.V} d={c.d} L={c.L} H={c.H} Hkv={c.Hkv} d_ffn={c.f} 上下文 {c.T}；参数 {n_params:,}")
    print("=" * 72)
    llama = Config(V=128256, d=4096, L=32, H=32, Hkv=8, f=14336)
    N8 = count_params(llama, tied=False)
    print(f"  同一个公式算 Llama-3-8B（输入、输出矩阵不共享）：{N8 / 1e9:.2f}B；每个训练 token 约 6N = {6 * N8 / 1e9:.0f} GFLOP")
    print(f"  初始损失 {evaluate(p, c, val_data):.3f}，均匀猜测是 ln {c.V} = {math.log(c.V):.3f}")

    prompt = tok.encode("注意力")
    print("  训练前生成：" + tok.decode(generate(p, c, prompt, 40)).replace("\n", " "))
    p = train(p, c, train_data, val_data, steps=800, log_every=100)
    vl = evaluate(p, c, val_data)
    print(f"  训练后 val loss {vl:.3f}，perplexity {math.exp(vl):.1f}；"
          f"每字节 {vl / math.log(2) * len(ids) / len(data):.2f} bit")
    for s in ["注意力", "梯度下降", "KV cache"]:
        print(f"  生成（提示 {s!r}）：" + tok.decode(generate(p, c, tok.encode(s), 60)).replace("\n", " "))

    print("\n" + "=" * 72)
    print("KV cache：和每步重算全部的结果一致吗？快多少？")
    print("=" * 72)
    seq = val_data[:100]
    full_logits = forward(p, c, seq[None, :])[0]
    cos, sin = rope_tables(c, c.T, p["wte"].dtype)
    kv = KVCache(c, p["wte"].dtype)
    inc = np.stack([step_with_cache(p, c, int(t), i, kv, cos, sin) for i, t in enumerate(seq)])
    print(f"  100 个位置的 logits，最大差异 {np.abs(full_logits - inc).max():.2e}（float32 舍入误差量级）")
    same = generate(p, c, prompt, 120, use_cache=True, seed=3) == generate(p, c, prompt, 120, use_cache=False, seed=3)
    c_long = Config(T=1024)                                   # 同一组权重，只是允许更长的上下文，用来计时
    for n_new in (128, 512, 1000):
        times = []
        for use in (False, True):
            t0 = time.time(); generate(p, c_long, prompt, n_new, use_cache=use, seed=3); times.append(time.time() - t0)
        print(f"  生成 {n_new:4d} 个 token：每步重算 {times[0]:6.2f}s，用 KV cache {times[1]:5.2f}s，快 {times[0] / times[1]:4.1f} 倍")
    print(f"  两种方式采样出的序列完全相同：{same}")

    print("\n" + "=" * 72)
    print("去掉 causal mask：模型能看见答案")
    print("=" * 72)
    rng = np.random.default_rng(20)
    q = init_params(c, seed=5)
    q = train(q, c, train_data, val_data, steps=200, causal=False)
    print(f"  训练 200 步。不带 mask 评估（训练时的样子）loss {evaluate(q, c, val_data, causal=False):.3f}；"
          f"带 mask 评估（生成时的样子）loss {evaluate(q, c, val_data, causal=True):.3f}")
    print("  生成：" + tok.decode(generate(q, c, prompt, 40)).replace("\n", " "))

    moe_routing_demo()
    rope_extrapolation_demo()
    print(f"\n总耗时 {time.time() - t_start:.0f}s")
