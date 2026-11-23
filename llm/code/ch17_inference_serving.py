r"""
第 17 章 · 推理服务的配套代码（llm/ch17-inference-serving.md）。只依赖 numpy，CPU 上运行：python3 ch17_inference_serving.py

需要真实 GPU 的实验在 ch17_gpu_benchmark.py（在 Kaggle / Colab / RunPod 上运行）。这里做的是：
1. 显存账本：权重、KV cache、一张卡能同时服务多少个请求。
2. Roofline 模型：decode 一步要多久、批大小怎样决定吞吐和延迟。
3. 在本机 CPU 上实测：矩阵 × 向量 vs 矩阵 × 矩阵，看"内存受限"长什么样。
4. 静态批处理 vs 连续批处理（continuous batching）。
5. KV cache 的分配：预留最大长度 vs 分页（PagedAttention）。
6. 前缀缓存：共享系统提示能省多少。
7. 投机解码：输出分布不变的证明做数值检验；加速比与接受率。
8. 量化：INT8 / INT4、按张量 / 按通道 / 分组，离群值的影响。
9. 排队：负载接近满载时，延迟怎样爆炸（M/G/1 与 Pollaczek–Khinchine 公式）。
"""
import time
import numpy as np

rng = np.random.default_rng(0)
GiB = 2 ** 30

# ---------------------------------------------------------------------------
# 模型与 GPU 的参数
# ---------------------------------------------------------------------------
# Llama 3 系列的公开配置：层数、隐藏维度、注意力头数、KV 头数（GQA）、头维度、FFN 维度、词表
MODELS = {
    "Llama-3-8B":  dict(L=32, d=4096, heads=32, kv_heads=8, hd=128, ffn=14336, vocab=128256),
    "Llama-3-70B": dict(L=80, d=8192, heads=64, kv_heads=8, hd=128, ffn=28672, vocab=128256),
}
# 显存容量（GiB 近似）、带宽（字节/秒）、BF16/FP16 稠密算力（FLOP/秒）
GPUS = {
    "H100 SXM":  dict(mem=80, bw=3.35e12, flops=989e12),
    "A100 80GB": dict(mem=80, bw=2.04e12, flops=312e12),
    "RTX 4090":  dict(mem=24, bw=1.008e12, flops=165e12),
    "T4":        dict(mem=16, bw=0.32e12, flops=65e12),
}


def n_params(m):
    """参数量：每层 = 注意力（Q、O 是 d×d，K、V 是 d×kv_dim）+ SwiGLU 前馈（3 个 d×ffn）；加上输入和输出 embedding。"""
    kv_dim = m["kv_heads"] * m["hd"]
    per_layer = 2 * m["d"] * m["d"] + 2 * m["d"] * kv_dim + 3 * m["d"] * m["ffn"]
    return m["L"] * per_layer + 2 * m["vocab"] * m["d"]


def kv_bytes_per_token(m, bytes_per=2):
    """每个 token 的 KV cache：K 和 V 各一份 × 层数 × KV 头数 × 头维度 × 每个数的字节数。"""
    return 2 * m["L"] * m["kv_heads"] * m["hd"] * bytes_per


print("=== 1. 显存账本 ===")
for name, m in MODELS.items():
    N = n_params(m)
    kv = kv_bytes_per_token(m)
    kv_mha = kv * m["heads"] // m["kv_heads"]
    print(f"  {name}: 参数 {N / 1e9:.2f}B，BF16 权重 {2 * N / GiB:.1f} GiB；"
          f"KV cache 每 token {kv / 1024:.0f} KiB（不用 GQA 会是 {kv_mha / 1024:.0f} KiB）")
m8 = MODELS["Llama-3-8B"]; N8 = n_params(m8); kv8 = kv_bytes_per_token(m8)
print("  Llama-3-8B（BF16）放在不同的卡上，留 10% 给激活和框架：")
for g, spec in GPUS.items():
    free = spec["mem"] * GiB * 0.9 - 2 * N8
    if free <= 0:
        print(f"    {g:<10} 放不下权重"); continue
    toks = free / kv8
    print(f"    {g:<10} KV 可用 {free / GiB:5.1f} GiB → 共 {toks / 1e3:6.0f}k token → 每个请求 4k 上下文时最多 {int(toks // 4096):>3} 个并发")

# ---------------------------------------------------------------------------
# 2. Roofline
# ---------------------------------------------------------------------------
print("\n=== 2. Roofline：decode 一步的时间与吞吐（Llama-3-8B，BF16，上下文 2048） ===")


def decode_step(m, gpu, B, T, wbytes=2, kvbytes=2):
    """一步 decode（批里每个序列各生成 1 个 token）的时间下界 = max(读数据的时间, 计算的时间)。"""
    N = n_params(m)
    bytes_moved = N * wbytes + B * T * kv_bytes_per_token(m, kvbytes)   # 权重读一次，KV cache 每个序列各读一遍
    flops = 2 * N * B + 4 * m["L"] * m["heads"] * m["hd"] * T * B      # 矩阵乘 + 注意力（QKᵀ 和 AV）
    t_mem, t_cmp = bytes_moved / gpu["bw"], flops / gpu["flops"]
    return max(t_mem, t_cmp), t_mem, t_cmp


for g in ["H100 SXM", "RTX 4090"]:
    spec = GPUS[g]
    cap_tokens = (spec["mem"] * GiB * 0.9 - 2 * N8) / kv8
    print(f"  {g}（脊点 = 算力/带宽 = {spec['flops'] / spec['bw']:.0f} FLOP/字节）")
    print("     批大小   每步(ms)   读数据(ms)  计算(ms)   吞吐(token/s)  受限于   显存放得下吗")
    for B in [1, 8, 32, 64, 128, 256]:
        t, tm, tc = decode_step(m8, spec, B, 2048)
        fits = "是" if B * 2048 <= cap_tokens else "否"
        print(f"     {B:>5}   {t * 1e3:8.2f}   {tm * 1e3:9.2f}  {tc * 1e3:8.2f}   {B / t:>12,.0f}  {'带宽' if tm > tc else '算力'}     {fits}")


def crossover_batch(m, gpu, T):
    """解 读数据时间 = 计算时间，得到进入算力受限所需的批大小；分母 ≤ 0 表示永远受带宽限制。"""
    N = n_params(m)
    denom = (2 * N + 4 * m["L"] * m["heads"] * m["hd"] * T) / gpu["flops"] - T * kv_bytes_per_token(m) / gpu["bw"]
    return (2 * N / gpu["bw"]) / denom if denom > 0 else float("inf")


print("  H100 上进入算力受限需要的批大小（随上下文长度）：",
      "，".join(f"T={T}: {crossover_batch(m8, GPUS['H100 SXM'], T):.0f}" for T in [0, 128, 512, 1024, 2048]))
t1, *_ = decode_step(m8, GPUS["H100 SXM"], 1, 2048)
print(f"  单个请求在 H100 上：每 token 至少 {t1 * 1e3:.1f} ms，即每秒最多约 {1 / t1:.0f} 个 token —— 由读一遍权重的时间决定")
tp, *_ = decode_step(m8, GPUS["H100 SXM"], 1, 0)
prefill_flops = 2 * N8 * 2048
print(f"  prefill 一个 2048 token 的提示：约 {prefill_flops / 1e12:.0f} TFLOP，在 H100 上至少 {prefill_flops / 989e12 * 1e3:.0f} ms（受限于算力）")

# ---------------------------------------------------------------------------
# 3. 在本机 CPU 上实测
# ---------------------------------------------------------------------------
print("\n=== 3. 本机 CPU 实测：W(4096×4096, FP32) × X(4096×B) ===")
W = rng.standard_normal((4096, 4096)).astype(np.float32)
print("    B     每次(ms)   每列(ms)   GFLOP/s   有效带宽(GB/s)")
base = None
for B in [1, 4, 16, 64, 256]:
    X = rng.standard_normal((4096, B)).astype(np.float32)
    W @ X
    reps = 20 if B < 64 else 5
    t0 = time.perf_counter()
    for _ in range(reps): W @ X
    t = (time.perf_counter() - t0) / reps
    base = base or t
    print(f"  {B:>4}  {t * 1e3:9.2f}  {t / B * 1e3:9.3f}  {2 * W.size * B / t / 1e9:8.1f}  {(W.nbytes + X.nbytes) / t / 1e9:10.1f}")
print("  每列（每个 token）的时间随 B 下降：权重读一次，被 B 个 token 共用。")

# ---------------------------------------------------------------------------
# 4. 静态批处理 vs 连续批处理
# ---------------------------------------------------------------------------
print("\n=== 4. 静态批处理 vs 连续批处理（1000 个请求，批容量 32） ===")
n_req, cap = 1000, 32
out_len = np.clip(rng.lognormal(np.log(120), 0.9, n_req), 5, 2000).astype(int)   # 输出长度：中位数 120，长尾


def step_time(active):                      # 一步的时间：权重读取是固定开销，每多一个序列加一点 KV 读取
    return 1.0 + 0.02 * active


def static_batching():
    t, done_at, busy_slots = 0.0, np.zeros(n_req), 0
    for s in range(0, n_req, cap):
        batch = out_len[s:s + cap]
        for k in range(batch.max()):        # 整批要等最长的那个结束
            active = int((batch > k).sum())
            t += step_time(len(batch))      # 已结束的序列还占着位置
            busy_slots += active
        done_at[s:s + cap] = t
    return t, done_at, busy_slots / (t / step_time(cap) * cap)


def continuous_batching():
    t, queue, active, done_at, k = 0.0, list(range(n_req)), {}, np.zeros(n_req), 0
    busy, steps = 0, 0
    while queue or active:
        while queue and len(active) < cap:  # 每一步都把空出来的位置补上
            i = queue.pop(0); active[i] = out_len[i]
        t += step_time(len(active)); steps += 1; busy += len(active)
        for i in list(active):
            active[i] -= 1
            if active[i] == 0:
                done_at[i] = t; del active[i]
    return t, done_at, busy / (steps * cap)


for name, fn in [("静态批处理", static_batching), ("连续批处理", continuous_batching)]:
    total, done, util = fn()
    print(f"  {name}：总时间 {total:8.0f}，吞吐 {out_len.sum() / total:5.1f} token/单位时间，"
          f"平均完成时间 {done.mean():7.0f}，位置利用率 {util:.0%}")

# ---------------------------------------------------------------------------
# 5. KV cache 分配：预留 vs 分页
# ---------------------------------------------------------------------------
print("\n=== 5. KV cache：按最大长度预留 vs 分页（PagedAttention） ===")
budget, max_len, block = 200_000, 4096, 16       # KV 预算 20 万 token；最大长度 4096；每页 16 token
cur_len = np.clip(rng.lognormal(np.log(700), 0.8, 100_000), 20, max_len).astype(int)   # 进行中请求的当前长度
n_res = budget // max_len
used_res = cur_len[:n_res].sum()
pages = np.ceil(cur_len / block).astype(int)
n_paged = int(np.searchsorted(np.cumsum(pages * block), budget))
waste_paged = (pages[:n_paged] * block - cur_len[:n_paged]).sum()
print(f"  预留：最多 {n_res} 个并发，真正存了 token 的只占 {used_res / budget:.0%}，其余 {1 - used_res / budget:.0%} 是空着的预留")
print(f"  分页：最多 {n_paged} 个并发（{n_paged / n_res:.1f} 倍），浪费只来自每个请求最后一页没填满：{waste_paged / budget:.1%}")

# ---------------------------------------------------------------------------
# 6. 前缀缓存
# ---------------------------------------------------------------------------
print("\n=== 6. 前缀缓存：100 个请求共享 3000 token 的系统提示，各自再有 200 token ===")
S, U, R = 3000, 200, 100
no_cache = R * (S + U); with_cache = S + R * U
print(f"  prefill 要算的 token：不共享 {no_cache:,}，共享 {with_cache:,}（省 {1 - with_cache / no_cache:.0%}）；"
      f"KV cache 同样只存一份前缀")

# ---------------------------------------------------------------------------
# 7. 投机解码
# ---------------------------------------------------------------------------
print("\n=== 7. 投机解码 ===")
V = 40
P = rng.dirichlet(np.full(V, 0.3), size=V)                     # 目标模型：一个 V 状态的马尔可夫链 p(x_t | x_{t-1})
Q = 0.7 * P + 0.3 * rng.dirichlet(np.full(V, 1.0), size=V)     # 草稿模型：目标的粗糙近似


def speculative(n_tokens, k):
    """用草稿模型一次猜 k 个，目标模型一次验证。返回序列、目标模型的调用次数、每个位置是否被接受。"""
    seq, calls, accepts = [0], 0, []
    while len(seq) < n_tokens + 1:
        ctx = seq[-1]; drafts, qs = [], []
        for _ in range(k):                                     # 草稿模型自回归地猜 k 个
            q = Q[ctx]; x = rng.choice(V, p=q); drafts.append(x); qs.append(q); ctx = x
        calls += 1                                             # 目标模型一次前向，算出这 k+1 个位置的分布
        ctx = seq[-1]
        for x, q in zip(drafts, qs):
            p = P[ctx]
            if rng.random() < min(1.0, p[x] / q[x]):          # 以 min(1, p/q) 的概率接受
                seq.append(x); ctx = x; accepts.append(1)
            else:                                              # 拒绝：从 max(0, p−q) 归一化后的分布里重新采样
                r = np.maximum(p - q, 0); r /= r.sum()
                x = rng.choice(V, p=r); seq.append(x); accepts.append(0); break
        else:                                                  # k 个全被接受：目标模型顺便多给一个 token
            seq.append(rng.choice(V, p=P[ctx]))
    return np.array(seq[1:n_tokens + 1]), calls, np.array(accepts)


def plain(n_tokens):
    seq = [0]
    for _ in range(n_tokens): seq.append(rng.choice(V, p=P[seq[-1]]))
    return np.array(seq[1:])


def bigram_freq(s):
    M = np.zeros((V, V)); np.add.at(M, (s[:-1], s[1:]), 1)
    return M / M.sum()


n = 60_000
ref = plain(n)
for k in [2, 4, 8]:
    s, calls, acc = speculative(n, k)
    alpha = acc.mean()
    tv = 0.5 * np.abs(bigram_freq(s) - bigram_freq(ref)).sum()
    tv_ref = 0.5 * np.abs(bigram_freq(plain(n)) - bigram_freq(ref)).sum()      # 两次普通采样之间的差距，作为噪声基准
    print(f"  k={k}: 接受率 α={alpha:.3f}；每次目标模型调用产出 {n / calls:.2f} 个 token"
          f"（i.i.d. 公式 (1−α^(k+1))/(1−α) = {(1 - alpha ** (k + 1)) / (1 - alpha):.2f}）；"
          f"二元组分布与直接采样的总变差 {tv:.4f}（两次直接采样之间 {tv_ref:.4f}）")
c = 0.05     # 草稿模型一次前向的时间 / 目标模型一次前向的时间
for k in [2, 4, 8]:
    _, calls, acc = speculative(20_000, k)
    tok_per_call = 20_000 / calls
    print(f"  k={k}，草稿开销 c={c}：加速比 ≈ {tok_per_call / (1 + k * c):.2f} 倍")

# ---------------------------------------------------------------------------
# 8. 量化
# ---------------------------------------------------------------------------
print("\n=== 8. 权重量化（1024×1024，有 1% 的离群输入通道） ===")
Wq = rng.standard_normal((1024, 1024)) * 0.02
outlier_cols = rng.choice(1024, 10, replace=False)
Wq[:, outlier_cols] *= 25                                       # 少数输入通道的数值大得多
Xq = rng.standard_normal((1024, 64))


def quantize(Wm, bits, axis=None, group=None):
    """对称均匀量化：q = round(W/scale)，scale = max|W| / (2^(bits−1) − 1)。axis/group 决定 scale 的粒度。"""
    qmax = 2 ** (bits - 1) - 1
    if group:                                                   # 沿输入维度每 group 个一组
        Wg = Wm.reshape(Wm.shape[0], -1, group)
        s = np.abs(Wg).max(axis=2, keepdims=True) / qmax
        return (np.round(Wg / s).clip(-qmax, qmax) * s).reshape(Wm.shape)
    s = np.abs(Wm).max(axis=axis, keepdims=True) / qmax if axis is not None else np.abs(Wm).max() / qmax
    return np.round(Wm / s).clip(-qmax, qmax) * s


ref_out = Wq @ Xq
print("  方案                     每参数(bit)  输出相对误差")
for name, kw, bits_eff in [("INT8 按张量", dict(bits=8), 8), ("INT8 按输出通道", dict(bits=8, axis=1), 8),
                           ("INT4 按张量", dict(bits=4), 4), ("INT4 按输出通道", dict(bits=4, axis=1), 4),
                           ("INT4 分组 g=128", dict(bits=4, group=128), 4 + 16 / 128)]:
    err = np.linalg.norm(quantize(Wq, **kw) @ Xq - ref_out) / np.linalg.norm(ref_out)
    print(f"  {name:<22} {bits_eff:>8.3f}   {err:12.4f}")
keep = np.zeros(1024, bool); keep[outlier_cols] = True          # 混合精度：离群通道保留 16 位，其余 INT4 分组
W_mixed = Wq.copy()
rest = Wq[:, ~keep]; pad = (-rest.shape[1]) % 128
rest_q = quantize(np.pad(rest, ((0, 0), (0, pad))), 4, group=128)[:, :rest.shape[1]]
W_mixed[:, ~keep] = rest_q
err_mixed = np.linalg.norm(W_mixed @ Xq - ref_out) / np.linalg.norm(ref_out)
print(f"  {'离群通道留 16 位 + INT4 分组':<16} {4 + 16 / 128 + 10 * 12 / 1024:>8.3f}   {err_mixed:12.4f}")
W_clean = rng.standard_normal((1024, 1024)) * 0.02
e_clean = np.linalg.norm(quantize(W_clean, 4, group=128) @ Xq - W_clean @ Xq) / np.linalg.norm(W_clean @ Xq)
print(f"  对照：没有离群值时 INT4 分组的误差 {e_clean:.4f}")
for wb, label in [(2, "BF16"), (1, "INT8"), (0.5, "INT4")]:
    t, *_ = decode_step(m8, GPUS["RTX 4090"], 1, 2048, wbytes=wb)
    print(f"  Llama-3-8B 权重 {label}：{N8 * wb / GiB:5.1f} GiB，4090 上单请求 decode 下界 {t * 1e3:5.1f} ms/token")

# ---------------------------------------------------------------------------
# 9. 排队
# ---------------------------------------------------------------------------
print("\n=== 9. 排队：一台服务器，请求按泊松过程到达 ===")


def mg1(rho, service, n=200_000):
    """单服务台、先到先服务。service() 生成服务时间。返回平均等待、平均总时间、p99 总时间。"""
    S_ = service(n); lam = rho / S_.mean()
    A = np.cumsum(rng.exponential(1 / lam, n))
    start = np.empty(n); finish = np.empty(n); prev = 0.0
    for i in range(n):
        start[i] = max(A[i], prev); finish[i] = start[i] + S_[i]; prev = finish[i]
    wait, total = start - A, finish - A
    pk = lam * (S_ ** 2).mean() / (2 * (1 - rho))               # Pollaczek–Khinchine：平均等待时间
    return wait.mean(), pk, total.mean(), np.percentile(total, 99), lam * total.mean()   # 最后一项：Little 定律 L = λW


light = lambda n: rng.exponential(1.0, n)                       # 服务时间均值 1，指数分布
heavy = lambda n: rng.lognormal(-0.98, 1.4, n)                  # 均值约 1，但长尾（像长短悬殊的生成长度）
print("  服务时间   负载ρ   平均等待(模拟)  P-K 公式   平均总时间   p99 总时间   系统中平均请求数 L=λW")
for label, svc in [("指数", light), ("长尾", heavy)]:
    for rho in [0.5, 0.8, 0.9, 0.95]:
        w, pk, tot, p99, L = mg1(rho, svc)
        print(f"  {label:<6}   {rho:>4}   {w:12.2f}   {pk:9.2f}   {tot:9.2f}   {p99:9.1f}   {L:9.2f}")
