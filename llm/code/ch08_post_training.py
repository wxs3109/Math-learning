r"""
第 8 章 · 预训练与后训练的配套代码（llm/ch08-pre-post-training.md）。只依赖 numpy：python3 ch08_post_training.py

不训练真的大模型，而是用能在 CPU 上几秒跑完的小例子，把每个方法的核心机制单独拿出来看：
1. 训练的显存账本：全参微调、ZeRO、LoRA、QLoRA 各要多少显存。
2. Scaling law：给定算力，参数量和数据量怎么分。
3. LoRA：数据少时，低秩更新比全参更新学得准、"忘"得少。
4. RLHF 与 DPO：两者收敛到同一个最优策略；奖励模型有偏差时，优化过头会"奖励作弊"。
5. GRPO：组内基线让梯度估计的方差变小；全组都错的题没有学习信号。
6. 蒸馏：用老师的整个概率分布教学生，比只用老师的答案教，学得快得多。
"""
import numpy as np

rng = np.random.default_rng(0)
GiB = 2 ** 30


def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def log_softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))


# ===========================================================================
# 1. 训练的显存账本（8.3 节）
# ===========================================================================
def llama_linear_shapes(d=4096, ffn=14336, n_heads=32, n_kv=8, layers=32):
    """Llama-3-8B 每层的 7 个线性层 (输入维度, 输出维度)。"""
    hd = d // n_heads
    per_layer = [(d, d), (d, n_kv * hd), (d, n_kv * hd), (d, d),     # q k v o
                 (d, ffn), (d, ffn), (ffn, d)]                       # gate up down
    return per_layer * layers


def memory_ledger():
    print("=" * 72)
    print("1. 训练的显存账本（不含激活；Llama-3-8B 形状，总参数取 8.03B）")
    print("=" * 72)
    N = 8.03e9
    shapes = llama_linear_shapes()
    for r in (8, 16, 64):
        n_lora = sum(r * (i + o) for i, o in shapes)
        print(f"  LoRA r={r:<3} 加在全部 7 种线性层：可训练参数 {n_lora / 1e6:6.1f}M，占 {n_lora / N:.2%}")
    n_lora = sum(16 * (i + o) for i, o in shapes)

    rows = [
        ("全参，FP32 + Adam", 16 * N),                      # 权重 4 + 梯度 4 + Adam 两个矩 8
        ("全参，混合精度 + Adam", 16 * N),                   # BF16 权重 2 + 梯度 2 + FP32 主权重 4 + 两个矩 8
        ("  ZeRO-1，8 卡，每卡", 4 * N + 12 * N / 8),
        ("  ZeRO-2，8 卡，每卡", 2 * N + 14 * N / 8),
        ("  ZeRO-3，8 卡，每卡", 16 * N / 8),
        ("LoRA r=16（底座 BF16 冻结）", 2 * N + 16 * n_lora),
        ("QLoRA r=16（底座 4 bit）", 4.127 / 8 * N + 16 * n_lora),
        ("只做推理，BF16", 2 * N),
    ]
    for name, b in rows:
        print(f"  {name:<28}{b / GiB:8.1f} GiB")

    # 激活：Korthikanti et al. 2022 的估算，每层 s*b*h*(34 + 5*a*s/h) 字节（FP16，含 dropout 掩码）。
    # 用 FlashAttention 时不存注意力矩阵，5as/h 那一项消失。
    s, b, h, a, L = 2048, 1, 4096, 32, 32
    full = L * s * b * h * (34 + 5 * a * s / h)
    flash = L * s * b * h * 34
    ckpt = L * 2 * s * b * h + s * b * h * 34          # 只存每层输入，反向时一层一层重算
    print(f"\n  激活（序列 {s}，batch {b}）：普通注意力 {full / GiB:.1f} GiB；"
          f"FlashAttention {flash / GiB:.1f} GiB；再加梯度检查点 {ckpt / GiB:.2f} GiB")

    D = 15e12
    flops = 6 * N * D
    gpu_s = flops / (989e12 * 0.40)
    print(f"\n  预训练算力：6ND = 6 × {N:.2e} × {D:.1e} = {flops:.2e} FLOP；"
          f"H100 按 40% 利用率约 {gpu_s / 3600 / 1e3:.0f} 千 GPU 小时")


# ===========================================================================
# 2. Scaling law（8.2 节）
# ===========================================================================
def chinchilla():
    print("\n" + "=" * 72)
    print("2. Scaling law：L(N,D) = E + A/N^α + B/D^β，约束 C = 6ND 下的最优分配")
    print("=" * 72)
    fits = {"Hoffmann 等 2022 的拟合": (1.69, 406.4, 410.7, 0.34, 0.28),
            "Besiroglu 等 2024 的重新拟合": (1.82, 482.01, 2085.43, 0.3478, 0.3658)}
    for name, (E, A, B, al, be) in fits.items():
        print(f"  {name}：N* ∝ C^{be / (al + be):.3f}")
        G = (al * A / (be * B)) ** (1 / (al + be))
        for C in (1e21, 1e23, 1e25):
            N = G * (C / 6) ** (be / (al + be))
            D = C / 6 / N
            L = E + A / N ** al + B / D ** be
            print(f"    C={C:.0e}：N*={N / 1e9:7.2f}B，D*={D / 1e9:8.0f}B token，"
                  f"每参数 {D / N:5.1f} 个 token，损失 {L:.3f}")


# ===========================================================================
# 3. LoRA：低秩更新（8.5 节）
# ===========================================================================
def lora_experiment():
    print("\n" + "=" * 72)
    print("3. LoRA vs 全参微调：线性模型 y = W x，新任务 W* = W0 + 秩 4 的增量")
    print("=" * 72)
    d, true_rank, noise = 64, 4, 0.5
    W0 = rng.normal(0, 1 / np.sqrt(d), (d, d))
    dW_true = rng.normal(0, 1, (d, true_rank)) @ rng.normal(0, 1, (true_rank, d)) / d
    W_star = W0 + dW_true
    X_gen = rng.normal(0, 1, (2000, d))                 # "通用"输入，用来量遗忘：微调后它们的输出变了多少

    def task_data(n):
        X = rng.normal(0, 1, (n, d))
        return X, X @ W_star.T + noise * rng.normal(0, 1, (n, d))

    def fit_full(X, Y):
        dW, *_ = np.linalg.lstsq(X, Y - X @ W0.T, rcond=None)   # 最小范数解：等价于从 0 出发的梯度下降
        return W0 + dW.T

    def fit_lora(X, Y, r, steps=3000, lr=0.05):
        B = np.zeros((d, r))                              # B 初始化为 0：一开始 W = W0，不破坏原模型
        A = rng.normal(0, 1 / np.sqrt(d), (r, d))
        R0 = Y - X @ W0.T
        for _ in range(steps):
            E = X @ (B @ A).T - R0                         # 残差，形状 (n, d)
            G = E.T @ X / len(X)                           # 对 ΔW 的梯度
            B, A = B - lr * G @ A.T, A - lr * B.T @ G
        return W0 + B @ A

    X_test = rng.normal(0, 1, (2000, d))
    print(f"  全参可训练 {d * d} 个参数；LoRA r 个秩是 r×{2 * d} 个")
    print(f"  {'样本数':<8}{'方法':<12}{'新任务测试误差':>14}{'通用输入的改变量':>16}")
    for n in (48, 128, 1024):
        X, Y = task_data(n)
        for name, W in [("全参", fit_full(X, Y))] + [(f"LoRA r={r}", fit_lora(X, Y, r)) for r in (1, 4, 16)]:
            err = np.mean((X_test @ (W - W_star).T) ** 2)
            drift = np.mean((X_gen @ (W - W0).T) ** 2)
            print(f"  {n:<10}{name:<12}{err:>12.4f}{drift:>16.4f}")
    print(f"  （真实增量的改变量是 {np.mean((X_gen @ dW_true.T) ** 2):.4f}，噪声方差是 {noise ** 2}）")


# ===========================================================================
# 4. RLHF 与 DPO（8.6、8.7 节）
# ===========================================================================
def rlhf_dpo():
    print("\n" + "=" * 72)
    print("4. RLHF 与 DPO：一个提示，8 个候选回答")
    print("=" * 72)
    # 每个回答有两个特征：质量 q 和长度 ℓ。真正想要的奖励只看质量。
    q = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 0.8, 0.2, 1.2])
    ell = np.array([0.0, 0.5, 1.0, 1.0, 1.5, 2.5, 5.0, 2.0])   # 回答 6：又长又空
    true_r = q
    pi_ref = softmax(np.zeros(8) + 0.3 * rng.normal(size=8))
    # 标注者有长度偏好：他们按 q + 0.6ℓ 选（Bradley–Terry 模型）
    util = q + 0.6 * ell
    M = 4000
    i, j = rng.choice(8, M, p=pi_ref), rng.choice(8, M, p=pi_ref)
    keep = i != j
    i, j = i[keep], j[keep]
    i_wins = rng.random(len(i)) < 1 / (1 + np.exp(-(util[i] - util[j])))
    w, l = np.where(i_wins, i, j), np.where(i_wins, j, i)
    print(f"  偏好数据：{len(w)} 对，按 π_ref 采样回答，标注者按 q + 0.6ℓ 选")

    # 4a. 奖励模型：对特征差做逻辑回归
    F = np.stack([q, ell], 1)
    theta = np.zeros(2)
    for _ in range(3000):
        z = (F[w] - F[l]) @ theta
        theta += 0.5 * ((1 - 1 / (1 + np.exp(-z)))[:, None] * (F[w] - F[l])).mean(0)
    r_hat = F @ theta
    print(f"  奖励模型学到 r̂ = {theta[0]:.2f}·q + {theta[1]:.2f}·ℓ（标注者用的是 1·q + 0.6·ℓ，真正想要的是 1·q + 0·ℓ）")

    # 4b. KL 约束下的最优策略：π* ∝ π_ref · exp(r̂/β)
    print(f"\n  {'β':>6}{'代理奖励 E[r̂]':>16}{'真实奖励 E[q]':>14}{'KL(π‖π_ref)':>14}{'平均长度':>10}")
    print(f"  {'∞':>6}{pi_ref @ r_hat:>16.3f}{pi_ref @ true_r:>14.3f}{0:>14.3f}{pi_ref @ ell:>10.2f}   ← 就是 π_ref")
    for beta in (10, 3, 1, 0.5, 0.3, 0.2, 0.1, 0.05):
        pi = softmax(np.log(pi_ref) + r_hat / beta)
        kl = np.sum(pi * np.log(pi / pi_ref))
        print(f"  {beta:>6}{pi @ r_hat:>16.3f}{pi @ true_r:>14.3f}{kl:>14.3f}{pi @ ell:>10.2f}")

    # 4c. 同一个 β 下，三种方法得到的策略
    beta = 1.0
    closed = softmax(np.log(pi_ref) + r_hat / beta)

    # RLHF：在表格策略上做策略梯度（期望精确计算），目标 E_π[r̂] − β KL(π‖π_ref)
    z = np.log(pi_ref).copy()
    for _ in range(4000):
        pi = softmax(z)
        adv = r_hat - beta * (np.log(pi / pi_ref) + 1)           # 对 π 的导数
        z += 0.1 * pi * (adv - pi @ adv)                          # 链式法则穿过 softmax
    rlhf = softmax(z)

    # DPO：直接在偏好对上最小化 −log σ(β[log π(w)/π_ref(w) − log π(l)/π_ref(l)])
    z = np.log(pi_ref).copy()
    for _ in range(4000):
        lp = log_softmax(z) - np.log(pi_ref)
        m = beta * (lp[w] - lp[l])
        g = 1 - 1 / (1 + np.exp(-m))                              # σ(−m)：模型越"判错"，权重越大
        grad = np.zeros(8)
        np.add.at(grad, w, g); np.add.at(grad, l, -g)
        grad = beta * grad / len(w)
        z += 2.0 * (grad - softmax(z) * grad.sum())               # 对 log π 的梯度穿过 softmax
    dpo = softmax(z)

    tv = lambda a, b: 0.5 * np.abs(a - b).sum()
    print(f"\n  β = {beta}：")
    print(f"    TV(RLHF, 闭式最优) = {tv(rlhf, closed):.4f}")
    print(f"    TV(DPO,  闭式最优) = {tv(dpo, closed):.4f}")
    print(f"    TV(DPO,  RLHF)     = {tv(dpo, rlhf):.4f}")
    print("    回答      " + " ".join(f"{k:>6}" for k in range(8)))
    for name, p in [("π_ref", pi_ref), ("闭式最优", closed), ("RLHF", rlhf), ("DPO", dpo)]:
        print(f"    {name:<8}" + " ".join(f"{x:>6.3f}" for x in p))


# ===========================================================================
# 5. GRPO：组内基线（8.8 节）
# ===========================================================================
def grpo():
    print("\n" + "=" * 72)
    print("5. GRPO：可验证奖励 + 组内基线")
    print("=" * 72)
    K, G = 10, 8

    def grad_samples(z, correct, baseline, trials=20000):
        """返回 trials 次梯度估计（对 logits），每次采 G 个回答。"""
        pi = softmax(z)
        ys = rng.choice(K, (trials, G), p=pi)
        r = 1.0 + (ys == correct)                                   # 格式对得 1 分，答案对再加 1 分
        if baseline == "无":
            adv = r
        elif baseline == "留一均值":
            adv = r - (r.sum(1, keepdims=True) - r) / (G - 1)      # 不含自己的均值：仍然无偏
        else:
            mu, sd = r.mean(1, keepdims=True), r.std(1, keepdims=True)
            adv = (r - mu) / (sd + 1e-6)                            # GRPO：减组均值、除组标准差
        onehot = np.eye(K)[ys]                                      # ∇ log π(y) = e_y − π
        g = (adv[..., None] * (onehot - pi)).mean(1)
        return g

    z = np.zeros(K); z[0] = np.log(3.0)                             # 正确答案 0 的初始概率 = 0.25
    exact = softmax(z)[0] * (np.eye(K)[0] - softmax(z))            # 真梯度 ∇E[r]（常数 1 分的梯度是 0）
    print(f"  正确答案的初始概率 {softmax(z)[0]:.2f}，每组 {G} 个回答；奖励 = 1（格式）+ 1（答案对）")
    for b in ("无", "留一均值", "组均值/组标准差"):
        g = grad_samples(z, 0, b)
        ratio = np.linalg.norm(g.mean(0)) / np.linalg.norm(exact)
        cos = g.mean(0) @ exact / np.linalg.norm(g.mean(0)) / np.linalg.norm(exact)
        snr = np.linalg.norm(g.mean(0)) ** 2 / g.var(0).sum()      # 信噪比：均值的平方 / 方差
        print(f"  基线 = {b:<14} 方差 {g.var(0).sum():.4f}   信噪比 {snr:.3f}   方向余弦 {cos:.3f}   长度/真梯度 {ratio:.2f}")

    # 训练：60 道题，难度不同；每步每道题采一组。奖励只看答案对不对
    n_q = 60
    p0 = np.geomspace(0.002, 0.6, n_q)                               # 每道题正确答案的初始概率
    Z = np.zeros((n_q, K))
    Z[:, 0] = np.log(p0 * (K - 1) / (1 - p0))
    hard = p0 < 0.02
    print(f"\n  {n_q} 道题，初始正确率从 {p0[0]:.3f} 到 {p0[-1]:.2f}（几何间隔），其中 {hard.sum()} 道低于 2%；每步每题采 {G} 个")
    print(f"  {'步数':>6}{'平均正确率':>12}{'难题正确率':>12}{'全组都错':>10}{'全组都对':>10}")
    for step in range(401):
        P = softmax(Z)
        ys = np.array([rng.choice(K, G, p=p) for p in P])
        r = (ys == 0).astype(float)
        mu, sd = r.mean(1, keepdims=True), r.std(1, keepdims=True)
        adv = (r - mu) / (sd + 1e-6)                                 # 全组同分时 adv = 0：这道题这一步不学
        if step in (0, 25, 50, 100, 200, 400):
            print(f"  {step:>6}{P[:, 0].mean():>12.3f}{P[hard, 0].mean():>12.3f}"
                  f"{np.mean(mu[:, 0] == 0):>10.0%}{np.mean(mu[:, 0] == 1):>10.0%}")
        g = (adv[..., None] * (np.eye(K)[ys] - P[:, None, :])).mean(1)
        Z += 0.5 * g


# ===========================================================================
# 6. 蒸馏（8.9 节）
# ===========================================================================
def distillation():
    print("\n" + "=" * 72)
    print("6. 蒸馏：学生学老师的答案（硬标签），还是学老师的整个分布（软标签）")
    print("=" * 72)
    d, C = 20, 10
    Wt = rng.normal(0, 0.8, (C, d))                                  # 老师：softmax(W_t x)
    X_test = rng.normal(0, 1, (4000, d))
    P_test = softmax(X_test @ Wt.T)

    def train(X, T, steps=1500, lr=0.5):
        W = np.zeros((C, d))
        for _ in range(steps):
            W -= lr * (softmax(X @ W.T) - T).T @ X / len(X)          # 交叉熵对 W 的梯度
        return W

    def kl_to_teacher(W):
        return np.mean(np.sum(P_test * (np.log(P_test) - log_softmax(X_test @ W.T)), 1))

    print(f"  {'样本数':<8}{'硬标签（老师采样的答案）':>20}{'软标签（老师的分布）':>20}   KL(老师‖学生)")
    for n in (50, 200, 1000, 5000):
        X = rng.normal(0, 1, (n, d))
        P = softmax(X @ Wt.T)
        hard = np.eye(C)[[rng.choice(C, p=p) for p in P]]
        print(f"  {n:<10}{kl_to_teacher(train(X, hard)):>20.4f}{kl_to_teacher(train(X, P)):>22.4f}")


if __name__ == "__main__":
    memory_ledger()
    chinchilla()
    lora_experiment()
    rlhf_dpo()
    grpo()
    distillation()
