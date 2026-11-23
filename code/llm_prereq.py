r"""
大模型原理 · 前置知识的配套代码（topics/llm-00-prerequisites.md）。只依赖 numpy：python3 llm_prereq.py

1. 最大似然：类别分布的 MLE 就是频率；它等价于最小化"经验分布到模型"的 KL 散度。
2. softmax：平移不变、温度的作用、log-sum-exp 的数值稳定写法。
3. softmax + 交叉熵的梯度恰好是 p − y（用数值梯度核对）。
4. 反向传播：手写一个两层网络的前向和反向，用数值梯度检查，再用梯度下降训练它。
"""
import numpy as np

rng = np.random.default_rng(0)

# ---------------------------------------------------------------------------
# 1. 最大似然 = 数频率 = 最小化 KL(经验分布 ‖ 模型)
# ---------------------------------------------------------------------------
print("=== 1. 最大似然估计 ===")
true_p = np.array([0.5, 0.3, 0.2])                  # 真实分布（3 个类别）
data = rng.choice(3, size=200, p=true_p)            # 采样 200 个数据
freq = np.bincount(data, minlength=3) / len(data)   # 经验分布 = 频率

def avg_loglik(q):
    """平均对数似然 (1/N) Σ log q(x_i)。"""
    return np.mean(np.log(q[data]))

print("经验频率（就是 MLE）：", np.round(freq, 3))
for q in [np.array([1/3, 1/3, 1/3]), true_p, freq]:
    kl = np.sum(freq * np.log(freq / q))             # KL(经验 ‖ q)
    print(f"  q = {np.round(q, 3)}  平均对数似然 {avg_loglik(q):.4f}   KL(经验‖q) = {kl:.4f}")
print("  ⇒ 频率使似然最大、KL 为 0；真实分布反而稍差（有限样本的波动）。")

# ---------------------------------------------------------------------------
# 2. softmax 的性质
# ---------------------------------------------------------------------------
print("\n=== 2. softmax ===")
def softmax_naive(z):
    e = np.exp(z); return e / e.sum()

def softmax(z, T=1.0):
    """带温度 T 的 softmax，先减去最大值避免 exp 溢出。"""
    z = np.asarray(z, float) / T
    z = z - z.max()
    e = np.exp(z); return e / e.sum()

z = np.array([2.0, 1.0, 0.1])
print("softmax(z)        =", np.round(softmax(z), 4))
print("softmax(z + 100)  =", np.round(softmax(z + 100), 4), "（平移不变）")
with np.errstate(over="ignore", invalid="ignore"):
    print("朴素写法 softmax(z + 1000) =", softmax_naive(z + 1000), "（溢出成 nan）")
print("稳定写法 softmax(z + 1000) =", np.round(softmax(z + 1000), 4))
for T in [0.1, 1.0, 10.0]:
    print(f"温度 T={T:<4}：", np.round(softmax(z, T), 4))

# ---------------------------------------------------------------------------
# 3. softmax + 交叉熵的梯度 = p − y
# ---------------------------------------------------------------------------
print("\n=== 3. 交叉熵对 logits 的梯度 ===")
def ce_loss(z, y):
    """交叉熵 −log softmax(z)_y，用 log-sum-exp 稳定计算。"""
    m = z.max()
    return -(z[y] - m - np.log(np.exp(z - m).sum()))

z = rng.normal(size=5); y = 2
analytic = softmax(z); analytic[y] -= 1                 # p − onehot(y)
eps = 1e-6
numeric = np.array([(ce_loss(z + eps * np.eye(5)[i], y) - ce_loss(z - eps * np.eye(5)[i], y)) / (2 * eps)
                    for i in range(5)])
print("解析梯度 p − y：", np.round(analytic, 6))
print("数值梯度      ：", np.round(numeric, 6))
print(f"最大误差 {np.abs(analytic - numeric).max():.2e}")

# ---------------------------------------------------------------------------
# 4. 反向传播：两层网络
#    x → h = tanh(x W1 + b1) → z = h W2 + b2 → 损失 = 交叉熵(softmax(z), y)
# ---------------------------------------------------------------------------
print("\n=== 4. 反向传播与梯度下降 ===")
# 数据：平面上两个交错的半月形，二分类（线性模型分不开）
N = 400
t = rng.uniform(0, np.pi, N // 2)
X = np.vstack([np.c_[np.cos(t), np.sin(t)], np.c_[1 - np.cos(t), 0.5 - np.sin(t)]])
X += 0.1 * rng.normal(size=X.shape)
Y = np.r_[np.zeros(N // 2, int), np.ones(N // 2, int)]

def init(hidden=16):
    return {"W1": rng.normal(size=(2, hidden)) * 0.5, "b1": np.zeros(hidden),
            "W2": rng.normal(size=(hidden, 2)) * 0.5, "b2": np.zeros(2)}

def forward_backward(p, X, Y):
    """返回 (平均损失, 各参数的梯度)。每一步反向都对应前向的一行。"""
    n = len(X)
    a1 = X @ p["W1"] + p["b1"]               # (n, H)
    h = np.tanh(a1)                          # (n, H)
    z = h @ p["W2"] + p["b2"]                # (n, 2)  logits
    z = z - z.max(1, keepdims=True)
    P = np.exp(z); P /= P.sum(1, keepdims=True)
    loss = -np.mean(np.log(P[np.arange(n), Y]))
    # —— 反向：从损失往回，逐层用链式法则 ——
    dz = P.copy(); dz[np.arange(n), Y] -= 1; dz /= n     # ∂L/∂z = (p − y)/n
    g = {"W2": h.T @ dz, "b2": dz.sum(0)}
    dh = dz @ p["W2"].T                                   # ∂L/∂h
    da1 = dh * (1 - h ** 2)                               # tanh'(a) = 1 − tanh(a)²
    g["W1"] = X.T @ da1; g["b1"] = da1.sum(0)
    return loss, g

# 梯度检查：对每个参数随机挑几个分量，和数值梯度比较
p = init()
_, g = forward_backward(p, X, Y)
worst = 0.0
for name in p:
    for _ in range(3):
        idx = tuple(rng.integers(0, s) for s in p[name].shape)
        old = p[name][idx]
        p[name][idx] = old + 1e-5; lp, _ = forward_backward(p, X, Y)
        p[name][idx] = old - 1e-5; lm, _ = forward_backward(p, X, Y)
        p[name][idx] = old
        worst = max(worst, abs((lp - lm) / 2e-5 - g[name][idx]))
print(f"梯度检查：反向传播 vs 数值梯度，最大误差 {worst:.2e}")

# 梯度下降训练
lr = 0.5
for step in range(3001):
    loss, g = forward_backward(p, X, Y)
    for name in p:
        p[name] -= lr * g[name]              # θ ← θ − η ∇L
    if step in (0, 100, 500, 1000, 3000):
        h = np.tanh(X @ p["W1"] + p["b1"]); acc = np.mean((h @ p["W2"] + p["b2"]).argmax(1) == Y)
        print(f"  第 {step:>4} 步：损失 {loss:.4f}   训练准确率 {acc:.3f}")
