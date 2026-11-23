r"""
前置知识 P4 · 微积分与优化的配套代码（topics/llm-p4-calculus-optimization.md）。只依赖 numpy：python3 llm_p4_calculus_optimization.py

1. 数值梯度检查；三个矩阵微积分恒等式。
2. softmax 的数值稳定性（log-sum-exp）与梯度 p − y。
3. 梯度下降：步长条件 η < 2/L；条件数决定收敛速度。
4. 动量；SGD 的批量噪声按 1/√B 缩小；Adam。
5. 一个 60 行的标量自动微分（micrograd 风格）及梯度检查。
6. 两层网络的手写反向传播，训练到 100% 准确率。
"""
import math
import numpy as np

rng = np.random.default_rng(0)


def num_grad(f, x, eps=1e-6):
    """中心差分：g_i ≈ (f(x + eps e_i) − f(x − eps e_i)) / (2 eps)，误差 O(eps²)。"""
    g = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    for _ in it:
        i = it.multi_index
        old = x[i]
        x[i] = old + eps; fp = f(x)
        x[i] = old - eps; fm = f(x)
        x[i] = old
        g[i] = (fp - fm) / (2 * eps)
    return g


def rel_err(a, b):
    return np.abs(a - b).max() / max(np.abs(a).max(), np.abs(b).max(), 1e-12)


# ---------------------------------------------------------------------------
# 1. 矩阵微积分恒等式
# ---------------------------------------------------------------------------
print("=== 1. 矩阵微积分恒等式（解析梯度 vs 数值梯度）===")
d = 5
a = rng.normal(size=d); A = rng.normal(size=(d, d)); x = rng.normal(size=d)
print(f"  ∇ₓ(aᵀx) = a                 相对误差 {rel_err(a, num_grad(lambda x: a @ x, x)):.1e}")
print(f"  ∇ₓ(xᵀAx) = (A + Aᵀ)x        相对误差 {rel_err((A + A.T) @ x, num_grad(lambda x: x @ A @ x, x)):.1e}")
X = rng.normal(size=(7, d)); W = rng.normal(size=(d, 3)); G = rng.normal(size=(7, 3))
# 标量 L = sum(G ⊙ (XW))，即 L 对 XW 的梯度是 G。链式法则给出 ∂L/∂W = XᵀG，∂L/∂X = G Wᵀ
f_W = lambda W: np.sum(G * (X @ W)); f_X = lambda X: np.sum(G * (X @ W))
print(f"  Y = XW：∂L/∂W = Xᵀ(∂L/∂Y)   相对误差 {rel_err(X.T @ G, num_grad(f_W, W.copy())):.1e}")
print(f"          ∂L/∂X = (∂L/∂Y)Wᵀ   相对误差 {rel_err(G @ W.T, num_grad(f_X, X.copy())):.1e}")

# ---------------------------------------------------------------------------
# 2. softmax、log-sum-exp、交叉熵的梯度
# ---------------------------------------------------------------------------
print("\n=== 2. softmax 的数值稳定性与梯度 ===")
z = np.array([1000., 1001., 1002.])
with np.errstate(over="ignore", invalid="ignore"):
    naive = np.exp(z) / np.exp(z).sum()
stable = np.exp(z - z.max()) / np.exp(z - z.max()).sum()
print(f"  z = {z}：直接算 {naive}；减去最大值后 {np.round(stable, 4)}")


def ce(z, y):
    m = z.max()
    return -(z[y] - (m + np.log(np.exp(z - m).sum())))   # −log softmax(z)_y，用 log-sum-exp


z = rng.normal(size=6); y = 2
p = np.exp(z - z.max()); p /= p.sum()
g_analytic = p - np.eye(6)[y]
print(f"  交叉熵对 logits 的梯度 p − y，与数值梯度的相对误差 {rel_err(g_analytic, num_grad(lambda z: ce(z, y), z.copy())):.1e}")

# ---------------------------------------------------------------------------
# 3. 梯度下降：步长与条件数
# ---------------------------------------------------------------------------
print("\n=== 3. 梯度下降：步长与条件数 ===")


def gd_quadratic(lams, eta, steps):
    """在 f(x) = ½ Σ λ_i x_i² 上做梯度下降，从 (1,…,1) 出发，返回最终的 f。"""
    x = np.ones(len(lams))
    for _ in range(steps):
        x = x - eta * lams * x
    return 0.5 * np.sum(lams * x * x)


L = 10.0
print(f"  f = ½(1·x₁² + 10·x₂²)，L = 10，阈值 2/L = 0.2，100 步后的 f：")
for eta in [0.05, 0.1, 0.19, 0.21]:
    print(f"    η = {eta:<5} f = {gd_quadratic(np.array([1., L]), eta, 100):.3e}")
print("  条件数 κ = L/μ，取 η = 1/L，降到 f < 1e-8 需要的步数：")
for kappa in [1, 10, 100, 1000]:
    lams = np.array([1.0, kappa]); x = np.ones(2); k = 0
    while 0.5 * np.sum(lams * x * x) > 1e-8 and k < 10 ** 6:
        x = x - (1 / kappa) * lams * x; k += 1
    print(f"    κ = {kappa:<5} 步数 {k}")

# ---------------------------------------------------------------------------
# 4. 动量、SGD、Adam
# ---------------------------------------------------------------------------
print("\n=== 4. 动量、SGD 的噪声、Adam ===")
kappa = 100.0; lams = np.array([1.0, kappa])


def steps_to(update, tol=1e-8, cap=10 ** 5):
    x = np.ones(2); state = {}; k = 0
    while 0.5 * np.sum(lams * x * x) > tol and k < cap:
        x = update(x, lams * x, state); k += 1
    return k


plain = lambda x, g, s: x - (1 / kappa) * g


def heavy_ball(x, g, s, eta=1 / kappa, beta=0.9):
    s["v"] = beta * s.get("v", 0) + g          # 速度：过去梯度的指数加权和
    return x - eta * s["v"]


def adam(x, g, s, eta=0.02, b1=0.9, b2=0.999, eps=1e-12):
    s["t"] = s.get("t", 0) + 1
    s["m"] = b1 * s.get("m", 0) + (1 - b1) * g          # 一阶矩
    s["v"] = b2 * s.get("v", 0) + (1 - b2) * g * g      # 二阶矩
    mh = s["m"] / (1 - b1 ** s["t"]); vh = s["v"] / (1 - b2 ** s["t"])   # 偏差修正
    return x - eta * mh / (np.sqrt(vh) + eps)


print(f"  κ = 100 的二次函数，降到 f < 1e-8 的步数：GD {steps_to(plain)}，"
      f"动量(β=0.9) {steps_to(heavy_ball)}，Adam(η=0.02) {steps_to(adam)}")

# SGD：小批量梯度是全量梯度的无偏估计，标准差 ∝ 1/√B
N, dim = 10000, 20
Xs = rng.normal(size=(N, dim)); ws = rng.normal(size=dim); ys = Xs @ ws + rng.normal(size=N)
w0 = np.zeros(dim)
full = Xs.T @ (Xs @ w0 - ys) / N
print("  线性回归的小批量梯度（与全量梯度的偏差）：")
for B in [1, 10, 100, 1000]:
    errs = []
    for _ in range(500):
        idx = rng.integers(0, N, B)
        gb = Xs[idx].T @ (Xs[idx] @ w0 - ys[idx]) / B
        errs.append(np.linalg.norm(gb - full))
    print(f"    B = {B:<5} 平均 ‖g_B − g‖ = {np.mean(errs):7.3f}   ×√B = {np.mean(errs) * np.sqrt(B):.2f}")

# ---------------------------------------------------------------------------
# 5. 标量自动微分
# ---------------------------------------------------------------------------
print("\n=== 5. 标量自动微分（反向模式）===")


class Value:
    """一个标量节点：记录数值、梯度、以及"怎样把梯度传给子节点"。"""

    def __init__(self, data, children=()):
        self.data = data; self.grad = 0.0
        self._children = children; self._backward = lambda: None

    def __add__(self, o):
        o = o if isinstance(o, Value) else Value(o)
        out = Value(self.data + o.data, (self, o))
        def bw(): self.grad += out.grad; o.grad += out.grad          # 加法：梯度原样分给两边
        out._backward = bw; return out

    def __mul__(self, o):
        o = o if isinstance(o, Value) else Value(o)
        out = Value(self.data * o.data, (self, o))
        def bw(): self.grad += o.data * out.grad; o.grad += self.data * out.grad   # 乘法：乘以对方
        out._backward = bw; return out

    def tanh(self):
        t = math.tanh(self.data); out = Value(t, (self,))
        def bw(): self.grad += (1 - t * t) * out.grad
        out._backward = bw; return out

    def exp(self):
        e = math.exp(self.data); out = Value(e, (self,))
        def bw(): self.grad += e * out.grad
        out._backward = bw; return out

    def log(self):
        out = Value(math.log(self.data), (self,))
        def bw(): self.grad += out.grad / self.data
        out._backward = bw; return out

    __radd__ = __add__; __rmul__ = __mul__
    def __neg__(self): return self * -1
    def __sub__(self, o): return self + (-o)
    def __truediv__(self, o): return self * (o ** -1 if isinstance(o, Value) else Value(1 / o))
    def __pow__(self, k):
        out = Value(self.data ** k, (self,))
        def bw(): self.grad += k * self.data ** (k - 1) * out.grad
        out._backward = bw; return out

    def backward(self):
        order, seen = [], set()
        def topo(v):                       # 拓扑排序：保证一个节点的梯度收齐后才往下传
            if id(v) not in seen:
                seen.add(id(v))
                for c in v._children: topo(c)
                order.append(v)
        topo(self); self.grad = 1.0
        for v in reversed(order): v._backward()


# 一个 3 → 4 → 3 的 tanh 网络加交叉熵，全部用 Value 计算
W1 = rng.normal(size=(3, 4)) * 0.5; W2 = rng.normal(size=(4, 3)) * 0.5
xin = rng.normal(size=3); target = 1


def forward_numpy(W1, W2):
    h = np.tanh(xin @ W1); z = h @ W2
    return -(z[target] - np.log(np.exp(z).sum()))


V1 = [[Value(W1[i, j]) for j in range(4)] for i in range(3)]
V2 = [[Value(W2[i, j]) for j in range(3)] for i in range(4)]
h = [sum((V1[i][j] * xin[i] for i in range(3)), Value(0.0)).tanh() for j in range(4)]
zv = [sum((h[i] * V2[i][k] for i in range(4)), Value(0.0)) for k in range(3)]
loss = sum((zk.exp() for zk in zv), Value(0.0)).log() - zv[target]
loss.backward()
g1 = np.array([[V1[i][j].grad for j in range(4)] for i in range(3)])
g2 = np.array([[V2[i][j].grad for j in range(3)] for i in range(4)])
n1 = num_grad(lambda W: forward_numpy(W, W2), W1.copy()); n2 = num_grad(lambda W: forward_numpy(W1, W), W2.copy())
print(f"  损失 {loss.data:.6f}（numpy 前向 {forward_numpy(W1, W2):.6f}）")
print(f"  自动微分 vs 数值梯度：W1 相对误差 {rel_err(g1, n1):.1e}，W2 相对误差 {rel_err(g2, n2):.1e}")

# ---------------------------------------------------------------------------
# 6. 两层网络：手写矩阵形式的反向传播
# ---------------------------------------------------------------------------
print("\n=== 6. 两层网络（矩阵形式的反向传播）===")
# 数据：平面上三条螺旋臂，线性不可分
K, per = 3, 100
Xd = np.zeros((K * per, 2)); yd = np.repeat(np.arange(K), per)
for k in range(K):
    r = np.linspace(0.05, 1, per); t = np.linspace(k * 4, (k + 1) * 4, per) + rng.normal(0, 0.15, per)
    Xd[k * per:(k + 1) * per] = np.c_[r * np.sin(t), r * np.cos(t)]
H = 64
P = {"W1": rng.normal(size=(2, H)) * 0.5, "b1": np.zeros(H),
     "W2": rng.normal(size=(H, K)) * 0.1, "b2": np.zeros(K)}


def loss_and_grads(P, X, y):
    n = len(y)
    a1 = X @ P["W1"] + P["b1"]; h1 = np.maximum(a1, 0)          # ReLU
    z = h1 @ P["W2"] + P["b2"]
    z = z - z.max(1, keepdims=True); p = np.exp(z); p /= p.sum(1, keepdims=True)
    L = -np.log(p[np.arange(n), y]).mean()
    dz = p.copy(); dz[np.arange(n), y] -= 1; dz /= n             # p − y
    g = {"W2": h1.T @ dz, "b2": dz.sum(0)}                       # Y = XW ⇒ ∂W = Xᵀ ∂Y
    dh = dz @ P["W2"].T                                          #         ⇒ ∂X = ∂Y Wᵀ
    da = dh * (a1 > 0)                                           # ReLU 的导数是 0 或 1
    g["W1"] = X.T @ da; g["b1"] = da.sum(0)
    return L, g, p


_, g, _ = loss_and_grads(P, Xd, yd)
nW1 = num_grad(lambda W: loss_and_grads({**P, "W1": W}, Xd, yd)[0], P["W1"].copy())
print(f"  反向传播 vs 数值梯度：W1 相对误差 {rel_err(g['W1'], nW1):.1e}")
m = {k: 0 for k in P}; v = {k: 0 for k in P}
for t in range(1, 3001):                                         # Adam 训练
    L, g, p = loss_and_grads(P, Xd, yd)
    for k in P:
        m[k] = 0.9 * m[k] + 0.1 * g[k]; v[k] = 0.999 * v[k] + 0.001 * g[k] ** 2
        P[k] -= 0.01 * (m[k] / (1 - 0.9 ** t)) / (np.sqrt(v[k] / (1 - 0.999 ** t)) + 1e-8)
    if t in (1, 100, 500, 3000):
        print(f"  第 {t:>4} 步：损失 {L:.4f}，训练准确率 {(p.argmax(1) == yd).mean():.1%}")
