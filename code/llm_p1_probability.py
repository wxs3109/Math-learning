r"""
前置知识 P1 · 概率论的配套代码（topics/llm-p1-probability.md）。只依赖 numpy：python3 llm_p1_probability.py

1. 贝叶斯公式：检测阳性时真正患病的概率（基础比率的作用）。
2. 期望、方差、协方差；"不相关"不等于"独立"。
3. 大数定律与中心极限定理：样本均值的收敛速度是 1/√n，标准化后接近正态。
4. 多元高斯：用 Cholesky 分解从 N(μ, Σ) 采样，检查样本协方差。
5. 信息论：熵、交叉熵 ≥ 熵、KL 不对称、互信息。
"""
import numpy as np

rng = np.random.default_rng(0)

# ---------------------------------------------------------------------------
# 1. 贝叶斯公式
# ---------------------------------------------------------------------------
print("=== 1. 贝叶斯公式 ===")
prior = 0.01            # 患病率 P(D)
sens = 0.99             # 灵敏度 P(+ | D)
spec = 0.95             # 特异度 P(- | 非D)
p_pos = sens * prior + (1 - spec) * (1 - prior)          # 全概率公式 P(+)
post = sens * prior / p_pos                              # 贝叶斯 P(D | +)
print(f"P(+) = {p_pos:.4f}，P(患病 | 阳性) = {post:.4f}")
# 用模拟核对
N = 1_000_000
D = rng.random(N) < prior
pos = np.where(D, rng.random(N) < sens, rng.random(N) < 1 - spec)
print(f"模拟：阳性中真正患病的比例 = {D[pos].mean():.4f}")

# ---------------------------------------------------------------------------
# 2. 期望、方差、协方差；不相关 ≠ 独立
# ---------------------------------------------------------------------------
print("\n=== 2. 期望、方差、协方差 ===")
X = rng.normal(0, 1, 200_000)
Y = X ** 2                                   # Y 完全由 X 决定，显然不独立
print(f"Cov(X, X²) = {np.cov(X, Y)[0, 1]:.4f}（理论值 E[X³] = 0）⇒ 不相关，但不独立")
print(f"条件期望 E[Y | |X|>2] = {Y[np.abs(X) > 2].mean():.3f}，而 E[Y] = {Y.mean():.3f}")
# 线性组合的方差：Var(aX + bZ) = a²Var X + b²Var Z + 2ab Cov(X,Z)
Z = 0.6 * X + 0.8 * rng.normal(0, 1, X.size)  # Corr(X, Z) = 0.6
a, b = 2.0, -1.0
print(f"Var(2X − Z)：公式 {a**2 + b**2 + 2*a*b*0.6:.3f}，模拟 {np.var(a*X + b*Z):.3f}")

# ---------------------------------------------------------------------------
# 3. 大数定律与中心极限定理
# ---------------------------------------------------------------------------
print("\n=== 3. 大数定律与中心极限定理 ===")
p = 0.3                                      # 伯努利(0.3)：均值 0.3，标准差 √(0.21)
sd = np.sqrt(p * (1 - p))
for n in [10, 100, 1000, 10000]:
    means = rng.binomial(n, p, size=20000) / n          # 20000 次独立实验的样本均值
    z = (means - p) / (sd / np.sqrt(n))                 # 标准化
    print(f"n={n:<6} 样本均值的标准差 {means.std():.5f}（理论 σ/√n = {sd/np.sqrt(n):.5f}）"
          f"   |Z|<1.96 的比例 {np.mean(np.abs(z) < 1.96):.3f}（正态 0.950）")

# ---------------------------------------------------------------------------
# 4. 多元高斯
# ---------------------------------------------------------------------------
print("\n=== 4. 多元高斯 ===")
mu = np.array([1.0, -2.0, 0.5])
Sigma = np.array([[2.0, 0.8, 0.3],
                  [0.8, 1.0, -0.4],
                  [0.3, -0.4, 0.5]])
L = np.linalg.cholesky(Sigma)                # Σ = L Lᵀ
S = mu + rng.normal(size=(1_000_000, 3)) @ L.T # x = μ + L z，z ~ N(0, I)
print("样本均值：", np.round(S.mean(0), 3))
print("样本协方差与 Σ 的最大偏差：", f"{np.abs(np.cov(S.T) - Sigma).max():.4f}")
# 条件分布：x1 | x2 的均值是 x2 的线性函数
sub = S[np.abs(S[:, 1] - 0.0) < 0.05]       # 条件在 x2 ≈ 0
cond_mean = mu[0] + Sigma[0, 1] / Sigma[1, 1] * (0.0 - mu[1])
print(f"E[x1 | x2=0]：公式 {cond_mean:.3f}，模拟 {sub[:, 0].mean():.3f}")

# ---------------------------------------------------------------------------
# 5. 信息论
# ---------------------------------------------------------------------------
print("\n=== 5. 熵、交叉熵、KL、互信息（单位：bit）===")
def H(p): p = p[p > 0]; return -np.sum(p * np.log2(p))
def CE(p, q): return -np.sum(p * np.log2(q))
def KL(p, q): return np.sum(p * np.log2(p / q))

uniform = np.full(8, 1 / 8)
skewed = np.array([0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.0078125])
print(f"8 个结果均匀分布的熵 {H(uniform):.3f} bit（= log2 8），偏斜分布的熵 {H(skewed):.3f} bit")
print(f"H(p, q) = {CE(skewed, uniform):.3f} = H(p) + KL(p‖q) = {H(skewed):.3f} + {KL(skewed, uniform):.3f}")
p2, q2 = np.array([0.9, 0.1]), np.array([0.5, 0.5])
print(f"KL 不对称：KL(p‖q) = {KL(p2, q2):.3f}，KL(q‖p) = {KL(q2, p2):.3f}")
# 互信息：联合分布表
J = np.array([[0.30, 0.10],
              [0.05, 0.55]])                 # P(X=i, Y=j)
px, py = J.sum(1), J.sum(0)
MI = np.sum(J * np.log2(J / np.outer(px, py)))
print(f"互信息 I(X;Y) = {MI:.3f} bit；H(X) = {H(px):.3f}，H(X|Y) = {H(px) - MI:.3f}")
