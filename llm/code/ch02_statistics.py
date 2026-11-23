r"""
第 2 章 · 统计的配套代码（llm/ch02-statistics.md）。只依赖 numpy：python3 ch02_statistics.py

1. 估计量的偏差：样本方差除以 n 有偏，除以 n−1 无偏。
2. 最大似然：高斯的 MLE；标准误差 ∝ 1/√n；bootstrap 置信区间。
3. 偏差–方差权衡：多项式回归，训练误差一路下降，测试误差先降后升。
4. 正则化 = MAP：岭回归（高斯先验）；add-α 平滑（Dirichlet 先验）。
5. 为什么要留出验证集：用测试集反复挑模型，测试分数会变得过于乐观。
"""
import numpy as np

rng = np.random.default_rng(0)

# ---------------------------------------------------------------------------
# 1. 偏差
# ---------------------------------------------------------------------------
print("=== 1. 估计量的偏差 ===")
sigma2, n = 4.0, 5
X = rng.normal(0, np.sqrt(sigma2), size=(200_000, n))      # 20 万组、每组 5 个样本
v_n = X.var(axis=1, ddof=0)                                  # 除以 n
v_n1 = X.var(axis=1, ddof=1)                                 # 除以 n−1
print(f"真实方差 {sigma2}；除以 n 的平均 {v_n.mean():.3f}（理论 {sigma2*(n-1)/n:.3f}）；"
      f"除以 n−1 的平均 {v_n1.mean():.3f}")

# ---------------------------------------------------------------------------
# 2. 最大似然、标准误差、bootstrap
# ---------------------------------------------------------------------------
print("\n=== 2. 最大似然与不确定性 ===")
mu_true, sd_true = 3.0, 2.0
for n in [10, 100, 1000, 10000]:
    est = rng.normal(mu_true, sd_true, size=(2000, n)).mean(axis=1)   # 2000 次实验各算一次 MLE
    print(f"n={n:<6} μ 的 MLE 的标准差 {est.std():.4f}（理论 σ/√n = {sd_true/np.sqrt(n):.4f}）")

data = rng.normal(mu_true, sd_true, 200)
boot = np.array([rng.choice(data, data.size).mean() for _ in range(5000)])   # 有放回重抽样
lo, hi = np.percentile(boot, [2.5, 97.5])
se = data.std(ddof=1) / np.sqrt(data.size)
print(f"一份 n=200 的数据：均值 {data.mean():.3f}；bootstrap 95% 区间 [{lo:.3f}, {hi:.3f}]；"
      f"正态近似 [{data.mean()-1.96*se:.3f}, {data.mean()+1.96*se:.3f}]")

# ---------------------------------------------------------------------------
# 3. 偏差–方差：多项式回归
# ---------------------------------------------------------------------------
print("\n=== 3. 偏差–方差权衡（多项式回归）===")
f = lambda x: np.sin(2 * np.pi * x)
x_tr = rng.uniform(0, 1, 20); y_tr = f(x_tr) + 0.3 * rng.normal(size=20)
x_te = rng.uniform(0, 1, 2000); y_te = f(x_te) + 0.3 * rng.normal(size=2000)
def design(x, deg):
    return np.vander(2 * x - 1, deg + 1, increasing=True)      # 先把 x 映射到 [−1,1]，数值更稳定
def fit(deg, lam=0.0):
    A = design(x_tr, deg)
    return np.linalg.solve(A.T @ A + lam * np.eye(deg + 1), A.T @ y_tr)
def mse(w, x, y, deg):
    return np.mean((design(x, deg) @ w - y) ** 2)
print("  次数   训练 MSE   测试 MSE")
for deg in [1, 3, 5, 9, 15]:
    w = fit(deg)
    print(f"  {deg:>4}   {mse(w, x_tr, y_tr, deg):8.4f}   {mse(w, x_te, y_te, deg):8.4f}")
print(f"  （噪声方差 0.09 是测试 MSE 的下界）")

# ---------------------------------------------------------------------------
# 4. 正则化 = MAP
# ---------------------------------------------------------------------------
print("\n=== 4. 正则化 = 最大后验 ===")
deg = 15
print("  岭回归（15 次多项式）：λ   训练 MSE   测试 MSE   ‖w‖")
for lam in [0.0, 1e-4, 1e-2, 1.0]:
    w = fit(deg, lam)
    print(f"     {lam:<8g} {mse(w, x_tr, y_tr, deg):8.4f}   {mse(w, x_te, y_te, deg):8.4f}   {np.linalg.norm(w):10.1f}")

# 类别分布 + Dirichlet 先验 ⇒ add-α 平滑。没见过的类别不再是概率 0。
V = 50
true_p = rng.dirichlet(np.ones(V) * 0.5)
train = rng.choice(V, 60, p=true_p); test = rng.choice(V, 5000, p=true_p)
counts = np.bincount(train, minlength=V)
print("  类别分布（V=50，只有 60 个训练样本）：α   测试集平均负对数似然")
for alpha in [0.0, 0.01, 0.1, 0.5, 2.0]:
    q = (counts + alpha) / (counts.sum() + alpha * V)
    nll = -np.mean(np.log(q[test])) if alpha > 0 else np.inf if np.any(q[test] == 0) else -np.mean(np.log(q[test]))
    print(f"     {alpha:<5}  {nll:.4f}" + ("（测试集出现了训练没见过的类别 ⇒ 无穷大）" if not np.isfinite(nll) else ""))
print(f"  参考：真实分布的熵 {-(true_p * np.log(true_p)).sum():.4f}（任何模型的下界）")

# ---------------------------------------------------------------------------
# 5. 为什么要留出验证集
# ---------------------------------------------------------------------------
print("\n=== 5. 用测试集挑模型会高估效果 ===")
# 100 个"模型"其实都在瞎猜（准确率真值 0.5）；在 200 道题的测试集上挑分数最高的
scores = rng.binomial(200, 0.5, size=(5000, 100)) / 200
best = scores.max(axis=1)
print(f"每个模型真实准确率都是 0.500；在同一个测试集上挑出的最好模型，平均测得 {best.mean():.3f}")
print("⇒ 挑模型要用验证集，最终只在测试集上报告一次。")
