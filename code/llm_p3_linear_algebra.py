r"""
前置知识 P3 · 线性代数的配套代码（topics/llm-p3-linear-algebra.md）。只依赖 numpy：python3 llm_p3_linear_algebra.py

1. 矩阵乘法的四种读法，逐一验证。
2. 秩：外积之和的秩；随机矩阵满秩。
3. 对称矩阵的谱分解；幂迭代求最大特征值。
4. SVD 与低秩近似：截断误差 = 丢掉的奇异值平方和（Eckart–Young）；LoRA 的参数量。
5. 正交投影与最小二乘。
6. 高维几何：随机向量几乎正交；范数集中在 √d 附近。
7. 批量矩阵乘法与广播：(B, T, d) @ (d, k)。
"""
import numpy as np

rng = np.random.default_rng(0)

# ---------------------------------------------------------------------------
# 1. 矩阵乘法的四种读法
# ---------------------------------------------------------------------------
print("=== 1. 矩阵乘法的四种读法 ===")
A = rng.normal(size=(4, 3)); B = rng.normal(size=(3, 5))
C = A @ B
by_entry = np.array([[A[i] @ B[:, j] for j in range(5)] for i in range(4)])     # 内积
by_column = np.column_stack([A @ B[:, j] for j in range(5)])                    # A 作用在 B 的每一列
by_row = np.vstack([A[i] @ B for i in range(4)])                                # A 的每一行乘 B
by_outer = sum(np.outer(A[:, k], B[k]) for k in range(3))                       # 外积之和
for name, M in [("逐元素内积", by_entry), ("按列", by_column), ("按行", by_row), ("外积之和", by_outer)]:
    print(f"  {name:<8}与 A@B 的最大差：{np.abs(M - C).max():.1e}")

# ---------------------------------------------------------------------------
# 2. 秩
# ---------------------------------------------------------------------------
print("\n=== 2. 秩 ===")
r = 3
M = sum(np.outer(rng.normal(size=50), rng.normal(size=40)) for _ in range(r))    # 3 个外积之和
print(f"50×40 矩阵 = 3 个外积之和 ⇒ 秩 {np.linalg.matrix_rank(M)}")
print(f"50×40 随机高斯矩阵的秩 {np.linalg.matrix_rank(rng.normal(size=(50, 40)))}（满秩 = 40）")

# ---------------------------------------------------------------------------
# 3. 对称矩阵的谱分解与幂迭代
# ---------------------------------------------------------------------------
print("\n=== 3. 谱分解与幂迭代 ===")
X = rng.normal(size=(200, 6)) @ np.diag([3, 2, 1, .5, .3, .1])
S = X.T @ X / 200                                  # 协方差矩阵：对称、半正定
lam, Q = np.linalg.eigh(S)
print("特征值（从小到大）：", np.round(lam, 4))
print(f"Q 正交：‖QᵀQ − I‖ = {np.abs(Q.T @ Q - np.eye(6)).max():.1e}；"
      f"重构 ‖Q diag(λ) Qᵀ − S‖ = {np.abs(Q @ np.diag(lam) @ Q.T - S).max():.1e}")
v = rng.normal(size=6)
for _ in range(100):
    v = S @ v; v /= np.linalg.norm(v)              # 幂迭代：反复乘 S 再归一化
print(f"幂迭代 100 步：Rayleigh 商 vᵀSv = {v @ S @ v:.6f}，最大特征值 {lam[-1]:.6f}")

# ---------------------------------------------------------------------------
# 4. SVD 与低秩近似
# ---------------------------------------------------------------------------
print("\n=== 4. SVD 与低秩近似 ===")
# 一个"近似低秩"的矩阵：奇异值按幂律衰减
U0, _ = np.linalg.qr(rng.normal(size=(300, 300)))
V0, _ = np.linalg.qr(rng.normal(size=(200, 200)))
sv = 1.0 / (1 + np.arange(200)) ** 1.2
W = U0[:, :200] @ np.diag(sv) @ V0.T
U, s, Vt = np.linalg.svd(W, full_matrices=False)
print("  秩 k   ‖W − W_k‖_F²    丢掉的 σ² 之和    保留的能量")
for k in [1, 5, 20, 50]:
    Wk = U[:, :k] @ np.diag(s[:k]) @ Vt[:k]
    err = np.linalg.norm(W - Wk) ** 2
    print(f"  {k:>4}   {err:.6f}       {np.sum(s[k:] ** 2):.6f}         {np.sum(s[:k]**2)/np.sum(s**2):.3f}")
d_out, d_in, rk = 4096, 4096, 8
print(f"LoRA：{d_out}×{d_in} 的更新量 ΔW 全量要 {d_out*d_in:,} 个参数；"
      f"写成 BA（秩 {rk}）只要 {rk*(d_out+d_in):,} 个，是原来的 {rk*(d_out+d_in)/(d_out*d_in):.2%}")

# ---------------------------------------------------------------------------
# 5. 正交投影与最小二乘
# ---------------------------------------------------------------------------
print("\n=== 5. 投影与最小二乘 ===")
Xd = rng.normal(size=(100, 4)); y = Xd @ np.array([1., -2., .5, 3.]) + 0.1 * rng.normal(size=100)
w_normal = np.linalg.solve(Xd.T @ Xd, Xd.T @ y)          # 正规方程
w_lstsq = np.linalg.lstsq(Xd, y, rcond=None)[0]           # 基于 SVD 的求解器
P = Xd @ np.linalg.solve(Xd.T @ Xd, Xd.T)                 # 投影矩阵
resid = y - P @ y
print(f"正规方程与 lstsq 的差 {np.abs(w_normal - w_lstsq).max():.1e}；估计的 w = {np.round(w_normal, 3)}")
print(f"P² = P：{np.abs(P @ P - P).max():.1e}；Pᵀ = P：{np.abs(P.T - P).max():.1e}；"
      f"残差与每一列正交：{np.abs(Xd.T @ resid).max():.1e}")

# ---------------------------------------------------------------------------
# 6. 高维几何
# ---------------------------------------------------------------------------
print("\n=== 6. 高维几何 ===")
for d in [3, 64, 1024, 12288]:
    a = rng.normal(size=(2000, d)); b = rng.normal(size=(2000, d))
    cos = np.sum(a * b, 1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    norms = np.linalg.norm(a, axis=1) / np.sqrt(d)
    print(f"  d={d:<6} 两个随机向量夹角余弦的标准差 {cos.std():.4f}（≈1/√d = {1/np.sqrt(d):.4f}）；"
          f"‖x‖/√d 的范围 [{norms.min():.3f}, {norms.max():.3f}]")

# ---------------------------------------------------------------------------
# 7. 批量矩阵乘法与广播
# ---------------------------------------------------------------------------
print("\n=== 7. 形状：批量矩阵乘法与广播 ===")
Bsz, T, d, k = 2, 5, 16, 8
Xb = rng.normal(size=(Bsz, T, d)); Wq = rng.normal(size=(d, k)); bias = rng.normal(size=k)
Qb = Xb @ Wq + bias                                       # (B,T,d)@(d,k) → (B,T,k)，bias 沿前两维广播
scores = Qb @ Qb.transpose(0, 2, 1)                       # (B,T,k)@(B,k,T) → (B,T,T)
print(f"X {Xb.shape} @ W {Wq.shape} + b {bias.shape} → {Qb.shape}；Q Qᵀ → {scores.shape}")
print(f"与逐个样本、逐个位置计算的最大差：{np.abs(Qb[1, 3] - (Xb[1, 3] @ Wq + bias)).max():.1e}")
