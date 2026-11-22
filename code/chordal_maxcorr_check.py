r"""
两块分解 {σ_R, 其余} 的常数 1/(1-ρ)（ρ = σ_R 与其余部分的最大相关系数）对比真实 (⋆) 常数 M。
配合 problems/chordal-remove-b.md §12.5。用法：python3 chordal_maxcorr_check.py
"""
# 两块分解 {σ_R, 其余} 的常数 = 1/(1-ρ)，ρ = σ_R 与其余部分的最大相关系数（HGR）
# ρ² = 算子 T(x,x') = Σ_Y π(Y|x) π(x'|Y) 的第二大特征值；对比真实 (⋆) 常数 M
import sys; sys.path.insert(0, '.')
import numpy as np
from chordal_decomposition_check import glauber_matrix, decomposition_stats
from chordal_mixture_means_check import build as twin_build
from chordal_twin_blocker_combo import build   # 孪生 + 阻挡组合
def rho(n, adj, R, lam):
    states, P, w = glauber_matrix(n, adj, lam)
    Rm = sum(1 << v for v in R)
    xs = [None] + list(R)
    sig = [next((v for v in R if I >> v & 1), None) for I in states]
    Y = {}
    for i, I in enumerate(states): Y.setdefault(I & ~Rm, []).append(i)
    m = len(xs); J = np.zeros((m, len(Y)))                     # 联合分布 π(x, Y)
    for j, (yk, ids) in enumerate(Y.items()):
        for i in ids: J[xs.index(sig[i]), j] += w[i]
    px = J.sum(1); py = J.sum(0)
    K = J / np.sqrt(np.outer(px, py))                           # 归一化后奇异值 = 典型相关
    sv = np.linalg.svd(K, compute_uv=False)
    return sv[1]
print("族                    λ |   ρ_R    1/(1-ρ)   真实 M(⋆)")
for lam in [4, 64]:
    for s in [1, 4, 8]:
        n, adj, R = twin_build(s)
        r = rho(n, adj, R, lam); M = decomposition_stats(n, adj, R, lam)[3]
        print(f"twin s={s:<14}{lam:>4} | {r:7.4f} {1/(1-r):9.2f}   {M:7.3f}")
    for s, u in [(1, 8), (8, 8), (16, 16)]:
        n, adj, R = build(s, u)
        r = rho(n, adj, R, lam); M = decomposition_stats(n, adj, R, lam)[3]
        print(f"twin+blocker s={s},u={u:<4}{lam:>4} | {r:7.4f} {1/(1-r):9.2f}   {M:7.3f}")
