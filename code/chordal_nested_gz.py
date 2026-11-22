r"""
在嵌套孪生阻挡结构上计算引理 G′ 的关键量 ||Γ^Z|| / (λ̄ n(1+λ))，R = {y, r}。
用共轭梯度（CG）解 (I-S)x = b，避免稠密求逆，可以算到 ~10^5 个状态。
配合 problems/chordal-remove-b.md §10。用法：python3 chordal_nested_gz.py
"""
# 在嵌套孪生阻挡结构上计算 ||Γ^Z|| / (λ̄ n(1+λ))（R = {y, r}），用 CG 解 (I-S)x = b，避免稠密求逆
import sys, time; sys.path.insert(0, '.')
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spla
from chordal_nested_gadget import nested_twin_gadget, enumerate_is
def gz(depth, u, lam):
    n, adj = nested_twin_gadget(depth, u); R = [0, 1]
    states, nbr = enumerate_is(n, adj)
    ids = [I for I in states if not (I & 0b11)]
    idx = {I: k for k, I in enumerate(ids)}; N = len(ids); p_in = lam / (1 + lam)
    rows, cols, vals = [], [], []; diag = np.zeros(N)
    for k, I in enumerate(ids):
        stay = 0.0
        for v in range(n):
            b = 1 << v
            if v in R: stay += 1.0 / n; continue                  # R 冻结为空
            if I & b: rows.append(k); cols.append(idx[I ^ b]); vals.append((1 - p_in) / n); stay += p_in / n
            elif I & nbr[v]: stay += 1.0 / n
            else: rows.append(k); cols.append(idx[I | b]); vals.append(p_in / n); stay += (1 - p_in) / n
        diag[k] = stay
    sz = np.array([bin(I).count("1") for I in ids]); lw = sz * np.log(lam)
    mu = np.exp(lw - lw.max()); mu /= mu.sum(); d = np.sqrt(mu)
    rows, cols, vals = map(np.array, (rows, cols, vals))
    S = sp.csr_matrix((vals * d[rows] / d[cols], (rows, cols)), shape=(N, N)) + sp.diags(diag)
    L = sp.identity(N) - S
    # π_∅ 需要完整链：π_∅ = Z_∅ / Z_total
    allw = np.array([bin(I).count("1") for I in states]) * np.log(lam)
    pi0 = np.exp(lw - lw.max()).sum() / np.exp(allw - lw.max()).sum()
    B = []
    for y in R:
        a = np.array([0. if I & nbr[y] else 1. for I in ids]); p = a @ mu
        B.append(np.sqrt(lam * pi0 * p) * (a / p - 1) * d)
    B = np.array(B)
    X = []
    for b in B:
        x, info = spla.cg(L, b, rtol=1e-10, maxiter=50000); assert info == 0
        X.append(x - d * (d @ x))
    GZ = B @ np.array(X).T; GZ = (GZ + GZ.T) / 2
    return n, N, np.linalg.norm(GZ, 2) / (max(1, lam) * n * (1 + lam))
for lam in [4, 16]:
    for u in [2, 4, 8, 16, 24]:
        t0 = time.time(); n, N, r = gz(2, u, lam)
        print(f"λ={lam} depth=2 u={u:<3} n={n:<4} |Ω_∅|={N:<7} ||Γ^Z||/(λ̄ n(1+λ)) = {r:.4f} ({time.time()-t0:.0f}s)", flush=True)
