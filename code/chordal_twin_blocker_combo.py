r"""
"孪生 + 阻挡"组合（problems/chordal-remove-b.md §12）：
  R = S（s-团）∪ {r}；D = u-团，与 S 全连；v 与 D 全连。
检查：真实 t_rel、||Γ^Z||/(λ̄(n(1+λ)+τ_∅))（引理 G′ 的量）、真实 K 及其中心/叶子部分、(⋆) 常数 M。
用法：python3 chordal_twin_blocker_combo.py
"""
# "孪生 + 阻挡"组合：R = S（s-团）∪ {r}；D = u-团，与 S 全连；v 与 D 全连（公共阻挡点）
# 检查：真实 t_rel、τ_∅、||Γ^Z||（R 的团分解），看 G′ 是否被破坏
import sys; sys.path.insert(0, '.')
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spla
from chordal_nested_gadget import enumerate_is, relaxation_time
def build(s, u):
    adj = []
    def add(): adj.append(set()); return len(adj) - 1
    def e(a, b): adj[a].add(b); adj[b].add(a)
    S = [add() for _ in range(s)]; r = add(); R = S + [r]
    for i, a in enumerate(R):
        for b in R[i + 1:]: e(a, b)
    e(r, add())                                   # r 挂叶子
    D = [add() for _ in range(u)]
    for i, a in enumerate(D):
        for b in D[i + 1:]: e(a, b)
        for x in S: e(x, a)
    v = add()
    for a in D: e(a, v)
    return len(adj), adj, R
def gz_and_tau(n, adj, R, lam):
    states, nbr = enumerate_is(n, adj)
    Rm = sum(1 << v for v in R)
    ids = [I for I in states if not I & Rm]; idx = {I: k for k, I in enumerate(ids)}
    N = len(ids); p_in = lam / (1 + lam); rows, cols, vals = [], [], []; diag = np.zeros(N)
    for k, I in enumerate(ids):
        stay = 0.0
        for v in range(n):
            b = 1 << v
            if Rm >> v & 1: stay += 1 / n; continue
            if I & b: rows.append(k); cols.append(idx[I ^ b]); vals.append((1 - p_in) / n); stay += p_in / n
            elif I & nbr[v]: stay += 1 / n
            else: rows.append(k); cols.append(idx[I | b]); vals.append(p_in / n); stay += (1 - p_in) / n
        diag[k] = stay
    lw = np.array([bin(I).count("1") for I in ids]) * np.log(lam)
    allw = np.array([bin(I).count("1") for I in states]) * np.log(lam)
    mu = np.exp(lw - lw.max()); Z0 = mu.sum(); mu /= Z0; d = np.sqrt(mu)
    pi0 = Z0 / np.exp(allw - lw.max()).sum()
    rows, cols, vals = map(np.array, (rows, cols, vals))
    S = sp.csr_matrix((vals * d[rows] / d[cols], (rows, cols)), shape=(N, N)) + sp.diags(diag)
    L = (sp.identity(N) - S).tocsr()
    ev = spla.eigsh(spla.LinearOperator((N, N), matvec=lambda x: S @ x - d * (d @ x)), k=1, which="LA", return_eigenvectors=False)[0]
    tau0 = 1 / (1 - ev)
    B, X = [], []
    for y in R:
        a = np.array([0. if I & nbr[y] else 1. for I in ids]); p = a @ mu
        b = np.sqrt(lam * pi0 * p) * (a / p - 1) * d; B.append(b)
        x, info = spla.cg(L, b, rtol=1e-11, maxiter=100000); X.append(x - d * (d @ x))
    GZ = np.array(B) @ np.array(X).T; GZ = (GZ + GZ.T) / 2
    return np.linalg.norm(GZ, 2), tau0

if __name__ == "__main__":
    from chordal_mixture_means_check import parts
    from chordal_decomposition_check import decomposition_stats
    print(f"{'s':>3} {'u':>3} {'λ':>4} | t_rel/(n(1+λ))  ||Γ^Z||/(λ̄(n(1+λ)+τ_∅)) | K/(n(1+λ)) 中心 叶子 | M(⋆)")
    for lam in [4, 16]:
        for s, u in [(1, 2), (1, 32), (4, 8), (4, 32), (16, 2), (16, 8), (16, 32)]:
            n, adj, R = build(s, u); sc = n * (1 + lam)
            _, trel = relaxation_time(n, adj, lam)
            g, t0 = gz_and_tau(n, adj, R, lam)
            t, c, l = parts(n, adj, R, lam)
            M = decomposition_stats(n, adj, R, lam)[3]
            print(f"{s:>3} {u:>3} {lam:>4} | {trel/sc:12.3f}  {g/(max(1, lam)*(sc+t0)):14.4f}"
                  f"          | {t:7.3f} {c:6.3f} {l:6.3f} | {M:6.3f}", flush=True)
