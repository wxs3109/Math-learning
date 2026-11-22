r"""
核心事件 B = {D 全空} 的 Green 能量拆解（problems/chordal-remove-b.md §12）：
  G_B = <1_B/p-1, Z(1_B/p-1)>，拆成纤维部分 E1 与环境部分 E2，对比平凡界 τ_∅ κ_B。
用法：python3 chordal_core_env_check.py
"""
# 核心事件 B = {D 全空} 的 Green 能量拆解：纤维部分 E1 vs 环境部分 E2（都除以 p_B^2，即对 G_B 的贡献）
#   对比：E2 的平凡界 τ_∅ κ_B，以及候选界 n(1+λ)·(某个量)
import sys; sys.path.insert(0, '.')
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spla
from chordal_nested_gadget import nested_twin_gadget, nested_gadget, enumerate_is
from chordal_lemma_g_check import blocker

def analyze(n, adj, R, D, lam):
    states, nbr = enumerate_is(n, adj)
    Rm = sum(1 << v for v in R); Dm = sum(1 << v for v in D)
    ids = [I for I in states if not I & Rm]; idx = {I: k for k, I in enumerate(ids)}
    N = len(ids); p_in = lam / (1 + lam)
    rows, cols, vals = [], [], []; diag = np.zeros(N)
    for k, I in enumerate(ids):
        stay = 0.0
        for v in range(n):
            b = 1 << v
            if Rm >> v & 1: stay += 1 / n; continue
            if I & b: rows.append(k); cols.append(idx[I ^ b]); vals.append((1 - p_in) / n); stay += p_in / n
            elif I & nbr[v]: stay += 1 / n
            else: rows.append(k); cols.append(idx[I | b]); vals.append(p_in / n); stay += (1 - p_in) / n
        diag[k] = stay
    sz = np.array([bin(I).count("1") for I in ids]); lw = sz * np.log(lam)
    mu = np.exp(lw - lw.max()); mu /= mu.sum(); d = np.sqrt(mu)
    rows, cols, vals = map(np.array, (rows, cols, vals))
    S = sp.csr_matrix((vals * d[rows] / d[cols], (rows, cols)), shape=(N, N)) + sp.diags(diag)
    L = (sp.identity(N) - S).tocsr()
    def green(phi):                                   # <phi, Z phi>_mu，phi 均值为零
        b = phi * d; x, info = spla.cg(L, b, rtol=1e-11, maxiter=100000); assert info == 0
        x -= d * (d @ x); return b @ x
    ev = spla.eigsh(spla.LinearOperator((N, N), matvec=lambda x: S @ x - d * (d @ x)), k=1, which="LA",
                    return_eigenvectors=False)[0]
    tau0 = 1 / (1 - ev)
    A = np.array([0. if I & Dm else 1. for I in ids]); p = A @ mu
    # 环境 e = J \ D；p_e = P(A | e)
    env = {}
    for k, I in enumerate(ids): env.setdefault(I & ~Dm, []).append(k)
    pe = np.zeros(N)
    for e, ks in env.items():
        tot = mu[ks].sum(); a = sum(mu[k] for k in ks if A[k]); pe[ks] = a / tot
    phi1 = A - pe; phi2 = pe - p
    E1, E2, Eall = green(phi1) / p**2, green(phi2) / p**2, green(A - p) / p**2
    var2 = mu @ (pe - p) ** 2; kappa = var2 / p**2
    s = n * (1 + lam)
    return dict(p=p, kappa=kappa, tau0=tau0 / s, G=Eall / s, E1=E1 / s, E2=E2 / s, triv=tau0 * kappa / s)

print("（除 p、κ 外都以 n(1+λ) 为单位）")
print(f"{'族':<22}{'λ':>4} | {'p_B':>7} {'κ_B':>7} {'τ_∅':>7} | {'G_B':>7} {'E1':>7} {'E2':>7} | τ_∅κ_B（平凡界）")
for lam in [4, 16, 64]:
    for u in [2, 8, 32]:
        n, adj, R = blocker(u)                         # R={0,1}; D = U = 顶点 3..3+u-1
        D = list(range(3, 3 + u))
        r = analyze(n, adj, R, D, lam)
        print(f"{'blocker u='+str(u):<22}{lam:>4} | {r['p']:7.4f} {r['kappa']:7.3f} {r['tau0']:7.3f} | {r['G']:7.3f} {r['E1']:7.3f} {r['E2']:7.3f} | {r['triv']:7.3f}")
    for u in [2, 4, 8]:
        n, adj = nested_twin_gadget(2, u); R = [0, 1]; D = list(range(3, 3 + u))
        r = analyze(n, adj, R, D, lam)
        print(f"{'twin-nested d2 u='+str(u):<22}{lam:>4} | {r['p']:7.4f} {r['kappa']:7.3f} {r['tau0']:7.3f} | {r['G']:7.3f} {r['E1']:7.3f} {r['E2']:7.3f} | {r['triv']:7.3f}")
