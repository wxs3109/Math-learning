r"""
恒等式检验（problems/chordal-remove-b.md §13）：
  sup_f 中心部分(f)/E(f) = π_∅ ||Γ^tr||，其中 E_tr 是 Dirichlet 形式对叶子状态的 Schur 补（= Ω_∅ 上的迹链）。
用法：python3 chordal_trace_identity.py
"""
# 恒等式检验：sup_f 中心部分(f)/E(f) = ||Γ^{tr}||，
#   E_tr = Dirichlet 形式的 Schur 补（把叶子状态消去 = 迹链），Γ^{tr}_{yy'} = sqrt(w w') <φ_y, Z_tr φ_y'>
import sys; sys.path.insert(0, '.')
import numpy as np
from chordal_decomposition_check import glauber_matrix
from chordal_mixture_means_check import build as twin_build, parts
from chordal_lemma_g_check import whiskers, blocker
from chordal_twin_blocker_combo import build as combo_build
def trace_gamma(n, adj, R, lam):
    states, P, w = glauber_matrix(n, adj, lam); Nst = len(states)
    Lm = np.diag(w) @ (np.eye(Nst) - P); Lm = (Lm + Lm.T) / 2
    Rm = sum(1 << v for v in R); nbr = [sum(1 << u for u in adj[v]) for v in range(n)]
    c = [i for i, I in enumerate(states) if not I & Rm]; l = [i for i in range(Nst) if states[i] & Rm]
    Ltr = Lm[np.ix_(c, c)] - Lm[np.ix_(c, l)] @ np.linalg.solve(Lm[np.ix_(l, l)], Lm[np.ix_(l, c)])   # Schur 补
    piJ = w[c]; pi0 = piJ.sum(); mu0 = piJ / pi0
    vecs = []
    for y in R:
        a = np.array([0. if states[i] & nbr[y] else 1. for i in c]); p = a @ mu0
        if 0 < p < 1: vecs.append(np.sqrt(lam * pi0 * p) * mu0 * (a / p - 1))      # √w_y · μ0 φ_y
    B = np.array(vecs)
    X = np.linalg.lstsq(Ltr, B.T, rcond=None)[0]                                     # Ltr x = μ0φ（总和为 0）
    G = B @ X; return pi0 * np.linalg.norm((G + G.T) / 2, 2)        # 中心部分前面的 π_∅ 因子
for lam in [4, 16]:
    for name, g in [("whiskers r=4 m=3", whiskers(4, 3)), ("twin s=8", twin_build(8)), ("blocker u=8", blocker(8)),
                    ("blocker u=32", blocker(32)), ("twin+blocker s=16 u=32", combo_build(16, 32))]:
        n, adj, R = g; sc = n * (1 + lam)
        tr = trace_gamma(n, adj, R, lam) / sc; true = parts(n, adj, R, lam)[1]
        print(f"{name:<24}{lam:>4} | ||Γ^tr||={tr:8.4f}   true={true:8.4f}", flush=True)
