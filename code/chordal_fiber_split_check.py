r"""
把 f 拆成"纤维平均 + 纤维内偏差"（problems/chordal-remove-b.md §13）：
  A = sup E_Q(F̄)/E(f)（纤维平均放大能量的倍数），B = 纤维内偏差对中心部分的贡献。
用法：python3 chordal_fiber_split_check.py
"""
# 把任意 f 拆成 纤维平均 F̄∘pr + 纤维内偏差 g，分别测：
#   A = sup_f E_Q(F̄)/E(f)                     （坍缩到纤维平均会不会让能量变大）
#   B = sup_f 中心部分(g_c)/E(f)  [单位 n(1+λ)]  （纤维内偏差对中心部分的贡献）
# 若 A = O(1)、B = poly(λ)·O(1)，则 true ≤ 2·new·A + 2·B，修正方案成立
import sys; sys.path.insert(0, '.')
import numpy as np
from chordal_decomposition_check import glauber_matrix
from chordal_mixture_means_check import build as twin_build, sup_ratio
from chordal_lemma_g_check import whiskers, blocker
from chordal_twin_blocker_combo import build as combo_build
from chordal_nested_gadget import nested_twin_gadget

def split(n, adj, R, lam):
    states, P, w = glauber_matrix(n, adj, lam); Nst = len(states)
    Lm = np.diag(w) @ (np.eye(Nst) - P); Lm = (Lm + Lm.T) / 2          # E(f) = f^T Lm f
    Rm = sum(1 << v for v in R); nbr = [sum(1 << u for u in adj[v]) for v in range(n)]
    idx = {I: i for i, I in enumerate(states)}
    ctr = [i for i, I in enumerate(states) if not I & Rm]; m = len(ctr)
    # 纤维平均算子 A_bar: f ↦ F̄ (长度 m)
    Abar = np.zeros((m, Nst))
    for k, i in enumerate(ctr):
        J = states[i]; mem = [i] + [idx[J | (1 << y)] for y in R if not J & nbr[y]]
        tot = w[mem].sum()
        for j in mem: Abar[k, j] = w[j] / tot
    Ctr = np.zeros((m, Nst)); Ctr[np.arange(m), ctr] = 1                 # f ↦ f|_center
    # 坍缩链 Dirichlet 矩阵（在 Ω_∅ 上）
    piJ = w[ctr]; ins = np.array([[0. if states[i] & nbr[y] else 1. for y in R] for i in ctr])
    C = (piJ[:, None] * P[np.ix_(ctr, ctr)]) * (1 + lam * (ins @ ins.T)); np.fill_diagonal(C, 0); C = (C + C.T) / 2
    LQ = np.diag(C.sum(1)) - C
    # 中心部分二次型（作用在 Ω_∅ 上的 F）：π_∅ Σ w_y (a_y - a_0)·F)^2
    pi0 = piJ.sum(); a0 = piJ / pi0; Qc = np.zeros((m, m))
    for k, y in enumerate(R):
        sel = ins[:, k] > 0
        if not sel.any(): continue
        ay = np.where(sel, piJ, 0) / piJ[sel].sum(); wy = lam * piJ[sel].sum()
        Qc += pi0 * wy * np.outer(ay - a0, ay - a0)
    A = sup_ratio(Abar.T @ LQ @ Abar, Lm)
    G = Ctr - Abar                                                     # f ↦ g_c
    B = sup_ratio(G.T @ Qc @ G, Lm) / (n * (1 + lam))
    T = sup_ratio(Ctr.T @ Qc @ Ctr, Lm) / (n * (1 + lam))
    return A, B, T

print(f"{'族':<26}{'λ':>4} | A=sup E_Q(F̄)/E   B=纤维偏差贡献   true")
for lam in [4, 16, 64]:
    for name, g in [("whiskers r=4 m=3", whiskers(4, 3)), ("twin s=8", twin_build(8)),
                    ("blocker u=8", blocker(8)), ("blocker u=32", blocker(32)),
                    ("twin+blocker s=16 u=8", combo_build(16, 8)), ("twin+blocker s=16 u=32", combo_build(16, 32))]:
        A, B, T = split(*g, lam)
        print(f"{name:<26}{lam:>4} | {A:12.3f}   {B:12.3f}   {T:8.3f}", flush=True)
