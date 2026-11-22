r"""
统一刻画（problems/chordal-remove-b.md §14）：整个 K = sup_f Var_w(m)/E(f) 只依赖 F = f|_{Ω_∅}，
因此 K = sup_F Var_w(m(F)) / E_tr(F)，其中 E_tr 是 Dirichlet 形式对叶子状态的 Schur 补（迹链）。
Var_w(m) = Σ_x w_x (m_x - Σ w m)²，m_x = <a_x, F>，a_x = μ_x（x ∈ {∅} ∪ R）。
用法：python3 chordal_trace_full_K.py
"""
import numpy as np
from chordal_decomposition_check import glauber_matrix
from chordal_mixture_means_check import build as twin_build, parts
from chordal_lemma_g_check import whiskers, blocker
from chordal_twin_blocker_combo import build as combo_build


def full_K_trace(n, adj, R, lam):
    states, P, w = glauber_matrix(n, adj, lam); Nst = len(states)
    Lm = np.diag(w) @ (np.eye(Nst) - P); Lm = (Lm + Lm.T) / 2          # E(f) = f^T Lm f
    Rm = sum(1 << v for v in R); nbr = [sum(1 << u for u in adj[v]) for v in range(n)]
    c = [i for i, I in enumerate(states) if not I & Rm]; l = [i for i in range(Nst) if states[i] & Rm]
    Ltr = Lm[np.ix_(c, c)] - Lm[np.ix_(c, l)] @ np.linalg.solve(Lm[np.ix_(l, l)], Lm[np.ix_(l, c)])
    piJ = w[c]; pi0 = piJ.sum()
    A, W = [piJ / pi0], [pi0]                                           # x = ∅
    for y in R:
        a = np.array([0. if states[i] & nbr[y] else 1. for i in c])
        if a @ piJ > 0:
            A.append(a * piJ / (a @ piJ)); W.append(lam * (a @ piJ))   # w_y = π(Ω_y) = λ π(A_y)
    A, W = np.array(A), np.array(W)
    abar = W @ A                                                        # Σ_x w_x a_x
    Qv = sum(wx * np.outer(ax - abar, ax - abar) for wx, ax in zip(W, A))   # Var_w(m) = F^T Qv F
    ev, U = np.linalg.eigh(Ltr); keep = ev > 1e-12 * ev.max()
    B = U[:, keep] / np.sqrt(ev[keep])
    return np.linalg.eigvalsh(B.T @ Qv @ B).max()


if __name__ == "__main__":
    print("（单位 n(1+λ)）  迹链给出的 K   vs   直接在全空间上算的 K")
    for lam in [4, 16]:
        for name, g in [("whiskers r=4 m=3", whiskers(4, 3)), ("twin s=8", twin_build(8)),
                        ("blocker u=8", blocker(8)), ("blocker u=32", blocker(32)),
                        ("twin+blocker s=16 u=32", combo_build(16, 32))]:
            n, adj, R = g; sc = n * (1 + lam)
            kt = full_K_trace(n, adj, R, lam) / sc; kd = parts(n, adj, R, lam)[0]
            print(f"  {name:<24}λ={lam:<3}: {kt:8.4f}   {kd:8.4f}", flush=True)
