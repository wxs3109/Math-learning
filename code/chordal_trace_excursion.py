r"""
迹链的"远足分解"与孪生 + 阻挡例子的显式计算（problems/chordal-remove-b.md §15）。

1. 引理 X（远足分解）：R 是团 ⇒ 不同叶子块 Ω_y 之间没有直接转移，于是迹链的 Dirichlet 形式精确地是
       E_tr(F) = E_∅(F) + λ Σ_y  F^T [ cΠ_y (cΠ_y + L_y)^{-1} L_y ] F ,   c = 1/(n(1+λ))，
   其中 E_∅ 是中心块内部的边（被杀死的链），Π_y = diag(π|A_y)，L_y 是 A_y 的诱导子图上的 Laplacian
   （边权与中心块相同）。check_lemma_X 在一般小图上逐项核对。
2. 孪生 + 阻挡（R = S ∪ {r}，D 是 u-团且与 S 全连，v 与 D 全连，r 挂一个叶子 q）按对称性合并后，
   中心块只有 6 个状态 (b, q)，b ∈ {e(D、v 全空), v(v 占), d(D 有点占)}，q ∈ {0,1}。
   lumped_K 用 6×6 矩阵算出迹链下的 K 和被杀死链下的 K，单位 n(1+λ)（与 n 无关，可以取到 s,u ~ 10^5）。
用法：python3 chordal_trace_excursion.py
"""
import numpy as np
from chordal_decomposition_check import glauber_matrix, random_connected_chordal, adj_from_nx
from chordal_twin_blocker_combo import build as combo_build
from chordal_trace_full_K import full_K_trace


def gen_max_eig(Qv, L):
    """sup_F F^T Qv F / F^T L F（L 半正定，只在 L 的值域上取）。"""
    ev, U = np.linalg.eigh(L); keep = ev > 1e-12 * ev.max()
    B = U[:, keep] / np.sqrt(ev[keep])
    return np.linalg.eigvalsh(B.T @ Qv @ B).max()


def check_lemma_X(n, adj, R, lam):
    """返回 max |Schur 补 − (E_∅ + λ Σ_y cΠ(cΠ+L)^{-1}L)|，应为舍入误差量级。"""
    states, P, w = glauber_matrix(n, adj, lam); N = len(states)
    Lm = np.diag(w) @ (np.eye(N) - P); Lm = (Lm + Lm.T) / 2
    Rm = sum(1 << v for v in R); nbr = [sum(1 << u for u in adj[v]) for v in range(n)]
    cen = [i for i, I in enumerate(states) if not I & Rm]; lf = [i for i in range(N) if states[i] & Rm]
    Ltr = Lm[np.ix_(cen, cen)] - Lm[np.ix_(cen, lf)] @ np.linalg.solve(Lm[np.ix_(lf, lf)], Lm[np.ix_(lf, cen)])
    # 被杀死的链：只保留中心块内部的边（对角元 = 中心内部边权之和）
    C = -Lm[np.ix_(cen, cen)].copy(); np.fill_diagonal(C, 0); E0 = np.diag(C.sum(1)) - C
    c = 1 / (n * (1 + lam)); form = E0.copy()
    for y in R:
        A = [k for k, i in enumerate(cen) if not states[i] & nbr[y]]   # A_y 在中心块里的下标
        CA = C[np.ix_(A, A)]; LA = np.diag(CA.sum(1)) - CA                # A_y 诱导子图的 Laplacian
        Pi = np.diag(w[[cen[k] for k in A]])
        form[np.ix_(A, A)] += lam * (c * Pi) @ np.linalg.solve(c * Pi + LA, LA)
    form = (form + form.T) / 2
    return np.abs(Ltr - form).max() / np.abs(Ltr).max()


def lumped_K(s, u, lam, excursions=True):
    """孪生 + 阻挡的合并计算。返回 K/(n(1+λ))（excursions=False 时是被杀死链的版本）。"""
    c = 1.0                                     # 所有边权都正比于 c，取 c=1 即得以 n(1+λ) 为单位的 K
    nu = np.array([1.0, lam, u * lam])          # b = e, v, d 的权重（d 合并了 u 个状态）
    st = [(b, q) for q in (0, 1) for b in range(3)]; ix = {x: k for k, x in enumerate(st)}
    piu = np.array([nu[b] * lam ** q for b, q in st])                    # 中心块未归一化权重
    Z = piu.sum() + s * lam * piu[[ix[(0, 0)], ix[(1, 0)], ix[(0, 1)], ix[(1, 1)]]].sum() \
        + lam * piu[[ix[(b, 0)] for b in range(3)]].sum()
    pi = piu / Z
    def lap(edges, idx):
        L = np.zeros((len(idx), len(idx))); loc = {k: j for j, k in enumerate(idx)}
        for a, b, g in edges:
            i, j = loc[a], loc[b]; L[i, i] += g; L[j, j] += g; L[i, j] -= g; L[j, i] -= g
        return L
    # 中心块内部边：权 = 较大状态的 π × c
    E = []
    for q in (0, 1):
        E.append((ix[(0, q)], ix[(1, q)], pi[ix[(1, q)]] * c))           # e – v
        E.append((ix[(0, q)], ix[(2, q)], pi[ix[(2, q)]] * c))           # e – d（u 条边合并）
    for b in range(3):
        E.append((ix[(b, 0)], ix[(b, 1)], pi[ix[(b, 1)]] * c))           # q 翻转
    allc = list(range(6)); Ltr = lap(E, allc)
    AS = [ix[(0, 0)], ix[(1, 0)], ix[(0, 1)], ix[(1, 1)]]                  # A_y, y ∈ S：D 全空
    Ar = [ix[(b, 0)] for b in range(3)]                                    # A_r：q = 0
    if excursions:
        for A, mult in [(AS, s), (Ar, 1)]:
            LA = lap([e for e in E if e[0] in A and e[1] in A], A); Pi = np.diag(pi[A])
            Ltr[np.ix_(A, A)] += mult * lam * (c * Pi) @ np.linalg.solve(c * Pi + LA, LA)
    Ltr = (Ltr + Ltr.T) / 2
    pi0 = pi.sum(); vecs, wts = [pi / pi0], [pi0]
    for A, mult in [(AS, s), (Ar, 1)]:
        a = np.zeros(6); a[A] = pi[A]; pA = a.sum()
        vecs.append(a / pA); wts.append(mult * lam * pA)
    vecs, wts = np.array(vecs), np.array(wts)
    abar = wts @ vecs
    Qv = sum(wx * np.outer(ax - abar, ax - abar) for wx, ax in zip(wts, vecs))
    return gen_max_eig(Qv, Ltr)


if __name__ == "__main__":
    import networkx as nx
    print("① 引理 X（远足分解）的相对误差：")
    import random; rng = random.Random(0)
    for t in range(4):
        G = random_connected_chordal(9, rng)
        n, adj, G = adj_from_nx(G)
        R = list(max(nx.find_cliques(G), key=len))
        for lam in [0.5, 4, 16]:
            print(f"   随机弦图 #{t} |R|={len(R)} λ={lam:<4}: {check_lemma_X(n, adj, R, lam):.1e}")
    for s, u in [(2, 3), (4, 4)]:
        n, adj, R = combo_build(s, u)
        print(f"   孪生+阻挡 s={s} u={u} λ=4  : {check_lemma_X(n, adj, R, 4):.1e}")

    print("\n② 合并计算 vs 全空间计算（单位 n(1+λ)）：")
    for lam in [4, 16]:
        for s, u in [(1, 2), (4, 8), (16, 32)]:
            n, adj, R = combo_build(s, u)
            print(f"   s={s:<3} u={u:<3} λ={lam:<3}: 合并 {lumped_K(s, u, lam):.4f}   全空间 "
                  f"{full_K_trace(n, adj, R, lam) / (n * (1 + lam)):.4f}", flush=True)

    print("\n③ 大规模扫描：迹链 K  vs  被杀死链 K  vs  解析上界 6(1+λ)（单位 n(1+λ)）")
    print(f"   {'λ':>5} {'s':>6} {'u':>6} | {'迹链 K':>8} {'杀死链 K':>10} | {'K/(1+λ)':>8}")
    for lam in [1, 4, 16, 64, 256, 1024]:
        for s, u in [(1, 10**4), (10**2, 10**2), (10**2, 10**4), (10**4, 10**2), (10**4, 10**4), (10**5, 10**5)]:
            kt, kk = lumped_K(s, u, lam), lumped_K(s, u, lam, excursions=False)
            print(f"   {lam:>5} {s:>6} {u:>6} | {kt:8.3f} {kk:10.1f} | {kt / (1 + lam):8.4f}")
