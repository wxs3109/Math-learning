r"""
对角元与块结构的检查（配合 problems/chordal-remove-b.md §9）。

  命题 D′：U_y = N(y)\R 是团时，按"环境 e = J\U_y"做纤维分解，
           ⟨φ_y, Z φ_y⟩ ≤ 2 n(1+λ) Σ_e μ(e) λN(e)/Z(e)² + 2 τ_∅ Var_μ(p_e)
  引理 E ：若 U_y 与 U_y' 落在 G-R 的不相交连通分量集合里，则 Γ^Z_{yy'} = 0

用法：python3 chordal_diag_block_check.py
"""
import random
import numpy as np
import networkx as nx
from chordal_decomposition_check import (glauber_matrix, spectral_gap,
                                         random_connected_chordal, random_split_graph, adj_from_nx)
from chordal_lemma_g_check import blocker, whiskers


def blocker_k(u, k):                      # 阻挡者是 k-团 V（孪生阻挡）
    n, adj, R = blocker(u)
    v = n - 1; U = sorted(adj[v]); new = []
    for _ in range(k - 1):
        adj.append(set()); x = len(adj) - 1; new.append(x)
        for a in U: adj[a].add(x); adj[x].add(a)
        for b in [v] + new[:-1]: adj[b].add(x); adj[x].add(b)
    return len(adj), adj, R


def fiber_bound_ratio(n, adj, R, lam):
    """对每个 U_y 为团的 y，返回 ⟨φ,Zφ⟩ / [2n(1+λ)Σ_e μ(e)λN/Z² + 2τ_∅ Var(p_e)]（命题 D′ 预言 ≤ 1）。"""
    states, P, w = glauber_matrix(n, adj, lam)
    ids = [i for i, I in enumerate(states) if not any(I >> v & 1 for v in R)]
    Pe = P[np.ix_(ids, ids)].copy(); Pe[np.diag_indices_from(Pe)] += 1 - Pe.sum(1)
    mu = w[ids] / w[ids].sum(); d = np.sqrt(mu)
    S = (d[:, None] * Pe) / d[None, :]; S = (S + S.T) / 2
    Zs = np.linalg.inv(np.eye(len(ids)) - S + np.outer(d, d))       # 均值零子空间上的 Z
    tau0 = 1 / spectral_gap(Pe, mu)
    nbr = [sum(1 << u for u in adj[v]) for v in range(n)]
    idx = {i: k for k, i in enumerate(ids)}
    out = []
    for y in R:
        U = [z for z in adj[y] if z not in R]
        if not U or any(b not in adj[a] for a in U for b in U if a != b):
            continue                                                   # 只处理 U 是团
        Um = sum(1 << z for z in U)
        A = np.array([0. if states[i] & Um else 1. for i in ids]); p = A @ mu
        if not 0 < p < 1:
            continue
        lhs = ((A - p) * d) @ Zs @ ((A - p) * d)                      # ⟨φ, Zφ⟩, φ = 1_A - p
        env = {}                                                       # 环境 e = J \ U
        for i in ids:
            env.setdefault(states[i] & ~Um, []).append(i)
        pe = np.zeros(len(ids)); t1 = 0.0
        for e, lst in env.items():
            k0 = idx[next(i for i in lst if not states[i] & Um)]      # 纤维里 U 为空的那个状态
            N = sum(1 for z in U if not e & nbr[z]); Z = 1 + lam * N
            for i in lst:
                pe[idx[i]] = 1 / Z                                     # p_e = P(A | e)
            t1 += mu[k0] * lam * N / Z ** 2
        var = mu @ pe ** 2 - (mu @ pe) ** 2
        out.append(lhs / (2 * n * (1 + lam) * t1 + 2 * tau0 * var))
    return out


def max_offblock(n, adj, R, lam):
    """Γ^Z 中 U_y 落在不相交分量上的 (y,y') 的最大 |非对角元| / 最大对角元（引理 E 预言 = 0）。"""
    states, P, w = glauber_matrix(n, adj, lam)
    ids = [i for i, I in enumerate(states) if not any(I >> v & 1 for v in R)]
    Pe = P[np.ix_(ids, ids)].copy(); Pe[np.diag_indices_from(Pe)] += 1 - Pe.sum(1)
    mu = w[ids] / w[ids].sum(); d = np.sqrt(mu)
    S = (d[:, None] * Pe) / d[None, :]; S = (S + S.T) / 2
    Zs = np.linalg.inv(np.eye(len(ids)) - S + np.outer(d, d))
    nbr = [sum(1 << u for u in adj[v]) for v in range(n)]
    phis = []
    for y in R:
        a = np.array([0. if states[i] & nbr[y] else 1. for i in ids]); p = a @ mu
        phis.append((a / p - 1) * d)
    M = np.array(phis) @ Zs @ np.array(phis).T
    return np.abs(M - np.diag(np.diag(M))).max() / np.diag(M).max()


if __name__ == "__main__":
    print("(1) 命题 D′（纤维分解）：左/右 应 ≤ 1")
    for name, g in [("blocker u=16", blocker(16)), ("twin-blocker u=k=8", blocker_k(8, 8))]:
        for lam in [4, 64]:
            print(f"  {name:<20} λ={lam:<3}:", [round(r, 4) for r in fiber_bound_ratio(*g, lam)])
    rng = random.Random(9); worst = 0
    for k in range(120):
        G0 = (random_connected_chordal(rng.randint(7, 11), rng) if k % 2
              else random_split_graph(rng.randint(3, 6), rng.randint(3, 6), rng))
        n, adj, G = adj_from_nx(G0)
        for R in nx.chordal_graph_cliques(G):
            for lam in [4, 64]:
                for r in fiber_bound_ratio(n, adj, list(R), lam):
                    worst = max(worst, r)
    print(f"  随机弦图 / split 图：max 左/右 = {worst:.4f}")

    print("\n(2) 引理 E（块对角）：whiskers 中各 U_y 落在不相交分量上")
    for lam in [4, 64]:
        print(f"  whiskers r=4 m=2 λ={lam}: max|非对角|/max 对角 = {max_offblock(*whiskers(4, 2), lam):.2e}")
