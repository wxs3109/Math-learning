r"""
引理 G 的数值检查（配合 problems/chordal-remove-b.md §8）。

比较三个量（单位都是 n(1+λ)）：
  trace  = Σ_y w_y G_y                                    原来的引理 G（迹的形式）
  opnorm = ||Γ^Z||,  Γ^Z_{yy'} = sqrt(w_y w_y') <h_y-1, Z (h_y'-1)>_{μ0}
                                                          改进的引理 G'（算子范数）
  center = sup_f [π_∅ Σ_y w_y d_y²] / E(f,f)              真正需要控制的中心部分
其中 Z = (I - P_∅)^+，h_y = 1_{A_y}/p_y。

图族：
  whiskers(r, m)：r-团，每个点挂 m 个叶子（A_y 两两独立）——迹的形式会随 r 增长
  blocker(u)    ：R = {y, r}；U = K_u 全部连到 y；一个点 v 连到 U 全部（公共阻挡点）
                   ——真实的 t_rel 会随 u 增长，饱和在 ~λ 的量级

用法：python3 chordal_lemma_g_check.py
"""
import random
import numpy as np
import networkx as nx
from chordal_decomposition_check import (glauber_matrix, spectral_gap, decomposition_stats,
                                         random_connected_chordal, random_split_graph, adj_from_nx)
from chordal_mixture_means_check import parts


def green_gram(n, adj, R, lam):
    """返回 (trace, ||Γ^Z||, center)，单位 n(1+λ)。"""
    states, P, w = glauber_matrix(n, adj, lam)
    ids = [i for i, I in enumerate(states) if not any(I >> v & 1 for v in R)]
    Pe = P[np.ix_(ids, ids)].copy()
    Pe[np.diag_indices_from(Pe)] += 1 - Pe.sum(1)           # P_∅：R 冻结为空
    mu = w[ids] / w[ids].sum()
    pi0 = w[ids].sum()
    d = np.sqrt(mu)
    S = (d[:, None] * Pe) / d[None, :]
    S = (S + S.T) / 2                                         # 对称化
    Zs = np.linalg.pinv(np.eye(len(ids)) - S)                 # 对称坐标下的 Z
    nbr = [sum(1 << u for u in adj[v]) for v in range(n)]
    B = []
    for y in R:
        a = np.array([0.0 if states[i] & nbr[y] else 1.0 for i in ids])   # 1_{A_y}
        p = a @ mu
        if 0 < p < 1 - 1e-12:
            B.append(np.sqrt(lam * pi0 * p) * (a / p - 1) * d)            # sqrt(w_y)(h_y-1)
    B = np.array(B)
    GZ = B @ Zs @ B.T
    s = n * (1 + lam)
    return np.trace(GZ) / s, np.linalg.norm(GZ, 2) / s, parts(n, adj, R, lam)[1]


def whiskers(r, m):
    adj = [set() for _ in range(r)]
    for i in range(r):
        for j in range(i + 1, r):
            adj[i].add(j); adj[j].add(i)
    for i in range(r):
        for _ in range(m):
            adj.append({i}); adj[i].add(len(adj) - 1)
    return len(adj), adj, list(range(r))


def blocker(u):
    adj = []

    def add():
        adj.append(set()); return len(adj) - 1

    def e(a, b):
        adj[a].add(b); adj[b].add(a)
    y, r = add(), add(); e(y, r); e(r, add())                 # r 挂一个叶子
    U = [add() for _ in range(u)]
    for i, a in enumerate(U):
        e(y, a)
        for b in U[i + 1:]:
            e(a, b)
    v = add()
    for a in U:
        e(a, v)                                               # 公共阻挡点
    return len(adj), adj, [y, r]


if __name__ == "__main__":
    print("(1) whiskers：迹 vs 算子范数 vs 真实中心部分")
    for r, m in [(2, 3), (4, 3), (5, 2), (6, 1), (8, 1), (9, 1)]:
        n, adj, R = whiskers(r, m)
        for lam in [1, 16]:
            t, o, c = green_gram(n, adj, R, lam)
            print(f"  r={r} m={m} λ={lam:<3}: trace={t:6.3f}  ||Γ^Z||={o:6.3f}  center={c:6.3f}")

    print("\n(2) 公共阻挡点：真实 t_rel/(n(1+λ))、(⋆) 常数 M（R={y,r}）、||Γ^Z||/λ̄")
    for lam in [4, 16, 64]:
        for u in [1, 4, 16, 64]:
            n, adj, R = blocker(u)
            _, P, w = glauber_matrix(n, adj, lam)
            trel = 1 / spectral_gap(P, w) / (n * (1 + lam))
            M = decomposition_stats(n, adj, R, lam)[3]
            o = green_gram(n, adj, R, lam)[1]
            print(f"  λ={lam:<3} u={u:<3}: t_rel={trel:7.3f}  M={M:6.3f}  ||Γ^Z||/λ̄={o/max(1, lam):.3f}")

    print("\n(3) 随机弦图 / split 图：max ||Γ^Z|| / λ̄")
    rng = random.Random(5)
    worst = 0
    for k in range(100):
        G0 = (random_connected_chordal(rng.randint(7, 11), rng) if k % 2
              else random_split_graph(rng.randint(3, 6), rng.randint(3, 6), rng))
        n, adj, G = adj_from_nx(G0)
        for R in nx.chordal_graph_cliques(G):
            for lam in [4, 64]:
                worst = max(worst, green_gram(n, adj, list(R), lam)[1] / max(1, lam))
    print(f"  {worst:.3f}")
