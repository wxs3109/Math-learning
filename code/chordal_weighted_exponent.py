r"""
加权 hardcore 的"减速指数"搜索（problems/chordal-remove-b.md §17）。

由膨胀引理（chordal_tree_blowup.py），孪生类大小为 m_k 的弦图 ⇔ H 上活跃度 Λ_k = m_k λ 的 hardcore 动力学
（插入速率 Λ_k c、删除速率 c）。令 Λ_k = Λ^{e_k}，e_k ∈ [0,1]，看 t_rel·c 随 Λ 的增长指数
    α(H, e) = d log(t_rel c) / d log Λ    （用 Λ=10^6 与 10^9 两点的斜率估计）。
问题：max_{|H|=k, e} α 会不会随 k 无界增长？分裂竞争（4 点路径，e=(1/2,1,1,1/2)）给出 α=1/2。
H 取所有 k 点无标号树（networkx），以及若干随机弦图；e 在网格 {0,1/8,…,1} 上做随机重启 + 坐标搜索。
用法：python3 chordal_weighted_exponent.py
"""
import numpy as np, networkx as nx, itertools, random


def chain(H):
    """预计算：H 的所有独立集、转移（类型：删除 / 插入点 v）。返回可快速按活跃度重算的结构。"""
    k = H.number_of_nodes(); nb = [sum(1 << u for u in H[v]) for v in range(k)]
    st = [I for I in range(1 << k) if all(not (I >> v & 1 and I & nb[v]) for v in range(k))]
    ix = {I: j for j, I in enumerate(st)}
    occ = np.array([[I >> v & 1 for v in range(k)] for I in st], dtype=float)
    dele, ins = [], []
    for j, I in enumerate(st):
        for v in range(k):
            if I >> v & 1: dele.append((j, ix[I ^ (1 << v)]))
            elif not I & nb[v]: ins.append((j, ix[I | (1 << v)], v))
    return len(st), occ, np.array(dele), np.array(ins)


def log_trel(ch, logLam):
    """log(t_rel · c)，logLam[v] = log Λ_v。"""
    N, occ, dele, ins = ch
    lw = occ @ logLam; lw -= lw.max(); pi = np.exp(lw); pi /= pi.sum()
    Q = np.zeros((N, N))
    Q[dele[:, 0], dele[:, 1]] = 1.0
    Q[ins[:, 0], ins[:, 1]] = np.exp(logLam[ins[:, 2]])
    np.fill_diagonal(Q, -Q.sum(1))
    d = np.sqrt(pi); S = d[:, None] * Q / d[None, :]; S = (S + S.T) / 2
    ev = np.linalg.eigvalsh(-S)
    return -np.log(np.sort(ev)[1])


def alpha(ch, e, L1=np.log(1e6), L2=np.log(1e9)):
    e = np.asarray(e, float)
    return (log_trel(ch, e * L2) - log_trel(ch, e * L1)) / (L2 - L1)


def search(H, restarts=6, grid=8, rng=random.Random(0)):
    ch = chain(H); k = H.number_of_nodes(); best = (-1, None)
    for _ in range(restarts):
        e = [rng.randint(0, grid) for _ in range(k)]
        cur = alpha(ch, np.array(e) / grid); improved = True
        while improved:
            improved = False
            for v in range(k):
                for val in range(grid + 1):
                    if val == e[v]: continue
                    e2 = e.copy(); e2[v] = val; a = alpha(ch, np.array(e2) / grid)
                    if a > cur + 1e-9: e, cur, improved = e2, a, True
        if cur > best[0]: best = (cur, list(np.array(e) / grid))
    return best


if __name__ == "__main__":
    P4 = nx.path_graph(4)
    print(f"校验：分裂竞争（P4，e=(1/2,1,1,1/2)）α = {alpha(chain(P4), [0.5, 1, 1, 0.5]):.4f}（预期 0.5）\n")
    print("所有 k 点树上的最大减速指数 α(k)：")
    for k in range(2, 10):
        best = (-1, None, None)
        for T in nx.nonisomorphic_trees(k):
            T = nx.convert_node_labels_to_integers(T)
            a, e = search(T, restarts=3 if k >= 8 else 6)
            if a > best[0]: best = (a, e, sorted(d for _, d in T.degree()))
        print(f"  k={k}: α = {best[0]:.3f}   度序列 {best[2]}   指数 {best[1]}", flush=True)
    print("\n随机弦图（含非树）上的最大 α：")
    from chordal_decomposition_check import random_connected_chordal
    rng = random.Random(1)
    for k in [6, 8, 10]:
        best = -1
        for _ in range(12):
            G = nx.convert_node_labels_to_integers(random_connected_chordal(k, rng))
            best = max(best, search(G, restarts=2)[0])
        print(f"  k={k}: α = {best:.3f}", flush=True)
