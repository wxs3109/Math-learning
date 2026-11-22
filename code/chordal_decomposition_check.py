r"""
验证"按一个团 R 的状态做分解"时的三个量（弦图 hardcore 的 Glauber dynamics）：

  分块：Ω = ⊔_{x ∈ {∅}∪R} Ω_x，  Ω_x = { I : I ∩ R = x }   （独立集和团至多交 1 个点）

  (1) 投影链 P̄ 的谱间隙。理论预测：P̄ 是以 ∅ 为中心的星形链，
      每个叶子 x→∅ 的速率恰好等于删除 x 的概率 1/(n(1+λ))，
      所以 gap(P̄) = 1/(n(1+λ))，和 |R|、图结构都无关。
  (2) 实际的"分解损失"
          L = t_rel(P) / max_x t_rel(P_x)
      其中 P_x 是 P 限制在 Ω_x 上（跨块的移动改成自环）。
      粗糙的分解定理只能给出 L ≲ |R|·λ，从而递归得到 n^{O(log(|R|λ))}。
  (3) 在随机弦图上做对抗搜索：找使 L 最大的 (图, 团)。

用法：python3 chordal_decomposition_check.py
"""

import random
import numpy as np
import networkx as nx
from chordal_glauber_gap import clique_binary_tree


def random_connected_chordal(n, rng):
    """每个新点连向某个现有团的非空子集 ⇒ 连通弦图。"""
    G = nx.Graph(); G.add_node(0)
    for v in range(1, n):
        C = rng.choice([list(c) for c in nx.find_cliques(G)])
        G.add_edges_from((u, v) for u in rng.sample(C, rng.randint(1, len(C))))
    return G


def random_split_graph(k, m, rng):
    """split graph：k-团 K 加 m 个独立点，每个独立点随机连向 K 的非空子集（弦图，分隔集可以很大）。"""
    G = nx.complete_graph(k)
    for j in range(m):
        v = k + j
        G.add_node(v)
        G.add_edges_from((u, v) for u in rng.sample(range(k), rng.randint(1, k)))
    return G


# ---------------------------------------------------------------------------
# 基础：枚举独立集、构造 Glauber 转移矩阵（稠密，适合 ≤ 几千个状态）
# ---------------------------------------------------------------------------
def glauber_matrix(n, adj, lam):
    nbr = [sum(1 << u for u in adj[v]) for v in range(n)]
    states = []

    def rec(v, cur, forb):                          # 回溯枚举独立集
        if v == n:
            states.append(cur); return
        rec(v + 1, cur, forb)
        if not forb >> v & 1:
            rec(v + 1, cur | 1 << v, forb | nbr[v])
    rec(0, 0, 0)

    idx = {I: i for i, I in enumerate(states)}
    N, p_in = len(states), lam / (1 + lam)
    P = np.zeros((N, N))
    for i, I in enumerate(states):
        for v in range(n):
            b = 1 << v
            if I & b:
                P[i, idx[I ^ b]] += (1 - p_in) / n   # 删除 v
            elif not I & nbr[v]:
                P[i, idx[I | b]] += p_in / n         # 插入 v
        P[i, i] += 1 - P[i].sum()                    # 其余概率留在原地
    w = np.array([lam ** bin(I).count("1") for I in states], float)
    return states, P, w / w.sum()


def spectral_gap(P, w):
    """可逆链的谱间隙：对称化 D^{1/2} P D^{-1/2} 后求第二大特征值。"""
    if len(w) == 1:
        return np.inf
    d = np.sqrt(w)
    S = (d[:, None] * P) / d[None, :]
    ev = np.sort(np.linalg.eigvalsh((S + S.T) / 2))
    return 1 - ev[-2]


# ---------------------------------------------------------------------------
# 按团 R 的状态分解：返回 gap(P̄)、理论值、分解损失 L
# ---------------------------------------------------------------------------
def decomposition_stats(n, adj, R, lam):
    states, P, w = glauber_matrix(n, adj, lam)
    R = list(R)
    sig = [next((1 + j for j, v in enumerate(R) if I >> v & 1), 0) for I in states]
    m = len(R) + 1

    # 投影链 P̄(a,b) = Σ_{I∈Ω_a, J∈Ω_b} π(I)P(I,J) / π(Ω_a)
    piS = np.zeros(m)
    Q = np.zeros((m, m))
    for i, a in enumerate(sig):
        piS[a] += w[i]
    for i, a in enumerate(sig):
        for j in np.nonzero(P[i])[0]:
            if sig[j] != a:
                Q[a, sig[j]] += w[i] * P[i, j]
    keep = piS > 0
    Pbar = Q[np.ix_(keep, keep)] / piS[keep][:, None]
    np.fill_diagonal(Pbar, 1 - Pbar.sum(1))
    gap_bar = spectral_gap(Pbar, piS[keep])

    # 限制链 P_x 的谱间隙，取最差的一个
    worst = np.inf
    for x in range(m):
        ids = [i for i, a in enumerate(sig) if a == x]
        if len(ids) <= 1:
            continue
        Px = P[np.ix_(ids, ids)].copy()
        Px[np.diag_indices_from(Px)] += 1 - Px.sum(1)  # 跨块移动 → 自环
        worst = min(worst, spectral_gap(Px, w[ids] / w[ids].sum()))

    gap = spectral_gap(P, w)
    L = worst / gap                                     # = t_rel(P) / max_x t_rel(P_x)
    # 带"基础项"的损失：t_rel(P) / (t_rel(P̄) + max_x t_rel(P_x))
    # 理由：单个团本身就需要 ~ t_rel(P̄) = n(1+λ) 的时间，这不是递归损失
    M = (1 / gap) / (1 / gap_bar + 1 / worst)
    return gap_bar, 1 / (n * (1 + lam)), L, M


def adj_from_nx(G):
    G = nx.convert_node_labels_to_integers(G)
    return G.number_of_nodes(), [set(G.neighbors(v)) for v in G.nodes], G


if __name__ == "__main__":
    # ---- 实验 1 & 2：clique binary tree，按根团分解 ----
    print("== clique binary tree，按根团 R 分解 ==")
    print(f"{'(d,k,s)':>10} {'λ':>5} | {'gap(P̄)':>9} {'1/(n(1+λ))':>11} | {'损失 L':>7} | {'M':>5} | |R|·λ")
    for cfg in [(2, 4, 2), (3, 4, 3), (3, 8, 7)]:
        n, adj = clique_binary_tree(*cfg)
        R = range(cfg[1])                               # 根团 = 顶点 0..k-1
        for lam in [1, 16, 256, 1024]:
            gb, pred, L, M = decomposition_stats(n, adj, R, lam)
            print(f"{str(cfg):>10} {lam:>5} | {gb:9.6f} {pred:11.6f} | {L:7.3f} | {M:5.3f} | {cfg[1]*lam}")

    # ---- 实验 3：随机弦图上对抗搜索最大的 L ----
    print("\n== 随机弦图对抗搜索（每个图试所有极大团）==")
    rng = random.Random(7)
    best = []
    graphs = [random_connected_chordal(rng.randint(8, 13), rng) for _ in range(120)]
    graphs += [random_split_graph(rng.randint(3, 6), rng.randint(4, 8), rng) for _ in range(80)]
    for G0 in graphs:
        n, adj, G = adj_from_nx(G0)
        for lam in [4, 64, 1024]:
            for R in nx.chordal_graph_cliques(G):
                if len(R) < 2:
                    continue
                _, _, L, M = decomposition_stats(n, adj, R, lam)
                best.append((M, L, lam, n, len(R)))
    best.sort(reverse=True)
    print(f"共测试 {len(best)} 个 (图, 团, λ) 组合")
    print(f"max L = {max(b[1] for b in best):.2f}（含单团基础时间，会随 λ 线性涨）")
    print("M = t_rel(P)/(t_rel(P̄)+max t_rel(P_x)) 最大的 5 个：")
    for M, L, lam, n, r in best[:5]:
        print(f"  M = {M:6.3f}   (L = {L:7.2f})   λ = {lam:>4}   n = {n:>2}   |R| = {r}")
    for lam in [4, 64, 1024]:
        print(f"  λ = {lam:>4}: max M = {max(b[0] for b in best if b[2] == lam):.3f}")
