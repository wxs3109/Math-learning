r"""
"团分解不等式"证明尝试的数值检查（配合 problems/chordal-remove-b.md §7）。

记号（R 是一个团，Ω_∅ = {I : I∩R = ∅}，μ_0 = π(·|Ω_∅)，A_y = {J∈Ω_∅ : N(y)∩J = ∅}）：
  w_y = π(Ω_y),  m_y = E_{μ_0}[F | A_y],  m_0 = E_{μ_0}F,  d_y = m_y - m_0
  G_y = <h_y-1, (I-P_∅)^+ (h_y-1)>_{μ_0},  h_y = 1_{A_y}/p_y        （Green 函数量）
  Γ   = 事件 {A_y} 的归一化协方差矩阵（Gram 矩阵）

检查三件事：
  (1) 孪生族上 (⋆) 的常数 M 是否随孪生数 s 增长
  (2) Σ_y w_y G_y / (n(1+λ)) 和 ||Γ|| 的大小
  (3) sup_f Var_w(m)/E(f,f) 及其"中心部分 / 叶子部分"（广义特征值）

用法：python3 chordal_mixture_means_check.py
"""
import random
import numpy as np
import networkx as nx
from chordal_decomposition_check import (glauber_matrix, decomposition_stats,
                                         random_connected_chordal, random_split_graph, adj_from_nx)
from chordal_glauber_gap import clique_binary_tree


# 孪生族：R = S ∪ {u}，S 的 s 个点外部邻域完全相同（都只连到子团 D = {d1,d2}）；
#         子团 C = S ∪ D；每个 d_i 和 u 各挂一个叶子。
# 测试"孪生"情形：R 中有 s 个点的外部邻域完全相同（都只连到子团 D）
# 图：R = K_{s+1}（S = R 的前 s 个点 + 1 个点 u）；子团 C = S ∪ D，D = {d1,d2}；
#     每个 d_i 挂一个叶子；u 挂一个叶子。

def build(s, d=2):
    n = 0; adj = []
    def add():
        nonlocal n; adj.append(set()); n += 1; return n - 1
    def edge(a, b): adj[a].add(b); adj[b].add(a)
    S = [add() for _ in range(s)]; u = add(); R = S + [u]
    D = [add() for _ in range(d)]
    for i in range(len(R)):
        for j in range(i + 1, len(R)): edge(R[i], R[j])
    C = S + D
    for i in range(len(C)):
        for j in range(i + 1, len(C)): edge(C[i], C[j])
    for x in D + [u]: edge(x, add())          # 叶子
    return n, adj, R


def stats(n, adj, R, lam):
    states, P, w = glauber_matrix(n, adj, lam)
    R = list(R)
    ids = [i for i, I in enumerate(states) if not any(I >> v & 1 for v in R)]
    Pe = P[np.ix_(ids, ids)].copy(); Pe[np.diag_indices_from(Pe)] += 1 - Pe.sum(1)
    mu = w[ids] / w[ids].sum()
    nbr = [sum(1 << u for u in adj[v]) for v in range(n)]
    A = np.array([[0 if states[i] & nbr[y] else 1 for i in ids] for y in R], float)  # A[y][J]
    p = A @ mu
    d = np.sqrt(mu); S = (d[:, None] * Pe) / d[None, :]; S = (S + S.T) / 2
    Lp = np.linalg.pinv(np.eye(len(ids)) - S)          # 对称化后的伪逆
    G = []
    Zall = w.sum()                                    # w 已归一化 ⇒ 1
    pi0 = w[ids].sum()
    for y in range(len(R)):
        if p[y] <= 0 or p[y] >= 1 - 1e-12: continue
        phi = (A[y] / p[y] - 1) * d
        wy = lam * pi0 * p[y]                          # w_y = π(Ω_y)
        G.append(wy * (phi @ Lp @ phi))                # 记录 w_y·G_y
    H = np.array([(A[y] - p[y]) / np.sqrt(p[y]) for y in range(len(R)) if 0 < p[y] < 1])
    Gam = (H * mu) @ H.T if len(H) else np.zeros((1, 1))
    return (sum(G) if G else 0) / (n * (1 + lam)), np.linalg.norm(Gam, 2), len(R)


def sup_ratio(Qm, Lm):
    # max f^T Q f / f^T L f, 在 L 的值域上（L 半正定，核为常数）
    ev, U = np.linalg.eigh(Lm); keep = ev > 1e-12 * ev.max()
    B = U[:, keep] / np.sqrt(ev[keep])
    return np.linalg.eigvalsh(B.T @ Qm @ B).max()

def parts(n, adj, R, lam):
    states, P, w = glauber_matrix(n, adj, lam)
    N = len(states); R = list(R)
    Lm = np.diag(w) @ (np.eye(N) - P); Lm = (Lm + Lm.T) / 2      # Dirichlet 形式矩阵
    nbr = [sum(1 << u for u in adj[v]) for v in range(n)]
    ctr = [i for i, I in enumerate(states) if not any(I >> v & 1 for v in R)]
    pi0 = w[ctr].sum()
    # 每个分量的均值泛函 m_x = <a_x, f>
    a0 = np.zeros(N); a0[ctr] = w[ctr] / pi0
    leaves = []
    for y in R:
        a = np.zeros(N); sel = [i for i in ctr if not states[i] & nbr[y]]
        if not sel: continue
        a[sel] = w[sel] / w[sel].sum()
        leaves.append((lam * w[sel].sum(), a))           # (w_y, 均值泛函)
    W = sum(wy for wy, _ in leaves)
    Qc = sum(wy * np.outer(a - a0, a - a0) for wy, a in leaves) * pi0          # 中心部分
    aL = sum(wy * a for wy, a in leaves) / W
    Ql = W * sum(wy * np.outer(a - aL, a - aL) for wy, a in leaves)             # 叶子部分 = W·Σ w_y (m_y - m_L)²（§14 更正：原先漏了因子 W）
    s = n * (1 + lam)
    return sup_ratio(Qc + Ql, Lm) / s, sup_ratio(Qc, Lm) / s, sup_ratio(Ql, Lm) / s

if __name__ == "__main__":
    print("(1) 孪生族上 (⋆) 的经验常数 M")
    for s in [1, 2, 4, 8, 12]:
        n, adj, R = build(s)
        Ms = [decomposition_stats(n, adj, R, lam)[3] for lam in [4, 64, 1024]]
        print(f"  s={s:>2}: " + "  ".join(f"M(λ={l})={m:.3f}" for l, m in zip([4, 64, 1024], Ms)))

    print("\n(2) Σ w_y G_y/(n(1+λ)) 与 ||Γ||")
    for s in [1, 4, 8, 12]:
        n, adj, R = build(s)
        for lam in [4, 1024]:
            g, gam, r = stats(n, adj, R, lam)
            print(f"  twin s={s:<2} λ={lam:<5}: ΣwG={g:6.3f}  ||Γ||={gam:6.3f}  |R|={r}")
    for cfg in [(2, 4, 2), (3, 8, 7)]:
        n, adj = clique_binary_tree(*cfg)
        for lam in [4, 1024]:
            g, gam, r = stats(n, adj, range(cfg[1]), lam)
            print(f"  tree {cfg} λ={lam:<5}: ΣwG={g:6.3f}  ||Γ||={gam:6.3f}  |R|={r}")

    print("\n(3) sup_f Var_w(m)/E(f,f)，单位 n(1+λ)：总量 / 中心部分 / 叶子部分")
    for s in [1, 4, 8]:
        n, adj, R = build(s)
        for lam in [4, 1024]:
            t, c, l = parts(n, adj, R, lam)
            print(f"  twin s={s:<2} λ={lam:<5}: {t:6.3f} / {c:6.3f} / {l:6.3f}")
    rng = random.Random(11)
    worst = [0, 0, 0]
    for k in range(120):
        G0 = (random_connected_chordal(rng.randint(7, 11), rng) if k % 2
              else random_split_graph(rng.randint(3, 6), rng.randint(3, 6), rng))
        n, adj, G = adj_from_nx(G0)
        for R in nx.chordal_graph_cliques(G):
            for lam in [4, 256]:
                worst = [max(a, b) for a, b in zip(worst, parts(n, adj, R, lam))]
    print("  随机弦图 / split 图上的最大值：", [round(x, 3) for x in worst])
