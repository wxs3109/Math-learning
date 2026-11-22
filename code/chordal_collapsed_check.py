r"""
修正方案的数值检验：坍缩链 Q 几何下的中心部分界（problems/chordal-remove-b.md §13）。

记号：Ω_∅ = {I : I∩R = ∅}，pr 把 I 中 R 的点删掉，ν = pr_* π，即
    ν(J) = π(J) (1 + λ N(J)),   N(J) = #{y∈R : y 在 J 处可插入}
坍缩链（把每颗星 {J} ∪ {J∪{y}} 压成一点）：
    Q(J,J') = π(J) P_∅(J,J') (1 + λ N(J,J')) / ν(J),   J ≠ J'
    N(J,J') = #{y : y 在 J 和 J' 处都可插入}
中心部分  π_∅ Σ_y w_y d_y²，d_y = E_{μ_y}F - E_{μ_0}F = <F, ψ_y>_ν，ψ_y = (μ_y - μ_0)/ν。
于是 sup_F 中心部分 / E_Q(F,F) = π_∅ ||Γ^Q||，
    Γ^Q_{yy'} = sqrt(w_y w_y') <ψ_y, Z_Q ψ_y'>_ν,   Z_Q = (I - Q)^+ （ν 下）

本脚本对比（单位 n(1+λ)）：
    old   = ||Γ^Z||          ——引理 2′（只用中心能量），§12 已知会随结构增长
    new   = π_∅ ||Γ^Q||      ——坍缩链版本（中心 + 叶子能量）
    true  = sup_f 中心部分 / E(f,f)  ——真正要控制的量
用法：python3 chordal_collapsed_check.py
"""
import random
import numpy as np
import networkx as nx
from chordal_decomposition_check import glauber_matrix, random_connected_chordal, random_split_graph, adj_from_nx
from chordal_mixture_means_check import build as twin_build, parts
from chordal_lemma_g_check import whiskers, blocker, green_gram
from chordal_twin_blocker_combo import build as combo_build
from chordal_nested_gadget import nested_twin_gadget


def collapsed(n, adj, R, lam):
    states, P, w = glauber_matrix(n, adj, lam)
    R = list(R)
    Rm = sum(1 << v for v in R)
    nbr = [sum(1 << u for u in adj[v]) for v in range(n)]
    ids = [i for i, I in enumerate(states) if not states[i] & Rm]
    J = [states[i] for i in ids]
    m = len(ids)
    ins = np.array([[0. if Jk & nbr[y] else 1. for y in R] for Jk in J])   # ins[J, y]
    piJ = w[ids]                                       # π(J)（整体已归一化）
    pi0 = piJ.sum()
    nu = piJ * (1 + lam * ins.sum(1))                  # ν(J)，总和 = 1
    # Q 的对称导纳 c(J,J') = π(J) P_∅(J,J') (1 + λ N(J,J'))
    Psub = P[np.ix_(ids, ids)]
    C = (piJ[:, None] * Psub) * (1 + lam * (ins @ ins.T))
    np.fill_diagonal(C, 0)
    C = (C + C.T) / 2
    Lap = np.diag(C.sum(1)) - C                        # ν 加权的 Laplacian：E_Q(F,F) = F^T Lap F
    # ψ_y = (μ_y - μ_0)/ν；μ_0 = π(J)/π_∅，μ_y = π(J) ins/π(A_y)
    mu0 = piJ / pi0
    vecs, ws = [], []
    for k, y in enumerate(R):
        piA = piJ @ ins[:, k]
        if piA <= 0 or piA >= pi0 * (1 - 1e-12):
            continue
        muy = piJ * ins[:, k] / piA
        vecs.append((muy - mu0) / nu)
        ws.append(lam * piA)                          # w_y = π(Ω_y) = λ π(A_y)
    if not vecs:
        return 0.0
    # Z_Q 作用：解 Lap x = ν ψ（ν ψ 的总和为 0），Green 能量 <ψ, Z_Q ψ>_ν = (νψ)^T x
    Lp = np.linalg.pinv(Lap)
    Bm = np.array([np.sqrt(wy) * nu * v for wy, v in zip(ws, vecs)])
    G = Bm @ Lp @ Bm.T
    return pi0 * np.linalg.norm((G + G.T) / 2, 2)


def row(name, n, adj, R, lam):
    s = n * (1 + lam)
    new = collapsed(n, adj, R, lam) / s
    old = green_gram(n, adj, list(R), lam)[1]          # 已是 n(1+λ) 单位
    true = parts(n, adj, list(R), lam)[1]
    print(f"{name:<26}{lam:>4} | old={old:7.3f}  new={new:7.3f}  true={true:7.3f}", flush=True)
    return new / max(true, 1e-12)


if __name__ == "__main__":
    print("（单位 n(1+λ)；old = ||Γ^Z||，new = π_∅||Γ^Q||，true = 真实中心部分）")
    ratios = []
    for lam in [4, 64]:
        for r_, m_ in [(4, 3), (9, 1)]:
            ratios.append(row(f"whiskers r={r_} m={m_}", *whiskers(r_, m_), lam))
        for s in [1, 8]:
            ratios.append(row(f"twin s={s}", *twin_build(s), lam))
        for u in [8, 32]:
            ratios.append(row(f"blocker u={u}", *blocker(u), lam))
        for s, u in [(4, 8), (16, 8), (16, 32)]:
            ratios.append(row(f"twin+blocker s={s} u={u}", *combo_build(s, u), lam))
        for u in [2, 3]:
            n, adj = nested_twin_gadget(2, u)
            ratios.append(row(f"nested-twin d2 u={u}", n, adj, [0, 1], lam))
    rng = random.Random(21); worst_new = 0.0
    for k in range(80):
        G0 = (random_connected_chordal(rng.randint(7, 11), rng) if k % 2
              else random_split_graph(rng.randint(3, 6), rng.randint(3, 6), rng))
        n, adj, G = adj_from_nx(G0)
        for Rc in nx.chordal_graph_cliques(G):
            for lam in [4, 64]:
                worst_new = max(worst_new, collapsed(n, adj, list(Rc), lam) / (n * (1 + lam)))
    print(f"\n随机弦图 / split 图：max new = {worst_new:.3f}")
    print(f"所有结构化例子里 new / true 的范围：[{min(ratios):.2f}, {max(ratios):.2f}]")
