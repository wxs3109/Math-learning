"""
弦图上 hardcore model 的 Glauber dynamics：精确计算谱间隙（relaxation time）。

目的：在小规模弦图上数值观察 relaxation time  t_rel = 1 / gap
如何随 (n, 分隔集大小 s, 逸度 λ) 变化，给"能否去掉 bounded separator 限制"
这个问题提供直觉。

图族：clique binary tree
    - 一棵深度为 depth 的完全二叉树，每个树节点是一个大小为 k 的团 (bag)
    - 子团和父团共享 s 个顶点（s = 最小分隔集大小）
    - 这样构造出来的图一定是弦图（它天然带一个 clique tree）
    - s = 0 时额外用一条边连接父子团，对应 Bezáková–Sun 论文里的例子（分隔集大小 1）

Glauber dynamics（heat-bath 版本）：
    每步均匀随机选顶点 v：
        - 以概率 λ/(1+λ) 尝试把 v 加入独立集（若 v 的邻居都不在集合里才成功）
        - 否则把 v 移出
    该链关于 π(I) ∝ λ^{|I|} 可逆，且转移矩阵半正定，所以 gap = 1 - λ_2。

用法：python3 chordal_glauber_gap.py
"""

import itertools
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


# ---------------------------------------------------------------------------
# 1. 构造图
# ---------------------------------------------------------------------------
def clique_binary_tree(depth, k, s):
    """返回 (n, adj)，adj[v] 是 v 的邻居集合。

    depth: 二叉树深度（根深度 0），共 2^{depth+1}-1 个团
    k:     每个团的大小
    s:     子团与父团共享的顶点数（0 <= s < k）
    """
    assert 0 <= s < k
    bags = []          # bags[i] = 第 i 个团的顶点列表
    n = 0

    def new_vertices(cnt):
        nonlocal n
        vs = list(range(n, n + cnt))
        n += cnt
        return vs

    bags.append(new_vertices(k))                 # 根团
    num_nodes = 2 ** (depth + 1) - 1
    for i in range(1, num_nodes):
        parent = bags[(i - 1) // 2]
        which_child = (i - 1) % 2                # 0 = 左孩子, 1 = 右孩子
        # 左右孩子尽量共享父团里不同的 s 个顶点（父团够大时）
        start = (which_child * s) % k
        shared = [parent[(start + j) % k] for j in range(s)]
        bags.append(shared + new_vertices(k - s))

    adj = [set() for _ in range(n)]
    for bag in bags:                             # 每个 bag 内两两相连
        for u, v in itertools.combinations(bag, 2):
            adj[u].add(v)
            adj[v].add(u)
    if s == 0:                                   # s=0: 父子团之间连一条边
        for i in range(1, num_nodes):
            u = bags[(i - 1) // 2][-1]
            v = bags[i][0]
            adj[u].add(v)
            adj[v].add(u)
    return n, adj


# ---------------------------------------------------------------------------
# 2. 枚举所有独立集（用 bitmask 表示）
# ---------------------------------------------------------------------------
def enumerate_independent_sets(n, adj):
    """回溯枚举所有独立集，返回 bitmask 列表。"""
    nbr_mask = [sum(1 << u for u in adj[v]) for v in range(n)]
    result = []

    def rec(v, cur, forbidden):
        if v == n:
            result.append(cur)
            return
        rec(v + 1, cur, forbidden)                       # 不选 v
        if not (forbidden >> v) & 1:                     # 选 v（若允许）
            rec(v + 1, cur | (1 << v), forbidden | nbr_mask[v])

    rec(0, 0, 0)
    return result, nbr_mask


# ---------------------------------------------------------------------------
# 3. 构造 Glauber 转移矩阵，计算谱间隙
# ---------------------------------------------------------------------------
def glauber_gap(n, adj, lam):
    states, nbr_mask = enumerate_independent_sets(n, adj)
    N = len(states)
    index = {I: i for i, I in enumerate(states)}
    p_in = lam / (1 + lam)                               # 加入概率

    rows, cols, vals = [], [], []
    for i, I in enumerate(states):
        stay = 0.0
        for v in range(n):
            bit = 1 << v
            if I & bit:
                # v 在集合里：以 1/(1+λ) 移出
                j = index[I ^ bit]
                rows.append(i); cols.append(j); vals.append((1 - p_in) / n)
                stay += p_in / n
            elif I & nbr_mask[v]:
                stay += 1.0 / n                          # 有邻居被占，不能加
            else:
                j = index[I | bit]
                rows.append(i); cols.append(j); vals.append(p_in / n)
                stay += (1 - p_in) / n
        rows.append(i); cols.append(i); vals.append(stay)
    P = sp.csr_matrix((vals, (rows, cols)), shape=(N, N))

    # 平稳分布 π(I) ∝ λ^{|I|}；对称化 S = D^{1/2} P D^{-1/2}
    sizes = np.array([bin(I).count("1") for I in states])
    logw = sizes * np.log(lam)
    pi = np.exp(logw - logw.max())
    pi /= pi.sum()
    d = np.sqrt(pi)
    S = sp.diags(d) @ P @ sp.diags(1 / d)
    S = (S + S.T) / 2                                    # 消除数值误差

    if N <= 2000:
        eig = np.linalg.eigvalsh(S.toarray())
        lam2 = eig[-2]
    else:
        eig = spla.eigsh(S, k=2, which="LA", return_eigenvectors=False)
        lam2 = min(eig)
    gap = 1 - lam2
    return N, gap


# ---------------------------------------------------------------------------
# 4. 实验
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    lams = [1, 4, 16, 64]

    def run(title, configs):
        print(f"\n== {title} ==")
        print(f"{'depth':>5} {'k':>3} {'s':>3} {'n':>4} {'#IS':>8} | "
              + " ".join(f"λ={l}".rjust(16) for l in lams))
        for depth, k, s in configs:
            n, adj = clique_binary_tree(depth, k, s)
            out = []
            for lam in lams:
                N, gap = glauber_gap(n, adj, lam)
                t = 1 / gap
                # 括号里是 t_rel / (n·λ)：单点更新链在大 λ 下的自然尺度
                out.append(f"{t:8.1f}({t/(n*lam):5.2f})")
            print(f"{depth:>5} {k:>3} {s:>3} {n:>4} {N:>8} | " + " ".join(out))

    # 实验 A：深度 2，团大小 4，分隔集 s = 0..3
    run("A: depth=2, k=4, 变化 s", [(2, 4, s) for s in range(4)])
    # 实验 B：深度 3（状态数可控的组合）
    run("B: depth=3", [(3, 3, 2), (3, 4, 2), (3, 4, 3)])
    # 实验 C：深度 3 固定，s = k-1 随 k 增大（分隔集变大，树结构不变）
    run("C: depth=3, s=k-1, k 增大", [(3, k, k - 1) for k in range(3, 11)])
