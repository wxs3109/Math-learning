r"""
弦图上 hardcore model 的精确计算：clique tree DP。

结论：对弦图 G 和任意 λ>0，
    Z_G(λ) = Σ_{I 独立集} λ^{|I|}
可以在 O(Σ_t |C_t| · deg(t)) 时间内精确算出（C_t 是 clique tree 的节点 = 极大团），
并且可以在同样的时间内从 π(I) ∝ λ^{|I|} 中做精确采样。

关键事实：
  (1) 独立集和每个团至多相交 1 个点，所以 "I ∩ C_t" 只有 |C_t|+1 种取值。
  (2) clique tree 满足 running intersection：包含顶点 v 的 bag 构成一棵连通子树。
      ⇒ 若 x ∈ C_t \ S、y ∈ C_c \ S（S = C_t ∩ C_c，c 是 t 的孩子），
         则 x、y 不在同一个 bag 里，因此不相邻。
         所以父子之间唯一的约束来自分隔集 S。

DP 定义（自底向上）：
  V_t    = t 的子树里所有 bag 的并
  f_t(x) = Σ { λ^{|I|} : I 是 G[V_t] 的独立集, I ∩ C_t = x },  x ∈ {∅} ∪ C_t
  转移：
    f_t(x) = w(x) · Π_{孩子 c} g_c(x)
    w(∅) = 1,  w(v) = λ
    g_c(x) = f_c(x) / λ                       若 x ∈ S（x 被父子共享，权重别算两次）
           = Σ_{y ∈ {∅} ∪ (C_c \ S)} f_c(y)   若 x ∉ S（含 x = ∅）
  Z = Σ_x f_root(x)

用法：python3 chordal_hardcore_dp.py   （随机弦图上和暴力枚举对拍）
"""

import random
from fractions import Fraction
from collections import Counter
import networkx as nx


# ---------------------------------------------------------------------------
# 1. 随机弦图：每次新加一个点，让它连向当前图里某个团的一个子集
#    （新点的邻域是团 ⇒ 它是 simplicial 点 ⇒ 图保持弦图）
# ---------------------------------------------------------------------------
def random_chordal_graph(n, rng):
    G = nx.Graph()
    G.add_node(0)
    for v in range(1, n):
        cliques = [list(c) for c in nx.find_cliques(G)]
        C = rng.choice(cliques)
        k = rng.randint(0, len(C))                 # 允许 k=0（新连通分量）
        G.add_node(v)
        for u in rng.sample(C, k):
            G.add_edge(u, v)
    assert nx.is_chordal(G)
    return G


# ---------------------------------------------------------------------------
# 2. clique tree：极大团的交图上求最大权生成树（权 = 交的大小）
#    对弦图这是标准做法，得到的树满足 running intersection
#    不连通的图：权为 0 的边把各个分量的 clique tree 连成一棵树，
#    此时分隔集 S = ∅，DP 转移自动退化成乘积，仍然正确
# ---------------------------------------------------------------------------
def clique_tree(G):
    cliques = [frozenset(c) for c in nx.chordal_graph_cliques(G)]
    H = nx.Graph()
    H.add_nodes_from(range(len(cliques)))
    for i in range(len(cliques)):
        for j in range(i + 1, len(cliques)):
            H.add_edge(i, j, weight=len(cliques[i] & cliques[j]))
    T = nx.maximum_spanning_tree(H) if len(cliques) > 1 else H
    return cliques, T


def check_running_intersection(G, cliques, T):
    for v in G.nodes:
        nodes = [i for i, C in enumerate(cliques) if v in C]
        assert nx.is_connected(T.subgraph(nodes)), "running intersection 失败"


# ---------------------------------------------------------------------------
# 3. DP：计算每个节点的表 f_t，返回 Z
# ---------------------------------------------------------------------------
def hardcore_dp(G, lam):
    cliques, T = clique_tree(G)
    check_running_intersection(G, cliques, T)
    root = 0
    parent = {root: None}
    order = list(nx.dfs_preorder_nodes(T, root))   # 父在子之前
    for t in order:
        for c in T.neighbors(t):
            if c not in parent:
                parent[c] = t
    children = {t: [c for c in T.neighbors(t) if parent.get(c) == t] for t in order}

    f = {}                                          # f[t][x]，x=None 表示 ∅
    for t in reversed(order):                       # 子在父之前处理
        C = cliques[t]
        table = {}
        for x in [None] + sorted(C):
            val = Fraction(1) if x is None else Fraction(lam)
            for c in children[t]:
                S = C & cliques[c]
                if x is not None and x in S:
                    val *= f[c][x] / lam
                else:
                    val *= f[c][None] + sum(f[c][y] for y in cliques[c] - S)
            table[x] = val
        f[t] = table
    Z = sum(f[root].values())
    return Z, (cliques, children, f, root)


# ---------------------------------------------------------------------------
# 4. 精确采样：自顶向下，按条件概率依次决定每个 bag 的 I ∩ C_t
# ---------------------------------------------------------------------------
def hardcore_sample(G, lam, rng, dp=None):
    if dp is None:
        _, dp = hardcore_dp(G, lam)
    cliques, children, f, root = dp
    I = set()

    def choose(weights):                            # weights: {x: 权重}
        tot = sum(weights.values())
        r = Fraction(rng.random()) * tot
        for x, w in weights.items():
            if r < w:
                return x
            r -= w
        return x

    stack = [(root, choose(f[root]))]               # 根：按 f_root(x) 选
    while stack:
        t, x = stack.pop()
        if x is not None:
            I.add(x)
        for c in children[t]:
            S = cliques[t] & cliques[c]
            if x is not None and x in S:
                y = x                               # 共享点已被父节点决定
            else:
                opts = {None: f[c][None]}
                opts.update({y: f[c][y] for y in cliques[c] - S})
                y = choose(opts)
            stack.append((c, y))
    return frozenset(I)


# ---------------------------------------------------------------------------
# 5. 暴力：枚举所有子集
# ---------------------------------------------------------------------------
def brute_force_Z(G, lam):
    nodes = list(G.nodes)
    Z = Fraction(0)
    for mask in range(1 << len(nodes)):
        S = [nodes[i] for i in range(len(nodes)) if mask >> i & 1]
        if all(not G.has_edge(u, v) for i, u in enumerate(S) for v in S[i + 1:]):
            Z += Fraction(lam) ** len(S)
    return Z


if __name__ == "__main__":
    rng = random.Random(2026)

    # 对拍 Z：200 个随机弦图，n ≤ 16，λ 取若干有理数
    cnt = 0
    for trial in range(200):
        n = rng.randint(1, 16)
        G = random_chordal_graph(n, rng)
        lam = Fraction(rng.choice([1, 2, 3, 7]), rng.choice([1, 2, 5]))
        Z_dp, _ = hardcore_dp(G, lam)
        Z_bf = brute_force_Z(G, lam)
        assert Z_dp == Z_bf, (trial, n, lam, Z_dp, Z_bf)
        cnt += 1
    print(f"Z 对拍：{cnt} 个随机弦图全部精确相等（有理数运算，无浮点误差）")

    # λ=1 时 Z = 独立集个数；顺便检查一个已知值：路 P_n 的独立集数 = Fibonacci(n+2)
    P = nx.path_graph(10)
    Z, _ = hardcore_dp(P, 1)
    print(f"P_10 的独立集个数 = {Z}（应为 F_12 = 144）")

    # 采样检验：小图上的经验分布 vs 精确分布（总变差距离）
    G = random_chordal_graph(7, rng)
    lam = Fraction(2)
    Z, dp = hardcore_dp(G, lam)
    N = 200_000
    emp = Counter(hardcore_sample(G, lam, rng, dp) for _ in range(N))
    nodes = list(G.nodes)
    tv, num_states = 0.0, 0
    for mask in range(1 << len(nodes)):
        S = frozenset(nodes[i] for i in range(len(nodes)) if mask >> i & 1)
        if all(not G.has_edge(u, v) for u in S for v in S if u < v):
            num_states += 1
            p = float(lam ** len(S) / Z)
            tv += abs(emp[S] / N - p)
    print(f"采样检验：n=7, {num_states} 个独立集, N={N} 次采样, TV 距离 = {tv/2:.4f}")
