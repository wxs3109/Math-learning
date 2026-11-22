r"""
证伪实验：嵌套"公共阻挡点"结构，检验慢化是否随团大小 u 叠加。

单层 gadget（见 problems/chordal-remove-b.md §8.4）：
    y —— U (u-团，全部连 y) —— v (连 U 全部)
  慢化因子约 min(u, λ)，固定 λ 时随 u 饱和。

嵌套：把阻挡点 v 本身当作下一层 gadget 的 "y"，挂 `branch` 个子 gadget：
    v —— U' (u-团) —— v'   （每个子 gadget 各一份）
  深度 d 的结构是一棵 gadget 树。它是弦图（天然有 clique tree：
  {y}∪U, U∪{v}, {v}∪U', U'∪{v'}, ...）。

判据（固定 λ）：
  若 t_rel / (n(1+λ)) 在 u → ∞ 时对每个深度 d 都饱和  ⇒ 支持猜想（每层 poly(λ)）
  若随 u 无界增长（尤其增长速度随 d 变快）           ⇒ 猜想被证伪

实现：回溯枚举独立集 + 稀疏转移矩阵 + Lanczos（eigsh）求第二大特征值。
用法：python3 chordal_nested_gadget.py
"""
import sys
import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def nested_gadget(depth, u, branch=2):
    """返回 (n, adj)。根 gadget 的 y 额外挂一个点 r（再挂一个叶子），和 §8.4 一致。"""
    adj = []

    def add():
        adj.append(set())
        return len(adj) - 1

    def e(a, b):
        adj[a].add(b)
        adj[b].add(a)

    y = add()
    r = add(); e(y, r); e(r, add())              # y 旁边的 r 和叶子

    def grow(top, level):
        """top 是本层 gadget 的 'y'，在它下面挂 gadget，并递归。"""
        if level > depth:
            return
        k = 1 if level == 1 else branch          # 根只挂一个，下面每层分叉
        for _ in range(k):
            U = [add() for _ in range(u)]
            for i, a in enumerate(U):
                e(top, a)
                for b in U[i + 1:]:
                    e(a, b)
            v = add()
            for a in U:
                e(a, v)                          # 公共阻挡点
            grow(v, level + 1)                   # v 作为下一层的 'y'
    grow(y, 1)
    return len(adj), adj


def enumerate_is(n, adj):
    nbr = [0] * n
    for v in range(n):
        for w in adj[v]:
            nbr[v] |= 1 << w
    out = []
    sys.setrecursionlimit(10000)

    def rec(v, cur, forb):
        if v == n:
            out.append(cur)
            return
        rec(v + 1, cur, forb)
        if not forb >> v & 1:
            rec(v + 1, cur | (1 << v), forb | nbr[v])
    rec(0, 0, 0)
    return out, nbr


def relaxation_time(n, adj, lam):
    states, nbr = enumerate_is(n, adj)
    N = len(states)
    idx = {I: i for i, I in enumerate(states)}
    p_in = lam / (1 + lam)
    rows, cols, vals = [], [], []
    diag = np.zeros(N)
    for i, I in enumerate(states):
        stay = 0.0
        for v in range(n):
            b = 1 << v
            if I & b:                                    # 删除 v
                rows.append(i); cols.append(idx[I ^ b]); vals.append((1 - p_in) / n)
                stay += p_in / n
            elif I & nbr[v]:                             # 被挡住
                stay += 1.0 / n
            else:                                        # 插入 v
                rows.append(i); cols.append(idx[I | b]); vals.append(p_in / n)
                stay += (1 - p_in) / n
        diag[i] = stay
    sizes = np.array([bin(I).count("1") for I in states])
    logw = sizes * np.log(lam)
    pi = np.exp(logw - logw.max()); pi /= pi.sum()
    d = np.sqrt(pi)
    rows = np.array(rows); cols = np.array(cols); vals = np.array(vals)
    # 对称化 S = D^{1/2} P D^{-1/2}（可逆链，S 对称）
    svals = vals * d[rows] / d[cols]
    S = sp.csr_matrix((svals, (rows, cols)), shape=(N, N)) + sp.diags(diag)
    # 去掉平稳方向 d 后求最大特征值 = λ_2
    op = spla.LinearOperator((N, N), matvec=lambda x: S @ x - d * (d @ x) * 1.0, dtype=float)
    ev = spla.eigsh(op, k=1, which="LA", return_eigenvectors=False, tol=1e-9, maxiter=20000)
    lam2 = ev[0]
    return N, 1 / (1 - lam2)


def run(builder, plan, lams):
    for lam in lams:
        print(f"\n===== λ = {lam} =====   （表中为 t_rel / (n(1+λ))）")
        for depth, us in plan.items():
            row = []
            for u in us:
                n, adj = builder(depth, u)
                t0 = time.time()
                N, trel = relaxation_time(n, adj, lam)
                row.append(f"u={u}:{trel / (n * (1 + lam)):.2f}")
                print(f"  depth={depth} u={u:<3} n={n:<4} #IS={N:<8} t_rel/(n(1+λ))={trel/(n*(1+lam)):8.3f}"
                      f"  ({time.time()-t0:.0f}s)", flush=True)


def nested_twin_gadget(depth, u, branch=2):
    """变体：阻挡者是 u-团 V（孪生阻挡），子 gadget 的 U' 连到整个 V。
    这样相邻两层共享大小为 u 的分隔集 —— 正是"分隔集很大"的情形。"""
    adj = []

    def add():
        adj.append(set())
        return len(adj) - 1

    def e(a, b):
        adj[a].add(b)
        adj[b].add(a)

    def clique(vs):
        for i, a in enumerate(vs):
            for b in vs[i + 1:]:
                e(a, b)
    y = add()
    r = add(); e(y, r); e(r, add())

    def grow(top, level):
        if level > depth:
            return
        for _ in range(1 if level == 1 else branch):
            U = [add() for _ in range(u)]
            clique(U)
            for t in top:
                for a in U:
                    e(t, a)
            V = [add() for _ in range(u)]
            clique(V)
            for a in U:
                for b in V:
                    e(a, b)
            grow(V, level + 1)
    grow([y], 1)
    return len(adj), adj


if __name__ == "__main__":
    # 用法：python3 chordal_nested_gadget.py          —— 单点阻挡嵌套
    #       python3 chordal_nested_gadget.py --twin   —— 孪生阻挡嵌套（相邻层共享大小 u 的分隔集）
    if "--twin" in sys.argv:
        run(nested_twin_gadget, {1: [2, 4, 8, 16, 32], 2: [1, 2, 4, 8, 16, 24, 32], 3: [1, 2, 3]}, [4, 16])
    else:
        run(nested_gadget, {1: [2, 4, 8, 16, 32, 64], 2: [2, 4, 8, 16, 24, 32], 3: [1, 2, 3, 4]}, [4, 16])
