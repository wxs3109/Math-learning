r"""
树的"孪生膨胀"（problems/chordal-remove-b.md §17）：损失会不会沿 clique tree 逐层相乘？

归约（膨胀引理）：把加权图 H 的每个点 k 换成一个 m_k 点的团（孪生类），H 的每条边换成两团之间的完全二部图。
  - H 是弦图 ⇒ 膨胀图是弦图（树 ⇒ 每条边变成一个 2m 团，clique tree 就是 H 的线图结构）。
  - 按类内置换对称性合并后，Glauber 恰好是 H 上的 hardcore 动力学：
        类 k 插入速率 Λ_k c（Λ_k = m_k λ），删除速率 c，   c = 1/(n(1+λ))。
  - 合并链的谱是原链谱的子集，所以合并链的 t_rel 是真实 t_rel 的下界；
    真实 t_rel 还至少是 1/c（类内"哪个孪生被占"的模式）。
本文件：H = 深度 h 的完全二叉树，内部点活跃度 Λ，叶子活跃度 Λ_b（Λ_b 很大 = "叶子全占"边界）。
再按"交换左右子树"的自同构合并（深度 4 时 27652 个轨道），精确求 t_rel（单位 1/c = n(1+λ)）。
用法：python3 chordal_tree_blowup.py
"""
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spla
from functools import lru_cache


def orbits(h):
    """深度 h 子树的所有轨道（规范形）。规范形 = (占?, 左规范形, 右规范形)，左右按排序。叶子：(占?,)。"""
    if h == 0:
        return [(0,), (1,)]
    sub = orbits(h - 1); out = []
    for i, a in enumerate(sub):
        for b in sub[i:]:
            pair = tuple(sorted((a, b)))                      # 与 canon 的排序一致
            out.append((0,) + pair)
            if a[0] == 0 and b[0] == 0:
                out.append((1,) + pair)
    return out


def canon(t):
    if len(t) == 1: return t
    a, b = canon(t[1]), canon(t[2])
    return (t[0],) + ((a, b) if a <= b else (b, a))


def flips(t, parent_occ):
    """枚举在规范形 t 上翻转一个点的所有转移，产出 (新子树, 类型)。类型：0=删除，1=插入内部点，2=插入叶子。"""
    occ = t[0]
    if occ:
        yield (0,) + t[1:], 0
    else:
        kids_free = len(t) == 1 or (t[1][0] == 0 and t[2][0] == 0)
        if not parent_occ and kids_free:
            yield (1,) + t[1:], (2 if len(t) == 1 else 1)
    if len(t) > 1:
        for side in (1, 2):
            for new, kind in flips(t[side], occ):
                s = list(t); s[side] = new; yield tuple(s), kind


def counts(t):
    """(内部被占点数, 叶子被占点数)。"""
    if len(t) == 1: return (0, t[0])
    a, b = counts(t[1]), counts(t[2]); return (t[0] + a[0] + b[0], a[1] + b[1])


def size(t):
    if len(t) == 1: return 1
    s = size(t[1]) * size(t[2]); return s * (2 if t[1] != t[2] else 1)


_struct = {}
def structure(h):
    """与 Λ 无关的结构：轨道、每个轨道的 (内部数, 叶子数, log 轨道大小)，以及转移列表。只建一次。"""
    if h not in _struct:
        O = orbits(h); ix = {o: k for k, o in enumerate(O)}
        cnt = np.array([counts(o) for o in O]); lsz = np.log([size(o) for o in O])
        rows, cols, kinds = [], [], []
        for k, o in enumerate(O):
            for new, kind in flips(o, 0):
                rows.append(k); cols.append(ix[canon(new)]); kinds.append(kind)
        _struct[h] = (len(O), cnt, lsz, np.array(rows), np.array(cols), np.array(kinds))
    return _struct[h]


def trel(h, Lam, Lamb):
    """合并链的 relaxation time（单位 1/c）。"""
    N, cnt, lsz, rows, cols, kinds = structure(h)
    lw = cnt[:, 0] * np.log(Lam) + cnt[:, 1] * np.log(Lamb) + lsz; lw -= lw.max()
    pi = np.exp(lw); pi /= pi.sum()
    vals = np.choose(kinds, [1.0, Lam, Lamb])                       # 删除速率 c=1，插入速率 Λ 或 Λ_b
    Q = sp.csr_matrix((vals, (rows, cols)), shape=(N, N))
    out = np.asarray(Q.sum(1)).ravel()
    d = np.sqrt(pi); S = sp.diags(d) @ Q @ sp.diags(1 / d); S = (S + S.T) / 2 - sp.diags(out)
    if N < 3000:
        return 1 / np.sort(np.linalg.eigvalsh(-S.toarray()))[1]
    # 大规模：在与平稳向量 d 正交的子空间上，用 CG 求解 (−S)x = b 作为算子 (−S)^{-1}，
    # 再用 Lanczos 求它的最大特征值 = 1/谱间隙（−S 半正定，零空间只有 d）
    A = (-S).tocsr(); proj = lambda x: x - d * (d @ x)
    Dinv = 1 / np.maximum(A.diagonal(), 1e-300)
    M = spla.LinearOperator((N, N), matvec=lambda x: Dinv * x)          # Jacobi 预条件
    def solve(b):
        x, info = spla.cg(A, proj(b), rtol=1e-12, maxiter=200000, M=M)
        return proj(x)
    op = spla.LinearOperator((N, N), matvec=solve, dtype=float)
    return spla.eigsh(op, k=1, which="LA", return_eigenvectors=False, tol=1e-8)[0]



if __name__ == "__main__":
    print("t_rel（单位 n(1+λ)；括号里是相邻深度之比）")
    for Lamb_mode in ["free", "boundary"]:
        print(f"\n叶子边界：{'自由（叶子活跃度 = Λ）' if Lamb_mode == 'free' else '叶子活跃度 1e6（叶子全占）'}")
        for Lam in [4, 16, 64, 256, 1024]:
            Lamb = Lam if Lamb_mode == "free" else 1e6
            ts = [max(1.0, trel(h, Lam, Lamb)) for h in range(1, 5)]
            ratio = " ".join(f"{t:9.2f}" + (f"(×{t / ts[i - 1]:5.2f})" if i else " " * 8) for i, t in enumerate(ts))
            print(f"  Λ={Lam:<5} h=1..4: {ratio}", flush=True)
