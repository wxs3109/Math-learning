r"""
分层星套星的大规模（稀疏）计算：深度 4（b=2，27652 个合并状态）。
在 d⊥ 上用 CG 求 (−S)^{-1}，Lanczos 取最大特征值 = t_rel·c。只在双精度可靠的 Λ ≤ 1e4 使用。
"""
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spla
import chordal_hier_star as H


def log_trel_sparse(tree, ch, e, logLam):
    N, rows, cols, coef, layer = ch
    loglam = [ei * logLam for ei in e]
    lw = np.array([tree.logw(0, t, loglam) for t in tree.types[0]]); lw -= lw.max()
    pi = np.exp(lw); pi /= pi.sum()
    rate = coef * np.where(layer >= 0, np.exp(np.array(loglam + [0.0])[layer]), 1.0)
    Q = sp.csr_matrix((rate, (rows, cols)), shape=(N, N)); out = np.asarray(Q.sum(1)).ravel()
    d = np.sqrt(pi); S = sp.diags(d) @ Q @ sp.diags(1 / d); S = (S + S.T) / 2 - sp.diags(out)
    A = (-S).tocsr(); proj = lambda x: x - d * (d @ x)
    Dinv = 1 / A.diagonal(); M = spla.LinearOperator((N, N), matvec=lambda x: Dinv * x)
    def solve(b):
        x, _ = spla.cg(A, proj(b), rtol=1e-13, maxiter=500000, M=M); return proj(x)
    op = spla.LinearOperator((N, N), matvec=solve, dtype=float)
    return np.log(spla.eigsh(op, k=1, which="LA", return_eigenvectors=False, tol=1e-9)[0])


if __name__ == "__main__":
    b = 2
    for dep in (3, 4):
        tr = H.Tree((b,) * dep); ch = tr.chain(); g = 1 / (b + 1)
        e = [b * g] + [1.0] * (dep - 1) + [g]
        ps = [2.0, 2.5, 3.0, 3.5, 4.0]
        v = [log_trel_sparse(tr, ch, e, np.log(10 ** p)) for p in ps]
        sl = [(v[i + 1] - v[i]) / np.log(10 ** 0.5) for i in range(len(v) - 1)]
        print(f"b={b} 深度={dep} 状态数={ch[0]}  局部斜率 10^2→10^4: {' '.join('%.3f' % s for s in sl)}"
              f"   t·c(10^4)={np.exp(v[-1]):.4g}", flush=True)
