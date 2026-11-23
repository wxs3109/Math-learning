r"""
用 mpmath 高精度计算分层星套星合并链的谱间隙（problems/chordal-remove-b.md §18），
以便把 Λ 推到 1e12 以上，看渐近减速指数。双精度在 Λ≳1e6 时误差和间隙同阶。
用法：python3 chordal_hier_mp.py
"""
import mpmath as mp
import numpy as np
import chordal_hier_star as H
mp.mp.dps = 80


def log_trel_mp(tree, ch, e, Lam):
    """log(t_rel·c)，高精度。Lam 是 mpmath 数。"""
    N, rows, cols, coef, layer = ch
    loglam = [mp.mpf(ei) * mp.log(Lam) for ei in e]
    lw = [mp.mpf(0)] * N
    for k, t in enumerate(tree.types[0]):
        lw[k] = mp.mpf(tree.logw(0, t, [float(x) for x in loglam]))  # 对数权重的结构部分
    # 为了避免 logw 的双精度误差，直接用高精度重算：logw 是 loglam 的线性函数 + 常数（多项式系数）
    base = np.array([tree.logw(0, t, [0.0] * len(e)) for t in tree.types[0]])
    coeff = []
    for i in range(len(e)):
        unit = [0.0] * len(e); unit[i] = 1.0
        coeff.append(np.array([tree.logw(0, t, unit) for t in tree.types[0]]) - base)
    lw = [mp.mpf(base[k]) + sum(mp.mpf(round(coeff[i][k])) * loglam[i] for i in range(len(e))) for k in range(N)]
    m = max(lw); pi = [mp.e ** (x - m) for x in lw]; Z = sum(pi); pi = [p / Z for p in pi]
    lam_layer = [mp.e ** x for x in loglam]
    Q = mp.zeros(N, N)
    for r, c_, cf, ly in zip(rows, cols, coef, layer):
        Q[int(r), int(c_)] += mp.mpf(cf) * (lam_layer[int(ly)] if ly >= 0 else 1)
    for i in range(N):
        Q[i, i] = -sum(Q[i, j] for j in range(N) if j != i)
    d = [mp.sqrt(p) for p in pi]
    S = mp.matrix(N, N)
    for i in range(N):
        for j in range(N):
            S[i, j] = -(d[i] * Q[i, j] / d[j] + d[j] * Q[j, i] / d[i]) / 2
    ev = sorted(mp.eigsy(S, eigvals_only=True))
    return mp.log(1 / ev[1])


if __name__ == "__main__":
    for b, d in [(2, 1), (2, 2), (3, 1), (3, 2), (2, 3)]:
        tr = H.Tree((b,) * d); ch = tr.chain(); g = 1 / (b + 1)
        e = [b * g] + [1.0] * (d - 1) + [g]
        ps = [4, 6, 8, 10, 12] if ch[0] < 100 else [4, 6, 8, 10]
        v = [log_trel_mp(tr, ch, e, mp.mpf(10) ** p) for p in ps]
        sl = [float((v[i + 1] - v[i]) / mp.log(mp.mpf(10) ** (ps[i + 1] - ps[i]))) for i in range(len(v) - 1)]
        print(f"b={b} 深度={d} 状态数={ch[0]} 预测α={(d*b-1)/(b+1):.3f}  局部斜率（10^{ps[0]}→10^{ps[-1]}，每两个数量级）:",
              " ".join("%.3f" % s for s in sl), flush=True)
