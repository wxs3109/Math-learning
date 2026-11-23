r"""
分层"星套星"：减速指数会不会随层数增长？（problems/chordal-remove-b.md §18）

H = 球对称树：根在第 0 层，第 ℓ 层每个点有 b_{ℓ+1} 个孩子，叶子在第 d 层。第 ℓ 层活跃度 Λ^{e_ℓ}。
由膨胀引理（§17.2），这对应一个弦图：每个点换成 m = Λ^{e_ℓ}/λ 个孪生点，时间以 c = 1/(n(1+λ)) 为单位。

合并：同一个父亲下的孩子可以互换，所以子树状态只需记录"各类孩子子树各有几个"（类型计数）。
合并链的谱是原链谱的子集 ⇒ 合并链的 t_rel 是真实 t_rel 的下界。
用法：python3 chordal_hier_star.py
"""
import numpy as np, itertools
from functools import lru_cache
from math import lgamma, log


def compositions(total, parts):
    """把 total 拆成 parts 个非负整数之和的所有方式。"""
    if parts == 1:
        yield (total,); return
    for i in range(total + 1):
        for rest in compositions(total - i, parts - 1):
            yield (i,) + rest


class Tree:
    def __init__(self, branching):
        self.b = list(branching); self.d = len(branching)
        self.types = {self.d: [(0,), (1,)]}                       # 叶子：占 / 不占
        for lv in range(self.d - 1, -1, -1):
            ch = self.types[lv + 1]; out = []
            for cnt in compositions(self.b[lv], len(ch)):
                out.append((0, cnt))
                if all(c == 0 or ch[i][0] == 0 for i, c in enumerate(cnt)):
                    out.append((1, cnt))                          # 自己占 ⇒ 孩子都不占
            self.types[lv] = out
        self.index = {lv: {t: k for k, t in enumerate(T)} for lv, T in self.types.items()}

    def logw(self, lv, t, loglam):
        """类型 t 的对数权重（含多项式系数），loglam[ℓ] = 第 ℓ 层的 log 活跃度。"""
        if lv == self.d: return t[0] * loglam[lv]
        occ, cnt = t; ch = self.types[lv + 1]
        s = occ * loglam[lv] + lgamma(self.b[lv] + 1) - sum(lgamma(c + 1) for c in cnt)
        return s + sum(c * self.logw(lv + 1, ch[i], loglam) for i, c in enumerate(cnt) if c)

    @lru_cache(maxsize=None)
    def moves(self, lv, t, parent_occ):
        """从类型 t 出发的转移：[(新类型, 速率系数, 活跃度所在层或 -1)]。速率 = 系数 × Λ_层（层=-1 表示删除，速率 1）。"""
        out = []
        if lv == self.d:
            if t[0]: out.append(((0,), 1, -1))
            elif not parent_occ: out.append(((1,), 1, lv))
            return tuple(out)
        occ, cnt = t; ch = self.types[lv + 1]
        if occ: out.append(((0, cnt), 1, -1))
        elif not parent_occ and all(c == 0 or ch[i][0] == 0 for i, c in enumerate(cnt)):
            out.append(((1, cnt), 1, lv))
        for i, c in enumerate(cnt):
            if not c: continue
            for new, coef, layer in self.moves(lv + 1, ch[i], occ):
                j = self.index[lv + 1][new]; cnt2 = list(cnt); cnt2[i] -= 1; cnt2[j] += 1
                out.append(((occ, tuple(cnt2)), coef * c, layer))
        return tuple(out)

    def chain(self):
        T = self.types[0]; N = len(T)
        rows, cols, coef, layer = [], [], [], []
        for k, t in enumerate(T):
            for new, cf, ly in self.moves(0, t, 0):
                rows.append(k); cols.append(self.index[0][new]); coef.append(cf); layer.append(ly)
        return N, np.array(rows), np.array(cols), np.array(coef, float), np.array(layer)


def log_trel(tree, ch, e, logLam):
    """log(t_rel·c)，各层活跃度 Λ^{e_ℓ}。稠密对称特征值。"""
    N, rows, cols, coef, layer = ch
    loglam = [ei * logLam for ei in e]
    lw = np.array([tree.logw(0, t, loglam) for t in tree.types[0]]); lw -= lw.max()
    pi = np.exp(lw); pi /= pi.sum()
    rate = coef * np.where(layer >= 0, np.exp(np.array(loglam + [0.0])[layer]), 1.0)
    Q = np.zeros((N, N)); np.add.at(Q, (rows, cols), rate)
    np.fill_diagonal(Q, -Q.sum(1))
    d = np.sqrt(pi); S = d[:, None] * Q / d[None, :]; S = (S + S.T) / 2
    return -np.log(np.sort(np.linalg.eigvalsh(-S))[1])


def best_exponents(tree, ch, grid=12, L1=np.log(1e2), L2=np.log(10**3.5), restarts=6, seed=0):
    rng = np.random.default_rng(seed); D = tree.d + 1; best = (-1, None)
    f = lambda e: (log_trel(tree, ch, e, L2) - log_trel(tree, ch, e, L1)) / (L2 - L1)
    for _ in range(restarts):
        e = list(rng.integers(0, grid + 1, D)); cur = f([x / grid for x in e]); imp = True
        while imp:
            imp = False
            for i in range(D):
                for v in range(grid + 1):
                    if v == e[i]: continue
                    e2 = e.copy(); e2[i] = v; a = f([x / grid for x in e2])
                    if a > cur + 1e-9: e, cur, imp = e2, a, True
        if cur > best[0]: best = (cur, [x / grid for x in e])
    return best


if __name__ == "__main__":
    for br in [(3,), (2, 2), (3, 3), (2, 4), (2, 2, 2), (2, 2, 3), (3, 3, 2)]:
        tr = Tree(br); ch = tr.chain()
        if ch[0] > 6000:
            print(f"分支 {br}: 状态数 {ch[0]} 太大，跳过"); continue
        a, e = best_exponents(tr, ch)
        sl = [(log_trel(tr, ch, e, np.log(10.0 ** (p + 0.5))) - log_trel(tr, ch, e, np.log(10.0 ** p))) / np.log(10 ** 0.5)
              for p in (1.5, 2.0, 2.5, 3.0, 3.5)]
        print(f"分支 {br}（深度 {len(br)}，状态数 {ch[0]}）：各层指数 {e}  局部斜率 "
              f"{' '.join('%.3f' % s for s in sl)}", flush=True)
