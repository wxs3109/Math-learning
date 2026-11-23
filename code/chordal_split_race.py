r"""
"分裂竞争"族（problems/chordal-remove-b.md §16）：对 (⋆) 的一个反例候选。

图：R = S1 ∪ S2 是 2s-团；D_i 是 u-团，与 S_i 全连（i=1,2）。clique tree 是一条路
    (S1∪D1) — (S1∪S2) — (S2∪D2)，所以这是一个区间图。
直觉：两个"相"——相 1 = {S1 有点占, D2 有点占}，相 2 对称。相 1 → 相 2 必须经过 R 全空，
     而且 S2 要能插入还需要 D2 空：
       路线 a：在 D2 空的时刻删掉 S1 的点，概率 ~1/(1+uλ)；
       路线 b：删掉 S1 的点后 D1 先抢到空位（概率 ~u/(s+u)），再等 D2 空出来。
     取 u ~ sqrt(s/λ) 时两条路线都窄，t_rel ~ sqrt(sλ)·n(1+λ)，而所有限制链的 τ = n(1+λ)。

1. 小实例：全空间精确计算 t_rel，并用 decomposition_stats 给出 (⋆) 的经验常数 M。
2. 大实例：按 S_i、D_i 内的置换对称性合并成 8 个状态（合并链的谱是原链谱的子集，
   所以合并链的 t_rel 是真实 t_rel 的下界）。τ 可以精确算出：τ = n(1+λ)（见 §16）。
用法：python3 chordal_split_race.py
"""
import numpy as np
from chordal_nested_gadget import relaxation_time
from chordal_decomposition_check import decomposition_stats


def build(s, u):
    adj = []
    def add(): adj.append(set()); return len(adj) - 1
    def e(a, b): adj[a].add(b); adj[b].add(a)
    S1 = [add() for _ in range(s)]; S2 = [add() for _ in range(s)]; R = S1 + S2
    for i, a in enumerate(R):
        for b in R[i + 1:]: e(a, b)
    for S in (S1, S2):
        D = [add() for _ in range(u)]
        for i, a in enumerate(D):
            for b in D[i + 1:]: e(a, b)
            for x in S: e(x, a)
    return len(adj), adj, R


def lumped_trel(s, u, lam):
    """合并链（8 个状态）的 relaxation time，单位 n(1+λ)。返回 (t_rel, π(相1), π_∅)。"""
    c = 1.0                                       # 速率都正比于 c = 1/(n(1+λ))，取 c=1 即得以 n(1+λ) 为单位
    # 状态 (r, b1, b2)：r ∈ {0:R 空, 1:S1 某点占, 2:S2 某点占}；b_i ∈ {0:D_i 空, 1:D_i 某点占}
    st = [(0, a, b) for a in (0, 1) for b in (0, 1)] + [(1, 0, b) for b in (0, 1)] + [(2, a, 0) for a in (0, 1)]
    ix = {x: k for k, x in enumerate(st)}
    wt = np.array([(s * lam if r else 1.0) * (u * lam) ** (b1 + b2) for r, b1, b2 in st])
    pi = wt / wt.sum()
    N = len(st); Q = np.zeros((N, N))             # Q = 生成元（连续时间速率，按合并后的类）
    def add(a, b, rate): Q[ix[a], ix[b]] += rate
    for r, b1, b2 in st:
        x = (r, b1, b2)
        if r == 0:
            if b1 == 0: add(x, (1, 0, b2), s * lam * c); add(x, (0, 1, b2), u * lam * c)
            if b2 == 0: add(x, (2, b1, 0), s * lam * c); add(x, (0, b1, 1), u * lam * c)
            if b1 == 1: add(x, (0, 0, b2), c)
            if b2 == 1: add(x, (0, b1, 0), c)
        elif r == 1:
            add(x, (0, 0, b2), c)
            add(x, (1, 0, 1 - b2), (u * lam if b2 == 0 else 1) * c)
        else:
            add(x, (0, b1, 0), c)
            add(x, (2, 1 - b1, 0), (u * lam if b1 == 0 else 1) * c)
    np.fill_diagonal(Q, -Q.sum(1))
    d = np.sqrt(pi); S = (d[:, None] * Q) / d[None, :]; S = (S + S.T) / 2   # 对称化
    ev = np.sort(np.linalg.eigvalsh(-S))
    phase1 = pi[[ix[(1, 0, 0)], ix[(1, 0, 1)]]].sum(); pi0 = pi[:4].sum()
    return 1 / ev[1], phase1, pi0


if __name__ == "__main__":
    print("① 小实例：全空间精确值 vs 合并链（单位 n(1+λ)），以及 (⋆) 的经验常数 M")
    for lam in [1, 4]:
        for s, u in [(2, 1), (4, 2), (6, 2), (8, 3)]:
            n, adj, R = build(s, u); sc = n * (1 + lam)
            _, tfull = relaxation_time(n, adj, lam)
            M = decomposition_stats(n, adj, R, lam)[3]
            print(f"   λ={lam} s={s} u={u} n={n:<3}: 全空间 {tfull / sc:7.3f}   合并 {lumped_trel(s, u, lam)[0]:7.3f}"
                  f"   M(⋆)={M:6.3f}", flush=True)

    print("\n② 大实例（合并链；τ = n(1+λ) 精确成立，所以 M(⋆) = t_rel/(2n(1+λ))）")
    print(f"   {'λ':>4} {'s':>8} {'u*':>6} | {'t_rel/(n(1+λ))':>14} {'M(⋆)':>8} {'√(sλ)':>8} | π(相1)  π_∅")
    for lam in [1, 4, 16]:
        for s in [10**2, 10**3, 10**4, 10**5, 10**6]:
            best = max(range(1, int(4 * np.sqrt(s)) + 2), key=lambda u: lumped_trel(s, u, lam)[0])
            t, p1, p0 = lumped_trel(s, best, lam)
            print(f"   {lam:>4} {s:>8} {best:>6} | {t:14.2f} {t / 2:8.2f} {np.sqrt(s * lam):8.1f} | {p1:.3f}  {p0:.2e}")


def nested_lumped(s, s2, u, lam):
    r"""孪生组 S（s 个点，就是 R）全连到一个分裂竞争零件的 S1'（零件参数 s2,u）。
    G−R 只有一个分量（整个零件），P_∅ 就是零件本身，τ = t_rel(零件)。
    返回 (t_rel(全图), τ) ，单位 n(1+λ)；都用合并链。"""
    c = 1.0
    st = []
    for r in (0, 1):                               # r=1：S 中某点被占（要求 S1' 空）
        for rho in (0, 1, 2):
            for b1 in (0, 1):
                for b2 in (0, 1):
                    if rho == 1 and b1: continue
                    if rho == 2 and b2: continue
                    if r == 1 and rho == 1: continue
                    st.append((r, rho, b1, b2))
    ix = {x: k for k, x in enumerate(st)}
    wt = np.array([(s * lam) ** r * (s2 * lam if rho else 1.0) * (u * lam) ** (b1 + b2) for r, rho, b1, b2 in st])
    pi = wt / wt.sum(); N = len(st); Q = np.zeros((N, N))
    def add(a, b, rate):
        if b in ix: Q[ix[a], ix[b]] += rate
    for x in st:
        r, rho, b1, b2 = x
        if r == 0 and rho != 1: add(x, (1, rho, b1, b2), s * lam * c)       # 插入 S
        if r == 1: add(x, (0, rho, b1, b2), c)                              # 删除 S
        if rho == 0:
            if b1 == 0 and r == 0: add(x, (r, 1, 0, b2), s2 * lam * c)       # 插入 S1'（要求 S 空）
            if b2 == 0: add(x, (r, 2, b1, 0), s2 * lam * c)
        else:
            add(x, (r, 0, b1, b2), c)
        if b1 == 0 and rho != 1: add(x, (r, rho, 1, b2), u * lam * c)
        if b1 == 1: add(x, (r, rho, 0, b2), c)
        if b2 == 0 and rho != 2: add(x, (r, rho, b1, 1), u * lam * c)
        if b2 == 1: add(x, (r, rho, b1, 0), c)
    np.fill_diagonal(Q, -Q.sum(1))
    d = np.sqrt(pi); S = (d[:, None] * Q) / d[None, :]; S = (S + S.T) / 2
    ev = np.sort(np.linalg.eigvalsh(-S))
    return 1 / ev[1], lumped_trel(s2, u, lam)[0]
