# 弦图上 hardcore model 的快速混合：能否去掉 bounded separator 限制？

> 状态：文献梳理完成 + 小规模数值实验 · 更新：2026-11-22
> 标注约定：**[已证]** 文献中已证明 · **[推论]** 由已知定理直接推出 · **[猜想] / [思路]** 我们自己的想法，未验证 · **[需核实]** 需要回原文确认

---

## 1. 问题

给定弦图 $G$ 和任意逸度 $\lambda>0$，在独立集上做局部 Markov 链（Glauber dynamics，或 Dyer–Greenhill 的 insert/delete/drag 链），目标分布是

$$\pi(I) = \frac{\lambda^{|I|}}{Z_G(\lambda)},\qquad Z_G(\lambda)=\sum_{I\in\mathcal I(G)}\lambda^{|I|}.$$

**核心问题**：对**所有**弦图（不限制最小分隔集大小），混合时间是否是 $\mathrm{poly}(n)$？

已知的最好结果（包括我们 2020 年的论文）都只给出 $n^{O(\log b)}$，其中 $b$ 是某种"宽度"：只有 $b=O(1)$ 时才是多项式，一般弦图只得到拟多项式 $n^{O(\log n)}$。

---

## 2. 背景与动机

- 一般图上，$\lambda$ 大于唯一性阈值 $\lambda_c(\Delta)$ 时近似采样是 NP-hard，所以"任意 $\lambda$ 都快速混合"只能指望特殊图类。
- "任意 $\lambda$"的已知图类主要靠**结构**，不靠衰减相关性：
  - claw-free 图（Matthews；Jerrum–Sinclair 匹配链的推广）
  - 有界 bipartite pathwidth（Dyer–Greenhill–Müller）
  - 有界 treewidth（Eppstein–Frishberg）
  - 有界分隔集的弦图（Bezáková–Sun，就是我们的论文）
  - 有界 tree-independence number（Marin，包含所有弦图，但只给出拟多项式）
- 有意思的一点：弦图上 $Z_G(\lambda)$ **可以精确地在多项式时间内算出来**，所以**采样问题本身是容易的**。这里研究的纯粹是"自然局部链能不能快速混合"。弦图是检验 Glauber dynamics 的一个干净的测试场：结构很强，但度数、treewidth 都没有上界。

### 2.1 精确计算：clique tree DP **[已证 + 已对拍]**

- **文献**：Okamoto–Uno–Uehara 给出了弦图上独立集计数的**线性时间**算法（$\lambda=1$）；也给出了固定大小独立集计数的多项式算法，所以独立集多项式 $\sum_k i_k\lambda^k$ 的全部系数都能算。但在弦图上，**极大**独立集计数是 #P-complete 的。
- **带权版本的 DP**：取一棵 clique tree，以 $t$ 为根的子树顶点集记为 $V_t$。定义
  $$f_t(x)=\sum\{\lambda^{|I|}: I\in\mathcal I(G[V_t]),\ I\cap C_t=x\},\qquad x\in\{\emptyset\}\cup C_t,$$
  对每个孩子 $c$，设分隔集 $S=C_t\cap C_c$，转移为
  $$f_t(x)=w(x)\prod_{c}g_c(x),\qquad
  g_c(x)=\begin{cases} f_c(x)/\lambda, & x\in S\\[2pt] \sum_{y\in\{\emptyset\}\cup(C_c\setminus S)} f_c(y), & x\notin S\end{cases}$$
  其中 $w(\emptyset)=1$、$w(v)=\lambda$。最后 $Z=\sum_x f_{\text{root}}(x)$。
- **为什么正确**：
  1. 独立集和每个团至多交 1 个点，所以每个 bag 只有 $|C_t|+1$ 种状态。
  2. 由 running intersection，$x\in C_t\setminus S$ 和 $y\in C_c\setminus S$ 不会同时出现在任何一个 bag 里，所以不相邻。父子之间唯一的耦合就是分隔集 $S$。
- **复杂度**：$O\big(\sum_t |C_t|\cdot\deg_T(t)\big)$，相对 clique tree 的大小是线性的。
- **采样**：自顶向下，按 $f$ 的条件比例依次决定每个 bag 的状态，就能从 $\pi$ 精确采样，代价相同。
- **对拍**：[`code/chordal_hardcore_dp.py`](../code/chordal_hardcore_dp.py)
  - 200 个随机弦图（$n\le16$，有理数 $\lambda$），DP 得到的 $Z$ 和暴力枚举**精确相等**（全程用有理数运算）。
  - $P_{10}$ 的独立集数是 144，等于 $F_{12}$。
  - 在 $n=7$、64 个独立集的图上采样 20 万次，经验分布和精确分布的 TV 距离是 0.0068，和采样噪声的量级一致。

---

## 3. 前置知识

**弦图与 clique tree** **[已证，经典]**
- 弦图：没有长度 $\ge 4$ 的诱导环。
- 等价刻画：存在一个 tree decomposition，使每个 bag 都是**极大团**（clique tree）。
- 极小分隔集都是团，并且恰好是 clique tree 上相邻两个极大团的交。所以"极小分隔集大小 $\le b$"等价于"任意两个相邻极大团的交 $\le b$"。

**关键结构事实**：一个独立集和每个团最多相交 1 个顶点。所以独立集在一个分隔集 $S$ 上的限制只有 $|S|+1$ 种可能（空，或者选 $S$ 中的某一个点）。

**Tree-independence number**（Dallard–Milanič–Štorgel）：
$$\text{tree-}\alpha(G)=\min_{\text{tree decomp.}}\ \max_{t}\ \alpha(G[X_t]).$$
弦图恰好是 tree-$\alpha=1$ 的图（bag 都是团）。

**Canonical paths 界**（Jerrum–Sinclair，Diaconis–Stroock）：给每对状态 $(I,J)$ 指定一条路径 $\gamma_{IJ}$，定义拥塞度

$$\rho = \max_{e=(x,y)} \frac{1}{\pi(x)P(x,y)} \sum_{\gamma_{IJ}\ni e} \pi(I)\pi(J)\,|\gamma_{IJ}|,$$

则 $t_{\mathrm{rel}} \le \rho$，混合时间 $\tau(\varepsilon)\le \rho\,(\ln \pi_{\min}^{-1} + \ln \varepsilon^{-1})$。

---

## 4. 已知结果与证明思路

### 4.1 Bezáková–Sun（COCOON 2020）**[已证]**

- **对象**：Dyer–Greenhill 的 insert/delete/drag 链（论文说结论对 Glauber 也成立，但没有给出证明）。
- **条件**：弦图，极小分隔集大小 $\le b$。
- **结论**：拥塞度 $\le n\,(3+\log_2(b\bar\lambda))\,\bar\lambda$，其中 $\bar\lambda=\max\{1,\lambda\}$；混合时间 $O\!\left(n^{4+\log_2(b\bar\lambda)}\right)$。
- **证明思路**：
  1. 在 clique tree 上反复取平衡分割，递归深度是 $O(\log n)$。
  2. 沿着这个层次结构构造从 $I$ 到 $J$ 的 canonical path。
  3. 用注入式编码（encoding）界住拥塞度。
- **$\log b$ 的来源**：编码里需要记录大约 $\log_2 n$ 个"被临时删掉的顶点"，每个顶点用它在所属分隔集（大小 $\le b$）里的排名表示，所以编码数是 $b^{\log_2 n}=n^{\log_2 b}$。
  - ⚠️ 也就是说，拟多项式来自**编码的计数损失**，不是来自真正的能量瓶颈。这是想改进时最该盯住的地方。
- **例子**：一棵 $\sqrt n$ 个节点的完全二叉树，每个节点换成大小 $\sqrt n$ 的团，相邻团之间连一条边。这个图度数、treewidth、bipartite pathwidth 都没有上界，但分隔集大小是 1。
- **论文自己提的开放问题**：推广到任意弦图，也就是去掉分隔集限制。

### 4.2 Eppstein–Frishberg（ISAAC 2023）**[已证]**

你记得的"bounded treewidth 上什么都能采样"应该就是这篇。

- hardcore Glauber 在 treewidth $t$ 的图上对任意固定 $\lambda>0$ 的混合时间是 $n^{O(t)}$。
- 同一个框架还覆盖了：dominating sets、$b$-edge covers（需要 treewidth 有界），以及 $b$-matchings、极大独立集、极大 $b$-matchings（还需要度数有界）；有界 treewidth 加有界度数时的 $q$-colorings（$q\ge\Delta+2$）。
- 技术：multicommodity flow + projection–restriction 的分治框架。
- 和弦图的关系：弦图的 treewidth $=\omega-1$ 可以很大，所以这个结果不覆盖我们的图类。
- 后续：Efthymiou–Hayes–Štefankovič–Vigoda 在任意树上得到了最优 relaxation time $O(n)$（$\lambda=1$），并在小 $\lambda$ 时得到 $O(n\log n)$ 混合时间。

### 4.3 Marin（arXiv 2504.18427，WALCOM 2026）**[已证]**

- **Thm（path decomposition）**：$\tau=(b\tilde\lambda)^{2\,\text{path-}\alpha(G)+2}\cdot n^{O(1)}$，其中 $\tilde\lambda=e^{|\ln\lambda|}$，$b$ 是达到该 path-$\alpha$ 的 path decomposition 的最小 bag 大小。
- **Thm（tree decomposition）**：$\tau=n^{O(\text{tree-}\alpha(G))\cdot\ln(b\tilde\lambda)}$，$b$ 是达到 tree-$\alpha$ 的 tree decomposition 的最小 bag 大小。
- 论文说这推广了 Bezáková–Sun（因为弦图就是 tree-$\alpha=1$）。
- **下界**：存在 tree-$\alpha=2t$ 的图，混合时间 $\ge(\lambda n)^{ct\ln n}$，所以对一般的有界 tree-$\alpha$ 图类，拟多项式是紧的。但这个构造含诱导 $C_4$，**不是弦图**。
- **开放问题 4**：是否存在弦图 $G$，使得对足够大的常数 $\lambda$ 有 $\tau_{G,\lambda}=\Omega(n^{\ln n})$？
  - 这正是我们的问题的另一面：**上界能不能降到多项式，下界能不能做到超多项式，目前都不知道。**

**两个由此直接得到的观察：**

- **[推论] 区间图是多项式的。** 区间图有一个所有 bag 都是团的 path decomposition（clique path），所以 path-$\alpha=1$。代入 path 定理得到 $(b\tilde\lambda)^4 n^{O(1)}=\mathrm{poly}(n)$，其中 $b\le n$。
  - ⇒ 难点完全在**分叉**：clique tree 是路时没问题，是"茂密的树"时才有 $\log$ 深度的递归开销。
- **[需核实] 对弦图，Marin 的界可能比我们的弱。** 在 tree-$\alpha=1$ 的分解里每个 bag 都是团，而每个极大团必须被某个 bag 包含，所以 Marin 的 $b\ge\omega(G)$（团数）。我们的 $b$ 是分隔集大小，最多 $\omega-1$，可能小得多。
  - 例子：上面那个"团二叉树"中 $\omega=\sqrt n$、分隔集是 1。Marin 的界给出 $n^{O(\log n)}$，我们的给出多项式。
  - 所以 Marin 推广的是**图类**，不是**界本身**。待核实：他的证明能否改写成只依赖 adhesion（相邻 bag 的交）大小。

### 4.4 相关但不同的方向

- **Heinrich（2020）**：弦图上 $k\ge(1+\varepsilon)(\Delta+1)$ 种颜色时，colorings 的 Glauber dynamics 是多项式混合；有界 treewidth、$k\ge\Delta+2$ 时也是。这说明弦图上 colorings 也有类似的结构型结果。
- **Chen（SODA 2024，arXiv 2307.08212）**：treewidth $t$ 时混合时间是 $n^{O(1+t\log(1+\lambda))}$，加强了 Eppstein–Frishberg。方法是在平衡分隔集树上做 block factorization，是和我们问题最接近的框架（详见 [chordal-remove-b.md](chordal-remove-b.md) §11.2）。
- **Split graph**：取 bag $K\cup\{s\}$ 得 path-$\alpha\le2$，由 Marin 的定理 2 是多项式的 **[推论]**。加上区间图，开放的只剩"分叉的"弦图。
- **Spectral independence 一系列工作**（Anari–Liu–Oveis Gharan，Chen–Liu–Vigoda，Chen–Feng–Yin–Zhang，…）只覆盖 $\lambda<\lambda_c(\Delta)$，处理不了任意 $\lambda$。

---

## 5. 数值实验

代码：[`code/chordal_glauber_gap.py`](../code/chordal_glauber_gap.py)

**设置**：
- 图族 "clique binary tree"：深度为 depth 的完全二叉树，每个节点是 $k$-团，子团和父团共享 $s$ 个顶点（所以分隔集大小是 $s$）。$s=0$ 时父子团之间连一条边，对应论文里的例子。
- 链：heat-bath Glauber。
- 方法：精确枚举所有独立集，对称化转移矩阵，求第二大特征值，得到 $t_{\mathrm{rel}}=1/\text{gap}$。
- 表格括号里是 $t_{\mathrm{rel}}/(n\lambda)$。单点更新链在大 $\lambda$ 下，删除一个点的速率是 $\frac{1}{n(1+\lambda)}$，所以 $n\lambda$ 是自然的时间尺度。

```
== A: depth=2, k=4, 变化 s ==
depth   k   s    n      #IS |       λ=1          λ=4          λ=16          λ=64
    2   4   0   28    62105 |  76.5( 2.73)  209.9( 1.87)  739.0( 1.65)  2854.3( 1.59)
    2   4   1   22     8528 |  51.1( 2.32)  130.0( 1.48)  447.0( 1.27)  1714.2( 1.22)
    2   4   2   16      405 |  57.2( 3.57)  180.9( 2.83)  688.0( 2.69)  2726.7( 2.66)
    2   4   3   10       57 |  29.1( 2.91)   86.3( 2.16)  323.1( 2.02)  1281.6( 2.00)

== B: depth=3 ==
    3   3   2   17     2376 |  50.7( 2.98)  147.4( 2.17)  549.6( 2.02)  2178.8( 2.00)
    3   4   2   32   155925 | 127.5( 3.98)  413.4( 3.23) 1591.7( 3.11)  6335.0( 3.09)
    3   4   3   18     2336 |  52.8( 2.93)  155.5( 2.16)  581.7( 2.02)  2306.9( 2.00)

== C: depth=3, s=k-1, k 增大（分隔集变大，树形不变）==
    3   3   2   17     2376 |  50.7( 2.98)  ...                         2178.8( 2.00)
    3   6   5   20     2338 |  58.7( 2.94)  ...                         2563.2( 2.00)
    3  10   9   24     2342 |  72.1( 3.01)  ...                         3075.9( 2.00)
```

**解读**：
1. 所有实例在大 $\lambda$ 下都有 $t_{\mathrm{rel}}\approx c\cdot n\lambda$，$c\in[1.2,3.1]$。看不到任何超多项式迹象。
2. 实验 C：树形固定时，把分隔集从 2 增大到 9，$c$ 稳定在 2.00。说明只增大分隔集、不增加递归深度，不会产生瓶颈。这和"$\log b$ 来自编码而非能量壁垒"的判断一致。
3. 实验 A 对比 B：$k=4,s=2$ 时深度从 2 到 3，$c$ 从 2.66 涨到 3.09。如果 $c$ 随深度持续增长（比如每层乘一个常数），总体会是 $n^{O(1)}$。如果每层乘的因子随 $s$ 增大，就接近 $n^{\log s}$。**这正是需要区分的东西，但现在的规模（深度 $\le3$）区分不了。**
4. 局限：精确枚举到大约 $10^5$ 个状态就到头了（depth=3、$k=3$、$s=1$ 已经有 190 万个状态）。

**下一步实验想法**：
- 用 clique tree 上的 transfer/DP 精确计算**具体割的 conductance**，不枚举状态，这样可以做到深度 10 以上。候选割：按根团的占据状态划分。
- 或者用 coupling-from-the-past / 经验自相关时间估计 $t_{\mathrm{rel}}$。精确 $Z$ 可以由 DP 得到，用来做无偏检验。

---

## 6. 结论与疑问

**目前搞清楚的**：
- 一般弦图的混合时间，最好的上界是拟多项式 $n^{O(\log n)}$（我们的结果和 Marin 的结果都给出这个），下界没有超多项式的。**两边都是开的**（我们论文的 open problem，也是 Marin 的 open problem 4）。
- 区间图（clique tree 是路）已经是多项式的 **[推论]**，所以困难只在 clique tree 分叉很多、分隔集又大的情况。
- 拟多项式来自 canonical path 的编码损失 $b^{\log n}$。小规模数值实验里看不到相应的瓶颈。

**可能的研究方向**：
1. **[思路] 用 multicommodity flow 代替单一 canonical path。** 编码损失来自"必须记下每层删掉的是分隔集里的哪个点"。如果把流在这 $\le b$ 个选择上**随机平摊**，可能把乘性的 $b$ 换成常数。Eppstein–Frishberg 用 flow 的方式值得对照。
2. **[思路] 沿 clique separator 用分解定理**（Martin–Randall / Jerrum–Son–Tetali–Vigoda）。
   - 投影链只有 $|S|+1$ 个状态，是一个"星形链"：中心是"$S$ 全空"，叶子是"选了 $S$ 中某一点"。
   - 关键量是这个星形链的谱间隙，它取决于中心的质量 $\pi(S\cap I=\emptyset)$。
   - 如果能证明每层的损失因子是和 $|S|$ 无关的常数，那么 $O(\log n)$ 层合起来是 $\mathrm{poly}(n)$。
   - **[猜想]** 风险点：大 $\lambda$ 时中心质量可能很小，星形链变慢。需要算一个具体例子。
3. **下界方向（Marin OP4）**：尝试构造弦图，使大 $\lambda$ 下存在需要同时清空 $\Theta(\log n)$ 个分隔集的瓶颈。Marin 的下界用的是含 $C_4$ 的二叉树构造。能否"弦化"它同时保住瓶颈？
4. **Glauber vs. Dyer–Greenhill 链**：我们的证明是针对 insert/delete/drag 链的。用比较定理（Diaconis–Saloff-Coste）把结论严格转移到 Glauber，应该只损失多项式因子。值得补全。
5. **[需核实]** 读 Marin 的 Lemma 17 和 Theorem 3 的证明，确认 bag 大小 $b$ 能不能换成 adhesion 大小。如果可以，就同时覆盖"有界 tree-$\alpha$ + 有界 adhesion"这个更大的类。

**疑问**：
- 星形投影链在大 $\lambda$ 时到底多慢？能不能用"条件在父团状态上的子树配分函数比值"界住？
- 有没有比 clique tree 更好的分解，比如用 $\omega$ 很大但"几乎是路"的部分来摊薄深度？

---

## 7. 参考资料

1. I. Bezáková, W. Sun. *Mixing of Markov Chains for Independent Sets on Chordal Graphs with Bounded Separators.* COCOON 2020, pp. 664–676. [PDF](https://par.nsf.gov/servlets/purl/10193339)
2. D. Eppstein, D. Frishberg. *Rapid Mixing for the Hardcore Glauber Dynamics and Other Markov Chains in Bounded-Treewidth Graphs.* ISAAC 2023, LIPIcs 283, 30:1–30:13. [arXiv:2111.03898](https://arxiv.org/abs/2111.03898)
3. M. Marin. *Subexponential and Parameterized Mixing Times of Glauber Dynamics on Independent Sets.* WALCOM 2026. [arXiv:2504.18427](https://arxiv.org/abs/2504.18427)
4. M. Dyer, C. Greenhill, H. Müller. *Counting independent sets in graphs with bounded bipartite pathwidth.* Random Structures & Algorithms 59(2), 2021. [arXiv:1812.03195](https://arxiv.org/abs/1812.03195)
5. M. Heinrich. *Glauber dynamics for colourings of chordal graphs and graphs of bounded treewidth.* 2020. [arXiv:2010.16158](https://arxiv.org/abs/2010.16158)
6. Y. Okamoto, T. Uno, R. Uehara. *Counting the number of independent sets in chordal graphs.* J. Discrete Algorithms 6(2), 2008.
7. C. Efthymiou, T. Hayes, D. Štefankovič, E. Vigoda. *Optimal Mixing via Tensorization for Random Independent Sets on Arbitrary Trees.* APPROX/RANDOM 2023. [arXiv:2307.07727](https://arxiv.org/abs/2307.07727)
8. M. Dyer, C. Greenhill. *On Markov chains for independent sets.* J. Algorithms 35(1), 2000.
