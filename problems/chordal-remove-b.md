# 去掉 bounded separator 限制：一条基于"团分解"的路线

> 接续 [chordal-hardcore-mixing.md](chordal-hardcore-mixing.md) · 状态：把问题归约为一个分解不等式 + 数值证据 · 更新：2026-11-22
> 标注：**[已证]** 本文给出了完整证明 · **[猜想]** 有数值支持、未证明 · **[需核实]** 引用的定理需要回原文确认常数和条件

---

## 1. 问题

设 $G$ 是 $n$ 个点的弦图，$\lambda>0$，$P$ 是 hardcore 模型上的 heat-bath Glauber dynamics：
- 均匀随机选一个点 $v$；
- 以概率 $\frac{\lambda}{1+\lambda}$ 尝试把 $v$ 放进独立集（$v$ 的邻居都不在集合里时才成功），否则把 $v$ 拿出来。

目标：证明 $t_{\mathrm{rel}}(P)\le \mathrm{poly}(n)\cdot(1+\lambda)$，并且**多项式的次数和分隔集大小、团大小都无关**。

本文的主要结论：
1. 这个目标可以**归约**到一个"按团分解"的单步不等式（§4）。
2. 现有的分解定理只能给出 $|R|\lambda$ 的损失，这个损失**正好**对应 $n^{O(\log(b\lambda))}$。
3. 数值上，真实的损失是一个 $\le 2.4$ 的常数，和 $|R|$、$\lambda$ 都无关。

---

## 2. 背景：为什么想换方法

我们 2020 年的论文用 canonical paths，拥塞度里有一个 $\bar\lambda$ 的一次方，说明路径只需要"一个空位"，不需要跨越很高的能量壁垒。$n^{\log b}$ 来自编码：大约 $\log n$ 层，每层要记录分隔集里的哪个点被拿掉，这是**计数损失**。

所以我们换一个不需要编码的框架：**沿着团做谱分解（decomposition），再在 clique tree 上递归**。

---

## 3. 前置知识

**弦图的平衡团分隔集** **[需核实：Gilbert–Rose–Edenbrandt 1984]**：每个弦图都存在一个团 $R$，使 $G-R$ 的每个连通分量至多有 $n/2$ 个点。

**乘积链的谱间隙** **[已证，标准]**：设 $Q=\sum_i \frac{n_i}{n}\,(I\otimes\cdots\otimes Q_i\otimes\cdots\otimes I)+\frac{n_0}{n}I$，其中 $Q_i$ 是第 $i$ 个分量上自己的链（$n_i$ 个点），$n_0$ 个点被冻结。则
$$\mathrm{gap}(Q)=\min_i \frac{n_i}{n}\,\mathrm{gap}(Q_i).$$

**分解定理（Jerrum–Son–Tetali–Vigoda 2004）** **[需核实常数]**：设 $\Omega=\bigsqcup_x\Omega_x$。投影链 $\bar P(x,y)=\pi(\Omega_x)^{-1}\sum_{I\in\Omega_x,J\in\Omega_y}\pi(I)P(I,J)$，限制链为 $P_x$，记 $\gamma=\max_x\max_{I\in\Omega_x}P(I,\Omega\setminus\Omega_x)$。则
$$\mathrm{gap}(P)\ \ge\ \min\Big\{\tfrac{\mathrm{gap}(\bar P)}{3},\ \frac{\mathrm{gap}(\bar P)\,\min_x\mathrm{gap}(P_x)}{3\gamma+\mathrm{gap}(\bar P)}\Big\}.$$

---

## 4. 推导

### 4.1 按一个团分块

取任意团 $R$。独立集和 $R$ 至多交一个点，所以
$$\Omega=\bigsqcup_{x\in\{\emptyset\}\cup R}\Omega_x,\qquad \Omega_x=\{I: I\cap R=x\}.$$

### 4.2 引理 A：投影链是星形，谱间隙恰为 $\frac{1}{n(1+\lambda)}$ **[已证]**

**证明**：
1. 对 $x\ne y$ 都属于 $R$：单步最多改一个点，而 $x,y$ 不能同时在 $I$ 里，所以 $\bar P(x,y)=0$。投影链是以 $\emptyset$ 为中心的星形链。
2. 对 $x\in R$：任何 $I\in\Omega_x$ 都可以删掉 $x$，概率是 $\frac1n\cdot\frac{1}{1+\lambda}$。所以每个叶子到中心的速率都是 $r=\frac{1}{n(1+\lambda)}$。
3. 星形链的特征值。设中心到叶子 $x$ 的速率是 $q_x$，$Q=\sum_x q_x$。由细致平衡，$q_x\propto\pi(\Omega_x)$。
   - 取 $f(\emptyset)=0$，并且 $\sum_x\pi(\Omega_x)f(x)=0$。这样的 $f$ 满足 $\sum_x q_xf(x)=0$，从而 $(I-\bar P)f=r\,f$。所以特征值 $r$ 的重数是 $|R|-1$。
   - 剩下"叶子整体 vs 中心"的两态链，特征值是 $r+Q$。
   - 所以 $\mathrm{gap}(\bar P)=\min(r,\,r+Q)=r$。$\blacksquare$

关键点：这个间隙**和 $|R|$ 无关，和中心的质量 $\pi(\Omega_\emptyset)$ 也无关**。

### 4.3 引理 B：进入叶子时恰好是平稳分布 **[已证]**

对 $y\in R$，记 $A_y=\{J\in\Omega_\emptyset: N(y)\cap J=\emptyset\}$。

- 映射 $J\mapsto J\cup\{y\}$ 是 $A_y\to\Omega_y$ 的双射，并且 $\pi(J\cup\{y\})=\lambda\pi(J)$。
- 从 $J\in A_y$ 插入 $y$ 的概率是常数 $\frac{\lambda}{n(1+\lambda)}$。

所以：从 $\pi|_{\Omega_\emptyset}$ 出发、发生一次 $\emptyset\to y$ 跳跃之后，状态的分布**恰好**是 $\pi|_{\Omega_y}$。$\blacksquare$

### 4.4 引理 C：单个团的割永远不是瓶颈 **[已证]**

由 B 中的双射（反方向：从 $\Omega_x$ 删掉 $x$）得到 $\pi(\Omega_\emptyset)\ge\pi(\Omega_x)/\lambda$。所以
$$\pi(\Omega_\emptyset)\ \ge\ \frac{1}{1+|R|\lambda}.$$

推论：如果想构造弦图上的**超多项式下界**（Marin 的开放问题 4），瓶颈不能是"某一个团被清空"，必须是多个团的集体行为。

### 4.5 现有分解定理的损失正好是 $|R|\lambda$

对 4.1 的分块：
- $\mathrm{gap}(\bar P)=\frac{1}{n(1+\lambda)}$（引理 A）；
- $\gamma$ 由 $\Omega_\emptyset$ 的出口决定，至多 $\frac{|R|\lambda}{n(1+\lambda)}$。

代入 JSTV 后取倒数：
$$t_{\mathrm{rel}}(P)\ \le\ \max\Big\{3n(1+\lambda),\ \big(3|R|\lambda+1\big)\max_x t_{\mathrm{rel}}(P_x)\Big\}.$$

损失因子 $\gamma/\mathrm{gap}(\bar P)=|R|\lambda$，也就是"中心的出口速率"和"叶子的出口速率"之比。

### 4.6 递归 **[已证，给定单步不等式]**

**命题**：假设对所有弦图 $G$、所有团 $R$、所有 $\lambda$，都有
$$t_{\mathrm{rel}}(G)\ \le\ C\Big(n(1+\lambda)+\max_x t_{\mathrm{rel}}(P_x)\Big).\tag{$\star$}$$
那么 $t_{\mathrm{rel}}(G)\le 2C\,n^{a}(1+\lambda)$，其中 $a=2+\log_2 C$。

**证明**：对 $n$ 归纳。取平衡团 $R$（§3）。

1. 在 $\Omega_x$ 里，$R$ 的状态固定为 $x$，$N(x)$ 中的点被迫为空。剩下的点构成 $G-R-N(x)$，它的连通分量 $G_i$ 都是弦图（诱导子图），并且 $|G_i|=n_i\le n/2$。
2. $P_x$ 恰好是这些分量上各自 Glauber 链的乘积链，所以由乘积链公式
   $$t_{\mathrm{rel}}(P_x)=\max_i\frac{n}{n_i}\,t_{\mathrm{rel}}(G_i).$$
3. 由归纳假设 $t_{\mathrm{rel}}(G_i)\le 2C\,n_i^a(1+\lambda)$，得到
   $$t_{\mathrm{rel}}(P_x)\le 2C\,n\,(n/2)^{a-1}(1+\lambda).$$
4. 代入 $(\star)$：
   $$t_{\mathrm{rel}}(G)\le C\,n(1+\lambda)+\frac{2C^2}{2^{a-1}}\,n^a(1+\lambda).$$
   因为 $2^{a-1}=2C$，第二项等于 $C\,n^a(1+\lambda)$。又因为 $n\le n^a$，第一项至多 $C\,n^a(1+\lambda)$。两项相加 $\le 2C\,n^a(1+\lambda)$。$\blacksquare$

**对照**：
- 如果 $(\star)$ 里的 $C$ 换成 $3|R|\lambda+1$（也就是 JSTV 能给的），同样的递归给出 $n^{O(\log(\omega\lambda))}$。这是 Marin 型的界：$\omega$ 是团数。
- 如果按分隔集 $S$ 而不是整个团分块，同样的论证给出 $n^{O(\log(b\lambda))}$，**重新得到我们 2020 年的结果**，而且不需要编码论证。
- **[需核实]** 这条"按分隔集分块"的路线还需要把引理 A、B 推广到 $S$。$S$ 也是团，所以 A、B 的证明应该原样成立，但需要检查递归时各分量的大小控制。

---

## 5. 数值实验

代码：[`code/chordal_decomposition_check.py`](../code/chordal_decomposition_check.py)

测量两个量：
- $L=\dfrac{t_{\mathrm{rel}}(P)}{\max_x t_{\mathrm{rel}}(P_x)}$：纯乘性损失；
- $M=\dfrac{t_{\mathrm{rel}}(P)}{t_{\mathrm{rel}}(\bar P)+\max_x t_{\mathrm{rel}}(P_x)}$：就是 $(\star)$ 里的 $C$ 的经验值。

**(a) 引理 A**：所有测试实例里，$\mathrm{gap}(\bar P)$ 与 $\frac{1}{n(1+\lambda)}$ 一致到小数点后 6 位，叶子出口速率的 min 和 max 完全相等。

**(b) clique binary tree，按根团分块**：

```
   (d,k,s)     λ |    损失 L |     M | |R|·λ
 (2, 4, 2)     1 |   1.786 | 0.893 | 4
 (2, 4, 2)  1024 |   2.654 | 1.327 | 4096
 (3, 4, 3)  1024 |   1.499 | 0.856 | 4096
 (3, 8, 7)  1024 |   1.499 | 0.856 | 8192
```

$|R|$ 从 4 增大到 8 时，$L$ 和 $M$ 完全不变；$\lambda$ 从 1 增大到 1024 时，二者都饱和。JSTV 的界 $|R|\lambda$ 在这里差了三个数量级。

**(c) 对抗搜索**：
- 实例：200 个随机连通弦图和随机 split graph（$n\le13$），尝试每个极大团作为 $R$，$\lambda\in\{4,64,1024\}$，共 3744 个组合。
- $\lambda=4,\,64,\,1024$ 时，$\max M$ 分别是 $1.64,\ 2.27,\ 2.36$。
- $\max L=2421$，但这是单团本身需要的 $n(1+\lambda)$ 时间（$P_x$ 平凡、$P$ 慢在团内）。它不是递归损失，所以 $(\star)$ 里保留了基础项。

**结论**：在所有测试过的实例上，$(\star)$ 对 $C\approx 2.4$ 成立。如果这在一般情况下成立，就有 $t_{\mathrm{rel}}=O(n^{3.3}(1+\lambda))$。

**局限**：$n\le 22$；clique tree 深度 $\le 3$。需要更大规模、更有针对性的搜索（§6 方向 4）。

---

## 6. 结论与疑问

**[猜想] 团分解不等式**：存在绝对常数 $C$，使得对所有弦图 $G$、团 $R$ 和 $\lambda>0$，$(\star)$ 成立。

⇒ 由 §4.6，任意弦图上 Glauber 的 relaxation time 是 $\mathrm{poly}(n)(1+\lambda)$。混合时间再乘 $\log\pi_{\min}^{-1}\le n\log(1+\lambda)$，仍然是多项式。

**这把开放问题变成了一个"单层"的问题**：只需要处理一个团和它挂着的若干个独立子问题，不再有递归深度的事。

**为什么 JSTV 会亏 $|R|\lambda$，怎么省回来**：
- **（推导草稿，没有完整写出）** 用方差分解 $\mathrm{Var}(f)=\mathbb E[\mathrm{Var}(f\mid\sigma_R)]+\mathrm{Var}(\mathbb E[f\mid\sigma_R])$。
- 用引理 B 的双射，把 $\mathbb E[f\mid\Omega_y]-\mathbb E[f\mid\Omega_\emptyset]$ 拆成两部分：
  1. 跨块边 $J\to J\cup\{y\}$ 上的差，这部分由 Dirichlet 形式直接控制；
  2. $\mathbb E_\emptyset[f\mid A_y]-\mathbb E_\emptyset f$，即条件期望的偏移。
- 第二部分里需要控制
  $$\sum_{y\in R}\lambda\,\pi(A_y\mid\emptyset)\,\big(\mathbb E_\emptyset[f\mid A_y]-\mathbb E_\emptyset f\big)^2.$$
- 由 Cauchy–Schwarz，这个量 $\le \mathbb E_\emptyset\big[(f-\mathbb E_\emptyset f)^2\cdot\lambda N\big]$，其中 $N(J)=\#\{y\in R: y\text{ 可插入}\}$。
- 它的**平均值**很小：$\lambda\,\mathbb E_\emptyset N=\dfrac{1-\pi(\Omega_\emptyset)}{\pi(\Omega_\emptyset)}$。
- 损失来自用 $\sup N\le|R|$ 代替平均值。
- 直觉：$N$ 大的状态在 $P_\emptyset$ 下是"短命"的。$y$ 可插入意味着 $y$ 的邻居全空，而在大 $\lambda$ 下这些邻居会很快被填上。所以集中在 $\{N$ 大$\}$ 上的函数 $f$ 要付出 Dirichlet 能量。
- **具体目标**：证明一个加权 Poincaré 型不等式
  $$\mathbb E_\emptyset\big[(f-\mathbb E_\emptyset f)^2\,\lambda N\big]\ \le\ C_1\,\mathrm{Var}_\emptyset(f)+C_2\,n(1+\lambda)\,\mathcal E_{P_\emptyset}(f,f),$$
  其中 $C_1,C_2$ 与 $|R|$ 无关。可以尝试 Lyapunov / drift 函数的方法，以 $N$ 为势函数。

**可以尝试的方向**：
1. **证明上面的加权不等式**。最值得先做的特例：$R$ 只挂一个子团（两个团共享 $s$ 个点）。这时可以显式算。
2. **查文献**：有没有利用"进入时恰好平稳"（引理 B）的分解定理？可能相关的有：Madras–Randall 2002 的 overlapping decomposition；Martinelli 的 block dynamics / "bisection" 技巧；Eppstein–Frishberg 的 projection–restriction 框架里对 "boundary" 的处理。
3. **先拿到 $\mathrm{poly}(\lambda)$ 版本**：哪怕证明 $C=\mathrm{poly}(\lambda)$、与 $|R|$ 无关，也已经去掉了 $b$。得到的是 $n^{O(\log\lambda)}$，对固定 $\lambda$ 是多项式。这比去掉 $\lambda$ 依赖容易，可能是第一篇论文的合理目标。
4. **更大规模的数值**：用 DP（[`chordal_hardcore_dp.py`](../code/chordal_hardcore_dp.py)）精确计算加权不等式左右两边，对特殊函数类测试，比如只依赖子树根团状态的 $f$。也可以对 $(\star)$ 做更大实例的 Lanczos。

**疑问**：
- 平衡团分隔集的大小可以是 $\omega$，但引理 A 对任意大小的团都成立。所以有望连 $\omega$ 一起消掉。这和 Marin 的 tree-$\alpha$ 结果的下界构造（含 $C_4$）冲突吗？不冲突：tree-$\alpha\ge2$ 时 bag 不是团，引理 A 不成立。这也说明"bag 是团"正是关键。
- 我们 2020 年证明的是 Dyer–Greenhill 链。这条路线直接针对 Glauber，如果走通，可以顺便补上 Glauber 的情况。

---

## 7. 参考资料

1. M. Jerrum, J.-B. Son, P. Tetali, E. Vigoda. *Elementary bounds on Poincaré and log-Sobolev constants for decomposable Markov chains.* Ann. Appl. Probab. 14(4), 2004.
2. R. A. Martin, D. Randall. *Sampling adsorbing staircase walks using a new Markov chain decomposition method.* FOCS 2000.
3. N. Madras, D. Randall. *Markov chain decomposition for convergence rate analysis.* Ann. Appl. Probab. 12(2), 2002.
4. J. R. Gilbert, D. J. Rose, A. Edenbrandt. *A separator theorem for chordal graphs.* SIAM J. Algebraic Discrete Methods 5(3), 1984.
5. 以及 [chordal-hardcore-mixing.md](chordal-hardcore-mixing.md) 里的全部参考资料。
