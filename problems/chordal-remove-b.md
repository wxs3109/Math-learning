# 去掉 bounded separator 限制：一条基于"团分解"的路线

> 接续 [chordal-hardcore-mixing.md](chordal-hardcore-mixing.md) · 状态：G′ 的难点已定位到非对角元；有块对角引理 + 核心事件分解方案 · 目标 n^{O(log λ)} · 更新：2026-11-22
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
3. 数值上，在随机弦图上真实的损失 $\le 2.4$。但 §8.4 的"公共阻挡点"结构表明，对某些团 $R$，损失必须随 $\lambda$ 增长。所以正确的目标是 $\mathrm{poly}(\lambda)$、与 $|R|$ 无关，得到 $n^{O(\log\bar\lambda)}$。

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

**结论**：在这些随机实例上，$(\star)$ 对 $C\approx 2.4$ 成立。（⚠️ §8.4 给出了 $C$ 随 $\lambda$ 增长的结构化例子，所以这里的常数不能推广成绝对常数。）

**局限**：$n\le 22$；clique tree 深度 $\le 3$。需要更大规模、更有针对性的搜索（§6 方向 4）。

---

## 6. 结论与疑问

**[猜想] 团分解不等式**：存在绝对常数 $C$，使得对所有弦图 $G$、团 $R$ 和 $\lambda>0$，$(\star)$ 成立。

> ⚠️ **更正（见 §8.4）**：这个"绝对常数"版本是错的。公共阻挡点结构给出 $C=\Omega(\lambda)$。正确的目标是 $C=\mathrm{poly}(\bar\lambda)$，与 $|R|$ 无关，由此得到 $n^{O(\log\bar\lambda)}$。下面"$O(n^{3.3})$"的说法相应作废。

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

## 7. 证明尝试：$C$ 与 $|R|$ 无关（2026-11-22）

代码：[`code/chordal_mixture_means_check.py`](../code/chordal_mixture_means_check.py)

### 7.1 记号

- 固定团 $R$。令 $\pi_\emptyset=\pi(\Omega_\emptyset)$，$\mu_0=\pi(\cdot\mid\Omega_\emptyset)$。
- 对 $y\in R$：$A_y=\{J\in\Omega_\emptyset: N(y)\cap J=\emptyset\}$，$p_y=\mu_0(A_y)$，$\mu_y=\mu_0(\cdot\mid A_y)$，$w_y=\pi(\Omega_y)=\lambda\pi_\emptyset p_y$。令 $w_0=\pi_\emptyset$，$W=\sum_y w_y=1-\pi_\emptyset$。
- 投影 $\mathrm{pr}:\Omega\to\Omega_\emptyset$ 把 $I$ 中属于 $R$ 的那个点删掉（如果有的话）。由引理 B，$\mathrm{pr}_*\pi=\nu:=\sum_{x\in\{0\}\cup R}w_x\mu_x$。
- 对 $f:\Omega\to\mathbb R$：
  - $F=f|_{\Omega_\emptyset}$；
  - $\delta_y(J)=f(J\cup\{y\})-f(J)$，其中 $J\in A_y$；
  - $m_x=\mathbb E_{\mu_x}F$，$d_y=m_y-m_0$。
- $\tau=\max_x t_{\mathrm{rel}}(P_x)$，$\mathcal E$ 是 $P$ 的 Dirichlet 形式。

### 7.2 命题 1（归约到"混合分量均值"） **[已证]**

$$t_{\mathrm{rel}}(P)\ \le\ 4\tau+6n(1+\lambda)+2K,\qquad K:=\sup_f\frac{\mathrm{Var}_w(m)}{\mathcal E(f,f)}.$$

**证明**：
1. 记 $D(f)=\sum_y\sum_{J\in A_y}\pi(J\cup\{y\})\,\delta_y(J)^2$。边 $J\cup\{y\}\to J$ 的转移概率是 $\frac{1}{n(1+\lambda)}$，所以 $D(f)\le n(1+\lambda)\,\mathcal E(f,f)$。
2. 把 $f$ 写成 $f=F\circ\mathrm{pr}+\delta$。由 $(a+b)^2\le 2a^2+2b^2$，
   $$\mathrm{Var}_\pi f\le 2\,\mathrm{Var}_\nu F+2D.$$
3. 混合测度的全方差公式：
   $$\mathrm{Var}_\nu F=\sum_x w_x\mathrm{Var}_{\mu_x}F+\mathrm{Var}_w(m).$$
4. 估计分量内方差：
   - $x=0$：$\mathrm{Var}_{\mu_0}F\le\tau\,\mathcal E_{P_\emptyset}(F)$。
   - $x=y$：$\mathrm{Var}_{\mu_y}F\le 2\,\mathrm{Var}_{\mu_y}f(\cdot\cup\{y\})+2\,\mathbb E_{\mu_y}\delta_y^2$，其中第一项 $\le 2\tau\,\mathcal E_{P_y}$。
   - 各个限制链的 Dirichlet 形式（按 $w_x$ 加权）之和不超过 $\mathcal E$。所以
     $$\sum_x w_x\mathrm{Var}_{\mu_x}F\le 2\tau\,\mathcal E+2D.$$
5. 合起来：$\mathrm{Var}_\pi f\le(4\tau+6n(1+\lambda))\,\mathcal E+2\,\mathrm{Var}_w(m)$。$\blacksquare$

⇒ 要证 $(\star)$，只需证 **$K\le C'\,(n(1+\lambda)+\tau)$，其中 $C'$ 与 $|R|$ 无关**。

### 7.3 恒等式：中心部分与叶子部分 **[已证]**

由 $\mathrm{Var}_w(m)=\sum_y w_yd_y^2-(\sum_y w_yd_y)^2$ 直接整理得到
$$\mathrm{Var}_w(m)=\underbrace{\pi_\emptyset\sum_y w_y d_y^2}_{\text{中心部分}}+\underbrace{\sum_y w_y\,(m_y-m_L)^2}_{\text{叶子部分}},\qquad m_L=\frac{\sum_y w_ym_y}{W}.$$

要点：中心部分多了一个因子 $\pi_\emptyset$。旧的论证用 $\sum_y w_yd_y^2$ 而不是这个精确恒等式，正是这里丢了 $\pi_\emptyset$。

### 7.4 引理 2（中心部分：Green 函数界） **[已证]**

令 $h_y=\mathbf 1_{A_y}/p_y$，定义
$$G_y=\big\langle h_y-1,\ (I-P_\emptyset)^{+}(h_y-1)\big\rangle_{\mu_0}.$$
则
$$\pi_\emptyset\sum_y w_y d_y^2\ \le\ \Big(\sum_y w_yG_y\Big)\,\mathcal E(f,f).$$

**证明**：
1. $d_y=\langle F,h_y-1\rangle_{\mu_0}$。设 $(I-P_\emptyset)u_y=h_y-1$。$\Omega_\emptyset$ 是若干个独立集空间的乘积，所以 $P_\emptyset$ 不可约，并且 $h_y-1$ 的均值为零，这个方程有解。
2. 由 Cauchy–Schwarz，$d_y^2=\mathcal E_{P_\emptyset}(F,u_y)^2\le\mathcal E_{P_\emptyset}(F,F)\,G_y$。
3. 再用 $\pi_\emptyset\,\mathcal E_{P_\emptyset}(F,F)\le\mathcal E(f,f)$。$\blacksquare$

$G_y$ 的含义：$p_yG_y=\sum_{t\ge0}\big(\Pr_{\mu_0|A_y}(X_t\in A_y)-p_y\big)$，也就是从 $A_y$ 出发、"多于平稳水平"地停留在 $A_y$ 里的总时间。所以 $w_yG_y=\lambda\pi_\emptyset\cdot(\text{这个超额停留时间})$。

### 7.5 推论 3（弱相关情形） **[已证]**

令 $\Gamma_{yy'}=\mathrm{Cov}_{\mu_0}(\mathbf 1_{A_y},\mathbf 1_{A_{y'}})/\sqrt{p_yp_{y'}}$（事件的归一化 Gram 矩阵）。用 Bessel 不等式：
$$\sum_y p_yd_y^2=\sum_y\langle \tilde F,\tilde h_y\rangle^2\le\|\Gamma\|\,\mathrm{Var}_{\mu_0}F,$$
其中 $\tilde F=F-\mathbb E_{\mu_0}F$，$\tilde h_y=(\mathbf 1_{A_y}-p_y)/\sqrt{p_y}$。于是 $\mathrm{Var}_w(m)\le\sum_yw_yd_y^2\le\lambda\|\Gamma\|\,\tau\,\mathcal E$，从而
$$t_{\mathrm{rel}}(P)\le(4+2\lambda\|\Gamma\|)\,\tau+6n(1+\lambda).$$

- 如果每一层选的团都满足 $\|\Gamma\|=O(1)$（例如 $R$ 中各点的外部邻域互相独立），就得到 $C=O(\lambda)$，与 $b$ 无关。递归后是 $n^{O(\log\bar\lambda)}$。
- **但这条路一般走不通**：孪生族里 $\|\Gamma\|\approx0.6\,s$，随孪生数线性增长（见 7.6）。

### 7.6 数值结果

**(1) 孪生族**：$R$ 中有 $s$ 个点的外部邻域完全相同。$(\star)$ 的经验常数 $M$ 在 $s=1\to12$ 时从 1.01 **降**到 0.79（$\lambda=1024$）。真实动力学没有任何损失。

**(2) $\|\Gamma\|$ 与 $\sum_yw_yG_y/(n(1+\lambda))$**：

| 实例 | $\lambda$ | $\sum w_yG_y/(n(1+\lambda))$ | $\|\Gamma\|$ | $\lvert R\rvert$ |
|---|---|---|---|---|
| twin $s=1$ | 1024 | 1.33 | 1.00 | 2 |
| twin $s=12$ | 1024 | 1.33 | 8.00 | 13 |
| tree (3,8,7) | 1024 | 1.48 | 4.00 | 8 |
| 随机弦图 / split 图 | ≤256 | ≤ 2.79 | — | — |

结论：Gram 矩阵的界随 $|R|$ 涨，**Green 函数的界不涨**。所以 Green 函数是对的工具。

**(3) 直接算 $K$**：把 $K$ 化成广义特征值问题 $\sup_f \mathrm{Var}_w(m)/\mathcal E(f,f)$ 精确求解，单位是 $n(1+\lambda)$：
- 孪生族：总量 ≤ 2.0，其中中心部分 ≤ 0.82，叶子部分 ≤ 1.51；
- 随机弦图 / split 图：总量 ≤ 2.68，中心部分 ≤ 1.52，叶子部分 ≤ 2.39。

所以在所有测试实例上 $K\le 2.7\,n(1+\lambda)$，和 $|R|$、$\lambda$ 都无关。

### 7.7 还缺什么

剩下两个引理，证出来就得到 $(\star)$，从而得到任意弦图上的多项式混合：

- **引理 G（待证）**：$\sum_y w_yG_y\le C_1\,(n(1+\lambda)+\tau)$。
  - 直觉：$\lambda\pi_\emptyset\cdot(\text{在 }A_y\text{ 里的超额停留时间})$ 对 $y$ 求和。
  - 难点：平凡界 $p_yG_y\le\tau(1-p_y)$ 求和后会产生 $|R|$。需要利用两点：$\sum_y\lambda\pi_\emptyset p_y=W\le1$，以及在 $A_y$ 里停留时间短的点（$y$ 的外部邻居容易被填上）占据了大部分权重。
  - 孪生族上可以手算验证：$\pi_\emptyset=1/(1+s\lambda p)$ 正好抵消 $s$。
- **引理 L（待证）**：叶子部分 $\sum_yw_y(m_y-m_L)^2\le C_2\,\mathrm{poly}(\lambda)\,(n(1+\lambda)+\tau)\,\mathcal E(f,f)$。
  - 难点：叶子均值之差只能"经过中心"比较，而中心的权重 $\pi_\emptyset$ 可能很小。
  - 直觉：$A_y\cap A_{y'}$ 中的状态 $J$ 同时连着 $J\cup\{y\}$ 和 $J\cup\{y'\}$，中心状态和叶子状态的质量比是 $1:\lambda$。所以经由中心"搭桥"的代价应该是 $\lambda$，而不是 $|R|\lambda$。如果能证明各个 $A_y$ 之间有足够的重叠，或者把重叠少的情况单独处理，就能拿到 $\mathrm{poly}(\lambda)$。
  - 这正是"先拿 $\mathrm{poly}(\lambda)$ 版本"的意义：叶子部分允许丢一个 $\lambda$。

---

## 8. 引理 G 的尝试：修正与反例（2026-11-22）

代码：[`code/chordal_lemma_g_check.py`](../code/chordal_lemma_g_check.py)

### 8.1 结论速览

1. **§7 的引理 G（迹的形式）是错的**：它会随 $|R|$ 增长，而真正要控制的中心部分不会。
2. **改用算子范数（引理 2′）**：严格成立，数值上几乎是紧的。
3. **常数不可能与 $\lambda$ 无关**："公共阻挡点"结构里，真实的 relaxation time 会多出一个约为 $\lambda$ 的因子。
4. **修正后的目标**：$\|\Gamma^Z\|\le C\bar\lambda\,(n(1+\lambda)+\tau)$。数值上 $C\le0.45$，与图的大小无关。递归后得到 $n^{O(\log\bar\lambda)}$，对固定 $\lambda$ 是多项式。

### 8.2 反例 1：迹的形式不成立

图：$r$-团 $R$，每个点挂 $m$ 个叶子。此时各个 $A_y$ 依赖的是不相交的叶子，所以相互独立。对角元 $w_yG_y$ 每个都不小，迹 $\sum_yw_yG_y$ 随 $r$ 线性增长；但真实的中心部分并不增长。

| $r$ | $m$ | $\lambda$ | 迹 $\sum w_yG_y$ | $\|\Gamma^Z\|$ | 真实中心部分 |
|---|---|---|---|---|---|
| 2 | 3 | 16 | 0.684 | 0.342 | 0.342 |
| 4 | 3 | 16 | 1.360 | 0.340 | 0.340 |
| 5 | 2 | 16 | 2.041 | 0.409 | 0.406 |
| 9 | 1 | 16 | 0.842 | 0.094 | 0.076 |

（单位都是 $n(1+\lambda)$。）

### 8.3 引理 2′（算子范数版） **[已证]**

令 $\varphi_y=h_y-1$，$Z=(I-P_\emptyset)^+$（在均值为零的子空间上），
$$\Gamma^Z_{yy'}=\sqrt{w_yw_{y'}}\;\langle\varphi_y,\,Z\varphi_{y'}\rangle_{\mu_0}.$$
则
$$\pi_\emptyset\sum_yw_yd_y^2\ \le\ \|\Gamma^Z\|_{\mathrm{op}}\ \mathcal E(f,f).$$

**证明**：heat-bath Glauber 是半正定的，所以 $I-P_\emptyset\succeq0$。
1. 令 $a=(I-P_\emptyset)^{1/2}\tilde F$、$b_y=Z^{1/2}\varphi_y$，则 $d_y=\langle \tilde F,\varphi_y\rangle=\langle a,b_y\rangle$，并且 $\|a\|^2=\mathcal E_{P_\emptyset}(F,F)$。
2. 由此
   $$\sum_yw_y\langle a,b_y\rangle^2=a^{\!\top}\Big(\sum_yw_yb_yb_y^{\!\top}\Big)a\le\Big\|\sum_yw_yb_yb_y^{\!\top}\Big\|\,\|a\|^2 .$$
3. 而 $\big\|\sum_yw_yb_yb_y^{\!\top}\big\|=\|\Gamma^Z\|$：对 $B=[\sqrt{w_y}b_y]$，$BB^{\!\top}$ 和 $B^{\!\top}B$ 的非零特征值相同。
4. 再用 $\pi_\emptyset\mathcal E_{P_\emptyset}\le\mathcal E$。$\blacksquare$

它同时推广了前面两个结果：
- 由 $\|\Gamma^Z\|\le\mathrm{tr}\,\Gamma^Z$ 得到引理 2；
- 由 $Z\preceq\tau I$ 得到 $\|\Gamma^Z\|\le\tau\lambda\pi_\emptyset\|\Gamma\|$，即推论 3。

所以在 whiskers 例子（Gram 矩阵好、迹不好）和孪生例子（迹好、Gram 矩阵不好）上，它都是好的。

### 8.4 反例 2：公共阻挡点 → 常数必须依赖 $\lambda$

**图**：$R=\{y,r\}$（$r$ 挂一个叶子）；$U=K_u$，全部连到 $y$；另有一个点 $v$ 连到 $U$ 的全部（$U\cup\{v\}$ 是团）。

**机制**：
- $v$ 被占时，$U$ 全被挡住，$y$ 可以被占。
- 从"$v$ 占"切换到"$U$ 中某点占"，需要先清空 $v$ 和 $y$。
- 反方向：从"$U$ 中某点占"出发，空位产生后，$U$ 以约 $u/(u+1)$ 的概率把空位"抢回来"，链又回到原来那一侧。
- 这是**熵陷阱**。慢化因子约为 $\min(u,\lambda)$。

**数值**（单位 $n(1+\lambda)$）：

| $\lambda$ | $u=1$ | $u=4$ | $u=16$ | $u=64$ | $u=256$ |
|---|---|---|---|---|---|
| 真实 $t_{\mathrm{rel}}$，$\lambda=16$ | 2.36 | 3.71 | 5.76 | 7.27 | 7.85 |
| 真实 $t_{\mathrm{rel}}$，$\lambda=64$ | 2.59 | 4.74 | 10.41 | 19.10 | 25.07 |

- 对 $R=\{y,r\}$，$(\star)$ 的经验常数 $M$ 随 $u$ 增长（$\lambda=1024,u=12$ 时 $M=5.9$），最终饱和在 $\Theta(\lambda)$。**所以"对所有团 $R$，$C$ 是绝对常数"是错的。**
- 换一个团做分块时 $M\le1.5$：慢化被吸收进了子问题的 $\tau$。这说明递归时**选哪个团**有影响。
- 这个结构的慢化对固定 $\lambda$ 是有界的（约 $\lambda/2$），所以它**不能**直接构造出超多项式下界：逐层叠加只给出 $\lambda^{O(\log n)}=n^{O(\log\lambda)}$。

### 8.5 修正后的目标 **[猜想]**

**引理 G′**：存在绝对常数 $C_1$，使得对所有弦图、所有团 $R$ 和 $\lambda>0$，
$$\|\Gamma^Z\|_{\mathrm{op}}\ \le\ C_1\,\bar\lambda\,\big(n(1+\lambda)+\tau\big).$$

数值上 $\|\Gamma^Z\|/(\bar\lambda\,n(1+\lambda))$ 的最大值：
- 公共阻挡点族：0.42（$u\to\infty$ 时趋近约 0.5）；
- 随机弦图 / split 图：0.33。

**如果 G′ 和引理 L 的 $\mathrm{poly}(\lambda)$ 版本都成立**：$(\star)$ 对 $C=O(\mathrm{poly}(\bar\lambda))$ 成立，由 §4.6 得到 $t_{\mathrm{rel}}=n^{O(\log\bar\lambda)}$。这去掉了 $b$，也去掉了 $\omega$。

### 8.6 证明 G′ 的思路（草稿）

- **对角元**：用 Thomson 原理（Green 能量 = 最小流能量）。以 $U=N(y)\setminus R$ 是团的情形为例：
  - 把每个 $J\in A^c$ 的质量沿"删掉 $U$ 中那个点"这一条边送回 $J\setminus U\in A$。这部分能量是 $p^2(1-p)\,n(1+\lambda)$。
  - 剩下的残差 $\mu(1-pZ_U)\mathbf 1_A$ 用 $\tau$ 路由，其中 $Z_U=1+\lambda N_U$，$N_U$ 是 $U$ 中可插入点的个数。
  - 合起来得到
    $$w_yG_y\le w_y\big[\,2n(1+\lambda)+2\tau\,D_y\big],\qquad D_y=\frac{\mathrm{Var}_{\nu_{A_y}}(Z_U)}{\mathbb E_{\nu_{A_y}}Z_U}\ (\text{离散指数}).$$
  - 在公共阻挡点例子里，$D_y\approx\lambda u$ 但 $w_y\approx\lambda\pi_\emptyset/u$，乘积是 $\mathrm{poly}(\lambda)$。但这只是对角元。
- **非对角元**：$\langle\varphi_y,Z\varphi_{y'}\rangle$。$P_\emptyset$ 是 $G-R$ 各连通分量上链的乘积，所以只依赖不相交分量的 $\varphi_y,\varphi_{y'}$ 的交叉项为 0。需要处理的是共享分量的 $y,y'$。
  - 在 clique tree 里，这对应 $R$ 的同一个子树。可以按子树分组，把 $\Gamma^Z$ 写成分块矩阵。块之间的耦合来自 $h_y=\prod_i h_y^{(i)}$ 的乘积结构。
- **难点**：一般 $U$ 不是团（$y$ 的外部邻居可以分布在多个子树里），这时 $Z_U$ 是多元的配分函数，残差的离散指数可能很大。

---

## 9. 对角元与块结构（2026-11-22）

代码：[`code/chordal_diag_block_check.py`](../code/chordal_diag_block_check.py)

### 9.1 更正：对角元本身是平凡的

由 $\langle\varphi,Z\varphi\rangle\le\tau_\emptyset\|\varphi\|^2$，有 $G_y\le\tau_\emptyset(1-p_y)/p_y$，所以
$$\Gamma^Z_{yy}=w_yG_y=\lambda\pi_\emptyset p_yG_y\ \le\ \lambda\pi_\emptyset(1-p_y)\,\tau_\emptyset\ \le\ \bar\lambda\,\tau .$$

也就是说 G′ 的对角元自动满足目标。我在 §8.7 说"先证 $U$ 是团时的对角元"，这个下一步选错了。**G′ 的全部难度在非对角元**：有多少个 $y$ 会"串通"起来，让算子范数比最大对角元大很多。
- whiskers 例子：非对角元全为零。
- 孪生例子：矩阵接近秩一。

### 9.2 命题 D′：纤维分解（$U_y$ 是团） **[已证]**

这个命题的价值不在于给对角元一个更好的界，而在于它给出的分解方式：把"$U$ 内部的快速动力学"和"环境的慢变化"分开。这正是处理非对角元需要的工具。

**设定**：$U=N(y)\setminus R$ 是团。把 $J\in\Omega_\emptyset$ 写成 $J=e\cup S$，其中环境 $e=J\setminus U$，$S=J\cap U$，$|S|\le1$。
- $N(e)$ 是给定 $e$ 时 $U$ 中可插入点的个数，$Z(e)=1+\lambda N(e)$。
- 记 $p_e=\Pr(A\mid e)=1/Z(e)$。$e$ 在 $\mu_0$ 下的边缘分布记为 $\rho$，于是 $\rho(e)=\mu_0(e)Z(e)$，$p=\mathbb E_\rho[1/Z]$。

**命题**：令 $\varphi=\mathbf 1_A-p$，则
$$\langle\varphi,Z\varphi\rangle\ \le\ 2n(1+\lambda)\,\mathbb E_\rho\!\Big[\frac{Z-1}{Z^3}\Big]+2\tau_\emptyset\,\mathrm{Var}_\rho(1/Z).$$

**证明**：把 $\varphi$ 拆成 $\varphi_1=\mathbf 1_A-p_e$（纤维内部）和 $\varphi_2=p_e-p$（只依赖环境）。由 $Z\succeq0$，$\langle\varphi,Z\varphi\rangle\le2\langle\varphi_1,Z\varphi_1\rangle+2\langle\varphi_2,Z\varphi_2\rangle$。

1. **Thomson 原理**：对可逆链和均值为零的 $\psi$，$\langle\psi,Z\psi\rangle$ 等于所有"净流出量 $=\mu\psi$"的流 $\theta$ 中，能量 $\sum_{\text{边}}\theta^2/Q$ 的最小值，其中 $Q(x,x')=\mu(x)P(x,x')$。最优流是 $\theta^*=Q\,\nabla(Z\psi)$。
2. **纤维部分**：在每个纤维里，从 $e$ 沿插入边向每个可插入的 $e\cup\{z\}$ 送 $\lambda\mu_0(e)/Z(e)$ 的流量。
   - 散度检查：$e$ 处的净流出是 $\mu_0(e)\lambda N/Z=\mu_0(e)\varphi_1(e)$；$e\cup\{z\}$ 处的净流入是 $\lambda\mu_0(e)/Z=-\mu_0(e\cup\{z\})\varphi_1(e\cup\{z\})$。✓
   - 边的导纳 $Q=\lambda\mu_0(e)/(n(1+\lambda))$。
   - 能量 $=n(1+\lambda)\sum_e\mu_0(e)\lambda N/Z^2=n(1+\lambda)\,\mathbb E_\rho[(Z-1)/Z^3]$。
3. **环境部分**：$\langle\varphi_2,Z\varphi_2\rangle\le\tau_\emptyset\,\mathrm{Var}_{\mu_0}(p_e)=\tau_\emptyset\,\mathrm{Var}_\rho(1/Z)$。$\blacksquare$

**推论**：由 $(Z-1)/Z^3\le 1/Z^2\le1/Z$ 和 $\mathrm{Var}_\rho(1/Z)\le p(1-p)$，
$$w_yG_y\le 2\lambda\pi_\emptyset\big(n(1+\lambda)+(1-p_y)\tau_\emptyset\big).$$

**数值**：在公共阻挡点族上，左边/右边 ≈ 0.50–0.52，也就是只差一个因子 2。旧的"离散指数"版本（命题 D）在同样的例子上是 0.008，松了两个数量级。随机弦图 / split 图上，左边/右边的最大值是 0.64。

### 9.3 引理 E：块对角 **[已证]**

**引理**：若 $U_y$ 与 $U_{y'}$ 落在 $G-R$ 的不相交连通分量集合里，则 $\Gamma^Z_{yy'}=0$。

**证明**：
1. $\mu_0$ 是 $G-R$ 各分量上 hardcore 测度的乘积。$P_\emptyset$ 每步选点的随机性与状态无关，所以不同分量的坐标在任意时刻都相互独立地演化。
2. 设 $\varphi_y$ 只依赖分量集 $\mathcal C$ 上的坐标，$\varphi_{y'}$ 只依赖与 $\mathcal C$ 不相交的分量集 $\mathcal C'$ 上的坐标。那么 $\mathbb E[\varphi_y(X_0)\varphi_{y'}(X_t)]=\mathbb E\varphi_y\cdot\mathbb E\varphi_{y'}=0$ 对每个 $t$ 成立。
3. 对 $t$ 求和得 $\langle\varphi_y,Z\varphi_{y'}\rangle=0$。$\blacksquare$

**数值**：whiskers 上非对角元 / 对角元 $\approx10^{-16}$，即机器精度。

⇒ $\|\Gamma^Z\|=\max_{\text{块}}\|\Gamma^Z_{\text{块}}\|$。这里的"块"指通过共享 $G-R$ 的分量而连在一起的那些 $y$。

**结构事实** **[已证]**：在以 $R$ 为一个节点的 clique tree 里，$G-R$ 的每个分量都只落在 $R$ 的某一个子树 $T_c$ 中。理由：$R$ 之外的点出现的 bag 构成一棵不含 $R$ 的连通子树；不同子树中 $R$ 之外的点之间没有边。

设某个子树的根团是 $C_c$，分隔集 $S_c=R\cap C_c$：
- 每个 $y\in S_c$ 都和 $C_c\setminus S_c$ 中的所有点相邻（因为 $C_c$ 是团）；
- 于是 $A_y\subseteq B_c:=\{C_c\setminus S_c\text{ 全空}\}$。

**同一个子树对应的那些 $y$，所有 $A_y$ 都共享一个核心事件 $B_c$。** 孪生例子就是 $A_y=B_c$ 的极端情形。

⚠️ 一个 $y$ 可能同时属于多个子树的分隔集。这时它会把多个子树连进同一个块，并且 $A_y\subseteq\bigcap_{c:\,y\in S_c}B_c$。下面 9.4 先处理"一个块只涉及一个子树"的情形；多子树的块要按 $y$ 所属的子树集合进一步分组。

### 9.4 核心事件分解：下一步的具体方案

在一个块里，令 $B=B_c$，$p_B=\mu_0(B)$，$A_y=B\cap E_y$，$q_y=\mu_0(E_y\mid B)$。把 $\varphi_y$ 写成
$$\varphi_y=\underbrace{\big(\mathbf 1_B/p_B-1\big)}_{\text{核心，与 }y\text{ 无关}}+\underbrace{\frac{\mathbf 1_B}{p_B}\Big(\frac{\mathbf 1_{E_y}}{q_y}-1\Big)}_{\text{条件在 }B\text{ 上的剩余部分}} .$$
则在半正定意义下 $\Gamma^Z\preceq 2\,\Gamma_{\text{core}}+2\,\Gamma_{\text{rest}}$。

**核心部分** **[已证]**：$\Gamma_{\text{core}}$ 是秩一矩阵，范数等于 $\big(\sum_{y\in\text{块}}w_y\big)G_B\le G_B$。$C_c\setminus S_c$ 是团，所以对 $B$ 用命题 D′，并记 $\kappa_B=\mathrm{CV}^2_\rho(1/Z_B)$（$1/Z_B$ 在环境分布下的平方变异系数），得到
$$\|\Gamma_{\text{core}}\|\ \le\ 2(1+\kappa_B)\,n(1+\lambda)+2\,\kappa_B\,\tau .$$

- 孪生例子里 $\kappa_B$ 很小，这就解释了 §7.6 的观察：Gram 矩阵的界涨，真实值不涨。
- **但 $\kappa_B$ 本身可以很大**：公共阻挡点让 $1/Z_B$ 在 $1$ 和 $1/(1+\lambda|D|)$ 之间跳。这时需要把"$\kappa_B\cdot\tau$"细化成"$1/Z_B$ 这个环境函数在环境动力学下的 Green 能量"。而这个 Green 能量是下一层子树上的同类问题。

**剩余部分**：$\Gamma_{\text{rest}}$ 是"条件在 $C_c\setminus S_c$ 全空"之后的同一个问题。条件在 $B$ 上以后，$C_c$ 下面的子树进一步解耦，可以再次使用引理 E 和结构事实。

⇒ 这提示了一个**沿 clique tree 向下的归纳**：每层要么分出秩一的核心部分（代价是 $\mathrm{poly}(\lambda)$ 乘环境波动），要么按引理 E 块对角化。需要证明的是：环境波动 $\kappa$ 这一类量在归纳中保持 $\mathrm{poly}(\lambda)$ 有界，而不随团的大小增长。

---

## 10. 参考资料

1. M. Jerrum, J.-B. Son, P. Tetali, E. Vigoda. *Elementary bounds on Poincaré and log-Sobolev constants for decomposable Markov chains.* Ann. Appl. Probab. 14(4), 2004.
2. R. A. Martin, D. Randall. *Sampling adsorbing staircase walks using a new Markov chain decomposition method.* FOCS 2000.
3. N. Madras, D. Randall. *Markov chain decomposition for convergence rate analysis.* Ann. Appl. Probab. 12(2), 2002.
4. J. R. Gilbert, D. J. Rose, A. Edenbrandt. *A separator theorem for chordal graphs.* SIAM J. Algebraic Discrete Methods 5(3), 1984.
5. 以及 [chordal-hardcore-mixing.md](chordal-hardcore-mixing.md) 里的全部参考资料。
