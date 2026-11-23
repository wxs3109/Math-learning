# 去掉 bounded separator 限制：一条基于"团分解"的路线

> 接续 [chordal-hardcore-mixing.md](chordal-hardcore-mixing.md) · 状态：G′ 被证伪；整个 K 已精确化为单一迹链 Green 范数（§14）；迹链有精确的远足分解，孪生 + 阻挡族已严格证明 K ≤ 6(1+λ)n(1+λ)（§15）；**(⋆) 被分裂竞争族推翻（区间图，C ~ √(|R|λ)），改为加法形式 (⋆′)，仍给出多项式（§16）**；膨胀归约 + 减速指数搜索（§17）；**⚠️ 重大转折：分层星套星的减速指数随深度线性增长（高精度数值，d≤3），弦图上 Glauber 很可能是 n^{Θ(log n)}，b 的依赖可能去不掉（§18）**；文献核查无撞车 · 目标 n^{O(1+log(1+λ))}（与树上已知最好界一致） · 更新：2026-11-22
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

**弦图的平衡团分隔集** **[标准事实：Gilbert–Rose–Edenbrandt 1984；也可由 clique tree 重心论证得到，见 §11.5]**：每个弦图都存在一个团 $R$，使 $G-R$ 的每个连通分量至多有 $n/2$ 个点。

**乘积链的谱间隙** **[已证，标准]**：设 $Q=\sum_i \frac{n_i}{n}\,(I\otimes\cdots\otimes Q_i\otimes\cdots\otimes I)+\frac{n_0}{n}I$，其中 $Q_i$ 是第 $i$ 个分量上自己的链（$n_i$ 个点），$n_0$ 个点被冻结。则
$$\mathrm{gap}(Q)=\min_i \frac{n_i}{n}\,\mathrm{gap}(Q_i).$$

**分解定理（Jerrum–Son–Tetali–Vigoda 2004，定理 1）** **[已核对原文，§11.3]**：设 $\Omega=\bigsqcup_x\Omega_x$。投影链 $\bar P(x,y)=\pi(\Omega_x)^{-1}\sum_{I\in\Omega_x,J\in\Omega_y}\pi(I)P(I,J)$，限制链为 $P_x$，记 $\gamma=\max_x\max_{I\in\Omega_x}P(I,\Omega\setminus\Omega_x)$。则
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

> ⚠️ **更正（§16）**：(⋆) 本身是错的。分裂竞争族（一个区间图）上，对每个平衡团都有 $C\gtrsim\sqrt{|R|\lambda}$。正确的目标是加法形式 (⋆′)：$t_{\mathrm{rel}}\le C(\lambda)\tau+B(n,\lambda)\,n(1+\lambda)$，递归后仍是多项式，见 §16.4。

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
$$\mathrm{Var}_w(m)=\underbrace{\pi_\emptyset\sum_y w_y d_y^2}_{\text{中心部分}}+\underbrace{W\sum_y w_y\,(m_y-m_L)^2}_{\text{叶子部分}},\qquad m_L=\frac{\sum_y w_ym_y}{W}.$$

> ⚠️ **更正（§14）**：最初写的叶子部分漏了因子 $W=1-\pi_\emptyset\le1$，上式已改正。§7.6、§12 等处按旧公式算出的"叶子部分"和 $K$ 因此都偏大，是保守的上界，结论不受影响。

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
4. **修正后的目标**：$\|\Gamma^Z\|\le C\bar\lambda\,(n(1+\lambda)+\tau)$。数值上 $C\le0.6$（§10），与图的大小无关。递归后得到 $n^{O(\log\bar\lambda)}$，对固定 $\lambda$ 是多项式。

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

> ⚠️ **更正（见 §12）**：下面的引理 G′ 已被"孪生 + 阻挡"组合证伪。真正需要的中心部分界仍然成立，但必须同时用上叶子块的 Dirichlet 能量。

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

## 10. 证伪实验：嵌套阻挡结构（2026-11-22）

代码：[`code/chordal_nested_gadget.py`](../code/chordal_nested_gadget.py)（`--twin` 跑孪生变体）

**目的**：§8.4 的单层公共阻挡点结构给出的慢化约为 $\min(u,\lambda)$。如果把它沿 clique tree 嵌套，慢化会不会随团的大小 $u$ **逐层叠加**？如果会，就说明 $(\star)$ 的常数不可能与团的大小无关，猜想 $n^{O(\log\bar\lambda)}$ 被证伪，方向应转向构造下界。

**方法**：精确计算谱间隙。回溯枚举独立集，构造稀疏转移矩阵，用 Lanczos 求第二大特征值。最大实例约 120 万个状态。

**两种嵌套方式**：
- **单点阻挡嵌套**：阻挡点 $v$ 作为下一层的 "$y$"，下面挂两个子结构。
- **孪生阻挡嵌套**：阻挡者是 $u$-团 $V$，子结构的 $U'$ 连到整个 $V$。相邻层共享大小为 $u$ 的分隔集，正是"分隔集很大"的情形，最接近反例。

**结果**（表中是 $t_{\mathrm{rel}}/(n(1+\lambda))$）：

单点阻挡嵌套：

| 深度 | $\lambda$ | $u$ 小 | $u$ 中 | $u$ 大 |
|---|---|---|---|---|
| 1 | 16 | 2.92 ($u$=2) | 5.76 ($u$=16) | 7.27 ($u$=64) |
| 2 | 16 | 5.47 ($u$=2) | 3.93 ($u$=16) | 3.64 ($u$=32) |
| 3 | 16 | 6.45 ($u$=1) | 3.33 ($u$=3) | 2.79 ($u$=4) |

孪生阻挡嵌套：

| 深度 | $\lambda$ | 数据 |
|---|---|---|
| 1 | 4 | $u$=2: 1.99 · $u$=8: 2.13 · $u$=32: 2.18（饱和） |
| 2 | 4 | $u$=1: 2.56 · $u$=4: 3.91 · $u$=8: 4.58 · $u$=16: 5.10 · $u$=32: 5.43 · $u$=40: 5.50 |
| 2 | 16 | $u$=1: 4.28 · $u$=4: 6.60 · $u$=8: 7.60 · $u$=16: 8.30 · $u$=24: 8.57 |
| 3 | 16 | $u$=1: 6.45 · $u$=2: 9.43 · $u$=3: 11.60 |

**解读**：
1. **单点阻挡嵌套反而越嵌套越快**：深层的阻挡点很少被占，陷阱机制失效。所以这不是一个好的对抗构造。
2. **孪生阻挡嵌套在深度 2 随 $u$ 增长，但增长在减速**。$\lambda=4$ 时，$u$ 每翻倍一次，增量依次是 $0.72,\,0.67,\,0.51,\,0.33$，近似按 $u^{-0.6}$ 衰减。外推极限约为 6.0，和深度 1 的饱和值 2.2 相比，每层的因子约为 2.7。**看不到随 $u$ 无界增长的迹象。**
3. **深度方向**：固定较小的 $u$，比值随深度增长。$\lambda=16$、$u=1$ 时，深度 1, 2, 3 分别是 2.9, 4.3, 6.4。这正是预期中"每层乘一个 $\mathrm{poly}(\lambda)$ 因子"的重构型增长，对固定 $\lambda$ 递归后是 $n^{O(\log\lambda)}$。

4. **G′ 的关键量**（代码 [`code/chordal_nested_gz.py`](../code/chordal_nested_gz.py)）：在孪生阻挡嵌套、深度 2 上，$\|\Gamma^Z\|/(\bar\lambda\,n(1+\lambda))$ 随 $u=2\to24$ 的变化是：$\lambda=4$ 时 $0.30\to0.59$，$\lambda=16$ 时 $0.25\to0.32$。两者都在饱和，而且这里的分母**没有**计入 $\tau$ 项，比 G′ 要求的更严格。§8.5 里的经验常数相应从 0.45 更新为 0.6。

**结论**：在这两种最自然的嵌套构造里，慢化**不随团的大小叠加**。猜想（$(\star)$ 的常数是 $\mathrm{poly}(\lambda)$、与 $|R|$ 无关）通过了这次检验，主线可以继续。

**局限**：
- 深度 3 只能算到 $u\le3$，状态数约 $10^6$。
- 对抗构造只试了两种。
- 外推极限是拟合出来的，不是证明。

---

## 11. 文献核查结果（2026-11-22）

这一轮由一个子 agent 做检索，我对关键条目回原文核对。"V"表示读过原文或摘要，"I"表示推断或未核实。

### 11.1 有没有人已经做了（防撞车）

**没有。**
- 引用 Bezáková–Sun 2020 的论文有 5 篇（OpenAlex 数据，V）：Eppstein–Frishberg；Hébert-Johnson–Lokshtanov–Vigoda（采样弦图本身，不是独立集）；Achlioptas–Zampetakis；Marin；Ju–Liu–Yin 2026（关于唯一性阈值附近的硬度，不涉及弦图）。
- Marin 只有 1 篇引用。他的开放问题 4（弦图上是否可能 $\Omega(n^{\ln n})$）目前没人回答。
- 补充一个推论（I，但很直接）：split graph 可以取 bag $K\cup\{s\}$，所以 path-$\alpha\le2$，由 Marin 的定理 2 是多项式的。加上区间图，**开放的情形只剩"分叉的"弦图**。

### 11.2 Chen（SODA 2024）——最接近的框架 **[V]**

文献：Zongchen Chen, *Combinatorial approach for factorization of variance and entropy in spin systems*, arXiv 2307.08212。

- **定理 1.1**：treewidth 为 $t$ 的图上，hardcore 的 Glauber 混合时间是 $n^{O(1+t\log(1+\lambda))}$。这是 Eppstein–Frishberg 的加强版。
- **方法**：在平衡分隔集树上递归地做 block factorization（Prop 3.1）。总常数等于从根到叶路径上各层常数的乘积。
- **每层常数（Lemma 4.2）**：$C=2(1+\lambda)^t$，来源是 $\mu(\sigma_S=\vec 0)\ge(1+\lambda)^{-|S|}$。
- **两块分解（Lemma 3.4）**：若对所有条件 $y,y'$ 有 $d_{TV}(\pi_X^y,\pi_X^{y'})\le1-\varepsilon_X$，对所有 $x,x'$ 有 $d_{TV}(\pi_Y^x,\pi_Y^{x'})\le1-\varepsilon_Y$，则方差分解常数为 $\frac{2}{\varepsilon_X+\varepsilon_Y}$。

**和我们路线的关系**：
1. 套到弦图上，取团分隔集 $S$，有 $\mu(\sigma_S=\emptyset)\ge1/(1+|S|\lambda)$（引理 C 的类比）。每层损失 $\sim|S|\lambda$，递归给出 $n^{O(\log(b\lambda))}$，正好重新得到 2020 年的结果（I）。
2. Lemma 3.4 的常数是 $\varepsilon_X+\varepsilon_Y$ 的**和**，所以只要一侧重叠大就够了。孪生例子就是这样：$\pi_Y^x$ 与 $x$ 无关，$\varepsilon_Y=1$。但 whiskers 例子两侧都差（$\varepsilon_X\sim1/(r\lambda)$，$\varepsilon_Y\sim(1+\lambda)^{-m}$），而真实动力学并不慢。原因是 Lemma 3.4 对边界条件取**最坏情况**的 TV 距离。
3. 我们的 $\Gamma^Z$（引理 2′）本质上是它的**平均情形**版本：只看平稳分布下的 Green 能量，不对边界条件取上确界。这正是能同时处理 whiskers 和孪生两种例子的原因。

**一个更自然的目标表述**：Chen 在树上（$t=1$）给出的是 $n^{O(1+\log(1+\lambda))}$。我们的猜想等价于：**弦图和树一样，混合时间是 $n^{O(1+\log(1+\lambda))}$，与团的大小、分隔集的大小都无关。**

### 11.3 现成的分解定理能否直接用 **[V]**

- **JSTV 定理 1**：形式为 $\mathrm{gap}\ge\min\{\bar\lambda/3,\ \bar\lambda\lambda_{\min}/(3\gamma+\bar\lambda)\}$，§3 的记忆是对的，"需核实"可以去掉。
- **JSTV 推论 2/3**：若各块之间转移的"落点分布"在**两个方向**上都接近平稳分布，$\gamma$ 可以换成很小的 $\hat\gamma$；完全精确时 $\mathrm{gap}\ge\min\{\bar\lambda,\lambda_{\min}\}$。
- **我们的星形分解只满足一个方向**：从中心到叶子的落点恰好平稳（引理 B）；反方向的落点是 $\pi_\emptyset(\cdot\mid A_y)$，和 $\pi_\emptyset$ 差得很远。所以推论 2/3 不适用。没有找到只用单向条件的现成定理。
- ⇒ 命题 1（§7.2）可以看作一个"**单向精确落点**"的分解引理：缺的那个方向的损失，恰好就是 $\mathrm{Var}_w(m)$ 这一项。这本身可能值得单独写成一个一般引理。
- **其它候选**：
  - Madras–Randall 的重叠覆盖：取 $X_y=\Omega_y\cup A_y$，重叠度 $\Theta\le|K|+1$，又丢了 $|K|$（I）。
  - Miracle–Streib–Streib（arXiv 1910.05184）：用一个 Rayleigh 商 $\hat\delta_2$ 和 $\varepsilon$-正交性代替 $\gamma$。形式上可以用，但需要估计 $\hat\delta_2$，难度和 G′ 相当。
  - Caracciolo–Pelissetto–Sokal 型定理：没有 $\gamma$，但针对的是复合链，不直接适用。

### 11.4 目标是不是最优的 **[V]**

- **完全 $b$-叉树、自由边界**：Martinelli–Sinclair–Weitz 证明了对所有 $\lambda$ relaxation time 都是 $O(n)$（Restrepo 等人在文中引用了这一点）。所以在**完全**树上，指数不会随 $\lambda$ 增长。
- **Restrepo 等人的下界**：需要特意构造的"坏"边界条件，没有自由边界版本。
- **一般树**：Efthymiou–Hayes–Štefankovič–Vigoda 指出，存在一些树，在 $\lambda>C$（$C\approx28$）时有多项式级的慢化。Chen–Yang–Yin–Zhang（arXiv 2404.04668）对所有树证明了 $\lambda<e^2$ 时 relaxation time 是最优的 $O(n)$。
- **至今没有人证明自由边界树上的指数必须随 $\lambda$ 增长**（例如 $n^{\Omega(\log\lambda)}$）。子 agent 提出一个思路：用 $C\log n$ 个悬挂叶子模拟边界条件。这还没有核实（I）。
- ⇒ **我们的目标 $n^{O(1+\log(1+\lambda))}$ 和树上目前已知最好的上界一致，但不一定是最优的。** 是否最优，在树上本身就是开放的。

### 11.5 其它核对

- **Gilbert–Rose–Edenbrandt**，SIAM J. Algebraic Discrete Methods 5(3):306–313, 1984（V：题目、期刊、摘要）。"存在一个团，删去后每个连通分量至多 $n/2$ 个点"这个团形式，是通过二手来源确认的。它也可以由 clique tree 重心论证直接得到。§3 里的"需核实"可以降级为"标准事实"。
- **Okamoto–Uno–Uehara**：WG 2005（LNCS 3787）和 J. Discrete Algorithms 6(2), 2008（V）。

### 11.6 对下一步的影响

1. **没有现成工具能直接给出 G′**，所以主线（核心事件归纳）有必要继续。它的定位也清楚了：这是 Chen 框架的"平均情形"加强。
2. **结果的定位**：如果做成，结论是"弦图和树一样，混合时间是 $n^{O(1+\log(1+\lambda))}$"。这去掉了 2020 年结果和 Marin 结果中的分隔集 / bag 大小依赖。
3. **"单向精确落点分解引理"**（命题 1）可以作为一个独立的技术贡献来写。

---

## 12. G′ 被证伪；问题出在引理 2′ 丢掉了叶子的能量（2026-11-22）

代码：[`code/chordal_core_env_check.py`](../code/chordal_core_env_check.py)、[`code/chordal_twin_blocker_combo.py`](../code/chordal_twin_blocker_combo.py)

### 12.1 核心事件的 Green 能量拆解

在公共阻挡点族里取 $B=\{D=U\text{ 全空}\}$，用命题 D′ 把 $G_B$ 拆成纤维部分 $E_1$ 和环境部分 $E_2$，单位是 $n(1+\lambda)$：

| 实例 | $\lambda$ | $p_B$ | $\kappa_B$ | $G_B$ | $E_1$ | $E_2$ | 平凡界 $\tau_\emptyset\kappa_B$ |
|---|---|---|---|---|---|---|---|
| blocker $u=2$ | 4 | 0.385 | 1.14 | 1.40 | 0.05 | 1.10 | 1.14 |
| blocker $u=32$ | 4 | 0.038 | 20.3 | 20.7 | 0.04 | 20.3 | 20.3 |
| blocker $u=32$ | 64 | 0.031 | 31.0 | 31.0 | 0.00 | 31.0 | 31.0 |
| twin-nested d2 $u=8$ | 16 | 0.206 | 3.71 | 7.72 | 0.00 | 7.66 | 8.82 |

- 纤维部分可以忽略，**$G_B$ 几乎全部来自环境部分**。
- 在 blocker 族里，平凡界 $\tau_\emptyset\kappa_B$ 是**紧的**，而且 $\kappa_B\approx0.76/p_B$ 随 $u$ 增长。所以 $G_B$ 本身确实很大。
- 核心部分之所以小，只是因为它前面的权重 $\sum_y w_y$ 小。§9.4 用 $\sum_yw_y\le1$ 这一步太粗了。

### 12.2 反例：G′ 不成立

**"孪生 + 阻挡"组合**：$R=S\cup\{r\}$，其中 $S$ 是 $s$-团；$D$ 是 $u$-团，与 $S$ 全部相连；$v$ 与 $D$ 全部相连。

| $s$ | $u$ | $\lambda$ | 真实 $t_{\mathrm{rel}}/(n(1+\lambda))$ | $\|\Gamma^Z\|/(\bar\lambda(n(1+\lambda)+\tau_\emptyset))$ | 真实 $K$（中心 / 叶子） | $M(\star)$ |
|---|---|---|---|---|---|---|
| 1 | 32 | 4 | 2.87 | 0.20 | 2.22（0.82 / 1.43） | 1.43 |
| 4 | 32 | 4 | 3.80 | 0.65 | 3.53（1.43 / 2.20） | 1.90 |
| 16 | 32 | 4 | 3.67 | **1.48** | 3.19（1.45 / 1.93） | 1.84 |
| 32 | 32 | 4 | — | — | 2.77（1.30 / 1.69） | 1.68 |
| 16 | 32 | 16 | 5.28 | 0.72 | 4.75（2.15 / 2.94） | 2.64 |

- **$\|\Gamma^Z\|$ 随 $s$ 和 $u$ 一起增长**：$\lambda=4$、$s=16$ 时，$u=2,8,32$ 对应 $0.16,0.55,1.48$。所以 G′ 在现在的写法下是错的。
- **但真正要控制的量都有界**：真实 $t_{\mathrm{rel}}$、真实的 $K$（包括中心部分）、$(\star)$ 的常数 $M$ 都不随 $s,u$ 增长。$s$ 从 16 加到 32 时，它们甚至略微下降。
- ⇒ **猜想 $(\star)$ 仍然成立；失效的是引理 2′ 这个证明工具。**

### 12.3 诊断

引理 2′ 的第 4 步用的是 $\pi_\emptyset\,\mathcal E_{P_\emptyset}(F,F)\le\mathcal E(f,f)$。也就是说，它只用中心块 $\Omega_\emptyset$ 上的 Dirichlet 能量来"支付" $F$ 的变化，而**丢掉了叶子块的能量** $\sum_yw_y\mathcal E_{P_y}$。

在孪生 + 阻挡例子里，$s$ 个孪生点让叶子的总质量 $W=1-\pi_\emptyset$ 接近 1，于是 $\pi_\emptyset$ 很小。可是在 $A_y$ 上，$F(J)\approx f(J\cup\{y\})$，所以 $F$ 在 $A_y$ 内部的变化本来可以由叶子链 $P_y$ 的能量来支付，权重是 $w_y$，而不是 $\pi_\emptyset$。引理 2′ 没有用到这部分，损失大约是 $W/\pi_\emptyset$ 量级。

### 12.4 修正方向

真正需要的不等式就是
$$\pi_\emptyset\sum_yw_yd_y^2\ \le\ C\,\big(n(1+\lambda)+\tau\big)\cdot\Big[\pi_\emptyset\mathcal E_{P_\emptyset}(F)+\sum_yw_y\,\mathcal E_{P_y}\big(f(\cdot\cup\{y\})\big)+\frac{D(f)}{n(1+\lambda)}\Big],$$
其中右边方括号里的量 $\lesssim\mathcal E(f,f)$。数值上这就是 §7.6 和本节算出的"真实中心部分"，在所有实例里都 $\le 2.7\,n(1+\lambda)$。

两个候选做法：
1. **"坍缩链"几何**：按投影 $\mathrm{pr}$ 把每颗星 $\{J\}\cup\{J\cup\{y\}\}$ 压成一个点。坍缩后的链 $Q$ 在 $\Omega_\emptyset$ 上，平稳分布是 $\nu$，边权是 $\pi(J)P_\emptyset(J,J')\,(1+\lambda N(J,J'))$，这同时包含了中心和叶子的能量。在 $Q$ 的几何下重新定义 $\Gamma^Z$，用 $Z_Q$ 代替 $Z$。
   - 已知的障碍：把 $f$ 换成纤维平均 $\bar F$ 时，相邻纤维的大小 $1+\lambda N$ 不同，会多出 $\delta$ 项。粗略处理会损失一个因子 $n(1+\lambda)$，需要更细的比较。
2. **按核心事件分组时用精确权重**：核心部分的范数精确地是 $(\sum_{y\in\text{块}}w_y)\,G_B$，并且 $G_B\approx\tau\kappa_B$ 是紧的。所以要证明的是 $(\sum_{y}w_y)\,\kappa_B\lesssim\mathrm{poly}(\lambda)$ 这一类"权重乘以环境波动"的量有界。孪生 + 阻挡例子说明，单靠中心能量做不到，必须同时用叶子能量。

### 12.5 为什么不换成 Chen 的两块分解框架

两块分解 $\{\sigma_R,\text{其余}\}$ 的最优常数是 $1/(1-\rho_R)$，其中 $\rho_R$ 是 $R$ 的状态与外部配置之间的最大相关系数（HGR 相关）。这已经是"平均情形"的最好版本，比 Chen 的 Lemma 3.4（最坏 TV）更强。代码：[`code/chordal_maxcorr_check.py`](../code/chordal_maxcorr_check.py)

| 实例 | $\lambda$ | $\rho_R$ | $1/(1-\rho_R)$ | 真实 $M(\star)$ |
|---|---|---|---|---|
| twin $s=8$ | 4 | 0.774 | 4.43 | 0.76 |
| twin+blocker $s=1,u=8$ | 4 | 0.817 | 5.45 | 1.30 |
| twin+blocker $s=16,u=16$ | 4 | 0.951 | 20.4 | 1.60 |
| twin+blocker $s=16,u=16$ | 64 | 0.970 | 33.0 | 2.26 |

**两块分解的常数随结构增长，真实常数不增长。**

原因：块动力学"给定其余部分，把 $R$ 整体重采样"会抹掉 Glauber 的一个关键机制，即**空位出现时的竞争**。$R$ 中的点被删掉后，是 $D$ 先抢到空位还是 $R$ 自己抢回来，取决于两边可插入点数的比例。块动力学每次都让 $R$ 的纤维完全平衡，于是在"$R$ 占 ⟺ $D$ 空"强相关时卡住。我们的星形分解靠引理 B（进入叶子时恰好平稳）保留了这个机制。

⇒ **继续留在命题 1 的框架里修，不换框架。**

**状态更新**：
- 引理 2′ 本身仍然成立，只是太松。
- G′（用 $\|\Gamma^Z\|$ 表述的版本）是错的。
- 真正的目标（中心部分 $\lesssim\mathrm{poly}(\lambda)(n(1+\lambda)+\tau)\,\mathcal E$）在所有实例上数值成立。

---

## 13. 修正方案的检验：坍缩链与迹链（2026-11-22）

代码：[`code/chordal_collapsed_check.py`](../code/chordal_collapsed_check.py)、[`code/chordal_fiber_split_check.py`](../code/chordal_fiber_split_check.py)、[`code/chordal_trace_identity.py`](../code/chordal_trace_identity.py)

### 13.1 坍缩链 $Q$：只对"纤维常数"的函数有效

**定义**：$\nu=\mathrm{pr}_*\pi$，即 $\nu(J)=\pi(J)(1+\lambda N(J))$。坍缩链为
$$Q(J,J')=\frac{\pi(J)P_\emptyset(J,J')\,(1+\lambda N(J,J'))}{\nu(J)},$$
其中 $N(J,J')$ 是在 $J,J'$ 处都可插入的 $R$ 中点数。

**刻画**：对在每颗星 $\{J\}\cup\{J\cup\{y\}\}$ 上取常数的 $f$，有 $\mathcal E(f)=\mathcal E_Q(F)$ 精确成立。于是在这类函数上，中心部分 / 能量的上确界恰好是 $\pi_\emptyset\|\Gamma^Q\|$，其中 $\Gamma^Q_{yy'}=\sqrt{w_yw_{y'}}\langle\psi_y,Z_Q\psi_{y'}\rangle_\nu$，$\psi_y=(\mu_y-\mu_0)/\nu$。

**数值**（单位 $n(1+\lambda)$）：

| 实例 | $\lambda$ | 旧 $\|\Gamma^Z\|$ | 新 $\pi_\emptyset\|\Gamma^Q\|$ | 真实中心部分 |
|---|---|---|---|---|
| twin+blocker $s=16,u=32$ | 4 | 11.83 | 0.57 | 1.45 |
| twin+blocker $s=16,u=32$ | 64 | 29.20 | 0.71 | 2.52 |
| blocker $u=32$ | 4 | 1.59 | 0.33 | 0.82 |
| blocker $u=32$ | 64 | 15.46 | 0.66 | 6.71 |
| nested-twin d2 $u=2$ | 64 | 6.71 | 1.59 | 2.42 |

- 新量在所有族上都有界（$\le1.6$），旧量的增长完全消失了。
- **但新量可以远小于真实值**：$\lambda=64$ 的 blocker $u=32$ 上，新量 0.66，真实值 6.71。差出来的部分来自**纤维内部的变化**，也就是空位出现时的竞争。所以只看纤维常数函数是不够的。

### 13.2 直接对纤维取平均也不行

把任意 $f$ 写成 $\bar F\circ\mathrm{pr}+g$，其中 $\bar F$ 是纤维平均，$g$ 是纤维内偏差。测两个量：
- $A=\sup_f\mathcal E_Q(\bar F)/\mathcal E(f)$：在 blocker 族上随 $u$ 增长。$\lambda=4$ 时 $u=8,32$ 对应 4.5 和 14.0；$\lambda=16$ 时对应 6.5 和 18.1。
- $B$（纤维内偏差对中心部分的贡献）：很小，$\le0.9$。

⇒ 损失来自"取平均"这一步本身，它会放大能量。所以 $\bar F$ 不是合适的代表元。

### 13.3 精确刻画：迹链 **[已证 + 数值核对]**

**命题 T**：设 $\mathcal E_{\mathrm{tr}}$ 是 $P$ 的 Dirichlet 形式对叶子状态 $\Omega\setminus\Omega_\emptyset$ 取的 **Schur 补**，也就是 $P$ 在 $\Omega_\emptyset$ 上的**迹链**（只在链位于 $\Omega_\emptyset$ 时记录它）的 Dirichlet 形式。则
$$\sup_f\frac{\pi_\emptyset\sum_yw_yd_y^2}{\mathcal E(f,f)}\ =\ \pi_\emptyset\,\big\|\Gamma^{\mathrm{tr}}\big\|,\qquad \Gamma^{\mathrm{tr}}_{yy'}=\sqrt{w_yw_{y'}}\,\big\langle\varphi_y,\ Z_{\mathrm{tr}}\varphi_{y'}\big\rangle,$$
其中 Green 函数用迹链的 Dirichlet 形式来定义。

**证明**：中心部分只依赖 $F=f|_{\Omega_\emptyset}$。对固定的 $F$，在所有延拓中取 $\mathcal E(f)$ 的最小值，按定义就是 Schur 补 $\mathcal E_{\mathrm{tr}}(F)$，这是电网络化简的标准事实。剩下的就是引理 2′ 的同一套论证。$\blacksquare$

**数值核对**（$\lambda=4$，单位 $n(1+\lambda)$）：whiskers 0.3230 / 0.3230，twin $s=8$ 0.4995 / 0.4995，blocker $u=8$ 0.6684 / 0.6684，blocker $u=32$ 0.8163 / 0.8163，twin+blocker 1.4515 / 1.4515。左右**完全一致**。

### 13.4 三种几何的关系（按半正定序）

$$\pi_\emptyset\,\mathcal E_{P_\emptyset}\ \preceq\ \mathcal E_{\mathrm{tr}},\qquad\text{从而}\quad \pi_\emptyset\|\Gamma^{\mathrm{tr}}\|\ \le\ \|\Gamma^Z\|\ (\text{引理 2′}).$$
- **引理 2′** 用的是被杀死（限制）的链 $P_\emptyset$，丢掉了经过叶子的那些转移。
- **坍缩链** $Q$ 只覆盖纤维常数函数，给出的是真实值的下界。
- **迹链**是精确的：除了 $P_\emptyset$ 本身的转移，它还包含"**插入 $y$ → 在 $\Omega_y$ 里游走约 $n(1+\lambda)$ 步 → 删掉 $y$**"这样的长跳跃。正是这些**经过叶子的远足**让中心的混合变快，从而抵消了孪生和阻挡带来的损失。

### 13.5 对下一步的意义

问题被精确地化成了：
$$\textbf{目标：}\quad \pi_\emptyset\,\|\Gamma^{\mathrm{tr}}\|\ \le\ \mathrm{poly}(\bar\lambda)\,\big(n(1+\lambda)+\tau\big).$$
数值上它在所有实例中成立（例如 $\lambda=64$ 时最大约 $6.7\,n(1+\lambda)\approx0.1\lambda\,n(1+\lambda)$）。

**证明思路**：把迹链的 Dirichlet 形式拆开，
$$\mathcal E_{\mathrm{tr}}\ \succeq\ \pi_\emptyset\mathcal E_{P_\emptyset}+\sum_y\lambda\pi(A_y)\cdot\frac{1}{n(1+\lambda)}\cdot\mathcal E_{K_y},$$
其中 $K_y$ 是"在 $\Omega_y$ 中用 $P_y$ 跑一段几何分布时间（均值 $n(1+\lambda)$）后返回"的核。它的谱满足：对 $P_y$ 的慢模，$\mathcal E_{K_y}\approx n(1+\lambda)\,\mathcal E_{P_y}$；对快模，$\mathcal E_{K_y}\approx$ 方差（相当于完全重采样）。
- 这一步把"叶子能量"精确地放回了几何里，正是 §12.3 诊断出缺失的那部分。
- 接下来，在这个更丰富的 Dirichlet 形式下重做 Green 函数估计。可以先在孪生 + 阻挡例子上手算，看远足项如何抵消 $W/\pi_\emptyset$ 的损失，再推广。

---

## 14. 统一刻画：整个 $K$ 就是一个迹链 Green 范数（2026-11-22）

代码：[`code/chordal_trace_full_K.py`](../code/chordal_trace_full_K.py)

**观察**：$\mathrm{Var}_w(m)$ 里的每个 $m_x=\mathbb E_{\mu_x}F$（$x\in\{\emptyset\}\cup R$）都只依赖 $F=f|_{\Omega_\emptyset}$。所以命题 T 的论证（对 $F$ 的所有延拓取最小能量，得到 Schur 补）**对整个 $K$ 都适用**，不只是中心部分。

**命题 T′** **[已证 + 数值核对]**：
$$K=\sup_f\frac{\mathrm{Var}_w(m)}{\mathcal E(f,f)}=\sup_F\frac{\sum_xw_x\big(\langle F,a_x\rangle-\langle F,\bar a\rangle\big)^2}{\mathcal E_{\mathrm{tr}}(F,F)},\qquad a_x=\mu_x,\ \ \bar a=\sum_xw_xa_x,$$
也就是迹链几何下以 $\{\sqrt{w_x}(a_x-\bar a)\}_{x\in\{\emptyset\}\cup R}$ 为列的 Green 矩阵的算子范数。

**数值核对**（单位 $n(1+\lambda)$；左边用迹链，右边按改正后的恒等式在全空间上计算）：

| 实例 | $\lambda=4$ | $\lambda=16$ |
|---|---|---|
| whiskers $r=4,m=3$ | 0.3639 / 0.3639 | 0.3442 / 0.3442 |
| blocker $u=32$ | 1.4912 / 1.4912 | 5.2834 / 5.2834 |
| twin+blocker $s=16,u=32$ | 2.7550 / 2.7550 | 4.4787 / 4.4787 |

完全一致。这个核对同时暴露出 §7.3 恒等式漏掉的因子 $W$，已更正。

**意义**：
- **引理 G 和引理 L 合并成一个不等式**，不再需要分开处理中心部分和叶子部分：
  $$\boxed{\ K=\big\|\Gamma^{\mathrm{tr}}_{\text{full}}\big\|\ \le\ \mathrm{poly}(\bar\lambda)\,\big(n(1+\lambda)+\tau\big)\ }$$
  ⇒（命题 1）$(\star)$ ⇒（§4.6）任意弦图上 $n^{O(\log\bar\lambda)}$。
- 数值上，所有实例中 $K\le 5.3\,n(1+\lambda)$（$\lambda\le16$）。
- 证明的对象从此是**一条具体的链**，即 Glauber 在 $\Omega_\emptyset$ 上的迹链；要证的是**一个具体的量**，即一族固定向量在这条链下的 Green 矩阵范数。

---

## 15. 远足分解与"孪生 + 阻挡"的显式计算（2026-11-22）

代码：[`code/chordal_trace_excursion.py`](../code/chordal_trace_excursion.py)

### 15.1 结论速览

1. **引理 X（远足分解）[已证 + 数值核对]**：迹链的 Dirichlet 形式可以按 $y\in R$ **精确**拆开，每个叶子贡献一个显式的"截断"形式。证明只用到 $R$ 是团。
2. **孪生 + 阻挡族上 $K\le 6(1+\lambda)\,n(1+\lambda)$，对所有 $s,u$ 一致成立 [已证]。** 被杀死的链在同一族上给出的 $K$ 约为 $\lambda\min(u,s\lambda)\,n(1+\lambda)$，没有上界。
3. **抵消机制看清了**：被杀死链的极值函数恰好是 $F=\mathbf 1_v$。它只在 $A_y$ **内部**变化，也就是 D 块空时阻挡点 $v$ 占没占，这正是空位竞争的那个坐标。中心块里这个坐标的电导只有 $\lambda(1+\lambda)c/Z$，但 $s$ 个叶子各提供一份放大 $\lambda$ 倍、再乘截断因子 $\frac1{2+\lambda}$ 的副本。两者相乘，$s$ 正好抵消。
4. **$\lambda$ 因子是真实的**：$s=1$、$u$ 很大时数值上 $K\approx0.4\lambda\,n(1+\lambda)$，和 §8.4 一致。所以这里的界已经紧到常数。

记号：$c=\frac1{n(1+\lambda)}$ 是单个点的删除速率。本节所有能量都按未归一化的电导 $\pi(I)P(I,I')$ 计。

### 15.2 引理 X：远足分解 **[已证]**

$$\mathcal E_{\mathrm{tr}}(F)=\mathcal E_\emptyset(F)+\lambda\sum_{y\in R}\mathcal M^{(y)}_c\big(F|_{A_y}\big),\qquad \mathcal M^{(y)}_c(F)=\inf_g\Big\{c\sum_{J\in A_y}\pi(J)\big(F(J)-g(J)\big)^2+\mathcal E_{A_y}(g)\Big\}.$$

- $\mathcal E_\emptyset$ 是中心块内部边的能量，即被杀死的链。
- $\mathcal E_{A_y}$ 是 $A_y$ 的**诱导子图**上的能量，边权和中心块里一样。
- 写成矩阵形式：$\mathcal M^{(y)}_c=c\Pi_y(c\Pi_y+L_y)^{-1}L_y$。对 $L_y$ 关于 $\pi|_{A_y}$ 的特征值 $\theta$，权重是 $\varphi_c(\theta)=\dfrac{c\theta}{c+\theta}\approx\min(c,\theta)$：慢模按 $\theta$ 计，快模被截断在 $c$。

**证明**：
1. $R$ 是团，所以一个独立集最多含 $R$ 的一个点。单点翻转不可能把 $y$ 换成 $y'$，因此**不同叶子块之间没有边**，叶子部分的 Laplacian 按 $y$ 分块对角。于是 Schur 补等于中心块加上各个 $y$ 的 Schur 补之和。
2. 叶子块 $\Omega_y=\{J\cup\{y\}:J\in A_y\}$ 和 $A_y$ 一一对应，权重是 $\pi(J\cup\{y\})=\lambda\pi(J)$。
3. 叶子内部的转移 $J\cup\{y\}\to J'\cup\{y\}$ 翻转的是某个 $x\ne y$。如果 $x$ 被插入，那么 $x\in J'\in A_y$，所以 $x\notin N(y)$，转移概率和 $J\to J'$ 相同。删除的情形是平凡的。所以叶子内部的电导恰好是 $\lambda\times$（$A_y$ 诱导子图的电导）。
4. 从 $J\cup\{y\}$ 出发，除了删掉 $y$（辐条，电导 $\lambda\pi(J)c$），其它转移都留在 $\Omega_y$ 里：$N(y)$ 中的点不能插入，删掉 $J$ 中的点仍在 $A_y$ 里。
5. 所以第 $y$ 块是 $\lambda(c\Pi_y+L_y)$，和中心的耦合是 $-\lambda c\Pi_y$。它的 Schur 补是 $\lambda\big[c\Pi_y-c\Pi_y(c\Pi_y+L_y)^{-1}c\Pi_y\big]=\lambda\,c\Pi_y(c\Pi_y+L_y)^{-1}L_y$，这就是 $\lambda\mathcal M^{(y)}_c$。$\blacksquare$

**数值核对**：4 个随机弦图 × $\lambda\in\{0.5,4,16\}$，以及两个孪生 + 阻挡实例，相对误差都 $\le7\times10^{-15}$。

**与 §13.5 的关系**：§13.5 只猜了一个下界"$\succeq$"，这里是**等式**。坍缩链 $Q$ 的边权因子 $1+\lambda N(J,J')$ 相当于把每个叶子的贡献都按慢模计，即 $\varphi_c(\theta)\approx\theta$。§13.1 里坍缩链低估真实值，原因正是快模被截断在 $c$。

### 15.3 合并后的孪生 + 阻挡

图：$R=S\cup\{r\}$，$S$ 是 $s$-团；$D$ 是 $u$-团，与 $S$ 全连；$v$ 与 $D$ 全连；$r$ 挂一个叶子 $q$。$G-R$ 有两个分量：$\{D,v\}$ 和 $\{q\}$。

按 $S$ 内、$D$ 内的置换对称性合并。取对称化不改变 $\mathrm{Var}_w(m)$，也不增加 $\mathcal E_{\mathrm{tr}}$，所以极值 $F$ 可以取为对称的。中心块只剩 6 个状态 $(b,q)$：
- $b\in\{e,v,d\}$：$e$ = D 块和 $v$ 全空，$v$ = $v$ 被占，$d$ = $D$ 中某点被占（合并 $u$ 个状态）；
- $\mu_0$ 是乘积测度：$b$ 的权重 $(1,\lambda,u\lambda)/Z_b$，其中 $Z_b=1+\lambda+u\lambda$；$q$ 的权重 $(1,\lambda)/(1+\lambda)$。

两类叶子：
- $A_y=\{b\in\{e,v\}\}$（$y\in S$，共 $s$ 个相同的事件），总权重 $W_S=s\lambda(1+\lambda)^2/Z$；
- $A_r=\{q=0\}$，权重 $w_r=\lambda Z_b/Z$；
- 中心 $w_0=\pi_\emptyset=Z_b(1+\lambda)/Z$，其中 $Z=Z_b(1+2\lambda)+s\lambda(1+\lambda)^2$。

**每个事件只限制一个分量。** 这是后面的方差分解能干净进行的原因。

### 15.4 被杀死的链为什么失效

取 $F=\mathbf 1_v$：
- $m_0=m_r=\lambda/Z_b$，$m_S=\lambda/(1+\lambda)$，所以 $\mathrm{Var}_w(m)=W_S(1-W_S)\Big(\dfrac{u\lambda^2}{(1+\lambda)Z_b}\Big)^2$；
- 中心块里只有 $e$–$v$ 边在付钱，电导 $\lambda(1+\lambda)c/Z$。

所以被杀死链下
$$K_{\text{kill}}\ \ge\ \frac{s\,u^2\lambda^4(1+2\lambda)}{Z\,Z_b\,(1+\lambda)}\cdot n(1+\lambda)\ \approx\ \lambda\min\!\big(u,\ s\lambda\big)\cdot n(1+\lambda).$$
数值上这正是 $K_{\text{kill}}$ 的精确值：$s=u=10^4$、$\lambda=4$ 时闭式 $8470.8$，计算值 $8470.8$。

**直观**：$\mathbf 1_v$ 在 $\mu_0$ 下几乎不动（$A_S$ 的概率只有 $p_S=(1+\lambda)/Z_b$），所以中心为它付的能量很小。但 $m_S$ 是对 $A_S$ 取的条件期望，把它放大了 $1/p_S$ 倍，权重 $W_S$ 又接近 1。

### 15.5 迹链怎样补回来

由引理 X，$s$ 个 $S$-叶子各贡献 $\lambda\mathcal M_c$。$\mathbf 1_v$ 只依赖 $b$，在 $A_S$ 上是两状态链 $e\rightleftarrows v$ 的特征向量，特征值 $\theta=c(1+\lambda)$。所以
$$\lambda s\,\mathcal M_c(\mathbf 1_v)=\lambda s\cdot\frac{c}{c+\theta}\cdot\frac{\lambda(1+\lambda)c}{Z}=\frac{s\lambda}{2+\lambda}\cdot\frac{\lambda(1+\lambda)c}{Z}.$$
也就是说，**$e$–$v$ 方向的电导被放大了 $1+\dfrac{s\lambda}{2+\lambda}$ 倍**：
- 因子 $s$：每个孪生点都提供一份 $A_S$ 的副本；
- 因子 $\lambda$：叶子的质量是对应中心状态的 $\lambda$ 倍；
- 因子 $\frac1{2+\lambda}$：截断。$v$ 在叶子里的翻转速率约为 $c(1+\lambda)$，而远足的平均长度只有 $1/c$。

损失是 $W_S/(\ldots)\propto s$，补偿是 $\propto s$，正好抵消。剩下一个 $\frac{2+\lambda}{\cdots}$，就是最终界里的 $(1+\lambda)$。

### 15.6 定理：孪生 + 阻挡族上 $K\le6(1+\lambda)\,n(1+\lambda)$ **[已证]**

对 $\mu_0$（乘积测度）做方差分解：$F(b,q)=\bar F+g(b)+h(q)+k(b,q)$，其中 $g,h$ 均值为零，$k$ 的两个边缘都为零。

**第 1 步：$m_x$ 只依赖主效应。**
- $A_S$ 只限制 $b$，$A_r$ 只限制 $q$，所以 $k$ 和"另一个分量"的主效应在条件期望里都消失：
$$m_S-m_0=\gamma:=\mathbb E[g\mid A_S],\qquad m_r-m_0=h(0).$$
- 用成对形式 $\mathrm{Var}_w(m)=\tfrac12\sum_{x,x'}w_xw_{x'}(m_x-m_{x'})^2$ 和 $(a-b)^2\le2a^2+2b^2$：
$$\mathrm{Var}_w(m)\le W_S(w_0+2w_r)\,\gamma^2+2w_r\,h(0)^2,\qquad w_0+2w_r\le3\pi_\emptyset.$$

**第 2 步：能量也只看主效应。**
- 被杀死的链是 $b$-链与 $q$-链的乘积。由凸性（对另一个坐标取平均不增加能量），$\mathcal E_\emptyset(F)\ge\pi_\emptyset\big[\mathcal E_b(g)+\mathcal E_q(h)\big]$。
- 在 $A_S=\{e,v\}\times\{0,1\}$ 上，"对 $q$ 取平均"是一个与 $L$ 交换的正交投影，所以 $\mathcal M_c(F)\ge\mathcal M_c(g)$。
- 丢掉 $r$-叶子的贡献：
$$\mathcal E_{\mathrm{tr}}(F)\ \ge\ \underbrace{\pi_\emptyset\mathcal E_b(g)+\lambda s\,\mathcal M_c(g)}_{\text{管 }\gamma}+\underbrace{\pi_\emptyset\mathcal E_q(h)}_{\text{管 }h(0)}.$$

**第 3 步：逐项比较。** 记 $\Delta_{ed}=g_e-g_d$，$\Delta_{ev}=g_v-g_e$，$1-p_S=u\lambda/Z_b$。
- $\gamma=(1-p_S)\big[\Delta_{ed}+\tfrac{\lambda}{1+\lambda}\Delta_{ev}\big]$，所以 $\gamma^2\le2(1-p_S)^2\big[\Delta_{ed}^2+(\tfrac{\lambda}{1+\lambda})^2\Delta_{ev}^2\big]$。
- 能量：$\pi_\emptyset\mathcal E_b(g)=\frac{(1+\lambda)c}{Z}\big[u\lambda\Delta_{ed}^2+\lambda\Delta_{ev}^2\big]$，$\lambda s\mathcal M_c(g)=\frac{s\lambda}{2+\lambda}\cdot\frac{(1+\lambda)\lambda c}{Z}\Delta_{ev}^2$。

| 方向 | 谁付钱 | 比值（× $n(1+\lambda)$） | 界 |
|---|---|---|---|
| $\Delta_{ed}$（跨过 $\partial A_S$） | 中心的 $e$–$d$ 边 | $\dfrac{6\,s\lambda(1+\lambda)^2\cdot u\lambda}{Z\,Z_b}$ | $\le6$ |
| $\Delta_{ev}$（$A_S$ 内部，空位竞争） | $s$ 个叶子的远足 | $\dfrac{6\,s\,u^2\lambda^4}{Z\,Z_b\,(1+s\lambda/(2+\lambda))}$ | $\le6(1+\lambda)$ |
| $h(0)$（$q$ 方向） | 中心的 $q$ 翻转 | $\dfrac{2\lambda^2}{(1+\lambda)^2}$ | $\le2$ |

- 第一行用 $s\lambda(1+\lambda)^2\le Z$、$u\lambda\le Z_b$。
- 第二行用 $Z\ge Z_b(1+\lambda)$、$Z_b\ge u\lambda$、$\lambda(2+\lambda)\le(1+\lambda)^2$。**这一行如果没有远足项（分母里的 $s\lambda/(2+\lambda)$），就只能得到 $\approx\lambda\min(u,s\lambda)$，正是 §15.4 的失效。**

二次型之比不超过各项比值的最大值，所以 $K\le6(1+\lambda)\,n(1+\lambda)$。再由命题 1，
$$t_{\mathrm{rel}}\le4\tau+\big(6+12(1+\lambda)\big)\,n(1+\lambda)\quad\text{对所有 }s,u\text{ 一致成立。}\ \blacksquare$$

### 15.7 数值（单位 $n(1+\lambda)$；合并计算与全空间计算在小实例上一致到 4 位）

| $\lambda$ | $(s,u)$ | 迹链 $K$ | 杀死链 $K$ | $K/(1+\lambda)$ |
|---|---|---|---|---|
| 4 | $(10^4,10^4)$ | 2.81 | 8471 | 0.56 |
| 4 | $(10^5,10^5)$ | 2.81 | 84706 | 0.56 |
| 64 | $(10^2,10^4)$ | 48.8 | 4747 | 0.75 |
| 64 | $(10^5,10^5)$ | 4.37 | 186702 | 0.07 |
| 1024 | $(1,10^4)$ | 405.8 | 973 | 0.40 |
| 1024 | $(10^5,10^5)$ | 4.55 | 199125 | 0.004 |

- 所有实例都满足 $K/(1+\lambda)\le0.93$，定理的常数 6 有余量。
- $\lambda$ 增长只出现在 $u\gg s\lambda$ 的"纯阻挡"区域，这是真实的减速（§8.4）。孪生点多时 $K$ 反而趋于常数。

### 15.8 对第 2 步（一般引理）的启示

这个例子的证明结构可以直接推广，但有两处需要补：

1. **乘积结构是一般的。** $G-R$ 的各个分量在 $\Omega_\emptyset$ 上独立（引理 E），被杀死的链是乘积链，每个 $A_y$ 都是乘积事件 $\prod_CA_y^C$。所以总可以按分量做方差分解。
   - **缺口 1**：一般情况下一个 $y$ 会同时限制多个分量，交互项 $k$ 不再从 $m_y$ 里消失。需要证明交互项可以用"每个分量各付一份"的方式控制，或者损失只有 $\mathrm{poly}(\lambda)$。
2. **每个分量内部的模板**：把 $\mathbb E[g\mid A_y]$ 拆成"跨过 $\partial A_y$"和"$A_y$ 内部"两部分。
   - 跨边界的部分由中心付钱，比值自动 $\le O(1)$，因为边界电导和 $A_y^c$ 的质量同阶；
   - 内部的部分由远足付钱，放大因子是 $\lambda\sum_{y':A_{y'}\supseteq\cdot}\varphi_c(\theta)/\theta$。
   - **缺口 2**：截断因子 $\varphi_c(\theta)/\theta=\frac{c}{c+\theta}$。这里 $A_y$ 内部只有一个慢坐标，$\theta=c(1+\lambda)$，只损失 $1+\lambda$。一般情况下 $A_y$ 内部可能有很多快模（$\theta\gg c(1+\lambda)$），需要证明决定 $\mathbb E[F\mid A_y]$ 的那部分 $F$ 不会集中在快模上，或者快模的方差另有人付。

⇒ **第 2 步的具体目标**：
- 先在"每个 $y$ 只限制一个分量"的弦图子类上把 15.6 的论证写成一般定理。这个子类已经包含孪生、阻挡、whiskers 这些族。
- 再处理缺口 1（交互项）和缺口 2（快模截断）。

---

## 16. 第二步：(⋆) 被"分裂竞争"族推翻；修正为加法形式 (⋆′)（2026-11-22）

代码：[`code/chordal_split_race.py`](../code/chordal_split_race.py)

### 16.1 结论速览

1. **(⋆) 是错的，而且对每一个平衡团都错。** 反例是一个区间图：在一个 $2s$-团的两半上各挂一个 $u$-团。取 $u\approx\sqrt{s/\lambda}$，则
   - $t_{\mathrm{rel}}\approx\tfrac12\sqrt{s\lambda}\cdot n(1+\lambda)$；
   - 对每个平衡团 $R$，所有限制链都满足 $\tau=n(1+\lambda)$（精确）。

   所以 (⋆) 的常数 $C\gtrsim\sqrt{s\lambda}/4$，随 $|R|$ 无界增长。这个图在"每个 $y$ 只限制一个分量"的子类里，所以 §15.8 计划的强形式定理也是错的。
2. **这不影响多项式结论。** 损失出现在**加法项** $n(1+\lambda)$ 上，没有乘在 $\tau$ 上。把 (⋆) 换成
   $$t_{\mathrm{rel}}(G)\ \le\ C(\lambda)\,\tau+B(n,\lambda)\,n(1+\lambda),\qquad C(\lambda)=\mathrm{poly}(\lambda),\ \ B=\mathrm{poly}(n,\lambda),\tag{$\star'$}$$
   递归后仍然得到 $\mathrm{poly}(n)\cdot n^{O(\log\bar\lambda)}$（16.4）。
3. **数值支持 (⋆′)**：把一组孪生点压在分裂竞争零件上，让 $\tau$ 本身变大。此时 $t_{\mathrm{rel}}/\tau\le1.32$，并且孪生点越多越小。
4. 命题 1 本来就是 (⋆′) 的形状（$\tau$ 前面的系数是 4）。所以**新目标是 $K\le C_1(\lambda)\,\tau+B\,n(1+\lambda)$**：只需把 JSTV 那个 $|R|\lambda$ 从 $\tau$ 的系数**挪到**加法项上，不需要消掉它。

### 16.2 分裂竞争族

**图**：
- $R=S_1\cup S_2$ 是 $2s$-团；
- $D_i$ 是 $u$-团，与 $S_i$ 全连（$i=1,2$）；
- clique tree 是一条路 $(S_1\cup D_1)-(S_1\cup S_2)-(S_2\cup D_2)$，所以这是**区间图**。

**两个相**：
- 相 1 = $\{S_1$ 有点被占$\}$，此时 $D_1$ 被迫为空，$D_2$ 自由（大概率被占）；
- 相 2 对称；
- 两相的质量各约 $1/2$。中心 $\Omega_\emptyset$ 的质量 $\pi_\emptyset\approx u/(2s)$，很小。

**相 1 → 相 2 必须经过 $R$ 全空，而且 $S_2$ 要能插入还需要 $D_2$ 空。** 只有两条路线：
- **路线 a**：删掉 $S_1$ 的点时 $D_2$ 恰好是空的，概率约 $\frac1{1+u\lambda}$。之后在 $R$ 全空的竞争里，$S_2$ 以约 $1/2$ 的概率赢。
- **路线 b**：删掉 $S_1$ 的点后，$D_1$ 先抢到空位，概率约 $\frac{u}{s+u}$。这样就到了"$D_1,D_2$ 都被占"的枢纽状态，之后以约 $1/2$ 的概率走向相 2。

每单位时间离开相 1 的速率约为
$$\frac c2\Big(\frac1{1+u\lambda}+\frac{u}{s+u}\Big),\qquad c=\frac1{n(1+\lambda)}.$$
取 $u^2\lambda\approx s$ 时两项相等，这个速率约为 $c/\sqrt{s\lambda}$。两相对称，所以 $t_{\mathrm{rel}}\approx\frac{\sqrt{s\lambda}}{2c}$。

**$\tau$ 的精确值**：
- 平衡团必须包含 $S_1$ 或 $S_2$，否则 $S_1\cup S_2$ 剩下的部分会把两个 $D_i$ 连成一个大分量。
- 对这样的 $R$，每个限制链 $P_x$ 都是若干个团上 Glauber 链的乘积。
- 团 $K_k$ 上（按全图的时间尺度）的谱间隙恰好是 $c$：取 $f(\emptyset)=0$、$\sum_vf(v)=0$，就有 $Lf=-cf$；另一个特征值是 $c(1+k\lambda)$。
- 乘积链的谱间隙取最小值，所以 **$\tau=1/c=n(1+\lambda)$ 精确成立**。

**数值**（合并成 8 个状态；合并链的谱是原链谱的子集，所以合并链的 $t_{\mathrm{rel}}$ 是真实值的下界。小实例上两者完全一致）：

| $\lambda$ | $s$ | 最坏的 $u$ | $t_{\mathrm{rel}}/(n(1+\lambda))$ | $\tfrac12\sqrt{s\lambda}$ | $M(\star)$ |
|---|---|---|---|---|---|
| 1 | $10^4$ | 99 | 50.75 | 50 | 25.4 |
| 1 | $10^6$ | 999 | 500.75 | 500 | 250.4 |
| 4 | $10^6$ | 500 | 1000.75 | 1000 | 500.4 |
| 16 | $10^6$ | 250 | 2000.75 | 2000 | 1000.4 |

和闭式 $\tfrac12\sqrt{s\lambda}$ 吻合到 0.2%。**[数值 + 推导；下界可以用相指示函数作测试函数写成严格证明]**

**为什么之前没发现**：
- §5 的随机图只到 $n\le13$。这个效应在 $s\approx10$ 时只有约 1.5 倍，和随机图里的 $M\le2.4$ 分不开（见小实例：$s=8,u=3$ 时 $M=1.49$）。
- §7–§15 的结构族都只有**一组**重叶子：孪生、阻挡、孪生 + 阻挡（$r$-叶子很轻）。这里是**两个分量各有一组重叶子**，中心被两边挤得很小。§15.6 的证明里正好用到 $w_0+2w_r\le3\pi_\emptyset$，在这里不成立。

### 16.3 这和引理 C、引理 A 不矛盾

- **引理 A**：投影链的谱间隙确实是 $c$。但投影链把中心看成一个点，看不到"中心里 $D_2$ 空不空"这个内部状态。
- **引理 C**：$\pi_\emptyset\ge1/(1+|R|\lambda)$ 仍然成立，这里 $\pi_\emptyset\approx u/(2s)$。
- 慢的根源是**中心里有一个依赖方向的内部瓶颈**：从相 1 进入中心后，必须先等 $D_2$ 空出来。

这正是 $K$ 里"中心部分 × $W/\pi_\emptyset$"那种损失，只不过这次迹链的远足救不回来：相 1 的叶子只能放大 $A_{S_1}=\{D_1\text{ 空}\}$ 内部的几何，而瓶颈在 $D_2$ 方向。

### 16.4 加法形式 (⋆′) 仍然给出多项式 **[已证，给定 (⋆′)]**

记 $t(G)=t_{\mathrm{rel}}(G)/(n(1+\lambda))$。§4.6 第 2 步的乘积链公式给出 $\tau/(n(1+\lambda))=\max_i t(G_i)$，其中 $|G_i|\le n/2$。所以 (⋆′) 变成
$$t(G)\le C\max_it(G_i)+B.$$
迭代 $\log_2n$ 层：
$$t(G)\le B\sum_{k=0}^{\log_2 n}C^k\le 2B\,C^{\log_2n}=2B\,n^{\log_2C}.$$
所以 $t_{\mathrm{rel}}\le 2B\,n^{1+\log_2C}(1+\lambda)$。$C=\mathrm{poly}(\lambda)$、$B=\mathrm{poly}(n,\lambda)$ 时，这是 $\mathrm{poly}(n)\cdot n^{O(\log\bar\lambda)}$。$\blacksquare$

**对照**：
- 分裂竞争族满足 (⋆′)，取 $C=1$、$B\approx\sqrt{|R|\lambda}$ 即可。
- JSTV 给出的是 $C=3|R|\lambda+1$、$B=3$：损失乘在 $\tau$ 上，递归时会复合，得到 $n^{O(\log(|R|\lambda))}$。
- **要做的只是把 $|R|$ 从乘法系数挪到加法项。**

### 16.5 换成 $K$ 的语言：新目标

由命题 1（$t_{\mathrm{rel}}\le4\tau+6n(1+\lambda)+2K$），只需证
$$\boxed{\ K\ \le\ C_1(\lambda)\,\tau\ +\ B(n,\lambda)\,n(1+\lambda)\ }\qquad C_1=\mathrm{poly}(\lambda),\ B=\mathrm{poly}(n,\lambda).$$
- 加法项允许有很大的余量，比如 $B=|R|\lambda$，甚至 $B=n^{10}$。
- 难点只剩下：**中心链 $P_\emptyset$ 的慢模**（时间尺度 $\gg 1/c$）在 $\mathrm{Var}_w(m)$ 里的放大倍数不能和 $|R|$ 有关。
- 快模（时间尺度 $\lesssim1/c$）随便怎么放大都没关系，因为平凡界只给出 $|R|\lambda\cdot\frac1c$，可以算进 $B$。

**为什么这可能容易得多**：把 $F$ 按 $P_\emptyset$ 的谱分成慢部分和快部分。
- 快部分的贡献直接用 Gram/平凡界，代价 $|R|\lambda/c$，算进 $B$。
- 慢部分要证的是"$h_y-1$ 在慢子空间里的投影"不会被 $|R|$ 个 $y$ 叠加放大。

孪生压在零件上的实验（16.6）正是在测慢部分，结果是不放大。

### 16.6 检验：$\tau$ 的系数会不会随 $|R|$ 增长

**构造**：把 $s$ 个孪生点（即 $R$）全连到一个分裂竞争零件（参数 $s'=10^4$，取最坏的 $u$）的 $S_1'$ 上。
- $G-R$ 只有一个分量，就是整个零件，所以 $\tau=t_{\mathrm{rel}}(\text{零件})\gg n(1+\lambda)$；
- $A_y=\{S_1'\text{ 空}\}$ 和零件的慢模（相变量）强相关。这正是"慢模被 $|R|$ 个 $y$ 放大"最可能发生的情形。

| $\lambda$ | $\tau/(n(1+\lambda))$ | $s=1$ | $s=10$ | $s=10^2$ | $s=10^3$ | $s=10^4$ | $s=10^8$ |
|---|---|---|---|---|---|---|---|
| 1 | 50.75 | 1.21 | 1.27 | 0.82 | 0.19 | 0.04 | 0.02 |
| 4 | 100.75 | 1.32 | 1.19 | 0.58 | 0.10 | 0.02 | 0.01 |
| 16 | 200.75 | 1.32 | 1.06 | 0.37 | 0.05 | 0.01 | 0.005 |

（表中是 $t_{\mathrm{rel}}/\tau$。小实例上合并链与全空间精确值一致。）

- **$t_{\mathrm{rel}}/\tau\le1.32$，并且随 $s$ 增大而减小。** 孪生点多了以后，叶子把零件"钉"在相 2，慢模直接消失了。
- 所以在这个最危险的构造里，$\tau$ 的系数没有任何 $|R|$ 依赖。

### 16.7 对前面各节的影响

- **§4.6 / §6 / §12 / §15 中"(⋆) 仍成立"的说法作废**，改为 (⋆′)。§4.6 的递归改用 16.4 的版本。
- **§15 的结论都还对**：孪生 + 阻挡族上 $\tau=n(1+\lambda)$，所以 $K\le6(1+\lambda)n(1+\lambda)$ 同时满足两种形式。引理 X、命题 T′ 不受影响。
- **§15.8 的计划要改**：强形式定理在子类上就是错的，不用再证。

### 16.8 下一步

1. **证 (⋆′) 的 $K$ 版本**（16.5）。按 $P_\emptyset$ 的谱分成慢、快两部分。快部分用平凡界算进 $B$；慢部分是新问题，需要证明慢模的放大和 $|R|$ 无关。引理 E（块对角）和引理 X 应该都用得上。
2. **检查损失会不会沿 clique tree 复合**：把分裂竞争零件嵌套进分裂竞争零件，看 $t_{\mathrm{rel}}$ 是像 $\sum_k\sqrt{s_k\lambda}$（加法，多项式）还是像 $\prod_k\sqrt{s_k\lambda}$（乘法）增长。
   - 如果是乘法，深度取 $\log n$ 就会得到 $n^{\Theta(\log n)}$ 量级的下界。这会**正面回答 Marin 的开放问题 4**，主线就要从"证上界"转向"构造下界"。
   - 机制上我倾向于加法：外层路线 a 只要求"删除那一刻 $D_2$ 处在合适的状态"，内层的慢只会以 $+\tau$ 的形式出现。但这必须算出来。

---

## 17. 损失会不会沿 clique tree 复合：膨胀归约与减速指数（2026-11-22）

代码：[`code/chordal_tree_blowup.py`](../code/chordal_tree_blowup.py)、[`code/chordal_weighted_exponent.py`](../code/chordal_weighted_exponent.py)

### 17.1 结论速览

1. **膨胀引理 [已证 + 数值核对]**：大孪生类的弦图，等价于加权弦图 $H$ 上活跃度为 $\Lambda_k=m_k\lambda$ 的 hardcore 动力学。分裂竞争就是 $H=P_4$ 的情形。
2. **均匀活跃度的树不复合**：完全二叉树做到深度 4，$t_{\mathrm{rel}}\approx1.3$–$2.1\,n(1+\lambda)$，和 $\Lambda$ 无关。
3. **减速指数**：令 $t_{\mathrm{rel}}\,c\sim\Lambda^{\alpha}$。在所有 $k\le9$ 点树上，最坏情形是**星形**，$\alpha\to1-\frac1{k-1}<1$。如果对任意 $H$ 都有 $\alpha\le1$，那么大活跃度最多贡献一个因子 $\Lambda_{\max}\le n\lambda$，**不会产生超多项式**。
4. **还没定**：分层的"星套星"（3 叉、深度 2）的局部斜率在 $10^5\to10^6$ 段是 0.974，还在上升，看不出会不会越过 1。

### 17.2 膨胀引理 **[已证]**

**构造**：设 $H$ 是图，$m_k\ge1$。把点 $k$ 换成 $m_k$-团 $V_k$，把 $H$ 的每条边 $kl$ 换成 $V_k,V_l$ 之间的完全二部图。

**结论**：
- $H$ 是弦图 ⇒ 膨胀图是弦图。理由：膨胀不产生新的无弦圈，任何无弦圈投影到 $H$ 上仍然是无弦圈。
- 膨胀图上的独立集至多从每个 $V_k$ 取一个点，并且被占的类在 $H$ 中构成独立集。
- 按各个 $V_k$ 内的置换对称性合并，得到 $H$ 上的链：类 $k$ 的插入速率是 $m_k\cdot\frac{\lambda}{n(1+\lambda)}=\Lambda_kc$，删除速率是 $c$。这就是活跃度为 $\Lambda_k$ 的 hardcore 动力学，时间尺度为 $c$。
- 合并链的谱是原链谱的子集，所以合并链的 $t_{\mathrm{rel}}$ 是真实值的**下界**。

**数值核对**：在 3 个真实膨胀图上，合并值与全空间精确值一致到 $10^{-6}$。对完全二叉树还按左右子树互换的自同构再合并一次（深度 4 时有 27652 个轨道），结果与 $H$ 上的完整链一致。

**意义**：
- 孪生类的大小 $m_k$ 可以随 $n$ 增长，所以有效活跃度 $\Lambda_k$ 可以达到 $n\lambda$。"固定 $\lambda$"并不意味着有效活跃度有界。
- **复合的问题因此变成**：$H$ 上的 hardcore，活跃度不均匀、最大到 $n\lambda$，时间以 $c$ 为单位，$t_{\mathrm{rel}}\,c$ 能不能超多项式？

### 17.3 均匀活跃度的完全二叉树

| $\Lambda$ | $h=1$ | $h=2$ | $h=3$ | $h=4$ |
|---|---|---|---|---|
| 4 | 1.36 | 1.54 | 2.17 | 2.14 |
| 64 | 1.49 | 1.36 | 1.74 | 1.41 |
| 1024 | 1.50 | 1.34 | 1.54 | 1.34 |

（单位 $n(1+\lambda)$，叶子与内部点活跃度相同。）

- 没有随深度或 $\Lambda$ 增长的趋势，和 Martinelli–Sinclair–Weitz 的"完全树、自由边界、$O(n)$"一致。
- 用"叶子全占"作边界（叶子活跃度 $10^6$），结果只相当于把深度减 2。
- ⇒ 慢化需要**不均匀**的活跃度。分裂竞争正是这样：$\Lambda_D\approx\sqrt{\Lambda_S}$。

### 17.4 减速指数 $\alpha$ 的搜索

**设定**：取 $\Lambda_k=\Lambda^{e_k}$，$e_k\in\{0,\frac18,\dots,1\}$，用随机重启加坐标搜索最大化 $\alpha=\frac{d\log(t_{\mathrm{rel}}c)}{d\log\Lambda}$。

**数值精度的教训**：$\Lambda=10^9$ 时谱间隙约 $10^{-9}$，矩阵范数约 $10^{10}$，双精度的误差已经和间隙同阶。第一轮在 $k=9$ 得到的 $\alpha=1.04$ 是假象。下表只用 $\Lambda\in[10^3,10^6]$ 的数据。

| $k$ | 搜索最优（结构） | 局部斜率 $10^3\!\to\!10^4,\ 10^4\!\to\!10^5,\ 10^5\!\to\!10^6$ | 星形解析值 $1-\frac1{k-1}$ |
|---|---|---|---|
| 3 | 星形（$P_3$） | 0.466, 0.488, 0.496 | 0.500 |
| 5 | 星形 | 0.685, 0.711, 0.727 | 0.750 |
| 7 | 星形 | 0.759, 0.768, 0.770 | 0.833 |
| 9 | 星形 | 0.839, 0.841, 0.840 | 0.875 |
| 10 | 非星形（搜索不完全） | 0.800, 0.851, 0.881 | 0.889 |

**星形的机制**：中心活跃度 $\Lambda$，$k-1$ 个叶子各为 $\Lambda^{1/(k-1)}$，于是"中心占"和"叶子占"两相的质量相当。
- 从"中心占"出发：删掉中心后，叶子必须抢在中心重新插入之前占位，概率约 $\Lambda^{1/(k-1)-1}$。
- 反方向：要求所有叶子同时空，概率约 $\Lambda^{-1}$，然后中心以速率 $\Lambda$ 插入。
- 所以 $t_{\mathrm{rel}}c\approx\Lambda^{1-1/(k-1)}$。这就是 §8.4 公共阻挡点的"$\min(u,\lambda)$"，只是活跃度换成了有效活跃度。

**分层"星套星"**：

| 结构 | $k$ | 各层最优指数 | 局部斜率 |
|---|---|---|---|
| 2 叉，深度 2 | 7 | $(1/2,\,1,\,3/8)$ | 0.717, 0.747, 0.758 |
| 3 叉，深度 2 | 13 | $(1,\,1,\,1/4)$ | 0.896, 0.946, **0.974** |

3 叉、深度 2 的斜率还在上升。更大的分层结构（4、5 叉，深度 3）直接枚举太慢，这一轮没有算完。

### 17.5 解读

- **如果 $\alpha\le1$ 对所有加权弦图成立**，大活跃度的贡献最多是 $\Lambda_{\max}\le n\lambda$。结合 §16 的加法形式，这强烈支持"任意弦图上 $t_{\mathrm{rel}}=\mathrm{poly}(n)\cdot n^{O(\log\bar\lambda)}$"。它也给 (⋆′) 的加法项一个自然的候选：$B\approx|R|\lambda$。
- **如果分层结构能让 $\alpha$ 随深度增长**，比如每层加一个常数，那么深度取 $\log n$、活跃度取 $n^\varepsilon$ 时，$t_{\mathrm{rel}}$ 会达到 $n^{\Theta(\log n)}$，正面回答 Marin 的开放问题 4。
- 现有证据倾向前者：单层的最坏情形是星形，指数严格小于 1。但 3 叉星套星斜率接近 1 且仍在上升，**这是目前唯一需要排除的危险信号**。

### 17.6 下一步

1. **把星套星按对称性合并**：同一个父亲下的孩子可以互换，所以状态只需记录"各类孩子子树各有多少个"。这样状态数是多项式级，可以算到 10 叉、深度 3、$\Lambda=10^6$。
2. **解析分析星套星**：写出两层的相结构和出口速率，判断 $\alpha$ 的极限是 1、某个大于 1 的常数，还是随深度增长。
3. 如果 $\alpha\le1$ 站得住，就转回证明 (⋆′)，把 $B=|R|\lambda$ 作为加法项的目标。

---

## 18. 重大转折：分层星套星让减速指数随深度线性增长（2026-11-22）

代码：[`code/chordal_hier_star.py`](../code/chordal_hier_star.py)（按层合并）、[`code/chordal_hier_mp.py`](../code/chordal_hier_mp.py)（80 位精度）、[`code/chordal_hier_sparse.py`](../code/chordal_hier_sparse.py)

### 18.1 结论速览

1. **[数值，高精度]** 取 $H$ 为 $b$ 叉、深度 $d$ 的完全树，各层活跃度为
   - 根：$\Lambda^{b/(b+1)}$；
   - 内部层：$\Lambda$；
   - 叶子：$\Lambda^{1/(b+1)}$。

   则 $t_{\mathrm{rel}}\,c\approx K_d\,\Lambda^{\alpha_d}$，其中
   $$\alpha_d=d\cdot\frac{b-1}{b+1}.$$
   已验证的情形：$b=2$ 时 $d=1,2,3$ 分别为 $\tfrac13,\tfrac23,1$；$b=3$ 时 $d=1,2$ 分别为 $\tfrac12,1$。**指数随深度线性增长，没有饱和。** 前置常数 $K_d$ 是 $O(1)$ 量级。
2. **[推论，待严格化]** 由膨胀引理（§17.2），这对应一个弦图（完全树的孪生膨胀），$\lambda=1$ 固定。取 $b=2$、$\Lambda\approx2^d$，则 $n\approx4^d$，并且
   $$t_{\mathrm{rel}}\ \gtrsim\ n(1+\lambda)\cdot K_d\,2^{d^2/3}=n^{\Omega(\log n)}.$$
   **也就是说，弦图上的 Glauber 不是多项式混合的。**
3. **意义**：
   - **(⋆′) 也是错的。** 每层乘上的因子约为 $\Lambda^{(b-1)/(b+1)}$，其中 $\Lambda\approx m\lambda$，$m$ 是孪生类的大小，与分隔集同阶。所以 $\tau$ 前面的系数必须依赖 $|R|$。JSTV 的 $|R|\lambda$ 损失在指数意义上是**真实的**。
   - **2020 年的界可能是紧的**：分隔集大小为 $b_{\mathrm{sep}}$ 时，这里的构造给出 $n^{\Theta(\log(b_{\mathrm{sep}}\lambda))}$ 量级，正好是 Bezáková–Sun 上界 $n^{O(\log(b\lambda))}$ 的形式（注意那里分析的是 Dyer–Greenhill 链）。
   - **这会正面回答 Marin 的开放问题 4**：存在弦图，混合时间是 $n^{\Omega(\log n)}$ 量级。具体常数还需要和他的原始表述对照。
   - **我们一开始的目标（去掉 $b$、得到多项式）对 Glauber 不成立。** 主线应改为：严格证明这个下界。

### 18.2 合并方法 **[已证 + 数值核对]**

- 球对称树里，同一个父亲下的孩子子树可以互换，所以子树的状态只需记录"各类孩子子树各有几个"（类型计数）。
- 状态数是多项式级：$b=2$ 时深度 1 到 4 分别是 4、16、191、27652；$b=3$ 时深度 1 到 3 分别是 5、55、37030。
- 合并链的谱是原链谱的子集，所以合并链的 $t_{\mathrm{rel}}$ 是真实值的**下界**。下界正是这里需要的方向。
- 与不合并的完整树链对照（分支 $(3)$、$(2,2)$、$(2,3)$、$(2,2,2)$，各取两个 $\Lambda$），一致到 $10^{-8}$。

### 18.3 数值精度

- 双精度在 $\Lambda\gtrsim10^6$ 时，误差（约 $\varepsilon\|S\|$）已经和谱间隙同阶，§17 里 $\alpha=1.04$ 的假象就是这么来的。
- 本节的渐近斜率都用 mpmath 的 80 位精度计算，$\Lambda$ 到 $10^{10}$–$10^{12}$。

| $b$ | $d$ | 合并状态数 | 局部斜率（每两个数量级一段，从 $10^4$ 起） | 渐近值 | $K_d$ |
|---|---|---|---|---|---|
| 2 | 1 | 4 | 0.305, 0.327, 0.332, 0.333 | 1/3 | 0.250 |
| 2 | 2 | 16 | 0.660, 0.665, 0.666, 0.667 | 2/3 | 0.375 |
| 2 | 3 | 191 | 0.973, 0.995, 0.999 | 1 | 0.113 |
| 3 | 1 | 5 | 0.468, 0.490, 0.497, 0.499 | 1/2 | 0.167 |
| 3 | 2 | 55 | 0.893, 0.955, 0.984, 0.995 | 1 | 0.028 |

**对比 §17 的单层结论**：单层（星形）的指数确实小于 1。但"星套星"每多一层就多 $\frac{b-1}{b+1}$，§17.5 担心的情形确实发生了。

### 18.4 机制（启发式，系数还没推对）

- 每个内部点 $v$（活跃度 $\Lambda$）配上它的孩子构成一个小相变："$v$ 占"是主相，"$v$ 空、孩子占"是稀有相，概率约 $\Lambda^{-1/(b+1)}$。叶子活跃度 $\Lambda^{1/(b+1)}$ 正好让这个比例在每一层保持不变（不动点）。
- 父亲要从"空"变成"占"，需要 $b$ 个孩子**同时**处在稀有相，或者在某个孩子被短暂删除时恰好抢到空位。孩子在两相之间切换本身就慢，所以父亲的翻转时间是孩子翻转时间乘上一个 $\Lambda$ 的正幂次。
- 根的活跃度 $\Lambda^{b/(b+1)}$ 用来让最顶层两相平衡，否则 $t_{\mathrm{rel}}$ 会被较小的一相"截短"。
- 我手推的两个版本分别给出每层 $\frac{b}{b+1}$ 和 0，都和数值不符。所以**机制的定量推导还没做对**，目前只能信数值。

### 18.5 还缺什么

1. **深度 4（$b=2$，27652 个状态）已算 [数值，双精度，$\Lambda\le10^3$]**：同一段 $\Lambda$ 范围内，局部斜率依次为

   | 深度 | $10^{1.5}\to10^2$ | $10^2\to10^{2.5}$ | $10^{2.5}\to10^3$ |
   |---|---|---|---|
   | 3 | 0.740 | 0.809 | 0.861 |
   | 4 | 0.907 | 0.999 | 1.054 |

   深度 4 明显更高，仍在上升。和深度 3 的渐近值 1 对照，深度 4 的渐近值应该是 4/3。**线性增长在深度 4 仍然成立。**
2. **严格证明**：对树的层数归纳，给出递归定义的瓶颈集合，用 Cheeger 或测试函数得到 $t_{\mathrm{rel}}c\ge K^{-d}\Lambda^{\alpha d}$，其中 $\alpha>0$、$K$ 与 $d,\Lambda$ 无关。这一步需要把 18.4 的机制推对。
3. **文献**：
   - 树上 hardcore 在大活跃度或特殊边界条件下的下界（Restrepo 等人、Efthymiou–Hayes–Štefankovič–Vigoda）里，可能已经有"每层乘一个 $\lambda$ 幂次"的结果。如果有，这里的构造本质上是"用孪生把 $\lambda$ 换成 $n^\varepsilon$"，证明可以借用。
   - **Marin 开放问题的原文已核对**（arXiv:2504.18427 结论部分）：是否存在弦图 $G$，使得在 $\lambda$ 为足够大的常数时 $\tau_{G,\lambda}=\Omega(n^{\ln n})$？
     - 他的拟多项式下界（定理 30）只覆盖 tree-independence number $\ge2$ 的图。弦图恰好是 tree-independence number 为 1 的图，所以没被覆盖。
     - 他的上界（定理 3 / 19）是 $n^{O(\ln(b\tilde\lambda))}$，$b$ 是团的大小，和我们构造给出的 $n^{\Theta(\log(b\lambda))}$ 形式一致。
     - **常数**：取 $b=2$、$\Lambda=n^{1/2}$，有 $\log t\approx\frac{b-1}{4(b+1)\ln b}(\ln n)^2\approx0.12(\ln n)^2$，也就是 $n^{0.12\ln n}$。要达到原文的 $n^{\ln n}$，还需要优化每层的指数设计，或者利用"$\lambda$ 足够大"这个自由度。定性上已经是 $n^{\Omega(\log n)}$。
   - 要核对 Dyer–Greenhill 链上是否同样成立，这决定 2020 年的界是否紧。
4. **对之前各节的影响**：§4.6、§16 的上界路线对一般弦图走不通。§7–§15 的工具（命题 1、引理 X、命题 T′）本身都还对；只是 $K$ 在这类图上确实很大。

---

## 19. 参考资料

1. M. Jerrum, J.-B. Son, P. Tetali, E. Vigoda. *Elementary bounds on Poincaré and log-Sobolev constants for decomposable Markov chains.* Ann. Appl. Probab. 14(4), 2004.
2. R. A. Martin, D. Randall. *Sampling adsorbing staircase walks using a new Markov chain decomposition method.* FOCS 2000.
3. N. Madras, D. Randall. *Markov chain decomposition for convergence rate analysis.* Ann. Appl. Probab. 12(2), 2002.
4. J. R. Gilbert, D. J. Rose, A. Edenbrandt. *A separator theorem for chordal graphs.* SIAM J. Algebraic Discrete Methods 5(3):306–313, 1984.
5. Z. Chen. *Combinatorial approach for factorization of variance and entropy in spin systems.* SODA 2024. [arXiv:2307.08212](https://arxiv.org/abs/2307.08212)
6. S. Miracle, A. P. Streib, N. Streib. *Iterated decomposition of biased permutations via new bounds on the spectral gap of Markov chains.* [arXiv:1910.05184](https://arxiv.org/abs/1910.05184)
7. F. Martinelli, A. Sinclair, D. Weitz. *Glauber dynamics on trees: boundary conditions and mixing time.* Comm. Math. Phys. 2004. [arXiv:math/0307336](https://arxiv.org/abs/math/0307336)
8. R. Restrepo, D. Štefankovič, J. C. Vera, E. Vigoda, L. Yang. *Phase transition for Glauber dynamics for independent sets on regular trees.* SIAM J. Discrete Math. 28(2), 2014. [arXiv:1007.2255](https://arxiv.org/abs/1007.2255)
9. Z. Chen, K. Yang, Y. Yin, X. Zhang. [arXiv:2404.04668](https://arxiv.org/abs/2404.04668)（任意树上 $\lambda<e^2$ 时的最优谱间隙）
10. 以及 [chordal-hardcore-mixing.md](chordal-hardcore-mixing.md) 里的全部参考资料。
