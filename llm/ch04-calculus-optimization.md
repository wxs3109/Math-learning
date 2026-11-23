# 第 4 章 微积分与优化

> 标注：**[已证]** 给出了推导 · **[事实]** 标准结论 · **[直觉]** 帮助理解、不严格

## 导读

训练大模型就是：写出一个损失函数 $L(\theta)$，然后让 $\theta$ 沿着负梯度方向一步步走。要真正理解这一过程，需要回答四个问题：
- 梯度是什么，向量和矩阵的梯度怎样求；
- 怎样高效地算出几十亿个参数的梯度（反向传播、自动微分）；
- 每一步走多远、往哪走（步长、条件数、动量、Adam）；
- 为什么每次只用一小批数据也能训练（随机梯度下降）。

**与大模型的关系**

| 概念 | 在大模型里出现的地方 |
|---|---|
| 梯度、链式法则 | 反向传播；PyTorch 的 `loss.backward()` |
| 矩阵微积分 | 每一层的梯度公式：$\partial W=X^\top\partial Y$ |
| 泰勒展开 | 为什么学习率不能太大；损失尖峰（loss spike） |
| 条件数 | 为什么需要归一化层（LayerNorm）、为什么 Adam 比 SGD 好用 |
| 随机梯度下降 | 每一步只用一个 batch；batch 大小和学习率的关系 |
| Adam、warmup、学习率衰减 | 几乎所有大模型的默认优化器设置 |
| log-sum-exp | softmax 和交叉熵的数值稳定实现 |

**前置知识**

单变量导数；[第 3 章 线性代数](ch03-linear-algebra.md)（矩阵乘法、谱分解）；[第 1 章 概率论](ch01-probability.md)（期望、方差）。

## 4.1 梯度、Jacobian、链式法则 **[事实]**

**梯度**：$f:\mathbb R^n\to\mathbb R$，梯度 $\nabla f(x)=\big(\partial f/\partial x_1,\dots,\partial f/\partial x_n\big)$。它满足一阶近似
$$f(x+\Delta)\approx f(x)+\nabla f(x)^\top\Delta.$$
由 Cauchy–Schwarz（[3.1 节](ch03-linear-algebra.md#3.1)），在 $\|\Delta\|$ 固定时，$\Delta$ 取 $-\nabla f$ 方向能让 $f$ 下降得最快。这就是"梯度下降"名字的来历。

**Jacobian**：$g:\mathbb R^n\to\mathbb R^m$，Jacobian $J_g$ 是 $m\times n$ 矩阵，$(J_g)_{ij}=\partial g_i/\partial x_j$。它满足 $g(x+\Delta)\approx g(x)+J_g\Delta$。

**链式法则**：复合函数 $h=f\circ g$ 的 Jacobian 是乘积 $J_h=J_f\,J_g$。

当最终输出是标量 $L$ 时，写成梯度形式：
$$\nabla_xL=J_g^\top\,\nabla_gL.$$
意思是：知道了 $L$ 对中间变量 $g$ 的梯度，乘上 $J_g^\top$，就得到 $L$ 对输入 $x$ 的梯度。**反向传播就是从输出往输入反复使用这个公式。**

## 4.2 矩阵微积分：几条够用的恒等式 **[已证]**

**约定**：$L$ 是标量；$\partial L/\partial W$ 和 $W$ 同形状，第 $(i,j)$ 个元素是 $\partial L/\partial W_{ij}$。

| 函数 | 梯度 |
|---|---|
| $a^\top x$ | $a$ |
| $x^\top Ax$ | $(A+A^\top)x$；$A$ 对称时是 $2Ax$ |
| $\|x\|^2$ | $2x$ |
| $Y=XW$，已知 $\bar Y=\partial L/\partial Y$ | $\partial L/\partial W=X^\top\bar Y$，$\partial L/\partial X=\bar Y\,W^\top$ |
| $Y=X+b$（$b$ 广播到每一行） | $\partial L/\partial b=\sum_{\text{行}}\bar Y$ |
| $h=\phi(a)$ 逐元素 | $\bar a=\bar h\odot\phi'(a)$ |

**最重要的一条：$Y=XW$** **[已证]**。$Y_{ik}=\sum_jX_{ij}W_{jk}$，所以
$$\frac{\partial L}{\partial W_{jk}}=\sum_i\frac{\partial L}{\partial Y_{ik}}\frac{\partial Y_{ik}}{\partial W_{jk}}=\sum_iX_{ij}\bar Y_{ik}=(X^\top\bar Y)_{jk}.$$
对 $X$ 的推导同理。

**检查形状的窍门**：$X$ 是 $(n,d)$，$W$ 是 $(d,k)$，$\bar Y$ 是 $(n,k)$。要得到 $(d,k)$ 的 $\partial W$，只能是 $X^\top\bar Y$；要得到 $(n,d)$ 的 $\partial X$，只能是 $\bar YW^\top$。

$X^\top\bar Y=\sum_iX_{i,:}^\top\bar Y_{i,:}$ 是**外积之和**（[3.2 节](ch03-linear-algebra.md#3.2)），每个样本（每个位置）贡献一个秩 1 的更新。

## 4.3 Softmax、log-sum-exp 与交叉熵的梯度 **[已证]**

**softmax**：$p_i=e^{z_i}/\sum_je^{z_j}$。

**数值稳定**：$z_i$ 很大时 $e^{z_i}$ 会溢出。由于 softmax 对所有 $z_i$ 同时加减常数不变，实现时先减去 $\max_jz_j$。交叉熵里的 $\log\sum_je^{z_j}$ 写成
$$\mathrm{LSE}(z)=m+\log\sum_je^{z_j-m},\qquad m=\max_jz_j.$$

**交叉熵对 logits 的梯度**：
- 损失 $L=-\log p_y=-z_y+\mathrm{LSE}(z)$；
- $\partial\,\mathrm{LSE}/\partial z_i=e^{z_i}/\sum_je^{z_j}=p_i$；
- 所以
$$\frac{\partial L}{\partial z}=p-e_y,$$
其中 $e_y$ 是第 $y$ 个位置为 1 的独热向量，也就是"预测分布减去真实分布"。

这个梯度结构简单，而且每个分量都落在 $[-1,1]$ 里，不会爆炸。这是交叉熵和 softmax 总是一起用的原因之一。

## 4.4 泰勒展开与步长：为什么学习率不能太大 **[已证]**

**二阶泰勒展开**：
$$f(x+\Delta)\approx f(x)+\nabla f^\top\Delta+\tfrac12\Delta^\top H\Delta,$$
其中 $H$ 是 Hessian（二阶导数矩阵，对称）。

**$L$-光滑**：若 $H$ 的特征值都不超过 $L$，代入梯度下降的一步 $\Delta=-\eta\nabla f$，得
$$f(x-\eta\nabla f)\le f(x)-\eta\big(1-\tfrac{L\eta}2\big)\|\nabla f\|^2.$$
只要 $\eta<2/L$，每一步都保证下降；取 $\eta=1/L$ 时下降最多。

**在二次函数上看清楚** **[已证]**：取 $f(x)=\frac12\sum_i\lambda_ix_i^2$，梯度下降的一步是 $x_i\leftarrow(1-\eta\lambda_i)x_i$。
- 收敛的充要条件是对所有 $i$ 有 $|1-\eta\lambda_i|<1$，即 $\eta<2/\lambda_{\max}=2/L$。
- 步长一旦超过阈值，最陡的方向就会来回震荡并发散。

**条件数**：$\kappa=\lambda_{\max}/\lambda_{\min}$。
- 步长受最陡方向 $\lambda_{\max}$ 的限制，最平方向每步只缩小到原来的 $1-\lambda_{\min}/\lambda_{\max}=1-1/\kappa$。
- 所以需要约 $\kappa\log(1/\varepsilon)$ 步。**条件数越大，越慢。**
- 几何上：等高线是又细又长的椭圆，梯度几乎垂直于长轴，于是路线呈锯齿形。

**在大模型里** **[直觉]**：
- 不同参数、不同层的"曲率"可以相差几个数量级，条件数很大。
- LayerNorm、residual connection、合适的初始化，都能让损失曲面更"圆"一些。
- 训练中途出现的损失尖峰，常被解释为某个方向的曲率变大，使当前学习率超过了 $2/\lambda_{\max}$。

## 4.5 动量 **[事实 + 直觉]**

**更新规则**（heavy-ball 动量）：
$$v\leftarrow\beta v+\nabla f(x),\qquad x\leftarrow x-\eta v.$$

**直觉**：
- $v$ 是过去梯度的指数加权和（权重 $\beta^k$）。
- 在锯齿方向上，梯度正负交替，相互抵消；在一致的方向上，梯度不断累加。
- 效果相当于把有效步长放大到约 $\eta/(1-\beta)$，但只放大在"方向一致"的部分。

**理论** **[事实]**：在二次函数上，调好参数的动量法只需要约 $\sqrt\kappa\log(1/\varepsilon)$ 步，而不是 $\kappa\log(1/\varepsilon)$ 步。

## 4.6 随机梯度下降 **[已证 + 事实]**

**设定**：损失是 $N$ 个样本的平均，$L(\theta)=\frac1N\sum_i\ell_i(\theta)$。$N$ 可以是万亿个 token，每步都算全量梯度不现实。

**小批量梯度**：每步随机抽 $B$ 个样本，用
$$g_B=\frac1B\sum_{i\in\text{batch}}\nabla\ell_i$$
代替真实梯度 $g$。

**两条性质** **[已证]**：
- **无偏**：$\mathbb E[g_B]=g$。
- **方差**：$\mathrm{Var}(g_B)=\frac1B\mathrm{Var}(\nabla\ell_i)$（[1.3 节](ch01-probability.md#1.3)，独立变量的平均），所以噪声的标准差按 $1/\sqrt B$ 缩小。

**结论** **[事实 + 直觉]**：
- batch 加大 4 倍，噪声只减半，但计算量是 4 倍。所以 batch 并不是越大越好。
- 常用的经验规则是：batch 增大时，学习率按比例或按平方根增大。
- 噪声并不全是坏事：它帮助跳出尖锐的极小值，可能带来更好的泛化。

## 4.7 Adam，以及大模型训练的标准配方 **[事实]**

**Adam** 对每个参数分别维护梯度的一阶矩 $m$ 和二阶矩 $v$：
$$m\leftarrow\beta_1m+(1-\beta_1)g,\qquad v\leftarrow\beta_2v+(1-\beta_2)g^2,$$
$$\hat m=\frac m{1-\beta_1^t},\quad\hat v=\frac v{1-\beta_2^t},\qquad\theta\leftarrow\theta-\eta\frac{\hat m}{\sqrt{\hat v}+\epsilon}.$$
- **$m$** 就是动量（4.5）。
- **除以 $\sqrt{\hat v}$**：每个参数的步长按它自己的梯度大小归一化。梯度一直很大的方向（陡），步长被缩小；梯度一直很小的方向（平），步长被放大。这相当于粗略地"对角预条件"，部分抵消了条件数的影响。
- **偏差修正** $1-\beta^t$：$m,v$ 从 0 开始，前几步会偏小，除以这个因子补回来。
- **默认值**：$\beta_1=0.9$，$\beta_2=0.95\sim0.999$，$\epsilon=10^{-8}$。

**AdamW**：把 weight decay（[2.6 节](ch02-statistics.md#2.6) 的 L2 正则化）从梯度里拿出来，单独做 $\theta\leftarrow\theta-\eta\lambda\theta$。这是训练大模型的默认优化器。

**学习率调度**：
- **warmup**：前几千步学习率从 0 线性升到峰值。一开始 $v$ 的估计不准，参数也处于很"陡"的区域，大步长容易发散。
- **衰减**：之后按余弦曲线或线性降到峰值的约 10%。

**梯度裁剪**：若 $\|g\|$ 超过阈值（常取 1.0），就把 $g$ 缩放到阈值。这是防止偶发尖峰的保险措施。

## 4.8 反向传播与自动微分 **[事实 + 已证]**

**计算图**：前向计算的每一步都是一个节点，它记录自己的输入和输出。

**反向模式自动微分**：
1. 先完成前向计算，保存所有中间结果；
2. 令 $\bar L=1$；
3. 按拓扑序**倒着**走，每个节点把收到的梯度 $\bar y$ 用 4.1 的链式法则分给它的输入，$\bar x\mathrel{+}=J^\top\bar y$。

**为什么用"+="**：一个变量可能被多个后续节点使用（比如 residual connection 中的 $x$ 同时进入了 $f(x)$ 和 $x+f(x)$），它的梯度是所有路径的贡献之和。

**常见节点的局部规则**：
- 加法 $z=x+y$：$\bar x\mathrel{+}=\bar z$，$\bar y\mathrel{+}=\bar z$（原样分发）；
- 乘法 $z=xy$：$\bar x\mathrel{+}=y\bar z$，$\bar y\mathrel{+}=x\bar z$（乘以对方）；
- 逐元素函数 $z=\phi(x)$：$\bar x\mathrel{+}=\phi'(x)\bar z$；
- 矩阵乘法：见 4.2。

**为什么是"反向"而不是"正向"** **[已证]**：
- 输出是 1 个标量，输入是 $n$ 个参数。
- 反向模式一次遍历就得到全部 $n$ 个偏导，代价约为前向计算的 2–3 倍。
- 正向模式每次只能得到对一个输入方向的导数，需要 $n$ 次遍历。

**代价**：反向传播要保存前向计算的所有中间结果，这是训练比推理耗显存得多的主要原因。"激活重计算"（activation checkpointing）是用多算一次前向换取少存。

**两层网络的反向传播**（4.9.6 节代码）。前向：
$$a_1=XW_1+b_1,\quad h_1=\mathrm{ReLU}(a_1),\quad z=h_1W_2+b_2,\quad L=\mathrm{CE}(z,y).$$
反向（从上往下逐行使用 4.2、4.3）：
$$\bar z=(p-Y)/n,\quad \bar W_2=h_1^\top\bar z,\quad \bar h_1=\bar zW_2^\top,\quad \bar a_1=\bar h_1\odot\mathbb 1[a_1>0],\quad \bar W_1=X^\top\bar a_1.$$

## 4.9 数值实验

代码：[`code/ch04_calculus_optimization.py`](code/ch04_calculus_optimization.py)，只依赖 numpy，`python3 ch04_calculus_optimization.py` 直接运行（不到 1 秒）。

### 4.9.1 矩阵微积分恒等式

用中心差分 $\big(f(x+\epsilon e_i)-f(x-\epsilon e_i)\big)/2\epsilon$ 做数值梯度检查。4.2 节表格里的四条公式（$a^\top x$、$x^\top Ax$、$\partial W$、$\partial X$），相对误差都在 $10^{-10}$ 量级。

**梯度检查**是写反向传播时最重要的调试手段：相对误差在 $10^{-6}$ 以下，基本说明公式写对了。

### 4.9.2 Softmax 的数值稳定性

- $z=(1000,1001,1002)$ 直接算 softmax，得到 `nan`（$e^{1000}$ 溢出）。
- 先减去最大值，得到 $(0.090,0.245,0.665)$。
- 交叉熵梯度 $p-e_y$ 与数值梯度的相对误差是 $2\times10^{-10}$。

### 4.9.3 步长与条件数

$f=\frac12(x_1^2+10x_2^2)$，$L=10$，阈值 $2/L=0.2$。100 步后的 $f$：

| 步长 $\eta$ | 0.05 | 0.1 | 0.19 | 0.21 |
|---|---|---|---|---|
| $f$ | $1.8\times10^{-5}$ | $3.5\times10^{-10}$ | $3.5\times10^{-9}$ | $9.5\times10^{8}$ |

- 刚过阈值（0.21）就发散。
- 0.19 比 0.1 还慢：$x_2$ 方向每步乘 $1-1.9=-0.9$，在来回震荡。

取 $\eta=1/L$，降到 $f<10^{-8}$ 需要的步数：

| 条件数 $\kappa$ | 1 | 10 | 100 | 1000 |
|---|---|---|---|---|
| 步数 | 1 | 85 | 882 | 8860 |

步数与 $\kappa$ 成正比，和 4.4 的推导一致。

### 4.9.4 动量、SGD、Adam

$\kappa=100$ 的二次函数，降到 $f<10^{-8}$ 的步数：
- 梯度下降：882 步；
- 动量（$\beta=0.9$）：176 步；
- Adam（$\eta=0.02$）：194 步。

**小批量梯度的噪声**：线性回归，1 万个样本，比较小批量梯度和全量梯度的差。

| batch 大小 $B$ | 1 | 10 | 100 | 1000 |
|---|---|---|---|---|
| 平均 $\lVert g_B-g\rVert$ | 14.19 | 5.61 | 1.81 | 0.58 |
| 乘以 $\sqrt B$ | 14.2 | 17.8 | 18.1 | 18.2 |

$B\ge10$ 时，乘以 $\sqrt B$ 后几乎是常数，即噪声按 $1/\sqrt B$ 缩小。$B=1$ 偏低，是因为单个样本的梯度分布有重尾，"范数的平均"明显小于"均方根"。

### 4.9.5 标量自动微分

代码里的 `Value` 类大约 60 行，支持加、乘、幂、tanh、exp、log，以及按拓扑序反向传播。用它搭一个 $3\to4\to3$ 的 tanh 网络加交叉熵：
- 前向损失 0.924386，与 numpy 计算一致；
- 两层权重的梯度与数值梯度的相对误差都约为 $10^{-10}$。

PyTorch 的 autograd 原理相同，只是节点是张量而不是标量。

### 4.9.6 两层网络

三条螺旋臂、300 个点（线性不可分），2 → 64（ReLU）→ 3 的网络，矩阵形式的手写反向传播，用 Adam 训练：
- 反向传播与数值梯度的相对误差 $1.4\times10^{-8}$；
- 训练过程：

| 步数 | 1 | 100 | 500 | 3000 |
|---|---|---|---|---|
| 损失 | 1.118 | 0.185 | 0.0098 | 0.0002 |
| 训练准确率 | 28.3% | 95.0% | 100% | 100% |

## 4.10 小结与疑问

**搞懂了什么**：
- 反向传播 = 从输出往回反复使用链式法则 $\bar x=J^\top\bar y$。矩阵乘法的梯度就是 $\bar W=X^\top\bar Y$ 和 $\bar X=\bar YW^\top$。
- 交叉熵对 logits 的梯度是 $p-e_y$；实现时用 log-sum-exp 保持数值稳定。
- 学习率必须小于 $2/\lambda_{\max}$；收敛步数与条件数成正比。
- 小批量梯度无偏，噪声按 $1/\sqrt B$ 缩小。
- Adam = 动量 + 逐参数的步长归一化；大模型的标准配方是 AdamW + warmup + 衰减 + 梯度裁剪。

**疑问**：
- 大模型的损失曲面并不是凸的，为什么梯度下降几乎总能找到好的解？
- Adam 的"逐参数归一化"在什么意义上接近二阶方法？Shampoo、Muon 等新优化器改进了什么？

## 参考资料

1. S. Boyd, L. Vandenberghe. *Convex Optimization.* 第 9 章（梯度下降、条件数）。
2. K. B. Petersen, M. S. Pedersen. *The Matrix Cookbook.* 矩阵求导公式速查。
3. D. P. Kingma, J. Ba. *Adam: A Method for Stochastic Optimization.* ICLR 2015。I. Loshchilov, F. Hutter. *Decoupled Weight Decay Regularization.* ICLR 2019（AdamW）。
4. A. Karpathy. *micrograd*（GitHub）与视频 *The spelled-out intro to neural networks and backpropagation*。4.9.5 节的 `Value` 类就是这个思路。
5. G. Goh. *Why Momentum Really Works.* Distill, 2017。动量的可视化讲解。
