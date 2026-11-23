# 第 7 章 从零实现并训练一个小 GPT

> 标注：**[已证]** 给出了推导 · **[事实]** 标准结论 · **[直觉]** 帮助理解、不严格 · **[经验]** 工程实践中总结的做法

## 导读

第 5、6 章讲了语言模型和 Transformer 的原理，但没有真正训练过一个模型。第 8 章一下子跳到了万亿 token 的预训练。这一章补上中间这一段：只用 numpy，不用任何深度学习框架，把一个结构和 Llama 一样、只是小得多的 GPT 从头写一遍，训练它，用它生成文本。具体包括：
- byte-level BPE tokenizer：在本书的正文上训练；
- 完整的前向传播：embedding、RMSNorm、带 RoPE 和 GQA 的 causal self-attention、SwiGLU FFN、residual connection、和 embedding 共享权重的输出层；
- **手推每一层的反向传播**，再用有限差分做 gradient check；
- 训练循环：取 batch、AdamW、warmup + cosine 学习率、梯度裁剪，看 loss 怎么降、什么时候开始过拟合；
- 生成：temperature 和 top-k 采样；KV cache 为什么结果不变、快多少；
- 参数量和计算量怎么数；
- 往大里走：MoE 的路由和负载均衡；RoPE 怎样外推到更长的上下文；
- 从 CPU 到 GPU：哪些东西会变。

**与大模型的关系**

真实的大模型和这一章的模型只差在规模上：层数、维度、词表、数据量、训练步数都大几个数量级，算法几乎一样。面试里的"手撕"题（multi-head attention、KV cache、RMSNorm、RoPE、top-k 采样、BPE，见附录 C 的 [C.12.1 节](appendix-c-interview-questions.md#C.12.1)）都在这一章的代码里。

**前置知识**

第 6 章，尤其是 6.3 节末尾"真实的大模型做了哪些改动"（[6.3 节](ch06-self-attention.md#6.3)）；BPE 的思路（[5.1 节](ch05-language-model.md#5.1)）；softmax 与交叉熵的梯度（[4.3 节](ch04-calculus-optimization.md#4.3)）；矩阵微积分（[4.2 节](ch04-calculus-optimization.md#4.2)）；反向传播（[4.8 节](ch04-calculus-optimization.md#4.8)）；Adam（[4.7 节](ch04-calculus-optimization.md#4.7)）。

## 7.1 整体结构：从 token 到 logits **[事实]**

一次前向传播，张量形状这样流动（$B$ 是 batch size，$T$ 是序列长度，$d$ 是隐藏维度，$V$ 是词表大小）：

| 步骤 | 计算 | 输出形状 |
|---|---|---|
| 输入 | token id | $(B,T)$ |
| embedding | 查表：$x=W_E[\text{id}]$ | $(B,T,d)$ |
| $L$ 个块，每块两个子层 | $x\leftarrow x+\mathrm{Attn}(\mathrm{RMSNorm}(x))$；$x\leftarrow x+\mathrm{FFN}(\mathrm{RMSNorm}(x))$ | $(B,T,d)$ |
| 最后的归一化 | $\mathrm{RMSNorm}(x)$ | $(B,T,d)$ |
| 输出层 | $\text{logits}=x\,W_E^\top$ | $(B,T,V)$ |
| 损失 | 每个位置对下一个 token 的交叉熵，取平均 | 标量 |

几个要点：
- **每个位置都在预测下一个 token**。一段长度为 $T$ 的序列一次给出 $T$ 个训练信号。训练时第 $t$ 个位置的输入是真实的前 $t$ 个 token（而不是模型自己生成的），这叫 **teacher forcing**；causal mask 保证位置 $t$ 看不到 $t$ 之后的 token。
- **残差流**（residual stream）：$x$ 从头到尾一直是 $(B,T,d)$，每个子层读它、往上加一点东西。
- **输出层和 embedding 共享权重**（weight tying）：同一个 $V\times d$ 矩阵，输入时按行查表，输出时和每一行做内积。小模型里这能省下不少参数；Llama 3 这样的大模型通常不共享。

本章的模型：$V=512$，$d=64$，$L=3$，4 个 query head、2 组 K/V（GQA），每个 head 16 维，FFN 中间维度 176，上下文 128，共 171,456 个参数。

## 7.2 BPE tokenizer **[事实]**

**训练**（5.1 节讲过思路）：
1. 把语料编码成 UTF-8 字节，每个字节是一个初始 token（0–255）；
2. 统计所有相邻 token 对的出现次数，把最多的那一对合并成一个新 token（256、257……），在语料里替换掉；
3. 重复，直到词表达到目标大小。记下每次合并的顺序。

**编码新文本**：转成字节，**按训练时的顺序**依次应用每条合并规则。**解码**：每个 token 对应一串字节，拼起来再按 UTF-8 解码。

**几个实现细节**：
- **重叠**：合并 "aa" 时，"aaa" 只能变成 [aa, a]，要从左到右贪心，跳过和前一处重叠的位置；
- **字节级的好处**：任何文本都能编码，没有"未知词"；
- **代价**：一个汉字是 3 个字节。词表小的时候，很多汉字还没被合并成一个 token，会被拆成几个字节片段（7.10.2 节）。这就是 5.1 节说的"中文更费 token"的直接原因。真实的 tokenizer 词表有十万以上，常用汉字基本是一个 token；
- **预切分**：GPT 系列的 tokenizer 先用正则把文本切成词、数字、标点，再在每段内部做 BPE，避免合并出跨越空格和标点的 token。本章为了简单没有做，所以会出现"；⏎- "这样的 token。

## 7.3 前向传播 **[事实]**

每个块里的计算，按顺序：

**1. RMSNorm**：对每个位置的向量单独做，
$$\mathrm{RMSNorm}(x)=g\odot\frac{x}{\mathrm{rms}(x)},\qquad \mathrm{rms}(x)=\sqrt{\tfrac1d\textstyle\sum_ix_i^2+\epsilon}.$$

**2. Q、K、V**：$q=nW_Q$ 拆成 $H$ 个 head；$k=nW_K$、$v=nW_V$ 只拆成 $H_{kv}$ 组。

**3. RoPE**：对每个位置 $t$，把 $q$ 和 $k$ 的每个 head 的前一半和后一半配成 $hd/2$ 对，第 $i$ 对旋转 $t\theta_i$ 角，$\theta_i=\text{base}^{-2i/hd}$：
$$\begin{pmatrix}x_1'\\x_2'\end{pmatrix}=\begin{pmatrix}\cos t\theta_i&-\sin t\theta_i\\ \sin t\theta_i&\cos t\theta_i\end{pmatrix}\begin{pmatrix}x_1\\x_2\end{pmatrix}.$$
只转 $q$ 和 $k$，不转 $v$：位置信息只需要影响"谁看谁"，不需要混进被读取的内容里。

**4. GQA**：每组 K、V 复制给组里的 $G=H/H_{kv}$ 个 query head。实现上就是沿 head 维度重复（`np.repeat`）。

**5. 注意力**：
$$S=\frac{QK^\top}{\sqrt{hd}}+M,\qquad P=\mathrm{softmax}(S),\qquad O=PV.$$
$M$ 是 causal mask，上三角为 $-\infty$（代码里用 $-10^9$）。各 head 的 $O$ 拼起来，乘 $W_O$，加回残差流。

**6. SwiGLU FFN**：
$$\mathrm{FFN}(n)=\big(\mathrm{SiLU}(nW_g)\odot nW_u\big)W_d,\qquad\mathrm{SiLU}(z)=z\,\sigma(z).$$

**7. 输出**：最后一个 RMSNorm，再乘 $W_E^\top$ 得到 logits，softmax 后对真实的下一个 token 取负对数，所有位置平均。

**初始化** **[经验]**：所有矩阵从 $N(0,0.02^2)$ 采样；写回残差流的两个矩阵（$W_O$、$W_d$）再除以 $\sqrt{2L}$。每一层都往残差流上加东西，层数越多，加的次数越多，这样缩小能让残差流的方差不随层数增长。归一化的增益 $g$ 初始化为 1。这样初始时 logits 接近 0，损失约等于 $\ln V$（均匀猜测），7.10.3 节的初始损失是 6.248，$\ln512=6.238$。

## 7.4 反向传播：逐层推梯度 **[已证]**

记号：$\bar X=\partial\mathcal L/\partial X$ 表示损失对 $X$ 的梯度，形状和 $X$ 相同。按前向的逆序，每一层拿到输出的梯度，算出输入和参数的梯度。用到的规则都来自 [4.2 节](ch04-calculus-optimization.md#4.2)、[4.3 节](ch04-calculus-optimization.md#4.3)。

**交叉熵 + softmax**：对 $N=BT$ 个位置取平均，
$$\overline{\text{logits}}=\frac1N\big(\mathrm{softmax}(\text{logits})-\mathrm{onehot}(y)\big).$$

**线性层** $Y=XW$：
$$\bar W=X^\top\bar Y,\qquad\bar X=\bar YW^\top.$$
$X$ 是 $(B,T,d)$ 时，先把前两维合并成 $N$ 行。$\bar W$ 把所有位置的贡献加起来：同一个权重被所有位置共用，梯度就是各位置之和。

**输出层与 embedding**：$\text{logits}=xW_E^\top$ 给 $W_E$ 一份梯度 $\overline{\text{logits}}^\top x$；输入的查表 $x=W_E[\text{id}]$ 又给一份：把每个位置的 $\bar x$ 加到它的 token 对应的那一行上（同一个 token 出现多次就加多次，代码里用 `np.add.at`）。两份相加。

**残差**：$y=x+f(x)$ 时，$\bar x=\bar y+\overline{f\text{ 的输入}}$。梯度一份直接穿过，一份经过子层再回来。这就是 6.3 节说的"梯度可以不经过任何归一化直接传回去"。

**RMSNorm**：$y=g\odot x\,r$，$r=(\tfrac1d\sum x_i^2+\epsilon)^{-1/2}$。记 $u=\bar y\odot g$。$r$ 依赖所有 $x_j$：
$$\frac{\partial r}{\partial x_j}=-\tfrac12\Big(\tfrac1d\sum_ix_i^2+\epsilon\Big)^{-3/2}\cdot\frac{2x_j}{d}=-\frac{r^3x_j}{d},$$
所以
$$\bar x=r\,u-x\,r^3\cdot\frac1d\sum_iu_ix_i,\qquad\bar g=\sum_{\text{位置}}\bar y\odot x\,r.$$
第一项是"当 $r$ 不变时"的梯度；第二项来自 $r$ 对 $x$ 的依赖，它把梯度在 $x$ 方向上的分量去掉一部分。

**attention 里的 softmax**：$P=\mathrm{softmax}(S)$ 按行。对一行，$\partial P_i/\partial S_j=P_i(\delta_{ij}-P_j)$，所以
$$\bar S=P\odot\Big(\bar P-\sum_j\bar P_jP_j\Big)\quad(\text{每行减去自己的加权平均}).$$
被 mask 的位置 $P=0$，梯度自动为 0。

**注意力的其余部分**：$O=PV$ 给出 $\bar P=\bar OV^\top$、$\bar V=P^\top\bar O$；$S=QK^\top/\sqrt{hd}$ 给出 $\bar Q=\bar SK/\sqrt{hd}$、$\bar K=\bar S^\top Q/\sqrt{hd}$。

**GQA**：一组 K、V 被复制给 $G$ 个 head 使用，反向时把这 $G$ 个 head 的梯度加回同一组。

**RoPE**：前向是对每一对乘旋转矩阵 $R$，反向乘 $R^\top=R^{-1}$，也就是转回去。

**SwiGLU**：$h=s\odot u$，$s=\mathrm{SiLU}(z)$，
$$\bar s=\bar h\odot u,\qquad\bar u=\bar h\odot s,\qquad\bar z=\bar s\odot\sigma(z)\big(1+z(1-\sigma(z))\big).$$
最后一个因子是 $\mathrm{SiLU}'(z)=\sigma(z)+z\,\sigma(z)(1-\sigma(z))$。

**gradient check**：手推的梯度很容易错一个转置或漏一项。检验办法是对每个参数取几个元素，用中心差分 $\frac{\mathcal L(\theta+h)-\mathcal L(\theta-h)}{2h}$ 估计梯度，和手写的比。用 float64、$h=10^{-5}$，正确的实现相对误差在 $10^{-7}$ 量级，写错了通常是 $10^{-1}$ 量级（7.10.1 节）。

实际工作里用 PyTorch 的自动微分（[4.8 节](ch04-calculus-optimization.md#4.8)），不需要手写反向；但手写一遍之后，就知道每个张量为什么要存下来（激活显存，[8.3 节](ch08-pre-post-training.md#8.3)），以及 FlashAttention 为什么要在反向时重算注意力矩阵（[17.3 节](ch17-inference-serving.md#17.3)）。

## 7.5 参数量与计算量 **[已证]**

**参数量**：每层
- attention：$W_Q$ 是 $d\times Hh_d$，$W_K,W_V$ 各 $d\times H_{kv}h_d$，$W_O$ 是 $Hh_d\times d$；
- SwiGLU FFN：$3\,d\,d_{\text{ffn}}$；
- 两个 RMSNorm 的增益：$2d$。

再加上 embedding $Vd$（不共享时输出层再加一个 $Vd$）和最后一个 RMSNorm 的 $d$。

- 本章的模型：171,456；代码里用公式算的结果和实际数出来的一致；
- 同一个公式代入 Llama-3-8B（$V=128256$，$d=4096$，$L=32$，32 个 query head、8 组 K/V，$d_{\text{ffn}}=14336$，不共享）：8.03B，和 [17.2 节](ch17-inference-serving.md#17.2) 一致。

**计算量**：每个参数对每个 token 做一次乘加（2 FLOP），前向约 $2N$；反向要分别算对输入和对权重的梯度（上面线性层的两个公式），约 $4N$。所以训练每个 token 约 $6N$ FLOP（[8.2 节](ch08-pre-post-training.md#8.2)）。注意力的 $QK^\top$ 和 $PV$ 另外还有约 $4LTd$，上下文短时可以忽略。

本章的模型训练 800 步，每步 $16\times128$ 个 token，共约 $6\times1.7\times10^5\times1.6\times10^6\approx1.7\times10^{12}$ FLOP，CPU 上花了约 60 秒，即每秒约 280 亿次。H100 按 40% 的利用率，几毫秒就能做完（7.9 节）。

## 7.6 训练循环 **[事实 + 经验]**

**取数据**：整个语料编码成一个 token 序列，90% 训练、10% 验证。每一步随机取 $B$ 个起点，切出长度 $T+1$ 的片段：前 $T$ 个是输入，后 $T$ 个是目标（错开一位）。

**AdamW**：Adam（[4.7 节](ch04-calculus-optimization.md#4.7)）加上**解耦的权重衰减**：
$$\theta\leftarrow\theta-\eta\lambda\theta-\eta\frac{\hat m}{\sqrt{\hat v}+\epsilon}.$$
和"把 $\lambda\|\theta\|^2$ 加进损失"不同：那样做的话，衰减项的梯度也会被 Adam 的 $\sqrt{\hat v}$ 缩放，梯度大的参数衰减得少，失去了正则化的意义。权重衰减只用在矩阵上，不用在 RMSNorm 的增益上。$\beta_2$ 用 0.95，比默认的 0.999 小，二阶矩对最近的梯度更敏感，大模型训练更稳定。

**学习率**：前 100 步从 0 线性升到峰值（warmup），之后按余弦降到峰值的 10%。
- warmup：一开始 Adam 的二阶矩估计还不准，学习率太大容易一步走飞；
- 余弦衰减：后期小步长，在一个较好的区域里慢慢收敛（[4.4 节](ch04-calculus-optimization.md#4.4)）。

**梯度裁剪**：算出所有梯度拼成一个向量的范数，超过 1 就整体缩放到 1。偶尔出现的一个特别大的梯度（比如遇到一段很奇怪的数据）不会把参数一下子推得很远。

**看什么** **[经验]**：
- 训练 loss 和验证 loss：两者开始分开，就是在过拟合（[2.5 节](ch02-statistics.md#2.5)）。本章的语料只有 11.5 万个 token，800 步共看了 160 万个 token，相当于把训练集过了约 16 遍，后期验证 loss 基本不再下降（7.10.3 节）。真实的预训练数据多到一般只过一遍，几乎不存在这个问题，SFT 才需要担心；
- 梯度范数：突然变大往往是训练要发散的信号；
- 生成的样本：loss 是平均值，样本能直观地看出模型学到了什么。

## 7.7 生成与 KV cache **[事实 + 已证]**

**采样**：取最后一个位置的 logits，除以 temperature，只保留最大的 $k$ 个（top-k），softmax 后采样（[9.6 节](ch09-prompt-engineering.md#9.6)），接到序列末尾，重复。

**不用 KV cache**：每生成一个 token，都把整段序列重新做一遍前向。生成第 $t$ 个 token 要处理 $t$ 个位置，生成 $n$ 个 token 共处理 $\sum_t t\approx n^2/2$ 个位置。

**用 KV cache**：
- 已经生成的位置，它们的 K、V 不会再变：causal mask 保证位置 $s$ 的计算只依赖 $s$ 之前的 token，而这些不会因为后面多了 token 而改变；
- 所以只需要算**新 token 这一个位置**：它的 q 去和缓存里所有的 K 做内积，读所有的 V；再把它自己的 K、V 追加进缓存；
- 新位置在序列最后，它本来就能看所有之前的位置，所以不需要 mask。

**结果完全相同** **[已证]**：上面第一点说明缓存里的 K、V 和整段重算时的 K、V 相等；新位置的计算用到的也正是这些量。所以两种方式给出的 logits 只差浮点舍入（7.10.4 节：最大差异 $8\times10^{-6}$，采样出的序列完全一样）。注意 RoPE：新位置要用它自己的位置 $t$ 去旋转，缓存里存的是旋转之后的 K。

**快多少**：不用缓存，生成 $n$ 个 token 的线性层计算量约 $n^2N$；用缓存约 $nN$。注意力部分两者分别是 $O(n^3)$ 和 $O(n^2)$。7.10.4 节在 CPU 上：生成 1000 个 token，快 64 倍。代价是显存：每个 token 在每层都要存一份 K 和 V，这就是第 17 章整章都在对付的东西（[17.2 节](ch17-inference-serving.md#17.2)）。

**prefill 和 decode**：真实系统里，提示部分是一次性并行算完、填进缓存的（prefill），之后才一个一个地生成（decode）。本章的代码为了简单，把提示也一个一个地喂进去，结果相同，只是慢一些（[17.1 节](ch17-inference-serving.md#17.1)）。

**去掉 causal mask 会怎样**：位置 $t$ 能直接看到第 $t+1$ 个 token，也就是答案。模型很快学会"抄答案"，训练 loss 降到接近 0；但生成时后面的 token 还不存在，抄不到，模型就完全不会了（7.10.5 节：训练时的 loss 0.14，按生成时的方式评估是 6.5，比训练前的 6.25 还差）。这说明 causal mask 不是一个可有可无的细节：训练和生成看到的信息必须一致。

## 7.8 往大里走：MoE 与长上下文 **[事实 + 已证]**

### 7.8.1 Mixture of Experts（MoE）

**动机**：参数越多，模型能记住和表达的东西越多；但每个 token 的计算量也随参数量线性增长（$2N$）。MoE 让每个 token 只用一部分参数。

**做法**：把每层的 FFN 换成 $E$ 个结构相同的 FFN（expert），加一个 router：
$$g=\mathrm{softmax}(W_rx)\in\mathbb R^E,\qquad y=\sum_{i\in\mathrm{TopK}(g)}\tilde g_i\,\mathrm{FFN}_i(x),$$
$\tilde g$ 是在选中的 $k$ 个 expert 上重新归一化的权重。每个 token 只经过 $k$ 个 expert（通常 $k=1$ 或 2，近年也有更多、更小的 expert 的设计）。

**总参数和激活参数** **[事实]**：
- Mixtral 8x7B：总参数约 47B，每个 token 激活约 13B；
- DeepSeek-V3：总参数 671B，每个 token 激活 37B。

计算量看激活参数，显存看总参数：所有 expert 都要放在显存里。推理时常把不同的 expert 放在不同的卡上（expert parallelism，[17.10 节](ch17-inference-serving.md#17.10)）。

**负载均衡**：router 是学出来的，很容易"偏心"：几个 expert 被选得越多、训练得越好，就越被选中。后果：
- 冷门的 expert 几乎学不到东西，参数白白浪费；
- 每个 expert 一次能处理的 token 数有上限（**capacity**，常设为平均值的 1.25 倍），超出的 token 被丢弃（直接走残差，跳过这一层的 FFN）。

常用的办法是加一个辅助损失（Switch Transformer）：
$$\mathcal L_{\text{aux}}=E\sum_{i=1}^Ef_iP_i,$$
$f_i$ 是分给 expert $i$ 的 token 比例，$P_i$ 是 router 给 expert $i$ 的平均概率。完全均衡时 $f_i=P_i=1/E$，$\mathcal L_{\text{aux}}=1$；越不均衡越大。7.10.6 节的例子里，router 偏心时损失是 1.66，37% 的 token 分配超出了容量。DeepSeek-V3 改用不加辅助损失的做法：给每个 expert 一个偏置，只用在选 expert 上，按负载动态调整。

### 7.8.2 长上下文与 RoPE 外推

**问题**：用 4096 长度训练的模型，直接用在 16k 的输入上，效果通常会崩。从 RoPE 的角度看原因：
- 第 $i$ 对维度的频率是 $\theta_i=\text{base}^{-2i/hd}$，高频的几对转得很快，训练时早已转过很多圈，所有角度都见过；
- 低频的几对转得很慢，训练长度内**还没转满一圈**，只见过一段角度。位置超出训练长度后，它们出现了训练时从没见过的角度。

7.10.7 节的设置（$hd=128$，base 10000，训练长度 4096）下，有 28% 的维度对属于后一类。

**几种做法**：
- **Position Interpolation**（PI）：把位置除以扩展倍数 $s$，所有角度都落回训练范围。代价是高频维度的分辨率也被压缩了：相邻两个 token 的角度差从 1 弧度变成 $1/s$，模型区分相邻位置的能力变差，通常需要再微调一下；
- **NTK-aware**：不改位置，改 base：$\text{base}'=\text{base}\cdot s^{hd/(hd-2)}$。高频维度几乎不变，最低频的那一对正好被压缩 $s$ 倍，中间的维度介于两者之间。7.10.7 节里，扩到 8 倍时，超出训练范围的程度从 8 倍降到 1.75 倍；
- **YaRN**：按维度区别对待。训练中转了很多圈的维度不动，没转满一圈的维度完全插值，中间的线性过渡；另外给注意力乘一个随 $s$ 略微增大的温度系数。7.10.7 节里它既保住了高频分辨率，又没有任何维度超出训练范围。

实践中，长上下文模型通常是先用短序列预训练，再用较长的文档做一段继续预训练，同时配合上面的某种 RoPE 调整。还要注意：上下文能放进去，不等于模型能用好其中的每一部分（"lost in the middle"，[9.8 节](ch09-prompt-engineering.md#9.8)）。

## 7.9 从 CPU 到 GPU **[事实]**

本章的模型在 CPU 上训练了一分钟。真实的模型要在 GPU 上训练，要知道的几件事：

**GPU 擅长什么**：几万个简单的计算单元同时做相同的运算。矩阵乘法是理想的工作负载，Transformer 的计算量几乎全是矩阵乘法。新的 GPU 还有专门的 **tensor core**，对 BF16、FP8 的矩阵乘法快得多（数值格式见 [8.3 节](ch08-pre-post-training.md#8.3)）。

**三个数决定性能**：

| | H100 SXM | 含义 |
|---|---|---|
| 显存（HBM） | 80 GB | 模型、梯度、优化器状态、激活、KV cache 都要放得下（[8.3 节](ch08-pre-post-training.md#8.3)、[17.2 节](ch17-inference-serving.md#17.2)） |
| 显存带宽 | 3.35 TB/s | 每秒能从显存读多少数据 |
| BF16 算力 | 约 989 TFLOP/s | 每秒能做多少次运算 |

算力除以带宽约是 295：每从显存读一个字节，要做约 295 次运算才能把算力用满。读得多、算得少的操作（比如 batch 为 1 的 decode）受限于带宽，算力大部分闲着。这就是 [17.3 节](ch17-inference-serving.md#17.3) 的 roofline 模型。

**片上存储**：每个计算单元旁边有一两百 KB 的 SRAM，比显存快一个数量级以上。高效的实现（比如 FlashAttention）尽量让数据在 SRAM 里多算几次再写回显存。

**框架做了什么**：PyTorch 这样的框架提供自动微分（不用像 7.4 节那样手写反向）、GPU 上的矩阵乘法和融合算子、mixed precision、分布式训练（[8.3 节](ch08-pre-post-training.md#8.3) 的 ZeRO、FSDP）。本章的代码改成 PyTorch，主要是把 `np` 换成 `torch`、删掉 `backward` 函数、加一行 `.cuda()`。

**规模的差距**：本章的模型训练共约 $1.7\times10^{12}$ FLOP；Llama-3-8B 的预训练约 $7\times10^{23}$ FLOP（[8.2 节](ch08-pre-post-training.md#8.2)），多了 4000 亿倍。

## 7.10 数值实验

代码：[`ch07_tiny_gpt.py`](code/ch07_tiny_gpt.py)。只依赖 numpy，CPU 上约两分钟。

### 7.10.1 Gradient check

小模型（$V=20$，$d=8$，2 层，2 个 query head、1 组 K/V），float64：

```
  检查了 20 个参数张量，每个 4 个元素；最大相对误差 1.02e-07
```

20 个参数张量（embedding、每层 9 个、最后的 RMSNorm）全部通过，说明 7.4 节的每一个公式和代码都对得上。

### 7.10.2 BPE

```
  语料 207,348 字节 → 115,042 个 token，平均每个 token 1.80 字节（0.3s）
  最长的几个 token： 节） | 模型 | ；⏎-  | 一个 | 。⏎-  | 什么 | ：⏎-  |  token | 事实 | 训练
  '梯度下降的学习率'             → 24 字节、13 个 token：<e6> | <a2> | <af> | 度 | 下 | <e999> | <8d> | 的 | <e5ad> | <a6> | <e4b9> | <a0> | 率
  'KV cache 的显存'         → 18 字节、12 个 token：K | V |   | c | a | ch | e  | 的 | <e698> | <be> | <e5ad> | <98>
  'Transformer'          → 11 字节、9 个 token：T | r | an | s | f | o | r | m | er
```

- 合并出来的都是本书里最常见的片段："模型""训练""事实"（来自 **[事实]** 标签）、" token"，还有"；⏎- "这样的列表格式；
- 只有 256 次合并，"梯度"的"梯"、"学习"都还没成为完整的 token，被拆成字节片段（尖括号里是十六进制）。真实的 tokenizer 合并十几万次，这些都会是单个 token；
- 平均每个 token 1.8 字节，比按字节切短了将近一半。

### 7.10.3 训练

```
  初始损失 6.248，均匀猜测是 ln 512 = 6.238
  训练前生成：注意力掀么掣d�文以��ⓓ��0 法�揍H��������G�to��它g
  step     0  lr 0.0000  train loss 6.251  val loss 6.248  |grad|   0.41  (  0.4s)
  step   100  lr 0.0030  train loss 4.137  val loss 4.293  |grad|   0.72  (  8.2s)
  step   200  lr 0.0029  train loss 3.393  val loss 3.722  |grad|   0.79  ( 15.9s)
  step   300  lr 0.0025  train loss 3.059  val loss 3.519  |grad|   0.98  ( 23.7s)
  step   400  lr 0.0020  train loss 2.858  val loss 3.306  |grad|   1.16  ( 31.3s)
  step   500  lr 0.0013  train loss 2.665  val loss 3.269  |grad|   1.39  ( 38.9s)
  step   600  lr 0.0008  train loss 2.441  val loss 3.212  |grad|   1.39  ( 46.5s)
  step   700  lr 0.0004  train loss 2.390  val loss 3.219  |grad|   1.44  ( 53.9s)
  step   799  lr 0.0003  train loss 2.222  val loss 3.195  |grad|   1.54  ( 61.2s)
  训练后 val loss 3.195，perplexity 24.4；每字节 2.56 bit
  生成（提示 '注意力'）：注意力扩展开交叉熵、不读文件、清楚： - 模型和铷必邮件； - 交叉熵很多靠清单
  生成（提示 '梯度下降'）：梯度下降（第 11 章 表认、梯度形式的 KL cache； - 不完散； - 大模型的心果（低秩近似（1.3 节）；如�
  生成（提示 'KV cache'）：KV cache（第 11 章 prompt、面的 sow-Modd、pretrie-Seco 与 LoRA。  11.4 LoRA 口：调用
```

- 初始损失和 $\ln V$ 几乎相等，说明初始化正确：模型一开始什么都不知道；
- 前 400 步 loss 下降很快；之后训练 loss 继续降，验证 loss 只从 3.31 降到 3.20，两者的差从 0.1 左右拉大到接近 1：模型开始背训练集。语料只有 11.5 万个 token，这是预料之中的；
- 训练前的生成是乱码（还有大量不完整的字节片段）；训练后学会了本书的格式（"（第 11 章""- "开头的列表、"（1.3 节）"这样的引用）、常见的词（交叉熵、低秩近似、LoRA、rerank），但没有意义。17 万个参数、11 万个 token，只能学到这一步；
- 每字节 2.56 bit：一个字节有 8 bit，模型把这本书的文字"压缩"到了约三分之一（[1.7 节](ch01-probability.md#1.7) 交叉熵就是平均码长）。

### 7.10.4 KV cache

```
  100 个位置的 logits，最大差异 8.34e-06（float32 舍入误差量级）
  生成  128 个 token：每步重算   0.13s，用 KV cache  0.03s，快  4.6 倍
  生成  512 个 token：每步重算   2.87s，用 KV cache  0.13s，快 22.9 倍
  生成 1000 个 token：每步重算  24.55s，用 KV cache  0.38s，快 63.9 倍
  两种方式采样出的序列完全相同：True
```

- 结果一致，验证了 7.7 节的论证；
- 加速比随长度增长：不用缓存的总时间随长度近似三次方增长（512 → 1000，长度约 2 倍，时间约 8.6 倍），用缓存的近似线性增长。

### 7.10.5 去掉 causal mask

```
  训练 200 步。不带 mask 评估（训练时的样子）loss 0.139；带 mask 评估（生成时的样子）loss 6.549
  生成：注意力��怶����（：����：：泶�注�：���：；、：����：���注
```

训练 loss 0.14，看起来好得不可思议，因为模型在抄答案。换成生成时的条件，loss 是 6.55，比完全不训练（6.25）还差，生成的全是乱码。

### 7.10.6 MoE 路由

8 个 expert，top-2，8192 个 token，capacity 设为平均值的 1.25 倍：

```
  均衡的 router           各 expert 的负载 [0.97 0.99 1.05 0.95 0.96 0.96 1.04 1.08]  aux loss 1.00  超出容量被丢弃 0.0%
  偏向前两个 expert 的 router 各 expert 的负载 [3.23 2.24 0.46 0.31 0.38 0.38 0.5  0.49]  aux loss 1.66  超出容量被丢弃 37.1%
  （负载以"平均值 = 1"为单位）
```

router 只要稍微偏心，前两个 expert 的负载就是平均值的 3.2 倍和 2.2 倍，37% 的 token 分配因为超出容量被丢掉；辅助损失从 1.00 升到 1.66，训练时把它加进总损失，就能把 router 推回均衡。

### 7.10.7 RoPE 外推

$hd=128$，base 10000，训练长度 4096：

```
  训练中转满过一圈的维度对：72%；其余 28% 只见过一段角度
  方法                        扩到        含没见过角度的维度对      最多超出几倍         最高频一对的相邻角度差
  直接外推                      2x             28%          2.00             1.000 rad
  Position Interpolation    2x              0%          1.00             0.500 rad
  NTK-aware                 2x             27%          1.21             1.000 rad
  YaRN                      2x              0%          1.00             1.000 rad
  直接外推                      4x             28%          4.00             1.000 rad
  Position Interpolation    4x              0%          1.00             0.250 rad
  NTK-aware                 4x             27%          1.45             1.000 rad
  YaRN                      4x              0%          1.00             1.000 rad
  直接外推                      8x             28%          8.00             1.000 rad
  Position Interpolation    8x              0%          1.00             0.125 rad
  NTK-aware                 8x             27%          1.75             1.000 rad
  YaRN                      8x              0%          1.00             1.000 rad
```

- 直接外推：28% 的维度对出现了训练时没见过的角度，扩多少倍就超出多少倍；
- PI：没有维度超出训练范围，但最高频一对的相邻角度差被压缩到 $1/s$；
- NTK-aware：高频分辨率保持不变，低频维度仍然略微超出，但超出的程度大大减小；
- YaRN：两者兼顾。

这只是从角度覆盖的角度看，模型实际能不能用好长上下文，还要用长文档的评测来确认。

## 7.11 小结与疑问

**搞懂了什么**：
- 一个 GPT 就是 embedding、$L$ 个"RMSNorm → attention → 残差，RMSNorm → FFN → 残差"的块、最后的 RMSNorm 和输出层。真实的大模型只是更大。
- BPE 从字节开始，反复合并最常见的相邻对；词表小时汉字会被拆成字节片段。
- 反向传播逐层套用几个规则：线性层的两个公式、softmax 的"减去加权平均"、RMSNorm 多出来的那一项、残差的梯度直通、RoPE 转回去、GQA 把梯度加回同一组。手写后要用 gradient check 验证。
- 训练循环：随机窗口、teacher forcing、AdamW、warmup + 余弦、梯度裁剪；看训练和验证 loss 的差判断过拟合。
- KV cache 的结果和每步重算完全一致，生成 $n$ 个 token 的线性层计算从 $O(n^2)$ 降到 $O(n)$。causal mask 让训练和生成看到的信息一致。
- MoE 用 router 让每个 token 只激活少数 expert，要靠辅助损失或偏置保持负载均衡；RoPE 外推的问题出在低频维度，PI、NTK-aware、YaRN 用不同的方式把角度拉回训练范围。

**疑问**：
- 如果数据量增加 100 倍、参数增加 10 倍，这个模型的生成能好到什么程度？loss 会按第 8 章的 scaling law 下降吗？可以在 GPU 上用 PyTorch 版本验证。
- 每层的 attention head 实际学到了什么？比如有没有专门看"上一个 token"或者"同一个词上一次出现的位置"的 head（induction head）？
- 为什么 MoE 的 expert 往往没有学出人能理解的分工（比如"数学 expert""代码 expert"）？

## 参考资料

1. A. Karpathy. *nanoGPT*、*minbpe*、*Let's build GPT: from scratch, in code, spelled out*（视频）。本章的结构大体沿着这条路线，只是改成了 Llama 式的组件并手写了反向。
2. A. Vaswani et al. *Attention Is All You Need.* NeurIPS 2017；H. Touvron et al. *LLaMA*（2023）；Meta. *The Llama 3 Herd of Models*（2024）。
3. R. Sennrich, B. Haddow, A. Birch. *Neural Machine Translation of Rare Words with Subword Units.* ACL 2016（BPE）。
4. B. Zhang, R. Sennrich. *Root Mean Square Layer Normalization.* NeurIPS 2019；N. Shazeer. *GLU Variants Improve Transformer.* 2020；J. Su et al. *RoFormer: Enhanced Transformer with Rotary Position Embedding.* 2021。
5. I. Loshchilov, F. Hutter. *Decoupled Weight Decay Regularization.* ICLR 2019（AdamW）。
6. W. Fedus, B. Zoph, N. Shazeer. *Switch Transformers.* JMLR 2022；A. Jiang et al. *Mixtral of Experts.* 2024；DeepSeek-AI. *DeepSeek-V3 Technical Report.* 2024。
7. S. Chen et al. *Extending Context Window of Large Language Models via Positional Interpolation.* 2023；B. Peng et al. *YaRN: Efficient Context Window Extension of Large Language Models.* ICLR 2024。
