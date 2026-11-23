# 第 6 章 Self-attention 与 Transformer

> 标注：**[已证]** 给出了推导 · **[事实]** 标准结论 · **[直觉]** 帮助理解、不严格

## 导读

[第 5 章](ch05-language-model.md) 留下的问题：神经网络 $f_\theta$ 读入一段文本的向量 $X\in\mathbb R^{T\times d}$，要在每个位置输出"下一个 token"的 logits。它怎样才能在每个位置有效地利用整段前文？

Transformer 的回答是**self-attention**：每个位置按相关性，从之前的所有位置读取信息。

**与大模型的关系**

- 位置 $t$ 预测下一个 token 时，有用的信息可能在前文的任何地方，比如十句话之前提到的人名。
- 更早的循环神经网络（RNN）把前文压缩进一个固定长度的状态向量，逐个位置往后传。距离一远，信息就丢了，而且没法并行计算。
- self-attention 让每个位置**直接**去看所有之前的位置，距离不再是障碍，所有位置也可以并行计算。

**前置知识**

- 内积与余弦相似度（[3.1 节](ch03-linear-algebra.md#3.1)）；矩阵乘法的读法，$QK^\top$ 是两两内积（[3.2 节](ch03-linear-algebra.md#3.2)）；高维随机向量的内积量级（[3.7 节](ch03-linear-algebra.md#3.7)）。
- softmax（[4.3 节](ch04-calculus-optimization.md#4.3)）；反向传播（[4.8 节](ch04-calculus-optimization.md#4.8)）。
- embedding（[5.4 节](ch05-language-model.md#5.4)）。

## 6.1 Positional encoding **[事实]**

注意力本身不区分顺序（见 6.2 的细节 3），所以要把"第几个位置"的信息加进去，即**positional encoding**。最简单的做法是给每个位置一个向量，直接加到 token 的 embedding 上。现在的大模型多用另一种做法 RoPE：不加到输入上，而是在算注意力时旋转 query 和 key（6.3 节末尾）。

## 6.2 Self-attention **[事实 + 直觉]**

**做法**：每个位置的向量 $x_t$ 经三个矩阵变换出三个向量：
- **query** $q_t=x_tW_Q$："我在找什么"；
- **key** $k_s=x_sW_K$："我这里有什么"；
- **value** $v_s=x_sW_V$："如果你选中我，我给你的内容"。

位置 $t$ 对位置 $s$ 的打分是内积 $q_t\cdot k_s$，经 softmax 变成权重。输出是 value 按这些权重的加权平均。写成矩阵形式：
$$\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\Big(\frac{QK^\top}{\sqrt{d}}+M\Big)V.$$

形状：$Q,K,V$ 都是 $T\times d$（$d$ 是每个头的维数），打分矩阵 $QK^\top$ 是 $T\times T$，输出是 $T\times d$。

**三个关键细节**：
1. **causal mask $M$**：语言模型预测第 $t+1$ 个 token 时不能偷看后面。所以 $M$ 在 $s>t$ 处取 $-\infty$，softmax 之后这些位置的权重恰好为 0。注意力矩阵因此是**下三角**的。
2. **为什么除以 $\sqrt d$** **[已证]**：
   - 设 $q,k$ 的各分量独立、均值 0、方差 1，则 $q\cdot k=\sum_{i=1}^dq_ik_i$ 的方差是 $d$。
   - $d$ 很大时，打分的量级很大，softmax 会退化成接近 one-hot 的分布。这时梯度几乎为 0，学不动。
   - 除以 $\sqrt d$ 把方差拉回 1。
3. **为什么说它不区分顺序**：如果把输入的行打乱，输出也只是相应地打乱。所以顺序信息必须靠 positional encoding 额外提供。

**multi-head attention**：把上面的计算在 $h$ 组较小的 $W_Q,W_K,W_V$ 上并行做 $h$ 次，输出拼起来。不同的头可以学到不同类型的关系，比如一个头看语法、一个头看指代。

## 6.3 一个 Transformer 块 **[事实]**

一个块（block）由两个子层组成，每个子层都带 residual connection 和归一化（LayerNorm）：
$$X\leftarrow X+\mathrm{MultiHeadAttention}(\mathrm{LN}(X)),\qquad X\leftarrow X+\mathrm{FFN}(\mathrm{LN}(X)).$$
- **注意力子层**：在位置之间交换信息。
- **FFN 子层（FFN）**：对每个位置单独做一个两层的全连接网络（结构和 [A.7 节](appendix-a-math-quick-reference.md#A.7) 的例子一样），可以理解成"对取到的信息做加工"。
- **residual connection**：让信息和梯度能直接穿过很多层，深层网络才训练得动。

GPT 类模型就是把这样的块堆几十到上百层，最后接一个输出层，在每个位置得到下一个 token 的 logits，再用 [第 5 章](ch05-language-model.md) 的交叉熵损失训练。

**归一化在做什么** **[事实]**：对每个位置的向量 $x\in\mathbb R^d$ 单独做（不跨位置、不跨样本）：
$$\mathrm{LayerNorm}(x)=\gamma\odot\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta,\qquad \mu=\frac1d\sum_ix_i,\ \sigma^2=\frac1d\sum_i(x_i-\mu)^2.$$
$\gamma,\beta$ 是可学习的。它把每层的输入拉回固定的尺度，训练更稳定。不用 BatchNorm 是因为它要在一批样本之间统计，而语言模型的序列长短不一，推理时又常常一次只有一个样本。

**上面写的是 Pre-Norm** **[事实]**：归一化放在子层**之前**，残差那条路上不做归一化。原始 Transformer 用的是 Post-Norm：$X\leftarrow\mathrm{LN}(X+\mathrm{Sublayer}(X))$。Pre-Norm 的残差路径从输入一直通到输出，梯度可以不经过任何归一化直接传回去，层数很深时更容易训练，现在的大模型基本都用它。

**真实的大模型在这个结构上做了哪些改动** **[事实 + 已证]**：第 8 章和第 17 章都以 Llama 3 为例，它和上面的结构有四处不同：

1. **RMSNorm** 代替 LayerNorm：不减均值，也没有 $\beta$，
$$\mathrm{RMSNorm}(x)=\gamma\odot\frac{x}{\sqrt{\frac1d\sum_ix_i^2+\epsilon}}.$$
少算一个均值，效果相当。
2. **SwiGLU FFN**：用三个矩阵代替两个，
$$\mathrm{FFN}(x)=W_{\text{down}}\big(\mathrm{SiLU}(W_{\text{gate}}x)\odot W_{\text{up}}x\big),\qquad\mathrm{SiLU}(z)=z\,\sigma(z).$$
一路算出"门"，逐元素乘到另一路上，控制哪些特征通过。中间维度 $d_{\text{ffn}}$ 取 $\frac83d$ 时，三个矩阵的参数量正好等于普通的"先放大 4 倍再缩回"的两层前馈；实际模型常取得更大一些（Llama-3-8B 是 3.5 倍）。第 8、17 章说的 gate、up、down 三个矩阵就是它们。
3. **RoPE（rotary position embedding）**：不给输入加位置向量，而是把 query 和 key 的分量两两配对，看成平面上的点，位置 $m$ 处转 $m\theta_i$ 角（每一对用不同的频率 $\theta_i$）。它的好处是注意力打分**只依赖相对位置**。证明：设 $R(\phi)$ 是转角为 $\phi$ 的旋转矩阵，它满足 $R(a)^\top R(b)=R(b-a)$，于是
$$\big(R(m\theta)q\big)^\top\big(R(n\theta)k\big)=q^\top R(m\theta)^\top R(n\theta)k=q^\top R\big((n-m)\theta\big)k,$$
只和 $n-m$ 有关。每一对分量都是如此，加起来也是。
4. **grouped-query attention（GQA）**：多个查询头共用一组 K、V，KV cache 变小几倍（[17.9 节](ch17-inference-serving.md#17.9)）。

另外，输入的 embedding 矩阵和输出层的矩阵各是 $V\times d$，词表大时它们占参数量的不小一部分。按这些结构数一遍参数量，见 [17.2 节](ch17-inference-serving.md#17.2)。第 7 章会把这些都写成代码。

## 6.4 由结构直接推出的几个现象 **[直觉]**

- **context window 有上限、长文本贵**：注意力要算 $T\times T$ 的打分矩阵，计算量随长度平方增长。
- **生成时可以缓存**：生成第 $t+1$ 个 token 时，前面位置的 $K,V$ 不变，可以存起来复用，这就是 KV cache。
- **"幻觉"**：模型的目标只是给出"像训练文本"的下一个 token 分布，并没有"核对事实"这一步。流畅和正确是两件事。

## 6.5 数值实验

代码：[`code/ch05_06_lm_attention.py`](code/ch05_06_lm_attention.py) 第 2 部分，`python3 ch05_06_lm_attention.py` 直接运行。

### 6.5.1 因果 self-attention

取 $T=6$ 个位置，模型维数 16，头维数 8，随机权重：

```
注意力矩阵（第 i 行 = 位置 i 分给各位置的权重）：
[[1.   0.   0.   0.   0.   0.  ]
 [0.72 0.28 0.   0.   0.   0.  ]
 [0.37 0.16 0.48 0.   0.   0.  ]
 [0.05 0.12 0.34 0.48 0.   0.  ]
 [0.05 0.1  0.28 0.38 0.19 0.  ]
 [0.44 0.13 0.09 0.07 0.08 0.19]]
```

- 下三角，右上角全为 0，每行之和为 1：每个位置只在"自己和之前"的位置上分配注意力。
- **因果性验证**：只改动最后一个 token 的输入，前 5 个位置的输出变化精确为 0，最后一个位置变化 8.47。

### 6.5.2 为什么除以 $\sqrt d$

| $d$ | 打分方差（不缩放） | 打分方差（除以 $\sqrt d$） |
|---|---|---|
| 4 | 4.1 | 1.02 |
| 64 | 63.6 | 0.99 |
| 1024 | 1042.6 | 1.02 |

$d=1024$ 时，不缩放的 softmax 最大权重是 0.999，几乎是 one-hot；缩放后是 0.284，分布正常。和 6.2 的推导一致。

## 6.6 小结与疑问

**搞懂了什么**：
- self-attention = 每个位置用 query 去匹配所有之前位置的 key，再按权重读取 value。
- causal mask 保证不看未来，$\sqrt d$ 保证 softmax 不退化，positional encoding 补上顺序信息。
- Transformer 块 = 注意力（位置间交换信息）+ FFN（每个位置单独加工），外加残差和归一化。

**疑问**：
- 注意力权重是不是真能解释成"模型在看哪里"？可解释性研究里对此有争议。
- positional encoding 有好几种（正弦、可学习、rotary position embedding RoPE），它们对长上下文外推的影响是什么？

## 参考资料

1. A. Vaswani et al. *Attention Is All You Need.* NeurIPS 2017。Transformer 的原始论文。
2. Jay Alammar. *The Illustrated Transformer.* 图解博客，适合在读原论文之前看。
3. 3Blue1Brown 关于 Transformer 与注意力机制的视频，适合建立直觉。
