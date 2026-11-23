# 第 17 章 推理服务

> 标注：**[已证]** 给出了推导 · **[事实]** 标准结论 · **[直觉]** 帮助理解、不严格 · **[经验]** 工程实践中总结的做法

## 导读

调用 API 时，我们只看到"每百万 token 多少钱""TTFT 多少毫秒"。这些数字由服务端的推理系统决定：一张 GPU 能同时服务多少个请求，每个请求每秒能吐出多少个 token，为什么长上下文又贵又慢，为什么输出比输入贵好几倍。本章从第一性原理把这些算出来：
- 一次生成的两个阶段：prefill 和 decode；
- 显存账本：权重和 KV cache 各占多少；
- Roofline 模型：为什么 decode 受限于显存带宽而不是算力；
- batching、continuous batching、PagedAttention、prefix caching：推理引擎（vLLM、SGLang、TensorRT-LLM）的核心技术；
- speculative decoding：怎样在**不改变输出分布**的前提下加速，附完整证明；
- 量化：误差从哪里来，离群值为什么麻烦；
- 多卡并行；
- 服务指标、排队论与容量规划。

**与大模型的关系**

这一章是"后端工程师的主场"：带宽、缓存、调度、排队、容量规划，都是系统工程的老问题，只是换成了 GPU 和 KV cache。理解它，才能回答这些实际问题：自己部署开源模型要几张什么卡；API 为什么要按输入和输出分别计价；prompt caching 为什么能便宜一个数量级；延迟要求和成本之间怎样取舍。

**前置知识**

矩阵乘法与外积（[3.2 节](ch03-linear-algebra.md#3.2)）、量化误差要用到的均匀分布方差（[1.3 节](ch01-probability.md#1.3)）、self-attention 与 KV cache（[6.2 节](ch06-self-attention.md#6.2)、[6.4 节](ch06-self-attention.md#6.4)）、prompt caching（[11.6 节](ch11-agent-harness.md#11.6)）、总变差距离（[1.7 节](ch01-probability.md#1.7)）与接受–拒绝采样的思想。排队论部分用到马尔可夫链的平稳分布，17.11 节会就地推导。

## 17.1 一次生成的两个阶段 **[事实]**

**Prefill**：把整段提示（$T$ 个 token）一次送进模型，并行算出每一层的 K、V，存进 KV cache，同时得到第一个输出 token 的分布。
- 所有 token 一起算，本质是**矩阵 × 矩阵**；
- 计算量约 $2NT$ FLOP（$N$ 是参数量，每个参数对每个 token 做一次乘加）；
- 决定了**TTFT**（time to first token）。

**Decode（解码）**：之后每一步只输入**一个**新 token，读出所有历史位置的 K、V 做注意力，生成下一个 token，再把它的 K、V 追加进缓存。
- 每步本质是**矩阵 × 向量**（批里有几个请求，就是几列）；
- 决定了**每个输出 token 的时间**（TPOT，time per output token，也叫 ITL）。

一个请求的总延迟约为
$$\text{延迟}\approx\text{TTFT}+\text{TPOT}\times(\text{输出 token 数}-1).$$

这两个阶段的性能特征截然不同（17.3 节），这是推理系统所有设计的出发点。

## 17.2 显存账本 **[已证]**

**权重**：参数量 × 每个参数的字节数。BF16 是 2 字节，INT8 是 1 字节，INT4 是 0.5 字节。

**参数量怎么算**：以 Llama 3 结构为例，每层有
- 注意力：$W_Q,W_O$ 各 $d\times d$，$W_K,W_V$ 各 $d\times d_{kv}$；
- SwiGLU FFN：3 个 $d\times d_{\text{ffn}}$ 的矩阵；

再加上输入、输出两个 embedding 矩阵（各 $V\times d$）。Llama-3-8B（$L=32$，$d=4096$，$d_{kv}=1024$，$d_{\text{ffn}}=14336$，$V=128256$）算出来是 **8.03B**，BF16 下 **15.0 GiB**。

**KV cache**：每个 token 在每一层都要存一份 K 和一份 V：
$$\text{每 token 的 KV 字节数}=2\times L\times n_{kv}\times d_{\text{head}}\times\text{每个数的字节数}.$$
- Llama-3-8B：$2\times32\times8\times128\times2=131072$ 字节，即 **128 KiB/token**；
- Llama-3-70B：**320 KiB/token**；
- 如果不用 grouped-query attention（GQA，$n_{kv}=n_{\text{heads}}$），8B 会是 512 KiB/token，大 4 倍（17.9 节）。

**一张卡能服务多少请求**：
$$\text{可存的 token 总数}=\frac{\text{显存}-\text{权重}-\text{激活和框架开销}}{\text{每 token 的 KV 字节数}}.$$
Llama-3-8B（BF16）放在 80 GB 的 H100 上，留 10% 余量，KV cache 可用 57 GiB，约 **46.7 万个 token**。每个请求 4k 上下文的话，最多约 **114 个并发**。放在 24 GB 的 RTX 4090 上，只剩 6.6 GiB，约 5.4 万 token，**13 个并发**。

**结论**：**并发能力主要由 KV cache 决定**，而 KV cache 与"并发数 × 上下文长度"成正比。长上下文贵，首先贵在显存。

## 17.3 Roofline：decode 为什么受限于带宽 **[已证]**

**Roofline 模型**：一个计算任务需要 $F$ 次浮点运算，需要从显存读写 $M$ 字节。GPU 的峰值算力为 $\Pi$ FLOP/s，显存带宽为 $\beta$ 字节/s。它的时间至少是
$$t\ \ge\ \max\Big(\frac{F}{\Pi},\ \frac{M}{\beta}\Big).$$
定义**arithmetic intensity** $I=F/M$（每读一个字节做多少次运算），以及 GPU 的**ridge point** $I^*=\Pi/\beta$：
- $I<I^*$：受限于带宽，算力闲着；
- $I>I^*$：受限于算力。

H100 SXM 的 BF16 稠密算力约 989 TFLOP/s，带宽 3.35 TB/s，ridge point 约 **295 FLOP/字节**。RTX 4090 约 164 FLOP/字节。

**Decode 一步**（批里 $B$ 个请求，每个上下文长度 $T$）：
- 读的字节：权重读一遍（所有请求共用）+ 每个请求读一遍自己的 KV cache：
$$M\approx2N+B\cdot T\cdot\kappa,$$
其中 $\kappa$ 是每 token 的 KV 字节数；
- 运算量：矩阵乘法 $2NB$，加上注意力（$QK^\top$ 和 $AV$）$4LdTB$：
$$F\approx B\,(2N+4LdT).$$

**情形 1：$B=1$**。$I\approx2N/2N=1$ FLOP/字节，比 ridge point 低两个数量级，**完全受限于带宽**。每生成一个 token，都要把 15 GiB 的权重从显存读一遍：
$$t\ \ge\ \frac{2N}{\beta}=\frac{16\text{ GB}}{3.35\text{ TB/s}}\approx4.9\text{ ms}.$$
所以单个请求在 H100 上最多约每秒 205 个 token，算力利用率不到 1%（17.12.2 节）。

**情形 2：增大 $B$**。权重只读一次却被 $B$ 个请求共用，矩阵乘法部分的强度变成约 $B$。忽略 KV，需要 $B\approx295$ 才能"喂饱"H100 的算力。

**情形 3：注意力的强度永远很低**。注意力部分的运算量和读的字节都与 $B\cdot T$ 成正比，两者相除：
$$I_{\text{attn}}=\frac{4LdT B}{2L\,n_{kv}d_{\text{head}}\cdot2\cdot TB}=\frac{d}{n_{kv}d_{\text{head}}}=\frac{n_{\text{heads}}}{n_{kv}}.$$
Llama-3-8B 是 $32/8=4$ FLOP/字节。**无论批多大，decode 阶段的注意力都受限于带宽**。上下文一长，读 KV cache 的时间就会盖过读权重的时间。解"读数据时间 = 计算时间"可以算出：在 H100 上，上下文为 128 时，batch size 要到约 424 才进入算力受限；上下文 512 以上，**永远**进不了（17.12.2 节）。

**Prefill**：$T$ 个 token 一起算，矩阵乘法的强度约为 $T$。只要提示有几百个 token，prefill 就受限于**算力**。2048 个 token 的提示约 33 TFLOP，在 H100 上至少 33 ms。

**FlashAttention：减少显存读写** **[事实 + 已证]**：GPU 的存储分两层：
- **显存**（HBM）：几十 GB，就是上面带宽 $\beta$ 说的那一层；
- **片上 SRAM**：每个计算单元只有一两百 KB，但读写速度快一个数量级以上。

标准的注意力实现先算出 $T\times T$ 的打分矩阵，写回显存，再读出来做 softmax，再写回、再读出来乘 $V$。打分矩阵的读写量随 $T^2$ 增长，序列一长，时间主要花在搬数据上。

FlashAttention 把 $Q,K,V$ 切成小块，每次只把一小块搬进 SRAM 算完，打分矩阵**从不完整地写回显存**。难点是 softmax 需要整行的最大值和总和，而一次只看到一块。解法是**在线 softmax**：对一行，依次处理各块，维护到目前为止的最大值 $m$、指数和 $\ell$、加权和 $o$。来了一块新的打分 $s_j$（对应 $v_j$）时：
$$m'=\max\big(m,\max_js_j\big),\quad \ell'=e^{m-m'}\ell+\sum_je^{s_j-m'},\quad o'=e^{m-m'}o+\sum_je^{s_j-m'}v_j.$$
全部处理完后，输出就是 $o/\ell$。

正确性：按归纳法，任何时刻都有 $\ell=\sum_{\text{已处理}}e^{s_j-m}$ 和 $o=\sum_{\text{已处理}}e^{s_j-m}v_j$；换新的最大值时乘 $e^{m-m'}$ 正好把旧的项换算到新的基准上。最后
$$\frac{o}{\ell}=\frac{\sum_je^{s_j}v_j}{\sum_je^{s_j}}=\sum_j\mathrm{softmax}(s)_jv_j.$$
减去 $m$ 只是为了数值稳定（[4.3 节](ch04-calculus-optimization.md#4.3)），不改变结果。

所以 FlashAttention **是精确计算，不是近似**：运算量不变，显存读写从 $O(T^2)$ 降到 $O(T)$ 量级，打分矩阵也不用存下来。训练时反向传播需要注意力矩阵，它选择重新算一遍，而不是存下来，这也是 [8.3 节](ch08-pre-post-training.md#8.3) 里"用 FlashAttention 时激活少了一大项"的原因。它主要加速 prefill 和训练；decode 时每步只有一个 query，瓶颈在读 KV cache。

**这解释了几件"商业"现象** **[直觉]**：
- **输出比输入贵**：输入在 prefill 里成批并行地算，GPU 满负荷；输出一个一个地生成，每个 token 都要读一遍权重和 KV。
- **prompt caching 便宜**：缓存命中就跳过了这一段的 prefill 计算。
- **长上下文贵**：KV cache 既占显存（压低并发），又占带宽（拖慢每一步）。

## 17.4 Batching：用延迟换吞吐 **[已证 + 事实]**

由 17.3 节，decode 一步的时间
$$t(B)\approx\frac{2N+B T\kappa}{\beta}\quad(\text{带宽受限时}),$$
总吞吐是 $B/t(B)$。
- $B$ 从 1 增加到 32，每步时间只从 4.9 ms 涨到 7.4 ms，吞吐却从 205 涨到约 4300 token/s；
- 代价是每个请求的 TPOT 变长了。

**吞吐和单请求延迟是一对此消彼长的量**。推理服务调的就是这个 batch size。

**static batching 的问题**：凑齐一批请求一起跑，要等**最长**的那个生成完才能换下一批。输出长度长短悬殊时，大部分位置早就空了，却还占着。

**continuous batching**（也叫 iteration-level scheduling，最早由 Orca 提出）：**每一步**都检查：有请求结束，就立刻让排队的新请求补进来。批里的请求各自处在不同阶段，有的刚开始，有的快结束。17.12.4 节的模拟里，输出长度是长尾分布，static batching 的位置利用率只有 20%，continuous batching 是 88%，吞吐高 4.5 倍。这是现代推理引擎的标配。

## 17.5 PagedAttention 与 prefix caching **[事实]**

**KV cache 的碎片问题**：请求开始时不知道它最终会生成多长。如果按最大长度（比如 4096）为每个请求预留一整块**连续**显存，大部分会空着。vLLM 的论文测得，当时已有的系统里只有约 20%–38% 的 KV cache 显存真正存了 token。17.12.5 节的模拟是 22%。

**PagedAttention**（vLLM）：借鉴操作系统的虚拟内存。
- 把 KV cache 切成固定大小的**块**（比如每块 16 个 token），就像内存页；
- 每个请求有一张**块表**，记录它的逻辑位置对应哪些物理块，物理块不必连续；
- 生成到哪里，才分配到哪里；浪费只来自每个请求最后一块没填满的部分，平均不到半块。

17.12.5 节的模拟里，同样的显存，并发数从 48 提高到 221（4.6 倍），浪费从 78% 降到 0.8%。论文报告，在相同延迟下吞吐提高 2–4 倍。

**块共享**：有了块表，多个请求可以指向**同一个**物理块。
- 一个请求采样多个回答（比如 $n=4$）：共享提示部分的块，写入时再复制（copy-on-write）；
- **prefix caching**：不同请求的开头相同（同一个 system prompt、同一份文档、agent 循环里不断增长的历史），对应的 KV 块算一次、存一份。vLLM 按块内容的哈希做自动 prefix caching，SGLang 用基数树（RadixAttention）组织所有前缀。

**API 的"prompt caching"就是这个机制**（[11.6 节](ch11-agent-harness.md#11.6)）：命中的前缀不用重新 prefill。100 个请求共享 3000 token 的 system prompt 时，要算的 prefill token 减少 93%（17.12.6 节）。这也解释了为什么前缀必须**逐字节相同**：哈希的是 token 块。

## 17.6 Prefill 与 decode 的调度 **[事实]**

一个长提示的 prefill 可能要几百毫秒。如果它和正在 decode 的请求排在同一个批里一起跑，那一步就会很慢，所有正在生成的请求都会卡一下。用户看到的就是输出"顿住"。

**chunked prefill**：把长 prefill 切成小块，每一步只做一块，和 decode 请求拼在一起。prefill 用的是算力，decode 用的是带宽，拼在一起正好互补，单步时间也更平稳。

**Prefill–decode 分离**（disaggregation，如 DistServe、Splitwise）：
- 两类工作的瓶颈不同，干脆放在不同的 GPU 池里；
- prefill 机器算完 KV cache，通过高速网络传给 decode 机器；
- 两个池可以各自按 TTFT 和 TPOT 的要求独立扩容。代价是 KV 传输的开销和系统复杂度。

## 17.7 Speculative decoding **[已证]**

**动机**：decode 受限于带宽，大模型做一次前向，算 1 个位置和算 5 个位置的时间差不多（读权重的时间占大头）。如果能"猜"出接下来几个 token，让大模型一次验证，就能一步产出好几个 token。

**做法**：
1. 用一个便宜的**draft model** $q$ autoregressive 地猜 $k$ 个 token：$x_1,\dots,x_k$；
2. **target model** $p$ 做**一次**前向，同时算出这 $k+1$ 个位置的分布；
3. 从左到右逐个检查：以概率 $\min\big(1,\frac{p(x_i)}{q(x_i)}\big)$ 接受 $x_i$；
4. 第一个被拒绝的位置，从修正分布 $r(x)=\frac{\max(0,\,p(x)-q(x))}{\sum_y\max(0,\,p(y)-q(y))}$ 里重新采样一个 token，然后结束这一轮；
5. 如果 $k$ 个全被接受，再从 target model 在第 $k+1$ 个位置的分布里免费多采一个。

**定理：输出的分布和直接从 $p$ 采样完全相同** **[已证]**。只看一个位置。记接受的总概率
$$\alpha=\sum_x q(x)\min\Big(1,\frac{p(x)}{q(x)}\Big)=\sum_x\min\big(p(x),q(x)\big).$$
最终输出 $x$ 有两条路：草稿给出 $x$ 并被接受；或者被拒绝后从 $r$ 里采到 $x$。所以
$$\Pr(\text{输出}=x)=\min\big(p(x),q(x)\big)+(1-\alpha)\,r(x).$$
而 $1-\alpha=\sum_y\big[p(y)-\min(p(y),q(y))\big]=\sum_y\max\big(0,p(y)-q(y)\big)$ 恰好是 $r$ 的归一化常数，所以
$$\Pr(\text{输出}=x)=\min\big(p(x),q(x)\big)+\max\big(0,p(x)-q(x)\big)=p(x).\qquad\blacksquare$$
这是一种接受–拒绝采样。**它不是近似**：用不用 speculative decoding，输出的分布一模一样，只是快了。

**接受率与总变差**：$\alpha=\sum\min(p,q)=1-\mathrm{TV}(p,q)$。draft model 和 target model 越接近，接受率越高。

**每轮产出多少 token**：如果每个位置以 $\alpha$ 的概率独立地被接受，接受的个数服从截断的几何分布，每轮（一次 target model 前向）期望产出
$$\mathbb E[\text{token 数}]=\sum_{i=0}^{k}\alpha^i=\frac{1-\alpha^{k+1}}{1-\alpha}.$$

**加速比**：设 draft model 一次前向的时间是 target model 的 $c$ 倍，则
$$\text{加速比}\approx\frac{(1-\alpha^{k+1})/(1-\alpha)}{1+kc}.$$
$k$ 太大，后面的猜测大多被拒，还白白付出了草稿的开销。最优的 $k$ 取决于 $\alpha$ 和 $c$。

**变体**：
- 草稿不一定是另一个模型：可以是 target model 上加几个预测头（Medusa、EAGLE），或者直接从提示里找重复的片段（n-gram / prompt lookup，适合改代码、改文档这类大量复制原文的任务）；
- **批很大时收益变小**：speculative decoding 利用的是"带宽受限、算力闲置"，批大到接近算力受限时，多算的位置就不再免费了。

## 17.8 量化 **[已证 + 事实]**

**为什么量化能加速 decode**：decode 受限于带宽（17.3 节），权重从 16 位变成 4 位，要读的字节少了 4 倍，单请求的 decode 时间下界也几乎降为 1/4。在 RTX 4090 上，Llama-3-8B 从 16.2 ms/token 降到 4.2 ms/token（17.12.8 节）。同时权重占的显存少了，留给 KV cache 的就多了，并发也更高。

**均匀量化的误差** **[已证]**：对称量化把 $w$ 映射成整数 $q=\mathrm{round}(w/s)$，其中缩放因子 $s=\max|w|/(2^{b-1}-1)$，再用 $\hat w=qs$ 近似。舍入误差 $e=\hat w-w$ 近似服从 $[-s/2,s/2]$ 上的均匀分布，方差为
$$\mathrm{Var}(e)=\frac{s^2}{12}.$$
- 每少 1 位，$s$ 翻倍，误差的标准差也翻倍；
- $s$ 由**最大值**决定。一个离群的大数会让 $s$ 变大，所有正常大小的权重都跟着被粗糙地量化。

**离群值问题**：大模型里确实存在少数数值特别大的通道，尤其在激活里。所以：
- **缩放因子的粒度要细**：按张量 → 按通道 → 按组（比如每 128 个权重一组）。组越小，离群值影响的范围越小，但要多存一些缩放因子（每组一个 16 位的数，128 个一组约多 0.125 位/参数）；
- **mixed precision**：离群通道单独用 16 位保存（LLM.int8() 的思路）；
- **把难量化的部分挪走**：SmoothQuant 把激活里的离群值按通道缩放后转移到权重上；AWQ 根据激活大小找出重要的权重通道，先放大再量化；
- **补偿舍入误差**：GPTQ 利用二阶信息（Hessian，[4.4 节](ch04-calculus-optimization.md#4.4)），量化一列后调整其余尚未量化的列，抵消已经产生的误差。

17.12.8 节的实验：1024×1024 的矩阵里有 1% 的离群输入通道（数值大 25 倍），用输出的相对误差衡量：
- INT4 按张量 41%，分组 24%；
- 离群通道留 16 位、其余 INT4 分组，降到 4.4%。

**两种量化，加速的环节不同** **[事实]**：
- **只量化权重**（W4A16、W8A16）：读权重的字节减少，加速**带宽受限**的 decode；算的时候再反量化回 16 位，prefill 基本不变快；
- **权重和激活都量化**（W8A8，INT8 或 FP8）：可以用低精度的张量核心直接计算，加速**算力受限**的 prefill。H100 的 FP8 算力是 BF16 的两倍；
- **KV cache 也可以量化**（比如 FP8）：省显存、省带宽，对长上下文特别有用。

## 17.9 让 KV cache 变小 **[事实]**

由 17.2、17.3 节，KV cache 同时限制了并发数和长上下文下的 decode 速度。模型结构上的改进：
- **multi-query attention**（MQA）：所有查询头共用 1 组 K、V；
- **grouped-query attention**（GQA）：每组查询头共用一组 K、V。Llama 3 是 8 组，KV cache 只有标准 multi-head attention 的 1/4；
- **multi-head latent attention**（MLA，DeepSeek-V2）：把 K、V 压缩成一个低维的潜在向量再缓存。这和 [3.5 节](ch03-linear-algebra.md#3.5)的低秩近似是同一个思路；
- **sliding window / 局部注意力**：部分层只看最近的若干个 token，KV cache 有上界；
- 推理时**丢弃不重要的 KV**：保留注意力权重大的位置。这是有损的，要用评测确认效果。

## 17.10 多卡并行 **[事实]**

模型放不下一张卡（Llama-3-70B 的 BF16 权重就有 131 GiB），或者想让单请求更快，就要多卡。

**tensor parallelism**（TP）：把每个权重矩阵按行或列切到 $G$ 张卡上，每张卡算一部分，每层做一到两次 all-reduce 合并结果。
- 每张卡只读 $1/G$ 的权重，带宽受限的 decode 接近 $G$ 倍加速；
- 但每层都要通信，必须用 NVLink 这类高速互联，通常只在一台机器的 8 张卡之内做。

**pipeline parallelism**（PP）：按层切，第 1–20 层放卡 1，第 21–40 层放卡 2……
- 通信量小，可以跨机器；
- 单个请求要依次经过每张卡，延迟不降。适合提高吞吐。

**expert parallelism**（EP）：Mixture of Experts（MoE）模型里，不同的专家放在不同的卡上，token 按路由发到对应的卡。

**经验** **[经验]**：能放进一张卡就不要拆（量化后放得下，往往比 TP=2 更划算）；要拆先在机器内做 TP；跨机器再考虑 PP。

## 17.11 服务指标、排队与容量规划 **[已证 + 经验]**

**指标**：
- **TTFT**：TTFT，用户感知的"反应速度"；
- **TPOT / ITL**：输出 token 之间的间隔，流式输出时的"打字速度"。人的阅读速度约每秒几个到十几个词，TPOT 低于约 50 ms 时，用户基本感觉不到在等；
- **吞吐**：每秒处理的 token 或请求数，决定成本；
- **Goodput**：**满足延迟目标（SLO）的**请求数每秒。只看吞吐会误导：把批开得很大，吞吐很高，但大部分请求超时了。

**Little 定律**：稳定的系统里，
$$L=\lambda W,$$
即系统中的平均请求数 = 到达率 × 平均停留时间。它对任何排队规则都成立。例：每秒 20 个请求，每个平均要 8 秒，系统里平均同时有 160 个请求，你的 KV cache 必须放得下 160 个并发。

**M/M/1 队列：一个马尔可夫链** **[已证]**。请求按速率 $\lambda$ 的泊松过程到达，服务时间服从速率 $\mu$ 的指数分布，单个服务台。系统中的请求数 $n$ 是一个生灭链：$n\to n+1$ 的速率为 $\lambda$，$n\to n-1$ 的速率为 $\mu$。由细致平衡 $\pi_n\lambda=\pi_{n+1}\mu$，得 $\pi_n=(1-\rho)\rho^n$，其中**利用率** $\rho=\lambda/\mu<1$。于是
$$L=\sum_nn\pi_n=\frac{\rho}{1-\rho},\qquad W=\frac{L}{\lambda}=\frac{1}{\mu-\lambda}.$$
$\rho\to1$ 时延迟发散：利用率 50% 时平均停留 2 个服务时间，90% 时 10 个，95% 时 20 个。

**服务时间不是指数分布时：Pollaczek–Khinchine 公式**（M/G/1）。平均排队等待时间为
$$W_q=\frac{\lambda\,\mathbb E[S^2]}{2(1-\rho)}=\frac{\rho}{1-\rho}\cdot\frac{\mathbb E[S^2]}{2\,\mathbb E[S]}.$$
它依赖服务时间的**二阶矩**。大模型请求的输出长度是长尾的：偶尔有一个请求要生成几千个 token，$\mathbb E[S^2]$ 很大。17.12.9 节的模拟里，平均服务时间同样为 1，长尾分布在利用率 90% 时的平均等待是指数分布的 3.4 倍，p99 是 3.8 倍。

**对推理服务的启示** **[经验]**：
- 不要按 100% 利用率规划容量。要满足延迟目标，峰值利用率通常要留出余量；
- 长尾请求会堵住短请求（队头阻塞）。continuous batching、preemption（KV cache 不够时把长请求换出到 CPU 内存或重算）、按预计长度分队列，都是在对付这个问题；
- 严格地说，带 batching 的推理服务器不是单服务台队列。这里的公式用来建立直觉、做粗略估计，精确的容量要靠压测（17.13 节的脚本是一个起点）。

**一个容量估算的例子**：要服务 Llama-3-8B，峰值每秒 10 个请求，平均提示 1000 token、输出 300 token，要求 TPOT < 30 ms。
1. 由 Little 定律，每个请求约 $0.1+300\times0.03\approx9$ 秒，平均同时约 90 个请求；
2. 每个请求平均约 1150 个 token 的 KV，90 个约 10 万 token、约 13 GiB。H100 放得下（17.2 节）；
3. batch size 约 90、上下文约 1150 时，decode 一步的下界约 $(16+90\times1150\times131072/10^9)\text{ GB}/3.35\text{ TB/s}\approx8.8$ ms。满足 30 ms，留有余量。

一张 H100 在理论上够用。实际效率通常只有理论峰值的 50%–70%，所以要用压测确认，并按峰值留出余量。

## 17.12 数值实验（CPU）

代码：[`code/ch17_inference_serving.py`](code/ch17_inference_serving.py)，只依赖 numpy，`python3 ch17_inference_serving.py` 运行，约 5 秒。理论计算用的 GPU 参数：

| GPU | 显存 | 带宽 | BF16/FP16 稠密算力 | ridge point（FLOP/字节） |
|---|---|---|---|---|
| H100 SXM | 80 GB | 3.35 TB/s | 989 TFLOP/s | 295 |
| A100 80GB | 80 GB | 2.04 TB/s | 312 TFLOP/s | 153 |
| RTX 4090 | 24 GB | 1.008 TB/s | 165 TFLOP/s | 164 |
| T4 | 16 GB | 0.32 TB/s | 65 TFLOP/s | 203 |

### 17.12.1 显存账本

| 模型 | 参数 | BF16 权重 | KV cache / token | 不用 GQA 时 |
|---|---|---|---|---|
| Llama-3-8B | 8.03B | 15.0 GiB | 128 KiB | 512 KiB |
| Llama-3-70B | 70.55B | 131.4 GiB | 320 KiB | 2560 KiB |

Llama-3-8B 在 H100 和 A100 80GB 上都能存约 46.7 万个 token 的 KV，即 4k 上下文 114 个并发；在 4090 上约 5.4 万 token，13 个并发；T4 连权重都放不下（16 GB 的卡留 10% 余量后只剩 14.4 GiB）。

### 17.12.2 Roofline

Llama-3-8B，BF16，上下文 2048，H100 SXM：

| batch size | 每步 | 读数据 | 计算 | 吞吐（token/s） | 显存放得下吗 |
|---|---|---|---|---|---|
| 1 | 4.87 ms | 4.87 ms | 0.02 ms | 205 | 是 |
| 8 | 5.44 ms | 5.44 ms | 0.14 ms | 1,472 | 是 |
| 32 | 7.36 ms | 7.36 ms | 0.55 ms | 4,349 | 是 |
| 128 | 15.05 ms | 15.05 ms | 2.22 ms | 8,505 | 是 |
| 256 | 25.31 ms | 25.31 ms | 4.44 ms | 10,116 | 否 |

- 每一行都受限于带宽，计算时间不到读数据时间的 1/5；
- 批从 1 到 128，吞吐提高 41 倍，每步时间只增加 3 倍；
- 批再大，KV cache 的读取占了大头，而且显存放不下了。

在 H100 上进入算力受限需要的 batch size：上下文 0 时 295，128 时 424，512 及以上时不存在（注意力部分的强度只有 4，17.3 节）。

### 17.12.3 本机 CPU 实测

在这个 2 核 CPU 上，$W$（4096×4096，FP32，64 MB）乘以 $X$（4096×B）：

| B | 每次 | 每列（每个 token） | GFLOP/s |
|---|---|---|---|
| 1 | 2.44 ms | 2.44 ms | 13.7 |
| 4 | 5.59 ms | 1.40 ms | 24.0 |
| 16 | 7.88 ms | 0.49 ms | 68.1 |
| 64 | 19.73 ms | 0.31 ms | 108.8 |
| 256 | 61.24 ms | 0.24 ms | 140.3 |

形状和 GPU 上一样：$B=1$ 时只用到 CPU 算力的十分之一，时间花在读 64 MB 的权重上；批变大，每个 token 分摊到的时间降到约 1/10。具体数字会随机器不同而变化。

### 17.12.4 Static batching 与 continuous batching

1000 个请求，输出长度服从中位数 120 的长尾分布，批容量 32：

| | 总时间 | 吞吐 | 平均完成时间 | 位置利用率 |
|---|---|---|---|---|
| static batching | 44,227 | 4.0 | 22,446 | 20% |
| continuous batching | 9,754 | 18.0 | 4,440 | 88% |

（时间单位是"一步 decode 的固定开销"。）

### 17.12.5 预留 vs 分页

KV 预算 20 万 token，最大长度 4096，进行中请求的长度服从中位数约 700 的长尾分布：
- 按最大长度预留：最多 48 个并发，真正存了 token 的只占 22%；
- 分页（每页 16 token）：最多 221 个并发，浪费 0.8%。

### 17.12.6 Prefix caching

100 个请求共享 3000 token 的 system prompt，各自 200 token：prefill 要算的 token 从 320,000 降到 23,000，省 93%。

### 17.12.7 Speculative decoding

target model 是一个 40 个状态的马尔可夫链，draft model 是它与一个随机分布的 7:3 混合。生成 6 万个 token，比较二元组（相邻两个 token）的频率分布：

| $k$ | 接受率 $\alpha$ | 每次 target model 调用产出 | 公式 $\frac{1-\alpha^{k+1}}{1-\alpha}$ | 与直接采样的总变差 | 两次直接采样之间的总变差 |
|---|---|---|---|---|---|
| 2 | 0.807 | 2.46 | 2.46 | 0.0639 | 0.0667 |
| 4 | 0.805 | 3.40 | 3.40 | 0.0626 | 0.0624 |
| 8 | 0.803 | 4.38 | 4.37 | 0.0637 | 0.0648 |

- speculative decoding 的输出和直接采样的差距，与"两次直接采样之间"的随机差距相同：分布没有改变，验证了 17.7 节的定理；
- 每次调用产出的 token 数与公式吻合；
- 草稿开销 $c=0.05$ 时，加速比：$k=2$ 为 2.22 倍，$k=4$ 为 2.84 倍，$k=8$ 为 3.20 倍。

### 17.12.8 量化

1024×1024 矩阵，1% 的输入通道数值大 25 倍，乘以 64 个随机向量，输出的相对误差：

| 方案 | 每参数位数 | 相对误差 |
|---|---|---|
| INT8 按张量 | 8 | 9.2% |
| INT8 按输出通道 | 8 | 4.2% |
| INT4 按张量 | 4 | 40.8% |
| INT4 按输出通道 | 4 | 37.7% |
| INT4 分组（每组 128） | 4.125 | 24.2% |
| 离群通道留 16 位 + INT4 分组 | 4.242 | 4.4% |

- 离群值在**输入通道**上，"按输出通道"的每个缩放因子仍然覆盖了离群值，帮助不大；分组把影响限制在少数组里；
- 对照：没有离群值时，INT4 分组的误差是 11.7%，与 17.8 节的估计一致：每组最大值约 $2.7\sigma$，$s\approx2.7\sigma/7$，误差标准差 $s/\sqrt{12}\approx0.11\sigma$；
- mixed precision 的误差最低，因为输出主要由离群通道贡献，而它们没有被量化；
- 实际使用的 INT4 方法（GPTQ、AWQ）会补偿舍入误差，模型的最终效果比"直接四舍五入"好得多。上表只用来说明误差从哪里来。

Llama-3-8B 在 RTX 4090 上单请求的 decode 下界：BF16 16.2 ms，INT8 8.2 ms，INT4 4.2 ms。

### 17.12.9 排队

一台服务器，泊松到达，平均服务时间为 1，模拟 20 万个请求：

| 服务时间 | 利用率 $\rho$ | 平均等待（模拟） | P-K 公式 | 平均总时间 | p99 总时间 |
|---|---|---|---|---|---|
| 指数 | 0.5 | 1.00 | 1.00 | 2.00 | 9.1 |
| 指数 | 0.9 | 8.99 | 9.05 | 9.99 | 49.8 |
| 指数 | 0.95 | 18.84 | 18.98 | 19.84 | 91.0 |
| 长尾 | 0.5 | 3.45 | 3.51 | 4.45 | 46.3 |
| 长尾 | 0.9 | 30.59 | 31.80 | 31.59 | 188.6 |
| 长尾 | 0.95 | 74.54 | 71.15 | 75.54 | 372.4 |

- 模拟与 Pollaczek–Khinchine 公式吻合（长尾的高利用率情形波动较大，是因为二阶矩估计本身噪声大）；
- 指数分布的平均总时间正是 M/M/1 的 $1/(1-\rho)$：2、10、20；
- 平均服务时间相同，长尾分布在利用率 90% 时，平均等待约为指数分布的 3.4 倍，p99 约为 3.8 倍。

## 17.13 GPU 实验（等你在 GPU 上运行）

代码：[`code/ch17_gpu_benchmark.py`](code/ch17_gpu_benchmark.py)。在 Kaggle Notebook（打开 GPU 和 Internet）、Colab 或 RunPod 上：
```
pip install vllm
python ch17_gpu_benchmark.py
```
vLLM 需要计算能力 7.5 以上的 GPU，Kaggle 和 Colab 免费的 T4 满足要求。脚本最后打印一段 JSON，把它发回来，我把实测结果写进这一节，和理论值对照。

**三个实验和理论预期**：

**A. 矩阵乘法从带宽受限走到算力受限**（只需要 PyTorch）：$W$ 为 8192×8192 FP16（128 MiB），batch size 从 1 到 4096。
- 预期 $B$ 很小时，时间几乎不随 $B$ 变化，有效带宽接近显存带宽（T4 约 320 GB/s，$B=1$ 时约 0.4 ms）；
- $B$ 超过 ridge point（T4 约 203）之后，时间开始随 $B$ 线性增长，算力接近峰值。

**B. vLLM 的并发与吞吐**：Qwen2.5-0.5B，同时提交 1、8、32、128 个请求，每个生成 256 个 token。
- 预期总吞吐随并发数大幅上升，单个请求的速度只略微下降（17.4 节）。

**C. prefix caching**：32 个请求共享约 2000 token 的前缀，只做 prefill，跑两次。
- 预期第二次（前缀已缓存）明显更快（17.5 节）。

## 17.14 小结与疑问

**搞懂了什么**：
- Prefill 一次算完整个提示，受限于算力，决定 TTFT；decode 每步一个 token，受限于显存带宽，决定输出速度。
- 显存 = 权重 + KV cache；KV cache 每 token $2Ln_{kv}d_{\text{head}}\times$字节数，它决定并发能力。
- Roofline：单请求 decode 的强度约为 1，要靠 batching 共用一次权重读取；decode 时注意力的强度只有 $n_{\text{heads}}/n_{kv}$，所以长上下文永远受限于带宽。
- continuous batching 每步补位；PagedAttention 用分页消除碎片，并支持前缀共享，API 的 prompt caching 就建立在它上面。
- speculative decoding 用接受–拒绝采样保证输出分布不变；每轮期望产出 $(1-\alpha^{k+1})/(1-\alpha)$ 个 token，$\alpha=1-\mathrm{TV}(p,q)$。
- 量化误差的方差是 $s^2/12$，$s$ 由最大值决定，离群值要靠细粒度缩放、mixed precision 或误差补偿处理。只量化权重加速 decode，权重和激活都量化才加速 prefill。
- 容量规划：Little 定律算并发；P-K 公式说明延迟随利用率和服务时间的二阶矩急剧上升，要留余量。

**疑问**：
- Mixture of Experts 模型（MoE）的推理特征有什么不同？每个 token 只激活一部分专家，但所有专家的权重都要放在显存里。
- 推理引擎怎样在 preemption（换出或重算 KV）和排队之间做选择？
- 实测效率和 roofline 上界差多少？差在哪里？（等 17.13 节的结果。）

## 参考资料

1. W. Kwon et al. *Efficient Memory Management for Large Language Model Serving with PagedAttention.* SOSP 2023（vLLM）。
2. G.-I. Yu et al. *Orca: A Distributed Serving System for Transformer-Based Generative Models.* OSDI 2022（continuous batching）。
3. Y. Leviathan, M. Kalman, Y. Matias. *Fast Inference from Transformers via Speculative Decoding.* ICML 2023；C. Chen et al. *Accelerating Large Language Model Decoding with Speculative Sampling.* 2023。
4. T. Dettmers et al. *LLM.int8().* NeurIPS 2022；G. Xiao et al. *SmoothQuant.* ICML 2023；E. Frantar et al. *GPTQ.* ICLR 2023；J. Lin et al. *AWQ.* MLSys 2024。
5. Y. Zhong et al. *DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving.* OSDI 2024；A. Agrawal et al. *Sarathi-Serve*（chunked prefill）。OSDI 2024。
6. L. Zheng et al. *SGLang: Efficient Execution of Structured Language Model Programs.* 2024（RadixAttention）。
7. DeepSeek-AI. *DeepSeek-V2.* 2024（multi-head latent attention MLA）；J. Ainslie et al. *GQA.* EMNLP 2023。
8. S. Williams, A. Waterman, D. Patterson. *Roofline: An Insightful Visual Performance Model for Multicore Architectures.* CACM 2009。
9. L. Kleinrock. *Queueing Systems, Volume 1: Theory.* 1975（M/M/1、M/G/1、Pollaczek–Khinchine 公式）。
