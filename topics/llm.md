# 大模型原理：学习路线

> 状态：前置知识 00、P1–P4 与正文 01、02 已写；03、04 待写 · 更新：2026-11-22

## 这个系列在讲什么

大模型（GPT、Claude、Llama 等）在数学上是一个条件概率分布 $p_\theta(\text{下一个 token}\mid\text{前文})$：
- 参数 $\theta$ 用最大似然在海量文本上拟合；
- 计算这个分布的函数是一个叫 Transformer 的神经网络。

这个系列把这句话从头拆开，目标是懂到"能自己从头推一遍、写一遍"的程度。每一篇都按同样的结构写：问题、背景、前置知识、推导、数值实验、结论与疑问、参考资料。每篇都配一份只依赖 numpy、可以直接运行的代码。

## 目录

| 篇 | 内容 | 代码 | 状态 |
|---|---|---|---|
| [00 前置知识速览](llm-00-prerequisites.md) | 一篇讲完 01、02 要用的结论：链式法则；熵、交叉熵、KL；最大似然；矩阵乘法；softmax；梯度下降；反向传播 | [`llm_prereq.py`](../code/llm_prereq.py) | 已写 |
| [P1 概率论](llm-p1-probability.md) | 概率空间与贝叶斯；期望、方差、协方差；常见分布；多元高斯；大数定律与中心极限定理；熵、KL、互信息 | [`llm_p1_probability.py`](../code/llm_p1_probability.py) | 已写 |
| [P2 统计](llm-p2-statistics.md) | 偏差与方差；最大似然与 Fisher 信息；bootstrap；线性回归；过拟合；MAP = 正则化；验证集与测试集 | [`llm_p2_statistics.py`](../code/llm_p2_statistics.py) | 已写 |
| [P3 线性代数](llm-p3-linear-algebra.md) | 矩阵乘法的四种读法；秩；谱分解；SVD、低秩近似与 LoRA；投影与最小二乘；高维几何；张量形状 | [`llm_p3_linear_algebra.py`](../code/llm_p3_linear_algebra.py) | 已写 |
| [P4 微积分与优化](llm-p4-calculus-optimization.md) | 矩阵微积分；log-sum-exp；步长与条件数；动量；SGD；Adam 与训练配方；自动微分 | [`llm_p4_calculus_optimization.py`](../code/llm_p4_calculus_optimization.py) | 已写 |
| [01 语言模型](llm-01-language-model.md) | 预测下一个 token；交叉熵与困惑度；bigram 模型及其局限；embedding | [`llm_basics.py`](../code/llm_basics.py) 第 1 部分 | 已写 |
| [02 自注意力与 Transformer](llm-02-self-attention.md) | query / key / value；因果掩码；为什么除以 $\sqrt d$；多头；Transformer 块；KV cache | [`llm_basics.py`](../code/llm_basics.py) 第 2 部分 | 已写 |
| 03 动手训练一个小 GPT | 把 00–02 组装起来，用反向传播真正训练，看损失下降、生成变好 | 计划跟 nanoGPT | 待写 |
| 04 "大"是怎么来的 | scaling law；指令微调与 RLHF；推理（KV cache、采样、长上下文） | — | 待写 |

## 怎么读

- 按顺序读。00 是速览，01、02 会回指它的具体小节；P1–P4 是完整的前置知识，可以先读，也可以读正文时遇到不懂的再回来查。
- 每篇读完都跑一下对应的代码，对照"数值实验"一节看输出。
- 有概率和线性代数基础的话，读 00 就够进入正文，重点看 4.3（最大似然 = 交叉熵）、4.5（softmax 的梯度）和 4.7（反向传播）。
- 想打牢地基：P1 → P2 → P3 → P4，每章约一小时，配合代码。
