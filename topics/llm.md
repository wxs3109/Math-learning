# 大模型原理：学习路线

> 状态：00、01、02 已写；03、04 待写 · 更新：2026-11-22

## 这个系列在讲什么

大模型（GPT、Claude、Llama 等）在数学上是一个条件概率分布 $p_\theta(\text{下一个 token}\mid\text{前文})$：
- 参数 $\theta$ 用最大似然在海量文本上拟合；
- 计算这个分布的函数是一个叫 Transformer 的神经网络。

这个系列把这句话从头拆开，目标是懂到"能自己从头推一遍、写一遍"的程度。每一篇都按同样的结构写：问题、背景、前置知识、推导、数值实验、结论与疑问、参考资料。每篇都配一份只依赖 numpy、可以直接运行的代码。

## 目录

| 篇 | 内容 | 代码 | 状态 |
|---|---|---|---|
| [00 前置知识](llm-00-prerequisites.md) | 条件概率与链式法则；熵、交叉熵、KL；最大似然；矩阵乘法的读法；softmax；梯度下降；反向传播 | [`llm_prereq.py`](../code/llm_prereq.py) | 已写 |
| [01 语言模型](llm-01-language-model.md) | 预测下一个 token；交叉熵与困惑度；bigram 模型及其局限；embedding | [`llm_basics.py`](../code/llm_basics.py) 第 1 部分 | 已写 |
| [02 自注意力与 Transformer](llm-02-self-attention.md) | query / key / value；因果掩码；为什么除以 $\sqrt d$；多头；Transformer 块；KV cache | [`llm_basics.py`](../code/llm_basics.py) 第 2 部分 | 已写 |
| 03 动手训练一个小 GPT | 把 00–02 组装起来，用反向传播真正训练，看损失下降、生成变好 | 计划跟 nanoGPT | 待写 |
| 04 "大"是怎么来的 | scaling law；指令微调与 RLHF；推理（KV cache、采样、长上下文） | — | 待写 |

## 怎么读

- 按顺序读。00 是地基，后面每篇都会回指 00 的具体小节。
- 每篇读完都跑一下对应的代码，对照"数值实验"一节看输出。
- 有概率和线性代数基础的话，00 可以快速扫过，重点看 4.3（最大似然 = 交叉熵）、4.5（softmax 的梯度）和 4.7（反向传播）。
