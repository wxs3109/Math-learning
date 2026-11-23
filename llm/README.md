# 大模型原理

> 从数学基础到工程实践 · 每章配可运行的代码 · 更新：2026-11-22

## 这本笔记在讲什么

大模型（GPT、Claude、Llama 等）在数学上是一个条件概率分布 $p_\theta(\text{下一个 token}\mid\text{前文})$：
- 参数 $\theta$ 用最大似然在海量文本上拟合；
- 计算这个分布的函数是一个叫 Transformer 的神经网络。

这本笔记把这句话从头拆开，目标是懂到"能自己从头推一遍、写一遍"的程度；然后讲怎样把模型用进真实的系统：提示、工具、agent、MCP、Skills、检索和评测。

## 目录

**第一部分 数学基础**

| 章 | 内容 | 代码 |
|---|---|---|
| [第 1 章 概率论](ch01-probability.md) | 概率空间与贝叶斯；期望、方差、协方差；常见分布；多元高斯；大数定律与中心极限定理；熵、KL、互信息 | [`ch01_probability.py`](code/ch01_probability.py) |
| [第 2 章 统计](ch02-statistics.md) | 偏差与方差；最大似然与 Fisher 信息；bootstrap；线性回归；过拟合；MAP = 正则化；验证集与测试集 | [`ch02_statistics.py`](code/ch02_statistics.py) |
| [第 3 章 线性代数](ch03-linear-algebra.md) | 矩阵乘法的四种读法；秩；谱分解；SVD、低秩近似与 LoRA；投影与最小二乘；高维几何；张量形状 | [`ch03_linear_algebra.py`](code/ch03_linear_algebra.py) |
| [第 4 章 微积分与优化](ch04-calculus-optimization.md) | 矩阵微积分；log-sum-exp；步长与条件数；动量；SGD；Adam 与训练配方；自动微分 | [`ch04_calculus_optimization.py`](code/ch04_calculus_optimization.py) |

**第二部分 语言模型与 Transformer**

| 章 | 内容 | 代码 |
|---|---|---|
| [第 5 章 语言模型](ch05-language-model.md) | 预测下一个 token；交叉熵与 perplexity；bigram 模型及其局限；embedding | [`ch05_06_lm_attention.py`](code/ch05_06_lm_attention.py) 第 1 部分 |
| [第 6 章 Self-attention 与 Transformer](ch06-self-attention.md) | query / key / value；causal mask；为什么除以 $\sqrt d$；多头；Transformer 块；KV cache | [`ch05_06_lm_attention.py`](code/ch05_06_lm_attention.py) 第 2 部分 |
| [第 7 章 从零实现并训练一个小 GPT](ch07-tiny-gpt.md) | byte-level BPE；Llama 式结构的前向（RMSNorm、RoPE、GQA、SwiGLU）；逐层手推反向与 gradient check；参数量与 6N；AdamW、warmup + cosine、梯度裁剪；KV cache；去掉 causal mask 会怎样；MoE；RoPE 外推（PI、NTK-aware、YaRN）；从 CPU 到 GPU | [`ch07_tiny_gpt.py`](code/ch07_tiny_gpt.py) |
| [第 8 章 预训练与后训练](ch08-pre-post-training.md) | 训练阶段全貌；6ND 与 scaling law；训练显存账本、mixed precision、ZeRO、gradient checkpointing；SFT；LoRA 与 QLoRA；RLHF、PPO；DPO 的推导；GRPO 与 verifiable reward；蒸馏；什么时候该微调 | [`ch08_post_training.py`](code/ch08_post_training.py) |

**第三部分 工程实践**

| 章 | 内容 | 代码 |
|---|---|---|
| [第 9 章 Prompt engineering](ch09-prompt-engineering.md) | 提示就是条件；chat template；写清楚、给例子、让模型先想；解码参数；constrained decoding；context engineering | [`ch09_prompting.py`](code/ch09_prompting.py) |
| [第 10 章 Tool calling](ch10-tool-calling.md) | 模型只写出调用请求；工具定义与设计；`tool_use` / `tool_result`；错误处理；并行调用；严格模式 | [`ch10_tool_calling.py`](code/ch10_tool_calling.py) |
| [第 11 章 Agent 与 harness](ch11-agent-harness.md) | workflow 还是 agent；agent 循环；context management 与压缩；权限、sandbox 与 hooks；prompt caching 与成本 | [`ch11_agent_harness.py`](code/ch11_agent_harness.py) |
| [第 12 章 MCP](ch12-mcp.md) | host / client / server；tools、resources、prompts；JSON-RPC 与无状态设计；stdio 与 Streamable HTTP；安全 | [`ch12_mcp_server.py`](code/ch12_mcp_server.py)、[`ch12_mcp_client.py`](code/ch12_mcp_client.py) |
| [第 13 章 Agent Skills](ch13-skills.md) | SKILL.md；progressive disclosure；description 决定触发；与 MCP、工具、subagent 的分工 | [`ch13_skills.py`](code/ch13_skills.py) |
| [第 14 章 LangChain 与 LangGraph](ch14-langchain.md) | 统一接口与 LCEL；`create_agent` 与 middleware；state graph、检查点与人工介入；要不要用框架 | [`ch14_langchain.py`](code/ch14_langchain.py) |
| [第 15 章 Retrieval-augmented generation（RAG）](ch15-rag.md) | 何时用 RAG；BM25 与 embedding；hybrid search、rerank、contextual chunking；评测检索 | [`ch15_rag.py`](code/ch15_rag.py) |
| [第 16 章 评测、安全与上线](ch16-evals-production.md) | 评测的组成；误差累积；pass@k 与 pass^k；置信区间与配对比较；LLM-as-a-judge；prompt injection；成本与延迟 | [`ch16_evals.py`](code/ch16_evals.py) |

**第四部分 系统与基础设施**

| 章 | 内容 | 代码 |
|---|---|---|
| [第 17 章 推理服务](ch17-inference-serving.md) | prefill 与 decode；显存账本与 KV cache；roofline；continuous batching、PagedAttention、prefix caching；speculative decoding（含证明）；量化；多卡并行；排队论与容量规划 | [`ch17_inference_serving.py`](code/ch17_inference_serving.py)、[`ch17_gpu_benchmark.py`](code/ch17_gpu_benchmark.py)（在 GPU 上运行） |

**第五部分 实战项目**

| 章 | 内容 | 代码 |
|---|---|---|
| [第 18 章 实战项目：设计](ch18-project-design.md) | 读书与面试陪练助手：自定义 SSE 与流式前端；模型网关；问书用 workflow、面试用 agent；记忆与间隔复习；用附录 C 做 eval set；成本、延迟与安全；里程碑 | [`ch18_eval_baseline.py`](code/ch18_eval_baseline.py) |

**附录**

| | 内容 | 代码 |
|---|---|---|
| [附录 A 数学速查](appendix-a-math-quick-reference.md) | 一页讲完第 5、6 章要用的结论：链式法则；熵、交叉熵、KL；最大似然；矩阵乘法；softmax；梯度下降；反向传播 | [`appendix_a_quick_reference.py`](code/appendix_a_quick_reference.py) |
| [附录 C 面试题清单](appendix-c-interview-questions.md) | 中英文面试题 366 道，按主题归类，每题标出本书哪一节能回答；最后汇总空白和补课顺序 | 无 |

## 每章的结构

- **导读**：这一章要回答什么问题，和大模型有什么关系，需要哪些前置知识。
- **正文各节**（如 1.1、1.2……）：定义、推导和例子。节标题旁的标签说明可信程度：**[已证]** 给出了推导，**[事实]** 是标准结论，**[直觉]** 帮助理解、不严格，**[经验]** 是工程实践中总结的做法。
- **数值实验**：用配套代码验证正文的结论，给出实际输出。
- **小结与疑问**，以及**参考资料**。

**术语**：工程和 AI 领域里大家平时就说英文的词，保留英文（prompt、tool calling、embedding、KV cache、speculative decoding、reward model……）；数学概念和已经通行的中文词用中文（概率、梯度、交叉熵、微调、预训练、蒸馏、量化）。

## 推荐的学习顺序（AI 工程岗）

这条路线是为"有扎实的数学基础、做过多年前端、目标是 AI 工程师 / AI 全栈"设计的。按每周 10 小时左右，大约 12 周。原则有三条：数学只补用得到的部分；尽早把模型从头写一遍；项目和读书并行，而不是读完再做。

| 阶段 | 时间 | 读什么 | 做什么 | 这一阶段结束时应该能 |
|---|---|---|---|---|
| 1 数学速通 | 第 1 周 | [附录 A](appendix-a-math-quick-reference.md)；第 1–4 章只读 [1.7](ch01-probability.md#1.7)、[2.2](ch02-statistics.md#2.2)、[2.5](ch02-statistics.md#2.5)、[3.5](ch03-linear-algebra.md#3.5)、[3.7](ch03-linear-algebra.md#3.7)、[4.3](ch04-calculus-optimization.md#4.3)、[4.7](ch04-calculus-optimization.md#4.7)、[4.8](ch04-calculus-optimization.md#4.8) | 跑附录 A 和第 4 章的代码 | 说清楚"最大似然 = 交叉熵 = KL"、softmax 的梯度、Adam |
| 2 模型本身 | 第 2–4 周 | [第 5 章](ch05-language-model.md) → [第 6 章](ch06-self-attention.md) → [第 7 章](ch07-tiny-gpt.md) | 跑第 7 章的代码；自己改一处再跑（比如把 GQA 改回普通 multi-head attention、去掉 RoPE），看 loss 和 gradient check | 不看书手写 attention、KV cache、RMSNorm、采样；估算任意模型的参数量和 KV cache |
| 3 应用核心 | 第 5–7 周 | [第 9 章](ch09-prompt-engineering.md) → [第 10 章](ch10-tool-calling.md) → [第 11 章](ch11-agent-harness.md) → [第 15 章](ch15-rag.md) → [第 16 章](ch16-evals-production.md) | 同时开始 [第 18 章](ch18-project-design.md) 的 M0、M1：骨架、模型网关、流式前端、问书 | 讲清楚 agent 循环和 harness；RAG 的每一步和评测；项目能流式问答、带出处 |
| 4 生态工具 | 第 6–8 周，随项目需要穿插 | [第 12 章](ch12-mcp.md)、[第 13 章](ch13-skills.md)、[第 14 章](ch14-langchain.md) | 项目 M2：面试官 agent | 说清楚 MCP 和 tool calling、Skills 和 MCP、要不要用框架的取舍 |
| 5 系统与训练 | 第 9–11 周 | [第 17 章](ch17-inference-serving.md) → [第 8 章](ch08-pre-post-training.md) | 项目 M3、M4：记忆、检索优化、评测进 CI；有空在 Kaggle 上跑 17.13 和 8.12 的 GPU 实验 | 用 roofline 和排队论估算服务容量；推一遍 DPO；回答"prompt、RAG、微调怎么选" |
| 6 冲刺 | 第 12 周起 | [附录 C](appendix-c-interview-questions.md) 全部 ★ 题 | 项目上线；每周两道 [C.11](appendix-c-interview-questions.md#C.11) 系统设计题 | 用自己的项目回答"介绍一个你做过的项目"，每个设计决定都有数字 |

**为什么是这个顺序**：
- **第 7 章放在很前面**：后面几乎每一章都默认你知道模型内部在算什么。KV cache、prefill 和 decode、显存账本、LoRA 加在哪些层，都建立在它上面；
- **第 8 章（训练）放在后面**：AI 工程岗很少要求亲手训练模型，但面试会问概念。它的数学（KL、策略梯度、Gibbs 分布）对有概率背景的人很快，等项目跑起来再读，更能体会"什么时候该微调"；
- **第 17 章在第 8 章之前**：推理服务直接决定线上的延迟和成本，系统设计题里出现得比训练多得多；
- **项目从第 5 周开始**：读到 RAG 和评测时就动手，读到的东西立刻在项目里用上，也立刻暴露出没读懂的地方。

**每一章怎么读**：
1. 读正文，推导亲手推一遍；
2. 跑配套代码，对照"数值实验"一节的输出；
3. 改一个参数或一处实现再跑，预测结果之后再看；
4. 合上书，用自己的话回答附录 C 里链接到这一章的 ★ 题，再请 Claude 按面试的方式追问三层（原理、数字、线上会坏在哪）。

## 怎么读

- **打地基**：按顺序读第 1–4 章，再读第 5、6 章。每章约一小时，读完跑一下配套代码，对照"数值实验"一节看输出。
- **准备面试**：先看附录 C，按岗位挑出对应的类别，再顺着每道题后面的链接回到正文。
- **快速入门**：有概率和线性代数基础的话，先读附录 A，然后直接读第 5、6 章，遇到不懂的再回到第 1–4 章对应的小节。
- **关心工程**：第 9–18 章可以直接读，遇到数学时按链接回查第 1–6 章。
- 代码进入 `code/` 目录后 `python3 文件名.py` 直接运行。第 1–11、13、15、16 章只依赖 numpy 或标准库；第 18 章另外可选 `pip install jieba`；第 12 章的互操作部分需要 `pip install "mcp>=2"`（没装会自动跳过）；第 14 章需要 `pip install "langchain>=1" "langgraph>=1"`。都不需要 API key。第 17 章的 `ch17_gpu_benchmark.py` 需要 GPU（Kaggle / Colab 的免费 T4 即可）。
