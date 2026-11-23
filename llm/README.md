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
| 第 7 章 动手训练一个小 GPT（待写） | 把前六章组装起来，用反向传播真正训练，看损失下降、生成变好 | 计划跟 nanoGPT |
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

## 怎么读

- **打地基**：按顺序读第 1–4 章，再读第 5、6 章。每章约一小时，读完跑一下配套代码，对照"数值实验"一节看输出。
- **准备面试**：先看附录 C，按岗位挑出对应的类别，再顺着每道题后面的链接回到正文。
- **快速入门**：有概率和线性代数基础的话，先读附录 A，然后直接读第 5、6 章，遇到不懂的再回到第 1–4 章对应的小节。
- **关心工程**：第 9–18 章可以直接读，遇到数学时按链接回查第 1–6 章。
- 代码进入 `code/` 目录后 `python3 文件名.py` 直接运行。第 1–6、8–11、13、15、16 章只依赖 numpy 或标准库；第 18 章另外可选 `pip install jieba`；第 12 章的互操作部分需要 `pip install "mcp>=2"`（没装会自动跳过）；第 14 章需要 `pip install "langchain>=1" "langgraph>=1"`。都不需要 API key。第 17 章的 `ch17_gpu_benchmark.py` 需要 GPU（Kaggle / Colab 的免费 T4 即可）。
