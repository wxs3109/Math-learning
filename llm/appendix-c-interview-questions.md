# 附录 C 面试题清单

> 标注：题目来自 2025–2026 年公开的中英文面试题库、面经和招聘方文章，我把措辞统一改写过，去掉了重复。每道题后面标出本书哪一节能回答它。
> 时间：整理于 2026 年 11 月。

## 导读

**这份清单是什么**：把中文面经（CSDN、掘金、阿里云开发者社区、GitHub 上的大模型面试笔记）和英文题库（DataCamp、InterviewBit、GitHub 上的 AI engineering 题库、Exponent 与宾大职业中心整理的大厂真题、techinterview.org 等）里出现过的题目放在一起，按主题归类。同一个意思的题只留一条。

**与大模型的关系**：面试题是一张"行业认为你应该懂什么"的地图。把它和本书的目录对一遍，就知道哪些已经学过、哪些还是空白。C.15 节汇总了空白，并对应到后续要写的章节。

**前置知识**：无。这一附录可以随时翻。

## C.1 怎么用这份清单 **[经验]**

**标记**：

- ★ 高频：在三个以上来源里出现，或者多篇面经都说"必问"；
- ✅ 本书已经讲到，后面是对应的节；
- ◐ 本书只讲了一部分，或者只提到没展开；
- ○ 本书还没有覆盖，括号里是计划补到哪里。

**按岗位挑着看** **[经验]**：从面经看，不同岗位的侧重差别很大。

| 岗位 | 重点 | 本清单中对应的节 |
|---|---|---|
| 大模型应用 / AI 工程师 | RAG、agent、工具调用、评测、成本、系统设计 | C.6–C.11、C.13 |
| 大模型算法岗 | Transformer 细节、训练、微调、对齐、手撕代码 | C.2–C.4、C.12 |
| 推理 / AI infra | KV cache、批处理、量化、并行、框架对比 | C.3.2、C.5、C.11 |
| AI 前端 / 全栈 | 流式输出、SSE、中断、长对话渲染、异步后端 | C.10，以及 C.8 的工程化部分 |

**怎么答** **[经验]**：几篇面经的共同观察是，第一问大家都能背，难在追问。面试官常沿三条线往下追：

1. **原理**：为什么这样设计？换一种会怎样？（例如"为什么只缓存 K 和 V，不缓存 Q"）
2. **数字**：能不能当场估算？（例如"7B 模型全参微调要多少显存""一个 token 的 KV cache 多大"）
3. **工程**：上线后哪里会坏，怎么发现、怎么修？（例如"rerank 太慢能不能省掉"）

所以准备每道题时，最好能讲出一个公式或一个数字，再讲出一个线上会遇到的坑。

## C.2 语言模型与 Transformer 基础 **[经验]**

### C.2.1 语言模型与生成

1. ★ 什么是大语言模型？它的训练目标是什么？ —— ✅ [5.1 节](ch05-language-model.md#5.1)、[5.2 节](ch05-language-model.md#5.2)
2. 自回归语言模型和掩码语言模型（BERT）有什么区别？ —— ◐ [5.1 节](ch05-language-model.md#5.1)
3. ★ 为什么最大似然就是最小化交叉熵？交叉熵、KL、熵之间是什么关系？ —— ✅ [1.7 节](ch01-probability.md#1.7)、[5.2 节](ch05-language-model.md#5.2)、[A.2 节](appendix-a-math-quick-reference.md#A.2)
4. 困惑度是什么，怎么从交叉熵算出来？ —— ✅ [5.2 节](ch05-language-model.md#5.2)
5. logits 是什么，在生成中起什么作用？ —— ✅ [9.6 节](ch09-prompt-engineering.md#9.6)、[A.5 节](appendix-a-math-quick-reference.md#A.5)
6. ★ 贪心、beam search、top-k、top-p、温度各有什么取舍？什么时候把温度调到接近 0？ —— ◐ [9.6 节](ch09-prompt-engineering.md#9.6)（beam search 未讲）
7. 你按下回车之后，ChatGPT 内部发生了什么？（从分词到流式返回） —— ◐ [9.2 节](ch09-prompt-engineering.md#9.2)、[17.1 节](ch17-inference-serving.md#17.1)
8. 为什么第一个 token 出来得慢，后面的快？ —— ✅ [17.1 节](ch17-inference-serving.md#17.1)
9. stop token 和 stop sequence 有什么区别？ —— ○（计划补到第 9 章）
10. 长输出里出现"复读"的原因是什么，怎么缓解？ —— ○（计划补到第 9 章：重复惩罚、采样）
11. 涌现能力是什么？有人认为它是度量方式造成的假象，你怎么看？ —— ○（计划第 8 章）
12. RNN 有什么局限，Transformer 怎样解决？ —— ◐ [5.4 节](ch05-language-model.md#5.4)

### C.2.2 Tokenizer 与 embedding

1. ★ token 是什么？token 数为什么同时影响费用和上下文上限？ —— ◐ [5.1 节](ch05-language-model.md#5.1)、[16.8 节](ch16-evals-production.md#16.8)
2. ★ BPE、WordPiece、SentencePiece、Unigram 有什么区别？ —— ○（计划第 7 章）
3. 手写 BPE 的训练和编码过程。 —— ○（计划第 7 章）
4. 分词器把专业术语切成没有意义的碎片，会带来什么问题，怎么办？ —— ○（计划第 7 章）
5. 中文分词和英文分词有什么不同？词表大小怎么定？ —— ○（计划第 7 章）
6. embedding 是什么？语义相似在向量上是什么意思？ —— ✅ [5.4 节](ch05-language-model.md#5.4)、[15.4 节](ch15-rag.md#15.4)
7. 余弦相似度、点积、欧氏距离的区别是什么？归一化之后有什么关系？ —— ✅ [3.1 节](ch03-linear-algebra.md#3.1)、[15.4 节](ch15-rag.md#15.4)
8. 为什么高维空间里随机向量几乎正交？这对检索意味着什么？ —— ✅ [3.7 节](ch03-linear-algebra.md#3.7)

### C.2.3 注意力与位置编码

1. ★ Q、K、V 分别是什么？为什么要用三个不同的矩阵？ —— ✅ [6.2 节](ch06-self-attention.md#6.2)
2. ★ 为什么注意力分数要除以 $\sqrt{d_k}$？不除会怎样？ —— ✅ [6.2 节](ch06-self-attention.md#6.2)、[6.5.2 节](ch06-self-attention.md#6.5.2)
3. ★ 多头注意力"多"在哪里？参数量变了吗？ —— ✅ [6.2 节](ch06-self-attention.md#6.2)
4. 因果掩码加在哪里？为什么在 softmax 之前加 $-\infty$，而不是之后置零？ —— ✅ [6.2 节](ch06-self-attention.md#6.2)、[6.5.1 节](ch06-self-attention.md#6.5.1)
5. 自注意力和交叉注意力有什么区别？ —— ◐ [6.2 节](ch06-self-attention.md#6.2)
6. 注意力的时间和空间复杂度是多少？长文本时怎么降？ —— ◐ [6.4 节](ch06-self-attention.md#6.4)、[17.9 节](ch17-inference-serving.md#17.9)
7. ★ MHA、MQA、GQA、MLA 的区别是什么？为什么它们都围绕 KV cache 展开？ —— ✅ [17.9 节](ch17-inference-serving.md#17.9)
8. ★ 为什么需要位置编码？正弦位置编码、可学习位置编码、RoPE 各有什么特点？ —— ◐ [6.1 节](ch06-self-attention.md#6.1)（RoPE 只提到）
9. ★ RoPE 的原理是什么？为什么它能表达相对位置？ —— ○（计划第 7 章）
10. 长上下文外推：位置插值、NTK-aware、YaRN 分别做了什么？ —— ○（计划第 8 章）
11. 滑动窗口注意力、attention sink 是什么？ —— ○（计划第 17 章补充）
12. FlashAttention 为什么快？它减少了计算还是减少了显存读写？ —— ○（计划第 17 章补充）

### C.2.4 Transformer 块与架构变体

1. ★ 画出一个 decoder-only Transformer 块，讲清楚一次前向传播的每一步和张量形状。 —— ✅ [6.3 节](ch06-self-attention.md#6.3)、[3.8 节](ch03-linear-algebra.md#3.8)
2. ★ encoder-only、decoder-only、encoder-decoder 的区别？为什么现在的大模型几乎都是 decoder-only？ —— ◐ [6.3 节](ch06-self-attention.md#6.3)
3. ★ LayerNorm、RMSNorm、BatchNorm 的区别？为什么大模型不用 BatchNorm？ —— ○（计划第 7 章）
4. ★ Pre-Norm 和 Post-Norm 有什么区别？各自的训练稳定性如何？ —— ◐ [6.3 节](ch06-self-attention.md#6.3)
5. 残差连接为什么让深层网络训练得动？ —— ✅ [6.3 节](ch06-self-attention.md#6.3)、[4.8 节](ch04-calculus-optimization.md#4.8)
6. FFN 在 Transformer 里起什么作用？SwiGLU 相比 ReLU 有什么好处？ —— ◐ [6.3 节](ch06-self-attention.md#6.3)（SwiGLU 未讲）
7. 给定层数、隐藏维度、词表大小，估算模型参数量。 —— ◐ [17.2 节](ch17-inference-serving.md#17.2)（计划第 7 章给出完整推导）
8. ★ MoE（混合专家）是什么？稠密模型和稀疏模型怎么比较？负载均衡怎么做？ —— ○（计划第 8 章）
9. 介绍 DeepSeek 系列模型在架构上的改动（MLA、MoE、多 token 预测）。 —— ◐ [17.9 节](ch17-inference-serving.md#17.9)
10. 扩散语言模型和自回归模型有什么不同？ —— ○
11. Transformer 能理解图像吗？视觉 token 是怎么来的？ —— ○（多模态，暂未计划）

## C.3 预训练、分布式训练与微调 **[经验]**

### C.3.1 预训练与 scaling

1. ★ 大模型的训练通常分哪几个阶段？每个阶段学到什么？ —— ○（计划第 8 章与后训练章）
2. Chinchilla scaling law 讲了什么？参数量和数据量该怎么配比？ —— ○（计划第 8 章）
3. 预训练数据怎么清洗、去重、配比？ —— ○（计划后训练章）
4. 继续预训练（continual pre-training）和微调有什么区别？什么时候用？ —— ○（计划后训练章）
5. 合成数据怎么生成和验证？ —— ○（计划后训练章）
6. 知识蒸馏是什么？蒸馏后的小模型推理能力差，怎么补？ —— ○（计划后训练章）

### C.3.2 显存与分布式训练

1. ★ 训练时显存被哪些东西占用？给出模型状态和激活值的估算。 —— ○（计划后训练章）
2. ★ 用 Adam 做混合精度训练，每个参数大约占多少字节？7B 模型全参微调大约要多少显存？ —— ◐ [4.7 节](ch04-calculus-optimization.md#4.7)（计划后训练章给出完整账本）
3. ★ 混合精度训练是怎么做的？FP16 和 BF16 有什么区别？为什么需要 loss scaling？ —— ○（计划后训练章）
4. ★ ZeRO 的三个阶段分别切分什么？通信量怎么变？ —— ○（计划后训练章）
5. FSDP 和 DeepSpeed ZeRO 有什么区别？ —— ○（计划后训练章）
6. ★ 数据并行、张量并行、流水线并行、3D 并行分别适合什么场景？ —— ◐ [17.10 节](ch17-inference-serving.md#17.10)（推理侧）
7. 梯度检查点（重计算）用时间换了多少显存？ —— ○（计划后训练章）
8. 梯度累积的作用是什么？和直接加大 batch 等价吗？ —— ◐ [4.6 节](ch04-calculus-optimization.md#4.6)
9. 多机训练时网络带宽不够会怎样？单个节点挂掉怎么办？ —— ○（计划后训练章）
10. 学习率预热和余弦衰减为什么有用？Adam 和 AdamW 的区别？ —— ✅ [4.7 节](ch04-calculus-optimization.md#4.7)

### C.3.3 SFT 与参数高效微调

1. ★ 什么时候该微调，什么时候用提示或 RAG 就够了？ —— ✅ [15.1 节](ch15-rag.md#15.1)、[9.8 节](ch09-prompt-engineering.md#9.8)
2. ★ 全参微调和参数高效微调（PEFT）的区别？ —— ○（计划后训练章）
3. ★ LoRA 的原理是什么？为什么可以用低秩分解？秩怎么选、加在哪些层？ —— ◐ [3.5 节](ch03-linear-algebra.md#3.5)（数学部分），○ 训练细节（计划后训练章）
4. ★ QLoRA 做了什么，为什么能在消费级显卡上微调？ —— ○（计划后训练章）
5. Prefix tuning、prompt tuning、adapter 和 LoRA 怎么比较？ —— ○（计划后训练章）
6. 多个 LoRA 适配器怎么合并？推理时怎么同时服务很多个 LoRA？ —— ○（计划后训练章）
7. 指令微调是什么？和普通的任务微调有什么不同？ —— ○（计划后训练章）
8. 微调数据怎么准备？需要多少条？质量和数量哪个更重要？ —— ○（计划后训练章）
9. ★ 微调后通用能力下降（灾难性遗忘），怎么处理？ —— ○（计划后训练章）
10. 微调时过拟合、模型背下训练数据，怎么发现、怎么避免？ —— ◐ [2.5 节](ch02-statistics.md#2.5)、[2.7 节](ch02-statistics.md#2.7)
11. 有人想用微调让模型记住新出的政策条文，你怎么建议？ —— ✅ [15.1 节](ch15-rag.md#15.1)
12. 微调后的模型怎么评测？除了指标还看什么？ —— ◐ [16.1 节](ch16-evals-production.md#16.1)
13. 关键超参数（学习率、epoch、batch、LoRA 秩）怎么调？ —— ○（计划后训练章）

## C.4 对齐与强化学习 **[经验]**

1. ★ 讲一遍 RLHF 的完整流程：SFT、奖励模型、PPO。每一步可能在哪里失败？ —— ○（计划后训练章）
2. 奖励模型怎么训练？为什么只有分数之差有意义（Bradley–Terry）？ —— ○（计划后训练章）
3. ★ PPO 的裁剪目标在防什么？为什么外面还要套一个 min？ —— ○（计划后训练章）
4. ★ 为什么要加 KL 惩罚？它和参考模型的关系是什么？ —— ◐ [1.7 节](ch01-probability.md#1.7)（KL 本身）
5. ★ DPO 是怎么从 RLHF 的目标推出来的？为什么不需要显式奖励模型？ —— ○（计划后训练章）
6. ★ DPO 和 PPO 各自什么时候用？ —— ○（计划后训练章）
7. DPO 训练后输出越来越长，是什么原因？怎么解决？ —— ○（计划后训练章）
8. ★ GRPO 是什么？它的优势函数怎么算？为什么可以不要价值网络？ —— ○（计划后训练章）
9. DAPO、GSPO、Dr. GRPO 分别修正了 GRPO 的什么问题？ —— ○（计划后训练章）
10. ★ 奖励作弊（reward hacking）是什么？实际中怎么发现和修？ —— ○（计划后训练章）
11. RLVR（可验证奖励）是什么？什么时候可以不用奖励模型？ —— ○（计划后训练章）
12. RLAIF、Constitutional AI 和 RLHF 有什么区别？ —— ○（计划后训练章）
13. 对齐税（alignment tax）是什么？ —— ○（计划后训练章）
14. 推理能力是在哪个训练阶段出现的？RL 是扩大了能力边界，还是只把已有能力变得更稳定？ —— ○（计划后训练章）
15. 拒绝采样微调和推理时的 best-of-N 有什么区别？ —— ○（计划后训练章）
16. 一次 GRPO 训练先涨后崩，你会看哪些日志？ —— ○（计划后训练章）
17. 多轮工具调用的 agentic RL，为什么信用分配更难？ —— ○（计划 agent 深入章）
18. 偏好数据的质量怎么保证？标注者意见不一致怎么办？ —— ◐ [16.5 节](ch16-evals-production.md#16.5)
19. 重要性采样在 PPO 里起什么作用？ —— ○（计划后训练章）
20. 手写 DPO 损失和 PPO 裁剪损失。 —— ○（计划后训练章）

## C.5 推理与部署 **[经验]**

### C.5.1 KV cache 与性能模型

1. ★ KV cache 是什么？为什么能加速？为什么只缓存 K 和 V，不缓存 Q？ —— ✅ [17.1 节](ch17-inference-serving.md#17.1)
2. ★ 估算一个模型每个 token 的 KV cache 大小，以及服务 70B 模型要多少显存。 —— ✅ [17.2 节](ch17-inference-serving.md#17.2)、[17.12.1 节](ch17-inference-serving.md#17.12.1)
3. ★ prefill 和 decode 有什么区别？为什么 prefill 受限于算力、decode 受限于带宽？ —— ✅ [17.1 节](ch17-inference-serving.md#17.1)、[17.3 节](ch17-inference-serving.md#17.3)
4. 用 roofline 估算 70B 模型在 H100 上的 decode 速度上限。 —— ✅ [17.3 节](ch17-inference-serving.md#17.3)
5. 长对话把 KV cache 撑爆、显存溢出，怎么办？ —— ✅ [17.5 节](ch17-inference-serving.md#17.5)、[17.9 节](ch17-inference-serving.md#17.9)
6. KV cache 压缩与驱逐（H2O、SnapKV、StreamingLLM）分别是什么思路？ —— ◐ [17.9 节](ch17-inference-serving.md#17.9)

### C.5.2 批处理与调度

1. ★ 静态批处理和连续批处理（continuous batching）有什么区别？ —— ✅ [17.4 节](ch17-inference-serving.md#17.4)、[17.12.4 节](ch17-inference-serving.md#17.12.4)
2. ★ PagedAttention 解决了什么问题？它借鉴了操作系统的什么机制？ —— ✅ [17.5 节](ch17-inference-serving.md#17.5)
3. ★ 前缀缓存（prefix caching / prompt caching）的原理？怎样设计提示才能命中？ —— ✅ [17.5 节](ch17-inference-serving.md#17.5)、[11.6 节](ch11-agent-harness.md#11.6)
4. chunked prefill 为什么能改善尾延迟？ —— ✅ [17.6 节](ch17-inference-serving.md#17.6)
5. prefill 和 decode 分离部署（disaggregation）有什么好处和代价？ —— ✅ [17.6 节](ch17-inference-serving.md#17.6)
6. ★ 设计一个单卡上的推理批处理系统，同步用户，每批最多 100 个请求。（Anthropic） —— ✅ [17.4 节](ch17-inference-serving.md#17.4)、[17.11 节](ch17-inference-serving.md#17.11)
7. 设计一个端到端的 LLM 请求批处理系统，包括排队和调度。（Anthropic） —— ✅ [17.4 节](ch17-inference-serving.md#17.4)、[17.11 节](ch17-inference-serving.md#17.11)
8. 请求优先级和公平调度怎么做？ —— ◐ [17.11 节](ch17-inference-serving.md#17.11)

### C.5.3 加速：投机解码、量化

1. ★ 投机解码的原理是什么？为什么输出分布不变？接受率怎么算？ —— ✅ [17.7 节](ch17-inference-serving.md#17.7)
2. Medusa、EAGLE、n-gram 投机分别怎么产生草稿？ —— ◐ [17.7 节](ch17-inference-serving.md#17.7)
3. ★ INT8、INT4、FP8、BF16 各有什么取舍？量化损失了什么？ —— ✅ [17.8 节](ch17-inference-serving.md#17.8)
4. ★ 权重量化和激活量化有什么区别？为什么激活更难量化（离群值）？ —— ✅ [17.8 节](ch17-inference-serving.md#17.8)
5. GPTQ、AWQ、SmoothQuant 分别怎么做？ —— ✅ [17.8 节](ch17-inference-serving.md#17.8)
6. PTQ 和 QAT 的区别？低比特下为什么会突然崩？ —— ◐ [17.8 节](ch17-inference-serving.md#17.8)
7. GGUF 是什么？端侧推理怎么优化？ —— ○（计划第 17 章补充）

### C.5.4 框架、并行与部署

1. ★ vLLM、SGLang、TensorRT-LLM、llama.cpp、Ollama 怎么选？ —— ◐ [17.5 节](ch17-inference-serving.md#17.5)（计划第 17 章补充对比表）
2. SGLang 的 RadixAttention 和 vLLM 的 PagedAttention 在前缀复用上有什么差别？ —— ◐ [17.5 节](ch17-inference-serving.md#17.5)
3. ★ 张量并行和流水线并行在推理时怎么用？通信发生在哪里？ —— ✅ [17.10 节](ch17-inference-serving.md#17.10)
4. ★ TTFT、TPOT、吞吐之间是什么关系？在线服务优先看吞吐还是延迟？ —— ✅ [17.11 节](ch17-inference-serving.md#17.11)
5. 用排队论估算需要几张卡，为什么利用率接近 100% 时延迟会爆炸？ —— ✅ [17.11 节](ch17-inference-serving.md#17.11)、[17.12.9 节](ch17-inference-serving.md#17.12.9)
6. 自建推理和调用 API 的成本怎么比较？ —— ◐ [16.8 节](ch16-evals-production.md#16.8)、[17.11 节](ch17-inference-serving.md#17.11)
7. Serverless 推理的冷启动怎么缓解？多模型如何共享显存？ —— ○
8. CPU/GPU 混合推理和 offloading 的代价是什么？ —— ○
9. 同步推理和异步推理（批量离线任务）的架构有什么不同？ —— ◐ [17.4 节](ch17-inference-serving.md#17.4)
10. 实现一个 GPU 额度管理系统，事件里有发放、使用和过期。（OpenAI） —— ○（编码题，见 C.12）

## C.6 提示工程与结构化输出 **[经验]**

1. ★ 提示工程是什么？它在数学上改变了什么？ —— ✅ [9.1 节](ch09-prompt-engineering.md#9.1)
2. 系统提示、用户消息、助手消息分别起什么作用？聊天模板是什么？ —— ✅ [9.2 节](ch09-prompt-engineering.md#9.2)
3. ★ 零样本、少样本、思维链分别是什么？少样本例子怎么挑？ —— ✅ [9.4 节](ch09-prompt-engineering.md#9.4)、[9.5 节](ch09-prompt-engineering.md#9.5)
4. ★ 思维链为什么能提升推理？什么时候它反而没用？ —— ✅ [9.5 节](ch09-prompt-engineering.md#9.5)
5. self-consistency、tree-of-thought 分别是什么？和 CoT 怎么递进？ —— ◐ [9.5 节](ch09-prompt-engineering.md#9.5)
6. ★ 怎样让模型稳定输出合法 JSON？受约束解码是怎么做的？ —— ✅ [9.7 节](ch09-prompt-engineering.md#9.7)、[10.6 节](ch10-tool-calling.md#10.6)
7. 模型输出了不合法的 JSON，线上怎么兜底？ —— ✅ [10.5 节](ch10-tool-calling.md#10.5)、[9.7 节](ch09-prompt-engineering.md#9.7)
8. 角色提示有用吗？什么时候有用？ —— ✅ [9.3 节](ch09-prompt-engineering.md#9.3)
9. 提示链（prompt chaining）是什么？和 agent 有什么区别？ —— ✅ [11.1 节](ch11-agent-harness.md#11.1)
10. ★ "lost in the middle" 是什么？长上下文提示里信息该放在哪里？ —— ◐ [9.8 节](ch09-prompt-engineering.md#9.8)
11. 从提示工程到上下文工程，变的是什么？ —— ✅ [9.8 节](ch09-prompt-engineering.md#9.8)
12. 少样本结果不稳定、对措辞很敏感，怎么办？ —— ◐ [9.4 节](ch09-prompt-engineering.md#9.4)、[16.4 节](ch16-evals-production.md#16.4)
13. 怎么系统地评估和迭代一个提示？改提示或换模型后怎么防退化？ —— ✅ [16.1 节](ch16-evals-production.md#16.1)、[16.4 节](ch16-evals-production.md#16.4)
14. 怎样防止系统提示泄露？ —— ✅ [16.7 节](ch16-evals-production.md#16.7)
15. 怎样让模型在不知道时说"不知道"？怎样控制回答长度？ —— ◐ [9.3 节](ch09-prompt-engineering.md#9.3)
16. 提示工程和 prompt tuning（软提示）有什么区别？ —— ○（计划后训练章）
17. 提示的成本和延迟怎么优化？ —— ✅ [11.6 节](ch11-agent-harness.md#11.6)、[16.8 节](ch16-evals-production.md#16.8)
18. 多语言提示要注意什么？跨语言零样本迁移为什么会失败？ —— ○

## C.7 RAG 与向量检索 **[经验]**

### C.7.1 原理与流程

1. ★ RAG 是什么？它为什么能减少幻觉？ —— ✅ [15.1 节](ch15-rag.md#15.1)、[15.2 节](ch15-rag.md#15.2)
2. ★ 画出一个 RAG 系统的完整流程。 —— ✅ [15.2 节](ch15-rag.md#15.2)
3. ★ RAG、微调、长上下文三者怎么选？什么时候不该用 RAG？ —— ✅ [15.1 节](ch15-rag.md#15.1)
4. ★ 为什么要把长文档切块？不切会怎样？ —— ✅ [15.2 节](ch15-rag.md#15.2)
5. ★ 定长、递归、语义三种切块方式的区别？块大小和重叠比例怎么定？ —— ◐ [15.2 节](ch15-rag.md#15.2)
6. 小块检索准但上下文太碎，大块上下文完整但检索不准，怎么调和（父子块）？ —— ◐ [15.5 节](ch15-rag.md#15.5)
7. 含表格和复杂版式的 PDF 怎么解析？ —— ○（计划数据与检索基础设施章）

### C.7.2 检索、混合检索与重排

1. ★ 稠密向量和稀疏向量的区别？BM25 的原理？ —— ✅ [15.3 节](ch15-rag.md#15.3)、[15.4 节](ch15-rag.md#15.4)
2. ★ 混合检索两路分数量纲不同，怎么融合？RRF 的 k 怎么取？ —— ✅ [15.5 节](ch15-rag.md#15.5)
3. ★ 初召回之后为什么还要重排？cross-encoder、ColBERT、LLM 重排有什么区别？ —— ◐ [15.5 节](ch15-rag.md#15.5)（ColBERT 未讲）
4. 重排太慢，线上扛得住吗？能不能省掉？精排取多少条？ —— ◐ [15.5 节](ch15-rag.md#15.5)、[15.8 节](ch15-rag.md#15.8)
5. ★ embedding 模型怎么选？召回率低的常见原因有哪些？ —— ◐ [15.4 节](ch15-rag.md#15.4)
6. 上下文化检索（contextual retrieval）是什么？ —— ✅ [15.5 节](ch15-rag.md#15.5)、[15.8.3 节](ch15-rag.md#15.8.3)
7. ★ 向量数据库和传统数据库的区别？ANN 是什么？HNSW、IVF、暴力检索怎么取舍？ —— ◐ [15.4 节](ch15-rag.md#15.4)（计划数据与检索基础设施章）
8. 十亿级向量检索怎么做？向量库内存太大怎么压（量化、PQ）？ —— ○（计划数据与检索基础设施章）
9. 相似度很高但结果不相关，可能是什么原因？短查询的语义检索为什么难？ —— ◐ [15.4 节](ch15-rag.md#15.4)、[15.8.2 节](ch15-rag.md#15.8.2)
10. 元数据过滤怎么和向量检索结合？ —— ○（计划数据与检索基础设施章）

### C.7.3 查询改写与进阶 RAG

1. ★ HyDE 的原理是什么？假设文档写错了会不会把检索带偏？ —— ○（计划第 15 章补充）
2. 查询改写、多查询、查询分解、step-back 有什么区别？ —— ○（计划第 15 章补充）
3. 多轮对话里用户省略指代（"那它还能退吗"），怎么补全成可检索的问题？ —— ○（计划第 15 章补充）
4. 多跳问题怎么检索？ —— ○（计划第 15 章补充）
5. ★ Agentic RAG、Self-RAG、GraphRAG 和普通 RAG 有什么区别？ —— ◐ [11.1 节](ch11-agent-harness.md#11.1)（计划 agent 深入章）
6. 怎么把 RAG 和 agent 结合起来？ —— ◐ [11.4 节](ch11-agent-harness.md#11.4)、[15.2 节](ch15-rag.md#15.2)
7. 表格、SQL、图片怎么纳入 RAG？Text-to-SQL 怎么做？ —— ○（计划数据与检索基础设施章）

### C.7.4 生成、幻觉与评测

1. ★ 怎样在提示里划定边界，防止模型答出文档以外的内容？拒答阈值怎么定？ —— ◐ [15.2 节](ch15-rag.md#15.2)、[9.3 节](ch09-prompt-engineering.md#9.3)
2. ★ 怎么让模型给出引用？怎么保证引用是真的？ —— ○（计划第 15 章补充）
3. 上下文是对的，模型还是答错，可能是什么原因？ —— ◐ [15.6 节](ch15-rag.md#15.6)
4. 多个来源互相矛盾，怎么处理？ —— ○（计划第 15 章补充）
5. ★ RAG 怎么评测？检索和生成分别看什么指标（召回率、MRR、nDCG、忠实度、相关性）？ —— ✅ [15.6 节](ch15-rag.md#15.6)、[16.5 节](ch16-evals-production.md#16.5)
6. ★ 没有标注数据时，评测集从哪来？ —— ◐ [16.1 节](ch16-evals-production.md#16.1)
7. RAGAS 这类框架的指标是怎么算的？ —— ◐ [16.5 节](ch16-evals-production.md#16.5)
8. ★ 检索召回了错误的文档，或者正确文档明明在库里却没召回，你从哪里查起？ —— ✅ [15.6 节](ch15-rag.md#15.6)、[15.8.2 节](ch15-rag.md#15.8.2)

### C.7.5 工程

1. ★ 文档局部更新时，怎样增量索引而不是全量重新向量化？最容易踩什么坑？ —— ○（计划数据与检索基础设施章）
2. embedding 模型升级后维度变了、检索质量崩了，怎么迁移？ —— ○（计划数据与检索基础设施章）
3. ★ 企业 RAG 怎么做按用户的权限控制？一个用户在回答里看到了别的团队的文档，是哪里出了问题？ —— ◐ [15.7 节](ch15-rag.md#15.7)
4. 几百万篇文档时，怎么保证检索延迟？ —— ◐ [15.4 节](ch15-rag.md#15.4)
5. 知识库频繁更新，怎么做版本管理，怎么保证答案是最新的？ —— ○（计划数据与检索基础设施章）
6. 记忆和 RAG 知识库要不要放在同一个库？ —— ○（计划 agent 深入章）

## C.8 Agent、工具调用与协议 **[经验]**

### C.8.1 Agent 基础

1. ★ agent 和普通聊天机器人、单次 LLM 调用的本质区别是什么？ —— ✅ [11.1 节](ch11-agent-harness.md#11.1)、[11.2 节](ch11-agent-harness.md#11.2)
2. ★ agent 由哪些部分组成（模型、工具、记忆、规划）？ —— ✅ [11.2 节](ch11-agent-harness.md#11.2)、[11.3 节](ch11-agent-harness.md#11.3)
3. ★ ReAct 的流程、优点和缺点？Plan-and-Execute 改进了什么？ —— ◐ [11.2 节](ch11-agent-harness.md#11.2)（Plan-and-Execute 计划 agent 深入章）
4. workflow 和 agent 怎么选？什么时候简单的 RAG 流水线就够了？ —— ✅ [11.1 节](ch11-agent-harness.md#11.1)
5. ★ harness 是什么？在编程 agent 里，模型和 harness 各占多大比重？ —— ✅ [11.3 节](ch11-agent-harness.md#11.3)、[11.7 节](ch11-agent-harness.md#11.7)
6. 为什么多步 agent 的成功率会随步数快速下降？ —— ✅ [16.2 节](ch16-evals-production.md#16.2)
7. 反思（reflection）在 agent 里怎么用？ —— ○（计划 agent 深入章）
8. 写代码的 agent 和调用工具的 agent 有什么区别？ —— ◐ [11.7 节](ch11-agent-harness.md#11.7)
9. computer-use agent、浏览器 agent 的难点在哪里？ —— ○（计划 agent 深入章）
10. 让 LLM 当 agent 的核心有哪些局限？ —— ✅ [11.9 节](ch11-agent-harness.md#11.9)

### C.8.2 工具调用

1. ★ function calling 的底层原理是什么？模型是怎么学会调用工具的？ —— ✅ [10.1 节](ch10-tool-calling.md#10.1)
2. ★ 一个工具定义包含哪些字段？从模型决定调用到结果回到上下文，消息是怎么一条条走的？ —— ✅ [10.2 节](ch10-tool-calling.md#10.2)、[10.4 节](ch10-tool-calling.md#10.4)
3. ★ 怎样设计好用的工具？工具粒度该多细（一个万能 SQL 工具还是多个专用工具）？ —— ✅ [10.3 节](ch10-tool-calling.md#10.3)
4. 结构化输出和 function calling 有什么区别？ —— ✅ [10.6 节](ch10-tool-calling.md#10.6)
5. ★ 模型生成的参数格式错误或缺失，怎么处理？ —— ✅ [10.5 节](ch10-tool-calling.md#10.5)
6. 并行工具调用怎么实现？工具之间有依赖怎么办？ —— ✅ [10.4 节](ch10-tool-calling.md#10.4)
7. ★ 工具调用失败、超时、返回异常，怎么重试和降级？ —— ✅ [10.5 节](ch10-tool-calling.md#10.5)
8. 模型调用了错误的工具，参数还看起来很合理，怎么防？ —— ◐ [10.3 节](ch10-tool-calling.md#10.3)、[11.5 节](ch11-agent-harness.md#11.5)
9. 怎样防止模型臆造出不存在的工具能力？ —— ◐ [10.3 节](ch10-tool-calling.md#10.3)
10. 工具返回的结果很大，怎么交给模型？ —— ✅ [10.3 节](ch10-tool-calling.md#10.3)、[11.4 节](ch11-agent-harness.md#11.4)

### C.8.3 记忆与上下文

1. ★ agent 的记忆分哪几类？工作记忆、短期记忆、长期记忆分别怎么实现？ —— ◐ [11.4 节](ch11-agent-harness.md#11.4)（计划 agent 深入章）
2. ★ 上下文有限时，怎么设计分层记忆的写入、读取、压缩和过期？ —— ◐ [11.4 节](ch11-agent-harness.md#11.4)
3. 滑动窗口、摘要、向量检索三种记忆方案怎么比较？ —— ◐ [11.4 节](ch11-agent-harness.md#11.4)
4. 新旧记忆冲突时听谁的？"值不值得存"由规则判断还是模型判断？ —— ○（计划 agent 深入章）
5. 摘要会不会把细节合丢？检索记忆时相关性和时间远近怎么平衡？ —— ○（计划 agent 深入章）
6. ★ 什么是上下文工程？上下文压缩（compaction）怎么做？ —— ✅ [9.8 节](ch09-prompt-engineering.md#9.8)、[11.4 节](ch11-agent-harness.md#11.4)
7. 长期运行的 agent 会"漂移"，怎么处理？ —— ◐ [11.4 节](ch11-agent-harness.md#11.4)
8. 设计一个跨会话的个人助理记忆系统。 —— ○（计划 agent 深入章）

### C.8.4 多 agent

1. ★ 单 agent 和多 agent 怎么选？多 agent 什么时候会崩？ —— ◐ [11.4 节](ch11-agent-harness.md#11.4)（子 agent）
2. 层级、辩论、市场三种协作模式分别适合什么？ —— ○（计划 agent 深入章）
3. 子 agent 和 agent 团队有什么区别？ —— ◐ [11.4 节](ch11-agent-harness.md#11.4)
4. 多 agent 之间共享记忆还是各自独立？ —— ○（计划 agent 深入章）
5. 多 agent 的死锁、信息过载、目标冲突怎么解决？ —— ○（计划 agent 深入章）
6. 多 agent 的工具输出互相矛盾，怎么协调？ —— ○（计划 agent 深入章）

### C.8.5 MCP、A2A 与 Skills

1. ★ MCP 是什么？它解决了 function calling 的哪些痛点？ —— ✅ [12.1 节](ch12-mcp.md#12.1)
2. ★ MCP 里 host、client、server 各是什么角色？ —— ✅ [12.1 节](ch12-mcp.md#12.1)
3. ★ tool、resource、prompt 三种原语有什么区别？分别由谁决定何时调用？什么东西该做成 resource 而不是 tool？ —— ✅ [12.2 节](ch12-mcp.md#12.2)
4. MCP 和传统 function calling 是什么关系？ —— ✅ [12.6 节](ch12-mcp.md#12.6)
5. ★ stdio 和 Streamable HTTP 两种传输有什么区别？为什么 Streamable HTTP 取代了 SSE？威胁模型有什么不同？ —— ✅ [12.4 节](ch12-mcp.md#12.4)、[12.5 节](ch12-mcp.md#12.5)
6. ★ 工具描述里的指令是怎么到达模型的？工具投毒、rug pull 怎么防？ —— ✅ [12.5 节](ch12-mcp.md#12.5)
7. 服务器拿着用户的 OAuth token，怎么防止混淆代理（confused deputy）攻击？ —— ✅ [12.5 节](ch12-mcp.md#12.5)
8. 设计一个面向分析师 agent、连接内部数仓的 MCP 服务器：用共享账号还是按用户鉴权？大结果和错误怎么返回？ —— ◐ [12.2 节](ch12-mcp.md#12.2)、[12.5 节](ch12-mcp.md#12.5)、[10.3 节](ch10-tool-calling.md#10.3)
9. agent 反复调用 list_tables，延迟很高，怎么优化？ —— ◐ [12.3 节](ch12-mcp.md#12.3)（缓存提示）
10. A2A 协议是什么？和 MCP 是什么关系？ —— ○（计划 agent 深入章）
11. ★ Agent Skills 是什么？渐进式披露怎样节省上下文？和 MCP 怎么分工？ —— ✅ [13.1 节](ch13-skills.md#13.1)、[13.3 节](ch13-skills.md#13.3)、[13.5 节](ch13-skills.md#13.5)

### C.8.6 框架

1. ★ LangChain、LangGraph、LlamaIndex 有什么区别？怎么选？ —— ◐ [14.1 节](ch14-langchain.md#14.1)、[14.5 节](ch14-langchain.md#14.5)（LlamaIndex 未讲）
2. LangChain 的核心组件有哪些？链和 agent 有什么区别？ —— ✅ [14.2 节](ch14-langchain.md#14.2)
3. LangChain 1.x 的 `create_agent` 和中间件是怎么工作的？ —— ✅ [14.3 节](ch14-langchain.md#14.3)
4. LangGraph 的 StateGraph 是什么？检查点和人工中断怎么实现？ —— ✅ [14.4 节](ch14-langchain.md#14.4)
5. LCEL、输出解析器、文本切分器是做什么的？ —— ◐ [14.2 节](ch14-langchain.md#14.2)
6. AutoGen、CrewAI、MetaGPT 各有什么特点？ —— ○
7. 现有框架有什么缺点？如果你自己设计一个轻量的 agent 框架，会怎么做？ —— ✅ [14.5 节](ch14-langchain.md#14.5)、[11.3 节](ch11-agent-harness.md#11.3)

### C.8.7 工程化

1. ★ agent 陷入死循环，怎么检测和打断？ —— ✅ [11.2 节](ch11-agent-harness.md#11.2)、[14.3 节](ch14-langchain.md#14.3)
2. ★ 长任务的检查点和断点续跑怎么设计？ —— ✅ [14.4 节](ch14-langchain.md#14.4)、[11.4 节](ch11-agent-harness.md#11.4)
3. ★ 删库、执行命令这类高危操作怎么防？人在回路怎么设计？ —— ✅ [11.5 节](ch11-agent-harness.md#11.5)
4. 安全地执行模型生成的代码：沙箱怎么做？ —— ✅ [11.5 节](ch11-agent-harness.md#11.5)
5. ★ 长流程里的 token 成本怎么控制？每个任务的预算怎么强制执行？ —— ✅ [11.6 节](ch11-agent-harness.md#11.6)、[16.8 节](ch16-evals-production.md#16.8)
6. agent 的可观测性：日志、链路追踪、监控指标看什么？ —— ✅ [16.6 节](ch16-evals-production.md#16.6)
7. ★ 怎么评测一个 agent？SWE-bench 这类基准有什么局限？ —— ✅ [16.1 节](ch16-evals-production.md#16.1)、[16.3 节](ch16-evals-production.md#16.3)
8. agent 任务太慢，怎么提速？ —— ◐ [10.4 节](ch10-tool-calling.md#10.4)、[16.8 节](ch16-evals-production.md#16.8)

## C.9 评测、幻觉与安全 **[经验]**

### C.9.1 评测

1. ★ 为什么 LLM 评测难？没有唯一正确答案时怎么评？ —— ✅ [16.1 节](ch16-evals-production.md#16.1)
2. ★ LLM 当评委（LLM-as-a-judge）怎么用？有哪些失效方式（位置偏差、长度偏差、偏爱自家模型）？怎么加固？ —— ✅ [16.5 节](ch16-evals-production.md#16.5)、[16.9.4 节](ch16-evals-production.md#16.9.4)
3. ★ 改了提示或换了模型，怎么发现退化？托管模型更新后行为变了，什么能发现？ —— ✅ [16.4 节](ch16-evals-production.md#16.4)、[16.6 节](ch16-evals-production.md#16.6)
4. 怎么从统计上比较两个模型或两个提示？分数差多少才算显著？ —— ✅ [16.4 节](ch16-evals-production.md#16.4)、[2.3 节](ch02-statistics.md#2.3)
5. pass@k 和 pass^k 的区别？为什么上线更该看 pass^k？ —— ✅ [16.3 节](ch16-evals-production.md#16.3)
6. BLEU、ROUGE、BERTScore、G-Eval 分别是什么？为什么传统指标不够？ —— ○（计划评测工程章）
7. MMLU、HumanEval、GSM8K 分别测什么？基准污染是什么？ —— ○（计划评测工程章）
8. 基准分数变高了，用户体验却变差了，可能是为什么？ —— ◐ [16.1 节](ch16-evals-production.md#16.1)
9. 离线评测和线上评测（A/B 测试）怎么配合？ —— ◐ [16.6 节](ch16-evals-production.md#16.6)
10. 评审员之间意见不一致，怎么办？ —— ◐ [16.5 节](ch16-evals-production.md#16.5)
11. 怎样为一个 AI 功能定义成功指标？用户不用这个功能，你查什么？ —— ○（计划评测工程章）
12. 怎样从零搭一个评测框架和"黄金数据集"？ —— ◐ [16.1 节](ch16-evals-production.md#16.1)

### C.9.2 幻觉

1. ★ 幻觉是什么？为什么会产生？ —— ◐ [5.1 节](ch05-language-model.md#5.1)、[15.1 节](ch15-rag.md#15.1)
2. ★ 怎样检测和缓解幻觉？ —— ◐ [15.6 节](ch15-rag.md#15.6)、[16.5 节](ch16-evals-production.md#16.5)
3. 模型在高风险场景里自信地给出错误答案，你怎么调查和缓解？（Anthropic） —— ◐ [16.1 节](ch16-evals-production.md#16.1)、[16.6 节](ch16-evals-production.md#16.6)
4. 来源很弱时模型还很笃定，应该改哪里？ —— ◐ [15.2 节](ch15-rag.md#15.2)
5. agent 的幻觉主要出现在哪几步？ —— ◐ [16.2 节](ch16-evals-production.md#16.2)
6. 怎么向产品经理解释"边缘情况下 15% 的幻觉率"，以及为什么 AI 不可能 100% 准确？ —— ◐ [16.4 节](ch16-evals-production.md#16.4)

### C.9.3 安全

1. ★ 提示注入的原理是什么？直接注入和间接注入有什么区别？ —— ✅ [16.7 节](ch16-evals-production.md#16.7)
2. ★ 在架构层面怎样防提示注入和数据外泄？ —— ✅ [16.7 节](ch16-evals-production.md#16.7)、[16.9.5 节](ch16-evals-production.md#16.9.5)、[11.5 节](ch11-agent-harness.md#11.5)
3. 越狱有哪些常见手法？红队测试怎么做？ —— ○（计划安全章）
4. 输入和输出护栏（guardrails）怎么设计？PII 怎么处理？ —— ◐ [16.7 节](ch16-evals-production.md#16.7)（计划安全章）
5. 怎样防止模型输出训练数据中记住的隐私或受版权保护的内容？ —— ○（计划安全章）
6. 数据投毒、开源模型后门是什么？怎么防？ —— ○（计划安全章）
7. 消费级产品里的生成式 AI 安全怎么做？（OpenAI、Anthropic、Google） —— ◐ [16.7 节](ch16-evals-production.md#16.7)（计划安全章）
8. 用户想看模型隐藏的推理过程，应用该给什么？ —— ○（计划安全章）
9. 模型给出了不安全的建议（医疗等），需要哪些控制？ —— ○（计划安全章）
10. 上线一个 LLM 应用之前，你会问哪些问题？ —— ◐ [16.10 节](ch16-evals-production.md#16.10)
11. 可解释性和机制可解释性是什么？ —— ○
12. GDPR、欧盟 AI 法案对 LLM 应用有什么要求？ —— ○

## C.10 AI 应用工程与前端 **[经验]**

这一类题集中出现在"前端转 AI 全栈"和 AI 应用开发的面经里，和你的前端背景最相关。

1. ★ 流式输出的原理是什么？为什么"打字机效果"不是前端动画，而是自回归生成的直接体现？ —— ◐ [17.1 节](ch17-inference-serving.md#17.1)（计划 AI 前端章）
2. ★ SSE、WebSocket、HTTP 分块传输怎么选？为什么 AI 回复多用 SSE？ —— ○（计划 AI 前端章）
3. ★ `EventSource` 只能发 GET，为什么很多实现改用 `fetch` + `ReadableStream`？怎么手写一个 SSE 解析器？ —— ○（计划 AI 前端章）
4. ★ 用户点"停止生成"、切换页面或组件卸载时，怎么中断请求（`AbortController`）并避免内存泄漏？后端怎么停止推理？ —— ○（计划 AI 前端章）
5. 用户快速连点发送，导致多个流交错，怎么处理？ —— ○（计划 AI 前端章）
6. 流式 Markdown 怎么增量渲染？代码块没闭合时怎么办？怎么防 XSS？ —— ○（计划 AI 前端章）
7. 很长的对话怎么做虚拟滚动和懒加载？自动滚动到底部和用户手动上翻怎么兼容？ —— ○（计划 AI 前端章）
8. 流式输出时如何同时展示工具调用的进度？Function calling 给流式带来什么挑战？ —— ◐ [10.4 节](ch10-tool-calling.md#10.4)（计划 AI 前端章）
9. 部署到 nginx 后流式变成一次性返回，可能是什么原因？ —— ○（计划 AI 前端章）
10. 设计一个 AI 聊天应用的文件上传：大文件、多模态，最终放进模型上下文。（xAI） —— ○（计划 AI 前端章）
11. 设计 OpenAI Playground。（OpenAI） —— ○（计划 AI 前端章）
12. 用 FastAPI + asyncio 调用大模型 API，为什么高并发下有优势？和多线程比呢？怎样避免阻塞事件循环？ —— ○（计划实战项目章）
13. 给一段 Python 代码加上带超时和退避的异步重试。（Sierra） —— ○（计划实战项目章）
14. 多家模型厂商怎么用一套接口适配？模型网关（AI gateway）做什么？ —— ◐ [14.1 节](ch14-langchain.md#14.1)
15. API 密钥怎么管理？为什么不能放在前端？ —— ○（计划实战项目章）
16. 生成式 UI：让模型输出界面组件而不是文本，怎么做？ —— ○（计划 AI 前端章）

## C.11 系统设计题 **[经验]**

系统设计题是英文面试的重头，几乎每家都有。下面按主题列出，答案大多需要组合书中多章的内容。

### C.11.1 RAG 与知识类

1. ★ 设计一个基于公司文档的问答助手。 —— ✅ 第 15 章、[16.1 节](ch16-evals-production.md#16.1)
2. ★ 设计一个覆盖一千万篇文档、带权限控制的企业 RAG。 —— ◐ 第 15 章（权限与规模计划数据与检索基础设施章）
3. 设计一个异构文档的摄取与索引流水线。（Databricks） —— ○（计划数据与检索基础设施章）
4. 设计一个保险理赔 agent，用 RAG，并控制 token 成本。（Scale AI） —— ◐ 第 15 章、[16.8 节](ch16-evals-production.md#16.8)
5. 设计法律文档审阅助手、金融文档处理系统。 —— ◐ 第 15 章
6. 设计一个面向几千张表的 Text-to-SQL 系统。 —— ○（计划数据与检索基础设施章）
7. 设计一个深度研究（deep research）agent。 —— ◐ 第 11 章（计划 agent 深入章）

### C.11.2 Agent 与产品类

1. ★ 设计一个客服 AI agent，能升级到人工。（Sierra 多次考到：户外用品公司、Spotify、流媒体） —— ◐ 第 10、11 章
2. 设计一个和第三方公司的数据与业务流程集成的 AI 系统。（Sierra） —— ◐ 第 12 章
3. ★ 设计一个 AI 编程 agent / 代码审查系统 / Copilot。 —— ✅ [11.7 节](ch11-agent-harness.md#11.7)
4. 设计一个 AI 邮件助理、会议纪要系统（每天几千场会议）。 —— ○
5. 设计一个多 agent 工作流系统。 —— ○（计划 agent 深入章）
6. 设计一个端侧 AI 助理。 —— ○

### C.11.3 平台与基础设施类

1. ★ 端到端设计 ChatGPT。 —— ◐ 第 9、17 章、C.10
2. ★ 设计一个 LLM 推理平台；设计一个高效采样的 API，支持批处理和请求编排。（Anthropic） —— ✅ 第 17 章
3. 设计一个 LLM 评测平台；设计一个实验追踪平台。（Google） —— ◐ 第 16 章
4. ★ LLM 应用的缓存策略：精确缓存、语义缓存、提示缓存。 —— ◐ [11.6 节](ch11-agent-harness.md#11.6)、[17.5 节](ch17-inference-serving.md#17.5)
5. ★ 限流、配额与成本管理怎么设计？厂商限流时怎么办？ —— ◐ [16.8 节](ch16-evals-production.md#16.8)
6. ★ 主模型挂了怎么回退？怎么在不停机的情况下换厂商或换模型？ —— ◐ [16.8 节](ch16-evals-production.md#16.8)
7. 语义路由：按难度把请求分给大模型或小模型。 —— ✅ [16.8 节](ch16-evals-production.md#16.8)、[16.9.6 节](ch16-evals-production.md#16.9.6)
8. 从每秒 100 个请求扩到每秒 5000 个，要改什么？ —— ✅ [17.11 节](ch17-inference-serving.md#17.11)
9. 多租户聊天机器人平台怎么隔离？ —— ○
10. 内容审核系统（含直播）怎么设计？ —— ○

### C.11.4 实时语音

1. 设计一个实时语音 AI 应用：延迟预算怎么分？ —— ○
2. 级联（ASR→LLM→TTS）和端到端语音模型怎么选？打断（barge-in）和轮次判断怎么做？ —— ○
3. 工具调用很慢时，语音 agent 怎么"填空"？ —— ○

## C.12 手撕代码 **[经验]**

中文算法岗面经里几乎必有"手撕"，英文 AI 工程岗则偏实用编码。

### C.12.1 模型相关

1. ★ 实现缩放点积注意力，带因果掩码和 padding 掩码。 —— ✅ [6.5.1 节](ch06-self-attention.md#6.5.1)、`ch05_06_lm_attention.py`
2. ★ 实现多头注意力，讲清每一步的张量形状。 —— ◐ [6.2 节](ch06-self-attention.md#6.2)（计划第 7 章给出完整实现）
3. 把多头注意力改成 GQA / MQA。 —— ○（计划第 7 章）
4. ★ 实现 KV cache 和单步解码。 —— ○（计划第 7 章）
5. ★ 实现数值稳定的 softmax 和交叉熵。 —— ✅ [4.3 节](ch04-calculus-optimization.md#4.3)、[A.5 节](appendix-a-math-quick-reference.md#A.5)
6. 实现 LayerNorm、RMSNorm。 —— ○（计划第 7 章）
7. 实现 RoPE。 —— ○（计划第 7 章）
8. ★ 实现 top-k、top-p、温度采样。 —— ✅ [9.6 节](ch09-prompt-engineering.md#9.6)、`ch09_prompting.py`
9. 实现 beam search。 —— ○（计划第 7 章）
10. 实现 BPE 的训练和编码。 —— ○（计划第 7 章）
11. 实现一个 LoRA 线性层。 —— ○（计划后训练章）
12. 实现 DPO 损失、PPO 的裁剪目标。 —— ○（计划后训练章）
13. 实现一个最小的 decoder-only Transformer 并训练它。 —— ○（计划第 7 章）
14. 实现 Adam 优化器。 —— ✅ [4.7 节](ch04-calculus-optimization.md#4.7)

### C.12.2 应用相关

1. ★ 实现一个最小的 RAG：切块、向量化、余弦检索。 —— ✅ `ch15_rag.py`
2. 实现 BM25 和 RRF 融合。 —— ✅ [15.3 节](ch15-rag.md#15.3)、[15.5 节](ch15-rag.md#15.5)
3. ★ 实现一个带工具调用和预算上限的最小 agent 循环。 —— ✅ [11.2 节](ch11-agent-harness.md#11.2)、`ch11_agent_harness.py`
4. 实现一个 function calling 处理器（解析、校验、执行、回填）。 —— ✅ `ch10_tool_calling.py`
5. 实现对话记忆：滑动窗口、摘要、缓冲。 —— ◐ `ch11_agent_harness.py`
6. 实现一个 LLM 评委流水线，并消除位置偏差。 —— ✅ [16.9.4 节](ch16-evals-production.md#16.9.4)、`ch16_evals.py`
7. ★ 实现带指数退避和抖动的重试。 —— ○（计划实战项目章）
8. 实现流式 SSE 解析器。 —— ○（计划 AI 前端章）
9. 实现带 TTL 的 LRU 缓存、语义缓存。 —— ○（计划实战项目章）
10. 实现令牌桶限流器，再改成分布式的。 —— ○（计划实战项目章）
11. 实现一个处理五万篇文档的异步批处理器（控制并发）。 —— ○（计划实战项目章）
12. 实现提示注入检测、PII 输出过滤。 —— ◐ [16.9.5 节](ch16-evals-production.md#16.9.5)

### C.12.3 大厂通用编码题（AI 公司真题）

1. 实现一个支持事务的内存键值存储：set、begin、commit、rollback。（OpenAI、xAI） —— ○
2. 字符串列表的编码与解码。（OpenAI） —— ○
3. 电子表格单元格依赖遍历，检测循环引用。（Sierra） —— ○
4. 扫描文件系统，找出重复文件。（Anthropic） —— ○
5. 把性能分析器的栈采样转换成调用轨迹。（Anthropic） —— ○
6. 按每张图的指令文件顺序执行图像变换的流水线。（Anthropic） —— ○
7. 只用 SQL 和 Python 从演示数据库里抽取并清洗出可用数据集。（Anthropic） —— ○
8. 编辑距离；统计每个商家去重后的评论数（SQL）。（腾讯） —— ○

## C.13 场景排障题 **[经验]**

这类题不问定义，直接给一个线上症状，看你怎么查。是近两年英文面试里增长最快的一类。

1. ★ RAG 答错了，但正确的文档就在库里。你按什么顺序排查？ —— ✅ [15.6 节](ch15-rag.md#15.6)、[15.8.2 节](ch15-rag.md#15.8.2)
2. ★ 长上下文提示的回答质量变差，为什么？ —— ◐ [9.8 节](ch09-prompt-engineering.md#9.8)、[11.4 节](ch11-agent-harness.md#11.4)
3. ★ token 费用太高，你先砍什么？ —— ✅ [11.6 节](ch11-agent-harness.md#11.6)、[16.8 节](ch16-evals-production.md#16.8)
4. ★ 上线后 p99 延迟回退了，或者高峰期延迟飙升，怎么查？ —— ✅ [17.6 节](ch17-inference-serving.md#17.6)、[17.11 节](ch17-inference-serving.md#17.11)
5. 多轮对话中途换话题，模型还在答上一个话题，怎么办？ —— ○（计划 agent 深入章）
6. 量化后精度掉了，怎么定位是哪一层？ —— ◐ [17.8 节](ch17-inference-serving.md#17.8)
7. 多个 LLM 串成的流水线里某一步出错，怎么定位？ —— ✅ [16.6 节](ch16-evals-production.md#16.6)
8. 模型生成的代码用了已经废弃的库接口，怎么办？ —— ◐ 第 15 章
9. 用户说模型"变笨了"，但你什么都没改。 —— ✅ [16.6 节](ch16-evals-production.md#16.6)
10. 新的 embedding 模型上线后搜索质量崩了。 —— ○（计划数据与检索基础设施章）

## C.14 行为面与开放题 **[经验]**

1. 介绍你做过的 RAG 或 agent 项目：背景、最难的点、怎么评估、遇到过什么坑？（中文面经几乎每场都从这里开始，追问可以有二三十个） —— 见 C.15 的建议
2. 你做过端到端的 agent 系统吗？（Glean） —— 同上
3. AI 工程和传统机器学习工程有什么区别？什么时候不该用 AI？ —— ✅ [11.1 节](ch11-agent-harness.md#11.1)
4. 直接调 API 还是自己部署模型？ —— ◐ [17.11 节](ch17-inference-serving.md#17.11)
5. 怎样衡量 AI 功能的投入产出？怎么管理利益相关方的预期？ —— ○
6. 讲一次你对方案很有信心、后来发现错了的经历。（Anthropic） —— 行为题
7. 讲一次你要做的东西和个人价值观冲突的经历。（Anthropic） —— 行为题
8. 为什么想来这家公司？最近哪些 AI 进展让你觉得有意思？未来一年哪些技能最重要？（Anthropic） —— 行为题
9. 你怎么向不懂技术的人解释一个技术概念？（Anthropic） —— 行为题
10. 你怎么跟进 AI 领域的新进展？三到五年后 AI 工程会变成什么样？ —— 行为题
11. 给 OpenAI 的微调能力制定一个产品策略。（OpenAI） —— ○

## C.15 覆盖情况与补课顺序 **[经验]**

上面一共 366 道题，其中 128 道标了 ★。按标记数：✅ 119 道，◐ 97 道，○ 143 道，其余 7 道是行为题。按类别粗略看：

| 类别 | 已覆盖 ✅ | 部分 ◐ | 未覆盖 ○ | 主要空白 |
|---|---|---|---|---|
| C.2 基础与 Transformer | 多 | 中 | 中 | RoPE、tokenizer、归一化、FlashAttention、MoE |
| C.3 训练与微调 | 少 | 少 | **多** | 显存账本、ZeRO、混合精度、LoRA 训练、灾难性遗忘 |
| C.4 对齐与强化学习 | 无 | 少 | **几乎全部** | RLHF、DPO、GRPO、奖励作弊 |
| C.5 推理与部署 | **多** | 中 | 少 | 框架对比、端侧推理 |
| C.6 提示 | **多** | 中 | 少 | 多语言 |
| C.7 RAG | 中 | 多 | 中 | 查询改写、引用、增量索引、向量库内部 |
| C.8 Agent 与协议 | **多** | 中 | 中 | 记忆系统、多 agent、A2A |
| C.9 评测与安全 | 中 | 中 | 中 | 传统指标与基准、红队、护栏 |
| C.10 AI 前端 | 无 | 少 | **几乎全部** | SSE、流式渲染、中断、长对话 |
| C.11 系统设计 | 少 | 多 | 中 | 需要把多章串起来练 |
| C.12 手撕代码 | 中 | 少 | 多 | 多头注意力、KV cache、BPE、重试与限流 |

**结论** **[经验]**：

1. **工程实践部分（第 9–17 章）覆盖得最好**。RAG、agent、MCP、推理这几类高频题，大部分已经能在书里找到原理和数字。
2. **最大的两块空白是训练和对齐（C.3、C.4）**，而它们恰好是中文算法岗面经的必考内容。这和路线图里的"微调与后训练"一章正好对上，应该优先写。
3. **AI 前端（C.10）是第二个空白**，而且和你的背景最贴近：这些题本质上是前端工程问题加上一点对自回归生成的理解，补起来成本低、区分度高。
4. **第 7 章（动手训练一个小 GPT）能一次补上手撕代码里的大半**：多头注意力、KV cache、RoPE、RMSNorm、BPE、采样，都可以在同一份代码里实现。
5. 系统设计题不需要新章节，需要的是练习：拿 C.11 里的题，按"需求 → 架构图 → 关键数字 → 失效方式 → 评测"的顺序各写一页。

**建议的补课顺序**：

| 顺序 | 内容 | 补上的题 |
|---|---|---|
| 1 | 第 7 章：动手训练一个小 GPT | C.2 的 RoPE、tokenizer、归一化；C.12.1 的大部分 |
| 2 | 微调与后训练（SFT、LoRA、显存与分布式、RLHF、DPO、GRPO） | C.3、C.4 几乎全部 |
| 3 | AI 前端工程 | C.10 全部，C.12.2 的 SSE 解析 |
| 4 | 真实 API 的端到端项目 | C.10 的后端部分、C.12.2 的重试和限流、C.14 的项目介绍 |
| 5 | Agent 深入（记忆、多 agent、A2A） | C.8.3、C.8.4 |
| 6 | 数据与检索基础设施、评测工程、安全 | C.7.5、C.9 的剩余部分 |

**关于"介绍你的项目"** **[经验]**：几篇面经里，一个带记忆的 RAG 项目被追问了四十多个问题，从记忆冲突、指代补全、混合检索打分、增量索引，一直问到 asyncio 和 SGLang。所以实战项目那一章最好做成一个能被这样追问的项目：每个设计决定都留下理由和一个实测数字。

## 参考资料

下面是整理时用到的主要来源。题目都经过改写和合并，具体措辞以原文为准。

**英文**

1. University of Pennsylvania Career Services（转载 Exponent）. *45+ AI Engineer Interview Questions & Answers (2026 Guide).* 2026-06。含 Anthropic、OpenAI、Sierra、Scale AI、xAI、Databricks、Glean、Perplexity 的真题。
2. Amit Shekhar. *ai-engineering-interview-questions*（GitHub）。约 400 题，覆盖 LLM 基础、提示、RAG、agent、微调、向量库、系统设计、LLMOps、评测、安全、多模态、基础设施、编码。
3. Hyring. *LLM Interview Questions.* 45 题，含大量场景排障题。
4. DataCamp. *Top LLM Interview Questions*；*RAG Interview Questions*；*MCP Interview Questions*；*AI Engineer Interview Questions.* 2026。
5. InterviewBit. *LLM Interview Questions.*
6. techinterview.org. *MCP Interview Questions*；RLHF 面试题相关文章。
7. J. Zhang. *Everything about RL and Agentic RL*（博客，2026）。后训练与 agentic RL 自测题。
8. KDnuggets、PracHub 等的生成式 AI 系统设计题整理。

**中文**

9. wdndev. *llm_interview_note*（GitHub）。大模型算法与应用工程师知识点与面试题，按基础、架构、数据、分布式训练、微调、推理、强化学习、RAG、评估、应用分章。
10. km1994. *LLMs_interview_notes*（GitHub）；jackaduma. *awesome_LLMs_interview_notes*（GitHub）。
11. ricardokevins. *大模型面试题库*（GitHub Pages）。十个核心模块、手撕代码专项、分布式训练专项、RL 专项。
12. 阿里云开发者社区. *AI Agent 面试题*（基础、记忆、工具调用、RAG、框架、多 agent、工程化、全栈）；*大模型推理加速面试题.*
13. CSDN. *大模型应用开发面试题*（agent、部署、评测、微调、LangChain、提示、RAG 共七十余题）。
14. CSDN. 字节跳动大模型应用岗面经（带记忆的对话 agent + RAG 项目追问）；腾讯大模型算法岗面经。
15. 80aj. *大模型面试 100 问：推理与部署篇*，2026-01。
16. 掘金、CSDN 上"前端转 AI 全栈"系列：SSE 流式输出、fetch + ReadableStream、React/Vue 流式渲染。
17. CSDN、掘金. DeepSpeed ZeRO、混合精度与显存计算相关面试整理。
