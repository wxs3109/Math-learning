# 第 14 章 LangChain 与 LangGraph

> 标注：**[已证]** 给出了推导 · **[事实]** 标准结论 · **[直觉]** 帮助理解、不严格 · **[经验]** 工程实践中总结的做法

## 导读

第 10–13 章的东西，都可以直接用模型厂商的 SDK 从头写。框架把其中通用的部分包装起来。**LangChain** 是用得最广的一个，**LangGraph** 是同一团队做的底层编排库。本章讲：
- 框架解决什么问题；
- LangChain 的核心抽象：模型、消息、工具、Runnable 与 LCEL；
- LangChain 1.x 的 `create_agent` 和 middleware，以及它们和第 11 章 Harness 的对应关系；
- LangGraph：state graph、conditional edge、checkpoint、human-in-the-loop；
- 什么时候用框架，什么时候不用。

**与大模型的关系**

LangChain 和 LangGraph 在 2025 年 10 月发布 1.0，接口大幅简化：
- agent 统一用 `langchain.agents.create_agent` 构建，它运行在 LangGraph 之上；
- 旧的 `AgentExecutor` 和 LangGraph 的 `create_react_agent` 都已弃用；
- 定制 agent 循环的方式变成**middleware**。

网上很多教程还是 0.x 的写法，读的时候要注意版本。

**前置知识**

[第 10 章](ch10-tool-calling.md)（tool calling）、[第 11 章](ch11-agent-harness.md)（agent 循环与 harness）。Python 的类型标注。

## 14.1 框架解决什么问题 **[事实 + 经验]**

**统一不同厂商的接口**：Claude、GPT、Gemini、开源模型的 API 各不相同，比如消息格式、tool calling 格式、流式输出。LangChain 提供统一的聊天模型接口和消息类型，换模型通常只改一行。

**现成的部件**：
- prompt 模板、输出解析、structured output；
- 工具定义（从函数签名生成 schema）；
- agent 循环，以及摘要压缩、人工审批、重试等常用 middleware；
- 大量第三方集成：向量数据库、文档加载器、检索器（第 15 章）。

**编排与持久化**（LangGraph）：把多步流程画成 state graph，每一步之后自动保存状态，可以暂停、恢复、回放，适合长时间运行和需要 human-in-the-loop 的流程。

**observability**：和 LangSmith 等工具集成，记录每一次模型调用和 tool calling（第 16 章）。

## 14.2 核心抽象 **[事实]**

**聊天模型**：所有模型实现同一个接口。
- `invoke(messages)`：输入一组消息，返回一条 `AIMessage`；
- `batch`：并发处理多条输入；
- `stream`：逐块输出；
- `ainvoke`：异步版本。

**消息**：`SystemMessage`、`HumanMessage`、`AIMessage`（可以带 `tool_calls`）、`ToolMessage`（工具结果，带 `tool_call_id`）。它们对应第 9 章的角色和第 10 章的 `tool_use` / `tool_result`。

**工具**：`@tool` 装饰一个带类型标注和文档字符串的函数，自动生成名字、描述和参数 schema（[10.2 节](ch10-tool-calling.md#10.2)）。

**Runnable 与 LCEL**：模型、prompt 模板、解析器、普通函数都是 `Runnable`，有相同的 `invoke` / `batch` / `stream` 接口。用 `|` 把它们串起来（LangChain Expression Language）：
```python
chain = prompt | model | StrOutputParser()
chain.invoke({"subject": "线性代数", "question": "什么是秩？"})
```
`RunnableParallel` 把同一个输入同时送给几个分支。

## 14.3 Create_agent 与 middleware **[事实]**

```python
from langchain.agents import create_agent

agent = create_agent(
    model="anthropic:claude-sonnet-5-5",   # 或者一个模型对象
    tools=[get_weather],
    system_prompt="你是天气助手。",
    middleware=[...],
)
agent.invoke({"messages": [{"role": "user", "content": "温哥华天气？"}]})
```
它运行的就是第 11 章的循环：调用模型 → 有 tool calling 就执行 → 结果放回 → 直到没有 tool calling。

**middleware**是定制这个循环的方式，可以挂在几个位置：
- `before_model` / `after_model`：每次调用模型前后，比如修改上下文、记录日志；
- `wrap_model_call`：包住模型调用，比如换模型、重试、降级到备用模型；
- `wrap_tool_call`：包住每次 tool call，比如拦截、改参数、重试；
- `before_agent` / `after_agent`：整次运行的开始和结束。

**内置 middleware 和第 11 章 Harness 部件的对应**：

| 内置 middleware | 对应的 harness 部件 |
|---|---|
| `SummarizationMiddleware` | 上下文超过阈值时自动摘要压缩（[11.4 节](ch11-agent-harness.md#11.4)） |
| `ContextEditingMiddleware` | 清理旧的工具结果 |
| `HumanInTheLoopMiddleware` | 敏感 tool calling 前请人审批（[11.5 节](ch11-agent-harness.md#11.5)） |
| `ModelCallLimitMiddleware`、`ToolCallLimitMiddleware` | 轮数和调用次数上限 |
| `ModelRetryMiddleware`、`ToolRetryMiddleware`、`ModelFallbackMiddleware` | 失败重试、换备用模型 |
| `PIIMiddleware` | 检测和遮盖个人信息 |
| `TodoListMiddleware` | 让 agent 维护待办清单 |
| `LLMToolSelectorMiddleware` | 工具很多时，先用小模型挑出相关的工具 |

框架做的事情，第 11 章都讲过原理，框架把它们做成了可以拼装的部件。

## 14.4 LangGraph：state graph **[事实]**

当流程不只是"模型–工具"循环时，比如有固定的步骤、分支、多个 agent 协作、需要人工审批，就用 LangGraph 直接画图。

**四个概念**：
- **状态**（State）：一个类型化的字典，在节点之间传递；
- **节点**（Node）：一个函数，读状态，返回要更新的字段。节点里可以调用模型、工具或任意代码；
- **边**（Edge）：固定的下一步；**conditional edge**：由一个函数根据状态决定下一步，这就是分支和循环；
- **checkpoint**（Checkpointer）：每一步之后把状态存下来。有了它就能：
  - 程序崩溃后从断点继续（持久执行）；
  - 在任意节点里调用 `interrupt(...)` 暂停，等人回复后用 `Command(resume=...)` 继续；
  - 用同一个 `thread_id` 进行多轮对话；
  - 回放和调试历史状态。

14.6.5 节用它实现了第 11 章的"evaluator–optimizer"模式：写 → 审 → 不通过就回去改 → 通过后暂停等人工审批。

**LangChain 和 LangGraph 的关系**：先用 `create_agent` 快速起步；需要自定义流程时，再下到 LangGraph。`create_agent` 返回的本身就是一个编译好的 LangGraph 图，所以 checkpoint、流式输出、中断等能力它都有。

## 14.5 要不要用框架 **[经验]**

**适合用**：
- 需要在多家模型之间切换；
- 需要现成的集成（各种向量库、文档加载器）；
- 需要持久执行、human-in-the-loop、复杂的分支流程（LangGraph 的强项）；
- 团队已经在用 LangSmith 做观测和评测。

**可以不用**：
- 只用一家模型，流程简单：直接用厂商 SDK，几十行代码就能写出 agent 循环（[11.2 节](ch11-agent-harness.md#11.2)），每一步都看得见；
- 需要用到厂商最新的功能，而框架还没跟上。

**无论用不用** **[经验]**：要知道框架在底层做了什么。很多问题出在"以为框架做了某件事，其实没做"，比如 prompt 到底长什么样、工具结果有没有被截断、缓存有没有命中。打开调试日志，看实际发出去的请求。Anthropic 的建议也是：可以从直接调用 API 开始；用框架的话，一定要理解它底层的代码。

## 14.6 数值实验

代码：[`code/ch14_langchain.py`](code/ch14_langchain.py)。依赖 `pip install "langchain>=1" "langgraph>=1"`，运行 `python3 ch14_langchain.py`。代码用脚本化的假模型（按顺序返回预先写好的回复）代替真实模型，所以不需要 API key。换成真实模型只需改模型那一行，其余代码不变。以下输出来自 langchain 1.5、langgraph 1.2。

### 14.6.1 统一的模型接口

`model.invoke([SystemMessage(...), HumanMessage("梯度是什么？")])` 返回一个 `AIMessage`。

### 14.6.2 @tool

`get_weather(city: str, unit: str = "celsius")` 自动生成的定义：`city` 必填，`unit` 可选且默认 `celsius`，描述来自文档字符串。

### 14.6.3 LCEL

- prompt 模板填好后是两条消息：system "你是线性代数助教，回答不超过一句话。"，human "什么是秩？"。
- `prompt | model | StrOutputParser()` 用 `batch` 一次处理两个问题，返回两个字符串。
- `RunnableParallel(upper=..., length=...)` 对 `"lcel"` 同时得到 `{'upper': 'LCEL', 'length': 4}`。

### 14.6.4 `create_agent` 与 middleware

消息序列：
1. `HumanMessage`：温哥华和北京天气？
2. `AIMessage`：两个并行的 tool call `get_weather(Vancouver)`、`get_weather(Beijing)`
3. 两条 `ToolMessage`：`8°C 多云`、`2°C 晴`
4. `AIMessage`：最终回答

middleware 日志：调用模型前上下文有 1 条消息 → 两次 tool calling → 调用模型前上下文有 4 条消息。两个 tool calling 是并行执行的，日志里谁先谁后每次运行可能不同（这次是先 Beijing 后 Vancouver）。顺序不同说明它们确实是并发的，但顺序相同不能说明是串行的；要确认并行，看两次调用的时间戳是否重叠，或让工具各 sleep 1 秒，看总耗时是 1 秒还是 2 秒。

### 14.6.5 LangGraph：写—审—改 + 人工审批

| 步 | 节点 | 结果 |
|---|---|---|
| 1 | write | 草稿："softmax 把分数变成概率。" |
| 2 | review | 缺少：logits、最大值 → 回到 write |
| 3 | write | 第 2 稿：加上了 logits |
| 4 | review | 缺少：最大值 → 回到 write |
| 5 | write | 第 3 稿：加上"先减去最大值防止溢出" |
| 6 | review | 通过 → 去 human |
| 7 | human | `interrupt`：图停在 `('human',)`，状态存进 checkpoint |
| 8 | （人回复"同意"，`Command(resume="同意")`） | `approved_by_human = True`，结束 |

第 7 步之后，程序可以退出；只要 checkpoint 存在持久化的存储里（这里用的是内存版），几小时后用同一个 `thread_id` 也能接着跑。

## 14.7 小结与疑问

**搞懂了什么**：
- LangChain 统一了模型、消息、工具的接口；LCEL 用 `|` 组合 Runnable。
- `create_agent` 就是第 11 章的 agent 循环；middleware 是定制循环的方式，内置 middleware 一一对应 harness 的部件。
- LangGraph 用状态、节点、conditional edge 描述流程；checkpoint 带来持久执行、中断和恢复。
- 简单场景直接用厂商 SDK；无论用不用框架，都要知道底层发出了什么请求。

**疑问**：
- middleware 的执行顺序怎样影响结果？比如先做摘要还是先做 PII 遮盖。
- 多 agent 的协作模式（主管、群体、交接）在 LangGraph 里怎样表达，各自适合什么场景？

## 参考资料

1. LangChain. *LangChain v1 release notes*、*Migration guide*、*Middleware*（docs.langchain.com）。
2. LangChain. *LangGraph v1 release notes*、*Graph API*、*Persistence*、*Interrupts*。
3. S. Runkle and the LangChain OSS team. *LangChain and LangGraph Agent Frameworks Reach v1.0 Milestones.* 2025。
4. E. Schluntz, B. Zhang (Anthropic). *Building effective agents.* 2024：关于何时使用框架的建议。
