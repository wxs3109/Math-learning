r"""
第 14 章 · LangChain 与 LangGraph 的配套代码（llm/ch14-langchain.md）。
依赖：pip install "langchain>=1" "langgraph>=1"。不需要任何 API key：用脚本化的假模型代替真实模型。
运行：python3 ch14_langchain.py

1. 统一的聊天模型接口与消息类型。
2. @tool：从函数签名生成工具 schema。
3. LCEL：用 | 把提示模板、模型、输出解析器串成一条链。
4. create_agent：LangChain 1.x 的标准 agent；用中间件（middleware）在循环里插入自己的逻辑。
5. LangGraph：自己画状态图，实现"写—审—改"循环；用检查点（checkpointer）在人工审批处暂停和恢复。
"""
import json
from typing import TypedDict

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain_core.tools import tool
from langchain_core.utils.function_calling import convert_to_openai_tool


class ScriptedModel(FakeMessagesListChatModel):
    """按顺序返回预先写好的回复。真实代码里换成 init_chat_model("anthropic:...") 之类，其余代码不变。"""

    def bind_tools(self, tools, **kwargs):
        return self


# ---------------------------------------------------------------------------
# 1. 聊天模型接口
# ---------------------------------------------------------------------------
print("=== 1. 统一的模型接口 ===")
model = ScriptedModel(responses=[AIMessage(content="梯度是函数增长最快的方向。")])
reply = model.invoke([SystemMessage("你是数学助教。"), HumanMessage("梯度是什么？")])
print(f"  invoke → {type(reply).__name__}: {reply.content}")
print("  同一个接口还有 .batch()（并发处理多条输入）、.stream()（逐块输出）、.ainvoke()（异步）。")

# ---------------------------------------------------------------------------
# 2. @tool
# ---------------------------------------------------------------------------
print("\n=== 2. @tool 生成的工具定义 ===")


@tool
def get_weather(city: str, unit: str = "celsius") -> str:
    """查询城市当前天气。city 用英文名。"""
    return {"Vancouver": "8°C 多云", "Beijing": "2°C 晴"}.get(city, f"未知城市 {city}")


print(json.dumps(convert_to_openai_tool(get_weather), ensure_ascii=False))

# ---------------------------------------------------------------------------
# 3. LCEL
# ---------------------------------------------------------------------------
print("\n=== 3. LCEL：prompt | model | parser ===")
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是{subject}助教，回答不超过一句话。"),
    ("human", "{question}"),
])
print("  模板填好后的消息：", [(m.type, m.content) for m in prompt.invoke(
    {"subject": "线性代数", "question": "什么是秩？"}).to_messages()])
model = ScriptedModel(responses=[AIMessage(content="秩是线性无关的列数。"), AIMessage(content="SVD 把矩阵分解为 UΣVᵀ。")])
chain = prompt | model | StrOutputParser()            # 每一段都是 Runnable，| 把它们串起来
print("  chain.batch →", chain.batch([{"subject": "线性代数", "question": "什么是秩？"},
                                     {"subject": "线性代数", "question": "什么是 SVD？"}]))
fanout = RunnableParallel(upper=RunnableLambda(str.upper), length=RunnableLambda(len))
print("  RunnableParallel →", fanout.invoke("lcel"))

# ---------------------------------------------------------------------------
# 4. create_agent + 中间件
# ---------------------------------------------------------------------------
print("\n=== 4. create_agent 与中间件 ===")
from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware, before_model, wrap_tool_call

LOG = []


@before_model
def log_context(state, runtime):
    """每次调用模型之前运行：这里只记录上下文里有几条消息。"""
    LOG.append(f"调用模型前：上下文有 {len(state['messages'])} 条消息")


@wrap_tool_call
def guard(request, handler):
    """包住每一次工具调用：可以改参数、拦截、记录、重试。"""
    call = request.tool_call
    LOG.append(f"工具调用：{call['name']}({call['args']})")
    return handler(request)


model = ScriptedModel(responses=[
    AIMessage(content="", tool_calls=[{"name": "get_weather", "args": {"city": "Vancouver"}, "id": "c1"},
                                      {"name": "get_weather", "args": {"city": "Beijing"}, "id": "c2"}]),
    AIMessage(content="温哥华 8°C 多云，北京 2°C 晴。"),
])
agent = create_agent(model=model, tools=[get_weather], system_prompt="你是天气助手。",
                     middleware=[log_context, guard, ToolCallLimitMiddleware(run_limit=5)])
result = agent.invoke({"messages": [{"role": "user", "content": "温哥华和北京天气？"}]})
for m in result["messages"]:
    extra = f" {[(c['name'], c['args']) for c in m.tool_calls]}" if getattr(m, "tool_calls", None) else ""
    print(f"  {type(m).__name__:<12} {m.content!r}{extra}")
print("  中间件日志：")
for line in LOG: print("   ", line)

# ---------------------------------------------------------------------------
# 5. LangGraph：写—审—改循环 + 人工审批
# ---------------------------------------------------------------------------
print("\n=== 5. LangGraph：状态图、条件边、检查点与中断 ===")
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt


class State(TypedDict):
    topic: str
    draft: str
    feedback: str
    rounds: int
    approved_by_human: bool


DRAFTS = ["softmax 把分数变成概率。",
          "softmax 把 logits 变成概率分布：取指数再归一化。",
          "softmax 把 logits 变成概率分布：每个分量取指数再除以总和；实现时先减去最大值防止溢出。"]


def write(state):                         # 节点：写（或按反馈重写）。真实系统里是一次模型调用
    return {"draft": DRAFTS[min(state["rounds"], 2)], "rounds": state["rounds"] + 1}


def review(state):                        # 节点：审。这里用规则代替"评审模型"
    missing = [k for k in ["logits", "最大值"] if k not in state["draft"]]
    return {"feedback": "通过" if not missing else f"缺少：{missing}"}


def route(state):                         # 条件边：通过就去人工审批，否则回去重写（最多 3 轮）
    return "human" if state["feedback"] == "通过" or state["rounds"] >= 3 else "write"


def human(state):                         # 节点：暂停，等人确认
    decision = interrupt({"请审批": state["draft"]})
    return {"approved_by_human": decision == "同意"}


g = StateGraph(State)
g.add_node("write", write); g.add_node("review", review); g.add_node("human", human)
g.add_edge(START, "write"); g.add_edge("write", "review")
g.add_conditional_edges("review", route, ["write", "human"])
g.add_edge("human", END)
app = g.compile(checkpointer=InMemorySaver())   # 检查点：每一步之后保存状态，可以暂停、恢复、回放

cfg = {"configurable": {"thread_id": "demo-1"}}
for event in app.stream({"topic": "softmax", "draft": "", "feedback": "", "rounds": 0,
                         "approved_by_human": False}, cfg, stream_mode="updates"):
    for node, update in event.items():
        print(f"  [{node}] {update}")
snap = app.get_state(cfg)
print(f"  图停在：{snap.next}（等待人工审批，状态已存进检查点）")
for event in app.stream(Command(resume="同意"), cfg, stream_mode="updates"):   # 人回复后从断点继续
    for node, update in event.items():
        print(f"  [{node}] {update}")
print(f"  最终状态：rounds={app.get_state(cfg).values['rounds']}，"
      f"approved_by_human={app.get_state(cfg).values['approved_by_human']}")
