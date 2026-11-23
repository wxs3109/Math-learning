r"""
第 10 章 · 工具调用的配套代码（llm/ch10-tool-calling.md）。只用标准库：python3 ch10_tool_calling.py

不调用真实模型。用一个"脚本化模型"代替真实模型的决策，把工具调用的完整协议走一遍：
1. 从 Python 函数签名自动生成工具的 JSON Schema（框架里的 @tool 做的就是这件事）。
2. 一个最小的 JSON Schema 校验器。
3. 消息协议：assistant 发出 tool_use 块，程序执行后用 tool_result 块（带同一个 id）送回。
4. 参数出错时把错误作为 tool_result（is_error=True）送回，模型据此自我修正。
5. 并行工具调用：一条回复里有多个 tool_use，程序并发执行，结果一次送回。
"""
import inspect, json, time, typing
from concurrent.futures import ThreadPoolExecutor

# ---------------------------------------------------------------------------
# 1. 从函数签名生成 JSON Schema
# ---------------------------------------------------------------------------
PY2JSON = {int: "integer", float: "number", str: "string", bool: "boolean"}


def tool_schema(fn):
    """读取类型标注和文档字符串，生成 {name, description, input_schema}。"""
    sig = inspect.signature(fn)
    hints = typing.get_type_hints(fn)
    props, required = {}, []
    for name, param in sig.parameters.items():
        t = hints[name]
        if typing.get_origin(t) is typing.Literal:          # Literal["a", "b"] → enum
            props[name] = {"type": "string", "enum": list(typing.get_args(t))}
        else:
            props[name] = {"type": PY2JSON[t]}
        if param.default is inspect.Parameter.empty:
            required.append(name)
    doc = inspect.getdoc(fn) or ""
    return {"name": fn.__name__, "description": doc,
            "input_schema": {"type": "object", "properties": props, "required": required,
                             "additionalProperties": False}}


def get_weather(city: str, unit: typing.Literal["celsius", "fahrenheit"] = "celsius") -> str:
    """查询城市当前天气。city 用城市的英文名，例如 Vancouver。"""
    time.sleep(0.3)                                          # 假装是一次网络请求
    temp_c = {"Vancouver": 8, "Beijing": 2, "Tokyo": 12}.get(city)
    if temp_c is None:
        raise ValueError(f"未知城市 {city!r}。支持：Vancouver, Beijing, Tokyo")
    t = temp_c if unit == "celsius" else temp_c * 9 / 5 + 32
    return f"{city}: {t:g}°{'C' if unit == 'celsius' else 'F'}，多云"


def add(a: int, b: int) -> str:
    """两个整数相加。"""
    return str(a + b)


TOOLS = {f.__name__: f for f in [get_weather, add]}
print("=== 1. 由函数签名生成的工具定义 ===")
print(json.dumps(tool_schema(get_weather), ensure_ascii=False, indent=2))

# ---------------------------------------------------------------------------
# 2. 最小 JSON Schema 校验器（真实项目用 jsonschema 库或 pydantic）
# ---------------------------------------------------------------------------
JSON_TYPES = {"integer": int, "number": (int, float), "string": str, "boolean": bool, "object": dict}


def validate(value, schema, path="input"):
    errs = []
    t = schema.get("type")
    if t and not (isinstance(value, JSON_TYPES[t]) and not (t in ("integer", "number") and isinstance(value, bool))):
        return [f"{path} 应为 {t}，实际是 {type(value).__name__} {value!r}"]
    if "enum" in schema and value not in schema["enum"]:
        errs.append(f"{path} 必须是 {schema['enum']} 之一，实际是 {value!r}")
    if t == "object":
        for k in schema.get("required", []):
            if k not in value: errs.append(f"缺少必填字段 {path}.{k}")
        for k, v in value.items():
            if k in schema.get("properties", {}):
                errs += validate(v, schema["properties"][k], f"{path}.{k}")
            elif schema.get("additionalProperties") is False:
                errs.append(f"不认识的字段 {path}.{k}")
    return errs


# ---------------------------------------------------------------------------
# 3–5. 执行一轮工具调用
# ---------------------------------------------------------------------------
def run_tool(block, max_chars=200):
    """执行一个 tool_use 块，返回对应的 tool_result 块。任何错误都变成 is_error 结果，而不是让程序崩溃。"""
    fn = TOOLS.get(block["name"])
    if fn is None:
        return {"type": "tool_result", "tool_use_id": block["id"], "is_error": True,
                "content": f"没有名为 {block['name']} 的工具"}
    errs = validate(block["input"], tool_schema(fn)["input_schema"])
    if errs:
        return {"type": "tool_result", "tool_use_id": block["id"], "is_error": True,
                "content": "参数错误：" + "；".join(errs)}
    try:
        out = fn(**block["input"])
    except Exception as e:                                   # 工具内部的错误同样送回给模型
        return {"type": "tool_result", "tool_use_id": block["id"], "is_error": True, "content": str(e)}
    if len(out) > max_chars:                                 # 结果太长就截断，保护上下文
        out = out[:max_chars] + f"……（已截断，共 {len(out)} 字符）"
    return {"type": "tool_result", "tool_use_id": block["id"], "content": out}


def execute(assistant_content, parallel=True):
    calls = [b for b in assistant_content if b["type"] == "tool_use"]
    if parallel:
        with ThreadPoolExecutor() as ex:
            results = list(ex.map(run_tool, calls))
    else:
        results = [run_tool(c) for c in calls]
    return {"role": "user", "content": results}              # 全部结果放在同一条 user 消息里


# 脚本化模型：按顺序给出"真实模型可能给出的回复"。真实系统里这一步是一次 API 调用。
SCRIPT = [
    [{"type": "text", "text": "我先查一下两个城市的天气。"},
     {"type": "tool_use", "id": "toolu_01", "name": "get_weather", "input": {"city": "Vancouver"}},
     {"type": "tool_use", "id": "toolu_02", "name": "get_weather", "input": {"city": "北京", "unit": "kelvin"}}],
    [{"type": "text", "text": "第二个调用出错了：单位只能是 celsius 或 fahrenheit，城市要用英文名。重试。"},
     {"type": "tool_use", "id": "toolu_03", "name": "get_weather", "input": {"city": "Beijing"}}],
    [{"type": "text", "text": "温哥华 8°C，北京 2°C，都是多云。温哥华高 6 度。"}],
]


def scripted_model(messages, step):
    return SCRIPT[step]


print("\n=== 3–5. 一次完整的工具调用对话 ===")
messages = [{"role": "user", "content": "温哥华和北京现在各多少度？差几度？"}]
for step in range(len(SCRIPT)):
    reply = scripted_model(messages, step)
    messages.append({"role": "assistant", "content": reply})
    if not any(b["type"] == "tool_use" for b in reply):     # 没有工具调用 = 模型认为已经完成
        break
    t0 = time.time()
    messages.append(execute(reply))
    print(f"[第 {step + 1} 轮工具执行耗时 {time.time() - t0:.2f}s]")
for m in messages:
    print(json.dumps(m, ensure_ascii=False))

print("\n=== 并行 vs 串行 ===")
three = [{"type": "tool_use", "id": f"t{i}", "name": "get_weather", "input": {"city": c}}
         for i, c in enumerate(["Vancouver", "Beijing", "Tokyo"])]
for flag in [False, True]:
    t0 = time.time(); execute(three, parallel=flag)
    print(f"  {'并行' if flag else '串行'}执行 3 个各需 0.3s 的调用：{time.time() - t0:.2f}s")

print("\n=== 校验器能抓到的错误 ===")
s = tool_schema(get_weather)["input_schema"]
for bad in [{"city": 42}, {"unit": "celsius"}, {"city": "Tokyo", "units": "celsius"}, {"city": "Tokyo", "unit": "K"}]:
    print(f"  {json.dumps(bad, ensure_ascii=False):<38} → {validate(bad, s)}")
