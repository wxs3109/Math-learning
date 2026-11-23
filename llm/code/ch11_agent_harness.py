r"""
第 11 章 · Agent 与 harness 的配套代码（llm/ch11-agent-harness.md）。只用标准库：python3 ch11_agent_harness.py

"Agent = 模型 + harness"。这里用一个规则写成的"策略"代替模型，重点放在 harness 的各个部件上：
1. Agent 循环：调用模型 → 执行工具 → 把结果放回上下文 → 直到模型不再调用工具，或达到轮数上限。
2. 权限：只读工具直接放行，写操作需要在允许列表里，删除一律拒绝。拒绝也是一种 tool_result。
3. 钩子（hooks）：每次工具调用前后运行的检查和记录。
4. 上下文管理：截断过长的工具结果；上下文超过预算时做压缩（compaction）。
5. 成本：带提示缓存（prompt caching）时，每一轮实际要付多少钱。
"""
import json

# ---------------------------------------------------------------------------
# 一个玩具"工作区"：几个文件和一个测试
# ---------------------------------------------------------------------------
FS = {
    "calc.py": "def mean(xs):\n    return sum(xs) / len(xs) + 1   # bug: 多加了 1\n",
    "test_calc.py": "from calc import mean\nassert mean([1, 2, 3]) == 2\n",
    "notes.tmp": "临时文件\n",
}


def tool_list_files():
    return "\n".join(sorted(FS))


def tool_read_file(path):
    if path not in FS: raise FileNotFoundError(f"{path} 不存在")
    return FS[path]


def tool_write_file(path, content):
    FS[path] = content
    return f"已写入 {path}（{len(content)} 字符）"


def tool_delete_file(path):
    del FS[path]
    return f"已删除 {path}"


def tool_run_tests():
    env = {}
    try:
        exec(FS["calc.py"], env)                              # 玩具：真实 harness 会在沙箱里跑
        exec(FS["test_calc.py"].replace("from calc import mean\n", ""), env)
        return "1 passed"
    except AssertionError:
        return "1 failed: assert mean([1, 2, 3]) == 2"


TOOLS = {"list_files": tool_list_files, "read_file": tool_read_file, "write_file": tool_write_file,
         "delete_file": tool_delete_file, "run_tests": tool_run_tests}

# ---------------------------------------------------------------------------
# 权限与钩子
# ---------------------------------------------------------------------------
READ_ONLY = {"list_files", "read_file", "run_tests"}
WRITE_ALLOW = {"calc.py"}                                     # 只允许改这个文件


def permission(name, args):
    if name in READ_ONLY: return "allow"
    if name == "write_file" and args.get("path") in WRITE_ALLOW: return "allow"
    return "deny"                                             # 删除、写其他文件：拒绝（真实系统里可以改成"询问用户"）


AUDIT = []


def pre_tool_hook(name, args):
    AUDIT.append(("pre", name, json.dumps(args, ensure_ascii=False)[:40]))


def post_tool_hook(name, result):
    AUDIT.append(("post", name, result[:30].replace("\n", " ")))


# ---------------------------------------------------------------------------
# "模型"：一个看上下文做决定的规则策略（真实系统里这是一次 API 调用）
# ---------------------------------------------------------------------------
def policy(messages):
    seen = " ".join(json.dumps(m, ensure_ascii=False) for m in messages)
    if "list_files" not in seen:
        return [{"type": "tool_use", "id": "t1", "name": "list_files", "input": {}}]
    if "def mean" not in seen:
        return [{"type": "tool_use", "id": "t2", "name": "read_file", "input": {"path": "calc.py"}},
                {"type": "tool_use", "id": "t3", "name": "read_file", "input": {"path": "test_calc.py"}}]
    if "已写入" not in seen:
        return [{"type": "text", "text": "mean 多加了 1，去掉它。"},
                {"type": "tool_use", "id": "t4", "name": "write_file",
                 "input": {"path": "calc.py", "content": "def mean(xs):\n    return sum(xs) / len(xs)\n"}}]
    if "passed" not in seen:
        return [{"type": "tool_use", "id": "t5", "name": "run_tests", "input": {}}]
    if "notes.tmp" in seen and "删除" not in seen and "权限拒绝" not in seen:
        return [{"type": "text", "text": "顺手清理临时文件。"},
                {"type": "tool_use", "id": "t6", "name": "delete_file", "input": {"path": "notes.tmp"}}]
    return [{"type": "text", "text": "修好了：mean 去掉了多余的 +1，测试通过。notes.tmp 没有删除权限，留给你处理。"}]


# ---------------------------------------------------------------------------
# Agent 循环
# ---------------------------------------------------------------------------
def run_agent(task, max_turns=10):
    messages = [{"role": "user", "content": task}]
    for turn in range(1, max_turns + 1):
        reply = policy(messages)
        messages.append({"role": "assistant", "content": reply})
        for b in reply:
            if b["type"] == "text": print(f"  [第 {turn} 轮] 模型：{b['text']}")
        calls = [b for b in reply if b["type"] == "tool_use"]
        if not calls:
            return messages, f"完成（{turn} 轮）"
        results = []
        for c in calls:
            pre_tool_hook(c["name"], c["input"])
            if permission(c["name"], c["input"]) == "deny":
                out, err = f"权限拒绝：不允许 {c['name']} {c['input']}", True
            else:
                try:
                    out, err = TOOLS[c["name"]](**c["input"]), False
                except Exception as e:
                    out, err = str(e), True
            post_tool_hook(c["name"], out)
            print(f"  [第 {turn} 轮] 工具 {c['name']}({json.dumps(c['input'], ensure_ascii=False)[:30]}) → "
                  f"{'❌ ' if err else ''}{out.splitlines()[0] if out else ''}")
            results.append({"type": "tool_result", "tool_use_id": c["id"], "content": out, "is_error": err})
        messages.append({"role": "user", "content": results})
    return messages, f"达到轮数上限 {max_turns}，强制停止"


print("=== 1–3. 一个修 bug 的 agent ===")
msgs, status = run_agent("tests 挂了，修一下。")
print("  结果：", status, "；notes.tmp 还在：", "notes.tmp" in FS)
print("  审计日志（钩子记录）：")
for row in AUDIT: print("   ", row)

# ---------------------------------------------------------------------------
# 4–5. 长任务：上下文与成本
# ---------------------------------------------------------------------------
print("\n=== 4–5. 长任务里的上下文管理与成本 ===")
SYSTEM, TOOLDEFS = 2000, 3000        # 系统提示和工具定义的 token 数（固定前缀）
TURNS, READ = 40, 4000               # 40 轮，每轮读一个 4000 token 的文件
STEP = 150                           # 每轮模型自己的输出（推理 + 调用）
PRICE_IN, CACHE_READ, CACHE_WRITE = 1.0, 0.1, 1.25   # 相对价格：缓存命中 0.1 倍，写缓存 1.25 倍


def simulate(cap=None, budget=None, keep_last=4, summary=800, cache=True):
    """返回 (最大上下文, 总输入 token, 相对费用, 压缩次数)。"""
    history = []                     # 每一轮占用的 token 数
    prev_prefix = 0                  # 上一次请求里可被缓存的前缀长度
    total_in = cost = peak = 0
    compactions = 0
    for t in range(TURNS):
        result = READ if cap is None else min(READ, cap)
        history.append(STEP + result)
        ctx = SYSTEM + TOOLDEFS + sum(history)
        if budget and ctx > budget:  # 压缩：把旧的轮次总结成一段摘要，只留最近几轮原文
            history = [summary] + history[-keep_last:]
            compactions += 1
            prev_prefix = SYSTEM + TOOLDEFS          # 历史变了，只有固定前缀还能命中缓存
            ctx = SYSTEM + TOOLDEFS + sum(history)
        peak = max(peak, ctx)
        total_in += ctx
        if cache:
            hit = min(prev_prefix, ctx)
            cost += hit * CACHE_READ + (ctx - hit) * CACHE_WRITE
        else:
            cost += ctx * PRICE_IN
        prev_prefix = ctx            # 只追加不修改：这次的全部输入都是下次的前缀
    return peak, total_in, cost, compactions


rows = [("什么都不做，不用缓存", dict(cache=False)),
        ("什么都不做，用缓存", dict()),
        ("截断到 1000", dict(cap=1000)),
        ("截断 + 压缩（预算 20k）", dict(cap=1000, budget=20000)),
        ("截断 + 压缩，不用缓存", dict(cap=1000, budget=20000, cache=False))]
print(f"  {'策略':<22}{'最大上下文':>10}{'总输入 token':>14}{'相对费用':>12}{'压缩次数':>8}")
for name, kw in rows:
    peak, tot, cost, nc = simulate(**kw)
    print(f"  {name:<22}{peak:>10,}{tot:>14,}{cost:>12,.0f}{nc:>8}")
