r"""
第 9 章 · 提示工程的配套代码（llm/ch09-prompt-engineering.md）。只依赖 numpy：python3 ch09_prompting.py

不调用任何真实模型，而是把提示工程背后的几件"机械"事实单独拿出来验证：
1. 聊天模板：system / user / assistant 消息最终被拼成一条 token 序列。
2. 解码参数：temperature、top-k、top-p 怎样改变下一个 token 的分布。
3. 少样本（few-shot）= 贝叶斯更新：例子越多，"任务是什么"的后验越集中。
4. 受约束解码：用状态机屏蔽非法 token，让输出 100% 符合格式。
"""
import re
import numpy as np

rng = np.random.default_rng(0)

# ---------------------------------------------------------------------------
# 1. 聊天模板：消息列表 → 一条字符串（真实模型再把它切成 token）
# ---------------------------------------------------------------------------
print("=== 1. 聊天模板 ===")


def render_chat(messages, add_generation_prompt=True):
    """一个 ChatML 风格的模板。每个模型家族的特殊标记不同，但结构相同。"""
    out = []
    for m in messages:
        out.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>")
    if add_generation_prompt:
        out.append("<|im_start|>assistant\n")      # 模型从这里开始"续写"
    return "\n".join(out)


msgs = [
    {"role": "system", "content": "你是严谨的数学助教。只回答数学问题。"},
    {"role": "user", "content": "1+1=?"},
    {"role": "assistant", "content": "2"},
    {"role": "user", "content": "2+2=?"},
]
print(render_chat(msgs))
print("--> 模型看到的只是这一条序列；'角色'只是序列里的特殊标记。")

# ---------------------------------------------------------------------------
# 2. 解码参数
# ---------------------------------------------------------------------------
print("\n=== 2. temperature / top-k / top-p ===")
vocab = ["的", "是", "了", "在", "数学", "模型", "猫", "量子", "香蕉", "。"]
logits = np.array([3.0, 2.6, 2.2, 1.8, 1.5, 1.2, 0.3, -0.5, -1.5, 0.8])


def softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def entropy_bits(p):
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


print("  T      最大概率   熵(bit)   有效候选数 2^H")
for T in [0.2, 0.5, 1.0, 1.5, 3.0]:
    p = softmax(logits / T)
    H = entropy_bits(p)
    print(f"  {T:<5}  {p.max():.3f}      {H:.3f}     {2 ** H:.2f}")


def top_k_filter(p, k):
    q = np.zeros_like(p)
    idx = np.argsort(p)[::-1][:k]
    q[idx] = p[idx]
    return q / q.sum()


def top_p_filter(p, top_p):
    """nucleus：按概率从大到小累加，保留累计概率刚好超过 top_p 的最小集合。"""
    order = np.argsort(p)[::-1]
    csum = np.cumsum(p[order])
    keep = order[: int(np.searchsorted(csum, top_p) + 1)]
    q = np.zeros_like(p)
    q[keep] = p[keep]
    return q / q.sum()


p1 = softmax(logits)
for name, q in [("原分布", p1), ("top-k=3", top_k_filter(p1, 3)), ("top-p=0.9", top_p_filter(p1, 0.9))]:
    support = [vocab[i] for i in np.argsort(q)[::-1] if q[i] > 0]
    print(f"  {name:<9} 候选 {len(support):>2} 个：{' '.join(support)}")
draws = rng.choice(len(vocab), size=100000, p=p1)
print(f"  T=1 采样 10 万次，'香蕉' 出现 {np.mean(draws == 8):.4f}（理论 {p1[8]:.4f}）；"
      f"top-p=0.9 之后它的概率是 {top_p_filter(p1, 0.9)[8]:.1f}")

# ---------------------------------------------------------------------------
# 3. few-shot = 对"任务"的贝叶斯更新
# ---------------------------------------------------------------------------
print("\n=== 3. 少样本例子 = 贝叶斯更新 ===")
# 假想模型在预训练里见过几种"任务"。提示里的例子是证据，模型隐式地推断当前是哪个任务。
tasks = {
    "大写": lambda s: s.upper(),
    "反转": lambda s: s[::-1],
    "首字母": lambda s: s[0],
    "长度": lambda s: str(len(s)),
    "原样": lambda s: s,
}
names = list(tasks)
eps = 0.05                      # 例子本身有噪声：以 eps 概率"不符合"任何任务


def posterior(examples):
    logp = np.log(np.ones(len(names)) / len(names))          # 均匀先验
    for x, y in examples:
        for i, n in enumerate(names):
            logp[i] += np.log(1 - eps if tasks[n](x) == y else eps)
    return softmax(logp)


examples = [("a", "a"), ("cat", "CAT"), ("dog", "DOG"), ("moon", "MOON")]
print("  例子数   " + "  ".join(f"{n:>4}" for n in names))
for k in range(len(examples) + 1):
    post = posterior(examples[:k])
    print(f"  {k:<7}  " + "  ".join(f"{v:5.3f}" for v in post))
print("  第 1 个例子 a→a 同时符合'反转''首字母''原样'，反而把'大写'压低了；例子要有代表性，多给几个不同的例子后验才集中。")

# ---------------------------------------------------------------------------
# 4. 受约束解码
# ---------------------------------------------------------------------------
print("\n=== 4. 受约束解码：输出必须匹配 {\"age\": 数字} ===")
chars = list('{}":age 0123456789abcxyz,')
target = re.compile(r'\{"age": \d{1,3}\}')
prefix = '{"age": '


def allowed(s):
    """简单状态机：给出当前前缀 s 之后合法的下一个字符集合（None 表示已结束）。"""
    if len(s) < len(prefix):
        return {prefix[len(s)]}
    digits = s[len(prefix):]
    if digits.endswith("}"):
        return None
    nd = len(digits)
    ok = set()
    if nd < 3: ok |= set("0123456789")
    if nd >= 1: ok.add("}")
    return ok


def generate(constrained, max_len=16):
    s = ""
    for _ in range(max_len):
        z = rng.normal(size=len(chars)) * 2.0         # 一个"随机模型"：每步给出任意 logits
        if constrained:
            ok = allowed(s)
            if ok is None: break
            mask = np.array([c in ok for c in chars])
            z = np.where(mask, z, -np.inf)            # 非法 token 的 logit 设为 −∞，概率恰好为 0
        p = softmax(z)
        s += chars[rng.choice(len(chars), p=p)]
        if constrained and s.endswith("}") and len(s) > len(prefix): break
    return s


for flag in [False, True]:
    outs = [generate(flag) for _ in range(2000)]
    valid = np.mean([bool(target.fullmatch(o)) for o in outs])
    print(f"  {'有约束' if flag else '无约束'}：2000 次生成中格式合法的比例 {valid:.3f}；例子 {outs[0]!r}")
