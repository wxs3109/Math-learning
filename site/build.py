r"""
把 llm/ 里的《大模型原理》笔记和代码生成成一个单文件网页：site/index.html。

结构是一本书：部分 → 章 → 节。侧栏是可展开的章节目录，地址 #ch1/1.3 直接定位到第 1 章 1.3 节。
设计原则：网页**不依赖任何外部脚本**，打开即正确显示。
- 公式：构建时用 KaTeX（Node，site/tools/render_math.js）渲染成 HTML，KaTeX 字体以 data URI 嵌入页面。
- 代码：构建时用 Pygments 高亮。
- 导航、搜索、上一章 / 下一章：页面里的一小段原生 JS。
依赖：pip install markdown pygments；cd site/tools && npm install katex@0.16.11
用法：python3 site/build.py
"""
import base64, html, json, re, subprocess
from pathlib import Path
import markdown
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import HtmlFormatter

ROOT = Path(__file__).resolve().parent.parent
BOOK = ROOT / "llm"
SITE = ROOT / "site"
KATEX = SITE / "tools" / "node_modules" / "katex" / "dist"

# ---------------------------------------------------------------- 全书结构
# 每章：(页面 id, 源文件, 章号, 章名, 一句话简介)；源文件为 None 表示待写
PARTS = [
    ("第一部分 · 数学基础", [
        ("ch1", "ch01-probability.md", "1", "概率论",
         "贝叶斯；期望、方差、协方差；多元高斯；大数定律与中心极限定理；熵与 KL。"),
        ("ch2", "ch02-statistics.md", "2", "统计",
         "偏差与方差；最大似然；bootstrap；过拟合；正则化就是先验；验证集与测试集。"),
        ("ch3", "ch03-linear-algebra.md", "3", "线性代数",
         "矩阵乘法的四种读法；谱分解；SVD 与 LoRA；投影；高维几何；张量形状。"),
        ("ch4", "ch04-calculus-optimization.md", "4", "微积分与优化",
         "矩阵微积分；步长与条件数；动量、SGD、Adam；自动微分。"),
    ]),
    ("第二部分 · 语言模型与 Transformer", [
        ("ch5", "ch05-language-model.md", "5", "语言模型",
         "预测下一个 token；交叉熵与困惑度；bigram 模型及其局限；embedding。"),
        ("ch6", "ch06-self-attention.md", "6", "自注意力与 Transformer",
         "query / key / value；因果掩码；为什么除以 √d；Transformer 块。"),
        ("ch7", None, "7", "动手训练一个小 GPT", "把前六章组装起来，用反向传播真正训练。"),
        ("ch8", None, "8", "“大”是怎么来的", "scaling law、指令微调与 RLHF、推理优化。"),
    ]),
    ("第三部分 · 工程实践", [
        ("ch9", "ch09-prompt-engineering.md", "9", "提示工程",
         "提示就是条件；写清楚、给例子、让模型先想；解码参数；受约束解码；上下文工程。"),
        ("ch10", "ch10-tool-calling.md", "10", "工具调用",
         "模型只写出调用请求；工具定义与设计；tool_use / tool_result；错误处理；严格模式。"),
        ("ch11", "ch11-agent-harness.md", "11", "Agent 与 harness",
         "workflow 还是 agent；agent 循环；上下文管理与压缩；权限与钩子；提示缓存。"),
        ("ch12", "ch12-mcp.md", "12", "MCP",
         "host / client / server；tools、resources、prompts；无状态的 2026-07-28 规范；安全。"),
        ("ch13", "ch13-skills.md", "13", "Agent Skills",
         "SKILL.md 与渐进式披露；description 决定触发；和 MCP、工具、子 agent 的分工。"),
        ("ch14", "ch14-langchain.md", "14", "LangChain 与 LangGraph",
         "统一接口与 LCEL；create_agent 与中间件；状态图、检查点与人工介入。"),
        ("ch15", "ch15-rag.md", "15", "检索增强生成（RAG）",
         "BM25 与 embedding；混合检索与重排；上下文化分块；recall@k 与 MRR。"),
        ("ch16", "ch16-evals-production.md", "16", "评测、安全与上线",
         "误差累积；pass@k 与 pass^k；置信区间；LLM 评委；提示注入；成本与延迟。"),
    ]),
    ("第四部分 · 系统与基础设施", [
        ("ch17", "ch17-inference-serving.md", "17", "推理服务",
         "prefill 与 decode；KV cache 与 roofline；连续批处理、PagedAttention；投机解码；量化；排队与容量规划。"),
    ]),
    ("第五部分 · 实战项目", [
        ("ch18", "ch18-project-design.md", "18", "实战项目：设计",
         "读书与面试陪练助手：架构、流式前端、模型网关、检索、面试官 agent、记忆、评测、成本与安全；用附录 C 做检索基线。"),
    ]),
    ("附录", [
        ("appA", "appendix-a-math-quick-reference.md", "A", "数学速查",
         "一页讲完第 5、6 章要用的结论：交叉熵与 KL、最大似然、softmax、反向传播。"),
        ("appC", "appendix-c-interview-questions.md", "C", "面试题清单",
         "中英文面试题 366 道，按主题归类，每题标出本书哪一节能回答，并汇总还没覆盖的空白。"),
    ]),
]
CHAPTERS = [c for _, cs in PARTS for c in cs]
FILE_TO_ID = {src: pid for pid, src, *_ in CHAPTERS if src}
FILE_TO_ID["README.md"] = "home"
code_files = sorted((BOOK / "code").glob("*.py"))
CODE_ID = {p.name: "code-" + p.stem.replace("_", "-") for p in code_files}

# ---------------------------------------------------------------- 公式
MATH_RE = re.compile(r"\$\$(.+?)\$\$|\$(?!\s)([^$\n]+?)(?<!\s)\$", re.S)
CODE_RE = re.compile(r"(```.*?```|`[^`\n]+`)", re.S)
MATH = []          # 全书所有公式：{"tex", "display"}，最后一次性交给 KaTeX


def protect_math(text):
    """把代码之外的 $…$ / $$…$$ 换成占位符 MATHPH{n}END，公式本身存进 MATH。"""
    def sub(m):
        display = m.group(1) is not None
        MATH.append({"tex": (m.group(1) if display else m.group(2)).strip(), "display": display})
        return f"MATHPH{len(MATH) - 1}END"
    parts = CODE_RE.split(text)
    for i in range(0, len(parts), 2):
        parts[i] = MATH_RE.sub(sub, parts[i])
    return "".join(parts)


def tex_to_plain(tex):
    """公式的纯文本近似，用于目录和搜索。"""
    t = re.sub(r"\\(mathrm|mathbf|text|operatorname|mathcal|mathbb)\{([^}]*)\}", r"\2", tex)
    t = t.replace(r"\sqrt", "√").replace(r"\top", "ᵀ")
    t = re.sub(r"\\[a-zA-Z]+", " ", t)
    return re.sub(r"\s+", " ", re.sub(r"[{}_^\\]", "", t)).strip()


def plain_title(t):
    t = re.sub(r"\$(.+?)\$", lambda m: tex_to_plain(m.group(1)), t)
    return re.sub(r"\*\*\[[^\]]*\]\*\*|\*\*|`", "", t).strip()


# ---------------------------------------------------------------- Markdown 处理
LIST_RE = re.compile(r"^\s*([-*+]|\d+\.)\s+")


def normalize_lists(text):
    """列表前补空行、嵌套缩进统一成 4 的倍数、引用块逐行分段（代码块内不动）。"""
    out, in_code, prev = [], False, ""
    for line in text.split("\n"):
        if line.lstrip().startswith("```"):
            in_code = not in_code; out.append(line); prev = line; continue
        if in_code:
            out.append(line); prev = line; continue
        stripped = line.lstrip(" "); k = len(line) - len(stripped)
        if stripped and k:
            line = " " * (4 * ((k + 2) // 3)) + stripped
        if LIST_RE.match(line) and prev.strip() and not LIST_RE.match(prev) and not prev.startswith(" ") \
                and not prev.lstrip().startswith(">") and not line.startswith(" "):
            out.append("")
        if line.startswith(">") and prev.startswith(">"):
            out.append(">")
        out.append(line); prev = line
    return "\n".join(out)


def rewrite_link(href):
    """书内链接 → #页面 或 #页面/节号。"""
    path, _, frag = href.partition("#")
    name = path.split("/")[-1]
    if name in FILE_TO_ID:
        return "#" + FILE_TO_ID[name] + (f"/{frag}" if frag else "")
    if name in CODE_ID: return "#" + CODE_ID[name]
    return href


TAG_CLASS = [("已证", "proved"), ("事实", "fact"), ("直觉", "intuition"), ("猜想", "conjecture"),
             ("需核实", "verify"), ("数值", "numeric"), ("经验", "practice")]


def tag_html(label):
    cls = next((c for k, c in TAG_CLASS if label.startswith(k)), "fact")
    return f'<span class="tag tag-{cls}">{label}</span>'


NUM = r"([0-9A-Z]+(?:\.\d+)+)"


def postprocess(body, pid):
    """把 Markdown 生成的 HTML 改成本站的结构：节号、锚点、标签、表格、代码块。"""
    body = re.sub(r'href="([^"]+)"', lambda m: f'href="{rewrite_link(m.group(1))}"', body)
    body = re.sub(r"<strong>\[([^\]<]{1,40})\]</strong>", lambda m: tag_html(m.group(1)), body)
    body = body.replace("<hr />", "")
    body = re.sub(rf"<h([23])>{NUM}\s+(.*?)</h\1>",
                  lambda m: f'<h{m.group(1)} id="{pid}--{m.group(2)}"><span class="secnum">{m.group(2)}</span>'
                            f'<span>{m.group(3)}</span></h{m.group(1)}>', body)
    body = re.sub(r"<h2>(导读|参考资料)</h2>", lambda m: f'<h2 class="plain" id="{pid}--{m.group(1)}">{m.group(1)}</h2>', body)
    body = re.sub(r"<blockquote>\s*<p>⚠️\s*", '<blockquote class="callout"><p>', body)
    body = body.replace("<table>", '<div class="table-wrap"><table>').replace("</table>", "</table></div>")
    body = re.sub(r"<pre><code>", '<pre class="output"><span class="pre-label">输出</span><code>', body)
    return body


def md(raw, pid):
    return postprocess(markdown.markdown(normalize_lists(protect_math(raw)),
                                         extensions=["tables", "fenced_code", "sane_lists"]), pid)


def plain_text(src):
    t = re.sub(r"```.*?```", " ", src, flags=re.S)
    t = re.sub(r"\$(.+?)\$", lambda m: tex_to_plain(m.group(1)), t)
    return re.sub(r"\s+", " ", re.sub(r"[#>*`|\[\]()]", " ", t))


def sections_of(raw):
    """从 Markdown 源里取带编号的二级标题：[{id: "1.3", title}]。"""
    return [{"id": n, "title": plain_title(t)} for n, t in re.findall(rf"^## {NUM} (.+)$", raw, re.M)]


# ---------------------------------------------------------------- 各章页面
pages, order = [], ["home"]
for part, chs in PARTS:
    for pid, src, num, title, dek in chs:
        label = f"附录 {num}" if num.isalpha() else f"第 {num} 章"
        if not src:
            pages.append({"id": pid, "kind": "planned", "num": num, "title": title, "label": label,
                          "part": part, "dek": dek, "sections": [], "html": "", "text": ""})
            continue
        raw = (BOOK / src).read_text(encoding="utf-8")
        body = md(raw, pid)
        m = re.match(r"\s*<h1>.*?</h1>\s*(<blockquote>(.*?)</blockquote>)?", body, re.S)
        meta = re.findall(r"<p>(.*?)</p>", m.group(2) or "", re.S)
        rest = body[m.end():]
        head = (f'<header class="page-head chapter-head"><p class="eyebrow"><span>{label}</span>{part}</p>'
                f'<h1>{title}</h1><p class="dek">{dek}</p>'
                + "".join(f'<p class="meta-line">{x}</p>' for x in meta) + "</header>")
        pages.append({"id": pid, "kind": "chapter", "num": num, "title": title, "label": label, "part": part,
                      "dek": dek, "sections": sections_of(raw), "html": head + f'<div class="prose">{rest}</div>',
                      "text": plain_text(raw)})
        order.append(pid)

# ---------------------------------------------------------------- 代码页面（附录 B）
fmt = HtmlFormatter(nowrap=True)
for p in code_files:
    code = p.read_text(encoding="utf-8")
    doc = re.search(r'r?"""(.*?)"""', code, re.S)
    summary = html.escape(doc.group(1).strip().splitlines()[0]) if doc else ""
    body = (f'<header class="page-head"><p class="eyebrow"><span>附录 B</span>代码</p>'
            f'<h1><code>{p.name}</code></h1><p class="dek">{summary}</p>'
            f'<p class="meta-line">运行：<code>cd llm/code &amp;&amp; python3 {p.name}</code></p></header>'
            f'<div class="prose"><pre class="code"><span class="pre-label">llm/code/{p.name}</span>'
            f'<code>{highlight(code, PythonLexer(), fmt)}</code></pre></div>')
    pages.append({"id": CODE_ID[p.name], "kind": "code", "num": "", "title": p.name, "label": "附录 B",
                  "part": "附录", "dek": summary, "sections": [], "html": body, "text": plain_text(code)})

# ---------------------------------------------------------------- 封面与目录
readme = (BOOK / "README.md").read_text(encoding="utf-8")
intro_md = re.split(r"^## 目录\s*$", readme, flags=re.M)
before = re.sub(r"^# .*\n(>.*\n)?", "", intro_md[0])
after = re.split(r"^## ", intro_md[1], maxsplit=1, flags=re.M)
after = "## " + after[1] if len(after) > 1 else ""
by = {p["id"]: p for p in pages}


def contents_html():
    out = []
    for part, chs in PARTS:
        items = []
        for pid, src, num, title, dek in chs:
            p = by[pid]
            secs = "".join(f'<li><a href="#{pid}/{s["id"]}"><span class="toc-num">{s["id"]}</span>{html.escape(s["title"])}</a></li>'
                           for s in p["sections"])
            if not src:
                items.append(f'<li class="toc-chap planned"><div class="toc-chap-head"><span class="toc-chap-num">{num}</span>'
                             f'<span class="toc-chap-title">{title}</span><span class="status">待写</span></div>'
                             f'<p class="toc-dek">{dek}</p></li>')
                continue
            items.append(f'<li class="toc-chap"><a class="toc-chap-head" href="#{pid}"><span class="toc-chap-num">{num}</span>'
                         f'<span class="toc-chap-title">{title}</span></a><p class="toc-dek">{dek}</p><ol class="toc-secs">{secs}</ol></li>')
        if part == "附录":
            codes = "".join(f'<li><a href="#{CODE_ID[c.name]}"><code>{c.name}</code></a></li>' for c in code_files)
            items.append(f'<li class="toc-chap"><a class="toc-chap-head" href="#{CODE_ID[code_files[0].name]}">'
                         f'<span class="toc-chap-num">B</span><span class="toc-chap-title">代码</span></a>'
                         f'<p class="toc-dek">每章的配套代码，可以直接运行。大多只依赖 numpy 或标准库。</p><ol class="toc-secs code-list">{codes}</ol></li>')
        out.append(f'<section class="toc-part"><h2 class="part-title">{part}</h2><ol class="toc-chaps">{"".join(items)}</ol></section>')
    return "".join(out)


home = ('<header class="cover"><p class="eyebrow"><span>学习笔记</span></p><h1>大模型原理</h1>'
        '<p class="dek">从概率论讲到 Transformer，再到提示、工具、agent、MCP、Skills、RAG 和评测。每章都有推导或原理，以及一份可以直接运行的代码。</p></header>'
        f'<div class="prose">{md(before, "home")}</div>'
        f'<h2 class="contents-title">目录</h2>{contents_html()}'
        f'<div class="prose">{md(after, "home")}</div>')
pages.insert(0, {"id": "home", "kind": "home", "num": "", "title": "封面与目录", "label": "", "part": "",
                 "dek": "", "sections": [], "html": home, "text": plain_text(readme)})

# ---------------------------------------------------------------- 一次性渲染全部公式
rendered = json.loads(subprocess.run(["node", str(SITE / "tools" / "render_math.js")],
                                     input=json.dumps(MATH), capture_output=True, text=True, check=True).stdout)


def glue(m):
    """行内公式和紧跟的中文标点放进同一个不换行的盒子，避免句号被挤到下一行开头。"""
    html_math, punct = rendered[int(m.group(1))], m.group(2)
    return f'<span class="nobr">{html_math}{punct}</span>' if punct and not MATH[int(m.group(1))]["display"] else html_math + punct


for pg in pages:
    pg["html"] = re.sub(r"MATHPH(\d+)END([，。、；：！？）]?)", glue, pg["html"])

# ---------------------------------------------------------------- KaTeX 样式，字体内嵌为 data URI
css = (KATEX / "katex.min.css").read_text()


def embed_font(m):
    data = base64.b64encode((KATEX / "fonts" / f"{m.group(1)}.woff2").read_bytes()).decode()
    return f'src:url(data:font/woff2;base64,{data}) format("woff2")'


katex_css = re.sub(r'src:url\(fonts/([^.]+)\.woff2\) format\("woff2"\)(,url\([^)]*\) format\("[^"]*"\))*',
                   embed_font, css)

# ---------------------------------------------------------------- 写出
nav = [{"part": part, "items": [pid for pid, *_ in chs]} for part, chs in PARTS]
meta = [{k: p[k] for k in ("id", "kind", "num", "title", "label", "part", "dek", "sections")} for p in pages]
search = [{"id": p["id"], "text": p["text"]} for p in pages if p["text"]]
articles = "\n".join(f'<article class="doc" id="page-{p["id"]}" data-page="{p["id"]}" hidden>{p["html"]}</article>'
                     for p in pages if p["html"])
tpl = (SITE / "template.html").read_text(encoding="utf-8")
out = (tpl.replace("/*__KATEX_CSS__*/", katex_css)
          .replace("/*__META__*/", json.dumps(meta, ensure_ascii=False))
          .replace("/*__NAV__*/", json.dumps(nav, ensure_ascii=False))
          .replace("/*__SEARCH__*/", json.dumps(search, ensure_ascii=False))
          .replace("/*__ORDER__*/", json.dumps(order))
          .replace("<!--__ARTICLES__-->", articles))
(SITE / "index.html").write_text(out, encoding="utf-8")
errors = sum("katex-error" in h for h in rendered)
print(f"写出 site/index.html：{len(order) - 1} 章，{len(code_files)} 个代码文件，"
      f"{len(MATH)} 个公式（渲染报错 {errors} 个），{len(out) / 1e6:.2f} MB")
