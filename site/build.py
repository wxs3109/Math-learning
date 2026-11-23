r"""
把 Math learning 仓库里的笔记和代码生成成一个单文件网页：site/index.html。

设计原则：网页**不依赖任何外部脚本**，打开即正确显示。
- 公式：构建时用 KaTeX（Node，site/tools/render_math.js）渲染成 HTML，KaTeX 字体以 data URI 嵌入页面。
- 代码：构建时用 Pygments 高亮。
- 导航、搜索、本页目录、上一篇 / 下一篇：页面里的一小段原生 JS。
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
SITE = ROOT / "site"
KATEX = SITE / "tools" / "node_modules" / "katex" / "dist"

# ---------------------------------------------------------------- 页面清单
# (页面 id, 源文件, 分组, 侧栏短标题, 一句话简介)
PAGES = [
    ("llm", "topics/llm.md", "大模型原理", "学习路线",
     "整个系列的目录和阅读顺序。"),
    ("llm-00", "topics/llm-00-prerequisites.md", "大模型原理", "00 前置知识速览",
     "一篇讲完正文要用的结论：交叉熵与 KL、最大似然、softmax、反向传播。"),
    ("llm-01", "topics/llm-01-language-model.md", "大模型原理", "01 语言模型",
     "预测下一个 token；困惑度；bigram 模型及其局限；embedding。"),
    ("llm-02", "topics/llm-02-self-attention.md", "大模型原理", "02 自注意力",
     "query / key / value；因果掩码；为什么除以 √d；Transformer 块。"),
    ("llm-p1", "topics/llm-p1-probability.md", "前置知识", "P1 概率论",
     "贝叶斯；期望、方差、协方差；多元高斯；大数定律与中心极限定理；熵与 KL。"),
    ("llm-p2", "topics/llm-p2-statistics.md", "前置知识", "P2 统计",
     "偏差与方差；最大似然；bootstrap；过拟合；正则化就是先验；验证集与测试集。"),
    ("llm-p3", "topics/llm-p3-linear-algebra.md", "前置知识", "P3 线性代数",
     "矩阵乘法的四种读法；谱分解；SVD 与 LoRA；投影；高维几何；张量形状。"),
    ("llm-p4", "topics/llm-p4-calculus-optimization.md", "前置知识", "P4 微积分与优化",
     "矩阵微积分；步长与条件数；动量、SGD、Adam；自动微分。"),
    ("chordal-remove-b", "problems/chordal-remove-b.md", "研究笔记", "去掉 b 的尝试",
     "团分解路线的全过程：归约、反例、迹链刻画，以及 n^Θ(log n) 下界的数值证据。"),
    ("chordal-survey", "problems/chordal-hardcore-mixing.md", "研究笔记", "文献梳理",
     "弦图上 hardcore 模型的快速混合：已有结果与小规模实验。"),
]
SERIES = ["llm", "llm-00", "llm-p1", "llm-p2", "llm-p3", "llm-p4", "llm-01", "llm-02"]
PLANNED = [("03", "动手训练一个小 GPT", "把 00–02 组装起来，用反向传播真正训练。"),
           ("04", "“大”是怎么来的", "scaling law、指令微调与 RLHF、推理优化。")]
CODE_FIRST = ["llm_prereq.py", "llm_p1_probability.py", "llm_p2_statistics.py", "llm_p3_linear_algebra.py",
              "llm_p4_calculus_optimization.py", "llm_basics.py"]

FILE_TO_ID = {Path(src).name: pid for pid, src, *_ in PAGES}
FILE_TO_ID["README.md"] = "home"
code_files = sorted((ROOT / "code").glob("*.py"),
                    key=lambda p: (p.name not in CODE_FIRST,
                                   CODE_FIRST.index(p.name) if p.name in CODE_FIRST else 0, p.name))
CODE_ID = {p.name: "code-" + p.stem.replace("_", "-") for p in code_files}

# ---------------------------------------------------------------- 公式
MATH_RE = re.compile(r"\$\$(.+?)\$\$|\$(?!\s)([^$\n]+?)(?<!\s)\$", re.S)
CODE_RE = re.compile(r"(```.*?```|`[^`\n]+`)", re.S)
MATH = []          # 全站所有公式：{"tex", "display"}，最后一次性交给 KaTeX


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
    t = re.sub(r"\\[a-zA-Z]+", " ", t)
    return re.sub(r"\s+", " ", re.sub(r"[{}_^\\]", "", t)).strip()


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
    name = href.split("/")[-1].split("#")[0]
    if name in FILE_TO_ID: return "#" + FILE_TO_ID[name]
    if name in CODE_ID: return "#" + CODE_ID[name]
    return href


TAG_CLASS = [("已证", "proved"), ("事实", "fact"), ("直觉", "intuition"), ("猜想", "conjecture"),
             ("需核实", "verify"), ("数值", "numeric"), ("推论", "conjecture"), ("V", "fact"), ("I", "verify")]


def tag_html(label):
    cls = next((c for k, c in TAG_CLASS if label.startswith(k)), "fact")
    return f'<span class="tag tag-{cls}">{label}</span>'


def postprocess(body):
    """把 Markdown 生成的 HTML 改成本站的结构。"""
    body = re.sub(r'href="([^"]+)"', lambda m: f'href="{rewrite_link(m.group(1))}"', body)
    body = re.sub(r"<strong>\[([^\]<]{1,40})\]</strong>", lambda m: tag_html(m.group(1)), body)
    body = body.replace("<hr />", "")
    body = re.sub(r"<h2>(\d+)\.\s*(.*?)</h2>", r'<h2><span class="secnum">\1</span><span>\2</span></h2>', body)
    body = re.sub(r"<h3>(\d+\.\d+)\s*(.*?)</h3>", r'<h3><span class="subnum">\1</span><span>\2</span></h3>', body)
    body = re.sub(r"<blockquote>\s*<p>⚠️\s*", '<blockquote class="callout"><p>', body)
    body = body.replace("<table>", '<div class="table-wrap"><table>').replace("</table>", "</table></div>")
    body = re.sub(r"<pre><code>", '<pre class="output"><span class="pre-label">输出</span><code>', body)
    return body


def split_header(body):
    """拆出 <h1> 和紧跟的元信息引用块：(标题, 元信息行, 其余正文)。"""
    m = re.match(r"\s*<h1>(.*?)</h1>\s*(<blockquote>(.*?)</blockquote>)?", body, re.S)
    if not m: return "", [], body
    return m.group(1), re.findall(r"<p>(.*?)</p>", m.group(3) or "", re.S), body[m.end():]


def page_header(eyebrow, title, meta, dek=""):
    meta_html = "".join(f'<p class="meta-line">{x}</p>' for x in meta)
    dek_html = f'<p class="dek">{dek}</p>' if dek else ""
    return (f'<header class="page-head"><p class="eyebrow">{eyebrow}</p>'
            f'<h1>{title}</h1>{dek_html}<div class="meta">{meta_html}</div></header>')


def toc_texts(raw):
    """从 Markdown 源里取二、三级标题，作为右侧目录的文字。"""
    out = []
    for lvl, t in re.findall(r"^(##|###) (.+)$", raw, re.M):
        t = re.sub(r"\$(.+?)\$", lambda m: tex_to_plain(m.group(1)), t)
        t = re.sub(r"\*\*\[[^\]]*\]\*\*|\*\*|`", "", t).strip()
        out.append({"level": len(lvl), "text": t})
    return out


def plain_text(src):
    t = re.sub(r"```.*?```", " ", src, flags=re.S)
    t = re.sub(r"\$(.+?)\$", lambda m: tex_to_plain(m.group(1)), t)
    return re.sub(r"\s+", " ", re.sub(r"[#>*`|\[\]()]", " ", t))


# ---------------------------------------------------------------- 笔记页面
pages = []
for pid, src, group, short, dek in PAGES:
    raw = (ROOT / src).read_text(encoding="utf-8")
    body = postprocess(markdown.markdown(normalize_lists(protect_math(raw)),
                                         extensions=["tables", "fenced_code", "sane_lists"]))
    title, meta, rest = split_header(body)
    title = re.sub(r"^大模型原理\s*·\s*\d+\s*", "", title)
    eyebrow = (f"大模型原理 · 前置知识 {pid[-2:].upper()}" if pid.startswith("llm-p")
               else f"大模型原理 · 第 {pid[-2:]} 篇" if pid in SERIES[1:] else group)
    pages.append({"id": pid, "group": group, "short": short, "dek": dek, "toc": toc_texts(raw),
                  "html": page_header(eyebrow, title, meta, dek) + f'<div class="prose">{rest}</div>',
                  "text": plain_text(raw)})

# ---------------------------------------------------------------- 代码页面
fmt = HtmlFormatter(nowrap=True)
for p in code_files:
    code = p.read_text(encoding="utf-8")
    doc = re.search(r'r?"""(.*?)"""', code, re.S)
    summary = html.escape(doc.group(1).strip().splitlines()[0]) if doc else ""
    body = (page_header("代码", f"<code>{p.name}</code>",
                        [f"运行：<code>cd code &amp;&amp; python3 {p.name}</code>"], summary)
            + f'<div class="prose"><pre class="code"><span class="pre-label">code/{p.name}</span>'
              f'<code>{highlight(code, PythonLexer(), fmt)}</code></pre></div>')
    pages.append({"id": CODE_ID[p.name], "group": "代码", "short": p.name, "dek": summary, "toc": [],
                  "html": body, "text": plain_text(code)})

# ---------------------------------------------------------------- 首页
by = {p["id"]: p for p in pages}


def card(pid, num=None, status="已写", status_cls="done"):
    p = by[pid]
    title = p["short"].split(" ", 1)[-1] if num else p["short"]
    n = f'<span class="card-num">{num}</span>' if num else ""
    return (f'<a class="card" href="#{pid}">{n}<span class="card-title">{title}</span>'
            f'<span class="card-dek">{p["dek"]}</span><span class="status status-{status_cls}">{status}</span></a>')


planned = "".join(f'<div class="card card-planned"><span class="card-num">{n}</span><span class="card-title">{t}</span>'
                  f'<span class="card-dek">{d}</span><span class="status">待写</span></div>' for n, t, d in PLANNED)
home = f"""
<header class="hero">
  <p class="eyebrow">学习与研究笔记</p>
  <h1>Math learning</h1>
  <p class="dek">每篇按同一个顺序写：问题、背景、前置知识、推导、数值实验、结论与疑问、参考资料。每篇都配一份可以直接运行的代码。</p>
</header>
<section class="track">
  <div class="track-head"><h2>大模型原理</h2><a href="#llm">学习路线 →</a></div>
  <p class="track-dek">从"预测下一个词"讲到 Transformer，目标是能自己从头推一遍、写一遍。</p>
  <div class="cards">{card("llm-00", "00")}{card("llm-01", "01")}{card("llm-02", "02")}{planned}</div>
</section>
<section class="track">
  <div class="track-head"><h2>前置知识</h2><a href="#llm-00">先看速览 →</a></div>
  <p class="track-dek">概率、统计、线性代数、微积分与优化各一章：完整推导，每章配一份可运行的实验代码。</p>
  <div class="cards">{card("llm-p1", "P1")}{card("llm-p2", "P2")}{card("llm-p3", "P3")}{card("llm-p4", "P4")}</div>
</section>
<section class="track">
  <div class="track-head"><h2>研究笔记：弦图上的混合时间</h2></div>
  <p class="track-dek">能否去掉 bounded separator 限制，得到弦图上 Glauber 动力学的多项式混合？</p>
  <div class="cards cards-2">{card("chordal-remove-b", status="进行中", status_cls="active")}{card("chordal-survey")}</div>
</section>
<section class="track">
  <div class="track-head"><h2>代码</h2><a href="#{CODE_ID['llm_prereq.py']}">浏览代码 →</a></div>
  <p class="track-dek">仓库里全部 {len(code_files)} 个 Python 文件，只依赖 numpy、scipy、networkx，均可直接运行。</p>
</section>"""
pages.insert(0, {"id": "home", "group": "首页", "short": "首页", "dek": "", "toc": [], "html": home,
                 "text": plain_text((ROOT / "README.md").read_text(encoding="utf-8"))})

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
meta = [{k: p[k] for k in ("id", "group", "short", "dek", "toc")} for p in pages]
search = [{"id": p["id"], "text": p["text"]} for p in pages]
articles = "\n".join(f'<article class="doc" id="page-{p["id"]}" data-page="{p["id"]}" hidden>{p["html"]}</article>'
                     for p in pages)
tpl = (SITE / "template.html").read_text(encoding="utf-8")
out = (tpl.replace("/*__KATEX_CSS__*/", katex_css)
          .replace("/*__META__*/", json.dumps(meta, ensure_ascii=False))
          .replace("/*__SEARCH__*/", json.dumps(search, ensure_ascii=False))
          .replace("/*__SERIES__*/", json.dumps(SERIES))
          .replace("<!--__ARTICLES__-->", articles))
(SITE / "index.html").write_text(out, encoding="utf-8")
errors = sum(r.count("katex-error") + r.count("math-error") for r in rendered)
print(f"写出 site/index.html：{len(pages)} 个页面，{len(MATH)} 个公式（渲染报错 {errors} 个），{len(out) / 1024 / 1024:.2f} MB")
