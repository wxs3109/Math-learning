r"""
把 Math learning 仓库里的笔记和代码生成成一个单文件的在线文档平台：site/index.html。

- 每个页面是一个 <article>，用 URL 的 #页面名 切换（例如 #llm-00）。
- 公式交给 MathJax（SVG 输出，不依赖字体文件）在浏览器里渲染；转换前先把公式抠出来，
  免得 Markdown 解析把下划线、反斜杠吃掉。
- 代码文件整页展示，用 highlight.js 高亮。
- 侧栏导航、全文搜索、本页目录、上一篇 / 下一篇都在页面里的脚本完成。
用法：python3 site/build.py
"""
import html, json, re
from pathlib import Path
import markdown

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "site" / "index.html"

# 页面清单：(页面 id, 源文件, 分组)。顺序就是侧栏顺序，也决定"上一篇 / 下一篇"。
PAGES = [
    ("home", "README.md", "首页"),
    ("llm", "topics/llm.md", "大模型原理"),
    ("llm-00", "topics/llm-00-prerequisites.md", "大模型原理"),
    ("llm-01", "topics/llm-01-language-model.md", "大模型原理"),
    ("llm-02", "topics/llm-02-self-attention.md", "大模型原理"),
    ("chordal-survey", "problems/chordal-hardcore-mixing.md", "研究：弦图上的混合时间"),
    ("chordal-remove-b", "problems/chordal-remove-b.md", "研究：弦图上的混合时间"),
]
SERIES = ["llm", "llm-00", "llm-01", "llm-02"]          # 有上一篇 / 下一篇的系列
CODE_GROUP_FIRST = ["llm_prereq.py", "llm_basics.py"]    # 代码区里排在前面的文件

FILE_TO_ID = {Path(src).name: pid for pid, src, _ in PAGES}
code_files = sorted((ROOT / "code").glob("*.py"),
                    key=lambda p: (p.name not in CODE_GROUP_FIRST, CODE_GROUP_FIRST.index(p.name)
                                   if p.name in CODE_GROUP_FIRST else 0, p.name))
CODE_ID = {p.name: "code-" + p.stem.replace("_", "-") for p in code_files}


# ---------------------------------------------------------------- Markdown → HTML
MATH_RE = re.compile(r"\$\$(.+?)\$\$|\$(?!\s)([^$\n]+?)(?<!\s)\$", re.S)
CODE_RE = re.compile(r"(```.*?```|`[^`\n]+`)", re.S)


def protect_math(text):
    """把代码之外的 $…$ / $$…$$ 换成占位符，返回 (新文本, 公式列表)。"""
    store = []
    def sub_math(m):
        if m.group(1) is not None:
            tex = "\\[" + m.group(1) + "\\]"
        else:
            tex = "\\(" + m.group(2) + "\\)"
        store.append(tex)
        return f"MATHPH{len(store) - 1}END"
    parts = CODE_RE.split(text)
    for i in range(0, len(parts), 2):                    # 偶数下标是代码之外的文本
        parts[i] = MATH_RE.sub(sub_math, parts[i])
    return "".join(parts), store


def rewrite_link(href):
    """仓库内的相对链接改成站内的 #页面名。"""
    name = href.split("/")[-1].split("#")[0]
    if name in FILE_TO_ID:
        return "#" + FILE_TO_ID[name]
    if name in CODE_ID:
        return "#" + CODE_ID[name]
    return href


LIST_RE = re.compile(r"^\s*([-*+]|\d+\.)\s+")


def normalize_lists(text):
    """
    笔记按 GitHub 的习惯写：列表直接跟在段落后面、嵌套只缩进 2～3 格。
    Python-Markdown 要求列表前有空行、嵌套缩进 4 格，这里统一改过来（代码块内不动）。
    引用块的每一行单独成段，免得元信息挤成一行。
    """
    out, in_code, prev = [], False, ""
    for line in text.split("\n"):
        if line.lstrip().startswith("```"):
            in_code = not in_code
            out.append(line); prev = line; continue
        if in_code:
            out.append(line); prev = line; continue
        stripped = line.lstrip(" "); k = len(line) - len(stripped)
        if stripped and k:                                   # 缩进 1–3 → 4，4–6 → 8，……
            line = " " * (4 * ((k + 2) // 3)) + stripped
        is_item = bool(LIST_RE.match(line))
        if is_item and prev.strip() and not LIST_RE.match(prev) and not prev.startswith(" ") \
                and not prev.lstrip().startswith(">") and not line.startswith(" "):
            out.append("")                                   # 段落后面紧跟列表：补一个空行
        if line.startswith(">") and prev.startswith(">"):
            out.append(">")                                  # 引用块逐行分段
        out.append(line); prev = line
    return "\n".join(out)


def md_to_html(text):
    text, store = protect_math(text)
    text = normalize_lists(text)
    body = markdown.markdown(text, extensions=["tables", "fenced_code", "sane_lists"])
    body = re.sub(r'href="([^"]+)"', lambda m: f'href="{rewrite_link(m.group(1))}"', body)
    body = re.sub(r"MATHPH(\d+)END", lambda m: html.escape(store[int(m.group(1))]), body)
    # 表格外面套一层，窄屏时可以横向滚动
    body = body.replace("<table>", '<div class="table-wrap"><table>').replace("</table>", "</table></div>")
    body = body.replace('<code class="language-', '<code class="hljs language-')
    return body


def plain_text(src):
    """给搜索用的纯文本：去掉 Markdown 记号和公式里的反斜杠命令。"""
    t = re.sub(r"```.*?```", " ", src, flags=re.S)
    t = re.sub(r"\\[a-zA-Z]+", " ", t)
    t = re.sub(r"[#>*`|$\\{}_^\[\]()]", " ", t)
    return re.sub(r"\s+", " ", t)


pages = []
for pid, src, group in PAGES:
    raw = (ROOT / src).read_text(encoding="utf-8")
    title = re.search(r"^# (.+)$", raw, re.M).group(1).strip()
    pages.append({"id": pid, "group": group, "title": title, "src": src,
                  "html": md_to_html(raw), "text": plain_text(raw)})

for p in code_files:
    code = p.read_text(encoding="utf-8")
    doc = re.search(r'r?"""(.*?)"""', code, re.S)
    summary = doc.group(1).strip().splitlines()[0] if doc else ""
    pid = CODE_ID[p.name]
    body = (f'<h1><code>code/{p.name}</code></h1>'
            f'<p class="code-meta">{html.escape(summary)}</p>'
            f'<p class="code-meta">运行：<code>cd code &amp;&amp; python3 {p.name}</code></p>'
            f'<pre><code class="hljs language-python">{html.escape(code)}</code></pre>')
    pages.append({"id": pid, "group": "代码", "title": p.name, "src": f"code/{p.name}",
                  "html": body, "text": plain_text(code)})

# 侧栏里显示的短标题
SHORT = {"home": "项目首页", "llm": "学习路线", "llm-00": "00 前置知识", "llm-01": "01 语言模型",
         "llm-02": "02 自注意力与 Transformer", "chordal-survey": "文献梳理与小规模实验",
         "chordal-remove-b": "去掉 b 的尝试（主笔记）"}
for pg in pages:
    pg["short"] = SHORT.get(pg["id"], pg["title"])

meta = [{k: pg[k] for k in ("id", "group", "title", "short", "src")} for pg in pages]
search_index = [{"id": pg["id"], "text": pg["text"]} for pg in pages]
articles = "\n".join(f'<article class="doc" id="page-{pg["id"]}" data-page="{pg["id"]}" hidden>{pg["html"]}</article>'
                     for pg in pages)

TEMPLATE = (ROOT / "site" / "template.html").read_text(encoding="utf-8")
out = (TEMPLATE.replace("/*__META__*/", json.dumps(meta, ensure_ascii=False))
               .replace("/*__SEARCH__*/", json.dumps(search_index, ensure_ascii=False))
               .replace("/*__SERIES__*/", json.dumps(SERIES))
               .replace("<!--__ARTICLES__-->", articles))
OUT.write_text(out, encoding="utf-8")
print(f"写出 {OUT.relative_to(ROOT)}：{len(pages)} 个页面，{len(out) / 1024:.0f} KB")
