r"""
生成第 6 章的示意图（SVG）。python3 make_ch06_figures.py

颜色都写成 var(--ink, 默认值) 这样的形式：
- 单独打开 SVG（比如在 GitHub 上）时用默认值；
- 嵌进书站时跟着网站的浅色 / 深色主题变。
"""
import math, re
from pathlib import Path

OUT = Path(__file__).resolve().parent

STYLE = """<style>
.bg{fill:var(--paper,#fbfbfa)}
.t{fill:var(--ink,#161c28);font-family:var(--font-body,"Noto Sans SC","PingFang SC","Microsoft YaHei",sans-serif)}
.mu{fill:var(--muted,#586172)}
.ln{stroke:var(--muted,#586172);fill:none;stroke-width:1.4}
.lnb{stroke:var(--ink,#161c28);fill:none;stroke-width:3}
.box{fill:var(--wash,#f1f2f4);stroke:var(--rule,#c9cdd4);stroke-width:1}
.frame{fill:none;stroke:var(--rule,#c9cdd4);stroke-width:1}
.q{fill:#cf3b25}.qs{stroke:#cf3b25}
.k{fill:#d99a0b}.ks{stroke:#d99a0b}
.v{fill:#2f6fb5}.vs{stroke:#2f6fb5}
.acc{fill:var(--accent,#cf3b25)}
.mask{fill:var(--rule,#d5d8dd)}
.wh{fill:#ffffff}
</style>"""


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def rich(s):
    """'W_{Q}' 写下标、'x^{2}' 写上标，其余原样。"""
    out, i = [], 0
    for m in re.finditer(r"([_^])\{([^}]*)\}", s):
        out.append(esc(s[i:m.start()]))
        shift = "sub" if m.group(1) == "_" else "super"
        out.append(f'<tspan baseline-shift="{shift}" font-size="75%">{esc(m.group(2))}</tspan>')
        i = m.end()
    out.append(esc(s[i:]))
    return "".join(out)


class Fig:
    def __init__(self, name, w, h):
        self.name, self.w, self.h, self.el = name, w, h, []

    def text(self, x, y, s, size=14, cls="t", anchor="middle", weight=None, extra=""):
        fw = f' font-weight="{weight}"' if weight else ""
        self.el.append(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}" '
                       f'class="{cls}"{fw} {extra}>{rich(s)}</text>')

    def rect(self, x, y, w, h, cls="box", rx=4, extra=""):
        self.el.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" class="{cls}" {extra}/>')

    def line(self, x1, y1, x2, y2, cls="ln", arrow=False, extra=""):
        m = f' marker-end="url(#ah-{self.name})"' if arrow else ""
        self.el.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" class="{cls}"{m} {extra}/>')

    def path(self, d, cls="ln", arrow=False, extra=""):
        m = f' marker-end="url(#ah-{self.name})"' if arrow else ""
        self.el.append(f'<path d="{d}" class="{cls}"{m} {extra}/>')

    def raw(self, s):
        self.el.append(s)

    def save(self):
        head = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" '
                f'width="{self.w}" height="{self.h}" role="img">')
        defs = (f'<defs><marker id="ah-{self.name}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="10" '
                f'markerHeight="10" markerUnits="userSpaceOnUse" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="mu"/></marker></defs>')
        body = "\n".join(self.el)
        svg = f'{head}\n{STYLE}\n{defs}\n<rect width="100%" height="100%" class="bg"/>\n{body}\n</svg>\n'
        (OUT / f"{self.name}.svg").write_text(svg, encoding="utf-8")
        print("写出", self.name + ".svg")


def softmax(xs):
    m = max(xs); e = [math.exp(x - m) for x in xs]; s = sum(e)
    return [x / s for x in e]


# ---------------------------------------------------------------------------
# 图 6-1：一个位置怎样读取前文（query、key、value）
# ---------------------------------------------------------------------------
def fig_qkv():
    f = Fig("ch06-qkv", 760, 440)
    toks = ["我", "养", "了", "一只", "猫", "，", "它"]
    scores = [0.2, 0.0, -0.6, 0.6, 2.9, 0.0, 0.7]          # 示意数字
    w = softmax(scores)
    f.rect(20, 14, 720, 46, "box", 8)
    f.text(380, 34, "当前位置 t = 7（「它」）的 query q_{7}：「我在找什么」", 15, weight="bold")
    f.text(380, 52, "它去和每个位置 s ≤ 7 的 key k_{s}（「我这里有什么」）做内积", 13, "t mu")
    cols = [(72, "前文的 token"), (205, "打分 q_{7}·k_{s} / √d"), (420, "softmax 后的权重"), (612, "按权重读取 value")]
    for x, s in cols:
        f.text(x, 88, s, 13, "t mu")
    y0, rh = 100, 40
    for i, (tok, sc, wi) in enumerate(zip(toks, scores, w)):
        y = y0 + i * rh
        hl = tok == "猫"
        f.rect(30, y + 6, 84, 28, "box", 5, 'style="stroke:#cf3b25;stroke-width:2"' if hl else "")
        f.text(72, y + 25, f"{i + 1}  {tok}", 14, weight="bold" if hl else None)
        f.rect(128, y + 11, 30, 18, "k", 4)
        f.text(143, y + 25, f"k_{{{i + 1}}}", 12, "t wh")
        f.text(220, y + 25, f"{sc:+.1f}", 14, weight="bold" if hl else None)
        bw = 200 * wi
        f.rect(310, y + 12, max(bw, 1.5), 16, "q", 2, f'opacity="{0.35 + 0.65 * wi / max(w):.2f}"')
        f.text(316 + bw, y + 25, f"{wi:.2f}", 13, anchor="start", weight="bold" if hl else None)
        f.rect(588, y + 11, 48, 18, "v", 4, f'opacity="{0.25 + 0.75 * wi / max(w):.2f}"')
        f.text(612, y + 25, f"{wi:.2f}·v_{{{i + 1}}}", 11, "t wh")
        f.line(636, y + 20, 676, 240, "ln")
    f.text(690, 236, "Σ", 26, anchor="middle", weight="bold")
    f.rect(664, 252, 88, 52, "box", 6, 'style="stroke:#2f6fb5;stroke-width:2"')
    f.text(708, 274, "输出 o_{7}", 14, weight="bold")
    f.text(708, 294, "主要来自「猫」", 11, "t mu")
    ym = y0 + 7 * rh + 4
    f.rect(30, ym, 610, 28, "mask", 5, 'opacity="0.6"')
    f.text(335, ym + 19, "位置 8、9、…（后文）：被 causal mask 挡住，权重恒为 0", 13, "t mu")
    f.text(380, 428, "权重是 softmax 的结果，全部加起来等于 1。数字是为说明而编的，不是真实模型的输出。", 12, "t mu")
    f.save()


# ---------------------------------------------------------------------------
# 图 6-2：矩阵形式与形状
# ---------------------------------------------------------------------------
P_REAL = [[1.00, 0, 0, 0, 0, 0], [0.72, 0.28, 0, 0, 0, 0], [0.37, 0.16, 0.48, 0, 0, 0],
          [0.05, 0.12, 0.34, 0.48, 0, 0], [0.05, 0.10, 0.28, 0.38, 0.19, 0],
          [0.44, 0.13, 0.09, 0.07, 0.08, 0.19]]           # 6.5.1 节的实际输出


def grid(f, x, y, rows, cols, c, fill=lambda i, j: ("box", 1.0)):
    for i in range(rows):
        for j in range(cols):
            cls, op = fill(i, j)
            f.rect(x + j * c, y + i * c, c, c, cls, 0, f'opacity="{op:.2f}"')
    f.rect(x, y, cols * c, rows * c, "frame", 0)


def fig_matrix():
    f = Fig("ch06-matrix", 760, 262)
    c, T, d = 18, 6, 3
    top = 70
    x = 30
    f.text(380, 30, "一次算完所有位置：softmax(QK^{⊤}/√d + M) V", 16, weight="bold")
    # Q
    grid(f, x, top, T, d, c, lambda i, j: ("q", 0.55)); qx = x
    f.text(x + d * c / 2, top + T * c + 22, "Q", 15, weight="bold"); f.text(x + d * c / 2, top + T * c + 40, "T×d", 12, "t mu")
    x += d * c + 14; f.text(x + 4, top + T * c / 2 + 6, "·", 22); x += 14
    # K^T
    grid(f, x, top + (T - d) * c / 2, d, T, c, lambda i, j: ("k", 0.55))
    f.text(x + T * c / 2, top + T * c + 22, "K^{⊤}", 15, weight="bold"); f.text(x + T * c / 2, top + T * c + 40, "d×T", 12, "t mu")
    x += T * c + 10; f.text(x + 6, top + T * c / 2 + 6, "=", 18); x += 22
    # S with mask
    grid(f, x, top, T, T, c, lambda i, j: ("mask", 1.0) if j > i else ("acc", 0.18 + 0.1 * ((i * 3 + j * 5) % 4)))
    for i in range(T):
        for j in range(i + 1, T):
            f.text(x + j * c + c / 2, top + i * c + 13, "−∞", 8, "t mu")
    f.text(x + T * c / 2, top + T * c + 22, "打分 S（加上 mask）", 13, weight="bold"); f.text(x + T * c / 2, top + T * c + 40, "T×T", 12, "t mu")
    sx = x
    x += T * c + 8
    f.line(x, top + T * c / 2, x + 66, top + T * c / 2, "ln", arrow=True)
    f.text(x + 33, top + T * c / 2 - 8, "softmax", 12, "t mu"); f.text(x + 33, top + T * c / 2 + 20, "按行", 12, "t mu")
    x += 76
    grid(f, x, top, T, T, c, lambda i, j: ("mask", 0.35) if j > i else ("acc", 0.08 + 0.92 * P_REAL[i][j]))
    f.text(x + T * c / 2, top + T * c + 22, "权重 P", 15, weight="bold"); f.text(x + T * c / 2, top + T * c + 40, "T×T，每行和为 1", 12, "t mu")
    px = x
    x += T * c + 10; f.text(x + 4, top + T * c / 2 + 6, "·", 22); x += 14
    grid(f, x, top, T, d, c, lambda i, j: ("v", 0.55))
    f.text(x + d * c / 2, top + T * c + 22, "V", 15, weight="bold"); f.text(x + d * c / 2, top + T * c + 40, "T×d", 12, "t mu")
    x += d * c + 10; f.text(x + 6, top + T * c / 2 + 6, "=", 18); x += 22
    grid(f, x, top, T, d, c, lambda i, j: ("v", 0.25 + 0.12 * ((i + j) % 3)))
    f.text(x + d * c / 2, top + T * c + 22, "输出 O", 15, weight="bold"); f.text(x + d * c / 2, top + T * c + 40, "T×d", 12, "t mu")
    # 第 4 行的读法
    f.rect(px - 2, top + 3 * c - 2, T * c + 4, c + 4, "frame", 2, 'style="stroke:#cf3b25;stroke-width:2"')
    f.text(px + T * c / 2, top - 12, "第 4 行：位置 4 分给 1…4 的权重", 11, "t mu")
    f.text(sx + T * c / 2, top - 12, "灰格 s > t：未来，置为 −∞", 11, "t mu")
    f.text(380, 248, "P 的颜色深浅是 6.5.1 节实际算出的权重。右上角全为 0，所以注意力矩阵是下三角的。", 12, "t mu")
    f.save()


# ---------------------------------------------------------------------------
# 图 6-3：multi-head attention
# ---------------------------------------------------------------------------
def norm_rows(M):
    out = []
    for i, row in enumerate(M):
        r = [row[j] if j <= i else 0 for j in range(len(row))]
        s = sum(r) or 1
        out.append([v / s for v in r])
    return out


def fig_multihead():
    f = Fig("ch06-multihead", 760, 470)
    T = 6
    pats = [
        ("head 1：看上一个 token", [[(0.9 if j == i - 1 else 0.02) + (1 if i == 0 and j == 0 else 0) for j in range(T)] for i in range(T)]),
        ("head 2：看开头", [[0.8 if j == 0 else 0.04 for j in range(T)] for i in range(T)]),
        ("head 3：看自己", [[0.85 if j == i else 0.03 for j in range(T)] for i in range(T)]),
        ("head 4：按内容匹配", [[1, 0, 0, 0, 0, 0], [0.5, 0.5, 0, 0, 0, 0], [0.1, 0.8, 0.1, 0, 0, 0],
                                [0.7, 0.1, 0.1, 0.1, 0, 0], [0.05, 0.05, 0.8, 0.05, 0.05, 0],
                                [0.05, 0.05, 0.05, 0.75, 0.05, 0.05]]),
    ]
    f.rect(130, 16, 500, 34, "box", 6)
    f.text(380, 39, "输入 X：T × d_{model}（每个位置一个 d_{model} 维向量）", 14, weight="bold")
    xs = [110, 290, 470, 650]
    c = 13
    for k, (cx, (title, M)) in enumerate(zip(xs, pats)):
        f.line(380, 50, cx, 86, "ln", arrow=True)
        f.text(cx, 104, f"W_{{Q}}^{{({k + 1})}}, W_{{K}}^{{({k + 1})}}, W_{{V}}^{{({k + 1})}}", 12, "t mu")
        f.text(cx, 122, f"每个 head 只有 d_{{model}}/h 维", 11, "t mu")
        P = norm_rows(M)
        gx, gy = cx - T * c / 2, 134
        grid(f, gx, gy, T, T, c, lambda i, j, P=P: ("mask", 0.35) if j > i else ("acc", 0.06 + 0.94 * P[i][j]))
        f.text(cx, gy + T * c + 20, title, 13, weight="bold")
        f.line(cx, gy + T * c + 30, 380, 290, "ln", arrow=True)
    f.rect(130, 294, 500, 34, "box", 6)
    f.text(380, 317, "拼接：T × (h · d_{model}/h) = T × d_{model}", 14, weight="bold")
    f.line(380, 328, 380, 356, "ln", arrow=True)
    f.rect(230, 358, 300, 34, "box", 6)
    f.text(380, 381, "乘输出矩阵 W_{O}（d_{model} × d_{model}）", 14, weight="bold")
    f.line(380, 392, 380, 418, "ln", arrow=True)
    f.text(380, 436, "输出：T × d_{model}，和输入同样形状，加回残差流", 14, weight="bold")
    f.text(380, 460, "同样的输入，h 个 head 各算一张注意力图（深色 = 权重大）。四种模式是示意，真实模型里很多 head 的作用并不容易解释。", 12, "t mu")
    f.save()


# ---------------------------------------------------------------------------
# 图 6-4：Transformer 块（Pre-Norm）与整个模型
# ---------------------------------------------------------------------------
def fig_block():
    f = Fig("ch06-block", 760, 520)
    # 左：整个模型
    lx, bw = 40, 220
    stack = [
        (470, "token id（T 个整数）", "box"),
        (420, "embedding：查表得到 T × d", "box"),
        (360, "Transformer 块 1", "q"),
        (310, "Transformer 块 2", "q"),
        (262, "⋮", None),
        (210, "Transformer 块 L", "q"),
        (150, "最后的归一化", "box"),
        (100, "输出层：T × V 的 logits", "box"),
        (40, "softmax：每个位置下一个 token 的分布", "box"),
    ]
    prev = None
    for y, s, cls in stack:
        if cls is None:
            f.text(lx + bw / 2, y + 8, s, 20)
        else:
            f.rect(lx, y - 16, bw, 32, cls, 6, 'opacity="0.85"' if cls == "q" else "")
            f.text(lx + bw / 2, y + 5, s, 13, "t wh" if cls == "q" else "t", weight="bold" if cls == "q" else None)
        if prev is not None:
            f.line(lx + bw / 2, prev - 16, lx + bw / 2, y + 18 if cls else y + 14, "ln", arrow=True)
        prev = y if cls else y - 2
    f.text(lx + bw / 2, 512, "整个模型：同样的块堆 L 层（Llama-3-8B 是 32 层）", 12, "t mu")
    # 放大线
    f.line(lx + bw, 294, 330, 26, "ln", extra='stroke-dasharray="4 4"')
    f.line(lx + bw, 326, 330, 494, "ln", extra='stroke-dasharray="4 4"')
    f.rect(330, 16, 410, 478, "frame", 10, 'style="stroke-dasharray:4 4"')
    # 右：一个块的内部
    sx = 400
    f.line(sx, 470, sx, 72, "lnb")
    f.raw(f'<path d="M{sx - 9},74 L{sx},56 L{sx + 9},74 z" class="t"/>')
    f.text(sx, 484, "输入 X（T × d）", 13, weight="bold")
    f.text(sx, 46, "输出 X'（T × d，形状不变）", 13, weight="bold")
    f.text(sx - 16, 380, "残差流", 12, "t mu", anchor="end")
    f.text(sx - 16, 396, "residual stream", 11, "t mu", anchor="end")

    def branch(y_from, y_to, box2, note, note2):
        bx = 470
        f.path(f"M{sx},{y_from} L{bx + 70},{y_from} L{bx + 70},{y_from - 22}", "ln", arrow=True)
        f.rect(bx, y_from - 52, 140, 30, "box", 5)
        f.text(bx + 70, y_from - 32, "归一化（RMSNorm）", 12)
        f.line(bx + 70, y_from - 52, bx + 70, y_from - 72, "ln", arrow=True)
        f.rect(bx, y_from - 114, 140, 42, "q", 6, 'opacity="0.85"')
        f.text(bx + 70, y_from - 88, box2, 13, "t wh", weight="bold")
        f.path(f"M{bx + 70},{y_from - 114} L{bx + 70},{y_to} L{sx + 14},{y_to}", "ln", arrow=True)
        f.raw(f'<circle cx="{sx}" cy="{y_to}" r="12" class="box" style="stroke:var(--ink,#161c28);stroke-width:1.6"/>')
        f.text(sx, y_to + 6, "+", 18, weight="bold")
        f.text(bx + 150, y_from - 96, note, 12, "t mu", anchor="start")
        f.text(bx + 150, y_from - 80, note2, 12, "t mu", anchor="start")

    branch(440, 290, "注意力（causal）", "在位置之间", "交换信息")
    branch(256, 110, "FFN（SwiGLU）", "每个位置", "单独加工")
    f.text(535, 470, "Pre-Norm：归一化只在分支里，", 11, "t mu")
    f.text(535, 486, "主干上的 X 从输入一路加到输出", 11, "t mu")
    f.save()


# ---------------------------------------------------------------------------
# 图 6-5：RoPE 只依赖相对位置
# ---------------------------------------------------------------------------
def fig_rope():
    f = Fig("ch06-rope", 760, 340)
    theta = math.radians(25)
    alpha, beta = math.radians(10), math.radians(40)
    panels = [(190, 1, 3), (570, 4, 6)]
    R = 105
    for cx, m, n in panels:
        cy = 180
        f.raw(f'<circle cx="{cx}" cy="{cy}" r="{R}" class="frame" style="stroke-dasharray:3 4"/>')
        f.line(cx - R - 12, cy, cx + R + 12, cy, "ln", extra='opacity="0.4"')
        f.line(cx, cy + R + 12, cx, cy - R - 12, "ln", extra='opacity="0.4"')
        a_q, a_k = alpha + m * theta, beta + n * theta

        def vec(a, cls, label, r=R * 0.92):
            x2, y2 = cx + r * math.cos(a), cy - r * math.sin(a)
            f.el.append(f'<line x1="{cx}" y1="{cy}" x2="{x2:.1f}" y2="{y2:.1f}" class="{cls}" stroke-width="3" marker-end="url(#ah-{f.name})"/>')
            f.text(cx + (r + 18) * math.cos(a), cy - (r + 18) * math.sin(a) + 5, label, 13, weight="bold")

        # 原始方向（虚线）
        for a, cls in ((alpha, "qs"), (beta, "ks")):
            x2, y2 = cx + R * 0.55 * math.cos(a), cy - R * 0.55 * math.sin(a)
            f.el.append(f'<line x1="{cx}" y1="{cy}" x2="{x2:.1f}" y2="{y2:.1f}" class="{cls}" stroke-width="1.5" stroke-dasharray="4 3" opacity="0.6"/>')
        vec(a_q, "qs", f"q 转 {m}θ")
        vec(a_k, "ks", f"k 转 {n}θ")
        # 夹角弧
        r2 = 38
        x1, y1 = cx + r2 * math.cos(a_q), cy - r2 * math.sin(a_q)
        x2, y2 = cx + r2 * math.cos(a_k), cy - r2 * math.sin(a_k)
        f.path(f"M{x1:.1f},{y1:.1f} A{r2},{r2} 0 0 0 {x2:.1f},{y2:.1f}", "ln", extra='stroke-width="2"')
        f.text(cx, 330 - 24, f"query 在位置 m = {m}，key 在位置 n = {n}", 13, weight="bold")
    f.text(380, 30, "两个例子绝对位置不同，但 n − m 都是 2：转完之后 q、k 的夹角一样，内积也一样", 14, weight="bold")
    f.text(380, 52, "虚线是旋转前的 q、k；实线是旋转后的。夹角 = 原来的夹角 + (n − m)θ", 12, "t mu")
    f.text(380, 332, "真实的 RoPE 把每个 head 的向量分成很多对，每一对用不同的频率 θ_{i} 各自旋转。", 12, "t mu")
    f.save()


# ---------------------------------------------------------------------------
# 图：三个矩阵从哪来（X → Q、K、V），以及一个算到底的小例子
# ---------------------------------------------------------------------------
import numpy as np

EX_TOK = ["我", "养", "猫"]
EX_X = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 0, 0]], float)
EX_W = {
    "Q": np.array([[1, 0], [0, 1], [1, 0], [0, 1]], float) * 0.5,
    "K": np.array([[0, 1], [1, 0], [0, 1], [1, 0]], float),
    "V": np.array([[2, 0], [0, 0], [0, 0], [0, 2]], float),
}
CLS = {"Q": "q", "K": "k", "V": "v"}


def example():
    Q, K, V = (EX_X @ EX_W[n] for n in "QKV")
    S = Q @ K.T / math.sqrt(2)
    Z = S + np.triu(np.full((3, 3), -np.inf), 1)
    P = np.exp(Z - Z.max(1, keepdims=True)); P /= P.sum(1, keepdims=True)
    return Q, K, V, S, P, P @ V


def num(v):
    if v == -np.inf:
        return "−∞"
    s = f"{v:.2f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s.replace("-", "−")


def matrix(f, x, y, A, c, cls="box", op=0.18, size=12, hl_row=None, shade=None, upper_mask=False):
    r, k = A.shape
    for i in range(r):
        for j in range(k):
            o = op if shade is None else shade(i, j)
            cc = "mask" if (A[i, j] == -np.inf or (upper_mask and j > i)) else cls
            f.rect(x + j * c, y + i * c, c, c, cc, 0, f'opacity="{o:.2f}"')
            f.text(x + j * c + c / 2, y + i * c + c / 2 + 4.5, num(A[i, j]), size)
    f.rect(x, y, k * c, r * c, "frame", 2)
    if hl_row is not None:
        f.rect(x - 2, y + hl_row * c - 2, k * c + 4, c + 4, "frame", 3, 'style="stroke:#cf3b25;stroke-width:2.2"')


def fig_projection():
    f = Fig("ch06-projection", 760, 612)
    Q, K, V, *_ = example()
    c = 30
    f.text(380, 28, "self-attention 的第一步：同一个 X 乘三个不同的参数矩阵", 16, weight="bold")
    f.text(380, 50, "例子：T = 3 个 token，d_{model} = 4，每个 head 的维数 d = 2", 13, "t mu")
    # X
    xx, xy = 70, 262
    matrix(f, xx, xy, EX_X, c, "box", 1.0, hl_row=2)
    for i, t in enumerate(EX_TOK):
        f.text(xx - 14, xy + i * c + c / 2 + 5, t, 14, weight="bold", anchor="end")
    f.text(xx + 2 * c, xy - 12, "X（T × d_{model} = 3 × 4）", 13, weight="bold")
    f.text(xx + 2 * c, xy + 3 * c + 22, "每一行是一个 token 的向量", 12, "t mu")
    f.text(xx + 2 * c, xy + 3 * c + 40, "（embedding，或前一层的输出）", 12, "t mu")
    rows = [(80, "Q", "W_{Q}", "query：我在找什么"), (250, "K", "W_{K}", "key：我这里有什么"), (420, "V", "W_{V}", "value：被选中时交出的内容")]
    for y, n, wl, meaning in rows:
        out = {"Q": Q, "K": K, "V": V}[n]
        f.path(f"M{xx + 4 * c + 6},{xy + 1.5 * c} C{xx + 4 * c + 50},{xy + 1.5 * c} {290},{y + 2 * c} {306},{y + 2 * c}", "ln", arrow=True)
        wx = 330
        f.text(wx - 4, y + 2 * c + 7, "×", 20, anchor="end")
        matrix(f, wx + 6, y, EX_W[n], c, CLS[n], 0.22)
        f.text(wx + 6 + c, y - 10, f"{wl}（4 × 2）", 13, weight="bold")
        f.text(wx + 6 + 2 * c + 22, y + 2 * c + 7, "=", 20)
        ox = wx + 6 + 2 * c + 44
        oy = y + c / 2
        matrix(f, ox, oy, out, c, CLS[n], 0.45, hl_row=2)
        f.text(ox + c, oy - 10, f"{n}（3 × 2）", 13, weight="bold")
        f.text(ox + 2 * c + 16, oy + c + 4, meaning, 13, anchor="start")
        f.text(ox + 2 * c + 16, oy + 2 * c + 2, f"第 t 行就是 {n.lower()}_{{t}}", 12, "t mu", anchor="start")
    f.rect(20, 562, 720, 40, "box", 6)
    f.text(380, 580, "W_{Q}、W_{K}、W_{V} 是参数：训练学出来，所有位置共用同一份，大小和 T 无关。", 12)
    f.text(380, 596, "Q、K、V 是中间结果：换一句输入就全变。红框：「猫」那一行只由 X 的第 3 行算出。", 12)
    f.save()


def fig_example():
    f = Fig("ch06-example", 760, 330)
    Q, K, V, S, P, O = example()
    c = 40
    y = 92
    f.text(380, 28, "接着算完：打分 → mask → softmax → 加权读取 value", 16, weight="bold")
    f.text(380, 50, "接上图的 Q、K、V，d = 2，所以打分要除以 √2", 13, "t mu")
    x = 40
    for i, t in enumerate(EX_TOK):
        f.text(x - 8, y + i * c + c / 2 + 5, t, 13, weight="bold", anchor="end")
    for j, t in enumerate(EX_TOK):
        f.text(x + j * c + c / 2, y - 8, t, 12, "t mu")
    matrix(f, x, y, S, c, "acc", 0.15)
    f.text(x + 1.5 * c, y + 3 * c + 22, "S = QK^{⊤} / √d", 13, weight="bold")
    f.text(x + 1.5 * c, y + 3 * c + 40, "第 t 行第 s 列：q_{t}·k_{s} / √d", 11, "t mu")
    x += 3 * c + 12
    f.line(x, y + 1.5 * c, x + 40, y + 1.5 * c, "ln", arrow=True); f.text(x + 20, y + 1.5 * c - 8, "+ M", 12, "t mu")
    x += 52
    Z = S + np.triu(np.full((3, 3), -np.inf), 1)
    matrix(f, x, y, Z, c, "acc", 0.15, shade=lambda i, j: 1.0 if j > i else 0.15)
    f.text(x + 1.5 * c, y + 3 * c + 22, "加上 causal mask", 13, weight="bold")
    f.text(x + 1.5 * c, y + 3 * c + 40, "s > t 的位置变成 −∞", 11, "t mu")
    x += 3 * c + 12
    f.line(x, y + 1.5 * c, x + 40, y + 1.5 * c, "ln", arrow=True); f.text(x + 20, y + 1.5 * c - 8, "softmax", 11, "t mu")
    x += 52
    matrix(f, x, y, P, c, "acc", 0, shade=lambda i, j: 0.6 if j > i else 0.08 + 0.7 * P[i, j], hl_row=2, upper_mask=True)
    f.text(x + 1.5 * c, y + 3 * c + 22, "P：每行和为 1", 13, weight="bold")
    f.text(x + 1.5 * c, y + 3 * c + 40, "第 t 行 = 位置 t 的注意力分配", 11, "t mu")
    px = x
    x += 3 * c + 8
    f.text(x + 6, y + 1.5 * c + 7, "·", 22)
    x += 18
    matrix(f, x, y, V, c * 0.8, "v", 0.4, size=11)
    f.text(x + 0.8 * c, y + 2.4 * c + 22, "V", 13, weight="bold")
    x += 1.6 * c + 8
    f.text(x + 6, y + 1.5 * c, "=", 18)
    x += 22
    matrix(f, x, y, O, c * 0.8 + 4, "v", 0.6, size=11, hl_row=2)
    f.text(x + 0.8 * c + 4, y + 2.4 * c + 34, "输出 O", 13, weight="bold")
    o = O[2]
    f.rect(20, 262, 720, 56, "box", 6)
    f.text(380, 283, f"「猫」那一行：o_{{3}} = {P[2,0]:.2f}·v_{{1}} + {P[2,1]:.2f}·v_{{2}} + {P[2,2]:.2f}·v_{{3}} "
                     f"= {P[2,0]:.2f}·[2, 0] + {P[2,1]:.2f}·[0, 2] + {P[2,2]:.2f}·[2, 0] = [{o[0]:.2f}, {o[1]:.2f}]", 13, weight="bold")
    f.text(380, 305, "第 1 行只能看自己，权重必然是 1；第 3 行能看三个位置，「养」的打分最高，拿到约一半的权重。（P 显示的是四舍五入后的值）", 12, "t mu")
    f.save()


if __name__ == "__main__":
    fig_projection(); fig_qkv(); fig_matrix(); fig_example(); fig_multihead(); fig_block(); fig_rope()
