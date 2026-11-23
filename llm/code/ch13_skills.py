r"""
第 13 章 · Agent Skills 的配套代码（llm/ch13-skills.md）。只用标准库：python3 ch13_skills.py

在临时目录里建几个 skill（文件夹 + SKILL.md + 脚本 + 参考文档），然后实现 harness 一侧的加载逻辑：
1. 解析 SKILL.md 的 YAML 头（name、description），生成放进系统提示的"目录"。
2. 渐进式披露：第 1 层只有目录；任务相关时才读入第 2 层（SKILL.md 正文）；正文提到的文件按需读入（第 3 层）。
3. 脚本不进上下文：确定性的步骤交给脚本执行，只有输出进入上下文。
4. 统计 50 个 skill 时，各种加载方式的上下文开销。
5. description 写得含糊，skill 就不会被选中。
"""
import re, subprocess, sys, tempfile, textwrap
from pathlib import Path

# ---------------------------------------------------------------------------
# 建几个 skill
# ---------------------------------------------------------------------------
ROOT = Path(tempfile.mkdtemp(prefix="skills-"))


def make_skill(name, description, body, files=None):
    d = ROOT / name
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text(f"---\nname: {name}\ndescription: {description}\n---\n\n{body}", encoding="utf-8")
    for rel, content in (files or {}).items():
        (d / rel).parent.mkdir(parents=True, exist_ok=True)
        (d / rel).write_text(content, encoding="utf-8")


make_skill(
    "csv-cleaning",
    "清洗和检查 CSV 表格数据：去重、统一日期格式、检查缺失值。当用户要处理、清洗、检查 CSV 或表格数据时使用。",
    textwrap.dedent("""\
        # 清洗 CSV

        ## 步骤
        1. 先运行 `scripts/check_csv.py <文件>`，拿到行数、重复行、缺失值和日期格式的统计。不要自己逐行读文件。
        2. 日期统一成 ISO 8601（YYYY-MM-DD）。遇到含糊的日期（如 03/04/2025）时，读 `references/date-formats.md` 决定怎么处理。
        3. 删除完全重复的行；部分重复的行列出来给用户确认，不要自动删除。
        4. 输出清洗后的文件，并用一段话说明改了什么。
        """),
    {"scripts/check_csv.py": textwrap.dedent("""\
        import csv, re, sys
        rows = list(csv.reader(open(sys.argv[1], encoding="utf-8")))
        header, data = rows[0], rows[1:]
        dup = len(data) - len({tuple(r) for r in data})
        missing = sum(1 for r in data for v in r if v.strip() == "")
        fmts = {}
        for r in data:
            d = r[0]
            k = "ISO" if re.fullmatch(r"\\d{4}-\\d{2}-\\d{2}", d) else "斜杠" if "/" in d else "其他"
            fmts[k] = fmts.get(k, 0) + 1
        print(f"{len(data)} 行，{len(header)} 列；完全重复 {dup} 行；空值 {missing} 个；日期格式 {fmts}")
        """),
     "references/date-formats.md": "# 日期格式\n\n含糊日期（DD/MM 还是 MM/DD）：看同一列里是否出现大于 12 的数字来判断；"
                                   "判断不了就问用户，不要猜。\n" + "（此处省略约 3000 字的各国日期格式对照表）\n" * 40},
)
make_skill(
    "commit-message",
    "按团队规范写 git 提交信息：类型前缀、50 字以内的标题、说明动机的正文。当用户要提交代码或写 commit message 时使用。",
    "# 提交信息规范\n\n标题格式：`<type>: <summary>`，type 取 feat / fix / docs / refactor / test。\n" * 3,
)
make_skill("helper", "有用的工具。", "# 帮手\n\n其实这个 skill 是做 CSV 去重的，但描述没写清楚。\n")
for i in range(47):                                     # 再放 47 个无关的 skill，模拟真实环境里装了很多
    make_skill(f"domain-{i:02d}", f"领域 {i} 的内部流程说明：何时使用、怎样操作、注意事项。当用户提到领域 {i} 时使用。",
               f"# 领域 {i}\n\n" + "详细步骤……\n" * 120)

# 一个待处理的数据文件
data = ROOT / "sales.csv"
data.write_text("date,item,amount\n2025-01-03,apple,3\n03/04/2025,pear,5\n2025-01-03,apple,3\n2025-02-11,,7\n",
                encoding="utf-8")


# ---------------------------------------------------------------------------
# harness 一侧：解析、目录、加载
# ---------------------------------------------------------------------------
def approx_tokens(text):
    """粗略估计：中文约 1 字 1 token，英文约 4 字符 1 token。只用于比较量级。"""
    cjk = len(re.findall(r"[一-鿿]", text))
    return cjk + (len(text) - cjk) // 4


def parse_skill(path):
    text = (path / "SKILL.md").read_text(encoding="utf-8")
    m = re.match(r"---\n(.*?)\n---\n(.*)", text, re.S)
    meta = dict(line.split(": ", 1) for line in m.group(1).splitlines())
    return {"name": meta["name"], "description": meta["description"], "body": m.group(2), "dir": path}


skills = [parse_skill(p) for p in sorted(ROOT.iterdir()) if (p / "SKILL.md").exists()]
catalog = "\n".join(f"- {s['name']}: {s['description']}" for s in skills)   # 第 1 层：放进系统提示

print("=== 1. 系统提示里的 skill 目录（前 4 条） ===")
print("\n".join(catalog.splitlines()[:4]), "\n  ……")


def choose_skill(task):
    """代替模型做选择：看任务和 description 的词语重合（真实系统里由模型读目录后自己决定）。"""
    words = set(re.findall(r"[a-zA-Z]+|[一-鿿]{2}", task))
    scored = []
    for s in skills:
        desc = s["description"]
        score = sum(1 for w in words if w in desc)
        scored.append((score, s["name"]))
    scored.sort(reverse=True)
    return scored[:3]


task = "帮我清洗 sales.csv：日期格式乱了，还有重复行。"
print(f"\n=== 2. 任务：{task} ===")
print("  按 description 打分（前三）：", choose_skill(task))
chosen = next(s for s in skills if s["name"] == choose_skill(task)[0][1])
context = [("目录（第 1 层）", catalog), (f"{chosen['name']}/SKILL.md（第 2 层）", chosen["body"])]

print("\n=== 3. 按正文执行：先跑脚本，脚本本身不进上下文 ===")
script = chosen["dir"] / "scripts" / "check_csv.py"
out = subprocess.run([sys.executable, str(script), str(data)], capture_output=True, text=True).stdout.strip()
print("  脚本输出：", out)
context.append(("脚本输出", out))
if "斜杠" in out:                                         # 正文说：遇到含糊日期才读参考文档
    ref = (chosen["dir"] / "references" / "date-formats.md").read_text(encoding="utf-8")
    context.append(("references/date-formats.md（第 3 层）", ref))
    print("  发现斜杠日期 → 按需读入 references/date-formats.md")

print("\n=== 4. 上下文开销（估计 token 数） ===")
for label, text in context:
    print(f"  {label:<40}{approx_tokens(text):>7}")
loaded = sum(approx_tokens(t) for _, t in context)
everything = sum(approx_tokens((s['dir'] / 'SKILL.md').read_text(encoding='utf-8')) for s in skills) + \
             sum(approx_tokens(p.read_text(encoding='utf-8')) for p in ROOT.rglob("*") if p.is_file()
                 and p.suffix in (".md", ".py") and p.name != "SKILL.md")
print(f"  {'渐进式加载合计':<38}{loaded:>7}")
print(f"  {'把 50 个 skill 的全部内容都塞进提示':<34}{everything:>7}")
print(f"  {'脚本源码（没有进入上下文）':<36}{approx_tokens(script.read_text(encoding='utf-8')):>7}")

print("\n=== 5. description 决定能否被选中 ===")
for t in ["给 sales.csv 去重", "帮我写这次修复的 commit message"]:
    print(f"  任务 {t!r} → 打分 {choose_skill(t)}")
print("  'helper' 其实也会去重，但描述只有'有用的工具'，得分总是 0，永远不会被选中。")
