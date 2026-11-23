// 构建时把公式渲染成 HTML（KaTeX）。从 stdin 读 [{tex, display}]，向 stdout 写同样长度的 HTML 字符串数组。
const katex = require("katex");
let input = "";
process.stdin.on("data", d => input += d);
process.stdin.on("end", () => {
  const items = JSON.parse(input);
  const out = items.map(({ tex, display }) => {
    try {
      return katex.renderToString(tex, { displayMode: display, throwOnError: false, strict: "ignore", output: "html" });
    } catch (e) {
      return `<code class="math-error">${tex.replace(/</g, "&lt;")}</code>`;
    }
  });
  process.stdout.write(JSON.stringify(out));
});
