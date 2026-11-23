r"""
第 12 章 · MCP 客户端的配套代码（llm/ch12-mcp.md）。python3 ch12_mcp_client.py

1. 手写客户端：把 ch12_mcp_server.py 作为子进程启动，用 stdio 收发原始 JSON-RPC 消息，逐条打印。
2. 两种错误：协议版本不匹配（协议错误）、工具参数错误（工具执行错误）。
3. 互操作：如果装了官方 Python SDK（pip install "mcp>=2"），用官方 Client 连接这个手写服务器，
   再用官方 MCPServer 写一个等价的服务器，比较两者。没装 SDK 时跳过这一部分。
"""
import asyncio, json, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SERVER = [sys.executable, str(HERE / "ch12_mcp_server.py")]
VERSION = "2026-07-28"


class RawClient:
    """最小的 stdio 客户端：每行一条 JSON-RPC 消息。"""

    def __init__(self, cmd):
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL, text=True, encoding="utf-8")
        self.next_id = 0

    def request(self, method, params=None, version=VERSION, show=True):
        self.next_id += 1
        params = dict(params or {})
        params["_meta"] = {  # 2026-07-28：没有 initialize 握手，每个请求自带版本、能力和身份
            "io.modelcontextprotocol/protocolVersion": version,
            "io.modelcontextprotocol/clientCapabilities": {},
            "io.modelcontextprotocol/clientInfo": {"name": "raw-client", "version": "0.1"},
        }
        msg = {"jsonrpc": "2.0", "id": self.next_id, "method": method, "params": params}
        self.p.stdin.write(json.dumps(msg, ensure_ascii=False) + "\n"); self.p.stdin.flush()
        resp = json.loads(self.p.stdout.readline())
        if show:
            short = {k: v for k, v in params.items() if k != "_meta"}
            print(f"→ {method} {json.dumps(short, ensure_ascii=False)}")
            print(f"← {json.dumps(resp, ensure_ascii=False)[:300]}")
        return resp

    def close(self):
        self.p.stdin.close(); self.p.wait()


print("=== 1. 原始消息 ===")
c = RawClient(SERVER)
c.request("server/discover")
tools = c.request("tools/list")["result"]["tools"]
print(f"   （服务器提供 {len(tools)} 个工具：{[t['name'] for t in tools]}）")
c.request("tools/call", {"name": "search_notes", "arguments": {"query": "softmax"}})
c.request("tools/call", {"name": "add_note", "arguments": {"title": "MCP", "body": "模型上下文协议，JSON-RPC 2.0。"}})
c.request("resources/list")
c.request("resources/read", {"uri": "note://3"})
c.request("prompts/get", {"name": "review_notes", "arguments": {"style": "详细"}})

print("\n=== 2. 两种错误 ===")
c.request("tools/list", version="2024-11-05")                                   # 协议错误：版本不支持
c.request("tools/call", {"name": "search_notes", "arguments": {"query": 42}})    # 工具执行错误：isError=true
c.request("no/such/method")                                                     # 协议错误：方法不存在
c.close()

# ---------------------------------------------------------------------------
# 3. 与官方 SDK 互操作
# ---------------------------------------------------------------------------
print("\n=== 3. 官方 SDK ===")
try:
    from mcp import Client, StdioServerParameters
    from mcp.server.mcpserver import MCPServer
except ImportError:
    print("没有安装 mcp>=2（pip install \"mcp>=2\"），跳过。")
    sys.exit(0)


async def interop():
    params = StdioServerParameters(command=SERVER[0], args=SERVER[1:])
    async with Client(params) as client:                    # 官方客户端 ↔ 手写服务器
        names = [t.name for t in (await client.list_tools()).tools]
        r = await client.call_tool("search_notes", {"query": "步长"})
        print(f"官方 Client 连接手写服务器：协议 {client.protocol_version}，工具 {names}，"
              f"search_notes('步长') → {r.content[0].text!r}")

    app = MCPServer("notes-sdk")                             # 用 SDK 写同样的工具，只要几行

    @app.tool()
    def search_notes(query: str) -> str:
        """按关键词搜索笔记，返回匹配的标题和编号。"""
        return "#1 梯度下降的步长" if query in "梯度下降的步长" else f"没有包含 {query!r} 的笔记"

    async with Client(app) as client:                       # 测试时可以在进程内直接连接
        tool = (await client.list_tools()).tools[0]
        print(f"SDK 从函数签名生成的 inputSchema：{json.dumps(tool.input_schema, ensure_ascii=False)}")
        r = await client.call_tool("search_notes", {"query": "步长"})
        print(f"SDK 服务器 search_notes('步长') → {r.content[0].text!r}，isError={r.is_error}")


asyncio.run(interop())
