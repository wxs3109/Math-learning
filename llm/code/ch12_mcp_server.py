r"""
第 12 章 · 一个只用标准库手写的 MCP 服务器（llm/ch12-mcp.md）。

按 MCP 2026-07-28 规范实现 stdio 传输：从 stdin 逐行读 JSON-RPC 请求，向 stdout 逐行写响应，日志写到 stderr。
支持的方法：server/discover、tools/list、tools/call、resources/list、resources/read、prompts/list、prompts/get。
不要直接运行它（直接运行会停在那里等 stdin 输入，看起来像卡住了），而是运行客户端，由客户端把它作为子进程启动：
    python3 ch12_mcp_client.py
也可以手动喂一行请求试试：
    echo '{"jsonrpc":"2.0","id":1,"method":"server/discover"}' | python3 ch12_mcp_server.py
"""
import json, sys

PROTOCOL = "2026-07-28"
SERVER_INFO = {"name": "notes-server", "version": "0.1.0"}

# 服务器自己的"数据"：一个小笔记本。stdio 服务器是一个独立进程，状态就放在内存里。
NOTES = {
    1: {"title": "梯度下降的步长", "body": "学习率必须小于 2/L，L 是 Hessian 的最大特征值。"},
    2: {"title": "为什么除以 √d", "body": "q·k 的方差是 d，除以 √d 后 softmax 不会退化成 one-hot。"},
}

TOOLS = [  # 规范建议按确定的顺序返回，方便客户端缓存、提高提示缓存命中率
    {"name": "search_notes", "description": "按关键词搜索笔记，返回匹配的标题和编号。",
     "inputSchema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
    {"name": "add_note", "description": "新建一条笔记，返回新笔记的编号。",
     "inputSchema": {"type": "object", "properties": {"title": {"type": "string"}, "body": {"type": "string"}},
                     "required": ["title", "body"]}},
]
PROMPTS = [{"name": "review_notes", "description": "把全部笔记整理成复习提纲",
            "arguments": [{"name": "style", "description": "提纲风格，例如 简洁 / 详细", "required": False}]}]


def log(*a):
    print("[server]", *a, file=sys.stderr, flush=True)   # stdout 只能写协议消息，日志必须走 stderr


def meta():
    return {"io.modelcontextprotocol/serverInfo": SERVER_INFO}


def ok(result):
    return {"resultType": "complete", **result, "_meta": meta()}


def cacheable(result):
    return ok({"ttlMs": 60_000, "cacheScope": "private", **result})


def tool_text(text, is_error=False):
    return ok({"content": [{"type": "text", "text": text}], "isError": is_error})


def call_tool(name, args):
    if name == "search_notes":
        if not isinstance(args.get("query"), str):
            return tool_text("参数错误：query 必须是字符串", True)        # 校验错误作为工具结果返回，模型可据此修正
        q = args["query"]
        hits = [f"#{i} {n['title']}" for i, n in NOTES.items() if q in n["title"] or q in n["body"]]
        return tool_text("\n".join(hits) if hits else f"没有包含 {q!r} 的笔记")
    if name == "add_note":
        if not all(isinstance(args.get(k), str) for k in ("title", "body")):
            return tool_text("参数错误：title 和 body 都必须是字符串", True)
        new_id = max(NOTES) + 1
        NOTES[new_id] = {"title": args["title"], "body": args["body"]}
        return tool_text(f"已创建笔记 #{new_id}")
    return None


def handle(req):
    method, params = req.get("method"), req.get("params") or {}
    version = (params.get("_meta") or {}).get("io.modelcontextprotocol/protocolVersion")
    # 无状态：每个请求自带协议版本。server/discover 用来让客户端在不知道版本时先问一下。
    if method != "server/discover":
        if version is None:                                      # 没有握手，版本只能从请求里拿：缺了就是参数无效
            raise RpcError(-32602, "缺少 _meta.io.modelcontextprotocol/protocolVersion", {"supported": [PROTOCOL]})
        if version != PROTOCOL:
            raise RpcError(-32022, f"不支持的协议版本 {version}", {"supported": [PROTOCOL]})
    if method == "server/discover":
        return cacheable({"supportedVersions": [PROTOCOL],
                          "capabilities": {"tools": {}, "resources": {}, "prompts": {}},
                          "instructions": "一个笔记本服务器：可以搜索、新建、读取笔记。"})
    if method == "tools/list":
        return cacheable({"tools": TOOLS})
    if method == "tools/call":
        r = call_tool(params.get("name"), params.get("arguments") or {})
        if r is None: raise RpcError(-32602, f"未知工具 {params.get('name')!r}")
        return r
    if method == "resources/list":
        return cacheable({"resources": [{"uri": f"note://{i}", "name": n["title"], "mimeType": "text/plain"}
                                        for i, n in NOTES.items()]})
    if method == "resources/read":
        uri = params.get("uri", "")
        if not isinstance(uri, str): raise RpcError(-32602, "uri 必须是字符串")
        i = int(uri.removeprefix("note://")) if uri.removeprefix("note://").isdigit() else None
        if i not in NOTES: raise RpcError(-32602, f"资源不存在：{uri}")
        return cacheable({"contents": [{"uri": uri, "mimeType": "text/plain", "text": NOTES[i]["body"]}]})
    if method == "prompts/list":
        return cacheable({"prompts": PROMPTS})
    if method == "prompts/get":
        style = (params.get("arguments") or {}).get("style", "简洁")
        listing = "\n".join(f"- {n['title']}：{n['body']}" for n in NOTES.values())
        return ok({"messages": [{"role": "user", "content": {"type": "text",
                   "text": f"请把下面的笔记整理成{style}的复习提纲：\n{listing}"}}]})
    raise RpcError(-32601, f"方法不存在：{method}")


class RpcError(Exception):
    def __init__(self, code, message, data=None):
        self.code, self.message, self.data = code, message, data


def error(id_, code, message, data=None):
    err = {"code": code, "message": message}
    if data is not None: err["data"] = data
    return {"jsonrpc": "2.0", "id": id_, "error": err}


def main():
    log("启动，协议版本", PROTOCOL)
    for line in sys.stdin:
        if not line.strip(): continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as e:   # 读不出 id，按 JSON-RPC 规定 id 填 null
            resp = error(None, -32700, f"JSON 解析失败：{e}")
        else:
            if not isinstance(req, dict) or req.get("jsonrpc") != "2.0" or "method" not in req:
                resp = error(req.get("id") if isinstance(req, dict) else None, -32600, "不是合法的 JSON-RPC 请求")
            elif "id" not in req:           # 通知（notification）没有 id，也不需要回复
                continue
            else:
                try:
                    resp = {"jsonrpc": "2.0", "id": req["id"], "result": handle(req)}
                except RpcError as e:
                    resp = error(req["id"], e.code, e.message, e.data)
                except Exception as e:      # 服务器自己的 bug：回一个内部错误，进程不能崩
                    log("内部错误", repr(e))
                    resp = error(req["id"], -32603, "服务器内部错误")
            log("处理", req.get("method") if isinstance(req, dict) else req)
        sys.stdout.write(json.dumps(resp, ensure_ascii=False) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
