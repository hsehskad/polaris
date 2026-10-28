"""Dependency-free MCP stdio server for polaris_mcp's stdio transport test.

Speaks newline-delimited JSON-RPC 2.0 and implements just initialize,
tools/list, and tools/call with one tool: echo(text).
"""
import json
import sys

SUPPORTED = ["2025-06-18", "2025-03-26", "2024-11-05"]


def reply(msg_id, result=None, error=None):
    body = {"jsonrpc": "2.0", "id": msg_id}
    if error is not None:
        body["error"] = error
    else:
        body["result"] = result
    sys.stdout.write(json.dumps(body) + "\n")
    sys.stdout.flush()


for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    msg = json.loads(line)
    method, msg_id, params = msg.get("method"), msg.get("id"), msg.get("params") or {}
    if msg_id is None:
        continue  # notifications (e.g. notifications/initialized)
    if method == "initialize":
        requested = params.get("protocolVersion")
        version = requested if requested in SUPPORTED else SUPPORTED[0]
        reply(msg_id, {
            "protocolVersion": version,
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "stdio-fixture", "version": "0.0.1"},
        })
    elif method == "tools/list":
        reply(msg_id, {"tools": [{
            "name": "echo",
            "description": "Echo the text back.",
            "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
        }]})
    elif method == "tools/call" and params.get("name") == "echo":
        text = (params.get("arguments") or {}).get("text", "")
        reply(msg_id, {"content": [{"type": "text", "text": text}], "isError": False})
    elif method == "ping":
        reply(msg_id, {})
    else:
        reply(msg_id, error={"code": -32601, "message": f"method not found: {method}"})
