#!/usr/bin/env bash
# Health + Ironclad-shaped smoke: model id, greedy arithmetic, tool call.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

HOST="${HOST:-127.0.0.1}"
BASE="http://${HOST}:${PORT}"

echo ">> health"
curl -sf -m 10 "${BASE}/health" >/dev/null && echo "   OK" || echo "   (no /health body; HTTP may still be 200)"

echo ">> /v1/models"
python3 - "${BASE}" "${SERVED_NAME}" <<'PY'
import json, sys, urllib.request
base, expected = sys.argv[1], sys.argv[2]
r = json.load(urllib.request.urlopen(base + "/v1/models", timeout=10))
ids = [m["id"] for m in r["data"]]
print("  ", ids)
if expected not in ids:
    raise SystemExit(f"expected model id {expected!r}")
print("   OK")
PY

echo ">> 12*17 (thinking off)"
python3 - "${BASE}" "${SERVED_NAME}" <<'PY'
import json, sys, urllib.request
base, model = sys.argv[1], sys.argv[2]
body = {
    "model": model,
    "messages": [{"role": "user", "content": "12*17"}],
    "max_tokens": 2048,
    "temperature": 0,
    "chat_template_kwargs": {"enable_thinking": False},
}
req = urllib.request.Request(
    base + "/v1/chat/completions",
    data=json.dumps(body).encode(),
    headers={"Content-Type": "application/json"},
)
r = json.load(urllib.request.urlopen(req, timeout=300))
text = (r["choices"][0]["message"].get("content") or "").strip()
print("  ", repr(text[:200]))
if "204" not in text:
    raise SystemExit("expected 204 in content")
print("   OK")
PY

echo ">> tool call list_dir (thinking off)"
python3 - "${BASE}" "${SERVED_NAME}" <<'PY'
import json, sys, urllib.request
base, model = sys.argv[1], sys.argv[2]
body = {
    "model": model,
    "messages": [{"role": "user", "content": "List files in the current directory using the tool."}],
    "max_tokens": 128,
    "temperature": 0,
    "chat_template_kwargs": {"enable_thinking": False},
    "tools": [{
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List a directory",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    }],
}
req = urllib.request.Request(
    base + "/v1/chat/completions",
    data=json.dumps(body).encode(),
    headers={"Content-Type": "application/json"},
)
r = json.load(urllib.request.urlopen(req, timeout=300))
msg = r["choices"][0]["message"]
tools = msg.get("tool_calls") or []
print("  finish=", r["choices"][0].get("finish_reason"), "n_tools=", len(tools))
if not tools:
    raise SystemExit("expected a tool call")
print("  name=", tools[0]["function"]["name"], "args=", tools[0]["function"]["arguments"])
print("   OK")
PY
