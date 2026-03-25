#!/usr/bin/env python3
"""
demo.py — Demonstrate gguf-serve without requiring a real GGUF model.

Starts the server in MOCK mode, fires several API requests, and saves
all results to the outputs/ directory.

Usage:
    python demo.py
"""

import os
import sys
import json
import time
import threading
import datetime

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import requests  # type: ignore

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

HOST = os.getenv("GGUF_HOST", "127.0.0.1")
PORT = int(os.getenv("GGUF_PORT", "8000"))
BASE_URL = f"http://{HOST}:{PORT}"
OUTPUTS_DIR = os.getenv("GGUF_OUTPUTS_DIR", "outputs")

# Force mock mode for demo — no model file needed
os.environ["GGUF_MOCK_MODE"] = "1"

# ---------------------------------------------------------------------------
# Ensure outputs directory exists
# ---------------------------------------------------------------------------

os.makedirs(OUTPUTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Start server in background thread
# ---------------------------------------------------------------------------

def _run_server():
    import logging
    logging.getLogger("werkzeug").setLevel(logging.ERROR)
    from gguf_serve_run_any_g import create_app
    flask_app = create_app(mock=True)
    flask_app.run(host=HOST, port=PORT, debug=False, use_reloader=False)


def wait_for_server(timeout: int = 15) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"{BASE_URL}/health", timeout=1)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(0.3)
    return False


print("=" * 60)
print("  gguf-serve demo  (MOCK mode — no GGUF file required)")
print("=" * 60)

server_thread = threading.Thread(target=_run_server, daemon=True)
server_thread.start()

print(f"\nWaiting for server on {BASE_URL} …", end="", flush=True)
if not wait_for_server():
    print("\nERROR: server did not start in time.")
    sys.exit(1)
print(" ready!\n")

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

results = []  # collected for JSON output

def call(method: str, path: str, payload: dict | None = None) -> dict:
    url = f"{BASE_URL}{path}"
    try:
        if method == "GET":
            r = requests.get(url, timeout=10)
        else:
            r = requests.post(url, json=payload, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        return {"error": str(exc)}

# ---------------------------------------------------------------------------
# Demo calls
# ---------------------------------------------------------------------------

# 1 — health check
print("1. GET /health")
health = call("GET", "/health")
print(f"   {json.dumps(health, indent=2)}\n")
results.append({"endpoint": "/health", "method": "GET", "response": health})

# 2 — list models
print("2. GET /v1/models")
models = call("GET", "/v1/models")
print(f"   {json.dumps(models, indent=2)}\n")
results.append({"endpoint": "/v1/models", "method": "GET", "response": models})

# 3 — text completion
print("3. POST /v1/completions")
comp = call("POST", "/v1/completions", {
    "prompt": "The quick brown fox",
    "max_tokens": 50,
    "temperature": 0.7,
})
print(f"   {json.dumps(comp, indent=2)}\n")
results.append({"endpoint": "/v1/completions", "method": "POST",
                "request": {"prompt": "The quick brown fox"}, "response": comp})

# 4 — another completion
print("4. POST /v1/completions (second prompt)")
comp2 = call("POST", "/v1/completions", {
    "prompt": "Explain quantum computing in one sentence:",
    "max_tokens": 80,
    "temperature": 0.5,
})
print(f"   {json.dumps(comp2, indent=2)}\n")
results.append({"endpoint": "/v1/completions", "method": "POST",
                "request": {"prompt": "Explain quantum computing"}, "response": comp2})

# 5 — chat completion
print("5. POST /v1/chat/completions")
chat = call("POST", "/v1/chat/completions", {
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is a GGUF model?"},
    ],
    "max_tokens": 120,
    "temperature": 0.8,
})
print(f"   {json.dumps(chat, indent=2)}\n")
results.append({"endpoint": "/v1/chat/completions", "method": "POST",
                "request": {"messages": [{"role": "user", "content": "What is a GGUF model?"}]},
                "response": chat})

# 6 — multi-turn chat
print("6. POST /v1/chat/completions (multi-turn)")
multi = call("POST", "/v1/chat/completions", {
    "messages": [
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi! How can I help you today?"},
        {"role": "user", "content": "Tell me a short joke."},
    ],
    "max_tokens": 60,
})
print(f"   {json.dumps(multi, indent=2)}\n")
results.append({"endpoint": "/v1/chat/completions", "method": "POST",
                "request": {"messages": "multi-turn"}, "response": multi})

# 7 — bad request (missing prompt)
print("7. POST /v1/completions (missing prompt → 400)")
bad = call("POST", "/v1/completions", {"max_tokens": 10})
print(f"   {json.dumps(bad, indent=2)}\n")
results.append({"endpoint": "/v1/completions", "method": "POST",
                "note": "missing prompt", "response": bad})

# 8 — embeddings (single string)
print("8. POST /v1/embeddings (single string)")
emb = call("POST", "/v1/embeddings", {
    "input": "GGUF models are self-contained inference files.",
    "model": "mock-gguf",
})
# Truncate vector for display
if emb.get("data") and emb["data"][0].get("embedding"):
    preview = emb["data"][0]["embedding"][:6]
    emb["data"][0]["embedding"] = preview + ["..."]
print(f"   {json.dumps(emb, indent=2)}\n")
results.append({"endpoint": "/v1/embeddings", "method": "POST",
                "request": {"input": "GGUF models are self-contained inference files."},
                "response": emb})

# 9 — embeddings (batch)
print("9. POST /v1/embeddings (batch of 3)")
emb_batch = call("POST", "/v1/embeddings", {
    "input": ["What is a language model?", "How does quantization work?", "Run models locally."],
})
batch_note = f"{len(emb_batch.get('data', []))} vectors returned"
print(f"   {batch_note}  usage={emb_batch.get('usage')}\n")
results.append({"endpoint": "/v1/embeddings", "method": "POST",
                "note": "batch 3 inputs", "response": {"batch_size": len(emb_batch.get("data", [])),
                "usage": emb_batch.get("usage")}})

# 10 — tokenize
print("10. POST /v1/tokenize")
tok = call("POST", "/v1/tokenize", {"text": "The quick brown fox jumps over the lazy dog."})
print(f"   {json.dumps(tok, indent=2)}\n")
results.append({"endpoint": "/v1/tokenize", "method": "POST",
                "request": {"text": "The quick brown fox..."}, "response": tok})

# 11 — server stats
print("11. GET /v1/stats")
stats = call("GET", "/v1/stats")
print(f"   {json.dumps(stats, indent=2)}\n")
results.append({"endpoint": "/v1/stats", "method": "GET", "response": stats})

# ---------------------------------------------------------------------------
# Save outputs
# ---------------------------------------------------------------------------

timestamp = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

# results.json
json_path = os.path.join(OUTPUTS_DIR, "results.json")
with open(json_path, "w") as f:
    json.dump(
        {
            "generated_at": timestamp,
            "server": BASE_URL,
            "mock_mode": True,
            "total_calls": len(results),
            "results": results,
        },
        f,
        indent=2,
    )
print(f"Saved: {json_path}")

# report.html
html_rows = ""
for item in results:
    endpoint = item["endpoint"]
    method = item["method"]
    note = item.get("note", "")
    resp_text = json.dumps(item["response"], indent=2)
    html_rows += f"""
    <tr>
      <td><code>{method}</code></td>
      <td><code>{endpoint}</code></td>
      <td>{note}</td>
      <td><pre>{resp_text}</pre></td>
    </tr>
"""

html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>gguf-serve demo report</title>
  <style>
    body {{ font-family: monospace; background: #0d1117; color: #c9d1d9; margin: 2rem; }}
    h1 {{ color: #58a6ff; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th {{ background: #161b22; color: #58a6ff; padding: 8px 12px; text-align: left; border: 1px solid #30363d; }}
    td {{ padding: 8px 12px; border: 1px solid #30363d; vertical-align: top; }}
    pre {{ margin: 0; white-space: pre-wrap; word-break: break-word; font-size: 0.85em; }}
    code {{ background: #161b22; padding: 2px 6px; border-radius: 4px; }}
    .footer {{ margin-top: 2rem; color: #8b949e; font-size: 0.85em; }}
  </style>
</head>
<body>
  <h1>gguf-serve demo report</h1>
  <p>Generated: <strong>{timestamp}</strong> &nbsp;|&nbsp; Server: <strong>{BASE_URL}</strong> &nbsp;|&nbsp; Mock mode: <strong>yes</strong></p>
  <table>
    <thead>
      <tr><th>Method</th><th>Endpoint</th><th>Note</th><th>Response</th></tr>
    </thead>
    <tbody>
      {html_rows}
    </tbody>
  </table>
  <div class="footer">
    Made with <a href="https://heyneo.so" style="color:#58a6ff;">NEO</a> — autonomous AI engineer
  </div>
</body>
</html>
"""

html_path = os.path.join(OUTPUTS_DIR, "report.html")
with open(html_path, "w") as f:
    f.write(html_content)
print(f"Saved: {html_path}")

# demo_log.txt  — plain text summary
log_lines = [
    f"gguf-serve demo run — {timestamp}",
    f"Server: {BASE_URL}  |  Mock mode: enabled",
    "",
]
for item in results:
    log_lines.append(f"{item['method']} {item['endpoint']}")
    resp = item["response"]
    if "choices" in resp:
        choice = resp["choices"][0]
        text = (
            choice.get("text") or
            choice.get("message", {}).get("content") or
            "(no text)"
        )
        log_lines.append(f"  -> {text[:100]}")
    elif "status" in resp:
        log_lines.append(f"  -> status={resp['status']}")
    elif "error" in resp:
        log_lines.append(f"  -> error={resp['error']}")
    else:
        log_lines.append(f"  -> {list(resp.keys())}")
    log_lines.append("")

log_path = os.path.join(OUTPUTS_DIR, "demo_log.txt")
with open(log_path, "w") as f:
    f.write("\n".join(log_lines))
print(f"Saved: {log_path}")

print("\nDemo complete. Outputs written to:", OUTPUTS_DIR)
print(f"  {json_path}")
print(f"  {html_path}")
print(f"  {log_path}")
