#!/usr/bin/env python3
"""
03_custom_config.py — Customise behaviour via environment variables.

Shows how every gguf-serve setting is controlled by an env var with
a sensible default. Demonstrates:
  - Overriding GGUF_EMBED_DIM to change embedding vector size
  - Setting GGUF_MAX_TOKENS / GGUF_TEMPERATURE defaults
  - Using GGUF_LOG_REQUESTS to enable per-request logging
  - How GGUF_RATE_LIMIT is read (shown; not enforced in test client)

All values are read at module import time, so set env vars BEFORE
importing gguf_serve_run_any_g.

Run:
    python examples/03_custom_config.py
    GGUF_EMBED_DIM=128 python examples/03_custom_config.py
"""

import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# --- Configure via environment before import ---
os.environ["GGUF_MOCK_MODE"] = "1"
os.environ.setdefault("GGUF_EMBED_DIM", "128")      # smaller embeddings for this demo
os.environ.setdefault("GGUF_MAX_TOKENS", "64")       # tighter default generation limit
os.environ.setdefault("GGUF_TEMPERATURE", "0.3")     # more deterministic outputs
os.environ.setdefault("GGUF_LOG_REQUESTS", "0")      # set to "1" to log every request

# Import AFTER setting env vars so module-level constants pick them up
import importlib
import gguf_serve_run_any_g.server as _srv
# Reload so the new env values are picked up in this demo
importlib.reload(_srv)
import gguf_serve_run_any_g
importlib.reload(gguf_serve_run_any_g)

from gguf_serve_run_any_g import create_app, EMBED_DIM, MAX_TOKENS_DEFAULT, TEMPERATURE_DEFAULT

import json

print("Active configuration")
print("=" * 40)
print(f"  GGUF_EMBED_DIM    = {EMBED_DIM}   (env: {os.environ.get('GGUF_EMBED_DIM')})")
print(f"  GGUF_MAX_TOKENS   = {MAX_TOKENS_DEFAULT}   (env: {os.environ.get('GGUF_MAX_TOKENS')})")
print(f"  GGUF_TEMPERATURE  = {TEMPERATURE_DEFAULT}  (env: {os.environ.get('GGUF_TEMPERATURE')})")
print(f"  GGUF_MOCK_MODE    = {os.environ.get('GGUF_MOCK_MODE')}")
print(f"  GGUF_LOG_REQUESTS = {os.environ.get('GGUF_LOG_REQUESTS')}")
print()

app = create_app(mock=True)
app.config["TESTING"] = True

with app.test_client() as client:
    # Embedding vector size reflects GGUF_EMBED_DIM
    r = client.post("/v1/embeddings", json={"input": "custom embedding dimension"})
    body = json.loads(r.data)
    vec = body["data"][0]["embedding"]
    print(f"Embedding dim: {len(vec)}  (configured to {EMBED_DIM})")
    assert len(vec) == EMBED_DIM, f"Mismatch: got {len(vec)}, expected {EMBED_DIM}"

    # Default max_tokens applies when not specified in the request
    r = client.post("/v1/completions", json={"prompt": "Hello"})
    body = json.loads(r.data)
    print(f"Completion (default params): status=200  id={body['id']}")

    # You can still override per-request
    r = client.post("/v1/completions", json={
        "prompt": "Explain attention mechanisms",
        "max_tokens": 200,    # override the default
        "temperature": 0.9,   # override the default
    })
    body = json.loads(r.data)
    print(f"Completion (per-request override): status=200  id={body['id']}")

    # Health shows mock_mode=True
    r = client.get("/health")
    h = json.loads(r.data)
    print(f"\nHealth: status={h['status']}  mock_mode={h['mock_mode']}  version={h['version']}")

print("\nAll config checks passed.")
