#!/usr/bin/env python3
"""
01_quick_start.py — Minimal working example.

Starts gguf-serve in mock mode (no GGUF file required), sends a chat
completion request, and prints the response.

Run:
    python examples/01_quick_start.py
"""

import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import threading
import time
import json

os.environ["GGUF_MOCK_MODE"] = "1"

from gguf_serve_run_any_g import create_app

# Start the server in a background thread
app = create_app(mock=True)
app.config["TESTING"] = True

with app.test_client() as client:
    # Health check
    r = client.get("/health")
    print("Health:", json.loads(r.data)["status"])

    # Chat completion
    r = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "What is a GGUF model?"}],
        "max_tokens": 60,
    })
    body = json.loads(r.data)
    print("Response:", body["choices"][0]["message"]["content"])
