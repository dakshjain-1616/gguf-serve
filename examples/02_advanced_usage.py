#!/usr/bin/env python3
"""
02_advanced_usage.py — Advanced features: completions, chat, embeddings, tokenize, stats.

Demonstrates all seven API endpoints in mock mode using the in-process
test client. No network socket or running server required.

Run:
    python examples/02_advanced_usage.py
"""

import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import json

os.environ["GGUF_MOCK_MODE"] = "1"

from gguf_serve_run_any_g import create_app, EMBED_DIM, VERSION

app = create_app(mock=True)
app.config["TESTING"] = True

print(f"gguf-serve v{VERSION} — advanced usage demo\n{'=' * 50}")

with app.test_client() as client:

    # 1. List available models
    r = client.get("/v1/models")
    models = json.loads(r.data)
    print(f"\n[1] /v1/models → {len(models['data'])} model(s): {models['data'][0]['id']}")

    # 2. Text completion with custom sampling params
    r = client.post("/v1/completions", json={
        "prompt": "The future of local AI inference is",
        "max_tokens": 80,
        "temperature": 0.5,
        "top_p": 0.9,
        "top_k": 30,
    })
    body = json.loads(r.data)
    print(f"\n[2] /v1/completions →")
    print(f"    id: {body['id']}")
    print(f"    text: {body['choices'][0]['text']}")
    print(f"    tokens: {body['usage']['total_tokens']}  latency: {body.get('latency_ms', '?')}ms")

    # 3. Multi-turn chat with system prompt
    r = client.post("/v1/chat/completions", json={
        "messages": [
            {"role": "system", "content": "You are a concise technical assistant."},
            {"role": "user", "content": "What is quantization in ML?"},
            {"role": "assistant", "content": "Quantization reduces model weight precision."},
            {"role": "user", "content": "What are the trade-offs?"},
        ],
        "max_tokens": 100,
        "temperature": 0.7,
    })
    body = json.loads(r.data)
    print(f"\n[3] /v1/chat/completions (multi-turn) →")
    print(f"    {body['choices'][0]['message']['content']}")

    # 4. Embeddings — single string, inspect vector shape
    r = client.post("/v1/embeddings", json={"input": "GGUF enables portable model inference."})
    body = json.loads(r.data)
    vec = body["data"][0]["embedding"]
    print(f"\n[4] /v1/embeddings →")
    print(f"    dim={len(vec)} (expected {EMBED_DIM})  first 4 values: {vec[:4]}")

    # 5. Batch embeddings
    r = client.post("/v1/embeddings", json={
        "input": ["model inference", "vector embeddings", "local LLM"],
    })
    body = json.loads(r.data)
    print(f"\n[5] /v1/embeddings (batch) →")
    print(f"    {len(body['data'])} vectors returned  usage={body['usage']}")

    # 6. Tokenize
    r = client.post("/v1/tokenize", json={"text": "The quick brown fox jumps over the lazy dog."})
    body = json.loads(r.data)
    print(f"\n[6] /v1/tokenize → {body['count']} tokens: {body['tokens']}")

    # 7. Server stats
    r = client.get("/v1/stats")
    stats = json.loads(r.data)
    print(f"\n[7] /v1/stats →")
    print(f"    requests_total={stats['requests_total']}  errors={stats['errors_total']}")
    print(f"    avg_latency_ms={stats['avg_latency_ms']}  uptime={stats['uptime_seconds']}s")

print("\nDone.")
