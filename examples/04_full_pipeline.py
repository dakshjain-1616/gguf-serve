#!/usr/bin/env python3
"""
04_full_pipeline.py — End-to-end workflow showing the full project capability.

Simulates a real downstream integration pipeline:
  1. Spin up the server (in-process, mock mode)
  2. Verify server health and list available models
  3. Generate text completions for a batch of prompts
  4. Run a multi-turn chat conversation
  5. Embed a set of documents and compute cosine similarity
  6. Tokenize inputs and inspect token counts
  7. Validate error handling (bad inputs → 4xx)
  8. Collect and display server statistics

This is the pattern you'd use to build a RAG pipeline, chatbot,
or semantic search system against any local GGUF model.

Run:
    python examples/04_full_pipeline.py
"""

import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import json
import math

os.environ["GGUF_MOCK_MODE"] = "1"

from gguf_serve_run_any_g import create_app, VERSION, EMBED_DIM

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def cosine_similarity(a: list, b: list) -> float:
    """Compute cosine similarity between two unit-normed vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return round(dot / (mag_a * mag_b), 4)


def section(title: str) -> None:
    print(f"\n{'─' * 55}")
    print(f"  {title}")
    print(f"{'─' * 55}")


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

print(f"gguf-serve v{VERSION} — full pipeline demo")
print(f"Embedding dim: {EMBED_DIM}")

app = create_app(mock=True)
app.config["TESTING"] = True

with app.test_client() as client:

    # ------------------------------------------------------------------
    # 1. Health + models
    # ------------------------------------------------------------------
    section("1. Server health & model registry")

    r = client.get("/health")
    health = json.loads(r.data)
    assert r.status_code == 200
    print(f"  status:     {health['status']}")
    print(f"  mock_mode:  {health['mock_mode']}")
    print(f"  version:    {health['version']}")

    r = client.get("/v1/models")
    models = json.loads(r.data)
    model_id = models["data"][0]["id"]
    print(f"  model:      {model_id}")

    # ------------------------------------------------------------------
    # 2. Batch text completions
    # ------------------------------------------------------------------
    section("2. Batch text completions")

    prompts = [
        "GGUF format was designed to",
        "Quantization reduces model size by",
        "The key advantage of local inference is",
    ]
    for i, prompt in enumerate(prompts, 1):
        r = client.post("/v1/completions", json={
            "prompt": prompt,
            "max_tokens": 40,
            "temperature": 0.6,
        })
        body = json.loads(r.data)
        assert r.status_code == 200
        print(f"  [{i}] {body['choices'][0]['text'][:70]}")

    # ------------------------------------------------------------------
    # 3. Multi-turn chat conversation
    # ------------------------------------------------------------------
    section("3. Multi-turn chat conversation")

    conversation = [
        {"role": "system", "content": "You are an expert in machine learning."},
        {"role": "user",   "content": "What is a GGUF model?"},
    ]
    r = client.post("/v1/chat/completions", json={"messages": conversation, "max_tokens": 80})
    reply1 = json.loads(r.data)["choices"][0]["message"]
    conversation.append(reply1)
    print(f"  user:      {conversation[1]['content']}")
    print(f"  assistant: {reply1['content'][:80]}")

    conversation.append({"role": "user", "content": "How does quantization affect quality?"})
    r = client.post("/v1/chat/completions", json={"messages": conversation, "max_tokens": 80})
    reply2 = json.loads(r.data)["choices"][0]["message"]
    print(f"  user:      {conversation[-1]['content']}")
    print(f"  assistant: {reply2['content'][:80]}")

    # ------------------------------------------------------------------
    # 4. Semantic similarity via embeddings
    # ------------------------------------------------------------------
    section("4. Semantic similarity (cosine distance)")

    documents = [
        "GGUF is a binary format for storing quantized LLM weights.",
        "PyTorch tensors store model parameters in floating point.",
        "Quantization maps 32-bit floats to lower-bit representations.",
        "Flask is a lightweight Python web framework.",
    ]
    query = "How are model weights stored in GGUF?"

    all_texts = [query] + documents
    r = client.post("/v1/embeddings", json={"input": all_texts})
    emb_data = json.loads(r.data)["data"]
    assert r.status_code == 200

    query_vec = emb_data[0]["embedding"]
    print(f"  query: \"{query}\"")
    for i, doc in enumerate(documents):
        sim = cosine_similarity(query_vec, emb_data[i + 1]["embedding"])
        print(f"  [{sim:+.4f}] {doc[:60]}")

    # ------------------------------------------------------------------
    # 5. Tokenization
    # ------------------------------------------------------------------
    section("5. Token counting")

    texts = [
        ("short", "Hi"),
        ("medium", "The quick brown fox jumps over the lazy dog"),
        ("long", "Explain the transformer architecture, self-attention, and positional encoding"),
    ]
    for label, text in texts:
        r = client.post("/v1/tokenize", json={"text": text})
        body = json.loads(r.data)
        assert r.status_code == 200
        print(f"  {label:8s}: {body['count']:3d} tokens  {body['tokens'][:6]}{'...' if body['count'] > 6 else ''}")

    # ------------------------------------------------------------------
    # 6. Error handling
    # ------------------------------------------------------------------
    section("6. Input validation & error responses")

    cases = [
        ("/v1/completions",      {"max_tokens": 10},          400, "missing prompt"),
        ("/v1/completions",      {"prompt": "hi", "temperature": -1}, 400, "negative temp"),
        ("/v1/completions",      {"prompt": "hi", "max_tokens": 0},   400, "zero max_tokens"),
        ("/v1/chat/completions", {"messages": []},             400, "empty messages"),
        ("/v1/embeddings",       {},                           400, "missing input"),
    ]
    for path, payload, expected_status, label in cases:
        r = client.post(path, json=payload)
        assert r.status_code == expected_status, f"{label}: expected {expected_status}, got {r.status_code}"
        body = json.loads(r.data)
        print(f"  {expected_status} {label:25s} → {body['error']['type']}")

    # 404 for unknown route
    r = client.get("/unknown")
    assert r.status_code == 404
    print(f"  404 unknown route             → {json.loads(r.data)['error']['type']}")

    # ------------------------------------------------------------------
    # 7. Final stats
    # ------------------------------------------------------------------
    section("7. Server statistics")

    r = client.get("/v1/stats")
    stats = json.loads(r.data)
    assert r.status_code == 200
    print(f"  requests_total : {stats['requests_total']}")
    print(f"  errors_total   : {stats['errors_total']}")
    print(f"  avg_latency_ms : {stats['avg_latency_ms']}")
    print(f"  uptime_seconds : {stats['uptime_seconds']}")
    by_ep = stats["by_endpoint"]
    for ep, count in sorted(by_ep.items(), key=lambda x: -x[1]):
        print(f"    {count:3d}  {ep}")

print(f"\n{'=' * 55}")
print("  Full pipeline complete. All assertions passed.")
print(f"{'=' * 55}")
