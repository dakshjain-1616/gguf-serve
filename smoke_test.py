"""Quick smoke test to verify new features before running full pytest."""
import os
import json
os.environ["GGUF_MOCK_MODE"] = "1"
import gguf_serve_run_any_g as gguf_serve

# Helpers
vec = gguf_serve._mock_embeddings("hello", 8)
assert len(vec) == 8, f"expected 8, got {len(vec)}"
vec2 = gguf_serve._mock_embeddings("hello", 8)
assert vec == vec2, "embeddings not deterministic"

toks = gguf_serve._mock_tokenize("The quick brown fox")
assert len(toks) == 4, f"expected 4 tokens, got {toks}"

err, code = gguf_serve._validate_temperature(-0.1)
assert code == 400
err, code = gguf_serve._validate_temperature(2.0)
assert code is None
err, code = gguf_serve._validate_max_tokens(0)
assert code == 400
err, code = gguf_serve._validate_max_tokens(1)
assert code is None

# Full Flask app
app = gguf_serve.create_app(mock=True)
app.config["TESTING"] = True
with app.test_client() as c:
    # health + latency
    r = c.get("/health")
    body = json.loads(r.data)
    assert r.status_code == 200, r.status_code
    assert "latency_ms" in body, list(body.keys())
    assert body["status"] == "ok"

    # embeddings
    r2 = c.post("/v1/embeddings", json={"input": "test"})
    body2 = json.loads(r2.data)
    assert r2.status_code == 200
    assert len(body2["data"][0]["embedding"]) == 384
    assert "latency_ms" in body2

    # embeddings batch
    r_batch = c.post("/v1/embeddings", json={"input": ["a", "b", "c"]})
    body_batch = json.loads(r_batch.data)
    assert r_batch.status_code == 200
    assert len(body_batch["data"]) == 3

    # tokenize
    r3 = c.post("/v1/tokenize", json={"text": "Hello world"})
    body3 = json.loads(r3.data)
    assert r3.status_code == 200
    assert body3["count"] == len(body3["tokens"])

    # tokenize with prompt key
    r3b = c.post("/v1/tokenize", json={"prompt": "Hello world"})
    assert r3b.status_code == 200

    # stats
    r4 = c.get("/v1/stats")
    body4 = json.loads(r4.data)
    assert r4.status_code == 200
    assert "requests_total" in body4
    assert "avg_latency_ms" in body4
    assert "uptime_seconds" in body4
    assert "errors_total" in body4

    # stats increment
    before = json.loads(c.get("/v1/stats").data)["requests_total"]
    c.get("/health")
    after = json.loads(c.get("/v1/stats").data)["requests_total"]
    assert after > before

    # input validation
    r5 = c.post("/v1/completions", json={"prompt": "hi", "temperature": -0.1})
    assert r5.status_code == 400
    r6 = c.post("/v1/completions", json={"prompt": "hi", "temperature": 2.001})
    assert r6.status_code == 400
    r7 = c.post("/v1/completions", json={"prompt": "hi", "max_tokens": 0})
    assert r7.status_code == 400
    r8 = c.post("/v1/completions", json={"prompt": "hi", "max_tokens": -1})
    assert r8.status_code == 400
    r9 = c.post("/v1/completions", json={"prompt": "hi", "temperature": 2.0})
    assert r9.status_code == 200
    r10 = c.post("/v1/completions", json={"prompt": "hi", "temperature": 0.0})
    assert r10.status_code == 200

    # chat latency
    r_chat = c.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})
    body_chat = json.loads(r_chat.data)
    assert r_chat.status_code == 200
    assert "latency_ms" in body_chat

    # longer text → more tokens
    short = json.loads(c.post("/v1/tokenize", json={"text": "Hi"}).data)
    long_ = json.loads(c.post("/v1/tokenize", json={"text": "The quick brown fox jumps over the lazy dog"}).data)
    assert long_["count"] > short["count"]

print("All smoke tests passed!")
