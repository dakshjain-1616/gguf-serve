"""
tests/test_gguf_serve.py — pytest suite for gguf_serve_run_any_g

Tests cover:
  1. Server starts (Flask app factory, /health responds 200)
  2. API requests return well-formed responses
  3. Zero-config requirement (mock mode, no GGUF file needed)
  4. Edge cases and error responses
"""

import os
import json
import time
import pytest

# Force mock mode before importing so no GGUF file is required
os.environ["GGUF_MOCK_MODE"] = "1"

import gguf_serve_run_any_g as gguf_serve  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def client():
    """Return a Flask test client in mock mode."""
    app = gguf_serve.create_app(mock=True)
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


# ---------------------------------------------------------------------------
# Test 1 — Server starts / health check
# ---------------------------------------------------------------------------


class TestServerStarts:
    def test_health_returns_200(self, client):
        r = client.get("/health")
        assert r.status_code == 200

    def test_health_json_content_type(self, client):
        r = client.get("/health")
        assert "application/json" in r.content_type

    def test_health_status_ok(self, client):
        data = r = client.get("/health")
        body = json.loads(r.data)
        assert body["status"] == "ok"

    def test_health_mock_mode_flag(self, client):
        body = json.loads(client.get("/health").data)
        assert body["mock_mode"] is True

    def test_health_has_host_key(self, client):
        body = json.loads(client.get("/health").data)
        assert "host" in body

    def test_health_has_port_key(self, client):
        body = json.loads(client.get("/health").data)
        assert "port" in body

    def test_models_endpoint_200(self, client):
        r = client.get("/v1/models")
        assert r.status_code == 200

    def test_models_returns_list_object(self, client):
        body = json.loads(client.get("/v1/models").data)
        assert body.get("object") == "list"

    def test_models_data_is_list(self, client):
        body = json.loads(client.get("/v1/models").data)
        assert isinstance(body.get("data"), list)

    def test_models_data_not_empty(self, client):
        body = json.loads(client.get("/v1/models").data)
        assert len(body["data"]) >= 1


# ---------------------------------------------------------------------------
# Test 2 — API requests return model responses
# ---------------------------------------------------------------------------


class TestCompletionsEndpoint:
    def test_completions_200(self, client):
        r = client.post("/v1/completions", json={"prompt": "Hello world"})
        assert r.status_code == 200

    def test_completions_has_choices(self, client):
        body = json.loads(client.post("/v1/completions", json={"prompt": "Hello"}).data)
        assert "choices" in body

    def test_completions_choices_is_list(self, client):
        body = json.loads(client.post("/v1/completions", json={"prompt": "Hi"}).data)
        assert isinstance(body["choices"], list)

    def test_completions_choice_has_text(self, client):
        body = json.loads(client.post("/v1/completions", json={"prompt": "Hi"}).data)
        assert "text" in body["choices"][0]

    def test_completions_text_is_string(self, client):
        body = json.loads(client.post("/v1/completions", json={"prompt": "Hi"}).data)
        assert isinstance(body["choices"][0]["text"], str)

    def test_completions_has_usage(self, client):
        body = json.loads(client.post("/v1/completions", json={"prompt": "Hi"}).data)
        assert "usage" in body

    def test_completions_usage_has_total_tokens(self, client):
        body = json.loads(client.post("/v1/completions", json={"prompt": "Hi"}).data)
        assert "total_tokens" in body["usage"]

    def test_completions_has_id(self, client):
        body = json.loads(client.post("/v1/completions", json={"prompt": "Hi"}).data)
        assert "id" in body

    def test_completions_has_created(self, client):
        body = json.loads(client.post("/v1/completions", json={"prompt": "Hi"}).data)
        assert "created" in body
        assert isinstance(body["created"], int)

    def test_completions_missing_prompt_returns_400(self, client):
        r = client.post("/v1/completions", json={"max_tokens": 10})
        assert r.status_code == 400

    def test_completions_400_has_error_key(self, client):
        body = json.loads(client.post("/v1/completions", json={}).data)
        assert "error" in body

    def test_completions_respects_max_tokens_param(self, client):
        # Should not raise even with explicit max_tokens
        r = client.post("/v1/completions", json={"prompt": "Test", "max_tokens": 10})
        assert r.status_code == 200


class TestChatCompletionsEndpoint:
    _messages = [{"role": "user", "content": "What is 2+2?"}]

    def test_chat_200(self, client):
        r = client.post("/v1/chat/completions", json={"messages": self._messages})
        assert r.status_code == 200

    def test_chat_has_choices(self, client):
        body = json.loads(client.post("/v1/chat/completions", json={"messages": self._messages}).data)
        assert "choices" in body

    def test_chat_choice_has_message(self, client):
        body = json.loads(client.post("/v1/chat/completions", json={"messages": self._messages}).data)
        assert "message" in body["choices"][0]

    def test_chat_message_role_is_assistant(self, client):
        body = json.loads(client.post("/v1/chat/completions", json={"messages": self._messages}).data)
        assert body["choices"][0]["message"]["role"] == "assistant"

    def test_chat_message_content_is_string(self, client):
        body = json.loads(client.post("/v1/chat/completions", json={"messages": self._messages}).data)
        assert isinstance(body["choices"][0]["message"]["content"], str)

    def test_chat_has_usage(self, client):
        body = json.loads(client.post("/v1/chat/completions", json={"messages": self._messages}).data)
        assert "usage" in body

    def test_chat_missing_messages_returns_400(self, client):
        r = client.post("/v1/chat/completions", json={"max_tokens": 10})
        assert r.status_code == 400

    def test_chat_object_type(self, client):
        body = json.loads(client.post("/v1/chat/completions", json={"messages": self._messages}).data)
        assert body.get("object") == "chat.completion"

    def test_chat_multi_turn(self, client):
        msgs = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
        ]
        r = client.post("/v1/chat/completions", json={"messages": msgs})
        assert r.status_code == 200

    def test_chat_system_message_accepted(self, client):
        msgs = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
        ]
        r = client.post("/v1/chat/completions", json={"messages": msgs})
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# Test 3 — Zero-config requirement
# ---------------------------------------------------------------------------


class TestZeroConfig:
    def test_server_starts_without_model_path(self, client):
        """Server must start and respond even with no GGUF file."""
        r = client.get("/health")
        assert r.status_code == 200

    def test_completions_work_without_model_file(self, client):
        """Completions endpoint must work without a real model (mock mode)."""
        r = client.post("/v1/completions", json={"prompt": "no model needed"})
        assert r.status_code == 200

    def test_chat_works_without_model_file(self, client):
        """Chat endpoint must work without a real model (mock mode)."""
        r = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
        )
        assert r.status_code == 200

    def test_create_app_factory_no_args(self):
        """create_app() with no arguments returns a Flask app."""
        from gguf_serve_run_any_g import create_app
        app = create_app(mock=True)
        assert app is not None

    def test_all_env_vars_have_defaults(self):
        """Every env var must have a default so the server runs 0-config."""
        import importlib, sys

        # Save original env
        saved = {k: v for k, v in os.environ.items() if k.startswith("GGUF_")}

        # Remove all GGUF_ vars
        for k in list(os.environ.keys()):
            if k.startswith("GGUF_"):
                del os.environ[k]
        os.environ["GGUF_MOCK_MODE"] = "1"  # keep mock

        # Re-import should not raise
        for key in list(sys.modules.keys()):
            if key.startswith("gguf_serve_run_any_g"):
                del sys.modules[key]
        try:
            import gguf_serve_run_any_g as gs
            app = gs.create_app(mock=True)
            app.config["TESTING"] = True
            with app.test_client() as c:
                r = c.get("/health")
                assert r.status_code == 200
        finally:
            # Restore env
            for k in list(os.environ.keys()):
                if k.startswith("GGUF_"):
                    del os.environ[k]
            os.environ.update(saved)
            os.environ["GGUF_MOCK_MODE"] = "1"
            for key in list(sys.modules.keys()):
                if key.startswith("gguf_serve_run_any_g"):
                    del sys.modules[key]


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_unknown_route_returns_404(self, client):
        r = client.get("/nonexistent")
        assert r.status_code == 404

    def test_404_returns_json(self, client):
        r = client.get("/nonexistent")
        assert "application/json" in r.content_type

    def test_empty_json_body_completions(self, client):
        r = client.post("/v1/completions", json={})
        assert r.status_code == 400

    def test_high_temperature_accepted(self, client):
        r = client.post("/v1/completions", json={"prompt": "hi", "temperature": 2.0})
        assert r.status_code == 200

    def test_zero_temperature_accepted(self, client):
        r = client.post("/v1/completions", json={"prompt": "hi", "temperature": 0.0})
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# Test: /v1/stats endpoint
# ---------------------------------------------------------------------------


class TestStatsEndpoint:
    def test_stats_returns_200(self, client):
        r = client.get("/v1/stats")
        assert r.status_code == 200

    def test_stats_has_uptime(self, client):
        body = json.loads(client.get("/v1/stats").data)
        assert "uptime_seconds" in body
        assert body["uptime_seconds"] >= 0

    def test_stats_has_requests_total(self, client):
        body = json.loads(client.get("/v1/stats").data)
        assert "requests_total" in body
        assert isinstance(body["requests_total"], int)

    def test_requests_increment_after_call(self, client):
        body1 = json.loads(client.get("/v1/stats").data)
        client.get("/health")
        body2 = json.loads(client.get("/v1/stats").data)
        assert body2["requests_total"] > body1["requests_total"]

    def test_stats_has_avg_latency(self, client):
        body = json.loads(client.get("/v1/stats").data)
        assert "avg_latency_ms" in body
        assert body["avg_latency_ms"] >= 0

    def test_stats_has_errors_total(self, client):
        body = json.loads(client.get("/v1/stats").data)
        assert "errors_total" in body
        assert body["errors_total"] >= 0


# ---------------------------------------------------------------------------
# Test: /v1/embeddings endpoint
# ---------------------------------------------------------------------------


class TestEmbeddingsEndpoint:
    def test_embeddings_200(self, client):
        r = client.post("/v1/embeddings", json={"input": "Hello world"})
        assert r.status_code == 200

    def test_embeddings_list_object(self, client):
        body = json.loads(client.post("/v1/embeddings", json={"input": "test"}).data)
        assert body.get("object") == "list"

    def test_embeddings_data_is_list(self, client):
        body = json.loads(client.post("/v1/embeddings", json={"input": "test"}).data)
        assert isinstance(body["data"], list)

    def test_embeddings_has_vector(self, client):
        body = json.loads(client.post("/v1/embeddings", json={"input": "test"}).data)
        assert "embedding" in body["data"][0]
        assert isinstance(body["data"][0]["embedding"], list)

    def test_embeddings_vector_length(self, client):
        body = json.loads(client.post("/v1/embeddings", json={"input": "test"}).data)
        assert len(body["data"][0]["embedding"]) == gguf_serve.EMBED_DIM

    def test_embeddings_batch_input(self, client):
        r = client.post("/v1/embeddings", json={"input": ["Hello", "World", "Test"]})
        body = json.loads(r.data)
        assert r.status_code == 200
        assert len(body["data"]) == 3

    def test_embeddings_missing_input_returns_400(self, client):
        r = client.post("/v1/embeddings", json={})
        assert r.status_code == 400

    def test_embeddings_has_usage(self, client):
        body = json.loads(client.post("/v1/embeddings", json={"input": "test"}).data)
        assert "usage" in body
        assert "total_tokens" in body["usage"]

    def test_embeddings_deterministic(self, client):
        body1 = json.loads(client.post("/v1/embeddings", json={"input": "hello"}).data)
        body2 = json.loads(client.post("/v1/embeddings", json={"input": "hello"}).data)
        assert body1["data"][0]["embedding"] == body2["data"][0]["embedding"]


# ---------------------------------------------------------------------------
# Test: /v1/tokenize endpoint
# ---------------------------------------------------------------------------


class TestTokenizeEndpoint:
    def test_tokenize_200(self, client):
        r = client.post("/v1/tokenize", json={"text": "Hello world"})
        assert r.status_code == 200

    def test_tokenize_has_tokens(self, client):
        body = json.loads(client.post("/v1/tokenize", json={"text": "Hello world"}).data)
        assert "tokens" in body
        assert isinstance(body["tokens"], list)

    def test_tokenize_has_count(self, client):
        body = json.loads(client.post("/v1/tokenize", json={"text": "Hello world"}).data)
        assert "count" in body
        assert body["count"] == len(body["tokens"])

    def test_tokenize_accepts_prompt_key(self, client):
        r = client.post("/v1/tokenize", json={"prompt": "Hello world"})
        assert r.status_code == 200

    def test_tokenize_missing_text_returns_400(self, client):
        r = client.post("/v1/tokenize", json={})
        assert r.status_code == 400

    def test_tokenize_longer_text_more_tokens(self, client):
        short = json.loads(client.post("/v1/tokenize", json={"text": "Hi"}).data)
        long_ = json.loads(
            client.post(
                "/v1/tokenize",
                json={"text": "The quick brown fox jumps over the lazy dog"},
            ).data
        )
        assert long_["count"] > short["count"]


# ---------------------------------------------------------------------------
# Test: input validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    def test_negative_temperature_400(self, client):
        r = client.post("/v1/completions", json={"prompt": "hi", "temperature": -0.1})
        assert r.status_code == 400

    def test_temperature_above_2_400(self, client):
        r = client.post("/v1/completions", json={"prompt": "hi", "temperature": 2.001})
        assert r.status_code == 400

    def test_max_tokens_zero_400(self, client):
        r = client.post("/v1/completions", json={"prompt": "hi", "max_tokens": 0})
        assert r.status_code == 400

    def test_negative_max_tokens_400(self, client):
        r = client.post("/v1/completions", json={"prompt": "hi", "max_tokens": -5})
        assert r.status_code == 400

    def test_temperature_zero_boundary_ok(self, client):
        r = client.post("/v1/completions", json={"prompt": "hi", "temperature": 0.0})
        assert r.status_code == 200

    def test_temperature_two_boundary_ok(self, client):
        r = client.post("/v1/completions", json={"prompt": "hi", "temperature": 2.0})
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# Test: latency tracking
# ---------------------------------------------------------------------------


class TestLatencyTracking:
    def test_health_has_latency_ms(self, client):
        body = json.loads(client.get("/health").data)
        assert "latency_ms" in body

    def test_completions_has_latency_ms(self, client):
        body = json.loads(client.post("/v1/completions", json={"prompt": "hi"}).data)
        assert "latency_ms" in body

    def test_chat_has_latency_ms(self, client):
        body = json.loads(
            client.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "hi"}]},
            ).data
        )
        assert "latency_ms" in body

    def test_latency_is_non_negative(self, client):
        body = json.loads(client.get("/health").data)
        assert body["latency_ms"] >= 0
