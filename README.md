# gguf-serve

> **Autonomously built using [NEO — your Autonomous AI Agent](https://heyneo.so)** &nbsp;|&nbsp; [Get the VS Code Extension](https://marketplace.visualstudio.com/items?itemName=NeoResearchInc.heyneo)

Serve any GGUF model as a local OpenAI-compatible REST API. No Python environment hassle, no venv, no cloud — just run.

---

## What is gguf-serve?

**gguf-serve** is a zero-friction local LLM server that loads any [GGUF](https://github.com/ggerganov/llama.cpp)-format quantized model and exposes it as a fully OpenAI-compatible REST API on `localhost:8000`. Drop it in as a local replacement for the OpenAI API — your existing code just works.

Built on top of [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) and [Flask](https://flask.palletsprojects.com/), it supports GPU acceleration, rate limiting, request logging, embeddings, tokenization, and mock mode for testing — all configurable via environment variables.

---

## Architecture Infographic

```
┌─────────────────────────────────────────────────────────────────────┐
│                          gguf-serve                                 │
│                  Local OpenAI-Compatible LLM Server                 │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
            ┌──────────────────▼──────────────────┐
            │           Flask REST API             │
            │         localhost:8000               │
            └──────────────────┬──────────────────┘
                               │
     ┌─────────────────────────┼─────────────────────────┐
     │                         │                         │
     ▼                         ▼                         ▼
┌─────────┐             ┌─────────────┐           ┌──────────┐
│  /v1/   │             │    /v1/     │           │  /v1/    │
│  chat/  │             │ completions │           │embeddings│
│completns│             │             │           │/tokenize │
└────┬────┘             └──────┬──────┘           └────┬─────┘
     │                         │                       │
     └─────────────────────────┼───────────────────────┘
                               │
            ┌──────────────────▼──────────────────┐
            │         llama-cpp-python             │
            │      Model Inference Engine          │
            └──────────────────┬──────────────────┘
                               │
            ┌──────────────────▼──────────────────┐
            │            GGUF Model File           │
            │   (quantized LLM — any size/arch)   │
            └─────────────────────────────────────┘

  Configuration Layer (env vars / .env)
  ┌──────────┬────────────┬───────────┬────────────┐
  │ n_ctx    │ n_threads  │ gpu_layers│ rate_limit │
  │ 2048 tok │ 4 (default)│ 0=CPU only│ per-IP     │
  └──────────┴────────────┴───────────┴────────────┘

  Observability
  ┌──────────────────────────────────────────────┐
  │  /health  │  /v1/models  │  /v1/stats        │
  │  uptime   │  model list  │  latency/metrics  │
  └──────────────────────────────────────────────┘
```

---

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Server health, uptime, status |
| `/v1/models` | GET | List loaded models (OpenAI format) |
| `/v1/stats` | GET | Request count, latency, metrics |
| `/v1/chat/completions` | POST | Multi-turn chat (drop-in OpenAI replacement) |
| `/v1/completions` | POST | Raw text completion |
| `/v1/embeddings` | POST | Generate vector embeddings |
| `/v1/tokenize` | POST | Tokenize text, return token count |

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run with a GGUF model

```bash
GGUF_MODEL_PATH=/path/to/model.gguf python gguf_serve.py
```

### 3. Try mock mode (no model file needed)

```bash
GGUF_MOCK_MODE=1 python gguf_serve.py
```

### 4. Send a chat request

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'
```

---

## Configuration

All settings via environment variables (or `.env` file — copy `.env.example`):

| Variable | Default | Description |
|---|---|---|
| `GGUF_MODEL_PATH` | — | Path to your `.gguf` model file **(required)** |
| `GGUF_HOST` | `127.0.0.1` | Bind address |
| `GGUF_PORT` | `8000` | Port |
| `GGUF_N_CTX` | `2048` | Context window (tokens) |
| `GGUF_N_THREADS` | `4` | CPU threads for inference |
| `GGUF_N_GPU_LAYERS` | `0` | GPU layers to offload (`0` = CPU only) |
| `GGUF_MAX_TOKENS` | `256` | Default max tokens per response |
| `GGUF_TEMPERATURE` | `0.7` | Sampling temperature |
| `GGUF_TOP_P` | `0.95` | Nucleus sampling |
| `GGUF_TOP_K` | `40` | Top-k sampling |
| `GGUF_SEED` | `-1` | Seed (`-1` = random) |
| `GGUF_MOCK_MODE` | `0` | `1` = mock mode, no model needed |
| `GGUF_RATE_LIMIT` | `0` | Max requests/min per IP (`0` = unlimited) |
| `GGUF_LOG_REQUESTS` | `0` | `1` = log every request |
| `GGUF_LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |

---

## GPU Acceleration

```bash
GGUF_MODEL_PATH=/path/to/model.gguf \
GGUF_N_GPU_LAYERS=32 \
python gguf_serve.py
```

Increase `GGUF_N_GPU_LAYERS` to offload more layers to GPU (requires a compatible llama-cpp-python build).

---

## Examples

Progressive examples are in the `examples/` directory:

| File | What it covers |
|---|---|
| `01_quick_start.py` | Minimal setup, health check, first chat |
| `02_advanced_usage.py` | Streaming, sampling params, embeddings |
| `03_custom_config.py` | Custom env config patterns |
| `04_full_pipeline.py` | End-to-end workflow with metrics |

---

## Testing

```bash
# Start server in mock mode
GGUF_MOCK_MODE=1 python gguf_serve.py &

# Run smoke tests
python smoke_test.py
```

Smoke tests cover: health, completions, chat, embeddings, tokenization, parameter validation, and stats accuracy.

---

## Stack

- [Flask](https://flask.palletsprojects.com/) — REST API framework
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) — GGUF inference engine
- [Rich](https://github.com/Textualize/rich) — Terminal output formatting
- [python-dotenv](https://github.com/theskumar/python-dotenv) — Env config loading

---

## License

MIT

---
![](https://komarev.com/ghpvc/?username=dakshjain-1616)

> **Autonomously built using [NEO — your Autonomous AI Agent](https://heyneo.so)** &nbsp;|&nbsp; [Get the VS Code Extension](https://marketplace.visualstudio.com/items?itemName=NeoResearchInc.heyneo)
