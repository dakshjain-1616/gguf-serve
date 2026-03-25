#!/usr/bin/env python3
"""
server.py — Flask-based server for gguf_serve_run_any_g.

Serve any GGUF model as a local OpenAI-compatible API.
No Python venv gymnastics, no config files. Just run.

Usage:
    python -m gguf_serve_run_any_g --model path/to/model.gguf

Environment variables override all defaults (see .env.example).
"""

import hashlib
import json
import logging
import math
import os
import re
import struct
import sys
import threading
import time
import uuid
import argparse

# ---------------------------------------------------------------------------
# Rich — optional but strongly recommended for a beautiful terminal
# ---------------------------------------------------------------------------

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.status import Status
    from rich.text import Text
    _console = Console()
    _err_console = Console(stderr=True)
    _RICH = True
except ImportError:
    _console = None  # type: ignore[assignment]
    _err_console = None  # type: ignore[assignment]
    _RICH = False


def _print_ok(msg: str) -> None:
    """Print a green success/info message to stdout."""
    if _RICH:
        _console.print(f"[bold green]✓[/] {msg}")
    else:
        print(f"[OK] {msg}")


def _print_warn(msg: str) -> None:
    """Print a yellow warning message to stderr."""
    if _RICH:
        _err_console.print(f"[bold yellow]⚠[/] {msg}")
    else:
        print(f"[WARN] {msg}", file=sys.stderr)


def _print_err(msg: str) -> None:
    """Print a red error message to stderr."""
    if _RICH:
        _err_console.print(f"[bold red]✗[/] {msg}")
    else:
        print(f"[ERROR] {msg}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Configuration — every value from env with a sensible default
# ---------------------------------------------------------------------------

VERSION = "2.0.0"
MOCK_MODEL_ID = "mock-gguf"

HOST = os.getenv("GGUF_HOST", "127.0.0.1")
PORT = int(os.getenv("GGUF_PORT", "8000"))
MODEL_PATH = os.getenv("GGUF_MODEL_PATH", "")
N_CTX = int(os.getenv("GGUF_N_CTX", "2048"))
N_THREADS = int(os.getenv("GGUF_N_THREADS", str(os.cpu_count() or 4)))
N_GPU_LAYERS = int(os.getenv("GGUF_N_GPU_LAYERS", "0"))
MAX_TOKENS_DEFAULT = int(os.getenv("GGUF_MAX_TOKENS", "256"))
TEMPERATURE_DEFAULT = float(os.getenv("GGUF_TEMPERATURE", "0.7"))
TOP_P_DEFAULT = float(os.getenv("GGUF_TOP_P", "0.95"))
TOP_K_DEFAULT = int(os.getenv("GGUF_TOP_K", "40"))
SEED_DEFAULT = int(os.getenv("GGUF_SEED", "-1"))
LOG_LEVEL = os.getenv("GGUF_LOG_LEVEL", "INFO").upper()
MOCK_MODE = os.getenv("GGUF_MOCK_MODE", "0") == "1"
RATE_LIMIT = int(os.getenv("GGUF_RATE_LIMIT", "0"))   # req/min per IP, 0=disabled
EMBED_DIM = int(os.getenv("GGUF_EMBED_DIM", "384"))    # mock embedding dimension
LOG_REQUESTS = os.getenv("GGUF_LOG_REQUESTS", "0") == "1"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("gguf_serve")

# ---------------------------------------------------------------------------
# Server statistics (module-level, thread-safe)
# ---------------------------------------------------------------------------

_stats_lock = threading.Lock()
_stats = {
    "requests_total": 0,
    "by_endpoint": {},
    "errors_total": 0,
    "latency_sum_ms": 0.0,
    "avg_latency_ms": 0.0,
    "start_time": time.time(),
}

# ---------------------------------------------------------------------------
# Rate-limit state (module-level)
# ---------------------------------------------------------------------------

_rate_limit_lock = threading.Lock()
_rate_limit_windows: dict = {}   # ip -> list[float] of request timestamps

# ---------------------------------------------------------------------------
# Model state
# ---------------------------------------------------------------------------

_llm = None          # lazy-loaded Llama instance
_app_config: dict = {}


# ---------------------------------------------------------------------------
# Helper: stats recording
# ---------------------------------------------------------------------------

def _record_request(endpoint: str, latency_ms: float, is_error: bool) -> None:
    """Thread-safe increment of per-request counters and rolling latency average."""
    with _stats_lock:
        _stats["requests_total"] += 1
        _stats["by_endpoint"][endpoint] = _stats["by_endpoint"].get(endpoint, 0) + 1
        if is_error:
            _stats["errors_total"] += 1
        _stats["latency_sum_ms"] += latency_ms
        _stats["avg_latency_ms"] = _stats["latency_sum_ms"] / _stats["requests_total"]


def _get_stats_snapshot() -> dict:
    """Return a consistent point-in-time snapshot of server statistics."""
    with _stats_lock:
        snap = dict(_stats)
    return {
        "requests_total": snap["requests_total"],
        "by_endpoint": dict(snap["by_endpoint"]),
        "errors_total": snap["errors_total"],
        "avg_latency_ms": round(snap["avg_latency_ms"], 3),
        "uptime_seconds": round(time.time() - snap["start_time"], 3),
    }


# ---------------------------------------------------------------------------
# Helper: rate limiting
# ---------------------------------------------------------------------------

def _check_rate_limit(ip: str) -> bool:
    """Return True if the request is allowed, False if per-IP rate limit is exceeded."""
    if RATE_LIMIT <= 0:
        return True
    now = time.time()
    with _rate_limit_lock:
        window = _rate_limit_windows.get(ip, [])
        window = [t for t in window if now - t < 60.0]
        if len(window) >= RATE_LIMIT:
            _rate_limit_windows[ip] = window
            return False
        window.append(now)
        _rate_limit_windows[ip] = window
    return True


# ---------------------------------------------------------------------------
# Input validation helpers
# ---------------------------------------------------------------------------

def _validate_prompt(prompt) -> tuple:
    """Return (error_body, 400) if prompt is empty, else (None, None)."""
    if not prompt:
        return (
            {"error": {"message": "'prompt' is required", "type": "invalid_request_error"}},
            400,
        )
    return None, None


def _validate_temperature(temperature) -> tuple:
    """Return (error_body, 400) if temperature is outside [0.0, 2.0], else (None, None)."""
    if temperature < 0.0 or temperature > 2.0:
        return (
            {
                "error": {
                    "message": "temperature must be between 0.0 and 2.0",
                    "type": "invalid_request_error",
                }
            },
            400,
        )
    return None, None


def _validate_max_tokens(max_tokens) -> tuple:
    """Return (error_body, 400) if max_tokens <= 0, else (None, None)."""
    if max_tokens <= 0:
        return (
            {
                "error": {
                    "message": "max_tokens must be a positive integer",
                    "type": "invalid_request_error",
                }
            },
            400,
        )
    return None, None


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

def get_llm():
    """Return the loaded Llama model, initialising it on first call with a Rich progress spinner."""
    global _llm
    if _llm is not None:
        return _llm

    if MOCK_MODE:
        logger.info("MOCK_MODE=1 — returning mock model.")
        return None

    model_path = MODEL_PATH or _app_config.get("model_path", "")
    if not model_path:
        raise RuntimeError(
            "No model path provided. Set GGUF_MODEL_PATH or pass --model."
        )
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"GGUF file not found: {model_path}")

    try:
        from llama_cpp import Llama  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "llama-cpp-python is not installed. Run: pip install llama-cpp-python"
        ) from exc

    if _RICH:
        with Status(
            f"[bold cyan]Loading model:[/] {os.path.basename(model_path)} "
            f"(ctx={N_CTX}, threads={N_THREADS}, gpu_layers={N_GPU_LAYERS})",
            console=_console,
            spinner="dots",
        ):
            _llm = Llama(
                model_path=model_path,
                n_ctx=N_CTX,
                n_threads=N_THREADS,
                n_gpu_layers=N_GPU_LAYERS,
                seed=SEED_DEFAULT,
                verbose=False,
            )
    else:
        logger.info(
            "Loading model: %s  (n_ctx=%d, threads=%d, gpu_layers=%d)",
            model_path, N_CTX, N_THREADS, N_GPU_LAYERS,
        )
        _llm = Llama(
            model_path=model_path,
            n_ctx=N_CTX,
            n_threads=N_THREADS,
            n_gpu_layers=N_GPU_LAYERS,
            seed=SEED_DEFAULT,
            verbose=False,
        )

    _print_ok("Model loaded successfully.")
    return _llm


# ---------------------------------------------------------------------------
# Mock inference helpers
# ---------------------------------------------------------------------------

def _mock_completion(prompt: str, max_tokens: int, temperature: float) -> dict:
    """Return a plausible mock completion response (no model required)."""
    text = (
        f"[MOCK] This is a simulated response to: '{prompt[:60]}...'"
        if len(prompt) > 60
        else f"[MOCK] This is a simulated response to: '{prompt}'"
    )
    return {
        "id": f"cmpl-{uuid.uuid4().hex[:12]}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": MOCK_MODEL_ID,
        "choices": [
            {
                "text": text,
                "index": 0,
                "logprobs": None,
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": len(prompt.split()),
            "completion_tokens": len(text.split()),
            "total_tokens": len(prompt.split()) + len(text.split()),
        },
    }


def _mock_chat_completion(messages: list, max_tokens: int, temperature: float) -> dict:
    """Return a plausible mock chat completion response."""
    last_user = next(
        (m["content"] for m in reversed(messages) if m.get("role") == "user"),
        "Hello",
    )
    content = (
        f"[MOCK] Assistant reply to: '{last_user[:60]}...'"
        if len(last_user) > 60
        else f"[MOCK] Assistant reply to: '{last_user}'"
    )
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MOCK_MODEL_ID,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": sum(len(m.get("content", "").split()) for m in messages),
            "completion_tokens": len(content.split()),
            "total_tokens": sum(len(m.get("content", "").split()) for m in messages)
            + len(content.split()),
        },
    }


def _mock_embeddings(text: str, dim: int) -> list:
    """Return a deterministic unit-normed float vector derived from a SHA-256 hash of text."""
    seed_bytes = hashlib.sha256(text.encode("utf-8")).digest()
    raw: list = []
    block = seed_bytes
    while len(raw) < dim:
        block = hashlib.sha256(block).digest()
        for i in range(0, len(block) - 3, 4):
            val = struct.unpack_from(">i", block, i)[0]   # signed int32
            raw.append(val / 2_147_483_648.0)              # normalise to [-1, 1]
    raw = raw[:dim]
    norm = math.sqrt(sum(v * v for v in raw)) or 1.0
    return [round(v / norm, 6) for v in raw]


def _mock_tokenize(text: str) -> list:
    """Rough word/punctuation tokenizer — predictable and deterministic."""
    return re.findall(r"\w+|[^\w\s]", text)


# ---------------------------------------------------------------------------
# Routes / Flask app factory
# ---------------------------------------------------------------------------

def create_app(model_path: str = "", mock: bool = False):
    """Create and return a Flask app with all API routes registered."""
    global MOCK_MODE, _app_config
    from flask import Flask, request, jsonify, g

    app = Flask(__name__)
    if mock:
        MOCK_MODE = True
    _app_config = {"model_path": model_path or MODEL_PATH, "mock_mode": mock}
    app.config["model_path"] = model_path or MODEL_PATH

    # ------------------------------------------------------------------ hooks

    @app.before_request
    def _before():
        """Record request start time and enforce per-IP rate limit."""
        g.start_time = time.time()
        if RATE_LIMIT > 0:
            ip = request.remote_addr or "unknown"
            if not _check_rate_limit(ip):
                return jsonify(
                    {
                        "error": {
                            "message": f"Rate limit exceeded: {RATE_LIMIT} requests/minute",
                            "type": "rate_limit_error",
                        }
                    }
                ), 429

    @app.after_request
    def _after(response):
        """Record latency stats and inject latency_ms into successful JSON responses."""
        start = getattr(g, "start_time", time.time())
        latency_ms = (time.time() - start) * 1000.0
        is_error = response.status_code >= 400
        _record_request(request.path, latency_ms, is_error)
        if LOG_REQUESTS:
            logger.info(
                "%s %s -> %d (%.1fms)",
                request.method, request.path, response.status_code, latency_ms,
            )
        # Inject latency_ms into all successful JSON responses
        if response.content_type and "application/json" in response.content_type and not is_error:
            try:
                body = json.loads(response.get_data(as_text=True))
                if isinstance(body, dict) and "error" not in body:
                    body["latency_ms"] = round(latency_ms, 3)
                    response.set_data(json.dumps(body))
            except (ValueError, TypeError):
                pass
        return response

    # ------------------------------------------------------------------ routes

    @app.route("/health", methods=["GET"])
    def health():
        """Return server health, uptime, and version information."""
        mp = MODEL_PATH or app.config.get("model_path", "")
        snap = _get_stats_snapshot()
        return jsonify(
            {
                "status": "ok",
                "mock_mode": MOCK_MODE,
                "model": os.path.basename(mp) if mp else "none",
                "host": HOST,
                "port": PORT,
                "uptime_seconds": snap["uptime_seconds"],
                "requests_total": snap["requests_total"],
                "version": VERSION,
            }
        )

    @app.route("/v1/models", methods=["GET"])
    def list_models():
        """Return a list of available models in the OpenAI /v1/models format."""
        mp = MODEL_PATH or app.config.get("model_path", "")
        model_id = os.path.basename(mp) if mp else MOCK_MODEL_ID
        return jsonify(
            {
                "object": "list",
                "data": [
                    {
                        "id": model_id,
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": "gguf-serve",
                        "capabilities": {
                            "completions": True,
                            "chat_completions": True,
                            "embeddings": True,
                        },
                    }
                ],
            }
        )

    @app.route("/v1/stats", methods=["GET"])
    def stats():
        """Return request counts, per-endpoint breakdown, error count, and average latency."""
        return jsonify(_get_stats_snapshot())

    @app.route("/v1/completions", methods=["POST"])
    def completions():
        """Handle text completion requests — validate inputs and delegate to model or mock."""
        data = request.get_json(force=True, silent=True) or {}
        prompt = data.get("prompt", "")
        max_tokens = int(data.get("max_tokens", MAX_TOKENS_DEFAULT))
        temperature = float(data.get("temperature", TEMPERATURE_DEFAULT))
        top_p = float(data.get("top_p", TOP_P_DEFAULT))
        top_k = int(data.get("top_k", TOP_K_DEFAULT))

        err, code = _validate_prompt(prompt)
        if err:
            return jsonify(err), code
        err, code = _validate_temperature(temperature)
        if err:
            return jsonify(err), code
        err, code = _validate_max_tokens(max_tokens)
        if err:
            return jsonify(err), code

        logger.debug(
            "Completion: prompt=%r max_tokens=%d temp=%.2f",
            prompt[:80], max_tokens, temperature,
        )

        if MOCK_MODE or get_llm() is None:
            return jsonify(_mock_completion(prompt, max_tokens, temperature))

        llm = get_llm()
        output = llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            echo=False,
        )
        return jsonify(output)

    @app.route("/v1/chat/completions", methods=["POST"])
    def chat_completions():
        """Handle chat completion requests — validate messages and delegate to model or mock."""
        data = request.get_json(force=True, silent=True) or {}
        messages = data.get("messages", [])
        max_tokens = int(data.get("max_tokens", MAX_TOKENS_DEFAULT))
        temperature = float(data.get("temperature", TEMPERATURE_DEFAULT))
        top_p = float(data.get("top_p", TOP_P_DEFAULT))
        top_k = int(data.get("top_k", TOP_K_DEFAULT))

        if not messages:
            return jsonify(
                {"error": {"message": "'messages' is required", "type": "invalid_request_error"}}
            ), 400

        err, code = _validate_temperature(temperature)
        if err:
            return jsonify(err), code
        err, code = _validate_max_tokens(max_tokens)
        if err:
            return jsonify(err), code

        logger.debug(
            "Chat: %d messages, max_tokens=%d, temp=%.2f",
            len(messages), max_tokens, temperature,
        )

        if MOCK_MODE or get_llm() is None:
            return jsonify(_mock_chat_completion(messages, max_tokens, temperature))

        llm = get_llm()
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                prompt_parts.append(f"### System:\n{content}")
            elif role == "user":
                prompt_parts.append(f"### Human:\n{content}")
            elif role == "assistant":
                prompt_parts.append(f"### Assistant:\n{content}")
        prompt_parts.append("### Assistant:")
        prompt = "\n\n".join(prompt_parts)

        raw = llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            echo=False,
        )
        text = raw["choices"][0]["text"] if raw.get("choices") else ""
        return jsonify(
            {
                "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": os.path.basename(MODEL_PATH or app.config.get("model_path", "")),
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": text.strip()},
                        "finish_reason": raw["choices"][0].get("finish_reason", "stop"),
                    }
                ],
                "usage": raw.get("usage", {}),
            }
        )

    @app.route("/v1/embeddings", methods=["POST"])
    def embeddings():
        """Handle embedding requests — return float vectors for one or a batch of strings."""
        data = request.get_json(force=True, silent=True) or {}
        input_data = data.get("input")

        if input_data is None:
            return jsonify(
                {"error": {"message": "'input' is required", "type": "invalid_request_error"}}
            ), 400

        if isinstance(input_data, str):
            texts = [input_data]
        elif isinstance(input_data, list):
            texts = [str(t) for t in input_data]
        else:
            return jsonify(
                {
                    "error": {
                        "message": "'input' must be a string or list of strings",
                        "type": "invalid_request_error",
                    }
                }
            ), 400

        model_id = data.get("model", MOCK_MODEL_ID)
        result_data = []
        total_tokens = 0

        if MOCK_MODE or get_llm() is None:
            for idx, text in enumerate(texts):
                vector = _mock_embeddings(text, EMBED_DIM)
                toks = _mock_tokenize(text) if text else []
                total_tokens += len(toks)
                result_data.append({"object": "embedding", "embedding": vector, "index": idx})
        else:
            llm = get_llm()
            for idx, text in enumerate(texts):
                emb = list(llm.embed(text))
                toks = _mock_tokenize(text) if text else []
                total_tokens += len(toks)
                result_data.append({"object": "embedding", "embedding": emb, "index": idx})
            model_id = os.path.basename(MODEL_PATH or app.config.get("model_path", ""))

        return jsonify(
            {
                "object": "list",
                "data": result_data,
                "model": model_id,
                "usage": {"prompt_tokens": total_tokens, "total_tokens": total_tokens},
            }
        )

    @app.route("/v1/tokenize", methods=["POST"])
    def tokenize():
        """Handle tokenize requests — split text into tokens and return the token list and count."""
        data = request.get_json(force=True, silent=True) or {}
        text = data.get("text") or data.get("prompt") or ""

        if not text:
            return jsonify(
                {"error": {"message": "'text' or 'prompt' is required", "type": "invalid_request_error"}}
            ), 400

        if MOCK_MODE or get_llm() is None:
            tokens = _mock_tokenize(text)
            return jsonify({"tokens": tokens, "count": len(tokens)})

        llm = get_llm()
        raw_tokens = list(llm.tokenize(text.encode("utf-8")))
        return jsonify({"tokens": raw_tokens, "count": len(raw_tokens)})

    @app.errorhandler(404)
    def not_found(e):
        """Return a JSON 404 error for unrecognised routes."""
        return jsonify({"error": {"message": "Not found", "type": "not_found_error"}}), 404

    @app.errorhandler(500)
    def internal_error(e):
        """Return a JSON 500 error and log the full traceback."""
        logger.exception("Internal server error")
        return jsonify({"error": {"message": str(e), "type": "server_error"}}), 500

    return app


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    """Parse and return CLI arguments, applying env-var defaults for every option."""
    parser = argparse.ArgumentParser(
        description="Serve a GGUF model as a local OpenAI-compatible API",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"gguf-serve {VERSION}")
    parser.add_argument(
        "--model",
        default=os.getenv("GGUF_MODEL_PATH", ""),
        help="Path to the .gguf model file (or set GGUF_MODEL_PATH)",
    )
    parser.add_argument("--host", default=HOST, help="Bind host (or set GGUF_HOST)")
    parser.add_argument("--port", type=int, default=PORT, help="Bind port (or set GGUF_PORT)")
    parser.add_argument("--n-ctx", type=int, default=N_CTX, help="Context window size (or set GGUF_N_CTX)")
    parser.add_argument("--n-threads", type=int, default=N_THREADS, help="CPU threads (or set GGUF_N_THREADS)")
    parser.add_argument("--n-gpu-layers", type=int, default=N_GPU_LAYERS, help="GPU offload layers (or set GGUF_N_GPU_LAYERS)")
    parser.add_argument("--mock", action="store_true", default=MOCK_MODE, help="Run in mock mode (no GGUF required)")
    parser.add_argument("--rate-limit", type=int, default=RATE_LIMIT, help="Max requests/min per IP, 0=disabled (or set GGUF_RATE_LIMIT)")
    parser.add_argument("--embed-dim", type=int, default=EMBED_DIM, help="Mock embedding dimension (or set GGUF_EMBED_DIM)")
    return parser.parse_args(argv)


def _print_startup_banner(host: str, port: int, mock: bool, threads: int, ctx: int, model: str) -> None:
    """Render the Rich startup banner and configuration table, falling back to plain text."""
    if not _RICH:
        print(f"gguf-serve v{VERSION} — http://{host}:{port}  [mock={mock}]")
        return

    # Banner panel
    neo_line = Text()
    neo_line.append("Made with ", style="dim")
    neo_line.append("NEO", style="bold magenta")
    neo_line.append(" — autonomous AI Agent · ", style="dim")
    neo_line.append("heyneo.so", style="bold blue underline")

    title = Text()
    title.append("gguf-serve ", style="bold cyan")
    title.append(f"v{VERSION}", style="bold white")

    banner_text = Text.assemble(
        title, "\n",
        Text("Run any GGUF as a local OpenAI-compatible API\n", style="dim"),
        Text("0 config · 1 script · drop-in OpenAI replacement\n\n", style="dim"),
        neo_line,
    )
    _console.print(Panel(banner_text, border_style="cyan", padding=(0, 2)))

    # Config table
    table = Table(show_header=True, header_style="bold cyan", box=None, padding=(0, 2))
    table.add_column("Setting", style="bold white", no_wrap=True)
    table.add_column("Value", style="green")

    table.add_row("URL", f"http://{host}:{port}")
    table.add_row("Mode", "[yellow]MOCK (no model)[/yellow]" if mock else f"[green]{model or 'real model'}[/green]")
    table.add_row("Threads", str(threads))
    table.add_row("Context", f"{ctx} tokens")
    table.add_row("Endpoints", "/health  /v1/models  /v1/stats  /v1/completions  /v1/chat/completions  /v1/embeddings  /v1/tokenize")

    _console.print(table)
    _console.print()
    _print_ok(f"Listening on [bold]http://{host}:{port}[/]  — press Ctrl+C to stop")
    _console.print()


def main(argv=None) -> None:
    """Parse args, print the startup banner, and run the Flask development server."""
    args = parse_args(argv)

    global MOCK_MODE, N_CTX, N_THREADS, N_GPU_LAYERS, RATE_LIMIT, EMBED_DIM
    MOCK_MODE = args.mock
    N_CTX = args.n_ctx
    N_THREADS = args.n_threads
    N_GPU_LAYERS = args.n_gpu_layers
    RATE_LIMIT = args.rate_limit
    EMBED_DIM = args.embed_dim

    app = create_app(args.model, args.mock)
    app.config["model_path"] = args.model

    if not args.model and not MOCK_MODE:
        _print_warn(
            "No model path provided. Starting in MOCK mode. "
            "Pass --model path/to/model.gguf or set GGUF_MODEL_PATH."
        )
        MOCK_MODE = True

    _print_startup_banner(
        host=args.host,
        port=args.port,
        mock=MOCK_MODE,
        threads=N_THREADS,
        ctx=N_CTX,
        model=os.path.basename(args.model) if args.model else "",
    )

    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
