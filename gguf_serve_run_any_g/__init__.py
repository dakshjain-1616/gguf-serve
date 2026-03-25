"""
gguf_serve_run_any_g — Run any GGUF as a local OpenAI-compatible API.
0 config · 1 binary · drop-in OpenAI replacement.

Public API re-exported from .server for convenient top-level access:

    from gguf_serve_run_any_g import create_app, VERSION, EMBED_DIM

Usage:
    python -m gguf_serve_run_any_g --model path/to/model.gguf
    python -m gguf_serve_run_any_g --mock
"""

from .server import (
    # Constants
    VERSION,
    MOCK_MODEL_ID,
    HOST,
    PORT,
    MODEL_PATH,
    N_CTX,
    N_THREADS,
    N_GPU_LAYERS,
    MAX_TOKENS_DEFAULT,
    TEMPERATURE_DEFAULT,
    TOP_P_DEFAULT,
    TOP_K_DEFAULT,
    SEED_DEFAULT,
    LOG_LEVEL,
    RATE_LIMIT,
    EMBED_DIM,
    LOG_REQUESTS,
    # Public functions
    create_app,
    main,
    parse_args,
    get_llm,
    # Validation helpers (used by tests and smoke_test)
    _validate_prompt,
    _validate_temperature,
    _validate_max_tokens,
    # Mock inference helpers (used by tests and smoke_test)
    _mock_completion,
    _mock_chat_completion,
    _mock_embeddings,
    _mock_tokenize,
)

__all__ = [
    "VERSION",
    "MOCK_MODEL_ID",
    "HOST",
    "PORT",
    "MODEL_PATH",
    "N_CTX",
    "N_THREADS",
    "N_GPU_LAYERS",
    "MAX_TOKENS_DEFAULT",
    "TEMPERATURE_DEFAULT",
    "TOP_P_DEFAULT",
    "TOP_K_DEFAULT",
    "SEED_DEFAULT",
    "LOG_LEVEL",
    "RATE_LIMIT",
    "EMBED_DIM",
    "LOG_REQUESTS",
    "create_app",
    "main",
    "parse_args",
    "get_llm",
    "_validate_prompt",
    "_validate_temperature",
    "_validate_max_tokens",
    "_mock_completion",
    "_mock_chat_completion",
    "_mock_embeddings",
    "_mock_tokenize",
]
