# Examples

Runnable scripts that demonstrate different features of **gguf-serve**.
All examples work without a real GGUF file — they use `GGUF_MOCK_MODE=1`.

Run any script from the project root:

```bash
python examples/01_quick_start.py
python examples/02_advanced_usage.py
python examples/03_custom_config.py
python examples/04_full_pipeline.py
```

---

| Script | What it demonstrates |
|---|---|
| [`01_quick_start.py`](01_quick_start.py) | Minimal working example — create app, health check, single chat completion (~20 lines) |
| [`02_advanced_usage.py`](02_advanced_usage.py) | All seven endpoints: models, completions, multi-turn chat, embeddings (single + batch), tokenize, stats |
| [`03_custom_config.py`](03_custom_config.py) | Customise behaviour via `GGUF_EMBED_DIM`, `GGUF_MAX_TOKENS`, `GGUF_TEMPERATURE`, and other env vars |
| [`04_full_pipeline.py`](04_full_pipeline.py) | End-to-end workflow: health → batch completions → multi-turn chat → cosine similarity → token counting → error handling → stats |

---

Each script begins with:

```python
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
```

so it resolves `gguf_serve_run_any_g` from the project root and works
from any directory.
