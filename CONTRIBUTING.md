# Contributing

Issues and pull requests are welcome. Please describe the data protocol,
random seed, configuration, GPU/software versions, and evaluation split for
any experimental change. Keep generated data, checkpoints, caches, and local
paths out of commits; add a small reproducible manifest or receipt instead.

Before opening a pull request, run:

```bash
uv sync --dev
uv run ruff check .
uv run pytest -q
```

Changes that alter reported metrics should include the corresponding protocol
and a paired comparison rather than overwriting an earlier result.
