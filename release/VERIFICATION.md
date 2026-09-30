# Release verification record

This record is part of the submission package. It documents checks run from the
repository root before publication.

| Check | Result |
|---|---|
| `uv run python -m compileall -q release/code` | passed |
| `uv run ruff check release/code` | passed |
| `uv run python release/code/verify_release.py --assets-root /mnt/data/SHM2026/public_upload` | passed; three checkpoint hashes and byte counts matched |
| `uv run python release/code/render_blind.py --help` | passed |
| `uv run python release/code/train_full400.py --dry-run ...` | passed; prepare plus three train commands emitted |
| blind render of one official 1320 x 989 view with the final full-view checkpoint | passed; RGB uint8 PNG and mask IDs `{0,1,2,3,4}` |
| `CUDA_VISIBLE_DEVICES='' uv run pytest -q` | 2141 passed, 22 skipped |

The blind-render smoke output is kept outside Git at
`/mnt/data/SHM2026/release_preflight_render_20260930/`. It is a disposable
local verification artifact; the public entry point writes the same output
layout to the directory supplied by the evaluator.

The test suite is run with `CUDA_VISIBLE_DEVICES=''` for deterministic CPU
contract tests. Rendering and training require a CUDA-capable NVIDIA GPU as
specified in the main README.
