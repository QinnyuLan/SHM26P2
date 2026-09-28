# Trained checkpoints

Training checkpoints are release assets rather than Git source files. They are
large, machine-specific binary files and are ignored by `.gitignore`. A
public release should publish them through an artifact store or GitHub Release
and place the downloaded files under this directory.

The selected F deployment uses the component hashes and paths documented in
[`release/bridge_f_v1/README.md`](../release/bridge_f_v1/README.md). Verify
SHA-256 before using a checkpoint.
