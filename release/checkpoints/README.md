# Checkpoint material

Binary checkpoints are hosted outside Git in ModelScope and are downloaded by
the command in [`../README.md`](../README.md). The following manifest binds the
expected role, remote path, byte count, and SHA-256 digest.

The final blind-rendering entry point loads
`semantic_full400_cross.pt`. The RGB-only endpoint is included for RGB
inspection and for reproducing the training path; both endpoints use the same
corner-v2 coordinate contract.
