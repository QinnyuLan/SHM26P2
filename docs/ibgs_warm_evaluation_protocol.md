# Fixed warm-IBGS RGB endpoint evaluation

This is a predeclared engineering comparison of the two 6000-step warm endpoints, `full` and `no_source`. It is neither an exact reproduction of the original IBGS training recipe nor a new method. The sole primary candidate is `full/fused`; all four final RGB outputs (`full/{raw,fused}`, `no_source/{raw,fused}`) are published. No checkpoint, iteration, source threshold, coefficient, or output branch is selected from evaluation scores.

## Two processes, one prediction barrier

Preparation is allowed only after the training parent has naturally exited zero and both fixed endpoints are complete. It binds the parent plan/receipts, both last checkpoints, TRAIN contract, patched IBGS source and binary, evaluation source snapshots, existing references, main-project lockfile, and cached perceptual weights. It never opens an active endpoint or hashes/decodes VAL RGB or semantic payloads; expected VAL RGB hashes come from completed reference receipts.

1. **IBGS environment: render.** Restore each arm's exact field, learned normals/offsets, SH3, background and fusion network. Select at most four sources from all 350 TRAIN cameras using the fixed original geometric rules: angle <30 degrees, .01 < world-center distance <1.5, ordered by FP64 distance, then angle, then name; exclude a matching target name. Do not read any target image. Refresh all 350 source depths separately from that arm's final field, with the existing radius-one source RGB-valid erosion. Never share another arm's or a training-time depth cache. Evaluate the unit-skip final fusion stage with its own source ablation wrapper. Each of 50 target calls returns both raw and fused RGB; empty source support falls back to raw exactly. Save 200 finite native FP32 RGB arrays and their hashes before ending this process. Counts: 700 source-depth calls, 100 target calls, 200 native outputs, zero optimization/teacher/semantic calls. Costs include bank refresh and image I/O, and are not single-frame FPS.
2. **Main `uv` environment: deliver and score.** Verify natural completion and all 200 array identities. Use the frozen official `distortion_render_grid(..., 'colmap_corner_v2')`, clamp float RGB to [0,1] **before** OpenCV `INTER_LINEAR`/constant-border remapping, then `np.rint(255*RGB).astype(uint8)`. Save/hash all 200 original-grid PNGs before the first VAL RGB hash/decode. Then decode each of the 50 original RGB targets and score all four delivered PNGs. Do not read the 41 annotations, create masks, load DINO, or re-evaluate semantics. No dependency installation/download is permitted.

Metadata-only verification found this dataset's original camera has positive radial k1=.008987863345268233. The official covering canvas is exactly the native 1320×989 canvas, with FP32 fx=fy=925.7015991210938, cx=660, cy=494.5. Its remap lies within the native array. Preparation must check this equality for **all** 50 target cameras against the shared TRAIN camera; the current IBGS kernel cannot silently support differing source/target intrinsics or image sizes. Target `BridgeCamera` receives the original manifest-double common K, exactly as the TRAIN source cameras; the FP32 official-grid K is retained only for the map/equality check. Constructing full projection matrices from a prematurely FP32-rounded K is not assumed bitwise identical. Camera conversion uses the separately tested corner contract, including integer-array centers and corrected half-pixel rays. This does not authorize a generic overscan approximation.

## Frozen scoring and references

Copy the complete already-used scoring package from
`/mnt/data/SHM2026/runs/rgb_fixed_ensemble_v1/source_snapshot/bridge_rgs` into a separate scoring import root. In that package:

- `official_evaluate.py`, SHA256 `9b3799367a983495677122bf1bf75b06116342fde703466e43e3d8de092433f0`, supplies `score_official_arrays`, `_decode_rgb`, and `_lpips`.
- `evaluate.py`, SHA256 `5c180581ad680f76a708f73abcc5850dad0ec6f4c3aa25c664bd9d0d217e2b2f`, supplies the original-grid distortion map.
- RGB-only calls use the existing zero-mask API placeholder and `target_mask=None`; no semantic output is saved or scored.

Delivered uint8 RGB is decoded to FP32 /255. PSNR uses full-image RGB MSE accumulated in FP64; SSIM uses the fixed Gaussian 11×11/sigma1.5/population-covariance formula and complete window centers; LPIPS Alex v0.1 uses full-resolution [-1,1] RGB, no crop, resize, replacement or mask. Bind these existing local weights:

- `/home/sky/workspace/SHM2026/.venv/lib/python3.11/site-packages/lpips/weights/v0.1/alex.pth`, SHA `df73285e35b22355a2df87cdb6b70b343713b667eddbda73e1977e0c860835c0`.
- `/home/sky/.cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth`, SHA `7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02`.

Both references contain exactly the same 50 named, sized, original-grid views and common fingerprint `21a2f19c5d4d703403a0107402d5dd5e006dcd98e5f8e023820c388c3e69a85d`:

| Reference | Completed per-view metrics | SHA256 |
|---|---|---|
| Original AA 1M | `/mnt/data/SHM2026/runs/rgb_capacity_1m_reference_v1/evaluation_official/official_metrics.json` | `0b4b90fca81ad4f220794b9656ce308cca4dc68a8efe83769efbfde2cd51100e` |
| E delivered RGB | `/mnt/data/SHM2026/runs/rgb_fixed_ensemble_v1/rgb_metrics.json` | `40d022bc522315eaa864fa114261f6e66c5ca6991b3de74bf309a3a4e2ad2ef9` |

The latter is the same delivered composite RGB bound by `multifield_h3_teacher_v1` E, not a newly generated prediction. Reuse per-view scores only after camera/source-RGB/grid/scorer identity checks. The inherited semantic fingerprint may be carried as reference metadata; this RGB-only run does not revalidate annotation pixels.

## Fixed comparisons and interpretation

Report four sets of PSNR/SSIM/LPIPS means and all 50 per-view rows. Predeclare ten paired comparisons: every output against each of AA 1M and E (eight), plus full-source minus no-source separately for raw and fused (two). Use sorted camera names, 5000 shared view-bootstrap replicates, NumPy `default_rng(20260926)`, and candidate-minus-reference differences. No correction or new test subset is selected afterward.

For each comparison disclose the four engineering clauses: mean PSNR gain ≥.15 dB; paired PSNR 95% lower bound >0; mean SSIM nondecreasing; mean LPIPS nonincreasing. These are references for further investment, not automatic replacement of E, proof of generalization, or a semantic-system adoption rule. Main conclusions retain the fixed full-source/fused candidate even if another branch scores higher. All views have already been used in development.

The separately fixed 16 TRAIN start/end diagnostics distinguish AA-to-port restoration from later improvement. Report original AA, port step-zero, each endpoint raw/fused errors with their actual shared support and stage costs. They neither select the final endpoint nor change the official scoring grid. Source-input ablation changes its downstream geometry gradients during joint training: it measures the source-input pathway's total effect under common source-photo supervision, not a pure frozen-readout effect.

Semantics remain the previously measured E semantics, explicitly inherited and unchanged. This run supplies no new semantic evidence or measured joint deployment result. New RGB cannot be attributed to a shared semantic mechanism merely by associating it with E's existing numbers.

Worker interface: `--prepare --training <completed-root> --training-plan-sha256 <sha> --output <fresh-root>` in the main uv environment; then the frozen entry with `--render --plan ... --expected-plan-sha256 ...` in the IBGS environment. Root records its natural `render_launch_receipt.json` before the main-environment `--score --plan ... --expected-plan-sha256 ...`. Budgets are 600/660 seconds for rendering and 300/360 seconds for scoring. The main scoring flags are read and bound at CPU preparation, then set/read back during scoring; the renderer uses the training TF32-disabled flags.
