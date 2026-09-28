# Fixed renderer starting-point decomposition

Prepared design; no new forwards have been executed. This is one 48-forward TRAIN diagnostic, with no training, model updates, backend edits, parameter search, VAL or semantic targets. It does not select a new system or establish novelty. The original 1M checkpoint is `35b489fe…078092`, with 996009 Gaussians and SH3; all 16 target cameras were already seen during its training.

Use the exact `residual_transport_probe_v1` frozen scene package, native `colmap_corner_v2`, manifest K/w2c cast to FP32, and the old gsplat binary `b0867475…246106`. Fix names to 002/021/041/059/079/100/118/137/156/176/200/220/241/259/278/300. Run these conditions, in this order, with `eps2d=.3`, `far_plane=1e6`, RGB+ED, no semantic/head rendering:

| ID | Renderer | near plane | Work |
|---|---|---:|---|
| A | gsplat antialiased | .01 | 16 new forwards |
| C | gsplat classic | .01 | 16 new forwards |
| N | gsplat classic | .2 | 16 new forwards |
| I | original IBGS start-transfer | its existing .2 CUDA cull | reuse the 16 saved raw FP32 outputs |

I is the unchanged original warm starting field, not either trained IBGS endpoint. Bind its completed plan, execution, natural launch and prediction hashes. Both source packages remain unchanged. Actual high/low raster entry files, installed projection/SH/raster sources, binary, versions, runtime flags and environment are bound and checked. A uses the old probe flags (cuDNN TF32 true, matmul TF32 false, highest, benchmark false); I retains its historical execution provenance. This experiment does not promise identical arithmetic between renderer libraries.

Save and hash all 48 raw FP32 HWC RGB arrays on `/mnt/data` before loading any cached target payload. Then require every clipped A array to be `array_equal` to its old clipped AA cache; save exact/max-error records and stop on any mismatch, with no relaxed tolerance. Check old FP32 cameras and valid support too. Only afterwards load the 16 already-cached TRAIN target RGB/valid arrays, never the original image, annotation or semantic payloads. CPU preparation may hash those bound cache files but does not decode them or load the checkpoint.

Report FP64 raw and clipped MSE over all RGB channels on each fixed valid mask, equal-view MSE, and mean per-view PSNR (not PSNR of mean MSE). Report the fixed-order identity `MSE(I)-MSE(A) = [MSE(C)-MSE(A)] + [MSE(N)-MSE(C)] + [MSE(I)-MSE(N)]`, per view and for equal-view means. This is telescoping accounting, not independent causal attribution: AA also changes support/bounds; near culling interacts with compositing. The remaining I−N includes alpha cap (.99 vs .999), support/radius and numerical projection/SH ordering differences. Do not simulate a pixel-alpha cap by clipping Gaussian opacity. Common covariance blur .3, mathematically matching SH convention, and already-correct corner/array pixel-center mapping are retained.

Only record easy camera-z counts (CPU FP64 evaluation of stored FP32 means/poses, explicitly not CUDA projection-ledger identities) and actual `radii>0` projected counts. No compensation/rho hook or new threshold. `z==.2` is reported separately because IBGS rejects equality while gsplat's near comparison differs. No error-region selection, fitting or new gate.

One attempt, internal 90 seconds / external 120 seconds; expected 48 scene / 48 high / 48 low raster calls, zero backward/optimizer. Exclusive attempt marker precedes GPU work. Restore hooks, numerical flags, scene modes/gradient flags and verify exact model tensors. Serialize strict native JSON before exclusive file creation. Keep any failure and its completed predictions; do not retry or overwrite. Root alone launches GPU. Synthetic tests run with bytecode disabled and `-p no:cacheprovider`, including when copied beside the frozen worker.
