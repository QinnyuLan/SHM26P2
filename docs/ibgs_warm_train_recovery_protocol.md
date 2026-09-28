# Fixed TRAIN startup-to-endpoint IBGS diagnostic

This auxiliary diagnostic describes recovery on the same 16 TRAIN views used by
`ibgs_start_transfer_v1`. It is separate from the 50-view official evaluation and
has no adoption gate, best-checkpoint selection or additional training.

Preparation requires root-observed natural completion of both fixed 6000-step
arms (`full`, `no_source`) and of the starting-field comparison. It binds their
plans, receipts, final checkpoints, the original start-analysis and target-cache
bytes, the TRAIN data contract, and the runtime binary. Final field/head loading
uses the same frozen `load_saved_field`, `SourceAblatedNet` and `fuse_training`
interfaces as the official evaluation. The source package is inherited from the
completed training, rather than from a later live implementation.

The fixed names are 002, 021, 041, 059, 079, 100, 118, 137, 156, 176, 200, 220,
241, 259, 278 and 300. Each arm refreshes all 350 TRAIN source depths from its own
final field, then renders these 16 cameras once, producing raw and fused RGB:
700 source-depth calls, 32 target calls, 64 native FP32 HWC arrays. Camera inputs
are the same raw manifest K and original pose doubles as training; the resulting
FP32 camera pose is checked against the frozen start metadata. Neighbor lists
are the unchanged data-contract lists, excluding each target itself, without
new source ranking, threshold or fallback. Other diagnostic views may appear as
sources. Thus source RGB is permitted during prediction, and this is explicitly
an in-fit TRAIN description, not a held-out or target-RGB-independent result.

All 64 predictions must be saved before loading any of the 16 cached scoring
targets. No original target RGB, semantic annotation or VAL image is decoded for
scoring. The source bank may decode its existing TRAIN RGB as required by the
author's renderer. Target valid masks come from the original start cache and
must be byte-identical to the adapter's native valid support. Scoring reuses the
frozen start-transfer metric: FP64 mean squared RGB error over valid pixels and
all three channels, per-view PSNR, then equal-view means. Both unclipped and
clipped predictions are reported. Clipped per-view differences are shown against
the original AA baseline and the starting raw IBGS field, using the same support.
No official SSIM/LPIPS, bootstrap, threshold tuning or new acceptance gate is
introduced.

Execution has an internal 300-second and external 360-second limit, zero backward
and optimizer steps, fresh output files, and source/input hashes checked before
and after. Cached-target reads are counted separately from permitted source RGB
decodes. Results cannot establish generalization, an isolated antialiasing cause,
or novelty. Recovery from the documented startup renderer gap is not itself a
new performance gain. A failed diagnostic is preserved; it does not authorize a
retry or change the independent official evaluation.
