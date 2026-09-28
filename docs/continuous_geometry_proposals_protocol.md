# Continuous means proposals with actual batch acceptance

This independent protocol treats the installed gsplat VJP as a **local approximate proposal**, not a certified derivative of the finite floating-point rendered objective. It does not repair or replace the old `continuous_geometry_preflight_v1` result: its original 12 measurable checks passed only 3/12 at the fixed 5% threshold. Its independent audit passing means the records were verified, not that this numerical gate passed. The old `train_continuous_semantic_geometry.py` preparation gate is retained unchanged.

The separately completed `geometry_chain_diagnostic_v1` reconstructed TRAIN 041 at amplitude 1/128. A/B were approximately 2.23712e-4 / 2.237416e-4, C/D were 1.341876e-4 / 1.358423e-4, and fixed-ledger D was 2.367418e-4. These provide evidence for an important local tile-inclusion/order effect in that instance. Fixed-ledger D still differs from A; thresholds and other finite-precision effects were not eliminated. This is neither a CUDA-defect conclusion nor a global error bound. Preparation binds both completed executions, natural launches and independent audits by SHA, including the original failed numerical status.

## Fixed population and computation

Use the original H3 checkpoint (SHA `22bc8a2ddb260f93cb01b17857c97b2bb0873038efdb9318545cb2bdbb045226`) and the completed EM/FW `q_renderer` exactly as stored. Do not renormalize q or map it into features. Preserve Gaussian count, every non-means tensor, all 350 original TRAIN cameras, SH degree 3, legacy native grids and antialiased rasterization. The 259 labeled TRAIN names are sorted, then a private Python `Random(42)` shuffles a fresh sorted index list in each of four rounds. Each round has 37 batches of seven, giving 148 attempts per arm. All arms start from the same original means and fresh Adam state. There is no augmentation, head/teacher call, VAL scoring, topology update or production checkpoint replacement.

Each baseline and candidate evaluates all seven views through two unchanged standard gsplat calls: SH RGB+depth and explicit fixed-q raw semantics with shared alpha. RGB objective is FP64 mean squared error of clamped rendered RGB versus prepared uint8 RGB/255 over valid RGB pixels and channels. Semantic objective is the inherited weighted affine-noise raw CE, delta 5e-7, with the unchanged preflight class weights, divided by each view's known-valid pixel count. Both objectives are equal means over views; RGB and semantic support sizes may differ. Each task's FP32 means VJP is converted to FP64 before accumulation and division by seven.

## Three declared proposal rules

- `joint`: propose with `g_RGB + 0.03 g_semantic`.
- `pcgrad`: symmetric two-task PCGrad on `g_RGB` and `0.03 g_semantic`, using one global flattened inner product and both original task vectors. Conflicting projections are summed. Adam and the subsequent pointwise cap do not preserve the unpreconditioned projection's first-order properties.
- `trust`: propose with unweighted `g_semantic` and test the actual candidate below.

All arms use means-only Adam with constant LR `1.6e-6 * scene_scale`, betas `(0.9,0.999)`, epsilon `1e-15`, no weight decay. Geometry defines a fixed FP64 metric from normalized wxyz quaternions and `exp(log_scales)`. Each point's **incremental** displacement is capped at Mahalanobis length 1/64. After casting the capped candidate to FP32, a point exceeding the same cap is restored exactly. This is not a cumulative trust region about the original field. Single-point fallback retains the new Adam moments if the whole proposal is accepted; whole-proposal rejection restores means, moments and step. A current zero task combination need not imply a zero Adam step when historical moments exist.

`joint` and `pcgrad` commit every complete finite candidate, including no-ops. They are matched controls defined using this library VJP and have no objective-descent guarantee. `trust` commits only when all three actual batch checks hold:

1. Batch semantic CE strictly decreases.
2. Batch RGB MSE is at most its baseline plus `max(1e-6 * baseline, 1e-12)`.
3. Every view's RGB MSE is at most `1.001 *` its own baseline.

Otherwise the full transaction is rejected, with no alternate proposal or retry. Partial renders, exceptions, nonfinite values or exhausted time terminate the run and preserve failure. Actual objective acceptance is a well-defined discrete algorithm even when proposal derivatives are approximate. Its checks concern the current batch only; they imply neither monotonic full-TRAIN/VAL behavior nor stationary-point convergence. No result converts the old 5% gate into a pass.

## Budget, records and interpretation

For each arm, baseline+candidate means 148 * 7 * 2 view pairs. Across three arms there are 6,216 training pairs and 6,216 means VJPs. One shared initial and three fixed-last descriptive 259-view passes add 1,036 pairs. The complete budget is **14,504 standard rasters, 444 attempts, 6,216 means VJPs**, zero head/teacher/VAL calls. Targets are decoded once and cached, 777 file decodes total. Internal deadline is 1,800 seconds, external limit 1,860 seconds; no truncated-success interpretation or retry.

Save every batch's names, both objectives and support sizes, actual candidate changes, both `g · actual_FP32_delta` values, acceptance and optimizer step, cap scaling, post-cap rounding to the prior value and explicit over-cap fallback counts. Bind initial/final means, fixed q, SH and all other model state by SHA. Each arm writes only a diagnostic means-delta NPZ, separate Adam state for inspection, fixed-last TRAIN scores and receipt; these are not ordinary resumable scene checkpoints. Descriptive full-TRAIN RGB, raw CE and pooled raw confusion matrices use the fixed endpoints, never a best endpoint. Restore the complete scene and numerical flags in `finally`; recheck frozen sources, inputs and actual renderer binary.

The new worker is `scripts/train_continuous_geometry_proposals.py`. Its snapshot inherits the exact 72-source failed-preflight package, adding only the reviewed transaction helper, explicit worker/tests, unchanged old draft for regression and this protocol. CPU prepare hashes bytes without decoding pixels or loading checkpoints. Preparation does not launch a GPU job. The experiment is an engineering comparison of declared proposal/acceptance rules; there is no automatic adoption gate, new-formula claim, proof of a geometry bottleneck or held-out performance claim.
