# Frozen scene readout: full versus prior-only geometry gradients

2026-09-27. CPU implementation prepared for review; no preparation or GPU execution is authorized by this document alone. This is a matched path ablation, not a new stop-gradient algorithm or a performance adoption rule. The earlier raw-objective geometry FD calibration remains 3/12; the later two-camera readout probe established forward compatibility and finite actual descent examples, not a global means-derivative certificate.

The fixed source is `readout_geometry_probe_v1/source_snapshot`, including its tested standard-gsplat adapter and explicit differentiable H3 context. The original H3 checkpoint and the audited final EM/FW categorical sidecar remain identical to that probe. All 498136 Gaussian means start at original H3 values. q, features, head, SH, opacity, scale, rotation, cameras, normalization state and every other model tensor remain frozen. There is no q-to-feature conversion, production checkpoint, teacher, VAL input or RGB ensemble in this experiment.

Write the readout as `softmax(log p + refiner(p,e))`, where e contains rendered features, RGB, depth, alpha and all cross moments. Both arms use this exact forward. The full arm uses its complete standard-gsplat means VJP. The prior-only arm stops all five e inputs at the head boundary while retaining p in the external log prior and the head's p3d/entropy channels. It is not the adapter's `geometry_grad=False` option. Every gradient view runs both heads and requires bitwise-identical probabilities and identical scalar losses before making a proposal.

Both arms compute all three means VJPs: RGB MSE, complete scene CE, prior-only scene CE. Each per-view FP32 VJP is converted to FP64 before an equal-view batch mean. RGB MSE uses clamped RGB and valid pixels. Both semantic readout objectives use the probe's class-weighted affine-noise CE, delta=5e-7, evaluated in FP64; class weights and valid-known pixel denominator are unchanged. There is no Lovasz or residual penalty. Raw affine CE is descriptive only. The proposal is `gRGB + .03*gRoute`, with route full or prior-only. The descriptive context gradient is `gFull-gPrior`, not an independently validated derivative.

The fixed 259 labeled TRAIN names are sorted. Python Random(42) shuffles a fresh index list for each of four rounds: 37 batches of seven per round, 148 attempts per arm, in identical order. Both arms use a fresh Adam, lr `1.6e-6*scene_scale`, betas (.9,.999), eps 1e-15. The unchanged inherited transaction imposes the original-covariance Mahalanobis cap 1/64 per Gaussian per step, restores a point if FP32 rounding exceeds that cap, and rolls back means plus all Adam state on any failed candidate. Every complete finite candidate commits, including no-op or objective increases. There is no loss-based acceptance, line search, early endpoint selection or retry. The cap is not a cumulative bound about original H3.

Each attempt stores objective values, averaged-gradient norms and cross-path inner products, actual FP32 displacement norms and its dot with RGB/full/prior/context gradients. Negative cosine is not evidence of harmful interference. Adam and the cap can change the relation between gradients and actual updates. Different update magnitudes remain a possible explanation of arm differences; equal budget does not mean equal displacement.

One shared initial and each of the two fixed final endpoints are described on all 259 TRAIN views. Save per-view RGB MSE, raw/scene affine CE, known-pixel raw/scene 5×5 confusion matrices and pooled IoU. These are in-fit descriptions, not generalization or deployment gains. No per-step four-corner counterfactual head evaluations are added. The frozen head is used directly, without teacher fusion.

Before these TRAIN results are observed, the follow-up official comparison is fixed to the shared initial baseline (original H3 means with the same fixed EM/FW q) and both terminal arms, regardless of their TRAIN ordering. A separate evaluator will retain the old deployment direct-q readout and fresh fixed H+ inference. It is not part of this TRAIN worker or its budget and must wait for natural completion and the independent TRAIN audit. No endpoint may be selected or omitted using the TRAIN metrics.

| Fixed resource | Count |
|---|---:|
| Training baseline/candidate adapter calls | 4144 |
| Shared initial + two endpoint calls | 777 |
| Total adapter calls | 4921 |
| High-level standard gsplat calls | 9842 |
| Low-level raster calls (RGB4 + semantic39 split at 32) | 14763 |
| Means VJPs | 6216 |
| Head forwards | 6993 |
| TRAIN RGB/mask/valid decodes | 777 |
| Attempts | 296 |

Both arms pay the same three VJPs and additional prior-only head at every baseline batch view. Candidate and description views use only the complete head. Targets are cached by name and never read from VAL. Internal budget is 600 seconds, external budget 660 seconds, including input/source verification and output I/O. The two-view probe's 4.316 seconds includes loading and is not a reliable throughput extrapolation; this budget is a fixed upper bound, not a promised completion time. Timeout preserves a failed record and stops without reducing steps or extending the budget.

Preparation copies the already completed probe package unchanged, appends only the new worker, its synthetic tests and this protocol, and binds the probe plan/execution/natural-exit/independent audit plus original TRAIN input hashes from the completed three-arm plan. Execution is permitted only from that snapshot and checks actual imported source and the installed gsplat binary. Final means sidecars, optimizer state and attempt records are saved independently. The original scene, modes, requires-grad flags, gradient buffers, numerical flags and temporary counting wrappers are restored in finally. A completed run establishes execution of this fixed comparison, not convergence, superiority, causality or an innovation claim.
