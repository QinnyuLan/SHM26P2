# Fixed geometry-gradient chain diagnostic

This is an unexecuted diagnostic of one reconstructed instance: TRAIN `041.png`, the RGB-own direction, and covariance-unit amplitude **1/128** from `continuous_geometry_preflight_v1`. The prior run completed normally but passed only 3/12 own-gradient checks; that result and its 5% gate remain unchanged. This diagnostic neither admits training nor changes a model.

The old run did not retain complete gradients or perturbed means. Recompute its direction from the same frozen H3 scene, source package, RGB target, valid mask, original camera and numerical flags. Report differences from the old baseline loss, covariance normalizer, actual-displacement analytic dot and finite difference. Even matching summaries cannot establish byte-exact identity with the old direction. If they differ, the output describes this reconstructed instance, not a causal decomposition of the old endpoints.

## Fixed computation

Use three standard gsplat RGB+ED calls at original means and independently rounded FP32 means `base ± h*d`. All other parameters are frozen. The positive gradient direction is `d_i = Sigma_i*g_i / max_j sqrt(g_j^T Sigma_j g_j)`, computed in FP64; dots use the actual FP32 endpoint displacement. RGB MSE uses clamped RGB, FP64 arithmetic and the fixed valid-pixel mean (including the three channels). No semantic target, raw semantic render, head, teacher, optimizer or VAL data is used; the fixed q sidecar is bound only as parent provenance.

Intercept the existing Python call to `gsplat.rendering.rasterize_to_pixels`, retaining its actual tensors and output without replacing the forward result. One complete backward sweep collects means, projected centers, conics, effective opacity after AA compensation, SH color plus raw depth, and raw four-channel image VJPs. The means are also an explicit autograd input, so the backward continues through the intermediate inputs.

For `s(X) = (X_plus-X_minus)/(2h)`, report:

- **A**: `grad_means · s(means)`.
- **B**: sum of `grad_X · s(X)` over actual projected centers, conics, AA opacity and four-channel colors. Only nonzero-VJP rows with valid projections at all three states are evaluated. If any nonzero-VJP row lacks an endpoint projection, the complete B is unavailable; its valid-row partial sum is identified separately. Uninitialized culled projection storage is never interpreted as zero-valued evidence.
- **C**: original raw-image VJP dotted with the raw-image secant.
- **D**: `(F_plus-F_minus)/(2h)`.

Differences localize disagreements to portions of this finite secant chain, not necessarily a particular CUDA defect. Also report the exact FP64 quadratic identity for clamped images `P`, with the same valid support and scalar count `N`:

`D = 2<P0-Y, Pplus-Pminus>/(2h*N) + <Pplus-Pminus, Pplus+Pminus-2P0>/(2h*N)`.

The difference between raw-image C and this clamped-image term includes both the clamp contribution and FP32 image-VJP casting/rounding versus the FP64 analytic expression; it is not an exact isolated clamp effect. Report clamp crossings, endpoint rows whose actual Mahalanobis movement exceeds the intended h, their signed/absolute gradient-dot contributions, and projection/ledger identity changes. Counts or maximum motion alone do not establish causality.

## Conditional fixed-ledger replay

Save the exact low-level four-channel RGB+raw-depth colors, background, AA opacity, tile offsets and ordered flattened Gaussian IDs. Replay baseline with the same low-level callable; all four output channels and alpha must be byte-exact. Do not hide the depth channel or substitute three-channel rasterization.

Only if **every Gaussian referenced by the baseline ledger** has a valid projection at both endpoints, replay each endpoint's continuous attributes using baseline tile offsets and ordered IDs. Otherwise skip both endpoint replays and retain A/C/D and the explicitly partial B. Do not restrict to an intersection, synthesize missing attributes, or treat such a replay as the original scene. Freezing tile inclusion and order jointly does not freeze per-pixel sigma/alpha tests, alpha clipping or early termination. Natural-versus-fixed replay differences therefore identify a joint ledger effect only; they do not isolate sorting from support and need not be additive causal contributions.

Maximum work is **3 standard RGB rasters + 3 low-level replays, 1 backward sweep, 2 target-file decodes**. If endpoint replay is unsafe, there are only 1 baseline replay and 4 total rasters. Internal limit 120 seconds, external limit 180 seconds, including saving and hashing arrays; no retry or extra amplitudes. All products go to a new `/mnt/data` directory.

## Evidence and execution

The standalone worker is `scripts/diagnose_geometry_chain.py`; `--prepare OUTPUT` copies the original frozen preflight package plus this worker, its synthetic tests and this protocol. Preparation binds the original plan, natural launch, execution and independent CPU review; H3 checkpoint, manifest, q sidecar, 041 RGB/valid, runtime versions, numerical flags, installed gsplat sources and actual binary. The subsequent frozen entry uses `--run PLAN --expected-plan-sha256 SHA`. Preparation and GPU execution have not been performed for this revision.

Preserve per-state FP32 means, raw four-channel images/alpha, active projection attributes, radii and tile ledgers; baseline gradients, FP64 direction, original covariance factors, target and valid grid permit CPU recomputation. Invalid stored attributes are explicitly zero placeholders paired with a validity mask, not measurements. The receipt records actual frozen local imports, renderer binary, counts, memory scope, input/source rechecks and complete model/numerical restoration. Failed attempts remain failed. `completed_localization_only` is not a numerical pass, convergence, capacity finding or performance gain.
