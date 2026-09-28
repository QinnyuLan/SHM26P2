# IBGS starting-field transfer on 16 TRAIN views

The fixed 996009-Gaussian, SH3 field loses substantial RGB accuracy when its raw
parameters are moved from the completed antialiased gsplat renderer into the
patched IBGS renderer. Both use the same original camera inputs and cached valid
TRAIN targets. This is a starting-point comparison, not a trained IBGS result.

| Renderer/readout | Equal-view MSE | Mean per-view PSNR |
|---|---:|---:|
| Original AA cached RGB | 0.000885358904 | 32.300961 dB |
| Initial IBGS raw RGB | 0.001620793845 | 28.658327 dB |
| Initial IBGS clipped RGB | 0.001619526471 | 28.661841 dB |

The clipped IBGS difference is **−3.639120 dB** on average; all 16 per-view PSNR
differences are negative. The run naturally completed with 16 raw renders,
zero fusion calls, backward passes or optimizer steps. The field's eight tensor
hashes remained unchanged. All predictions were saved before loading the old
target caches. There was no new original-image decode, semantic annotation or
VAL input. Worker time was **5.970767 s**; an independent outer wall time was not
captured. Peak allocated memory was 1,237,152,768 bytes (about 1.152 GiB).

![Fixed 002 and 041 starting-field comparison](/mnt/data/SHM2026/runs/ibgs_start_transfer_v1/start_transfer_002_041.png)

The two fixed illustrative views show conspicuous structure-edge and texture
differences. These observations do not isolate antialiasing as the cause: the
renderers also differ in backend arithmetic and compositing implementation.
Identical input camera geometry does not mean identical pixel arithmetic. The
field already fitted these TRAIN views, so this comparison does not measure
generalization. Later recovery of this startup gap must be distinguished from
improvement over the original AA reference; no novelty or model gain follows
from this port check. The existing E system remains unchanged.

The immutable artifacts are [plan](/mnt/data/SHM2026/runs/ibgs_start_transfer_v1/plan.json),
[analysis](/mnt/data/SHM2026/runs/ibgs_start_transfer_v1/analysis.json),
[execution receipt](/mnt/data/SHM2026/runs/ibgs_start_transfer_v1/execution_receipt.json)
and [root natural-completion receipt](/mnt/data/SHM2026/runs/ibgs_start_transfer_v1/launch_receipt.json).
Their SHA256 values, respectively, are:

```
858b5601187e9d4d0841983614de46f816259d26bddc3e9b70cbb6f2da0ea42d
104335c2852e6b9056cc340c2c72b9c1b5e61d9c49d1904a05c964340c9714bf
bfd015f7c642a39e9284031de72c6b5756ffc4fa85f05ebe4800f142dfb24b61
50222d3206b4bfd3ea3b3dba9c29c441b321714d827e995cc05918d3852c89c2
```

The corrected backend binary was
`436b2b55df2b37cddda59e606ec0855ad125eb211058b502197674bd2b93adaf`;
the frozen warm-field loader was
`ac0be8afd879ee85dd5ac462af1da2b53675853d652c4b6dfcdf77fb1ef400d5`.
These are reported execution results and provenance, not an independent rerun
of rendering or a causal decomposition of the renderer differences.
