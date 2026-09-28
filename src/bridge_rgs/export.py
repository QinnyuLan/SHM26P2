"""Interoperable 3D Gaussian PLY with explicit official semantic properties."""
from pathlib import Path

import numpy as np
import torch


@torch.no_grad()
def export_ply(checkpoint, output):
    from .train import load_scene
    scene, _ = load_scene(checkpoint, device="cpu")
    scene.eval()
    s = scene.splats
    n = len(s["means"])
    data = {
        **{name: s["means"][:, i].numpy() for i, name in enumerate(["x", "y", "z"])},
        **{name: np.zeros(n, dtype=np.float32) for name in ["nx", "ny", "nz"]},
        **{f"f_dc_{i}": s["sh0"][:, 0, i].numpy() for i in range(3)},
    }
    rest = s["sh_rest"].transpose(1, 2).reshape(n, -1).numpy()
    data.update({f"f_rest_{i}": rest[:, i] for i in range(rest.shape[1])})
    data["opacity"] = s["opacity_logits"].numpy()
    data.update({f"scale_{i}": s["log_scales"][:, i].numpy() for i in range(3)})
    # gsplat normalizes quaternion parameters at render time; export that same rotation.
    quaternions = torch.nn.functional.normalize(s["quats"], dim=-1).numpy()
    data.update({f"rot_{i}": quaternions[:, i] for i in range(4)})
    probabilities = scene.semantic_decoder(s["sem_features"]).softmax(-1).numpy()
    data["semantic_confidence"] = probabilities.max(-1)
    data.update({f"semantic_p_{i}": probabilities[:, i] for i in range(5)})
    dtype = [(k, "<f4") for k in data] + [("semantic_id", "u1")]
    vertices = np.empty(n, dtype=dtype)
    for key, value in data.items():
        vertices[key] = value
    vertices["semantic_id"] = probabilities.argmax(-1).astype(np.uint8)
    header = ["ply", "format binary_little_endian 1.0", "comment Bridge-RGS original COLMAP coordinates",
              "comment semantic_id 0=background 1=deck 2=stay_cable 3=tower 4=foundation",
              "comment semantic probabilities are the explicit 3D field before the image refinement head",
              f"element vertex {n}", *[f"property float {key}" for key in data],
              "property uchar semantic_id", "end_header"]
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as stream:
        stream.write(("\n".join(header) + "\n").encode("ascii"))
        vertices.tofile(stream)
    return output
