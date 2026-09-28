"""Two real H+ updates verify adapter gradients and a frozen ModelScope base."""

import json
from pathlib import Path

import numpy as np
import torch

from bridge_rgs.teacher import ViewReader, _amp, aligned_crop, load_teacher, supervised_loss
from bridge_rgs.teacher_adapters import adapter_named_parameters, adapter_state_dict

torch.set_num_threads(8)
torch.manual_seed(11)
manifest = json.loads(Path("artifacts/prepared/manifest.json").read_text())
view = next(v for v in manifest["views"] if v["split"] == "train" and v.get("mask_path"))
image, labels, _ = aligned_crop(*ViewReader(5).read(view), 768, np.random.default_rng(11))
image, labels = image[None].cuda(), labels[None].cuda()
model = load_teacher("models/dinov3-vith16plus", 5, adapter_rank=16)
adapters = dict(adapter_named_parameters(model.backbone))
initial = adapter_state_dict(model.backbone)
parameters = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(parameters, lr=1e-4)
records = []
for step in range(2):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    with _amp("cuda"):
        output = model(image)
        loss, _ = supervised_loss(output, labels)
    loss.backward()
    assert all(p.grad is None for name, p in model.backbone.named_parameters() if name not in adapters)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in adapters.values())
    records.append({
        "step": step,
        "loss": float(loss.detach()),
        "A_gradient_l1": sum(float(p.grad.abs().sum()) for name, p in adapters.items() if name.endswith("lora_A")),
        "B_gradient_l1": sum(float(p.grad.abs().sum()) for name, p in adapters.items() if name.endswith("lora_B")),
    })
    optimizer.step()
assert records[-1]["A_gradient_l1"] > 0 and records[-1]["B_gradient_l1"] > 0
result = {
    "training_view": view["name"],
    "adapter_parameters": sum(p.numel() for p in adapters.values()),
    "base_parameters_have_no_gradient": True,
    "all_adapter_gradients_finite": True,
    "adapter_changed_l1": sum(float((p.detach() - initial[name]).abs().sum()) for name, p in adapters.items()),
    "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
    "updates": records,
}
destination = Path("runs/teacher_adapter_cuda_report.json")
destination.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
