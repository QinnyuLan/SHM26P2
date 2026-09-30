"""Small FP32 Q/V adapters for a frozen Hugging Face DINOv3 backbone.

Only adapter tensors enter the helpers' state dictionaries. The base model is
still loaded from its original checkpoint and never copied for EMA inference.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping, MutableMapping
from contextlib import contextmanager

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class LowRankLinear(nn.Module):
    """Frozen linear layer plus ``(alpha / rank) * B(A(x))``.

    A/B parameters and adapter arithmetic remain FP32, including inside AMP
    and after parent-module dtype conversion. The correction is cast to the
    base output dtype before addition. Zero B preserves the initial function;
    its first backward gives A a zero (but present) gradient, as in standard
    LoRA, and B a learning signal.
    """

    def __init__(self, base: nn.Linear, rank: int = 16, alpha: float = 16.):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("LowRankLinear requires an nn.Linear base")
        if type(rank) is not int or rank <= 0:
            raise ValueError("LoRA rank must be a positive integer")
        if not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("LoRA alpha must be finite and positive")
        self.base = base.requires_grad_(False)
        self.rank = rank
        self.alpha = float(alpha)
        self.scaling = self.alpha / rank
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features,
                                               device=base.weight.device, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank,
                                               device=base.weight.device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    @property
    def weight(self) -> nn.Parameter:
        return self.base.weight

    @property
    def bias(self) -> nn.Parameter | None:
        return self.base.bias

    def _apply(self, fn, recurse=True):
        # nn.Module.to/bfloat16 should convert the base, but must not quantize
        # the trainable master adapter weights or their accumulated gradients.
        if recurse:
            self.base._apply(fn)

        def preserve_fp32(value):
            converted = fn(value)
            if converted.is_floating_point() and converted.dtype != torch.float32:
                return value.to(device=converted.device, dtype=torch.float32)
            return converted

        return super()._apply(preserve_fp32, recurse=False)

    def forward(self, inputs: Tensor) -> Tensor:
        base_result = self.base(inputs)
        with torch.autocast(device_type=inputs.device.type, enabled=False):
            correction = F.linear(F.linear(inputs.float(), self.lora_A), self.lora_B)
            correction = correction * self.scaling
        return base_result + correction.to(dtype=base_result.dtype)


def install_dinov3_lora(backbone: nn.Module, last_n: int = 4,
                        rank: int = 16, alpha: float = 16.) -> list[str]:
    """Freeze a DINOv3 base and adapt Q/V in its final ``last_n`` layers.

    Returns paths such as ``layer.20.attention.q_proj``. All target paths are
    checked before modification. Reinstallation is rejected instead of
    silently changing which adapters are trainable. Save rank/alpha/last_n in
    checkpoint metadata and install the same configuration before state load.
    """
    layers = getattr(backbone, "layer", None)
    if not isinstance(layers, nn.ModuleList):
        raise TypeError("Expected Hugging Face DINOv3 backbone.layer as nn.ModuleList")
    if type(last_n) is not int or not 1 <= last_n <= len(layers):
        raise ValueError("last_n must select between one and all DINOv3 layers")
    if type(rank) is not int or rank <= 0 or not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("LoRA requires positive integer rank and finite positive alpha")
    if any(isinstance(module, LowRankLinear) for module in backbone.modules()):
        raise ValueError("DINOv3 LoRA adapters are already installed")
    targets = []
    for index in range(len(layers) - last_n, len(layers)):
        attention = getattr(layers[index], "attention", None)
        for name in ("q_proj", "v_proj"):
            target = getattr(attention, name, None)
            if not isinstance(target, nn.Linear):
                raise TypeError(f"Expected nn.Linear at layer.{index}.attention.{name}")
            targets.append((f"layer.{index}.attention.{name}", attention, name, target))
    backbone.requires_grad_(False)
    for _path, attention, name, target in targets:
        setattr(attention, name, LowRankLinear(target, rank=rank, alpha=alpha))
    return [path for path, *_rest in targets]


def adapter_named_parameters(backbone: nn.Module) -> Iterator[tuple[str, nn.Parameter]]:
    """Yield only trainable-adapter slots, irrespective of requires_grad flags."""
    for path, module in backbone.named_modules():
        if isinstance(module, LowRankLinear):
            prefix = f"{path}." if path else ""
            yield prefix + "lora_A", module.lora_A
            yield prefix + "lora_B", module.lora_B


def adapter_state_dict(backbone: nn.Module,
                       device: str | torch.device | None = None) -> dict[str, Tensor]:
    """Independent detached FP32 snapshot, optionally moved to e.g. CPU."""
    return {name: parameter.detach().to(device=device).clone()
            for name, parameter in adapter_named_parameters(backbone)}


def _validated_state(backbone: nn.Module, state: Mapping[str, Tensor]) -> dict[str, nn.Parameter]:
    parameters = dict(adapter_named_parameters(backbone))
    missing = sorted(parameters.keys() - state.keys())
    unexpected = sorted(state.keys() - parameters.keys())
    if missing or unexpected:
        raise ValueError(f"Adapter state keys differ: missing={missing}, unexpected={unexpected}")
    for name, parameter in parameters.items():
        value = state[name]
        if not isinstance(value, Tensor) or value.dtype != torch.float32:
            raise ValueError(f"Adapter state {name} must be an FP32 tensor")
        if value.shape != parameter.shape:
            raise ValueError(f"Adapter state shape mismatch for {name}: "
                             f"expected {tuple(parameter.shape)}, received {tuple(value.shape)}")
    return parameters


@torch.no_grad()
def load_adapter_state_dict(backbone: nn.Module, state: Mapping[str, Tensor]) -> None:
    """Strict keys/shapes/FP32 load, preserving parameter identity and device.

    CPU checkpoint tensors may load into CUDA adapters. Validate the complete
    state before copying so malformed checkpoints cannot partially overwrite
    an online model. This is a checkpoint operation, not an EMA context switch.
    """
    parameters = _validated_state(backbone, state)
    for name, parameter in parameters.items():
        parameter.copy_(state[name].detach().to(device=parameter.device, dtype=parameter.dtype))


@torch.no_grad()
def update_adapter_ema(ema_state: MutableMapping[str, Tensor], backbone: nn.Module,
                       decay: float) -> None:
    """Update detached adapter EMA in place on its existing device/dtype."""
    if not math.isfinite(decay) or not 0 <= decay <= 1:
        raise ValueError("EMA decay must lie in [0, 1]")
    parameters = _validated_state(backbone, ema_state)
    if any(value.requires_grad for value in ema_state.values()):
        raise ValueError("EMA state must be detached; use adapter_state_dict")
    for name, parameter in parameters.items():
        target = ema_state[name]
        target.lerp_(parameter.detach().to(device=target.device), 1 - decay)


@contextmanager
def temporary_adapter_state(backbone: nn.Module, state: Mapping[str, Tensor]):
    """Temporarily bind adapter copies and always restore original Parameters.

    Rebinding, instead of in-place parameter copies, preserves online parameter
    versions and optimizer references. Thus an online autograd graph can remain
    live across an EMA weak-view prediction. Nested contexts are supported.
    Base weights, module training mode and caller gradient mode are untouched;
    use ``torch.no_grad()`` for the evaluation/pseudo-label forward. Do not run
    an optimizer step while temporary parameters are bound.
    """
    parameters = _validated_state(backbone, state)
    bindings = []
    for name, original in parameters.items():
        module_path, _, attribute = name.rpartition(".")
        module = backbone.get_submodule(module_path) if module_path else backbone
        replacement = nn.Parameter(state[name].detach().to(
            device=original.device, dtype=original.dtype).clone(),
            requires_grad=original.requires_grad)
        bindings.append((module, attribute, original, replacement))
    try:
        for module, attribute, _original, replacement in bindings:
            setattr(module, attribute, replacement)
        yield backbone
    finally:
        for module, attribute, original, _replacement in bindings:
            setattr(module, attribute, original)
