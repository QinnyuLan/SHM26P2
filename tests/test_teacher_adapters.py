"""CPU contracts for LoRA weights, small checkpoints and safe EMA switching."""

from copy import deepcopy

import pytest
import torch
from torch import nn

from bridge_rgs.teacher_adapters import (
    LowRankLinear,
    adapter_named_parameters,
    adapter_state_dict,
    install_dinov3_lora,
    load_adapter_state_dict,
    temporary_adapter_state,
    update_adapter_ema,
)


class TinyAttention(nn.Module):
    def __init__(self, dim=8):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)

    def forward(self, x):
        return self.q_proj(x).tanh() + self.k_proj(x) + self.v_proj(x)


class TinyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = TinyAttention()

    def forward(self, x):
        return self.attention(x)


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.ModuleList([TinyBlock() for _ in range(6)])

    def forward(self, x):
        for layer in self.layer:
            x = layer(x)
        return x


@pytest.fixture(autouse=True)
def fixed_cpu_runtime():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(47)
    yield
    torch.set_num_threads(previous)


def adapted_backbone():
    backbone = TinyBackbone()
    install_dinov3_lora(backbone, last_n=2, rank=2, alpha=4)
    return backbone


def test_initial_function_scaling_and_frozen_base():
    base = nn.Linear(5, 3)
    x = torch.randn(2, 4, 5)
    expected = base(x).detach()
    adapter = LowRankLinear(base, rank=2, alpha=6)
    torch.testing.assert_close(adapter(x), expected, atol=0, rtol=0)
    assert adapter.weight is base.weight and adapter.bias is base.bias
    adapter(x).square().mean().backward()
    assert base.weight.grad is None and base.bias.grad is None
    assert adapter.lora_A.grad is not None and adapter.lora_A.grad.count_nonzero() == 0
    assert adapter.lora_B.grad.abs().sum() > 0
    with torch.no_grad():
        adapter.lora_B.normal_(std=.1)
    adapter.zero_grad(set_to_none=True)
    expected = base(x) + 3 * (x @ adapter.lora_A.T @ adapter.lora_B.T)
    torch.testing.assert_close(adapter(x), expected)
    adapter(x).square().mean().backward()
    assert adapter.lora_A.grad.abs().sum() > 0
    assert adapter.lora_B.grad.abs().sum() > 0
    assert base.weight.grad is None and base.bias.grad is None


@pytest.mark.parametrize("amp", [False, True])
def test_bfloat16_inputs_and_amp_keep_fp32_master_weights(amp):
    adapter = LowRankLinear(nn.Linear(8, 8), rank=2)
    original_a = adapter.lora_A.detach().clone()
    adapter = adapter.to(dtype=torch.bfloat16)
    assert adapter.base.weight.dtype == torch.bfloat16
    assert adapter.lora_A.dtype == adapter.lora_B.dtype == torch.float32
    torch.testing.assert_close(adapter.lora_A, original_a, atol=0, rtol=0)
    with torch.no_grad():
        adapter.lora_B.normal_(std=.1)
    inputs = torch.randn(3, 8, dtype=torch.bfloat16, requires_grad=True)
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=amp):
        result = adapter(inputs)
        loss = result.float().square().mean()
    loss.backward()
    assert result.dtype == torch.bfloat16
    for parameter in (adapter.lora_A, adapter.lora_B):
        assert parameter.grad.dtype == torch.float32
        assert torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0
    assert inputs.grad is not None and torch.isfinite(inputs.grad).all()
    assert adapter.base.weight.grad is None


def test_install_last_layers_freezes_base_and_saves_only_independent_adapters():
    backbone = TinyBackbone()
    x = torch.randn(2, 5, 8)
    before = backbone(x).detach()
    names = install_dinov3_lora(backbone, last_n=2, rank=2, alpha=4)
    assert names == [f"layer.{i}.attention.{projection}" for i in (4, 5)
                     for projection in ("q_proj", "v_proj")]
    torch.testing.assert_close(backbone(x), before, atol=0, rtol=0)
    assert isinstance(backbone.layer[3].attention.q_proj, nn.Linear)
    assert isinstance(backbone.layer[4].attention.k_proj, nn.Linear)
    adapters = dict(adapter_named_parameters(backbone))
    assert len(adapters) == 8
    assert all(parameter.requires_grad == (name in adapters)
               for name, parameter in backbone.named_parameters())
    state = adapter_state_dict(backbone, device="cpu")
    assert set(state) == set(adapters)
    assert sum(value.numel() for value in state.values()) == 2 * 2 * (8 * 2 + 2 * 8)
    for name, value in state.items():
        assert value.dtype == torch.float32 and not value.requires_grad
        assert value.data_ptr() != adapters[name].data_ptr()
    with torch.no_grad():
        next(iter(adapters.values())).add_(1)
    assert not torch.equal(next(iter(state.values())), next(iter(adapters.values())))


def test_strict_state_load_preserves_parameters_and_rejects_partial_overwrite():
    backbone = adapted_backbone()
    parameters = dict(adapter_named_parameters(backbone))
    state = {name: torch.randn_like(value) for name, value in adapter_state_dict(backbone).items()}
    load_adapter_state_dict(backbone, state)
    for name, parameter in adapter_named_parameters(backbone):
        assert parameter is parameters[name]
        torch.testing.assert_close(parameter, state[name], atol=0, rtol=0)
    online = adapter_state_dict(backbone)
    first, last = next(iter(online)), next(reversed(online))
    for malformed in (
        {name: value for name, value in online.items() if name != last},
        {**online, "base.weight": torch.zeros(1)},
        {**online, first: online[first] + 7, last: torch.zeros(1)},
        {**online, first: online[first] + 7, last: online[last].bfloat16()},
    ):
        with pytest.raises(ValueError):
            load_adapter_state_dict(backbone, malformed)
        for name, value in adapter_state_dict(backbone).items():
            torch.testing.assert_close(value, online[name], atol=0, rtol=0)


def test_ema_is_detached_and_preserves_state_storage():
    backbone = adapted_backbone()
    ema = adapter_state_dict(backbone)
    before = {name: value.clone() for name, value in ema.items()}
    pointers = {name: value.data_ptr() for name, value in ema.items()}
    with torch.no_grad():
        for _name, parameter in adapter_named_parameters(backbone):
            parameter.add_(2)
    update_adapter_ema(ema, backbone, .75)
    for name, value in ema.items():
        torch.testing.assert_close(value, before[name] + .5)
        assert value.data_ptr() == pointers[name]
        assert not value.requires_grad and value.dtype == torch.float32
    with pytest.raises(ValueError, match="decay"):
        update_adapter_ema(ema, backbone, float("nan"))


def test_temporary_state_restores_identity_gradients_and_values_after_exception():
    backbone = adapted_backbone()
    originals = dict(adapter_named_parameters(backbone))
    state = adapter_state_dict(backbone)
    for parameter in originals.values():
        parameter.grad = torch.ones_like(parameter)
    gradients = {name: parameter.grad for name, parameter in originals.items()}
    optimizer = torch.optim.Adam(originals.values())
    alternate = {name: value + .25 for name, value in state.items()}
    with pytest.raises(RuntimeError, match="test failure"), temporary_adapter_state(backbone, alternate):
        for name, parameter in adapter_named_parameters(backbone):
            assert parameter is not originals[name]
            assert parameter.device == originals[name].device
            assert parameter.dtype == originals[name].dtype
            torch.testing.assert_close(parameter, alternate[name], atol=0, rtol=0)
        raise RuntimeError("test failure")
    for name, parameter in adapter_named_parameters(backbone):
        assert parameter is originals[name] and parameter.grad is gradients[name]
        torch.testing.assert_close(parameter, state[name], atol=0, rtol=0)
    assert all(saved is live for saved, live in
               zip(optimizer.param_groups[0]["params"], originals.values(), strict=True))


def test_ema_forward_between_online_forward_and_backward_keeps_graph_valid():
    backbone = adapted_backbone()
    with torch.no_grad():
        for name, parameter in adapter_named_parameters(backbone):
            if name.endswith("lora_B"):
                parameter.normal_(std=.1)
    reference = deepcopy(backbone)
    x = torch.randn(2, 5, 8)
    online_output = backbone(x)
    state = adapter_state_dict(backbone)
    alternate = {name: value + .4 for name, value in state.items()}
    with torch.no_grad(), temporary_adapter_state(backbone, alternate):
        assert not torch.equal(backbone(x), online_output)
        outer_parameters = dict(adapter_named_parameters(backbone))
        with temporary_adapter_state(backbone, state):
            torch.testing.assert_close(backbone(x), online_output, atol=0, rtol=0)
        assert all(parameter is outer_parameters[name]
                   for name, parameter in adapter_named_parameters(backbone))
    online_output.square().mean().backward()
    reference(x).square().mean().backward()
    expected = dict(adapter_named_parameters(reference))
    for name, parameter in adapter_named_parameters(backbone):
        torch.testing.assert_close(parameter.grad, expected[name].grad, atol=0, rtol=0)
        assert parameter.grad.abs().sum() > 0


def test_invalid_install_leaves_backbone_unchanged():
    backbone = TinyBackbone()
    backbone.layer[-1].attention.v_proj = nn.Identity()
    with pytest.raises(TypeError, match="v_proj"):
        install_dinov3_lora(backbone, last_n=2)
    assert not any(isinstance(module, LowRankLinear) for module in backbone.modules())
    assert all(parameter.requires_grad for parameter in backbone.parameters())
    installed = adapted_backbone()
    original = dict(adapter_named_parameters(installed))
    with pytest.raises(ValueError, match="already installed"):
        install_dinov3_lora(installed, last_n=1)
    assert all(parameter is original[name] and parameter.requires_grad
               for name, parameter in adapter_named_parameters(installed))


@pytest.mark.parametrize("arguments", [{"rank": 0}, {"rank": 1.5}, {"alpha": 0},
                                       {"alpha": float("nan")}])
def test_invalid_adapter_hyperparameters(arguments):
    with pytest.raises(ValueError):
        LowRankLinear(nn.Linear(4, 4), **arguments)
