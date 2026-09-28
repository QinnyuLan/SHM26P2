"""Check pixel-center sampling against the real single-Gaussian rasterizer."""
import pytest
import torch
from torch.nn import functional as F

from bridge_rgs.coordinates import CORNER, LEGACY, sampling_grid
from bridge_rgs.model import configure_cuda


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires the real gsplat CUDA rasterizer")
def test_real_gsplat_alpha_is_centered_on_colmap_half_integer_pixels():
    configure_cuda()
    from gsplat import rasterization

    device = "cuda"
    width, height = 13, 11
    K = torch.tensor([[9., 0, 6.5], [0, 9., 5.5], [0, 0, 1]], device=device)
    _, alpha, info = rasterization(
        means=torch.tensor([[0., 0., 2.]], device=device),
        quats=torch.tensor([[1., 0., 0., 0.]], device=device),
        scales=torch.full((1, 3), .2, device=device),
        opacities=torch.tensor([.6], device=device), colors=torch.ones(1, 3, device=device),
        viewmats=torch.eye(4, device=device)[None], Ks=K[None], width=width, height=height,
        near_plane=.01, far_plane=100., packed=False, rasterize_mode="classic", render_mode="RGB")
    center = info["means2d"][0, 0]
    torch.testing.assert_close(center, K[:2, 2], atol=1e-6, rtol=0)
    values = alpha[0, ..., 0]
    assert divmod(int(values.argmax()), width) == (5, 6)
    torch.testing.assert_close(values[5, 6], torch.tensor(.6, device=device), atol=1e-6, rtol=0)
    torch.testing.assert_close(values[5, 5], values[5, 7], atol=1e-6, rtol=0)
    torch.testing.assert_close(values[4, 6], values[6, 6], atol=1e-6, rtol=0)
    uv = torch.stack((center, center+center.new_tensor([.5, 0])))
    observed = F.grid_sample(values[None, None], sampling_grid(uv, width, height, CORNER)[None, None],
                             align_corners=False)[0, 0, 0]
    torch.testing.assert_close(observed, torch.stack((values[5, 6], .5*(values[5, 6]+values[5, 7]))),
                               atol=1e-6, rtol=0)
    legacy = F.grid_sample(values[None, None], sampling_grid(uv, width, height, LEGACY)[None, None],
                           align_corners=False)[0, 0, 0]
    assert float((observed-legacy).abs().max()) > .01
