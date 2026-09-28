import math

import torch

from bridge_rgs.losses import opacity_entropy


def test_entropy_finite_symmetric_and_bounded_at_extreme_logits():
    x = torch.tensor([-1000., -100., -2., 0., 2., 100., 1000.], requires_grad=True)
    assert torch.equal(opacity_entropy(x), opacity_entropy(-x))
    assert 0 <= opacity_entropy(x) <= math.log(2)
    opacity_entropy(x).backward()
    assert torch.isfinite(x.grad).all()
    assert x.grad[2] > 0 and x.grad[4] < 0 and x.grad[3] == 0


def test_entropy_matches_bernoulli_definition():
    x = torch.tensor([-3., -.5, 0., .5, 3.], dtype=torch.float64)
    p = x.sigmoid()
    expected = -(p * p.log() + (1 - p) * (1 - p).log()).mean()
    torch.testing.assert_close(opacity_entropy(x), expected, atol=1e-14, rtol=0)
