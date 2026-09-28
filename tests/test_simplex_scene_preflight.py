"""CPU contracts for the fixed direct-q scene preflight; no real scene or payload."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
WORKER = HERE/'preflight_simplex_scene.py'
if not WORKER.exists():
    WORKER = HERE.parent/'scripts/preflight_simplex_scene.py'
spec = importlib.util.spec_from_file_location('simplex_preflight', WORKER)
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)


def fixture():
    generator = torch.Generator().manual_seed(42)
    W = torch.rand(6, 7, 4, generator=generator)*.1
    q0 = torch.softmax(torch.randn(4, 5, generator=generator), -1)
    background = torch.zeros(6, 7, 5)
    background[..., 0] = 1-W.sum(-1)
    return W, background, q0


def explicit(W, background, q):
    return torch.einsum('hwn,nc->hwc', W, q)+background


def test_same_leaf_adds_both_columns_and_background_cancels():
    W, background, q0 = fixture()
    q1, qm, direction = m.simplex_points(q0)
    q = qm.requires_grad_()
    info = {'means2d': torch.zeros(1, 4, 2), 'conics': torch.zeros(1, 4, 3),
            'opacities': torch.ones(1, 4), 'isect_offsets': torch.zeros(1, 1, 1, dtype=torch.int32),
            'flatten_ids': torch.arange(4, dtype=torch.int32)}
    def fake(*args, **kwargs):
        qi, qo, coeff = args[-3:]
        assert qi is qo is q
        assert coeff.tolist() == [m.SPEC['coefficients']]*4
        assert kwargs == {'width': 7, 'height': 6}
        return explicit(W, background, .3*qi+.7*qo), torch.ones(6, 7, 1)
    raw, _ = m.direct_q(info, q, fake, width=7, height=6)
    probe = m.fixed_probe(6, 7, 0, 'cpu')
    gradient, = torch.autograd.grad((raw.double()*probe.double()).sum(), (q,))
    expected = torch.einsum('hwn,hwc->nc', W, probe)
    torch.testing.assert_close(gradient, expected, rtol=2e-6, atol=1e-9)
    lhs = m.dot64(gradient, direction)
    rhs = m.dot64(probe, explicit(W, background, q1).double()-explicit(W, background, q0).double())
    assert m.adjoint_result(lhs, rhs)['within_tolerance']
    torch.testing.assert_close(explicit(W, background, q1)-explicit(W, background, q0),
                               torch.einsum('hwn,nc->hwc', W, q1-q0), atol=1e-7, rtol=1e-5)


def test_three_fixed_probes_detect_one_percent_common_vjp_scaling():
    W, background, q0 = fixture()
    q1, qm, direction = m.simplex_points(q0)
    qm.requires_grad_()
    raw = explicit(W, background, qm)
    good, bad = [], []
    for index in range(3):
        probe = m.fixed_probe(6, 7, index, 'cpu')
        assert probe.dtype == torch.float32
        gradient, = torch.autograd.grad((raw.double()*probe.double()).sum(), (qm,), retain_graph=True)
        A = m.dot64(gradient, direction)
        B = m.dot64(probe, explicit(W, background, q1).double()-explicit(W, background, q0).double())
        good.append(m.adjoint_result(A, B))
        bad.append(m.adjoint_result(A*1.01, B))
    assert m.adjoint_gate(good)['passed']
    assert all(row['measurable'] for row in bad)
    assert not any(row['within_tolerance'] for row in bad)
    assert not m.adjoint_gate(bad)['passed']


def test_zero_signal_cannot_pass_using_absolute_tolerance():
    row = m.adjoint_result(0., 1e-9)
    assert row['within_tolerance']
    assert not m.adjoint_gate([row]*3)['passed']


def test_primary_fd_is_not_replaced_by_sensitivity():
    q = torch.full((2, 5), .2)
    grad = torch.ones_like(q)
    row = m.fd_result(.01, grad, q, q, 2., 2.)
    assert row['primary'] and not row['passed']
    assert m.SPEC['primary_epsilon'] == .01
    assert m.SPEC['sensitivity_epsilons'] == [.005, .0025]


def test_simplex_endpoints_actual_direction_and_affine_ce():
    W, background, q0 = fixture()
    q1, qm, direction = m.simplex_points(q0)
    for q in (q0, q1, qm):
        assert m.simplex_error(q)['minimum'] >= 0
        assert m.simplex_error(q)['row_sum_max_abs_error'] < m.SPEC['simplex_absolute']
    qm.requires_grad_()
    labels = m.synthetic_labels(6, 7, 'cpu')
    def ce(q):
        raw = explicit(W, background, q)
        p = raw.double().gather(-1, labels[..., None])[..., 0]
        return -((1-m.SPEC['noise_delta'])*p+m.SPEC['noise_delta']/5).log().mean()
    grad, = torch.autograd.grad(ce(qm), (qm,))
    for epsilon in [.01, .005, .0025]:
        plus, minus = m.central_points(qm.detach(), direction, epsilon)
        assert min(float(plus.min()), float(minus.min())) > 0
        row = m.fd_result(epsilon, grad, plus, minus, float(ce(plus)), float(ce(minus)))
        assert row['passed']
        actual = (plus.double()-minus.double())/(2*epsilon)
        assert row['analytic_actual_displacement'] == m.dot64(grad, actual)


def test_restore_flags_modes_gradients_after_exception():
    scene = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.Dropout())
    scene[0].weight.grad = torch.ones_like(scene[0].weight)
    captured = m.capture_scene(scene)
    try:
        scene.eval().requires_grad_(False)
        scene.zero_grad(set_to_none=True)
        raise RuntimeError('synthetic exception')
    except RuntimeError:
        restored = m.restore_scene(scene, captured)
    assert restored['state_exact']
    assert restored['flags_modes_gradients_restored']


def test_prior_requires_natural_completion_and_bound_receipt():
    plan = {'sources': {'a.py': 'x'}}
    receipt = {'status': 'passed', 'plan_sha256': 'p', 'inputs_and_sources_unchanged': True}
    launch = {'exit_code': 0, 'natural_completion': True, 'plan_sha256': 'p', 'execution_receipt_sha256': 'r'}
    m.validate_prior(plan, receipt, launch, 'p', 'r')
    for bad in ({**launch, 'exit_code': 1}, {**launch, 'natural_completion': False},
                {**launch, 'execution_receipt_sha256': 'wrong'}, {**launch, 'status': 'failed'}):
        with pytest.raises(ValueError):
            m.validate_prior(plan, receipt, bad, 'p', 'r')


def test_fixed_budget_and_no_data_payload_contract():
    assert m.SPEC['counts'] == {'scene': 2, 'gsplat': 4, 'shader': 18, 'total_raster': 22, 'vjp': 8}
    assert m.SPEC['names'] == ['002.png', '118.png']
    assert m.SPEC['label_reads'] == m.SPEC['image_reads'] == m.SPEC['optimizer_steps'] == 0
    assert m.SPEC['noise_delta'] == 5e-7
    before = m.numerical_flags()
    try:
        m.numerical_flags(m.SPEC['numerics'])
        assert m.numerical_flags() == m.SPEC['numerics']
    finally:
        m.numerical_flags(before)
    assert m.numerical_flags() == before
    assert not torch.cuda.is_initialized()


def test_attempt_marker_consumes_plan_before_any_gpu_work(tmp_path):
    m.start_attempt(tmp_path, 'p')
    original = (tmp_path/'execution_started.json').read_bytes()
    with pytest.raises(ValueError, match='no retry'):
        m.start_attempt(tmp_path, 'p')
    assert (tmp_path/'execution_started.json').read_bytes() == original
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('existing', ['execution_receipt.json', 'analysis.json'])
def test_existing_result_without_marker_is_not_overwritten(tmp_path, existing):
    (tmp_path/existing).write_text('old failure')
    with pytest.raises(ValueError, match='no retry'):
        m.start_attempt(tmp_path, 'p')
    assert (tmp_path/existing).read_text() == 'old failure'
    assert not (tmp_path/'execution_started.json').exists()


def test_fd_result_whole_report_is_strict_json_and_reproduces_old_failure():
    _, _, q0 = fixture()
    _, qm, direction = m.simplex_points(q0)
    plus, minus = m.central_points(qm, direction, .01)
    gradient = torch.arange(qm.numel(), dtype=torch.float32).reshape_as(qm)*1e3
    args = (.01, gradient, plus, minus, 1.100001, 1.099999)
    row = m.fd_result(*args)
    # This invokes the actual immutable v1 reporting function, not a mock np.bool.
    previous_path = Path('/mnt/data/SHM2026/runs/simplex_scene_preflight_v1/source_snapshot/preflight_simplex_scene.py')
    old_spec = importlib.util.spec_from_file_location('failed_simplex_preflight_v1', previous_path)
    old = importlib.util.module_from_spec(old_spec)
    old_spec.loader.exec_module(old)
    old_row = old.fd_result(*args)
    with pytest.raises(TypeError, match='JSON serializable'):
        json.dumps(old_row, allow_nan=False)
    assert row == old_row  # identical values; only Python scalar types differ
    assert type(row['measurable']) is bool and type(row['passed']) is bool
    assert type(row['fp64_reduction_bound']) is float
    assert json.loads(json.dumps({'ce_fd': [row], 'passed': row['passed']}, allow_nan=False))['ce_fd'][0] == row


def test_serialization_failure_does_not_create_partial_file(tmp_path):
    path = tmp_path/'analysis.json'
    with pytest.raises(TypeError):
        m.write(path, {'bad': object()})
    assert not path.exists()
    with pytest.raises(ValueError):
        m.write(path, {'bad': float('nan')})
    assert not path.exists()
    m.write(path, {'completed': True})
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        m.write(path, {'replacement': True})
    assert path.read_bytes() == original
