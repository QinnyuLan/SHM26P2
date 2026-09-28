"""CPU contracts for the thin AA wrapper; inherited training mathematics is not rewritten."""
import copy
import importlib.util
import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

HERE = Path(__file__).resolve().parent
WORKER = HERE/'train_ibgs_aa_warm_matched.py'
if not WORKER.exists():
    WORKER = HERE.parent/'scripts/train_ibgs_aa_warm_matched.py'
spec = importlib.util.spec_from_file_location('aa_warm_contract_worker', WORKER)
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)
OLD = worker.load_worker(worker.PARENT/'source_snapshot/train_ibgs_warm_matched.py')


def test_only_protocol_and_renderer_profile_change_recipe():
    parent = worker.read(worker.PARENT/'plan.json')['specification']
    original = copy.deepcopy(parent)
    revised = worker.revised_specification(parent)
    assert parent == original
    assert {k for k in revised if revised[k] != parent.get(k)} == {'protocol', 'renderer_profile'}
    assert revised['steps_per_arm'] == 6000
    assert (revised['geometry_steps'], revised['fusion_detached_steps'], revised['fusion_joint_steps']) == (1000, 1000, 4000)
    assert revised['internal_seconds_per_arm'] == 1800 and revised['external_seconds'] == 3900
    json.dumps(revised, allow_nan=False)


def fd_receipt():
    return {'numerical_status': 'passed', 'primary_unsaturated_passed': True,
            'positive_case': [{'finite_differences': [{'passed': True}]} for _ in range(4)],
            'zero_control': {'rho_zero': True, 'finite_zero_opacity_gradient': True, 'background_exact': True},
            'forward_calls': 33, 'hook_calls': 33, 'backward_calls': 3, 'optimizer_steps': 0,
            'data_reads': 0, 'hook_restored': True, 'numeric_flags_restored': True,
            'cap_case': [{'passed': False}]}


def test_primary_gate_and_zero_control_not_cap_counterexample():
    worker.validate_fd_receipt(fd_receipt(), OLD)
    for key in ('numerical_status', 'primary_unsaturated_passed', 'hook_restored'):
        broken = fd_receipt(); broken[key] = False
        with pytest.raises(RuntimeError):
            worker.validate_fd_receipt(broken, OLD)
    broken = fd_receipt(); broken['zero_control']['rho_zero'] = False
    with pytest.raises(RuntimeError):
        worker.validate_fd_receipt(broken, OLD)


def test_evidence_binds_same_aa_and_binary():
    plan = {'source_hashes': {'bridge_rgs/ibgs_antialias.py': 'module'}, 'binary': {'sha256': 'binary'}}
    worker.validate_aa_binding(plan, 'module', 'binary', OLD)
    for source, binary in (('different', 'binary'), ('module', 'wrong')):
        with pytest.raises(RuntimeError):
            worker.validate_aa_binding(plan, source, binary, OLD)


def test_natural_numerical_completion_both_required(tmp_path):
    OLD.write(tmp_path/'plan.json', {'specification': {}})
    execution = {'status': 'completed', 'numerical_status': 'passed', 'plan_sha256': OLD.sha(tmp_path/'plan.json')}
    OLD.write(tmp_path/'execution_receipt.json', execution)
    launch = {'status': 'completed', 'natural_completion': True, 'exit_code': 0,
              'plan_sha256': execution['plan_sha256'],
              'execution_receipt_sha256': OLD.sha(tmp_path/'execution_receipt.json')}
    OLD.write(tmp_path/'launch_receipt.json', launch)
    worker.completed_evidence(tmp_path, OLD, numerical=True)
    launch['natural_completion'] = False
    (tmp_path/'launch_receipt.json').write_text(json.dumps(launch))
    with pytest.raises(RuntimeError):
        worker.completed_evidence(tmp_path, OLD, numerical=True)


def test_backward_evidence_requires_real_two_view_gradients_and_restoration():
    receipt = {'numerical_status': 'passed',
               'counts': {'depth_renders': 8, 'target_renders': 2, 'backward': 2, 'optimizer_steps': 0},
               'antialias_calls': 10, 'field_parameters_unchanged': True,
               'network_parameters_unchanged': True, 'hook_restored': True, 'numerics_restored': True,
               'target_rows': [{'name': name, 'fusion_gradients_finite': True,
                                'gradient_summary': {str(i): {'finite': True} for i in range(8)}}
                               for name in ('002.png', '041.png')],
               'rgb_decodes': 10, 'valid_decodes': 1, 'source_depth_updates': 8}
    worker.validate_backward_receipt(receipt, OLD)
    for key in ('field_parameters_unchanged', 'numerics_restored'):
        broken = copy.deepcopy(receipt); broken[key] = False
        with pytest.raises(RuntimeError):
            worker.validate_backward_receipt(broken, OLD)
    broken = copy.deepcopy(receipt)
    broken['target_rows'][1]['gradient_summary']['2']['finite'] = False
    with pytest.raises(RuntimeError):
        worker.validate_backward_receipt(broken, OLD)


def test_exact_inheritance_preserves_bound_py_under_cache_directory(tmp_path):
    source, target = tmp_path/'source', tmp_path/'new'
    (source/'official/scene/__pycache__').mkdir(parents=True)
    (source/'official/scene/__pycache__/__init__.py').write_text('# bound source\n')
    (source/'irrelevant.pyc').write_bytes(b'not a bound source')
    parent = {'source_snapshot': str(source), 'source_hashes': OLD.tree_hashes(source)}
    worker.copy_inherited_sources(parent, target, OLD)
    assert OLD.tree_hashes(target) == parent['source_hashes']
    assert not (target/'irrelevant.pyc').exists()


@pytest.mark.parametrize('mode', ['complete', 'partial', 'bad_count'])
def test_arm_scope_covers_all_renders_and_restores_on_failure(tmp_path, monkeypatch, mode):
    class Raster:
        def forward(self):
            return None
    original = Raster.forward
    @contextmanager
    def fake_aa(cls, *, capture_last, near_plane):
        assert not capture_last and near_plane == .01
        state = SimpleNamespace(calls=0)
        cls.forward = lambda self: None
        try:
            yield state
        finally:
            cls.forward = original
    captured = {}
    def endpoint(old, h, state, path, **kwargs):
        captured['state'] = state
    monkeypatch.setattr(worker, 'save_endpoint', endpoint)
    def inherited(plan, plan_sha, arm, report):
        (tmp_path/arm).mkdir()
        fake.save_endpoint(tmp_path/arm/'last.pt')
        captured['state'].calls = 6350 if mode != 'bad_count' else 6349
        if mode == 'partial':
            raise RuntimeError('partial inherited pass')
        report['steps'] = 6000
    saved = lambda *args, **kwargs: None
    fake = SimpleNamespace(train_arm=inherited, save_endpoint=saved, require=OLD.require, write=OLD.write, sha=OLD.sha)
    plan = {'output': str(tmp_path), 'aa_module_sha256': 'module', 'binary': {'sha256': 'binary'}}
    report = {}
    if mode == 'complete':
        worker.train_arm(fake, None, plan, 'plan', 'full', report, Raster, fake_aa)
    else:
        with pytest.raises(RuntimeError):
            worker.train_arm(fake, None, plan, 'plan', 'full', report, Raster, fake_aa)
    assert Raster.forward is original and fake.save_endpoint is saved
    scope = worker.read(tmp_path/'full/aa_scope_receipt.json')
    assert scope['hook_restored'] and not scope['capture_last']
    assert scope['aa_calls'] == (6349 if mode == 'bad_count' else 6350)


def test_endpoint_is_explicitly_new_profile_and_preserves_raw_parameters(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, 'get_rng_state_all', list)
    field = SimpleNamespace(_xyz=torch.nn.Parameter(torch.tensor([[1., 2., 3.]])), spatial_lr_scale=2.)
    inner = torch.nn.Linear(2, 3)
    optimizer = torch.optim.Adam([field._xyz], lr=.01)
    head_opt = torch.optim.Adam(inner.parameters(), lr=.02)
    plan = {'specification': {'protocol': worker.PROTOCOL}, 'camera_order_sha256': 'order',
            'data_contract': 'contract', 'data_contract_sha256': 'contractsha',
            'coordinate_profile': 'colmap_corner_v2', 'aa_module_sha256': 'module', 'binary': {'sha256': 'binary'}}
    path = tmp_path/'last.pt'
    worker.save_endpoint(OLD, SimpleNamespace(FIELD_KEYS=('_xyz',)), SimpleNamespace(calls=6350), path,
        field=field, net=SimpleNamespace(inner=inner), field_opt=optimizer, head_opt=head_opt,
        background=torch.tensor([.2, .3, .4]), arm='full', step=6000, plan=plan,
        plan_sha='plan', neighbors={}, counts={}, status='completed')
    saved = torch.load(path, weights_only=False, map_location='cpu')
    assert saved['protocol'] != 'ibgs_warm_matched_v1'
    assert saved['renderer_profile'] == worker.PROFILE
    assert saved['aa_module_sha256'] == 'module' and saved['renderer_binary'] == {'sha256': 'binary'}
    assert saved['counts']['aa_opacity_calls'] == 6350
    assert torch.equal(saved['field']['_xyz'], field._xyz.detach())
