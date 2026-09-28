"""CPU contracts for a recovery that never changes a failed execution claim."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location('visibility_recovery', Path(__file__).resolve().parents[1]/'scripts/recover_shared_visibility_intervention.py')
recovery = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(recovery)


def test_scalar_only_serialization_preserves_false_and_missing_evidence():
    value = {'status': 'recovered_measurements_only', 'original_status': 'failed',
             'runtime': None, 'passed': np.bool_(False), 'count': np.int64(264), 'loss': np.float64(.1)}
    restored = json.loads(json.dumps(value, allow_nan=False, default=recovery.numpy_scalar))
    assert restored == {'status': 'recovered_measurements_only', 'original_status': 'failed',
                        'runtime': None, 'passed': False, 'count': 264, 'loss': .1}


@pytest.mark.parametrize('bad', [object(), np.array([1., 2.])])
def test_unknown_objects_are_not_silently_stringified(bad):
    with pytest.raises(TypeError):
        json.dumps({'value': bad}, allow_nan=False, default=recovery.numpy_scalar)


def test_nan_is_still_rejected():
    with pytest.raises(ValueError):
        json.dumps({'loss': np.float32('nan')}, allow_nan=False, default=recovery.numpy_scalar)


def test_inventory_keeps_all_repeats_and_alpha_paths_without_scoring():
    original = {'output': '/never/read', 'views': [{'name': f'{i:03}.png'} for i in range(8)],
                'specification': {'order': ['base', 'minus', 'plus']*2}}
    arrays = recovery.expected_arrays(original)
    assert len(arrays) == 264
    for repeat in range(6):
        case = original['specification']['order'][repeat]
        assert arrays[f'/never/read/predictions/000.png.{repeat}.{case}.raw.npy'] == {'shape': [989, 1320, 5], 'dtype': 'float32'}
        assert arrays[f'/never/read/predictions/000.png.{repeat}.{case}.raw_ids.npy']['dtype'] == 'uint8'
    for key in ('mass', 'contribution_alpha', 'contribution_total'):
        assert f'/never/read/predictions/000.png.{key}.npy' in arrays
    original['views'] = original['views'][:7]
    with pytest.raises(ValueError, match='population'):
        recovery.expected_arrays(original)
