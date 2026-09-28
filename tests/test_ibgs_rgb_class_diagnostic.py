"""Population and prediction order contracts, entirely synthetic and CPU-only."""
import copy
import importlib.util
from pathlib import Path

import pytest

ENTRY = Path(__file__).resolve().parent/'diagnose_ibgs_layer_rgb_classes.py'
if not ENTRY.exists():
    ENTRY = Path(__file__).resolve().parents[1]/'scripts/diagnose_ibgs_layer_rgb_classes.py'
spec = importlib.util.spec_from_file_location('class_diagnostic', ENTRY)
worker = importlib.util.module_from_spec(spec); spec.loader.exec_module(worker)


def population():
    evaluation, annotations = [], []
    for i in range(50):
        row = {'camera': {'name': f'{i}.png'}, 'source_image_path': f'{i}.png', 'source_image_sha256': str(i)}
        annotation = {**copy.deepcopy(row), 'name': f'{i}.png',
            'source_annotation_path': f'{i}.json' if i < 41 else None,
            'source_annotation_sha256': str(i) if i < 41 else None,
            'rasterized_mask_sha256': str(i) if i < 41 else None}
        evaluation.append(row); annotations.append(annotation)
    return evaluation, annotations


def test_missing_annotation_never_invented_or_partially_bound():
    evaluation, annotations = population()
    selected, missing = worker.population(evaluation, annotations)
    assert len(selected) == 41 and missing == [f'{i}.png' for i in range(41, 50)]
    annotations[-1]['rasterized_mask_sha256'] = 'zero-background'
    with pytest.raises(ValueError, match='Partial'):
        worker.population(evaluation, annotations)


def test_camera_or_gt_hash_change_rejected():
    evaluation, annotations = population(); annotations[0]['source_image_sha256'] = 'other'
    with pytest.raises(ValueError, match='identity'):
        worker.population(evaluation, annotations)
    evaluation, annotations = population(); annotations[0]['camera']['new_focal'] = 1
    with pytest.raises(ValueError, match='identity'):
        worker.population(evaluation, annotations)


def test_all_arm_predictions_must_exist_once_before_gt():
    names = [f'{i}.png' for i in range(41)]
    records = [{'arm': a, 'name': n} for a in worker.ARMS for n in names]
    worker.prediction_barrier(records, names)
    for changed in (records[:-1], records[:-1]+[records[0]],
                    records[:-1]+[{'arm': 'selected_best', 'name': names[-1]}]):
        with pytest.raises(ValueError, match='164'):
            worker.prediction_barrier(changed, names)
