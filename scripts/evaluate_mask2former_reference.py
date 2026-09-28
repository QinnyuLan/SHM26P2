"""Fixed Mask2Former-only rendered RGB semantics on the shared original50/41 grid."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from train_mask2former_reference import verify_hashes, verify_plan, write_json

from bridge_rgs.coordinates import pixel_protocol, protocol_metadata
from bridge_rgs.evaluate import distortion_render_grid
from bridge_rgs.losses import iou_scores
from bridge_rgs.mask2former_reference import digest
from bridge_rgs.mask2former_training import (
    INFERENCE_PROTOCOL,
    load_trained_reference,
    predict_tiled,
)
from bridge_rgs.official_evaluate import (
    CAMERA_KEYS,
    FAMILY,
    SCORING_PROTOCOL,
    _decode_rgb,
    _hash,
    _lpips,
    _utc,
    official_fingerprint,
    rasterize_official_annotation,
    read_official_reference,
    score_official_arrays,
    validate_training_lineage,
)
from bridge_rgs.train import load_scene


@torch.inference_mode()
def predict_camera(scene, model, camera, scene_state):
    """Only camera metadata enters; no target image, label, valid or prior mask."""
    if set(camera) != CAMERA_KEYS:
        raise ValueError('Official camera whitelist mismatch')
    profile = pixel_protocol(scene_state)
    K, width, height, source = distortion_render_grid(camera['K'], camera['distortion'],
                                                      camera['width'], camera['height'], profile)
    device = scene.splats['means'].device
    result = scene.render(torch.tensor(K, device=device), torch.tensor(camera['w2c'], device=device).float(),
                          width, height, degree=3, semantics=False, refine=False, absgrad=False)
    rgb = result['rgb'].clamp(0, 1).cpu().numpy()
    probabilities = predict_tiled(model, (rgb*255).round().astype(np.uint8))
    if source is not None:
        rgb = cv2.remap(rgb, source[..., 0], source[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        probabilities = cv2.remap(probabilities, source[..., 0], source[..., 1], cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT)
    if (probabilities.shape != (camera['height'], camera['width'], 5)
            or not np.isfinite(probabilities).all() or (probabilities < 0).any()
            or (probabilities.sum(-1) <= 0).any()):
        raise ValueError('Invalid original-grid semantic probability support')
    return ((rgb*255).round().astype(np.uint8), probabilities.argmax(-1).astype(np.uint8),
            {'pinhole_canvas': [width, height], 'checkpoint_pixel_protocol': profile})


def evaluate(plan_path, checkpoint, output):
    plan_path, checkpoint, output = (Path(x).resolve() for x in (plan_path, checkpoint, output))
    plan = json.loads(plan_path.read_text())
    verify_plan(plan)
    if Path(__file__).resolve() != Path(plan['source_snapshot'])/Path(__file__).name:
        raise ValueError('Run the frozen evaluation entrypoint')
    config = plan['configuration']
    training_receipt_path = checkpoint.parent/'execution_receipt.json'
    training = json.loads(training_receipt_path.read_text())
    if (training.get('status') != 'completed' or training.get('completed_step') != 6000
            or training.get('successful_updates') != 6000 or training.get('checkpoint') != str(checkpoint)
            or training.get('checkpoint_sha256') != digest(checkpoint)
            or training.get('plan_sha256') != digest(plan_path)):
        raise ValueError('Require completed fixed6000 training receipt and endpoint')
    reference = read_official_reference(config['manifest'], plan['root'])
    selected_path = Path(config['selected_reference_receipt'])
    selected = json.loads(selected_path.read_text())
    selected_predictions = {p['name']: p for p in selected.get('predictions', [])}
    if (selected.get('status') != 'completed' or selected.get('scoring_protocol') != SCORING_PROTOCOL
            or selected.get('checkpoint_sha256') != plan['input_hashes'][config['base_checkpoint']]
            or len(selected_predictions) != 50 or set(selected_predictions) != {c['name'] for c in reference['cameras']}):
        raise ValueError('Require completed same-scene selected original50/41 reference')
    output.mkdir(parents=True, exist_ok=False)
    (output/'rgb').mkdir()
    (output/'mask').mkdir()
    inputs = plan['input_hashes'] | {str(plan_path): digest(plan_path), str(checkpoint): digest(checkpoint),
                                   str(training_receipt_path): digest(training_receipt_path)}
    receipt = {'status': 'predicting', 'evaluation_family': FAMILY, 'started_utc': _utc(),
               'checkpoint': str(checkpoint), 'checkpoint_sha256': digest(checkpoint),
               'plan_sha256': digest(plan_path), 'source_hashes': plan['source_hashes'],
               'entrypoint_source': {'path': str(Path(__file__).resolve()), 'sha256': digest(__file__)},
               'input_files_sha256': inputs, 'scoring_protocol': SCORING_PROTOCOL,
               'inference_protocol': INFERENCE_PROTOCOL,
               'selected_reference_receipt': str(selected_path),
               'source_read_policy': 'All50 model predictions complete before any original VAL RGB/annotation payload is opened',
               'semantics_source': 'Mask2Former on this scene rendered RGB only; no scene semantic head, teacher or ensemble'}
    receipt_path = output/'execution_receipt.json'
    write_json(receipt_path, receipt)
    started = time.monotonic()
    try:
        torch.set_num_threads(8)
        cv2.setNumThreads(8)
        scene, scene_state = load_scene(config['base_checkpoint'])
        scene.eval().requires_grad_(False)
        lineage = validate_training_lineage(scene_state, reference)
        model, state = load_trained_reference(checkpoint)
        if state['provenance']['source_hashes'] != plan['source_hashes'] or state['training_config'] != config:
            raise ValueError('Reference model source/config differs from fixed plan')
        receipt.update(training_manifest=lineage, checkpoint_pixel_protocol=protocol_metadata(scene_state),
                       reference_manifest={'path': reference['path'], 'sha256': reference['sha256']})
        predictions = []
        for camera in reference['cameras']:
            rgb, mask, info = predict_camera(scene, model, camera, scene_state)
            stem = Path(camera['name']).stem
            rgb_path, mask_path = output/'rgb'/f'{stem}.png', output/'mask'/f'{stem}.png'
            if not cv2.imwrite(str(rgb_path), rgb[..., ::-1]) or not cv2.imwrite(str(mask_path), mask):
                raise OSError('Cannot write official predictions')
            expected = selected_predictions[camera['name']]
            # Verify both actual PNG byte streams, not only cached digest metadata.
            if digest(expected['rgb']) != expected['rgb_sha256'] or digest(rgb_path) != expected['rgb_sha256']:
                raise ValueError('Same-scene RGB PNG did not exactly reproduce selected reference')
            predictions.append({'name': camera['name'], 'rgb': str(rgb_path), 'mask': str(mask_path),
                                'rgb_sha256': digest(rgb_path), 'mask_sha256': digest(mask_path), **info})
        del model, scene
        torch.cuda.empty_cache()
        verify_hashes(inputs)
        receipt.update(status='scoring', predictions=predictions, predictions_finished_utc=_utc(),
                       prediction_seconds_including_png_and_rgb_audit=time.monotonic()-started,
                       all50_rgb_png_sha_equal_to_selected=True)
        write_json(receipt_path, receipt)
        perceptual = _lpips('cuda')
        records, fingerprints = [], []
        matrix = np.zeros((5, 5), dtype=np.int64)
        receipt['source_scoring_started_utc'] = _utc()
        for camera, target, prediction in zip(reference['cameras'], reference['targets'], predictions, strict=True):
            image_bytes = Path(target['source_image_path']).read_bytes()
            truth = _decode_rgb(image_bytes, camera['width'], camera['height'], 'source image')
            annotation_bytes = Path(target['source_annotation_path']).read_bytes() if target['source_annotation_path'] else None
            truth_mask = (rasterize_official_annotation(annotation_bytes, camera['width'], camera['height'])
                          if annotation_bytes is not None else None)
            rgb_bytes, mask_bytes = Path(prediction['rgb']).read_bytes(), Path(prediction['mask']).read_bytes()
            if _hash(rgb_bytes) != prediction['rgb_sha256'] or _hash(mask_bytes) != prediction['mask_sha256']:
                raise ValueError('Predictions changed before scoring')
            rgb = _decode_rgb(rgb_bytes, camera['width'], camera['height'], 'prediction')
            mask = cv2.imdecode(np.frombuffer(mask_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
            values = score_official_arrays(rgb, mask, truth, truth_mask, perceptual, device='cuda')
            if 'confusion_matrix' in values:
                matrix += np.asarray(values['confusion_matrix'], np.int64)
            records.append({'name': camera['name'], 'width': camera['width'], 'height': camera['height'], **values})
            fingerprints.append({'camera': camera, 'source_image_sha256': _hash(image_bytes),
                                 'source_annotation_sha256': _hash(annotation_bytes) if annotation_bytes is not None else None,
                                 'rasterized_mask_sha256': _hash(truth_mask.tobytes(order='C')) if truth_mask is not None else None})
        fingerprint = official_fingerprint(fingerprints)
        if (fingerprint != selected['official_evaluation_fingerprint'] or len(records) != 50
                or sum('confusion_matrix' in row for row in records) != 41):
            raise ValueError('Common official50/41 scoring fingerprint mismatch')
        metrics = {'evaluation_family': FAMILY, 'official_evaluation_fingerprint': fingerprint,
                   'scoring_protocol': SCORING_PROTOCOL, 'inference_protocol': INFERENCE_PROTOCOL,
                   'validation_views': 50, 'semantic_validation_views': 41,
                   **{k: float(np.mean([r[k] for r in records])) for k in ['psnr', 'ssim', 'lpips']},
                   'confusion_matrix': matrix.tolist(), **iou_scores(torch.tensor(matrix)), 'views': records}
        verify_hashes(inputs)
        verify_plan(plan)
        write_json(output/'official_metrics.json', metrics)
        receipt.update(status='completed', finished_utc=_utc(), elapsed_seconds=time.monotonic()-started,
                       official_evaluation_fingerprint=fingerprint,
                       official_metrics_sha256=digest(output/'official_metrics.json'),
                       source_records=[dict(target, **row) for target, row in zip(reference['targets'], fingerprints, strict=True)])
        write_json(receipt_path, receipt)
        return metrics
    except BaseException as error:
        receipt.update(status='failed', error=f'{type(error).__name__}: {error}')
        write_json(receipt_path, receipt)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    evaluate(args.plan, args.checkpoint, args.output)


if __name__ == '__main__':
    main()
