"""CPU observations for the fixed saved TRAIN tracks; no model or prediction input."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt

CLASS_NAMES = ('background', 'deck', 'stay_cable', 'tower', 'foundation')
SPEC = {
    'protocol': 'saved_train_track_semantics_v1',
    'points': 60000, 'observations': 457102, 'train_views': 350, 'labeled_train_views': 259,
    'observation_population': 'Only init_points saved point/image pairs, in original CSR order',
    'duplicate_policy': 'Last original COLMAP keypoint for each saved image/track pair',
    'primary': 'Original distorted official polygon mask; floor COLMAP corner coordinates',
    'secondary': 'Legacy prepared mask/valid; cv2 undistort normalized then prepared K then numpy rint',
    'boundary': 'Own-class pixel-center EDT to other/ignore/invalid or one-pixel false exterior pad',
    'ray': 'Unit camera-to-observed-keypoint direction in world coordinates',
    'reprojection': 'Saved FP32 XYZ cast FP64 and original pose versus observed undistorted original-native UV',
    'error_bins': '[0,1], (1,2], (2,infinity); -1 nonfinite',
    'boundary_bins': '[0,3], (3,10], (10,infinity); -1 unavailable',
    'primary_interior': 'reprojection_error <= 1 and primary_boundary_distance > 10',
    'labels': '-1 no annotation; 255 ignore, out of bounds or invalid prepared support; 0..4 known class',
    'inference': 'No GPU, model, predictions, RGB payloads or VAL annotation/mask payloads',
}


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    text = json.dumps(value, indent=2, allow_nan=False)+'\n'
    with Path(path).open('x') as stream:
        stream.write(text)


def observation_layout(arrays, views):
    """Fail closed on held-out IDs, duplicate point/image pairs, or a changed CSR."""
    require(all(v['split'] == 'train' for v in views), 'TRAIN views only')
    ids = np.asarray(arrays['observation_image_ids'])
    offsets = np.asarray(arrays['observation_offsets'])
    tracks = np.asarray(arrays['track_ids'])
    points = np.asarray(arrays['points'])
    counts = np.asarray(arrays['num_observations'])
    require(ids.dtype.kind in 'iu' and offsets.dtype.kind in 'iu' and tracks.dtype.kind in 'iu', 'Integer identifiers required')
    require(points.shape == (len(tracks), 3) and np.isfinite(points).all()
            and len(np.unique(tracks)) == len(tracks) and np.all(tracks >= 0), 'Invalid points/track IDs')
    require(offsets.shape == (len(tracks)+1,) and offsets[0] == 0 and offsets[-1] == len(ids)
            and np.all(np.diff(offsets) > 0) and np.array_equal(np.diff(offsets), counts), 'Invalid observation CSR')
    view_ids = [v['image_id'] for v in views]
    require(len(set(view_ids)) == len(views) and np.isin(ids, view_ids).all(), 'Held-out or missing image ID')
    point_index = np.repeat(np.arange(len(tracks), dtype=np.int32), counts)
    # Stored observations are sorted and have one last-selected keypoint per image.
    same_point = point_index[1:] == point_index[:-1]
    require(np.all(ids[1:][same_point] > ids[:-1][same_point]), 'Repeated or unsorted per-track image ID')
    return point_index, {int(i): np.flatnonzero(ids == i) for i in view_ids}


def rotation_from_quaternion(q):
    q = np.asarray(q, np.float64)
    require(q.shape == (4,) and np.isfinite(q).all() and np.linalg.norm(q) > 1e-12, 'Invalid pose quaternion')
    w, x, y, z = q/np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                     [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])


def validate_source_cameras(path, source_cameras):
    """The present fixed dataset is SIMPLE_RADIAL; do not silently coerce models."""
    recovered = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip() or line.lstrip().startswith('#'): continue
        fields = line.split()
        require(len(fields) == 8 and fields[1] == 'SIMPLE_RADIAL', 'Expected source SIMPLE_RADIAL camera')
        cid = fields[0]
        require(cid not in recovered and cid in source_cameras, 'Camera identity mismatch')
        c = source_cameras[cid]; p = np.asarray(fields[4:], np.float64)
        K = np.array([[p[0], 0., p[1]], [0., p[0], p[2]], [0., 0., 1.]])
        require(c['model'] == fields[1] and (c['width'], c['height']) == (int(fields[2]), int(fields[3]))
                and np.array_equal(p, c['params']) and np.array_equal(K, c['K'])
                and np.array_equal([p[3], 0., 0., 0., 0.], c['opencv_distortion']), 'Source camera calibration differs')
        recovered[cid] = c
    require(set(recovered) == set(source_cameras), 'Incomplete source cameras')


def recover_observations(path, arrays, train_views):
    """Recover actual saved keypoints, never parse held-out observation values."""
    point_index, by_view = observation_layout(arrays, train_views)
    views = {v['image_id']: v for v in train_views}
    tracks = np.asarray(arrays['track_ids'])
    xy = np.full((len(point_index), 2), np.nan, np.float64)
    seen = set(); duplicates = 0; skipped = 0
    with Path(path).open() as stream:
        while line := stream.readline():
            if not line.strip() or line.lstrip().startswith('#'):
                continue
            fields = line.strip().split(maxsplit=9)
            require(len(fields) == 10, 'Malformed COLMAP image header')
            image_id = int(fields[0]); observations = stream.readline()
            require(observations != '', 'Missing COLMAP observation line')
            if image_id not in views:
                skipped += 1
                continue
            require(image_id not in seen, 'Duplicate COLMAP image header')
            seen.add(image_id); view = views[image_id]
            require(fields[9] == view['name'] and int(fields[8]) == view['camera_id'], 'COLMAP/manifest identity differs')
            pose = np.eye(4)
            pose[:3, :3] = rotation_from_quaternion([float(x) for x in fields[1:5]])
            pose[:3, 3] = [float(x) for x in fields[5:8]]
            require(np.allclose(pose, view['w2c_original'], rtol=0, atol=1e-12), 'Original pose differs from COLMAP')
            values = np.fromstring(observations, dtype=np.float64, sep=' ')
            require(len(values) % 3 == 0, 'Malformed TRAIN observations')
            values = values.reshape(-1, 3)
            lookup = {int(tracks[point_index[row]]): int(row) for row in by_view[image_id]}
            last = {}
            for j, track in enumerate(values[:, 2]):
                require(np.isfinite(track) and track == int(track) and abs(track) <= 2**53, 'Invalid track identifier')
                track = int(track)
                if track in lookup:
                    duplicates += int(track in last)
                    last[track] = j
            require(set(last) == set(lookup), f'Saved observed keypoint missing for TRAIN {image_id}')
            for track, j in last.items():
                xy[lookup[track]] = values[j, :2]
    require(seen == set(views) and np.isfinite(xy).all(), 'Incomplete TRAIN observation recovery')
    return xy, point_index, by_view, {
        'saved_observations_recovered': len(xy), 'selected_pair_duplicate_keypoints': duplicates,
        'duplicate_policy': SPEC['duplicate_policy'], 'nontrain_observation_lines_not_parsed': skipped,
    }


def geometry_for_view(raw_xy, points, view, source_camera):
    """No raster pixels used; preserve both observed ray and point-direction meanings."""
    raw_xy = np.asarray(raw_xy, np.float64); points = np.asarray(points, np.float64)
    K = np.asarray(source_camera['K'], np.float64)
    normalized = cv2.undistortPoints(raw_xy[:, None], K,
                                   np.asarray(source_camera['opencv_distortion'], np.float64))[:, 0]
    original_uv = normalized*K.diagonal()[:2]+K[:2, 2]
    prepared_K = np.asarray(view['K'], np.float64)
    legacy_uv = normalized*prepared_K.diagonal()[:2]+prepared_K[:2, 2]
    pose = np.asarray(view['w2c_original'], np.float64)
    camera_points = points@pose[:3, :3].T+pose[:3, 3]
    depth = camera_points[:, 2]
    require(np.isfinite(depth).all() and np.all(depth > 0), 'Saved observation has nonpositive depth; stop without silent exclusion')
    projected = camera_points[:, :2]/depth[:, None]*K.diagonal()[:2]+K[:2, 2]
    error = np.linalg.norm(projected-original_uv, axis=1)
    ray = np.column_stack([normalized, np.ones(len(normalized))])@pose[:3, :3]
    ray /= np.linalg.norm(ray, axis=1, keepdims=True)
    center = -pose[:3, :3].T@pose[:3, 3]
    direction = points-center
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    return {'undistorted_native_xy': original_uv, 'legacy_xy': legacy_uv,
            'reprojection_error': error, 'positive_depth': depth > 0,
            'viewing_ray_world': ray, 'point_direction_world': direction, 'camera_depth': depth}


def pixel_indices(xy, width, height, mode):
    xy = np.asarray(xy, np.float64)
    require(xy.ndim == 2 and xy.shape[1] == 2 and np.isfinite(xy).all(), 'Finite XY required')
    require(mode in ('floor', 'rint'), 'Unknown sampling convention')
    rounded = np.floor(xy) if mode == 'floor' else np.rint(xy)
    require(np.abs(rounded).max(initial=0) < np.iinfo(np.int64).max, 'Pixel index overflow')
    indices = rounded.astype(np.int64)
    inside = ((indices[:, 0] >= 0) & (indices[:, 0] < width)
              & (indices[:, 1] >= 0) & (indices[:, 1] < height))
    return indices, inside


def rasterize_annotation(content, width, height):
    """Same official LabelMe polygon draw order, with unknown labels as ignore 255."""
    data = json.loads(content)
    require((data['imageWidth'], data['imageHeight']) == (width, height), 'Annotation dimensions differ')
    canvas = Image.new('L', (width, height), 0); draw = ImageDraw.Draw(canvas)
    for shape in data['shapes']:
        require(shape.get('shape_type', 'polygon') == 'polygon' and len(shape['points']) >= 3, 'Unsupported annotation shape')
        label = shape['label'].strip()
        draw.polygon([tuple(p) for p in shape['points']], fill=CLASS_NAMES.index(label) if label in CLASS_NAMES else 255)
    return np.asarray(canvas, dtype=np.uint8)


def sample_labels_and_distance(mask, support, indices, inside):
    """Class distance is to other/ignore/invalid pixels AND exterior, not polygons."""
    mask, support = np.asarray(mask), np.asarray(support, bool)
    require(mask.ndim == 2 and support.shape == mask.shape and np.isin(mask, [0, 1, 2, 3, 4, 255]).all(), 'Invalid semantic/support grid')
    sampled = np.full(len(indices), 255, np.int16)
    support_sample = np.zeros(len(indices), bool)
    sampled[inside] = mask[indices[inside, 1], indices[inside, 0]]
    support_sample[inside] = support[indices[inside, 1], indices[inside, 0]]
    valid = inside & support_sample & (sampled < 5)
    labels = np.where(valid, sampled, 255).astype(np.int16)
    distance = np.full(len(indices), np.nan, np.float64)
    for label in np.unique(labels[valid]):
        region = (mask == label) & support
        edt = distance_transform_edt(np.pad(region, 1, constant_values=False))[1:-1, 1:-1]
        selected = valid & (labels == label)
        distance[selected] = edt[indices[selected, 1], indices[selected, 0]]
    return {'label': labels, 'sampled_label': sampled, 'valid': valid,
            'support_sample': support_sample, 'boundary_distance': distance}


def strata(values, thresholds):
    values = np.asarray(values)
    result = np.full(values.shape, -1, np.int8); valid = np.isfinite(values) & (values >= 0)
    result[valid] = np.searchsorted(np.asarray(thresholds), values[valid], side='left')
    return result


def verify(plan):
    require(plan['collector_specification'] == SPEC, 'Collector contract differs')
    snapshot = Path(plan['source_snapshot']).resolve()
    require(Path(__file__).resolve() == snapshot/Path(__file__).name, 'Use frozen collector only')
    actual = {str(p.relative_to(snapshot)): sha(p) for p in snapshot.rglob('*')
              if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}
    require(actual == plan['source_hashes'], 'Frozen source tree differs')
    require(all(str(Path(plan[key]).resolve()) in plan['input_hashes'] for key in
                ('manifest', 'init_points', 'colmap_images', 'colmap_cameras')), 'Unbound metadata input')
    manifest = read(plan['manifest'])
    forbidden = {str(Path(v[k]).resolve()) for v in manifest['views']
                 for k in ('image_path', 'source_image_path') if v.get(k)}
    forbidden |= {str(Path(v[k]).resolve()) for v in manifest['views'] if v['split'] != 'train'
                  for k in ('mask_path', 'source_annotation_path') if v.get(k)}
    require(not (forbidden & {str(Path(p).resolve()) for p in plan['input_hashes']}), 'RGB/VAL pixel source included in collector inputs')
    for path, expected in plan['input_hashes'].items():
        require(sha(path) == expected, f'Bound input changed: {path}')
    return manifest


def collect(plan):
    """Execute only an explicitly frozen plan; root owns prepare and statistical analysis."""
    output = Path(plan['output']); receipt_path = output/'collection_receipt.json'
    require(not receipt_path.exists() and not (output/'observations.npz').exists(), 'Never overwrite or retry collection')
    manifest = verify(plan)
    train = sorted([v for v in manifest['views'] if v['split'] == 'train'], key=lambda v: v['name'])
    require(len(train) == SPEC['train_views'] and sum(bool(v.get('mask_path')) for v in train) == SPEC['labeled_train_views'], 'TRAIN population differs')
    require(manifest['class_names'] == list(CLASS_NAMES), 'Class order differs')
    require(manifest.get('pixel_protocol', 'legacy_mixed_v1') == 'legacy_mixed_v1', 'Not the legacy prepared dataset')
    require(all(v['w2c'] == v['w2c_original'] for v in train), 'Prepared cameras were changed')
    validate_source_cameras(plan['colmap_cameras'], manifest['source_cameras'])
    keys = ('points', 'track_ids', 'observation_image_ids', 'observation_offsets', 'num_observations', 'reprojection_error')
    with np.load(plan['init_points'], allow_pickle=False) as f:
        arrays = {k: f[k] for k in keys}
    require(arrays['points'].dtype == np.float32 and len(arrays['points']) == SPEC['points']
            and len(arrays['observation_image_ids']) == SPEC['observations'], 'Saved cloud population/dtype differs')
    report = {'status': 'running', 'started_utc': datetime.now(UTC).isoformat(),
              'plan_sha256': sha(output/'plan.json'), 'collector_specification': SPEC,
              'source_hashes': plan['source_hashes'], 'input_hashes': plan['input_hashes'],
              'raw_annotation_payload_reads': 0, 'legacy_mask_payload_reads': 0,
              'legacy_valid_payload_reads': 0, 'rgb_payload_reads': 0, 'val_pixel_payload_reads': 0,
              'gpu_calls': 0, 'statistical_conflict_analysis_performed': False}
    started = time.perf_counter(); failure = None
    try:
        raw_xy, point_index, by_view, recovery = recover_observations(plan['colmap_images'], arrays, train)
        size = len(point_index)
        result = {'point_index': point_index, 'track_id': arrays['track_ids'][point_index],
                  'image_id': arrays['observation_image_ids'], 'observation_offsets': arrays['observation_offsets'],
                  'point_track_ids': arrays['track_ids'], 'point_positions': arrays['points'],
                  'saved_track_rms': arrays['reprojection_error'], 'raw_xy': raw_xy}
        for key, shape in (('undistorted_native_xy', (size, 2)), ('legacy_xy', (size, 2)),
                           ('viewing_ray_world', (size, 3)), ('point_direction_world', (size, 3)),
                           ('reprojection_error', (size,)), ('camera_depth', (size,))):
            result[key] = np.full(shape, np.nan, np.float64)
        for key in ('raw_index', 'legacy_index'):
            result[key] = np.zeros((size, 2), np.int64)
        for key in ('has_label', 'raw_inside', 'legacy_inside', 'positive_depth',
                    'primary_valid', 'legacy_valid', 'legacy_support_sample'):
            result[key] = np.zeros(size, bool)
        for prefix in ('primary', 'legacy'):
            result[prefix+'_label'] = np.full(size, -1, np.int16)
            result[prefix+'_sampled_label'] = np.full(size, -1, np.int16)
            result[prefix+'_boundary_distance'] = np.full(size, np.nan, np.float64)
        for view in train:
            rows = by_view[view['image_id']]
            require(len(rows) > 0, 'TRAIN camera has no saved observations')
            camera = manifest['source_cameras'][str(view['camera_id'])]
            geometry = geometry_for_view(raw_xy[rows], arrays['points'][point_index[rows]].astype(np.float64), view, camera)
            for key, value in geometry.items(): result[key][rows] = value
            ri, inside = pixel_indices(raw_xy[rows], camera['width'], camera['height'], 'floor')
            li, legacy_inside = pixel_indices(geometry['legacy_xy'], view['width'], view['height'], 'rint')
            result['raw_index'][rows], result['raw_inside'][rows] = ri, inside
            result['legacy_index'][rows], result['legacy_inside'][rows] = li, legacy_inside
            labeled = bool(view.get('source_annotation_path'))
            require(labeled == bool(view.get('mask_path')), 'Annotation/prepared label availability differs')
            result['has_label'][rows] = labeled
            if not labeled: continue
            paths = [view['source_annotation_path'], view['mask_path'], view['valid_path']]
            require(all(str(Path(p).resolve()) in plan['input_hashes'] for p in paths), 'Unbound TRAIN pixel path')
            primary = rasterize_annotation(Path(paths[0]).read_bytes(), camera['width'], camera['height'])
            report['raw_annotation_payload_reads'] += 1
            with Image.open(paths[1]) as f: legacy = np.asarray(f).copy()
            with Image.open(paths[2]) as f: valid = np.asarray(f) > 0
            report['legacy_mask_payload_reads'] += 1; report['legacy_valid_payload_reads'] += 1
            require(legacy.shape == valid.shape == (view['height'], view['width']), 'Prepared grid shape differs')
            for prefix, sampling in (
                ('primary', sample_labels_and_distance(primary, np.ones_like(primary, bool), ri, inside)),
                ('legacy', sample_labels_and_distance(legacy, valid, li, legacy_inside)),
            ):
                for key in ('label', 'sampled_label', 'valid', 'boundary_distance'):
                    result[prefix+'_'+key][rows] = sampling[key]
                if prefix == 'legacy': result['legacy_support_sample'][rows] = sampling['support_sample']
        result['reprojection_bin'] = strata(result['reprojection_error'], [1., 2.])
        for prefix in ('primary', 'legacy'):
            result[prefix+'_boundary_bin'] = strata(result[prefix+'_boundary_distance'], [3., 10.])
        require(all(report[k] == SPEC['labeled_train_views'] for k in
                    ('raw_annotation_payload_reads', 'legacy_mask_payload_reads', 'legacy_valid_payload_reads')), 'Wrong label read population')
        with (output/'observations.npz').open('xb') as f: np.savez_compressed(f, **result)
        report.update(status='completed', observation_recovery=recovery,
                      arrays={k: {'shape': list(v.shape), 'dtype': str(v.dtype)} for k,v in result.items()},
                      observations_path=str(output/'observations.npz'), observations_sha256=sha(output/'observations.npz'),
                      train_images=[{'image_id': v['image_id'], 'name': v['name']} for v in train],
                      finished_utc=datetime.now(UTC).isoformat())
    except BaseException as error:  # noqa: BLE001 - preserve failure evidence, then re-raise
        report.update(status='failed', error=repr(error)); failure = error
    finally:
        try:
            verify(plan); report['bound_sources_inputs_unchanged'] = True
        except BaseException as error:  # noqa: BLE001 - retain both calculation and provenance failures
            report.update(status='failed', provenance_error=repr(error), bound_sources_inputs_unchanged=False)
            failure = failure or error
        report['elapsed_seconds'] = time.perf_counter()-started
        write(receipt_path, report)
    if failure is not None: raise failure
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', type=Path, required=True)
    args = parser.parse_args()
    report = collect(read(args.execute))
    print(json.dumps({'status': report['status'], 'observations_sha256': report['observations_sha256']}))


if __name__ == '__main__':
    main()
