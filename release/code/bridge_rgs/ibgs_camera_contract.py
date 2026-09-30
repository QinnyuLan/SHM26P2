"""CPU coordinate contract for a corner-v2 to IBGS port, not a renderer.

IBGS uses integer pixel centers internally. Convert corner principal points to
array principal points for rays/texture projections, while its NDC projection
must retain corner principal points: ndc2Pix already subtracts half a pixel.
No image loading, CUDA, official repository imports, or camera mutation.
"""
from __future__ import annotations

import numpy as np


def _finite(value, shape, name):
    out = np.asarray(value, np.float64)
    if out.shape != shape or not np.isfinite(out).all():
        raise ValueError(f'{name}: finite shape {shape} required')
    return out


def _size(width, height):
    if any(isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or n < 2
           for n in (width, height)):
        raise ValueError('Image width/height must be integers >=2')
    return int(width), int(height)


def _K(K):
    K = _finite(K, (3, 3), 'K_corner')
    if (not np.array_equal(K[2], [0, 0, 1]) or K[0, 1] != 0 or K[1, 0] != 0
            or K[0, 0] <= 0 or K[1, 1] <= 0):
        raise ValueError('Positive-focal zero-skew pinhole K required')
    return K


def _pose(w2c):
    w2c = _finite(w2c, (4, 4), 'w2c')
    R = w2c[:3, :3]
    if (not np.array_equal(w2c[3], [0, 0, 0, 1])
            or not np.allclose(R@R.T, np.eye(3), rtol=0, atol=1e-5)
            or abs(np.linalg.det(R)-1) > 1e-5):
        raise ValueError('Homogeneous proper world-to-camera rigid transform required')
    return w2c


def array_intrinsics(K_corner):
    K = _K(K_corner).copy()
    K[:2, 2] -= .5
    return K


def make_ibgs_camera(K_corner, w2c, width, height, *, znear=.01, zfar=100.):
    """FP64 math values to set on an official Camera before CUDA conversion.

    Generic K is represented algebraically here. The current official CUDA uses
    target intrinsics for every source and hardcodes centered principal points;
    require_official_shared_centered must pass for the proposed minimal patch.
    R_argument is transpose(w2c rotation); world_view_transform is w2c.T for GLM.
    Near/far are explicit port choices, not copied from a different renderer.
    """
    width, height = _size(width, height)
    K, w2c = _K(K_corner), _pose(w2c)
    if not np.isfinite([znear, zfar]).all() or not 0 < znear < zfar:
        raise ValueError('Finite 0 < znear < zfar required')
    projection = np.zeros((4, 4), np.float64)
    projection[0, 0], projection[1, 1] = 2*K[0, 0]/width, 2*K[1, 1]/height
    projection[0, 2], projection[1, 2] = 2*K[0, 2]/width-1, 2*K[1, 2]/height-1
    projection[2, 2], projection[2, 3] = zfar/(zfar-znear), -zfar*znear/(zfar-znear)
    projection[3, 2] = 1
    array_K = array_intrinsics(K)
    return {'width': width, 'height': height, 'K_corner': K.copy(), 'K_array': array_K,
            'R_argument': w2c[:3, :3].T.copy(), 'T_argument': w2c[:3, 3].copy(),
            'world_view_transform': w2c.T.copy(), 'projection_matrix': projection.T.copy(),
            'full_proj_transform': w2c.T@projection.T,
            'camera_center': np.linalg.inv(w2c)[:3, 3],
            'Fx': float(K[0, 0]), 'Fy': float(K[1, 1]),
            'Cx': float(array_K[0, 2]), 'Cy': float(array_K[1, 2]),
            'FoVx': float(2*np.arctan(width/(2*K[0, 0]))),
            'FoVy': float(2*np.arctan(height/(2*K[1, 1]))),
            'znear': float(znear), 'zfar': float(zfar)}


def require_official_shared_centered(cameras):
    """Reject unsupported source K/dimensions instead of silently reusing target K."""
    if not cameras:
        raise ValueError('At least one camera required')
    first = cameras[0]
    for c in cameras:
        width, height = _size(c['width'], c['height'])
        K = _K(c['K_corner'])
        if not np.array_equal(K[:2, 2], [width/2, height/2]):
            raise ValueError('Minimal official CUDA patch requires centered corner K')
        if ((width, height) != (first['width'], first['height'])
                or not np.array_equal(K, first['K_corner'])):
            raise ValueError('Official source warp requires identical source/target K and size')
    return True


def ndc_to_array(ndc_xy, width, height):
    width, height = _size(width, height)
    xy = np.asarray(ndc_xy, np.float64)
    if xy.shape[-1:] != (2,) or not np.isfinite(xy).all():
        raise ValueError('Finite trailing-2 NDC coordinates required')
    return ((xy+1)*np.array([width, height])-1)/2


def project_world_ibgs(points, camera, *, homogeneous_epsilon=0.):
    """CPU replay of transposed GLM projection plus ndc2Pix.

    Default zero epsilon exposes the exact coordinate identity; the official
    preprocess adds 1e-7 to clip w. Tests report this separate finite-depth bias.
    """
    points = np.asarray(points, np.float64)
    if points.shape[-1:] != (3,) or not np.isfinite(points).all():
        raise ValueError('Finite world XYZ required')
    if not np.isfinite(homogeneous_epsilon) or homogeneous_epsilon < 0:
        raise ValueError('Finite nonnegative homogeneous epsilon required')
    homogeneous = np.concatenate([points, np.ones(points.shape[:-1]+(1,))], -1)
    clip = homogeneous@camera['full_proj_transform']
    if np.any(clip[..., 3] <= 0):
        raise ValueError('Projection contract requires positive camera z')
    ndc = clip[..., :2]/(clip[..., 3, None]+homogeneous_epsilon)
    return ndc_to_array(ndc, camera['width'], camera['height'])


def pixel_rays(K_corner, width, height):
    """z=1 camera rays, identical to integer pixels using the converted K_array."""
    width, height = _size(width, height)
    K = _K(K_corner)
    y, x = np.indices((height, width), dtype=np.float64)
    return np.stack([(x+.5-K[0, 2])/K[0, 0], (y+.5-K[1, 2])/K[1, 1], np.ones_like(x)], -1)


def reference_to_source(reference_w2c, source_w2c):
    """Column-vector transform, stored row-major by the IBGS warp kernel."""
    return _pose(source_w2c)@np.linalg.inv(_pose(reference_w2c))


def array_to_grid(array_xy, width, height, *, align_corners):
    width, height = _size(width, height)
    xy = np.asarray(array_xy, np.float64)
    if xy.shape[-1:] != (2,) or not np.isfinite(xy).all() or not isinstance(align_corners, bool):
        raise ValueError('Finite xy and explicit bool align_corners required')
    return 2*xy/np.array([width-1, height-1])-1 if align_corners else 2*(xy+.5)/np.array([width, height])-1


def texture_coordinates(array_xy):
    xy = np.asarray(array_xy, np.float64)
    if xy.shape[-1:] != (2,) or not np.isfinite(xy).all():
        raise ValueError('Finite array xy required')
    return xy+.5


def resize_corner_intrinsics(K_corner, old_size, new_size):
    old_w, old_h = _size(*old_size)
    new_w, new_h = _size(*new_size)
    K = _K(K_corner).copy()
    K[0] *= new_w/old_w
    K[1] *= new_h/old_h
    return K


def strided_array_intrinsics(K_corner, step, start=0):
    """For original array indices start+step*j; this is NOT image resizing."""
    if (isinstance(step, bool) or not isinstance(step, (int, np.integer)) or step < 1
            or isinstance(start, bool) or not isinstance(start, (int, np.integer)) or not 0 <= start < step):
        raise ValueError('Integer step>=1 and 0<=start<step required')
    K = array_intrinsics(K_corner)
    K[0, 0] /= step
    K[1, 1] /= step
    K[:2, 2] = (K[:2, 2]-start)/step
    return K
