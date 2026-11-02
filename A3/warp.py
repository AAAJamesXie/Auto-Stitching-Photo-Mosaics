"""Handwritten inverse warping and nearest/bilinear sampling for A.3.

Only NumPy arithmetic and linear algebra are used. No library image warp,
resampler, grid sampler, or interpolation routine is called here.
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class WarpResult:
    image: np.ndarray
    alpha: np.ndarray
    origin: tuple[int, int]
    H_source_to_canvas: np.ndarray


def _matrix(H):
    H = np.asarray(H, dtype=np.float64)
    if H.shape != (3, 3) or not np.isfinite(H).all() or not np.any(H):
        raise ValueError('H must be a finite, nonzero 3x3 matrix.')
    H = H / np.max(np.abs(H))
    try:
        inverse = np.linalg.inv(H)
    except np.linalg.LinAlgError as exc:
        raise ValueError('H must be invertible.') from exc
    if not np.isfinite(inverse).all():
        raise ValueError('H has no finite numerical inverse.')
    return H, inverse


def output_canvas(image_shape, H):
    """Return (world_x0, world_y0, width, height), bounding four corner centers.

    A denominator sign change means a projective horizon crosses the source:
    finite corner values would then falsely suggest a finite image bounding box.
    """
    H, _ = _matrix(H)
    h, w = image_shape[:2]
    corners = np.array([[0., 0, 1], [w-1, 0, 1], [w-1, h-1, 1], [0, h-1, 1]])
    q = corners @ H.T
    d = q[:, 2]
    if d.min() <= 0 <= d.max():
        raise ValueError('Projective horizon crosses/touches the source; choose an explicit finite destination ROI.')
    p = q[:, :2] / d[:, None]
    if not np.isfinite(p).all() or np.max(np.abs(p)) > 1e12:
        raise ValueError('Transformed image has impractically large or nonfinite bounds.')
    rounded = np.rint(p)
    p = np.where(np.abs(p-rounded) < 1e-9, rounded, p)
    lo = np.floor(p.min(axis=0)).astype(np.int64)
    hi = np.ceil(p.max(axis=0)).astype(np.int64)
    return int(lo[0]), int(lo[1]), int(hi[0]-lo[0]+1), int(hi[1]-lo[1]+1)


def _warp(im, H, mode, canvas, block_rows, max_pixels):
    im = np.asarray(im)
    if im.ndim not in (2, 3) or (im.ndim == 3 and im.shape[2] not in (1, 3)):
        raise ValueError('Input must be a grayscale or one/three-channel image.')
    if min(im.shape[:2]) < 1 or not np.issubdtype(im.dtype, np.number) or np.iscomplexobj(im):
        raise ValueError('Input image must be nonempty and real-valued.')
    if not np.isfinite(im).all():
        raise ValueError('Input image pixels must be finite.')
    H, inverse = _matrix(H)
    if not isinstance(block_rows, (int, np.integer)) or block_rows < 1:
        raise ValueError('block_rows must be a positive integer.')
    if not isinstance(max_pixels, (int, np.integer)) or max_pixels < 1:
        raise ValueError('max_pixels must be a positive integer.')
    if canvas is None:
        canvas = output_canvas(im.shape, H)
    if len(canvas) != 4 or any(not np.isfinite(v) or int(v) != v for v in canvas):
        raise ValueError('Canvas must contain four finite integers: x0, y0, width, height.')
    x0, y0, width, height = map(int, canvas)
    if width < 1 or height < 1 or width*height > max_pixels:
        raise ValueError(f'Canvas must be positive and contain at most {max_pixels} pixels.')
    ih, iw = im.shape[:2]
    output = np.zeros((height, width)+im.shape[2:], dtype=np.float32)
    alpha = np.zeros((height, width), dtype=bool)
    xx = np.arange(width, dtype=np.float64)[None, :] + x0
    for row in range(0, height, block_rows):
        stop = min(row+block_rows, height)
        yy = np.arange(row, stop, dtype=np.float64)[:, None] + y0
        qx = inverse[0, 0]*xx + inverse[0, 1]*yy + inverse[0, 2]
        qy = inverse[1, 0]*xx + inverse[1, 1]*yy + inverse[1, 2]
        d = inverse[2, 0]*xx + inverse[2, 1]*yy + inverse[2, 2]
        scale = abs(inverse[2, 0]*xx) + abs(inverse[2, 1]*yy) + abs(inverse[2, 2])
        safe = np.abs(d) > 1e-12*np.maximum(scale, np.finfo(float).tiny)
        sx = np.divide(qx, d, out=np.full(d.shape, np.nan), where=safe)
        sy = np.divide(qy, d, out=np.full(d.shape, np.nan), where=safe)
        valid = safe & np.isfinite(sx) & np.isfinite(sy)
        valid &= (sx >= -1e-8) & (sx <= iw-1+1e-8) & (sy >= -1e-8) & (sy <= ih-1+1e-8)
        # Roundoff at exact source-border centers must not erase the last row.
        sx = np.clip(np.where(valid, sx, 0), 0, iw-1)
        sy = np.clip(np.where(valid, sy, 0), 0, ih-1)
        if mode == 'nearest':
            ix = np.floor(sx+.5).astype(np.int64)
            iy = np.floor(sy+.5).astype(np.int64)
            values = im[iy, ix].astype(np.float32)
        else:
            ix0 = np.floor(sx).astype(np.int64)
            iy0 = np.floor(sy).astype(np.int64)
            ix1 = np.minimum(ix0+1, iw-1)
            iy1 = np.minimum(iy0+1, ih-1)
            dx, dy = sx-ix0, sy-iy0
            if im.ndim == 3:
                dx, dy = dx[..., None], dy[..., None]
            values = ((1-dx)*(1-dy)*im[iy0, ix0] + dx*(1-dy)*im[iy0, ix1]
                      + (1-dx)*dy*im[iy1, ix0] + dx*dy*im[iy1, ix1])
        output[row:stop] = np.where(valid[..., None] if im.ndim == 3 else valid, values, 0)
        alpha[row:stop] = valid
    T = np.array([[1., 0, -x0], [0, 1, -y0], [0, 0, 1]])
    return WarpResult(output, alpha, (x0, y0), T @ H)


def warpImageNearestNeighbor(im, H, *, canvas=None, block_rows=64, max_pixels=40_000_000):
    """Inverse-map destination centers, then sample floor(source_coordinate+0.5).

    Returns WarpResult(image=float32, alpha=bool, origin=(world_x0,world_y0),
    H_source_to_canvas). Explicit canvas is (x0,y0,width,height), otherwise corner
    center bounds are used. Both methods share the continuous pixel-center domain.
    """
    return _warp(im, H, 'nearest', canvas, block_rows, max_pixels)


def warpImageBilinear(im, H, *, canvas=None, block_rows=64, max_pixels=40_000_000):
    """Inverse-map and weight four neighboring source pixels, including exact borders."""
    return _warp(im, H, 'bilinear', canvas, block_rows, max_pixels)
