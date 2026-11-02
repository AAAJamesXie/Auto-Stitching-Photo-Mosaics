"""Reproduce bounded local refinement of visually selected architectural points.

This is an annotation aid: each source location and approximate target location
is explicitly selected in manual_seeds.json. It does not discover features,
search across an image, estimate a homography, or reject points by model error.
OpenCV is used only for reading grayscale pixels and local normalized correlation.
"""
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent


def preview_to_original(points, original_size, preview_size=(1008, 1344)):
    """Convert preview pixel centers to the corresponding original pixel centers."""
    return (np.asarray(points, dtype=float) + .5) * (np.asarray(original_size) / preview_size) - .5


def refine_scaled_target(source, target, source_point, target_seed, scale, radius=32, half_patch=15):
    """Refine a manually identified pair using a scaled local template, not a warp."""
    x, y = np.rint(source_point).astype(int)
    h = half_patch
    if x-h < 0 or y-h < 0 or x+h >= source.shape[1] or y+h >= source.shape[0]:
        raise ValueError('Source annotation patch is outside the preview.')
    size = max(3, int(round((2*h+1)*scale)) | 1)
    patch = Image.fromarray(np.asarray(source[y-h:y+h+1, x-h:x+h+1], dtype=np.float32))
    patch = np.asarray(patch.resize((size, size), Image.Resampling.BICUBIC))
    mid = size//2
    return refine_target(patch, target, [mid, mid], target_seed, radius=radius, half_patch=mid)


def refine_target(source, target, source_point, target_seed, radius=48, half_patch=25):
    """NCC-refine one target seed inside a fixed square; return xy, score, boundary."""
    sx, sy = np.rint(source_point).astype(int)
    tx, ty = np.rint(target_seed).astype(int)
    r, h = int(radius), int(half_patch)
    if r < 1 or h < 1:
        raise ValueError("Search radius and patch half width must be positive.")
    if sx-h < 0 or sy-h < 0 or sx+h >= source.shape[1] or sy+h >= source.shape[0]:
        raise ValueError("Source patch is outside image bounds.")
    # Restrict a search near a border to the available pixels; never pad pixels.
    x0, y0 = max(h, tx-r), max(h, ty-r)
    x1, y1 = min(target.shape[1]-h-1, tx+r), min(target.shape[0]-h-1, ty+r)
    if x1 < x0 or y1 < y0:
        raise ValueError("Target search region is outside image bounds.")
    patch = np.asarray(source[sy-h:sy+h+1, sx-h:sx+h+1], dtype=np.float32)
    if patch.std() < 1e-6:
        raise ValueError("Source patch has no texture; select a different architectural point.")
    region = np.asarray(target[y0-h:y1+h+1, x0-h:x1+h+1], dtype=np.float32)
    scores = cv2.matchTemplate(region, patch, cv2.TM_CCOEFF_NORMED)
    iy, ix = np.unravel_index(np.argmax(scores), scores.shape)
    boundary = ix in (0, scores.shape[1]-1) or iy in (0, scores.shape[0]-1)
    dx = dy = 0.
    if not boundary:
        for axis in (0, 1):
            minus = scores[iy-1, ix] if axis == 0 else scores[iy, ix-1]
            center = scores[iy, ix]
            plus = scores[iy+1, ix] if axis == 0 else scores[iy, ix+1]
            denominator = minus - 2 * center + plus
            offset = float(.5 * (minus-plus) / denominator) if abs(denominator) > 1e-8 else 0.
            if axis == 0:
                dy = float(np.clip(offset, -.5, .5))
            else:
                dx = float(np.clip(offset, -.5, .5))
    return [float(x0+ix+dx), float(y0+iy+dy)], float(scores[iy, ix]), bool(boundary)


def main():
    data = json.loads((ROOT/'manual_seeds.json').read_text(encoding='utf-8'))
    data['refinement'] = {'method': 'Bounded local grayscale NCC with manually seeded per-pair template scale; quadratic subpixel peak',
                          'preview_size': [1008, 1344], 'base_patch_size_px': 31, 'search_radius_preview_px': 32,
                          'scope': 'Refines the manually seeded target only; no feature detector, global matcher, RANSAC, or H-based filtering.'}
    for pair in data['pairs']:
        if pair.get('status') == 'insufficient_overlap':
            print(pair['id'], 'UNSOLVED:', pair['reason'])
            continue
        src = cv2.imread(str(ROOT.parent/'A1'/'previews'/pair['source']), cv2.IMREAD_GRAYSCALE)
        dst = cv2.imread(str(ROOT.parent/'A1'/'previews'/pair['target']), cv2.IMREAD_GRAYSCALE)
        if src is None or dst is None:
            raise FileNotFoundError(f"Missing A1 image for {pair['id']}")
        for point in pair['points']:
            xy, score, boundary = refine_scaled_target(src, dst, point['source_seed'], point['target_seed'],
                                                      pair['template_scale'], radius=pair.get('search_radius',32),
                                                      half_patch=pair.get('half_patch',15))
            point.update(source=preview_to_original(point['source_seed'], pair['source_size']).tolist(),
                         target=preview_to_original(xy, pair['target_size']).tolist(),
                         source_preview=point['source_seed'], target_preview=xy,
                         ncc=score, search_boundary=boundary)
        print(pair['id'], 'NCC min',round(min(p['ncc'] for p in pair['points']),3),
              'boundary IDs',[p['id'] for p in pair['points'] if p['search_boundary']],
              'NCC<0.75',[(p['id'],round(p['ncc'],3)) for p in pair['points'] if p['ncc']<.75])
    (ROOT/'correspondences.json').write_text(json.dumps(data,indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
