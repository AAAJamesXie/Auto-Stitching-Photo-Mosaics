"""A.2: normalized, eight-parameter least-squares homography recovery.

No library homography, perspective transform, image warp, or RANSAC calls.
Coordinates are (x, y) pixel centers, with (0, 0) at the top left.
"""
import numpy as np


def _point_pairs(src, dst):
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.ndim != 2 or src.shape[1] != 2 or src.shape != dst.shape:
        raise ValueError("Correspondences must be two arrays of matching shape (n, 2).")
    if len(src) < 4:
        raise ValueError("At least four point pairs are required.")
    if not np.isfinite(src).all() or not np.isfinite(dst).all():
        raise ValueError("Point coordinates must be finite.")
    return src, dst


def normalize_points(points):
    """Center points and scale their mean distance to sqrt(2); return points, T."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError("Expected a finite array of shape (n, 2).")
    if len(points) < 4:
        raise ValueError("At least four points are required.")
    center = points.mean(axis=0)
    centered = points - center
    if np.linalg.matrix_rank(centered) < 2:
        raise ValueError("Points must span two dimensions; repeated/collinear points are degenerate.")
    mean_distance = np.linalg.norm(centered, axis=1).mean()
    scale = np.sqrt(2.0) / mean_distance
    T = np.array([[scale, 0, -scale * center[0]],
                  [0, scale, -scale * center[1]], [0, 0, 1.]])
    return centered * scale, T


def build_system(im1_pts, im2_pts):
    """Return A, b in Ah=b, fixing h33=1; x and y equations are interleaved."""
    src, dst = _point_pairs(im1_pts, im2_pts)
    x, y = src.T
    u, v = dst.T
    A = np.zeros((2 * len(src), 8), dtype=np.float64)
    A[0::2, :3] = np.column_stack((x, y, np.ones(len(src))))
    A[1::2, 3:6] = np.column_stack((x, y, np.ones(len(src))))
    A[0::2, 6:] = -np.column_stack((u * x, u * y))
    A[1::2, 6:] = -np.column_stack((v * x, v * y))
    return A, dst.ravel().copy()


def computeH(im1_pts, im2_pts):
    """Recover H mapping im1_pts into im2_pts by normalized linear least squares.

    All supplied correspondences participate equally. There is no outlier rejection.
    The objective is algebraic least squares, not nonlinear geometric refinement.
    """
    src, dst = _point_pairs(im1_pts, im2_pts)
    src_n, T1 = normalize_points(src)
    dst_n, T2 = normalize_points(dst)
    A, b = build_system(src_n, dst_n)
    h, _, rank, _ = np.linalg.lstsq(A, b, rcond=None)
    if rank < 8:
        raise ValueError("The correspondence system is rank deficient; choose better-spread points.")
    Hn = np.append(h, 1.).reshape(3, 3)
    if np.linalg.matrix_rank(Hn) < 3:
        raise ValueError("Correspondences produce a singular homography.")
    H = np.linalg.solve(T2, Hn @ T1)
    # The h33=1 representation does not cover a homography with h33=0.
    if abs(H[2, 2]) <= np.finfo(float).eps * np.linalg.norm(H):
        raise ValueError("Recovered homography cannot be represented with h33=1.")
    H /= H[2, 2]
    if not np.isfinite(H).all():
        raise ValueError("Recovered homography has nonfinite entries.")
    return H


def project_points(points, H):
    """Apply H to coordinates only; this function does not resample an image."""
    points = np.asarray(points, dtype=np.float64)
    H = np.asarray(H, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or H.shape != (3, 3):
        raise ValueError("Expected points of shape (n, 2) and H of shape (3, 3).")
    if not np.isfinite(points).all() or not np.isfinite(H).all():
        raise ValueError("Points and H must be finite.")
    homogeneous = np.column_stack((points, np.ones(len(points)))) @ H.T
    if np.any(np.abs(homogeneous[:, 2]) <= np.finfo(float).eps * np.maximum(1., np.linalg.norm(homogeneous, axis=1))):
        raise ValueError("A projected point lies at or too close to infinity.")
    return homogeneous[:, :2] / homogeneous[:, 2:3]
