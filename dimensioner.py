"""Turn a 2D detection + aligned depth into real-world dimensions (millimeters).

The detector says *where* an object is in the color image (a pixel box); the depth
frame says *how far* every pixel is. Combined with the camera intrinsics, that's
enough to measure the physical part — which is the whole point of the welding trial
(a cobot needs a tube's length and diameter, not its pixel box).

Pipeline for one detection:
  1. crop aligned depth to the box
  2. isolate the object: keep the nearest coherent depth band, drop the far
     background wall and flyer pixels (the part is the closest thing in its box)
  3. deproject those pixels to a 3D point cloud (mm) via the intrinsics
  4. PCA on the cloud → principal axis = length, second axis = width, plus distance

Caveat we're honest about: the camera sees only the *facing* surface, so `length`
and `width` (the two in-plane dimensions) are trustworthy, while true thickness
(the depth dimension) is under-observed — reported as `thickness_seen` and marked
approximate. A segmentation mask (next layer) tightens this by removing the last
background pixels the box includes.
"""

from __future__ import annotations

import numpy as np

# The part is assumed to be the nearest object in its box; keep depths within this
# span of the nearest surface and treat anything beyond as background.
MAX_DEPTH_SPAN_MM = 500.0
# Too few 3D points to trust a measurement (object too far, too small, or no depth).
MIN_POINTS = 50
# Robust extent percentiles — reject the 2% tails so a few flyers don't inflate size.
_LO, _HI = 2.0, 98.0


def measure_detection(
    detection,
    depth_mm: np.ndarray,
    intrinsics,
    *,
    max_depth_span_mm: float = MAX_DEPTH_SPAN_MM,
    min_points: int = MIN_POINTS,
) -> dict | None:
    """Measure one Detection against the aligned depth frame. None if not measurable.

    Returns {length, width, thickness_seen, distance, points} in mm (counts for
    `points`). Also stored on `detection.dimensions_mm`.
    """
    h, w = depth_mm.shape
    x1, y1, x2, y2 = detection.box
    x1, x2 = max(0, min(x1, x2)), min(w, max(x1, x2))
    y1, y2 = max(0, min(y1, y2)), min(h, max(y1, y2))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None

    roi = depth_mm[y1:y2, x1:x2]
    valid = roi > 0
    if valid.sum() < min_points:
        return None

    depths = roi[valid]
    # Nearest surface (robust to a few too-close flyers), then keep the band in
    # front of the background.
    near = np.percentile(depths, _LO)
    keep = valid & (depth_mm[y1:y2, x1:x2] <= near + max_depth_span_mm) & (roi > 0)
    if keep.sum() < min_points:
        return None

    ys, xs = np.nonzero(keep)
    us = xs + x1  # back to full-image pixel coords
    vs = ys + y1
    zs = roi[keep]

    X, Y, Z = intrinsics.deproject(us, vs, zs)
    points = np.stack([np.asarray(X), np.asarray(Y), np.asarray(Z)], axis=1)

    dims = _pca_extents(points)
    distance = float(np.median(zs))

    result = {
        "length": round(dims[0], 1),
        "width": round(dims[1], 1),
        "thickness_seen": round(dims[2], 1),
        "distance": round(distance, 1),
        "points": int(points.shape[0]),
        "approximate": True,  # facing-surface only; refine with a mask later
    }
    detection.dimensions_mm = result
    return result


def _pca_extents(points: np.ndarray) -> tuple[float, float, float]:
    """Robust extents (mm) along the point cloud's 3 principal axes, largest first."""
    centroid = points.mean(axis=0)
    centered = points - centroid
    # Right singular vectors are the principal axes, ordered by variance.
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    extents = []
    for axis in vt:
        proj = centered @ axis
        extents.append(float(np.percentile(proj, _HI) - np.percentile(proj, _LO)))
    return tuple(extents)  # already largest-first (length, width, thickness)


def measure_all(detections, depth_mm, intrinsics, **kwargs) -> list:
    """Measure every detection in place (sets .dimensions_mm) and return the list."""
    if depth_mm is None or intrinsics is None:
        return detections
    for d in detections:
        measure_detection(d, depth_mm, intrinsics, **kwargs)
    return detections
