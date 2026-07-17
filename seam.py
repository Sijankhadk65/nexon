"""Seam detection (the joint between two parts) within an area of interest (AOI).

The seam is found with a PROFILE SCAN inside the AOI: along each scan line (across the
AOI's long axis) we pick the single pixel most likely to be the seam crossing, then
robustly fit a line to those points. Two complementary signals feed the scan:

  * DEPTH (primary when available): a plane is fit over the AOI's valid depth, and the
    pixel that dips FARTHEST BELOW that surface on each scan line is the groove. This is
    what "visible depth" buys you and is what the real bevelled/grooved joints key on.
  * RGB LOCAL CONTRAST (fallback / reinforcement): the pixel whose brightness deviates
    most from the local surface on each scan line. This is POLARITY-AGNOSTIC — it catches
    a bright ground bevel (shiny cut metal on dark plate) just as well as the dark hairline
    of a flush butt joint. The old "darkest pixel" rule only handled the latter and missed
    the common case where the groove is the BRIGHTEST thing in the AOI.

Candidates from both signals are merged and a robust line fit rejects the disagreeing set
(scratches, glare, bolt holes) — so a tight AOI plus either a depth step or a visible
contrast line is enough. The depth plane also gives each endpoint a clean on-surface
height (Z). A perfectly invisible joint (no depth step AND no contrast line) can't be
seen by any camera — you'd need teaching / touch.

Draw the AOI so its long side runs ALONG the seam and it hugs the joint tightly.

Set the AOI once with the live preview (needs the camera):
    uv run python seam.py
    drag a box over the joint (long side along the seam) -> 's' saves it -> 'q' quits.
detect_seam / follow_seam then load that AOI automatically.

Save a specific seam for the main program: in the same preview, once the green seam line
looks right, press 'w' to write the CURRENTLY DETECTED seam (its endpoints deprojected to
camera-frame XYZ) to seam.json AND the current AOI box to seam_aoi.json in one step. The
main program's follow_saved_seam tool loads the seam and traces it WITHOUT re-detecting
(handy for a repeatable capture, as long as the camera hasn't moved), while live detection
(follow_seam / detect_seam) picks up the same AOI. 's' saves only the AOI; 'w' saves both,
so the main program always loads the box you just drew rather than a stale one.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import cv2 as cv
import numpy as np

AOI_FILE = Path(__file__).resolve().parent / "seam_aoi.json"
SEAM_FILE = Path(__file__).resolve().parent / "seam.json"

CONTRAST_K = 3.0       # RGB: keep a scan line's peak only if its |deviation| from the
                       # local surface exceeds median + CONTRAST_K*MAD (polarity-agnostic)
DEPTH_MIN_MM = 1.5     # depth: a groove pixel must sit at least this far below the fitted
                       # surface (also gated against the AOI's own depth noise)
DEPTH_K = 2.0          # depth: groove threshold is max(DEPTH_MIN_MM, median + DEPTH_K*MAD)
INLIER_PX = 3.0        # line-fit inlier band (px) floor for the robust refit
MAX_RESID_FRAC = 0.35  # accept only if midline points hug the fitted line within this*win (px)
MIN_SPAN_FRAC = 0.4    # seam must span at least this fraction of the AOI's long side
MIN_PLANE_PTS = 200    # valid depth px in the AOI needed to fit the height plane
# --- fillet / corner / T-joint mode: seam = intersection line of two non-coplanar planes ---
PLANE_THRESH_MM = 3.0  # RANSAC 3D-plane inlier band (mm); ~ the depth sensor's plane noise
PLANE_MIN_FRAC = 0.2   # each of the two planes must claim at least this fraction of AOI points
DIHEDRAL_MIN_DEG = 20.0  # the two planes must meet at least this steeply (else it's ~one plane)
SEAM_BAND_MM = 6.0     # points within this of the intersection line define the seam's extent
WINDOW = "seam_aoi"


def load_aoi(path=AOI_FILE):
    """Return the saved AOI (x1, y1, x2, y2), or None if none set."""
    p = Path(path)
    if not p.exists():
        return None
    return tuple(json.loads(p.read_text())["aoi"])


def save_aoi(aoi, path=AOI_FILE):
    Path(path).write_text(json.dumps({"aoi": [int(v) for v in aoi]}))


def _sample_depth_mm(depth_mm, u, v, patch=5):
    """Median of the non-zero depths in an NxN patch around pixel (u, v), or 0 if all holes."""
    h, w = depth_mm.shape
    u, v = int(round(u)), int(round(v))
    r = patch // 2
    win = depth_mm[max(0, v - r):v + r + 1, max(0, u - r):u + r + 1]
    valid = win[win > 0]
    return float(np.median(valid)) if valid.size else 0.0


def seam_record(s, depth_mm, intrinsics):
    """Build a persistable seam record from a find_seam result + this frame's depth/intrinsics.

    Deprojects each endpoint to camera-frame XYZ (mm) — Z from the fitted AOI depth plane when
    available, else a local depth-patch median — the SAME representation vision.locate_seam
    hands the main program, so the saved seam maps cleanly through the camera->base extrinsic
    later. Returns the record dict, or None if the seam is missing or an endpoint has no usable
    depth (a hole) and so can't be turned into 3D.
    """
    if s is None or intrinsics is None or depth_mm is None:
        return None
    rec = {"p1_px": [round(s["p1"][0], 1), round(s["p1"][1], 1)],
           "p2_px": [round(s["p2"][0], 1), round(s["p2"][1], 1)],
           "length_px": round(s["length_px"], 1),
           "box": [int(v) for v in s["box"]]}
    # The fillet path already has true 3D endpoints (from the plane intersection); use them
    # directly. The groove path takes Z from the fitted AOI plane, else a local patch median.
    fillet_xyz = s.get("p1_cam_xyz_mm") is not None and s.get("p2_cam_xyz_mm") is not None
    cams = []
    for name, (u, v) in (("p1", s["p1"]), ("p2", s["p2"])):
        if fillet_xyz:
            x, y, zc = s[f"{name}_cam_xyz_mm"]
        else:
            z = plane_z(s["plane"], u, v) if s["plane"] is not None else _sample_depth_mm(depth_mm, u, v)
            if z <= 0:
                return None
            x, y, zc = intrinsics.deproject(u, v, z)
        rec[f"{name}_cam_xyz_mm"] = [float(x), float(y), float(zc)]
        cams.append(np.array([x, y, zc], float))
    rec["length_mm"] = round(float(np.linalg.norm(cams[1] - cams[0])), 1)
    rec["depth_z"] = s.get("mode", "plane" if s["plane"] is not None else "patch")
    return rec


def save_seam(record, path=SEAM_FILE):
    """Write a seam record (from seam_record) to disk for the main program to load.

    Stamps `saved_at` so the loader/operator can tell how fresh the capture is — the camera
    must not have moved since, as the endpoints are in the camera frame. Returns the path.
    """
    record = dict(record)
    record["saved_at"] = datetime.now().isoformat(timespec="seconds")
    Path(path).write_text(json.dumps(record, indent=2))
    return path


def load_seam(path=SEAM_FILE):
    """Return the saved seam record (dict), or None if none has been saved."""
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text())


def _fit_plane(us, vs, zs):
    """Least-squares depth plane z = a*u + b*v + c (u, v in FULL-FRAME px). Returns (a,b,c)."""
    A = np.column_stack([us, vs, np.ones_like(us)])
    coef, *_ = np.linalg.lstsq(A, zs, rcond=None)
    return float(coef[0]), float(coef[1]), float(coef[2])


def plane_z(plane, u, v):
    """Surface height (mm) predicted by the fitted plane at full-frame pixel (u, v)."""
    a, b, c = plane
    return a * u + b * v + c


def fit_plane_in_aoi(depth_mm, aoi):
    """Fit the surface height plane over valid depth in the AOI, or None if too sparse."""
    h, w = depth_mm.shape
    x1, y1, x2, y2 = _clamp(aoi, w, h)
    crop = depth_mm[y1:y2, x1:x2]
    vs, us = np.where(crop > 0)
    if us.size < MIN_PLANE_PTS:
        return None
    zs = crop[vs, us].astype(np.float32)
    return _fit_plane((us + x1).astype(np.float32), (vs + y1).astype(np.float32), zs)


def _clamp(aoi, w, h):
    x1, y1, x2, y2 = aoi
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))
    return x1, y1, x2, y2


def _robust_thr(vals, k, floor=0.0):
    """median + k*(1.4826*MAD) of `vals`, but never below `floor`. Empty -> inf."""
    if vals.size == 0:
        return np.inf
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med))) * 1.4826
    return max(floor, med + k * mad)


def _win(aoi):
    """Half-width (px) of the per-scan-line band window: a fraction of the AOI's SHORT side."""
    x1, y1, x2, y2 = aoi
    return max(8.0, 0.2 * min(x2 - x1, y2 - y1))


def _normalize(w):
    """Scale a non-negative weight map to peak 1 (0 stays 0). All-zero -> unchanged."""
    m = float(w.max()) if w.size else 0.0
    return w / m if m > 0 else w


def _rgb_weight(gray, horizontal, k=CONTRAST_K):
    """AOI-local map of RGB seam-likelihood: |brightness - local surface|, thresholded.

    Polarity-agnostic — a bright ground bevel and a dark hairline both score high. The
    baseline is the per-scan-line median (the surface), so only the joint stands out.
    """
    g = cv.GaussianBlur(gray, (5, 5), 0).astype(np.float32)
    axis = 0 if horizontal else 1
    dev = np.abs(g - np.median(g, axis=axis, keepdims=True))
    return np.where(dev > _robust_thr(dev.ravel(), k), dev, 0.0)


def _depth_weight(depth_mm, aoi, plane, min_mm=DEPTH_MIN_MM, k=DEPTH_K):
    """AOI-local map of depth seam-likelihood: how far each pixel sits BELOW the surface.

    Uses the fitted AOI plane as the reference surface; groove pixels are farther from the
    camera than it. The threshold's noise floor comes from the WHOLE surface (not the groove
    itself). Returns a zeros-shaped-like map when there's no plane/depth.
    """
    if plane is None or depth_mm is None:
        return None
    h, w = depth_mm.shape
    x1, y1, x2, y2 = _clamp(aoi, w, h)
    crop = depth_mm[y1:y2, x1:x2].astype(np.float32)
    a, b, c = plane
    ys, xs = np.mgrid[y1:y2, x1:x2].astype(np.float32)  # full-frame coords for the plane
    resid = crop - (a * xs + b * ys + c)                # >0 == farther than surface = groove
    valid = crop > 0                                    # depth holes can't be candidates
    thr = _robust_thr(resid[valid], k, floor=min_mm)
    return np.where(valid & (resid > thr), resid, 0.0)


def _peak_points(weight, horizontal):
    """Per-scan-line strongest pixel of a weight map -> (pts Nx2 in AOI-local u,v, ok mask).

    One rough candidate per scan line; scan lines with no weight are marked not-ok. These
    seed the RANSAC line (they're noisy across a wide band, but RANSAC only needs enough of
    them to agree on direction).
    """
    axis = 0 if horizontal else 1
    peak = np.argmax(weight, axis=axis)
    line = np.arange(weight.shape[1 - axis])
    ok = weight.max(axis=axis) > 0
    if horizontal:                      # scan columns: line=u (col), peak=v (row)
        return np.column_stack([line, peak]).astype(np.float32), ok
    return np.column_stack([peak, line]).astype(np.float32), ok  # scan rows: peak=u, line=v


def _ransac_line(pts, band, iters=200, seed=0):
    """Robust 2-point RANSAC line through `pts` -> (vx, vy, x0, y0) refit on inliers, or None.

    Tolerates the heavy outliers per-line peaks carry (glare, twin walls, scratches) far
    better than a single Huber fit; the winning line is the one the most points sit within
    `band` px of. Refined by L2 on that inlier set.
    """
    n = pts.shape[0]
    if n < 2:
        return None
    rng = np.random.default_rng(seed)
    best_line, best_count = None, -1
    for _ in range(iters):
        i, j = rng.integers(0, n, 2)
        d = pts[j] - pts[i]
        norm = np.hypot(*d)
        if norm < 1e-6:
            continue
        vx, vy = d / norm
        dist = np.abs((pts - pts[i]) @ np.array([-vy, vx]))
        count = int((dist < band).sum())
        if count > best_count:
            best_count, best_line = count, (vx, vy, pts[i])
    if best_line is None:
        return None
    vx, vy, p0 = best_line
    inl = pts[np.abs((pts - p0) @ np.array([-vy, vx])) < band]
    vx, vy, x0, y0 = cv.fitLine(inl, cv.DIST_L2, 0, 0.01, 0.01).flatten()
    return float(vx), float(vy), float(x0), float(y0)


def _centroid_near_line(weight, line, horizontal, win):
    """Per-scan-line weighted centroid of `weight`, within `win` px of `line`. AOI-local.

    `line` (vx, vy, x0, y0, in AOI-local coords) is the RANSAC rough seam. For each scan
    line we predict where the seam crosses and average only the weight within `win` of that
    prediction — so a wide bevel or twin walls average to the joint midline, while a distant
    reflection in the same scan line is excluded. Returns (us, vs) of sub-pixel centres.
    """
    vx, vy, x0, y0 = line
    axis = 0 if horizontal else 1
    H, W = weight.shape
    idx = np.arange(H if horizontal else W, dtype=np.float32)  # position along the scan
    idx = idx[:, None] if horizontal else idx[None, :]
    if horizontal:                       # scan columns; predict row v at each column u
        u = np.arange(W, dtype=np.float32)
        pred = y0 + (u - x0) * (vy / vx if abs(vx) > 1e-6 else 0.0)
        pred = pred[None, :]
    else:                                # scan rows; predict col u at each row v
        v = np.arange(H, dtype=np.float32)
        pred = x0 + (v - y0) * (vx / vy if abs(vy) > 1e-6 else 0.0)
        pred = pred[:, None]
    w = np.where(np.abs(idx - pred) <= win, weight, 0.0)
    wsum = w.sum(axis=axis)
    csum = (w * idx).sum(axis=axis)
    keep = wsum > 0
    centre = csum[keep] / wsum[keep]
    along = np.arange(weight.shape[1 - axis])[keep].astype(np.float32)
    return (along, centre) if horizontal else (centre, along)


def find_seam(bgr, depth_mm, aoi):
    """Find the seam LINE within `aoi` from depth + RGB contrast; endpoint Z from depth.

    Pipeline: build a per-pixel seam-likelihood map from depth (groove below the surface)
    and RGB (local contrast), combine them, seed a robust RANSAC line through the per-scan-
    line peaks (survives glare / twin bevel walls / scratches), then refine to the band's
    weighted midline and fit the final line. The depth plane also gives each endpoint its Z.

    Returns a dict or None:
      {"p1": (u, v), "p2": (u, v), "length_px": float, "box": (x1,y1,x2,y2),
       "plane": (a,b,c) or None, "resid_px": float, "n_depth": int, "n_rgb": int}
    p1/p2 are full-frame pixels; plane (if depth allowed a fit) gives a clean on-surface
    height at each endpoint via plane_z(). n_depth/n_rgb report how many scan lines each
    signal lit up, so you can see whether depth or contrast is carrying the detection.
    """
    if aoi is None:
        return None
    h, w = bgr.shape[:2]
    x1, y1, x2, y2 = _clamp(aoi, w, h)
    if x2 - x1 < 5 or y2 - y1 < 5:
        return None

    gray = cv.cvtColor(bgr[y1:y2, x1:x2], cv.COLOR_BGR2GRAY)
    horizontal = (x2 - x1) >= (y2 - y1)
    span = (x2 - x1) if horizontal else (y2 - y1)
    win = _win((x1, y1, x2, y2))

    # Fit the surface plane once: it references the depth groove scan AND gives endpoint Z.
    plane = fit_plane_in_aoi(depth_mm, aoi) if depth_mm is not None else None

    # Combined seam-likelihood map (AOI-local): normalised depth-groove + RGB-contrast.
    wr = _rgb_weight(gray, horizontal)
    wd = _depth_weight(depth_mm, aoi, plane)
    weight = _normalize(wr) + (_normalize(wd) if wd is not None else 0.0)
    n_rgb = int((wr.max(axis=(0 if horizontal else 1)) > 0).sum())
    n_depth = int((wd.max(axis=(0 if horizontal else 1)) > 0).sum()) if wd is not None else 0

    # Stage 1: rough line via RANSAC on per-scan-line peaks (robust to glare / twin walls).
    seeds, ok = _peak_points(weight, horizontal)
    seeds = seeds[ok]
    if seeds.shape[0] < MIN_SPAN_FRAC * span:
        return None
    rough = _ransac_line(seeds, band=max(2.0, 0.5 * win))
    if rough is None:
        return None

    # Stage 2: refine to the band's weighted midline within `win` of the rough line.
    cu, cv_ = _centroid_near_line(weight, rough, horizontal, win)
    if cu.size < MIN_SPAN_FRAC * span:
        return None
    pts = np.column_stack([cu + x1, cv_ + y1]).astype(np.float32)  # full-frame

    # Fit the seam line to the midline centroids, drop gross outliers, refit.
    vx, vy, x0, y0 = cv.fitLine(pts, cv.DIST_L2, 0, 0.01, 0.01).flatten()
    normal = np.array([-vy, vx])
    resid = np.abs((pts - np.array([x0, y0])) @ normal)
    band = max(INLIER_PX, 3.0 * float(np.median(resid)))  # a real seam curves a little in perspective
    pts = pts[resid < band]
    if pts.shape[0] < MIN_SPAN_FRAC * span:
        return None
    vx, vy, x0, y0 = cv.fitLine(pts, cv.DIST_L2, 0, 0.01, 0.01).flatten()
    normal = np.array([-vy, vx])
    resid = np.abs((pts - np.array([x0, y0])) @ normal)

    # Accept only a COHERENT, thin locus (rejects scatter) that SPANS the AOI (rejects a blob).
    if float(np.median(resid)) > MAX_RESID_FRAC * win:
        return None
    proj = (pts - np.array([x0, y0])) @ np.array([vx, vy])
    if float(proj.max() - proj.min()) < MIN_SPAN_FRAC * span:
        return None
    p1 = pts[int(np.argmin(proj))]
    p2 = pts[int(np.argmax(proj))]
    return {
        "p1": (float(p1[0]), float(p1[1])),
        "p2": (float(p2[0]), float(p2[1])),
        "length_px": float(np.linalg.norm(p2 - p1)),
        "box": (x1, y1, x2, y2),
        "plane": plane,
        "resid_px": float(resid.mean()),
        "n_depth": n_depth,
        "n_rgb": n_rgb,
        "mode": "groove",
    }


# --------------------------------------------------------------------------------------
# Fillet / corner / T-joint mode: the seam is the INTERSECTION LINE of two non-coplanar
# planes (not a groove in one surface). Needs true 3D, so this path takes the intrinsics
# and works on the deprojected point cloud. Sibling to find_seam; find_seam_auto dispatches.
# --------------------------------------------------------------------------------------

def _project(intr, X):
    """Camera-frame points X (N,3 mm) -> pixel (u, v) arrays. Inverse of intr.deproject."""
    x, y, z = X[:, 0], X[:, 1], X[:, 2]
    return x * intr.fx / z + intr.cx, y * intr.fy / z + intr.cy


def _deproject_aoi(depth_mm, aoi, intr):
    """Valid (depth>0) pixels in the AOI -> (P Nx3 camera-frame mm, us, vs full-frame px)."""
    h, w = depth_mm.shape
    x1, y1, x2, y2 = _clamp(aoi, w, h)
    crop = depth_mm[y1:y2, x1:x2]
    vv, uu = np.where(crop > 0)
    if uu.size == 0:
        return None
    us = (uu + x1).astype(np.float32)
    vs = (vv + y1).astype(np.float32)
    z = crop[vv, uu].astype(np.float32)
    x, y, z = intr.deproject(us, vs, z)
    return np.column_stack([x, y, z]).astype(np.float32), us, vs


def _fit_plane_svd(P):
    """Least-squares plane through points P (N,3) -> (unit normal n, offset d) with n·X = d."""
    c = P.mean(axis=0)
    _, _, vt = np.linalg.svd(P - c, full_matrices=False)
    n = vt[-1]                      # direction of least variance = plane normal
    return n, float(n @ c)


def _ransac_plane(P, thresh, iters=200, seed=0):
    """Robust 3-point RANSAC plane through P (N,3) -> (n, d, inlier_mask), or None.

    The winning plane has the most points within `thresh` mm; refined by SVD on its inliers.
    """
    n = P.shape[0]
    if n < 3:
        return None
    rng = np.random.default_rng(seed)
    best_mask, best_count = None, -1
    for _ in range(iters):
        i, j, k = rng.integers(0, n, 3)
        nrm = np.cross(P[j] - P[i], P[k] - P[i])
        norm = np.linalg.norm(nrm)
        if norm < 1e-6:
            continue
        nrm = nrm / norm
        mask = np.abs((P - P[i]) @ nrm) < thresh
        c = int(mask.sum())
        if c > best_count:
            best_count, best_mask = c, mask
    if best_mask is None or best_count < 3:
        return None
    nvec, d = _fit_plane_svd(P[best_mask])
    mask = np.abs(P @ nvec - d) < thresh          # final inliers under the refined plane
    return nvec, d, mask


def _plane_intersection(n1, d1, n2, d2):
    """Intersection of planes n1·X=d1, n2·X=d2 -> (unit dir, point p0), or None if ~parallel."""
    direction = np.cross(n1, n2)
    norm = np.linalg.norm(direction)
    if norm < 1e-6:                               # parallel planes: no unique line
        return None
    direction = direction / norm
    # p0 = the point on the line nearest the origin: solve [n1; n2; dir] x = [d1, d2, 0].
    A = np.array([n1, n2, direction])
    p0 = np.linalg.solve(A, np.array([d1, d2, 0.0]))
    return direction, p0


def find_fillet_seam(depth_mm, aoi, intrinsics):
    """Seam as the intersection line of TWO non-coplanar planes in the AOI (fillet/T/corner).

    Deprojects the AOI depth to a 3D cloud, RANSAC-fits a first plane, fits a second to the
    remainder, and — if the two meet steeply enough (>= DIHEDRAL_MIN_DEG) — returns their
    intersection line, clipped to the extent of the points that actually lie on the joint.
    Returns a dict shaped like find_seam (p1/p2 in full-frame px, length_px, box, resid_px,
    plane=None) plus 3D fields (p1_cam_xyz_mm, p2_cam_xyz_mm, length_mm, dihedral_deg) and
    mode="fillet". Returns None when there aren't two clear, steeply-meeting planes (e.g. a
    flat surface — let the groove path handle that).
    """
    if depth_mm is None or intrinsics is None or aoi is None:
        return None
    h, w = depth_mm.shape
    x1, y1, x2, y2 = _clamp(aoi, w, h)
    if x2 - x1 < 5 or y2 - y1 < 5:
        return None
    dep = _deproject_aoi(depth_mm, aoi, intrinsics)
    if dep is None or dep[0].shape[0] < 2 * MIN_PLANE_PTS:
        return None
    P, _, _ = dep
    n_total = P.shape[0]

    fit1 = _ransac_plane(P, PLANE_THRESH_MM, seed=1)
    if fit1 is None or fit1[2].sum() < PLANE_MIN_FRAC * n_total:
        return None
    n1, d1, m1 = fit1
    rest = P[~m1]
    if rest.shape[0] < PLANE_MIN_FRAC * n_total:
        return None
    fit2 = _ransac_plane(rest, PLANE_THRESH_MM, seed=2)
    if fit2 is None or fit2[2].sum() < PLANE_MIN_FRAC * n_total:
        return None
    n2, d2, _ = fit2

    # The planes must meet steeply — otherwise it's ~one surface (a groove/flat), not a fillet.
    dihedral = np.degrees(np.arccos(min(1.0, abs(float(n1 @ n2)))))
    dihedral = min(dihedral, 180.0 - dihedral)    # fold to the acute angle between the faces
    if dihedral < DIHEDRAL_MIN_DEG:
        return None

    inter = _plane_intersection(n1, d1, n2, d2)
    if inter is None:
        return None
    direction, p0 = inter

    # Clip the infinite line to the joint: keep points near it, project onto dir, take extremes.
    perp = P - p0 - np.outer((P - p0) @ direction, direction)
    near = P[np.linalg.norm(perp, axis=1) < SEAM_BAND_MM]
    if near.shape[0] < MIN_PLANE_PTS // 2:
        return None
    t = (near - p0) @ direction
    a3d = p0 + float(np.percentile(t, 1)) * direction   # trim to reject stray endpoints
    b3d = p0 + float(np.percentile(t, 99)) * direction
    length_mm = float(np.linalg.norm(b3d - a3d))

    (ua, ub), (va, vb) = _project(intrinsics, np.array([a3d, b3d]))
    resid_mm = float(np.median(np.abs(np.linalg.norm(perp[np.linalg.norm(perp, axis=1) < SEAM_BAND_MM], axis=1))))
    return {
        "p1": (float(ua), float(va)),
        "p2": (float(ub), float(vb)),
        "length_px": float(np.hypot(ub - ua, vb - va)),
        "box": (x1, y1, x2, y2),
        "plane": None,
        "resid_px": resid_mm,          # for fillet this is mm-off-line, not px (see mode)
        "n_depth": int(m1.sum() + fit2[2].sum()),
        "n_rgb": 0,
        "mode": "fillet",
        "dihedral_deg": float(dihedral),
        "p1_cam_xyz_mm": [float(a3d[0]), float(a3d[1]), float(a3d[2])],
        "p2_cam_xyz_mm": [float(b3d[0]), float(b3d[1]), float(b3d[2])],
        "length_mm": length_mm,
    }


def find_seam_auto(bgr, depth_mm, aoi, intrinsics=None):
    """Dispatch: try the fillet (two-plane) path first when depth+intrinsics allow, else groove.

    The fillet path self-rejects on flat surfaces (the two planes come out ~coplanar), so a
    butt/lap/grooved joint falls through to find_seam. Returns the same dict shape either way
    (with a `mode` of "fillet" or "groove").
    """
    if depth_mm is not None and intrinsics is not None:
        s = find_fillet_seam(depth_mm, aoi, intrinsics)
        if s is not None:
            return s
    return find_seam(bgr, depth_mm, aoi)


def main():
    """Live preview to set the AOI and see the seam detection."""
    from camera import OrbbecCamera

    try:
        cam = OrbbecCamera(with_depth=True)
    except RuntimeError as exc:
        print(f"[camera error: {exc}]", file=sys.stderr)
        return

    box = {"aoi": load_aoi()}
    drag = {"p0": None, "cur": None}

    def on_mouse(event, x, y, _flags, _param):
        if event == cv.EVENT_LBUTTONDOWN:
            drag["p0"] = (x, y); drag["cur"] = (x, y)
        elif event == cv.EVENT_MOUSEMOVE and drag["p0"] is not None:
            drag["cur"] = (x, y)
        elif event == cv.EVENT_LBUTTONUP and drag["p0"] is not None:
            (ax, ay) = drag["p0"]
            a = (min(ax, x), min(ay, y), max(ax, x), max(ay, y))
            drag["p0"] = drag["cur"] = None
            if a[2] - a[0] > 5 and a[3] - a[1] > 5:
                box["aoi"] = a

    cv.namedWindow(WINDOW, cv.WINDOW_AUTOSIZE)
    print("Drag a box over the joint (long side along the seam). "
          "'s' saves the AOI, 'w' saves the detected seam, 'q' quits.")
    callback_set = False
    last = {"s": None, "capf": None}   # most recent detection, for the 'w' save-seam key
    flash = {"text": "", "n": 0}       # transient on-screen confirmation (frames remaining)
    with cam:
        while True:
            capf = cam.capture()
            if capf is None:
                if cv.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
                continue
            view = capf.bgr.copy()
            a = box["aoi"]
            last["capf"], last["s"] = capf, None
            if a is not None:
                cv.rectangle(view, (a[0], a[1]), (a[2], a[3]), (0, 200, 255), 2)
                s = find_seam_auto(capf.bgr, capf.depth_mm, a, capf.intrinsics)
                last["s"] = s
                if s is not None:
                    p1 = tuple(np.round(s["p1"]).astype(int))
                    p2 = tuple(np.round(s["p2"]).astype(int))
                    cv.line(view, p1, p2, (0, 255, 0), 2)
                    if s.get("mode") == "fillet":
                        label = (f"seam[fillet] {s['length_mm']:.0f}mm "
                                 f"dihedral {s['dihedral_deg']:.0f}deg resid {s['resid_px']:.1f}mm")
                    else:
                        zt = "" if s["plane"] is None else " +depth-Z"
                        src = f"depth{s['n_depth']}/rgb{s['n_rgb']}"
                        label = f"seam[groove] {s['length_px']:.0f}px resid {s['resid_px']:.1f}px {src}{zt}"
                    cv.putText(view, label, (a[0], a[1] - 8),
                               cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    cv.putText(view, "no seam in AOI", (a[0], a[1] - 8),
                               cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            if drag["p0"] is not None and drag["cur"] is not None:
                cv.rectangle(view, drag["p0"], drag["cur"], (255, 255, 0), 1)
            cv.putText(view, "drag=AOI  s=save AOI  w=save seam  q=quit",
                       (10, view.shape[0] - 14), cv.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            if flash["n"] > 0:
                cv.putText(view, flash["text"], (10, 28), cv.FONT_HERSHEY_SIMPLEX,
                           0.6, (0, 255, 255), 2)
                flash["n"] -= 1
            cv.imshow(WINDOW, view)
            key = cv.waitKey(1) & 0xFF
            if not callback_set:
                cv.setMouseCallback(WINDOW, on_mouse)
                callback_set = True
            if key in (ord("q"), 27):
                break
            if key == ord("s") and box["aoi"] is not None:
                save_aoi(box["aoi"])
                print(f"saved AOI {box['aoi']} -> {AOI_FILE.name}")
                flash.update(text=f"saved AOI -> {AOI_FILE.name}", n=45)
            if key == ord("w"):
                # Capture the currently detected seam as camera-frame endpoints the main
                # program can load (follow_saved_seam) and trace without re-detecting.
                capf = last["capf"]
                rec = seam_record(last["s"], capf.depth_mm if capf else None,
                                  capf.intrinsics if capf else None)
                if rec is None:
                    print("can't save seam: no seam detected in the AOI, or an endpoint has "
                          "no depth (hole) — adjust the AOI / reposition and retry.")
                    flash.update(text="no seam to save (see console)", n=45)
                else:
                    # Persist the current AOI box too, so the main program's live detection
                    # (follow_seam / detect_seam) picks up THIS box, not a stale one.
                    if box["aoi"] is not None:
                        save_aoi(box["aoi"])
                    save_seam(rec)
                    print(f"saved seam {rec['length_mm']:.0f}mm ({rec['depth_z']}-Z) + AOI "
                          f"-> {SEAM_FILE.name}, {AOI_FILE.name}")
                    flash.update(text=f"saved seam {rec['length_mm']:.0f}mm + AOI", n=45)
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
