"""Red marker detection by classical color segmentation.

Finds a red blob (a red dot / sticker / marker, e.g. on white paper) and returns its
SUB-PIXEL centroid in pixels. This is the right tool for a small color marker: the
open-vocabulary object detector (Grounding DINO) can't see a tiny featureless color
blob, whereas a color mask + image-moment centroid is both reliable and sub-pixel
accurate — which is exactly what the camera->base pipeline needs for good precision.

Red is matched by two complementary cues, OR-ed together:
  * HSV hue near 0/180 (red wraps around the hue circle) with enough saturation+value —
    strong for a saturated red.
  * CIELab a* (positive a* = "toward red") — stays positive even when a light marker on
    bright white paper desaturates (white bleeds through and kills HSV saturation). This
    is the robust cue for red-on-white, per the farino_app red-line work.

The mask is morphologically cleaned (OPEN clears specks, CLOSE fills the blob), then the
largest region's centroid is returned. Run directly for a live mask+overlay preview to
tune the thresholds:  uv run python marker.py
"""

import sys

import cv2 as cv
import numpy as np

# HSV thresholds (OpenCV hue is 0..180). Red occupies both ends of the circle.
HSV_S_MIN = 80
HSV_V_MIN = 60
HSV_H_LO = 10        # 0..HSV_H_LO = red (low end)
HSV_H_HI = 170       # HSV_H_HI..180 = red (high end)

# CIELab a* cue: (a - 128) > LAB_A_MIN means reddish. Catches desaturated red on white.
LAB_A_MIN = 20

# Reject blobs smaller than this (px of contour area) as specks/noise.
MIN_AREA_PX = 20

# Seam (line) detection: a marked seam is an ELONGATED red region. Keep regions whose
# long/short side ratio clears MIN_ELONGATION (rejects round dots / fat clutter) and
# whose area clears MIN_LINE_AREA, then fit a line and take its extreme endpoints.
MIN_LINE_AREA = 60
MIN_ELONGATION = 3.0

# Default number of centerline waypoints sampled along a curved line (see
# line_waypoints_from_mask). More = a finer trace of the curve; fewer = coarser but faster.
LINE_WAYPOINTS = 12

# A marker dot is ROUND and sits on a BRIGHT surface (white paper, light wood, a clean
# plate) — these two cues reject other red clutter in a welding cell (elongated red cables,
# a red e-stop on the dark floor). "Bright surface" is deliberately looser than "white": a
# marker on tan wood has a bright but SATURATED ring, which a white-only test wrongly rejects.
MIN_CIRCULARITY = 0.5   # 4*pi*A/P^2: ~1 for a disc, low for a thin cable
MIN_LIGHT_FRAC = 0.5    # fraction of the ring around the blob that must be a bright background
LIGHT_V_MIN = 150       # HSV value above which a ring pixel counts as "bright" (wood/paper/plate)
LIGHT_RING_PX = 12      # thickness of that surrounding ring (px)


def build_red_mask(bgr):
    """Binary mask (0/255) of red pixels: (HSV hue-wrap AND sat/val) OR (CIELab a*)."""
    hsv = cv.cvtColor(bgr, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(hsv)
    sat_val = (s >= HSV_S_MIN) & (v >= HSV_V_MIN)
    hsv_red = ((h <= HSV_H_LO) | (h >= HSV_H_HI)) & sat_val

    a = cv.cvtColor(bgr, cv.COLOR_BGR2Lab)[:, :, 1].astype(np.int16)
    lab_red = (a - 128) > LAB_A_MIN

    mask = ((hsv_red | lab_red).astype(np.uint8)) * 255
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN,
                           cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3)))
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE,
                           cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5)))
    return mask


def _circularity(contour, area):
    """4*pi*A / P^2: ~1.0 for a disc, near 0 for a thin/elongated shape (cable)."""
    perim = cv.arcLength(contour, True)
    return 4.0 * np.pi * area / (perim * perim) if perim > 0 else 0.0


def _on_light_fraction(bgr, contour):
    """Fraction of the ring just OUTSIDE the blob that is a BRIGHT background (high value).

    A marker dot on paper/wood/a clean plate is surrounded by a bright surface; a red e-stop
    on the dark floor is not. Uses brightness only (not "white"), so a bright but saturated
    surface like tan wood still counts.
    """
    h, w = bgr.shape[:2]
    blob = np.zeros((h, w), np.uint8)
    cv.drawContours(blob, [contour], -1, 255, -1)
    k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (LIGHT_RING_PX * 2 + 1,) * 2)
    ring = cv.dilate(blob, k) & cv.bitwise_not(blob)
    idx = ring > 0
    if idx.sum() < 10:
        return 0.0
    v = cv.cvtColor(bgr, cv.COLOR_BGR2HSV)[:, :, 2]
    light = v > LIGHT_V_MIN
    return float(light[idx].sum()) / float(idx.sum())


def _marker_from_contour(contour, area, light_frac):
    """Build a marker dict from an accepted contour, or None if it has no mass.

    dict = {"center": (u, v) float sub-pixel centroid, "area_px": int,
            "radius_px": float, "box": (x1, y1, x2, y2), "on_light": float}.
    """
    M = cv.moments(contour)
    if M["m00"] == 0:
        return None
    u = M["m10"] / M["m00"]
    v = M["m01"] / M["m00"]
    (_, _), radius = cv.minEnclosingCircle(contour)
    x, y, w, hh = cv.boundingRect(contour)
    return {
        "center": (float(u), float(v)),
        "area_px": int(area),
        "radius_px": float(radius),
        "box": (int(x), int(y), int(x + w), int(y + hh)),
        "on_light": round(float(light_frac), 2),
    }


def find_red_markers(bgr, min_area=MIN_AREA_PX, require_on_light=True, max_markers=None):
    """Locate ALL red marker DOTS. Returns a list of dicts (largest first); [] if none.

    Keeps every red blob that is (a) big enough, (b) round (circularity >= MIN_CIRCULARITY,
    rejecting elongated red cables) and, if `require_on_light`, (c) on a BRIGHT background —
    white paper, light wood, or a clean plate (rejecting a red e-stop / red clutter on the
    dark floor). Set require_on_light=False to accept a red dot on a dark surface too. Unlike
    find_red_marker, this returns every survivor, so multiple markers in view are all reported.
    Results are sorted by area (largest first); `max_markers` caps the count if given.

    Each dict has the shape documented on _marker_from_contour.
    """
    mask = build_red_mask(bgr)
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    markers = []
    for c in contours:
        area = cv.contourArea(c)
        if area < min_area:
            continue
        if _circularity(c, area) < MIN_CIRCULARITY:
            continue
        light_frac = _on_light_fraction(bgr, c) if require_on_light else 1.0
        if require_on_light and light_frac < MIN_LIGHT_FRAC:
            continue
        m = _marker_from_contour(c, area, light_frac)
        if m is not None:
            markers.append(m)

    markers.sort(key=lambda m: m["area_px"], reverse=True)
    if max_markers is not None:
        markers = markers[:max_markers]
    return markers


def find_red_marker(bgr, min_area=MIN_AREA_PX, require_on_light=True):
    """Locate the single most prominent red marker DOT. Returns a dict, or None.

    Thin wrapper over find_red_markers that returns the largest survivor (same acceptance
    rules). Kept for callers that only want one marker; use find_red_markers to get all.
    """
    markers = find_red_markers(bgr, min_area, require_on_light, max_markers=1)
    return markers[0] if markers else None


def _line_shaped_contours(contours, min_area, min_elongation):
    """All contours that qualify as a line (area + elongation), LONGEST first.

    Rejects round blobs (dots, bolt holes) via the min-area-rect long/short side ratio and
    specks via area. Shared by the single- and multi-line extractors.
    """
    kept = []
    for c in contours:
        if cv.contourArea(c) < min_area:
            continue
        (w, h) = cv.minAreaRect(c)[1]
        long_side, short_side = max(w, h), max(1.0, min(w, h))
        if long_side / short_side < min_elongation:
            continue
        kept.append((long_side, c))
    kept.sort(key=lambda t: t[0], reverse=True)
    return [c for _, c in kept]


def _most_line_shaped_contour(contours, min_area, min_elongation):
    """The single longest line-shaped contour, or None."""
    kept = _line_shaped_contours(contours, min_area, min_elongation)
    return kept[0] if kept else None


def _centerline_from_contour(contour, shape, num_samples):
    """Ordered centerline polyline of one filled contour via PCA-axis binning, or None.

    Projects the contour's filled pixels onto their principal axis (PCA), splits that axis
    into `num_samples` bins, and takes the centroid of the pixels in each bin — giving
    head->tail waypoints that follow a curve (empty bins dropped). Robust for straight and
    gently curved lines (those that don't fold back on their principal axis).

    dict = {"waypoints": [(u, v), ...] ordered, "length_px": float, "box": (x1,y1,x2,y2)}.
    """
    h, w = shape[:2]
    region = np.zeros((h, w), np.uint8)
    cv.drawContours(region, [contour], -1, 255, -1)
    ys, xs = np.where(region > 0)
    pts = np.column_stack([xs, ys]).astype(np.float64)      # (N, 2) as (x, y)

    mean = pts.mean(axis=0)
    evals, evecs = np.linalg.eigh(np.cov((pts - mean).T))
    axis = evecs[:, int(np.argmax(evals))]                  # principal direction (x, y)
    t = (pts - mean) @ axis
    order = np.argsort(t)
    t_sorted, pts_sorted = t[order], pts[order]

    edges = np.linspace(t_sorted[0], t_sorted[-1], num_samples + 1)
    centers = []
    for i in range(num_samples):
        lo, hi = edges[i], edges[i + 1]
        if i == num_samples - 1:                            # last bin includes the far edge
            sel = (t_sorted >= lo) & (t_sorted <= hi)
        else:
            sel = (t_sorted >= lo) & (t_sorted < hi)
        if sel.any():
            centers.append(pts_sorted[sel].mean(axis=0))
    if len(centers) < 2:
        return None

    waypoints = [(float(c[0]), float(c[1])) for c in centers]
    length = float(sum(np.hypot(*(np.array(waypoints[i + 1]) - np.array(waypoints[i])))
                       for i in range(len(waypoints) - 1)))
    x, y, bw, bh = cv.boundingRect(contour)
    return {
        "waypoints": waypoints,
        "length_px": length,
        "box": (int(x), int(y), int(x + bw), int(y + bh)),
    }


def line_endpoints_from_mask(mask, min_area=MIN_LINE_AREA, min_elongation=MIN_ELONGATION):
    """Endpoints of the most LINE-SHAPED region in a binary mask, or None.

    Picks the most elongated region (see _most_line_shaped_contour), fits a line (total least
    squares), and returns the extreme points along it. Shared by the red-line and depth seam
    detectors. For a CURVED line use line_waypoints_from_mask instead — this only ever returns
    the two extreme endpoints (a straight chord).

    dict = {"p1": (u, v), "p2": (u, v), "length_px": float, "box": (x1, y1, x2, y2)}.
    """
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    best = _most_line_shaped_contour(contours, min_area, min_elongation)
    if best is None:
        return None

    pts = best.reshape(-1, 2).astype(np.float32)
    vx, vy, x0, y0 = cv.fitLine(pts, cv.DIST_L2, 0, 0.01, 0.01).flatten()
    proj = (pts - np.array([x0, y0])) @ np.array([vx, vy])
    p1 = pts[int(np.argmin(proj))]
    p2 = pts[int(np.argmax(proj))]
    x, y, w, hh = cv.boundingRect(best)
    return {
        "p1": (float(p1[0]), float(p1[1])),
        "p2": (float(p2[0]), float(p2[1])),
        "length_px": float(np.linalg.norm(p2 - p1)),
        "box": (int(x), int(y), int(x + w), int(y + hh)),
    }


def line_waypoints_from_mask(mask, min_area=MIN_LINE_AREA, min_elongation=MIN_ELONGATION,
                             num_samples=LINE_WAYPOINTS):
    """Ordered CENTERLINE waypoints of the most line-shaped region, or None.

    Like line_endpoints_from_mask, but returns a POLYLINE that follows a CURVED line (see
    _centerline_from_contour) rather than just its two extreme endpoints. Picks the single
    most-elongated region; use lines_waypoints_from_mask to get ALL lines in the mask.

    dict = {"waypoints": [(u, v), ...] ordered, "length_px": float (summed segment length),
            "box": (x1, y1, x2, y2)}.
    """
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    best = _most_line_shaped_contour(contours, min_area, min_elongation)
    if best is None:
        return None
    return _centerline_from_contour(best, mask.shape, num_samples)


def lines_waypoints_from_mask(mask, min_area=MIN_LINE_AREA, min_elongation=MIN_ELONGATION,
                              num_samples=LINE_WAYPOINTS, max_lines=None):
    """Centerline polylines of ALL line-shaped regions in a mask, longest first; [] if none.

    The multi-line form of line_waypoints_from_mask: builds a centerline for every qualifying
    region (each a separate red mark), sorted by length. `max_lines` caps the count if given.
    A region that can't yield >=2 centerline points is dropped. Each entry has the shape
    documented on _centerline_from_contour.
    """
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    kept = _line_shaped_contours(contours, min_area, min_elongation)
    if max_lines is not None:
        kept = kept[:max_lines]
    out = []
    for c in kept:
        poly = _centerline_from_contour(c, mask.shape, num_samples)
        if poly is not None:
            out.append(poly)
    return out


def find_red_seam(bgr, min_area=MIN_LINE_AREA, min_elongation=MIN_ELONGATION):
    """Find the most line-shaped RED region (a red-marked seam). Returns endpoints or None."""
    return line_endpoints_from_mask(build_red_mask(bgr), min_area, min_elongation)


def find_red_line_waypoints(bgr, min_area=MIN_LINE_AREA, min_elongation=MIN_ELONGATION,
                            num_samples=LINE_WAYPOINTS):
    """Ordered centerline waypoints of the most line-shaped RED region (follows a curve)."""
    return line_waypoints_from_mask(build_red_mask(bgr), min_area, min_elongation, num_samples)


def find_red_lines(bgr, min_area=MIN_LINE_AREA, min_elongation=MIN_ELONGATION,
                   num_samples=LINE_WAYPOINTS, max_lines=None):
    """Centerline polylines of ALL line-shaped RED regions (longest first); [] if none."""
    return lines_waypoints_from_mask(build_red_mask(bgr), min_area, min_elongation,
                                     num_samples, max_lines)


def main():
    """Live preview: shows the frame with the detected marker + the red mask, to tune."""
    from camera import OrbbecCamera

    try:
        cam = OrbbecCamera()
    except RuntimeError as exc:
        print(f"[camera error: {exc}]", file=sys.stderr)
        return

    win = "red_marker"
    cv.namedWindow(win, cv.WINDOW_AUTOSIZE)
    print("Live red-marker preview. q/Esc to quit.")
    with cam:
        for bgr in cam.frames():
            markers = find_red_markers(bgr)
            view = bgr.copy()
            mask = build_red_mask(bgr)
            for i, m in enumerate(markers):
                u, v = m["center"]
                cv.circle(view, (int(round(u)), int(round(v))),
                          max(4, int(m["radius_px"])), (0, 255, 0), 2)
                cv.drawMarker(view, (int(round(u)), int(round(v))), (0, 255, 0),
                              cv.MARKER_CROSS, 14, 1)
                cv.putText(view, f"#{i + 1}", (int(round(u)) + 8, int(round(v)) - 8),
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            if markers:
                cv.putText(view, f"{len(markers)} red marker(s)", (10, 24),
                           cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                cv.putText(view, "no red marker", (10, 24),
                           cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            combo = np.hstack([view, cv.cvtColor(mask, cv.COLOR_GRAY2BGR)])
            cv.imshow(win, combo)
            if cv.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
