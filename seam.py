"""Seam detection (the joint between two parts) within an area of interest (AOI).

The seam is found in RGB as the dark line the joint makes, using a PROFILE SCAN inside
the AOI: along the AOI's long axis, take the darkest pixel on each scan line (the seam
crossing) and robustly fit a line to those points. This works for a flush butt joint
(no depth step, but a clear dark line) as well as a grooved one, and — constrained to a
tight AOI — it ignores the plate outlines / bolt holes that confuse a full-frame edge
detector. A depth plane is fit over the AOI to give each endpoint a clean on-surface
height (Z); the seam LINE comes from RGB, the HEIGHT from depth.

Needs a visible seam line and a tight AOI. Draw the AOI so its long side runs ALONG the
seam. A perfectly invisible joint (no dark line, no depth step) can't be seen by any
camera — you'd need teaching / touch.

Set the AOI once with the live preview (needs the camera):
    uv run python seam.py
    drag a box over the joint (long side along the seam) -> 's' saves it -> 'q' quits.
detect_seam / follow_seam then load that AOI automatically.
"""

import json
import sys
from pathlib import Path

import cv2 as cv
import numpy as np

AOI_FILE = Path(__file__).resolve().parent / "seam_aoi.json"

DARK_K = 0.6           # a scan line's darkest pixel must be < mean - DARK_K*std to count
INLIER_PX = 3.0        # line-fit inlier band (px) for the robust refit
MIN_SPAN_FRAC = 0.4    # seam must span at least this fraction of the AOI's long side
MIN_PLANE_PTS = 200    # valid depth px in the AOI needed to fit the height plane
WINDOW = "seam_aoi"


def load_aoi(path=AOI_FILE):
    """Return the saved AOI (x1, y1, x2, y2), or None if none set."""
    p = Path(path)
    if not p.exists():
        return None
    return tuple(json.loads(p.read_text())["aoi"])


def save_aoi(aoi, path=AOI_FILE):
    Path(path).write_text(json.dumps({"aoi": [int(v) for v in aoi]}))


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


def _profile_points(gray, horizontal):
    """Darkest pixel per scan line, kept only where it's clearly dark. AOI-local coords.

    horizontal=True scans each COLUMN (seam runs across, ~horizontal); False scans each
    ROW (seam ~vertical). Returns (us, vs) arrays of candidate seam points.
    """
    g = cv.GaussianBlur(gray, (5, 5), 0).astype(np.float32)
    thr = g.mean() - DARK_K * g.std()
    if horizontal:
        rows = np.argmin(g, axis=0)
        mins = g[rows, np.arange(g.shape[1])]
        keep = mins < thr
        return np.arange(g.shape[1])[keep], rows[keep]
    cols = np.argmin(g, axis=1)
    mins = g[np.arange(g.shape[0]), cols]
    keep = mins < thr
    return cols[keep], np.arange(g.shape[0])[keep]


def find_seam(bgr, depth_mm, aoi):
    """Find the seam LINE (dark joint) in RGB within `aoi`; take endpoint Z from depth.

    Returns a dict or None:
      {"p1": (u, v), "p2": (u, v), "length_px": float, "box": (x1,y1,x2,y2),
       "plane": (a,b,c) or None, "resid_px": float}
    p1/p2 are full-frame pixels; plane (if depth allowed a fit) gives a clean on-surface
    height at each endpoint via plane_z().
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

    us, vs = _profile_points(gray, horizontal)
    if us.size < MIN_SPAN_FRAC * span:
        return None
    pts = np.column_stack([us + x1, vs + y1]).astype(np.float32)  # full-frame

    # Robust fit, then keep inliers and refit (rejects scratches / stray dark pixels).
    vx, vy, x0, y0 = cv.fitLine(pts, cv.DIST_HUBER, 0, 0.01, 0.01).flatten()
    normal = np.array([-vy, vx])
    resid = np.abs((pts - np.array([x0, y0])) @ normal)
    pts = pts[resid < INLIER_PX]
    if pts.shape[0] < MIN_SPAN_FRAC * span:
        return None
    vx, vy, x0, y0 = cv.fitLine(pts, cv.DIST_L2, 0, 0.01, 0.01).flatten()
    normal = np.array([-vy, vx])
    resid = np.abs((pts - np.array([x0, y0])) @ normal)

    proj = (pts - np.array([x0, y0])) @ np.array([vx, vy])
    p1 = pts[int(np.argmin(proj))]
    p2 = pts[int(np.argmax(proj))]
    return {
        "p1": (float(p1[0]), float(p1[1])),
        "p2": (float(p2[0]), float(p2[1])),
        "length_px": float(np.linalg.norm(p2 - p1)),
        "box": (x1, y1, x2, y2),
        "plane": fit_plane_in_aoi(depth_mm, aoi) if depth_mm is not None else None,
        "resid_px": float(resid.mean()),
    }


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
    print("Drag a box over the joint (long side along the seam). 's' saves, 'q' quits.")
    callback_set = False
    with cam:
        while True:
            capf = cam.capture()
            if capf is None:
                if cv.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
                continue
            view = capf.bgr.copy()
            a = box["aoi"]
            if a is not None:
                cv.rectangle(view, (a[0], a[1]), (a[2], a[3]), (0, 200, 255), 2)
                s = find_seam(capf.bgr, capf.depth_mm, a)
                if s is not None:
                    p1 = tuple(np.round(s["p1"]).astype(int))
                    p2 = tuple(np.round(s["p2"]).astype(int))
                    cv.line(view, p1, p2, (0, 255, 0), 2)
                    zt = "" if s["plane"] is None else " +depth-Z"
                    cv.putText(view, f"seam {s['length_px']:.0f}px resid {s['resid_px']:.1f}px{zt}",
                               (a[0], a[1] - 8), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    cv.putText(view, "no seam in AOI", (a[0], a[1] - 8),
                               cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            if drag["p0"] is not None and drag["cur"] is not None:
                cv.rectangle(view, drag["p0"], drag["cur"], (255, 255, 0), 1)
            cv.putText(view, "drag=AOI  s=save  q=quit", (10, view.shape[0] - 14),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
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
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
