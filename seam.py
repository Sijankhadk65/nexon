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
    cams = []
    for name, (u, v) in (("p1", s["p1"]), ("p2", s["p2"])):
        z = plane_z(s["plane"], u, v) if s["plane"] is not None else _sample_depth_mm(depth_mm, u, v)
        if z <= 0:
            return None
        x, y, zc = intrinsics.deproject(u, v, z)
        rec[f"{name}_cam_xyz_mm"] = [float(x), float(y), float(zc)]
        cams.append(np.array([x, y, zc], float))
    rec["length_mm"] = round(float(np.linalg.norm(cams[1] - cams[0])), 1)
    rec["depth_z"] = "plane" if s["plane"] is not None else "patch"
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
                s = find_seam(capf.bgr, capf.depth_mm, a)
                last["s"] = s
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
