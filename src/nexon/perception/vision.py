"""Shared vision runtime: one camera + detector for the whole app, plus a live window.

Only one process can open the 336L, so the Claude `detect_objects` tool and the
on-screen preview must share a single camera. `VisionHub` owns it:

  - a background thread continuously captures frames and, if a display is available,
    draws the latest detections over the live video and shows them;
  - `look()` (called by the tool) runs detection on the most recent frame and
    publishes the boxes/dimensions back, so the window overlays exactly what Claude
    just saw and measured.

Capture happens only on the hub thread (one consumer of the camera pipeline); the
detector is only ever called from `look()`. The two share frames/results behind a
lock. The window is best-effort: any GUI error disables the preview but never takes
down the chat. Access the process-wide instance via `get_hub()`.
"""

import logging
import os
import threading
import time

import cv2
import numpy as np

from nexon.perception import marker
from nexon.perception import seam
from nexon.perception.camera import OrbbecCamera
from nexon.perception.detector import Detection, load_detector
from nexon.perception.dimensioner import measure_all
from nexon.perception.viz import colorize_depth, draw_detections

log = logging.getLogger("nexon")

DETECTOR_BACKEND = os.environ.get("NEXON_DETECTOR", "grounding-dino")
WINDOW_NAME = "nexon — live vision"


def _has_display() -> bool:
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


class VisionHub:
    def __init__(self, backend: str = DETECTOR_BACKEND, show_window: bool | None = None):
        self.backend = backend
        self.show_window = _has_display() if show_window is None else show_window
        self._camera: OrbbecCamera | None = None
        self._detector = None
        self._lock = threading.Lock()
        self._cap = None          # latest CapturedFrame (published by the hub thread)
        self._dets: list = []     # latest detections (published by look())
        self._thread: threading.Thread | None = None
        self._running = False
        self._show_depth = False
        # Live seam-AOI drawing on the preview window ('a' toggles; drag a box).
        self._aoi_draw = False
        self._drag = {"p0": None, "cur": None}
        self._mouse_bound = False
        self._grid = False  # 'g' toggles a labeled pixel grid to read off AOI coordinates

    # -- lifecycle -----------------------------------------------------------

    def ensure_started(self):
        """Open the camera and start the capture/preview thread (idempotent)."""
        if self._running:
            return
        log.info("vision: starting Gemini 336L (color + depth)")
        cam = OrbbecCamera(with_depth=True)
        cam.start()
        self._camera = cam
        self._running = True
        # The capture thread warms up (auto-exposure) in the background; we don't
        # block the chat prompt on it. look() waits for the first frame via
        # _latest_capture, and by the time anyone asks, exposure has long settled.
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        try:
            cv2.destroyAllWindows()
        except Exception:  # noqa: BLE001
            pass
        if self._camera is not None:
            self._camera.stop()
            self._camera = None
        self._detector = None

    # -- capture + preview thread -------------------------------------------

    def _loop(self):
        window = self.show_window
        while self._running:
            cap = self._camera.capture()
            if cap is None:
                continue
            with self._lock:
                self._cap = cap
                dets = self._dets
            if window:
                window = self._render(cap, dets)

    def _on_mouse(self, event, x, y, _flags, _param):
        """Drag a seam AOI on the preview (only while AOI-draw mode is on). Saves on release."""
        if not self._aoi_draw:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            self._drag["p0"] = (x, y)
            self._drag["cur"] = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self._drag["p0"] is not None:
            self._drag["cur"] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and self._drag["p0"] is not None:
            (ax, ay) = self._drag["p0"]
            a = (min(ax, x), min(ay, y), max(ax, x), max(ay, y))
            self._drag["p0"] = self._drag["cur"] = None
            if a[2] - a[0] > 5 and a[3] - a[1] > 5:
                seam.save_aoi(a)
                log.info("vision: seam AOI set to %s", a)

    def _draw_seam_overlay(self, frame, cap):
        """Overlay the saved seam AOI + its live detection, and any in-progress drag box.

        Best-effort: any error here must never disable the whole preview.
        """
        try:
            aoi = seam.load_aoi()
            if aoi is not None:
                cv2.rectangle(frame, (aoi[0], aoi[1]), (aoi[2], aoi[3]), (0, 200, 255), 2)
                s = seam.find_seam(cap.bgr, cap.depth_mm, aoi)
                if s is not None:
                    p1 = tuple(np.round(s["p1"]).astype(int))
                    p2 = tuple(np.round(s["p2"]).astype(int))
                    cv2.line(frame, p1, p2, (0, 255, 0), 2)
            if self._aoi_draw:
                cv2.putText(frame, "AOI DRAW: drag a tight box along the seam ('a' to exit)",
                            (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
                d = self._drag
                if d["p0"] is not None and d["cur"] is not None:
                    cv2.rectangle(frame, d["p0"], d["cur"], (255, 255, 0), 1)
        except Exception as exc:  # noqa: BLE001 — overlay is optional, keep the video alive
            log.debug("vision: seam overlay skipped (%s)", exc)

    def _draw_grid(self, frame, step: int = 100):
        """Faint labeled pixel grid so you can read off AOI coordinates by eye ('g' toggles)."""
        h, w = frame.shape[:2]
        for x in range(0, w, step):
            cv2.line(frame, (x, 0), (x, h), (90, 90, 90), 1)
            cv2.putText(frame, str(x), (x + 2, 14), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (0, 255, 255), 1, cv2.LINE_AA)
        for y in range(0, h, step):
            cv2.line(frame, (0, y), (w, y), (90, 90, 90), 1)
            cv2.putText(frame, str(y), (2, y + 14), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (0, 255, 255), 1, cv2.LINE_AA)

    def _render(self, cap, dets) -> bool:
        """Draw + show one frame. Returns False (disable preview) on any GUI error."""
        try:
            frame = draw_detections(cap.bgr, dets)
            cv2.putText(frame, f"dets:{len(dets)}", (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            self._draw_seam_overlay(frame, cap)
            if self._grid:
                self._draw_grid(frame)
            cv2.imshow(WINDOW_NAME, frame)
            if self._show_depth and cap.depth_mm is not None:
                cv2.imshow("depth", colorize_depth(cap.depth_mm))

            key = cv2.waitKey(1) & 0xFF
            if not self._mouse_bound:
                # The Qt window handle isn't ready for a few frames; setMouseCallback
                # raises "NULL window handler" until it is. Retry each frame instead of
                # letting that error disable the whole preview.
                try:
                    cv2.setMouseCallback(WINDOW_NAME, self._on_mouse)
                    self._mouse_bound = True
                except cv2.error:
                    pass
            if key in (ord("q"), 27):        # close the preview; chat keeps running
                cv2.destroyAllWindows()
                return False
            if key == ord("d"):
                self._show_depth = not self._show_depth
                if not self._show_depth:
                    cv2.destroyWindow("depth")
            elif key == ord("s"):
                name = f"snapshot_{int(time.time())}.jpg"
                cv2.imwrite(name, frame)
                log.info("vision: saved %s", name)
            elif key == ord("a"):
                self._aoi_draw = not self._aoi_draw
                log.info("vision: seam AOI-draw %s", "on" if self._aoi_draw else "off")
            elif key == ord("g"):
                self._grid = not self._grid
            return True
        except Exception as exc:  # noqa: BLE001
            log.warning("vision: preview disabled (%s)", exc)
            try:
                cv2.destroyAllWindows()
            except Exception:  # noqa: BLE001
                pass
            return False

    # -- detection (called by the tool) -------------------------------------

    def latest(self):
        """The most recent CapturedFrame and the detections drawn on it, or (None, []).

        Non-blocking, for a UI that paints on its own clock. The returned frame is whatever
        the capture thread last published; callers that keep it beyond the next frame must
        copy it, since the camera's BGR buffer can be recycled.
        """
        with self._lock:
            return self._cap, list(self._dets)

    def _latest_capture(self, timeout: float = 2.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._lock:
                if self._cap is not None:
                    return self._cap
            time.sleep(0.02)
        return None

    def look(self, targets, min_confidence: float = 0.4, measure: bool = False) -> dict:
        """Detect `targets` in the latest frame; optionally measure. Publishes overlay."""
        self.ensure_started()
        if self._detector is None:
            log.info("vision: loading detector '%s' (first run downloads the model)",
                     self.backend)
            self._detector = load_detector(self.backend)

        cap = self._latest_capture()
        if cap is None or cap.bgr is None:
            return {"error": "no frame captured from camera"}

        detections = self._detector.detect(cap.bgr, targets, min_confidence=min_confidence)
        if measure and cap.depth_mm is not None:
            measure_all(detections, cap.depth_mm, cap.intrinsics)

        with self._lock:
            self._dets = detections  # window overlays these until the next look()

        h, w = cap.bgr.shape[:2]
        return {
            "image_size": {"width": w, "height": h},
            "measured": bool(measure and cap.depth_mm is not None),
            "detections": [d.as_dict() for d in detections],
        }

    def locate(self, target, min_confidence: float = 0.4, patch: int = 5) -> dict:
        """Locate the best `target` and return its 3D point in the CAMERA frame (mm).

        Detects `target`, takes the highest-confidence match, samples aligned depth at
        its box centre (median patch, with a nearest-band box fallback if the centre is
        a hole), and deprojects to camera-frame XYZ via the intrinsics. This stays
        robot-agnostic — the caller applies the camera->base extrinsic. Returns a dict
        with `cam_xyz_mm` and the pixel, or an `error`.
        """
        self.ensure_started()
        if self._detector is None:
            self._detector = load_detector(self.backend)

        cap = self._latest_capture()
        if cap is None or cap.bgr is None:
            return {"error": "no frame captured from camera"}
        if cap.depth_mm is None or cap.intrinsics is None:
            return {"error": "no depth/intrinsics available (camera not in depth mode?)"}

        detections = self._detector.detect(cap.bgr, [target], min_confidence=min_confidence)
        if not detections:
            return {"error": f"no '{target}' detected"}
        best = max(detections, key=lambda d: d.confidence)
        with self._lock:
            self._dets = [best]  # overlay just the located object

        u, v = best.center
        z = _depth_at(cap.depth_mm, u, v, patch, best.box)
        if z <= 0:
            return {"error": f"'{target}' found but no depth there (holes) — reposition"}
        x, y, zc = cap.intrinsics.deproject(u, v, z)
        return {
            "label": best.label,
            "confidence": round(best.confidence, 3),
            "center": [int(u), int(v)],
            "cam_xyz_mm": [float(x), float(y), float(zc)],
        }


    def locate_markers(self, patch: int = 5, max_markers: int | None = None) -> dict:
        """Locate ALL RED color markers and return their 3D points in the CAMERA frame (mm).

        Uses classical color segmentation (marker.find_red_markers), not the object
        detector — a small red blob is exactly what the neural detector can't see. Every
        marker in view is reported (largest first): sub-pixel centroid, aligned depth there,
        deprojected to camera-frame XYZ. Robot-agnostic; the caller applies the camera->base
        extrinsic. Returns {"markers": [...]} or an {"error": ...}. A marker whose depth is a
        hole is still listed, with `cam_xyz_mm` = None and a `note`, so it isn't silently lost.
        """
        self.ensure_started()
        cap = self._latest_capture()
        if cap is None or cap.bgr is None:
            return {"error": "no frame captured from camera"}

        found = marker.find_red_markers(cap.bgr, max_markers=max_markers)
        if not found:
            return {"error": "no red marker found"}

        with self._lock:  # overlay every found marker box in the preview
            self._dets = [
                Detection(label=f"red marker #{i + 1}", confidence=1.0, box=m["box"])
                for i, m in enumerate(found)
            ]

        if cap.depth_mm is None or cap.intrinsics is None:
            return {"error": "no depth/intrinsics available (camera not in depth mode?)"}

        markers = []
        for m in found:
            u, v = m["center"]
            entry = {
                "center": [round(u, 1), round(v, 1)],
                "radius_px": round(m["radius_px"], 1),
                "area_px": m["area_px"],
            }
            z = _depth_at(cap.depth_mm, int(round(u)), int(round(v)), patch, m["box"])
            if z <= 0:
                entry["cam_xyz_mm"] = None
                entry["note"] = "no depth there (holes) — reposition"
            else:
                x, y, zc = cap.intrinsics.deproject(u, v, z)
                entry["cam_xyz_mm"] = [float(x), float(y), float(zc)]
            markers.append(entry)
        return {"markers": markers}

    def locate_marker(self, patch: int = 5) -> dict:
        """Locate the single most prominent RED marker (camera-frame mm). See locate_markers.

        Thin wrapper returning the largest marker as a flat dict with `cam_xyz_mm` + pixel,
        or an `error`. Kept for callers that only act on one marker (e.g. move_to_red_marker).
        """
        res = self.locate_markers(patch, max_markers=1)
        if "error" in res:
            return res
        m = res["markers"][0]
        if m.get("cam_xyz_mm") is None:
            return {"error": "red marker found but no depth there (holes) — reposition"}
        return {
            "center": m["center"],
            "radius_px": m["radius_px"],
            "area_px": m["area_px"],
            "cam_xyz_mm": m["cam_xyz_mm"],
        }


    def locate_red_lines(self, patch: int = 5, num_samples: int = marker.LINE_WAYPOINTS,
                         max_lines: int | None = None) -> dict:
        """Locate ALL RED LINES (red-marked seams/paths) as ordered 3D POLYLINES (mm).

        The multi-line, color analog of locate_seam: segments EVERY line-shaped RED region in
        the whole frame (marker.find_red_lines, no AOI) and samples each one's centerline into
        `num_samples` ordered waypoints — so several separate red lines/tapes, straight or
        curved, are all returned (longest first). Each waypoint gets a local depth patch and is
        deprojected to camera-frame XYZ; waypoints over a depth hole are dropped, and a line
        left with <2 valid points is skipped. Robot-agnostic; the caller applies the extrinsic.
        Returns {"lines": [{"waypoints": [{"px", "cam_xyz_mm"}, ...], ...}, ...]} or an `error`.
        """
        self.ensure_started()
        cap = self._latest_capture()
        if cap is None or cap.bgr is None:
            return {"error": "no frame captured from camera"}

        found = marker.find_red_lines(cap.bgr, num_samples=num_samples, max_lines=max_lines)
        if not found:
            return {"error": "no red line found (need an elongated red mark — a dot won't do)"}

        with self._lock:  # overlay every found line's box in the preview
            self._dets = [Detection(label=f"red line #{i + 1}", confidence=1.0, box=s["box"])
                          for i, s in enumerate(found)]

        if cap.depth_mm is None or cap.intrinsics is None:
            return {"error": "no depth/intrinsics available (camera not in depth mode?)"}

        lines = []
        for s in found:
            waypoints = []
            for (u, v) in s["waypoints"]:
                z = _depth_at(cap.depth_mm, int(round(u)), int(round(v)), patch, s["box"])
                if z <= 0:
                    continue  # skip waypoints over a depth hole; keep the rest of this line
                x, y, zc = cap.intrinsics.deproject(u, v, z)
                waypoints.append({"px": [round(u, 1), round(v, 1)],
                                  "cam_xyz_mm": [float(x), float(y), float(zc)]})
            if len(waypoints) < 2:
                continue  # this line had too few points with depth — drop it
            lines.append({
                "waypoints": waypoints,
                "length_px": round(s["length_px"], 1),
                "num_sampled": len(s["waypoints"]),
                "num_with_depth": len(waypoints),
            })
        if not lines:
            return {"error": "red line(s) found but none had enough depth (holes) — reposition"}
        return {"lines": lines, "num_detected": len(found)}

    def locate_red_line(self, patch: int = 5,
                        num_samples: int = marker.LINE_WAYPOINTS) -> dict:
        """Locate the single most prominent RED line as an ordered 3D polyline. See locate_red_lines.

        Thin wrapper returning the longest line's `waypoints` dict, or an `error`. Kept for
        callers that only act on one line (detect_red_line).
        """
        res = self.locate_red_lines(patch, num_samples, max_lines=1)
        if "error" in res:
            return res
        return res["lines"][0]

    def frame_size(self):
        """(width, height) of the current camera frame in pixels, or None if unavailable."""
        self.ensure_started()
        cap = self._latest_capture(timeout=1.0)
        if cap is None or cap.bgr is None:
            return None
        h, w = cap.bgr.shape[:2]
        return (w, h)

    def locate_seam(self) -> dict:
        """Locate the seam LINE (joint between two parts) within the saved AOI.

        The seam line is found in RGB (the dark joint line) via seam.find_seam within the AOI
        set by `uv run python -m nexon.perception.seam` — this works for a flush joint too. Each endpoint's
        height (Z) is taken from a depth plane fit over the AOI (clean, on-surface); if depth
        is too sparse it falls back to a local depth patch. Deprojects to camera-frame XYZ;
        robot-agnostic, the caller applies the extrinsic. Returns p1/p2 pixels and
        `p1_cam_xyz_mm`/`p2_cam_xyz_mm`, or an `error`.
        """
        self.ensure_started()
        cap = self._latest_capture()
        if cap is None or cap.bgr is None:
            return {"error": "no frame captured from camera"}
        if cap.depth_mm is None or cap.intrinsics is None:
            return {"error": "no depth/intrinsics available (camera not in depth mode?)"}

        aoi = seam.load_aoi()
        if aoi is None:
            return {"error": "no seam AOI set — run 'uv run python -m nexon.perception.seam' to draw one"}

        s = seam.find_seam(cap.bgr, cap.depth_mm, aoi)
        if s is None:
            return {"error": "no seam found in the AOI (no clear joint line — check the AOI "
                             "is tight with its long side along the seam)"}

        with self._lock:  # overlay the seam box in the preview
            self._dets = [Detection(label="seam", confidence=1.0, box=s["box"])]

        out = {
            "p1_px": [round(s["p1"][0], 1), round(s["p1"][1], 1)],
            "p2_px": [round(s["p2"][0], 1), round(s["p2"][1], 1)],
            "length_px": round(s["length_px"], 1),
            "resid_px": round(s["resid_px"], 2),
        }
        for name, (u, v) in (("p1", s["p1"]), ("p2", s["p2"])):
            if s["plane"] is not None:                       # clean on-surface Z from plane
                z = seam.plane_z(s["plane"], u, v)
            else:                                            # fall back to a local depth patch
                z = _depth_at(cap.depth_mm, int(round(u)), int(round(v)), 5, s["box"])
            if z <= 0:
                return {"error": f"seam endpoint {name} has no depth (hole) — reposition"}
            x, y, zc = cap.intrinsics.deproject(u, v, z)
            out[f"{name}_cam_xyz_mm"] = [float(x), float(y), float(zc)]
        return out


def _depth_at(depth_mm, u, v, patch, box):
    """Median depth (mm) at (u,v): an NxN valid-median patch, else the box's near band.

    The centre patch is the most localized read; if it's all holes, fall back to the
    nearest coherent depth band inside the detection box (the object surface).
    """
    r = patch // 2
    win = depth_mm[max(0, v - r):v + r + 1, max(0, u - r):u + r + 1]
    valid = win[win > 0]
    if valid.size:
        return float(np.median(valid))
    x1, y1, x2, y2 = box
    roi = depth_mm[max(0, y1):y2, max(0, x1):x2]
    rv = roi[roi > 0]
    if rv.size < 10:
        return 0.0
    near = np.percentile(rv, 20)          # nearest surface band, ignore the far background
    band = rv[rv <= near + 30.0]
    return float(np.median(band)) if band.size else 0.0


# Process-wide singleton so the tool and the chat share one camera.
_hub: VisionHub | None = None


def get_hub(**kwargs) -> VisionHub:
    """Return the shared hub, creating it on first call (kwargs apply then only)."""
    global _hub
    if _hub is None:
        _hub = VisionHub(**kwargs)
    return _hub


def shutdown():
    global _hub
    if _hub is not None:
        _hub.stop()
        _hub = None
