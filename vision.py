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

import os
import sys
import threading
import time

import cv2

from camera import OrbbecCamera
from detector import load_detector
from dimensioner import measure_all
from viz import colorize_depth, draw_detections

DETECTOR_BACKEND = os.environ.get("NEXON_DETECTOR", "grounding-dino")
WINDOW_NAME = "nexon — live vision"
_WARMUP_FRAMES = 30


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

    # -- lifecycle -----------------------------------------------------------

    def ensure_started(self):
        """Open the camera and start the capture/preview thread (idempotent)."""
        if self._running:
            return
        print("[vision: starting Gemini 336L (color + depth)…]", file=sys.stderr)
        cam = OrbbecCamera(with_depth=True)
        cam.start()
        for _ in range(_WARMUP_FRAMES):  # let auto-exposure settle
            cam.read()
        self._camera = cam
        self._running = True
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

    def _render(self, cap, dets) -> bool:
        """Draw + show one frame. Returns False (disable preview) on any GUI error."""
        try:
            frame = draw_detections(cap.bgr, dets)
            cv2.putText(frame, f"dets:{len(dets)}", (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow(WINDOW_NAME, frame)
            if self._show_depth and cap.depth_mm is not None:
                cv2.imshow("depth", colorize_depth(cap.depth_mm))

            key = cv2.waitKey(1) & 0xFF
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
                print(f"[vision: saved {name}]", file=sys.stderr)
            return True
        except Exception as exc:  # noqa: BLE001
            print(f"[vision: preview disabled ({exc})]", file=sys.stderr)
            try:
                cv2.destroyAllWindows()
            except Exception:  # noqa: BLE001
                pass
            return False

    # -- detection (called by the tool) -------------------------------------

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
            print(f"[vision: loading detector '{self.backend}' "
                  f"(first run downloads the model)…]", file=sys.stderr)
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
