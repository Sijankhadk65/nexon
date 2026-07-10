"""Live camera view: pulls frames from the VisionHub and paints them.

The hub's capture thread publishes the newest CapturedFrame; this widget paints on its own
clock (a QTimer) rather than being driven by the camera. Decoupling them means a slow repaint
drops frames instead of stalling capture, and nothing in perception/ has to import Qt.

Frames are deep-copied into the QImage. QImage wraps the numpy buffer WITHOUT owning it, and
the array it wraps here is a local that is freed as soon as the frame is painted — so the copy
is what keeps the pixmap from pointing at freed memory. (viz.draw_detections and
colorize_depth already return fresh arrays, so this is the only copy on the path.)
"""

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QLabel, QSizePolicy

from nexon.perception.viz import colorize_depth, draw_detections

REPAINT_MS = 33          # ~30 fps; the camera, not this, is the rate limit


def bgr_to_pixmap(bgr) -> QPixmap:
    """Copy a contiguous HxWx3 BGR array into a QPixmap."""
    h, w = bgr.shape[:2]
    # Format_BGR888 consumes the array's own byte order, so no cvtColor is needed.
    # .copy() detaches the image from `bgr`, which QImage does not keep alive.
    image = QImage(bgr.data, w, h, bgr.strides[0], QImage.Format_BGR888).copy()
    return QPixmap.fromImage(image)


class VideoView(QLabel):
    """Scaled live view of the hub's latest frame. Emits frame_size once known."""

    frame_size = Signal(int, int)

    def __init__(self, hub, parent=None):
        super().__init__(parent)
        self._hub = hub
        self._show_depth = False
        self._last = None
        self._announced = False

        self.setMinimumSize(640, 360)
        self.setAlignment(Qt.AlignCenter)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("background:#101010; color:#888;")
        self.setText("waiting for camera…")

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(REPAINT_MS)

    def set_depth(self, on: bool) -> None:
        self._show_depth = bool(on)

    def _tick(self) -> None:
        cap, dets = self._hub.latest()
        if cap is None or cap is self._last:
            return                      # no camera yet, or no new frame since last repaint
        self._last = cap

        if self._show_depth and cap.depth_mm is not None:
            frame = colorize_depth(cap.depth_mm)
        else:
            frame = draw_detections(cap.bgr, dets)

        if not self._announced:
            h, w = frame.shape[:2]
            self.frame_size.emit(w, h)
            self._announced = True

        pix = bgr_to_pixmap(frame)
        self.setPixmap(pix.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
