"""The live camera, filling its half of the window.

The hub's capture thread publishes the newest CapturedFrame; this widget paints on its own
clock (a QTimer) rather than being driven by the camera. Decoupling them means a slow repaint
drops frames instead of stalling capture, and nothing in perception/ has to import Qt.

Frames are deep-copied into the QImage. QImage wraps the numpy buffer WITHOUT owning it, and
the array it wraps here is a local that is freed as soon as the frame is painted — so the copy
is what keeps the pixmap from pointing at freed memory. (viz.draw_detections and
colorize_depth already return fresh arrays, so this is the only copy on the path.)

Three things changed when this became a whole half of the window rather than a pane in it:

  IT COVERS, IT DOES NOT FIT. A letterboxed frame with black bars is a picture of a camera
  feed. Scaling to fill and cropping the overflow is a window onto the cell. The aspect
  ratio is preserved either way; only the bars go.

  IT IS BARELY VEILED. An earlier design floated the whole conversation on top of this and
  needed the frame dimmed almost into a backdrop to keep prose legible. The conversation has
  its own column now, so the veil is only deep enough to seat the arc panel and the action
  pills — the two things that still float here. What you are watching is the scene, not a
  dimmed photograph of it.

  IT IS THE BACKDROP FOR EVERY PANE OF GLASS. `backdrop()` hands out crops of a heavily
  downscaled copy of what was just painted; scaled back up, that is the blur behind the arc
  panel. It is computed once per frame here rather than once per panel, because the panels
  all sample the same image.

The first frame MATERIALIZES: it arrives blurred and slightly oversized and resolves into
place, rather than cutting in. A camera opening is a real event with a real duration, and
the two seconds it takes are more pleasant to watch than to be surprised by.
"""

from PySide6.QtCore import QRect, Qt, QTimer, Signal
from PySide6.QtGui import QImage, QLinearGradient, QPainter, QPixmap
from PySide6.QtWidgets import QSizePolicy, QWidget

from nexon.perception.viz import colorize_depth, draw_detections
from nexon.ui import motion, theme

REPAINT_MS = 33          # ~30 fps; the camera, not this, is the rate limit

# The backdrop is the painted frame at 1/BLUR_DIV scale. Smooth-scaling it back up to panel
# size is a box blur of radius ≈ BLUR_DIV screen pixels — deep enough to read as frosted
# glass, cheap enough to rebuild every frame.
BLUR_DIV = 14

# The bottom gradient runs up this fraction of the height, seating the action pills.
SCRIM_FRACTION = 0.42

# How much the first frame is oversized as it materializes. Small: this is a settle, not a
# zoom, and anything larger looks like the camera lurched.
REVEAL_SCALE = 0.06


def bgr_to_pixmap(bgr) -> QPixmap:
    """Copy a contiguous HxWx3 BGR array into a QPixmap."""
    h, w = bgr.shape[:2]
    # Format_BGR888 consumes the array's own byte order, so no cvtColor is needed.
    # .copy() detaches the image from `bgr`, which QImage does not keep alive.
    image = QImage(bgr.data, w, h, bgr.strides[0], QImage.Format_BGR888).copy()
    return QPixmap.fromImage(image)


class VideoView(QWidget):
    """Cover-cropped live view. Emits frame_size once known, and painted on every repaint."""

    frame_size = Signal(int, int)
    painted = Signal()

    def __init__(self, hub, parent=None):
        super().__init__(parent)
        self._hub = hub
        self._show_depth = False
        self._last = None
        self._announced = False

        self._display: QPixmap | None = None   # exactly what is on screen, veiled
        self._blur: QPixmap | None = None      # the same, at 1/BLUR_DIV — the backdrop
        self._soft: QPixmap | None = None      # the blur upscaled; only during the reveal

        self.setMinimumSize(360, 240)   # the splitter decides; this is only a floor
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setAutoFillBackground(False)

        # 0 while there is no frame, 1 once the first has resolved. Critically damped: the
        # camera coming up is not a gesture, so it must not overshoot.
        self._reveal = motion.Spring(0.0, damping=1.0, response=0.55, parent=self)
        self._reveal.value_changed.connect(lambda _v: self.update())

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(REPAINT_MS)

    def set_depth(self, on: bool) -> None:
        self._show_depth = bool(on)

    # ------------------------------------------------------------------ backdrop

    def backdrop(self, rect: QRect) -> QPixmap | None:
        """The blurred camera beneath `rect` (in this widget's coordinates), or None.

        Returns None before the first frame, which is not an error: a GlassPanel with no
        backdrop paints its tint and looks like frosted glass over the void, which is
        exactly what it is.
        """
        if self._blur is None or self._blur.isNull() or rect.isEmpty():
            return None

        clipped = rect.intersected(self.rect())
        if clipped.isEmpty():
            return None

        scale = self._blur.width() / max(1, self.width())
        source = QRect(
            int(clipped.left() * scale), int(clipped.top() * scale),
            max(1, int(clipped.width() * scale)), max(1, int(clipped.height() * scale)),
        ).intersected(self._blur.rect())
        if source.isEmpty():
            return None

        # Smooth upscale — this is where the blur actually happens.
        return self._blur.copy(source).scaled(
            rect.size(), Qt.IgnoreAspectRatio, Qt.SmoothTransformation)

    # --------------------------------------------------------------------- frames

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
            self._reveal.retarget(1.0)

        self._rebuild(bgr_to_pixmap(frame))
        self.update()
        self.painted.emit()             # the glass above must resample its backdrop

    def _rebuild(self, frame: QPixmap) -> None:
        """Cover-crop `frame` to the widget, veil it, and derive the backdrop from it."""
        if self.width() <= 0 or self.height() <= 0:
            return

        # KeepAspectRatioByExpanding fills the widget and overflows one axis; crop it out
        # from the centre, which is where the arm and the workpiece are.
        scaled = frame.scaled(self.size(), Qt.KeepAspectRatioByExpanding,
                              Qt.SmoothTransformation)
        x = max(0, (scaled.width() - self.width()) // 2)
        y = max(0, (scaled.height() - self.height()) // 2)
        display = scaled.copy(x, y, self.width(), self.height())

        painter = QPainter(display)
        painter.fillRect(display.rect(), theme.VEIL)
        gradient = QLinearGradient(0, display.height() * (1.0 - SCRIM_FRACTION),
                                   0, display.height())
        gradient.setColorAt(0.0, Qt.transparent)
        gradient.setColorAt(1.0, theme.SCRIM_BOTTOM)
        painter.fillRect(display.rect(), gradient)
        painter.end()

        self._display = display
        self._blur = display.scaled(max(1, self.width() // BLUR_DIV),
                                    max(1, self.height() // BLUR_DIV),
                                    Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
        # Only needed while the frame is still resolving; dropped once it has.
        self._soft = (self._blur.scaled(self.size(), Qt.IgnoreAspectRatio,
                                        Qt.SmoothTransformation)
                      if self._reveal.value < 0.999 else None)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._last = None               # force a rebuild at the new size on the next tick

    # ---------------------------------------------------------------------- paint

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), theme.VOID)

        if self._display is None:
            self._paint_waiting(painter)
            return

        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        reveal = max(0.0, min(1.0, self._reveal.value))

        if reveal < 0.999 and self._soft is not None:
            # Blurred and oversized underneath, sharp and settled on top. Cross-fading the
            # two IS the material arriving: it comes into focus and into place together,
            # which reads as a camera opening rather than a picture being pasted in.
            grow = REVEAL_SCALE * (1.0 - reveal)
            painter.setOpacity(1.0)
            painter.drawPixmap(self._grown(grow), self._soft)
            painter.setOpacity(reveal)
            painter.drawPixmap(self._grown(grow * 0.5), self._display)
            painter.setOpacity(1.0)
        else:
            painter.drawPixmap(0, 0, self._display)

    def _grown(self, amount: float) -> QRect:
        """The widget rect scaled up by `amount` about its centre."""
        dx = int(self.width() * amount / 2)
        dy = int(self.height() * amount / 2)
        return self.rect().adjusted(-dx, -dy, dx, dy)

    def _paint_waiting(self, painter: QPainter) -> None:
        painter.setFont(theme.font(theme.BODY))
        painter.setPen(theme.INK_TERTIARY)
        painter.drawText(self.rect(), Qt.AlignCenter, "waiting for camera…")

    # A background, not a control: clicks belong to whatever floats above it.
    def mousePressEvent(self, event) -> None:
        event.ignore()
