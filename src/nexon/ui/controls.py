"""Buttons: a round icon and a text pill, drawn rather than themed.

Qt's stylesheet engine can make a button any colour, but it cannot make it respond on
pointer-down with a spring, and it cannot antialias a translucent fill over live video the
way `QPainter` can. Both matter here, so both controls are painted.

Everything in this module obeys the same three rules:

  FEEDBACK STARTS ON PRESS. `mousePressEvent` retargets the scale spring before the click
  has been decided, and the springs retarget rather than restart, so a fast double-press
  continues from wherever the last one got to instead of snapping back to 1.0 first.

  WEIGHT ENCODES MEANING. `normal` is glass; `primary` is a saturated fill; `danger` is the
  arc red used nowhere else. A pill's kind is a claim about consequence, not a colour
  preference.

  A CONTROL THAT ONLY ANSWERS TO A MOUSE IS HALF-BUILT. Both subclass `QAbstractButton`, so
  focus, Space/Enter activation, checked state and accessibility come from Qt rather than
  being reimplemented badly. The focus ring is drawn explicitly because there is no
  platform style underneath to draw it for us.
"""

from PySide6.QtCore import QRectF, QSize, Qt
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QAbstractButton

from nexon.ui import motion, theme

# How far a control shrinks under the finger. Enough to feel, not enough to see as motion.
PRESS_SCALE = 0.92

# Pill fills, by consequence.
PILL_GLASS = QColor(255, 255, 255, 24)
PILL_GLASS_ON = QColor(255, 255, 255, 46)


class _Pressable(QAbstractButton):
    """Shared press/hover springs. Not used directly."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setCursor(Qt.PointingHandCursor)
        self.setFocusPolicy(Qt.StrongFocus)

        self._scale = motion.Spring(1.0, damping=1.0, response=0.20, parent=self)
        self._scale.value_changed.connect(lambda _v: self.update())
        self._glow = motion.Spring(0.0, damping=1.0, response=0.22, parent=self)
        self._glow.value_changed.connect(lambda _v: self.update())

    def enterEvent(self, event) -> None:
        if self.isEnabled():
            self._glow.retarget(1.0)
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:
        self._glow.retarget(0.0)
        super().leaveEvent(event)

    def mousePressEvent(self, event) -> None:
        self._scale.retarget(PRESS_SCALE)      # before the click is committed
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        self._scale.retarget(1.0)
        super().mouseReleaseEvent(event)

    def _apply_scale(self, painter: QPainter) -> None:
        scale = self._scale.value
        painter.translate(self.width() / 2, self.height() / 2)
        painter.scale(scale, scale)
        painter.translate(-self.width() / 2, -self.height() / 2)


class IconButton(_Pressable):
    """A round button whose glyph is a painted path. No icon assets anywhere in the app."""

    GLYPHS = ("send", "stop", "mic", "more")

    def __init__(self, glyph: str, parent=None, *, diameter: int | None = None,
                 fill: QColor | None = None, ink: QColor | None = None):
        super().__init__(parent)
        self._glyph = glyph
        self._fill = fill if fill is not None else PILL_GLASS
        self._ink = ink if ink is not None else theme.INK
        size = diameter or theme.em(2.4)
        self.setFixedSize(QSize(size, size))

    def set_glyph(self, glyph: str) -> None:
        self._glyph = glyph
        self.update()

    def set_colors(self, fill: QColor, ink: QColor) -> None:
        self._fill, self._ink = fill, ink
        self.update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        self._apply_scale(painter)

        fill = QColor(self._fill)
        if not self.isEnabled():
            fill.setAlpha(int(fill.alpha() * 0.4))
        elif self._glow.value:
            fill = fill.lighter(100 + int(16 * self._glow.value))

        painter.setPen(Qt.NoPen)
        painter.setBrush(fill)
        painter.drawEllipse(QRectF(self.rect()))

        ink = QColor(self._ink)
        if not self.isEnabled():
            ink.setAlpha(90)
        painter.setPen(QPen(ink, 2.0, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        painter.setBrush(Qt.NoBrush)

        # Glyphs are authored in a 24×24 box and scaled to whatever size the button is.
        painter.save()
        painter.scale(self.width() / 24.0, self.height() / 24.0)
        getattr(self, f"_draw_{self._glyph}")(painter, ink)
        painter.restore()

        if self.hasFocus():
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(theme.ACCENT, 2.0))
            painter.drawEllipse(QRectF(self.rect()).adjusted(1, 1, -1, -1))

    # ------------------------------------------------------------------- glyphs

    def _draw_send(self, painter: QPainter, _ink: QColor) -> None:
        path = QPainterPath()
        path.moveTo(12, 18.5)
        path.lineTo(12, 6)
        path.moveTo(6.5, 11.5)
        path.lineTo(12, 6)
        path.lineTo(17.5, 11.5)
        painter.drawPath(path)

    def _draw_stop(self, painter: QPainter, ink: QColor) -> None:
        painter.setPen(Qt.NoPen)
        painter.setBrush(ink)
        path = QPainterPath()
        path.addRoundedRect(QRectF(7.5, 7.5, 9, 9), 2.0, 2.0)
        painter.drawPath(path)

    def _draw_mic(self, painter: QPainter, ink: QColor) -> None:
        capsule = QPainterPath()
        capsule.addRoundedRect(QRectF(9, 3, 6, 10.5), 3.0, 3.0)
        painter.fillPath(capsule, ink)
        painter.drawArc(QRectF(5.5, 6.5, 13, 12), 180 * 16, 180 * 16)   # the cradle
        painter.drawLine(12, 18.5, 12, 21)                              # the stem

    def _draw_more(self, painter: QPainter, ink: QColor) -> None:
        painter.setPen(Qt.NoPen)
        painter.setBrush(ink)
        for x in (6.5, 12, 17.5):
            painter.drawEllipse(QRectF(x - 1.3, 10.7, 2.6, 2.6))


class Pill(_Pressable):
    """A rounded text button. `kind` states what pressing it costs, and picks the colour."""

    def __init__(self, text: str, parent=None, *, kind: str = "normal",
                 checkable: bool = False):
        super().__init__(parent)
        self._kind = kind
        # Metrics before text: setText() measures the label against the font and padding,
        # so both must already exist when it runs.
        self._pad = theme.em(0.95)
        self.setFont(theme.font(theme.CAPTION, weight=600))
        self.setFixedHeight(theme.em(2.05))
        self.setCheckable(checkable)
        self.setText(text)

    def setText(self, text: str) -> None:      # noqa: N802 — Qt's name
        super().setText(text)
        self._sync_width()

    def set_kind(self, kind: str) -> None:
        self._kind = kind
        self.update()

    def _sync_width(self) -> None:
        self.setFixedWidth(self.fontMetrics().horizontalAdvance(self.text()) + self._pad * 2)

    def _fill(self) -> QColor:
        if self._kind == "danger":
            return QColor(theme.ARC_LIVE)
        if self._kind == "primary":
            return QColor(theme.ACCENT)
        return QColor(PILL_GLASS_ON if self.isChecked() else PILL_GLASS)

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        self._apply_scale(painter)

        rect = QRectF(self.rect())
        radius = rect.height() / 2

        fill = self._fill()
        if not self.isEnabled():
            fill.setAlpha(int(fill.alpha() * 0.35) if fill.alpha() < 255 else 70)
        elif self._glow.value:
            fill = fill.lighter(100 + int(12 * self._glow.value))

        path = QPainterPath()
        path.addRoundedRect(rect.adjusted(0.5, 0.5, -0.5, -0.5), radius, radius)
        painter.fillPath(path, fill)

        if self._kind == "normal":
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(theme.stroke(), 1.0))
            painter.drawPath(path)

        ink = QColor(theme.INK)
        if not self.isEnabled():
            ink.setAlpha(90)
        painter.setPen(ink)
        painter.setFont(self.font())
        painter.drawText(rect, Qt.AlignCenter, self.text())

        if self.hasFocus():
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(theme.ACCENT, 2.0))
            painter.drawPath(path)

    def sizeHint(self) -> QSize:
        return QSize(self.fontMetrics().horizontalAdvance(self.text()) + self._pad * 2,
                     theme.em(2.05))
