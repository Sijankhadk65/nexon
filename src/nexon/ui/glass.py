"""The translucent material every floating surface in the window is made of.

CSS gets `backdrop-filter: blur()` for free. Qt does not, so the blur is produced where the
pixels already are: `ui/video.py` keeps a heavily downscaled copy of exactly what it just
painted, and a panel asks for the crop of it that lies beneath itself. Scaling that crop
back up with smooth interpolation IS a box blur — one cheap scale per frame instead of a
real convolution, and at this radius the difference is invisible. `set_backdrop_source`
installs the callable that performs the lookup; without one, the panel is simply a tinted
pane, which is what happens before the camera opens and in the settings sheet.

Three rules from Apple's material guidance are enforced here rather than left to callers:

  NEVER STACK GLASS ON GLASS. A translucent surface on another translucent surface loses
  its edge and the text on it loses contrast. `GlassPanel` therefore samples the VIDEO, not
  its parent, and the one surface that sits on top of another (a message bubble on the
  transcript) is opaque enough to need no backdrop at all — see `conversation.py`.

  BIGGER SURFACES READ AS THICKER. A large panel gets a deeper shadow and a larger radius
  than a small pill. That is the whole reason `shadow` and `radius` are per-instance: a
  wide composer and a small status chip are the same material at different weights, and if
  they carried the same shadow one of them would look wrong.

  THE TOP EDGE CATCHES THE LIGHT. A one-pixel bright arc along the upper curve is what
  separates "a pane of glass" from "a grey rectangle". It costs a stroke and it is the
  single detail that does most of the work.

The shadow is painted inside the widget's own bounds — a widget cannot draw outside itself —
so every panel reserves `shadow` pixels of margin on each side. `content_margins()` exists
so layouts add their padding to that margin instead of fighting it.
"""

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QWidget

from nexon.ui import theme

# How far the shadow reaches, and how dark it starts. Over live video — busy, high-contrast,
# constantly changing — a panel needs real separation or its edge dissolves into the frame.
_SHADOW_ALPHA = 46
_SHADOW_DROP = 2          # px downward: light comes from above, as it does everywhere else


class GlassPanel(QWidget):
    """A rounded, translucent surface with a blurred backdrop, a bright lip, and a shadow.

    `radius`, `tint` and `shadow` are the three dials. Leave them alone unless the surface
    genuinely differs in weight from the ones around it.
    """

    def __init__(self, parent: QWidget | None = None, *, radius: int = theme.RADIUS_PANEL,
                 tint: QColor | None = None, shadow: int = 12, backdrop: bool = True):
        super().__init__(parent)
        self._radius = radius
        self._tint = tint if tint is not None else theme.GLASS
        self._shadow = 0 if theme.more_contrast() else shadow
        self._wants_backdrop = backdrop
        self._backdrop_source = None
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_NoSystemBackground)

    # ------------------------------------------------------------------ backdrop

    def set_backdrop_source(self, fn) -> None:
        """Install `fn(widget, local_rect) -> QPixmap | None`, called on every repaint.

        Propagated to nested GlassPanels so the window installs it once at the root. A
        panel that returns None simply paints its tint — the material degrades to a solid,
        it never fails.
        """
        self._backdrop_source = fn
        for child in self.findChildren(GlassPanel):
            child._backdrop_source = fn          # noqa: SLF001 — same class

    def _backdrop(self, rect):
        if not self._wants_backdrop or self._backdrop_source is None:
            return None
        if theme.reduced_transparency() or theme.more_contrast():
            return None                          # the fill is opaque; blurring is wasted work
        return self._backdrop_source(self, rect)

    # -------------------------------------------------------------------- layout

    def content_margins(self) -> int:
        """The inset a layout must respect so content clears the shadow."""
        return self._shadow

    def surface_rect(self) -> QRectF:
        m = self._shadow
        return QRectF(self.rect().adjusted(m, m, -m, -m))

    def _path(self) -> QPainterPath:
        radius = min(self._radius, self.surface_rect().height() / 2)
        path = QPainterPath()
        path.addRoundedRect(self.surface_rect(), radius, radius)
        return path

    # --------------------------------------------------------------------- paint

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        path = self._path()

        self._paint_shadow(painter)

        backdrop = self._backdrop(self.surface_rect().toRect())
        if backdrop is not None and not backdrop.isNull():
            painter.save()
            painter.setClipPath(path)
            painter.drawPixmap(self.surface_rect(), backdrop, QRectF(backdrop.rect()))
            painter.restore()

        painter.fillPath(path, theme.surface(self._tint))
        self._paint_edges(painter, path)

    def _paint_shadow(self, painter: QPainter) -> None:
        if self._shadow <= 0:
            return
        rect = self.surface_rect().translated(0, _SHADOW_DROP)
        painter.setPen(Qt.NoPen)
        for i in range(self._shadow, 0, -1):
            # Quadratic falloff: linear layers band visibly, and the eye reads banding as
            # a rendering fault rather than as a shadow.
            fade = (1.0 - i / self._shadow) ** 2
            color = QColor(0, 0, 0, int(_SHADOW_ALPHA * fade))
            grown = rect.adjusted(-i, -i, i, i)
            radius = self._radius + i
            path = QPainterPath()
            path.addRoundedRect(grown, radius, radius)
            painter.fillPath(path, color)

    def _paint_edges(self, painter: QPainter, path: QPainterPath) -> None:
        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(theme.stroke(), 1.0))
        painter.drawPath(path)

        if theme.more_contrast():
            return                                # the border already does this job

        # The lip. Clipped to the upper third so the highlight sits where light would fall,
        # not ringing the whole shape like a selection outline.
        rect = self.surface_rect()
        painter.save()
        painter.setClipRect(QRectF(rect.left(), rect.top(),
                                   rect.width(), rect.height() / 3.0))
        painter.setPen(QPen(theme.GLASS_EDGE, 1.2))
        painter.drawPath(path)
        painter.restore()
