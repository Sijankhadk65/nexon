"""The machine's state, and the controls that are not conversation.

Almost everything in this window is a thing you ask for. These are the exceptions, and they
are exceptions for one reason each.

THE ARC. A live arc is the only state in this application whose cost of going unnoticed is
a burn, so it is the only thing allowed to shout: the panel floods red, the dot pulses, and
DISARM becomes the largest control on screen. `controller.py` states the invariant this
serves — "There is no state in which the arc cannot be shut off" — and a UI that renders
the arc anywhere other than in front of the operator's eyes breaks it just as surely as
code that refuses the call. Disarming is one press, always, with no confirmation: the
conservative answer must never be the slow one.

Note what is NOT here: a control that arms a live arc directly. `Arm live arc…` opens the
consent dialog, and consent is the only thing that arms. The agent can ask for an arc and
so can this pill; neither can grant it.

THE READOUT. Six fields permanently reading "—" is a panel pretending to be information.
This strip says the operation speed (always — it is the number that decides whether a weld
is a weld), and then only what has actually changed: a weave if one is set, axes if any are
locked, movement while the arm is moving, a transport speed if it is no longer what it was
when the window opened. Everything else, ask.

THE ACTIONS. Detect, dry-trace and depth-view survive as buttons because they are what an
operator does over and over while setting up, and a conversation is a poor place to put a
thing you do forty times an hour. They sit above the composer, near the words that would
otherwise have to request them.
"""

from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import QHBoxLayout, QLabel, QSizePolicy, QVBoxLayout, QWidget

from nexon.ui import motion, theme
from nexon.ui.controls import Pill
from nexon.ui.glass import GlassPanel

# Arc states, as the operator experiences them rather than as the flags spell them.
OFF, DRY, LIVE = "off", "dry", "live"

_ARC_TEXT = {
    OFF: "Welding off",
    DRY: "Dry weld",
    LIVE: "LIVE ARC ARMED",
}
_ARC_COLOR = {
    OFF: theme.INK_SECONDARY,
    DRY: theme.WELD_DRY,
    LIVE: theme.ARC_LIVE,
}


class ArcDot(QWidget):
    """A filled dot with a halo. The halo breathes only when the arc is live."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._color = QColor(theme.INK_SECONDARY)
        self._halo = 0.0
        size = theme.em(0.95)
        self.setFixedSize(size * 3, size * 3)

        self._pulse = motion.Pulse(0.0, 1.0, period_ms=1500, parent=self)
        self._pulse.valueChanged.connect(self._on_pulse)

    def _on_pulse(self, value) -> None:
        self._halo = float(value)
        self.update()

    def set_state(self, state: str) -> None:
        self._color = QColor(_ARC_COLOR[state])
        if state == LIVE:
            self._pulse.begin()
        else:
            self._pulse.end()
        self.update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(Qt.NoPen)

        centre = QRectF(self.rect()).center()
        core = self.width() / 3.0

        if self._halo > 0.01:
            halo = QColor(self._color)
            halo.setAlphaF(0.30 * self._halo)
            reach = core * (0.7 + 0.9 * self._halo)
            painter.setBrush(halo)
            painter.drawEllipse(centre, reach, reach)

        painter.setBrush(self._color)
        painter.drawEllipse(centre, core / 2, core / 2)


class MachinePanel(GlassPanel):
    """Arc state, the quiet readout, and the arc controls — one cluster, one subject."""

    weld_toggled = Signal(bool)
    arm_requested = Signal()
    disarm_requested = Signal()

    def __init__(self, parent=None):
        # A heavier material than the composer: this separates a structural region rather
        # than floating over one, and heavy materials are what carry structure.
        super().__init__(parent, radius=theme.RADIUS_PANEL, tint=theme.GLASS_DEEP, shadow=14)
        self.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
        self._baseline_transport = None
        self._state = OFF

        self._dot = ArcDot(self)
        self._arc = QLabel(_ARC_TEXT[OFF], self)
        self._arc.setFont(theme.font(theme.TITLE, weight=600))

        self._facts = QLabel("—", self)
        self._facts.setFont(theme.font(theme.CAPTION))
        self._facts.setStyleSheet(
            f"color:{theme.rgba(theme.INK_SECONDARY)}; background:transparent;")

        self._enable = Pill("Enable dry weld", self)
        self._enable.clicked.connect(lambda: self.weld_toggled.emit(True))

        self._disable = Pill("Turn welding off", self)
        self._disable.clicked.connect(lambda: self.weld_toggled.emit(False))

        self._arm = Pill("Arm live arc…", self, kind="primary")
        self._arm.clicked.connect(self.arm_requested.emit)

        self._disarm = Pill("DISARM", self, kind="danger")
        self._disarm.clicked.connect(self.disarm_requested.emit)

        margin = self.content_margins()
        pad = theme.em(0.75)
        column = QVBoxLayout(self)
        column.setContentsMargins(margin + pad, margin + pad, margin + pad, margin + pad)
        column.setSpacing(theme.em(0.5))

        headline = QHBoxLayout()
        headline.setSpacing(theme.em(0.2))
        headline.addWidget(self._dot)
        headline.addWidget(self._arc)
        headline.addStretch(1)
        column.addLayout(headline)
        column.addWidget(self._facts)

        self._actions = QHBoxLayout()
        self._actions.setSpacing(theme.em(0.4))
        for pill in (self._disarm, self._arm, self._enable, self._disable):
            self._actions.addWidget(pill)
        self._actions.addStretch(1)
        column.addLayout(self._actions)

    # ---------------------------------------------------------------- rendering

    def render_state(self, state) -> None:
        """Called on every controller transition, whoever caused it — human or agent."""
        if self._baseline_transport is None:
            self._baseline_transport = state.velocity["transport_pct"]

        arc = LIVE if state.live_armed else DRY if state.dry_weld else OFF
        self._state = arc
        self._dot.set_state(arc)
        self._arc.setText(_ARC_TEXT[arc])
        self._arc.setStyleSheet(
            f"color:{theme.rgba(theme.INK if arc == LIVE else _ARC_COLOR[arc])};"
            "background:transparent;")
        self._tint = theme.ARC_LIVE_DIM if arc == LIVE else theme.GLASS_DEEP

        self._facts.setText(self._describe(state))

        # Only disarming stays live during a pass. Everything else would change a plan
        # already in flight, and the controller would refuse it anyway.
        self._disarm.setVisible(arc == LIVE)
        self._arm.setVisible(arc == DRY)
        self._enable.setVisible(arc == OFF)
        self._disable.setVisible(arc == DRY)

        for pill in (self._arm, self._enable, self._disable):
            pill.setEnabled(not state.busy)
        self._disarm.setEnabled(True)

        self.adjustSize()
        self.update()

    def _describe(self, state) -> str:
        """The readout: the speed, plus only what has changed from where it started."""
        velocity = state.velocity
        physical = velocity["mode"] == "physical"
        speed = velocity["physical_mm_s"] if physical else velocity["percentage"]
        facts = [f"{speed} {'mm/s' if physical else '%'}"]

        weave = state.weave
        if weave.get("enabled"):
            facts.append(f"weave {weave.get('pattern', '?')} {weave.get('amplitude_mm', '?')}mm")
        if state.locked_axes:
            facts.append(f"{', '.join(state.locked_axes)} locked")
        if velocity["transport_pct"] != self._baseline_transport:
            facts.append(f"transport {velocity['transport_pct']}%")
        if state.busy:
            facts.append("moving")
        return "  ·  ".join(facts)


class ActionBar(QWidget):
    """The handful of things an operator does far too often to have to ask for."""

    detect_seam = Signal()
    dry_trace = Signal()
    depth_toggled = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)

        self._detect = Pill("Detect seam", self)
        self._detect.clicked.connect(self.detect_seam.emit)

        self._trace = Pill("Follow seam (dry)", self)
        self._trace.clicked.connect(self.dry_trace.emit)

        self._depth = Pill("Depth view", self, checkable=True)
        self._depth.toggled.connect(self.depth_toggled.emit)

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(theme.em(0.4))
        for pill in (self._detect, self._trace, self._depth):
            row.addWidget(pill)
        row.addStretch(1)

    def render_state(self, state) -> None:
        # Depth is a way of looking, not a way of moving — it stays live during a pass.
        for pill in (self._detect, self._trace):
            pill.setEnabled(not state.busy)
