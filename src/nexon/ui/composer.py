"""Where the operator speaks: a text field, a mic, and one button that changes its mind.

RESPOND ON PRESS, NOT ON RELEASE. Every button here highlights and shrinks the instant the
pointer goes down, driven by a spring rather than a transition so a quick double-press does
not queue up two animations. The moment feedback waits for the click to complete, the whole
interface stops feeling direct — and this is the surface the operator touches most.

THE MIC IS HELD, NOT TOGGLED. Push-to-talk means recording starts on press and ends on
release; there is no state to forget you left on, which matters when the thing listening is
wired to a welding arm. The field's placeholder narrates the whole way through — listening,
then transcribing — because a mic with no feedback is indistinguishable from a broken one.

SEND BECOMES STOP. While a reply is in flight the primary button turns into a stop control
in the same place. A reply that cannot be interrupted is a reply that owns the operator for
as long as it likes; making them hunt for a different control to escape is the same failure
with extra steps. Enter sends, Shift+Enter breaks a line, Escape stops.

The panel is the same glass as the arc panel opposite, but it sits on the conversation's
solid column rather than on the camera, so it is a tint and a lip with nothing to blur —
no backdrop source is installed on it. See ui/glass.py: a panel with no backdrop degrades
to a solid, it never fails.
"""

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QHBoxLayout, QTextEdit

from nexon.ui import theme
from nexon.ui.controls import PILL_GLASS, IconButton
from nexon.ui.glass import GlassPanel

# The field grows with what is typed, up to a point; past it, the field scrolls.
MIN_LINES = 1
MAX_LINES = 5

# The stop button's fill: lighter than the field, but not the accent — stopping is not the
# thing we want the eye drawn to, it is the thing we want reachable.
STOP_FILL = QColor(255, 255, 255, 40)


class Input(QTextEdit):
    """A field that grows with its content and sends on Enter."""

    submitted = Signal()
    escaped = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QTextEdit.NoFrame)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setFont(theme.font(theme.BODY))
        self.setAcceptRichText(False)
        self.setPlaceholderText("ask nexon…")
        self.setStyleSheet(f"""
            QTextEdit {{
                background:transparent; border:none;
                color:{theme.rgba(theme.INK)};
                selection-background-color:{theme.rgba(theme.ACCENT)};
            }}
        """)
        self.document().setDocumentMargin(0)
        self.textChanged.connect(self._sync_height)
        self._sync_height()

    def _line_height(self) -> int:
        return max(1, self.fontMetrics().lineSpacing())

    def _sync_height(self) -> None:
        doc = self.document()
        doc.setTextWidth(max(1, self.viewport().width()))
        wanted = doc.size().height()
        low = self._line_height() * MIN_LINES
        high = self._line_height() * MAX_LINES
        self.setFixedHeight(int(max(low, min(high, wanted))) + theme.em(0.2))
        self.setVerticalScrollBarPolicy(
            Qt.ScrollBarAsNeeded if wanted > high else Qt.ScrollBarAlwaysOff)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._sync_height()

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key_Escape:
            self.escaped.emit()
            return
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            if event.modifiers() & Qt.ShiftModifier:
                super().keyPressEvent(event)    # deliberate newline
                return
            self.submitted.emit()
            return
        super().keyPressEvent(event)

    def take(self) -> str:
        text = self.toPlainText().strip()
        self.clear()
        return text


class Composer(GlassPanel):
    """The glass bar at the bottom: mic, field, and the send/stop button."""

    submitted = Signal(str)
    stop_requested = Signal()
    stop_listening = Signal()
    hold_started = Signal()
    hold_ended = Signal()

    def __init__(self, parent=None):
        super().__init__(parent, radius=theme.RADIUS_PANEL, tint=theme.FIELD, shadow=10)
        self._busy = False
        self._hands_free = False
        self._listening_active = False

        self._mic = IconButton("mic", self)
        self._mic.pressed.connect(self.hold_started.emit)
        self._mic.released.connect(self.hold_ended.emit)
        self._mic.setToolTip("Hold to talk")

        self._input = Input(self)
        self._input.submitted.connect(self._on_submit)
        self._input.escaped.connect(self._on_escape)

        self._action = IconButton("send", self, fill=theme.ACCENT, ink=theme.INK)
        self._action.clicked.connect(self._on_action)
        self._action.setToolTip("Send  ·  Enter")

        margin = self.content_margins()
        pad = theme.em(0.5)
        row = QHBoxLayout(self)
        row.setContentsMargins(margin + pad, margin + pad, margin + pad, margin + pad)
        row.setSpacing(theme.em(0.6))
        row.addWidget(self._mic, alignment=Qt.AlignBottom)
        row.addWidget(self._input, stretch=1, alignment=Qt.AlignBottom)
        row.addWidget(self._action, alignment=Qt.AlignBottom)

    # -------------------------------------------------------------------- actions

    def _on_submit(self) -> None:
        if self._busy:
            return                              # Enter during a reply is not a queue
        text = self._input.take()
        if text:
            self.submitted.emit(text)

    def _on_escape(self) -> None:
        if self._busy:
            self.stop_requested.emit()
        elif self._listening_active:
            self.stop_listening.emit()

    def _on_action(self) -> None:
        if self._busy:
            self.stop_requested.emit()
        elif self._listening_active:
            self.stop_listening.emit()
        else:
            self._on_submit()

    # --------------------------------------------------------------------- states

    def set_busy(self, busy: bool) -> None:
        """A reply is in flight: send becomes stop, in the same place."""
        if busy == self._busy:
            return
        self._busy = busy
        self._refresh_action()

    def set_listening_active(self, active: bool) -> None:
        """Hands-free is armed/listening: the primary button becomes a stop, in the same place.

        This is the one always-visible way out of a hands-free session — the thing push-to-talk
        never needed, because it stopped the instant you let go of the mic.
        """
        if active == self._listening_active:
            return
        self._listening_active = active
        self._refresh_action()

    def _refresh_action(self) -> None:
        """Send when there's nothing to stop; stop when a reply or a listen loop is running."""
        if self._busy or self._listening_active:
            self._action.set_glyph("stop")
            self._action.set_colors(STOP_FILL, theme.INK)
            self._action.setToolTip("Stop  ·  Esc")
        else:
            self._action.set_glyph("send")
            self._action.set_colors(theme.ACCENT, theme.INK)
            self._action.setToolTip("Send  ·  Enter")

    def set_listening(self, listening: bool) -> None:
        self._input.setPlaceholderText("listening…" if listening else self._idle_text())
        self._mic.set_colors(theme.ARC_LIVE if listening else PILL_GLASS, theme.INK)
        self._input.setReadOnly(listening)

    def set_transcribing(self, busy: bool) -> None:
        if busy:
            self._input.setPlaceholderText("transcribing…")
        else:
            self._input.setPlaceholderText(self._idle_text())
            self._input.setReadOnly(False)

    def _idle_text(self) -> str:
        """The placeholder shown when nothing is in flight — a wake-word hint when armed."""
        return "say the wake word…" if self._hands_free else "ask nexon…"

    def set_hands_free(self, on: bool) -> None:
        """Reflect that the wake word is (un)armed, so the resting field says which mode it's in."""
        self._hands_free = on
        # Only rewrite the placeholder if the field is at rest; a live listening/transcribing
        # state owns it until it clears, and set_*() will pick up _idle_text() when it does.
        if not self._input.isReadOnly() and not self._busy:
            self._input.setPlaceholderText(self._idle_text())

    def set_voice_available(self, available: bool) -> None:
        self._mic.setVisible(available)
        if not available:
            self._mic.setToolTip("Set an ElevenLabs API key in Settings to talk")

    def set_text(self, text: str) -> None:
        self._input.setPlainText(text)
        cursor = self._input.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        self._input.setTextCursor(cursor)

    def focus(self) -> None:
        self._input.setFocus()
