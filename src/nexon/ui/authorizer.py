"""The Qt surface through which a human consents to a real welding arc.

Controller.arm_live_arc blocks its caller until a person answers, and Qt only permits
widgets to be touched from the main (GUI) thread. Those two facts collide:

  * The agent, or a UI button handler run off-thread, calls arm_live_arc on a WORKER
    thread. The dialog must therefore be created on the main thread and the worker made
    to wait for it. That is what a BlockingQueuedConnection does, and it is safe here
    precisely because the main thread is free — it is not the caller.

  * If arm_live_arc is ever called ON the main thread, that same blocking emit would make
    the main thread wait for itself. Qt prints "Dead lock detected" and hangs. Callers are
    supposed to run arming off-thread (see ui.window), but a mistake there must not freeze
    the machine, so `ask` checks which thread it is on and shows the dialog directly.

The Controller holds no lock while this runs, so `disarm` stays callable throughout and
every precondition is re-checked after consent — a person may take a minute to answer.

ON THE LOOK OF IT. This is the one modal in the application, and it is modal because it
guards the one action that is destructive, irreversible, and physical. A scrim dims the
window behind it — the task in front of you is the only task — and the dialog keeps its
native frame rather than becoming a floating pane of glass: a consent prompt is not the
place to find out whether this machine's compositor draws translucency correctly. Cancel is
both the default and the escape key. Consent is never the accidental answer, and it is
never the fast one.
"""

from PySide6.QtCore import QObject, Qt, QThread, Signal, Slot
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import (QApplication, QDialog, QHBoxLayout, QLabel, QVBoxLayout,
                               QWidget)

from nexon.controller import LIVE_ARC_CONFIRMATION
from nexon.ui import motion, theme
from nexon.ui.controls import Pill

SCRIM = QColor(0, 0, 0, 150)


class Scrim(QWidget):
    """Dims the window while the modal is up. Fades, so the room darkens rather than blinks."""

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setGeometry(parent.rect())

    def paintEvent(self, _event) -> None:
        QPainter(self).fillRect(self.rect(), SCRIM)

    def reveal(self) -> None:
        self.setGeometry(self.parentWidget().rect())
        self.show()
        self.raise_()
        motion.spring_opacity(self, 1.0, response=0.28)

    def dismiss(self) -> None:
        spring = motion.spring_opacity(self, 0.0, response=0.24)
        spring.settled.connect(self.hide)


class ArcDialog(QDialog):
    """Ask, in words the operator can act on, whether to energize the torch."""

    def __init__(self, reason: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Arm live arc?")
        self.setModal(True)
        self.setMinimumWidth(theme.em(30))
        self.setStyleSheet(f"""
            QDialog {{ background:{theme.rgba(QColor(24, 24, 26))}; }}
            QLabel {{ background:transparent; }}
        """)

        headline = QLabel("The next seam trace will strike a REAL welding arc.")
        headline.setWordWrap(True)
        headline.setFont(theme.font(theme.TITLE, weight=700))
        headline.setStyleSheet(f"color:{theme.rgba(theme.ARC_LIVE)};")

        why = QLabel(reason or "No reason given.")
        why.setWordWrap(True)
        why.setFont(theme.font(theme.BODY))
        why.setStyleSheet(f"color:{theme.rgba(theme.INK)};")

        checks = QLabel(
            "Check the cell is clear, the operator is shielded, and the workpiece is "
            "clamped. Arming energizes the torch on the next pass.")
        checks.setWordWrap(True)
        checks.setFont(theme.font(theme.CAPTION))
        checks.setStyleSheet(f"color:{theme.rgba(theme.INK_SECONDARY)};")

        # Focus starts on Cancel. There is no "default button" that Enter activates —
        # see keyPressEvent — so the safe answer is simply the one already under the hand.
        self._cancel = Pill("Cancel", self)
        self._cancel.clicked.connect(self.reject)

        self._arm = Pill(LIVE_ARC_CONFIRMATION, self, kind="danger")
        self._arm.clicked.connect(self.accept)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        buttons.addWidget(self._cancel)
        buttons.addWidget(self._arm)

        pad = theme.em(1.4)
        column = QVBoxLayout(self)
        column.setContentsMargins(pad, pad, pad, pad)
        column.setSpacing(theme.em(0.8))
        column.addWidget(headline)
        column.addWidget(why)
        column.addWidget(checks)
        column.addSpacing(theme.em(0.4))
        column.addLayout(buttons)

        self._cancel.setFocus()

    def keyPressEvent(self, event) -> None:
        # Enter must not arm. The only key that commits is the one on the armed button,
        # reached by tabbing to it first — consent is a deliberate act, not a reflex.
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            if self._arm.hasFocus():
                self.accept()
            else:
                self.reject()
            return
        super().keyPressEvent(event)           # Escape rejects, via QDialog


class ArcAuthorizer(QObject):
    """Register `authorizer.ask` with Controller.set_arc_authorizer."""

    # reason, out: a one-element list the main thread writes the answer into. A signal is
    # used rather than QMetaObject.invokeMethod so the blocking connection is explicit.
    _asked = Signal(str, list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._asked.connect(self._show, Qt.BlockingQueuedConnection)

    def ask(self, reason: str) -> bool:
        """Controller-facing authorizer: returns True only if a human consents."""
        out: list = []
        if QThread.currentThread() is QApplication.instance().thread():
            # Already on the GUI thread — a blocking emit would wait on ourselves.
            self._show(reason, out)
        else:
            self._asked.emit(reason, out)   # blocks this thread until the dialog closes
        return bool(out and out[0])

    @Slot(str, list)
    def _show(self, reason: str, out: list) -> None:
        window = self.parent()
        scrim = Scrim(window) if isinstance(window, QWidget) else None
        if scrim is not None:
            scrim.reveal()
        try:
            dialog = ArcDialog(reason, window)
            granted = dialog.exec() == QDialog.Accepted
        finally:
            if scrim is not None:
                scrim.dismiss()
        out.append(granted)
