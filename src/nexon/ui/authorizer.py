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
"""

from PySide6.QtCore import QObject, Qt, QThread, Signal, Slot
from PySide6.QtWidgets import QApplication, QMessageBox

from nexon.controller import LIVE_ARC_CONFIRMATION


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
        box = QMessageBox(self.parent())
        box.setIcon(QMessageBox.Warning)
        box.setWindowTitle("Arm live arc?")
        box.setText("<b>The next seam trace will strike a REAL welding arc.</b>")
        box.setInformativeText(
            f"<p>{reason or 'No reason given.'}</p>"
            "<p>Check the cell is clear, the operator is shielded, and the workpiece is "
            "clamped. Choosing <i>Arm</i> energizes the torch on the next pass.</p>")
        arm = box.addButton(f"{LIVE_ARC_CONFIRMATION}", QMessageBox.AcceptRole)
        cancel = box.addButton("Cancel", QMessageBox.RejectRole)
        # Default and escape both land on Cancel: consent is never the accidental answer.
        box.setDefaultButton(cancel)
        box.setEscapeButton(cancel)
        box.exec()
        out.append(box.clickedButton() is arm)
