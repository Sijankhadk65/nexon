"""The operator window: live view, machine state, and the arc controls.

Three threading rules hold this together, and breaking any of them hangs the arm:

  1. NOTHING that touches the controller runs on the GUI thread. arm_live_arc blocks for a
     human; run_motion blocks for a weld pass. Both go through `_run_async`.
  2. Controller.subscribe callbacks arrive on whatever thread caused the transition — the
     motion worker, the agent, this window's own workers. They are funnelled through a
     Qt signal, which marshals them onto the GUI thread before any widget is touched.
  3. The video widget paints on its own timer and never blocks on the camera.

The window is a CLIENT of the controller, exactly like the agent is. It reads state only
from the snapshots it is handed, never from robot.py's globals — so when Claude arms the
arc mid-conversation, this window lights up without being told about Claude at all.
"""

import json
import threading

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (QCheckBox, QGridLayout, QGroupBox, QHBoxLayout, QLabel,
                               QMainWindow, QMessageBox, QPushButton, QVBoxLayout, QWidget)

from nexon.controller import get_controller
from nexon.ui.authorizer import ArcAuthorizer
from nexon.ui.video import VideoView

ARC_ARMED_CSS = "background:#b00020; color:white; padding:8px; border-radius:4px;"
ARC_DRY_CSS = "background:#2e5c1f; color:white; padding:8px; border-radius:4px;"
ARC_OFF_CSS = "background:#333; color:#bbb; padding:8px; border-radius:4px;"


class MainWindow(QMainWindow):
    # Controller transitions and worker results arrive off-thread; these carry them home.
    _state_changed = Signal(object)
    _result = Signal(str, object)

    def __init__(self, hub):
        super().__init__()
        self.setWindowTitle("nex-ON")
        self._hub = hub
        self._ctl = get_controller()

        self._authorizer = ArcAuthorizer(self)
        self._ctl.set_arc_authorizer(self._authorizer.ask)

        self._build()

        # Queued by default because the emitters are other threads.
        self._state_changed.connect(self._render_state)
        self._result.connect(self._render_result)
        self._ctl.subscribe(self._state_changed.emit)
        self._render_state(self._ctl.snapshot())

    # ------------------------------------------------------------------ layout

    def _build(self) -> None:
        self._video = VideoView(self._hub)

        self._arc = QLabel("WELDING OFF")
        self._arc.setAlignment(Qt.AlignCenter)
        self._arc.setFont(QFont("", 14, QFont.Bold))
        self._arc.setStyleSheet(ARC_OFF_CSS)

        self._weld_on = QCheckBox("Enable welding (dry)")
        self._weld_on.clicked.connect(self._on_weld_toggled)

        self._arm = QPushButton("ARM LIVE ARC…")
        self._arm.clicked.connect(self._on_arm)
        self._disarm = QPushButton("Disarm")
        self._disarm.clicked.connect(self._on_disarm)

        self._detect = QPushButton("Detect seam")
        self._detect.clicked.connect(self._on_detect_seam)
        self._trace = QPushButton("Follow seam (dry run)")
        self._trace.clicked.connect(self._on_dry_trace)

        self._depth = QCheckBox("Depth view")
        self._depth.toggled.connect(self._video.set_depth)

        state = QGroupBox("Machine")
        grid = QGridLayout(state)
        self._fields = {}
        for row, key in enumerate(("welding", "weave", "operation speed",
                                   "transport speed", "locked axes", "motion")):
            grid.addWidget(QLabel(f"{key}:"), row, 0)
            value = QLabel("—")
            value.setTextInteractionFlags(Qt.TextSelectableByMouse)
            grid.addWidget(value, row, 1)
            self._fields[key] = value

        arc_box = QGroupBox("Arc")
        arc_layout = QVBoxLayout(arc_box)
        arc_layout.addWidget(self._arc)
        arc_layout.addWidget(self._weld_on)
        arc_layout.addWidget(self._arm)
        arc_layout.addWidget(self._disarm)

        seam_box = QGroupBox("Seam")
        seam_layout = QVBoxLayout(seam_box)
        seam_layout.addWidget(self._detect)
        seam_layout.addWidget(self._trace)

        side = QVBoxLayout()
        side.addWidget(state)
        side.addWidget(arc_box)
        side.addWidget(seam_box)
        side.addWidget(self._depth)
        side.addStretch(1)

        root = QHBoxLayout()
        root.addWidget(self._video, stretch=3)
        panel = QWidget()
        panel.setLayout(side)
        panel.setFixedWidth(300)
        root.addWidget(panel)

        central = QWidget()
        central.setLayout(root)
        self.setCentralWidget(central)
        self.statusBar().showMessage("ready")

    # ------------------------------------------------------------- threading

    def _run_async(self, label, fn, *args, **kwargs) -> None:
        """Run a blocking controller/tool call off the GUI thread; report via _result.

        Every controller entry point can block — on the motion worker, or on a human
        answering the arc dialog. Calling one from a button handler would freeze the
        window, and in the arm case would deadlock against the dialog it is waiting for.
        """
        def work():
            try:
                self._result.emit(label, fn(*args, **kwargs))
            except Exception as exc:  # noqa: BLE001 — surfaced in the status bar
                self._result.emit(label, exc)

        self.statusBar().showMessage(f"{label}…")
        threading.Thread(target=work, name=f"nexon-ui-{label}", daemon=True).start()

    # ---------------------------------------------------------------- actions

    def _on_weld_toggled(self, checked: bool) -> None:
        self._run_async("set weld", self._ctl.set_weld, enabled=checked)

    def _on_arm(self) -> None:
        # Off-thread on purpose: arm_live_arc blocks until the dialog is answered, and the
        # dialog needs this thread to run its event loop.
        self._run_async("arm", self._ctl.arm_live_arc, "armed from the operator console")

    def _on_disarm(self) -> None:
        self._run_async("disarm", self._ctl.disarm)

    def _on_detect_seam(self) -> None:
        from nexon.agent import tools
        self._run_async("detect seam", tools.detect_seam.invoke, {})

    def _on_dry_trace(self) -> None:
        from nexon.agent import tools
        self._run_async("dry trace", tools.follow_seam.invoke, {"dry_run": True})

    # ---------------------------------------------------------------- rendering

    def _render_state(self, state) -> None:
        weld, weave, vel = state.weld, state.weave, state.velocity

        if state.live_armed:
            self._arc.setText("LIVE ARC ARMED")
            self._arc.setStyleSheet(ARC_ARMED_CSS)
        elif state.dry_weld:
            self._arc.setText("DRY WELD — nothing energized")
            self._arc.setStyleSheet(ARC_DRY_CSS)
        else:
            self._arc.setText("WELDING OFF")
            self._arc.setStyleSheet(ARC_OFF_CSS)

        # The agent can enable welding too, so the checkbox follows state, not clicks.
        self._weld_on.blockSignals(True)
        self._weld_on.setChecked(bool(weld.get("enabled")))
        self._weld_on.blockSignals(False)

        unit = "mm/s" if vel["mode"] == "physical" else "%"
        speed = vel["physical_mm_s"] if vel["mode"] == "physical" else vel["percentage"]

        self._fields["welding"].setText(
            "LIVE" if state.live_armed else "dry" if state.dry_weld else "off")
        self._fields["weave"].setText(
            f"{weave.get('pattern', '—')} {weave.get('amplitude_mm', '')}mm"
            if weave.get("enabled") else "off")
        self._fields["operation speed"].setText(f"{speed} {unit}")
        self._fields["transport speed"].setText(f"{vel['transport_pct']} %")
        self._fields["locked axes"].setText(", ".join(state.locked_axes) or "none")
        self._fields["motion"].setText("RUNNING" if state.busy else "idle")

        # Only disarm stays live during a pass — everything else would change a plan
        # already in flight, and the controller would refuse it anyway.
        for w in (self._weld_on, self._arm, self._detect, self._trace):
            w.setEnabled(not state.busy)
        self._arm.setEnabled(not state.busy and state.dry_weld)
        self._disarm.setEnabled(True)

    def _render_result(self, label: str, result) -> None:
        if isinstance(result, Exception):
            self.statusBar().showMessage(f"{label}: {result}")
            return
        if isinstance(result, str):          # tools return JSON strings
            try:
                payload = json.loads(result)
            except ValueError:
                self.statusBar().showMessage(f"{label}: {result[:120]}")
                return
            if "error" in payload:
                self.statusBar().showMessage(f"{label}: {payload['error']}")
                return
            self.statusBar().showMessage(f"{label}: ok")
            return
        self.statusBar().showMessage(f"{label}: ok")   # MachineState from a mutator

    # ---------------------------------------------------------------- teardown

    def closeEvent(self, event) -> None:
        state = self._ctl.snapshot()
        if state.busy:
            QMessageBox.warning(self, "Motion running",
                                "A pass is still running. Wait for it to finish.")
            event.ignore()
            return
        # Disarm before anything else can fail. shutdown() is idempotent.
        self._ctl.shutdown()
        self._hub.stop()
        event.accept()
