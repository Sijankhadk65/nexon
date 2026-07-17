"""`nexon` — the operator console, and the only frontend.

Runs the window against the shared VisionHub and Controller. The hub is created with
show_window=False: the OpenCV preview must not be built alongside Qt, because
opencv-python is itself linked against Qt and two Qt instances in one process crash in
ways that look random. This window replaces that preview.

The camera takes seconds to open, so it is started on a worker thread — the window comes
up immediately and fills in when frames arrive. The agent takes seconds to build (torch,
transformers), so `MainWindow` starts it on another. Neither blocks the other, and neither
blocks the window.

ONE PROCESS. The conversation used to run in a separate `uv run nexon`, which meant a second
`Controller` singleton and a second copy of robot.py's module globals steering the same arm.
`nexon.session` now runs inside this process, on a worker thread, so there is exactly one
owner of the machine's state — which is what controller.py's docstring claims and what its
subscribe/notify machinery has always quietly assumed.
"""

import sys
import threading

from PySide6.QtGui import QPalette
from PySide6.QtWidgets import QApplication

from nexon import logs
from nexon.perception import vision
from nexon.ui import theme
from nexon.ui.window import MainWindow


def _apply_palette(app: QApplication) -> None:
    """A dark base, so Qt's own chrome (menus, tooltips, dialogs) matches the window.

    Widgets that paint themselves ignore this; it is here for the ones that don't.
    """
    palette = QPalette()
    palette.setColor(QPalette.Window, theme.VOID)
    palette.setColor(QPalette.Base, theme.VOID)
    palette.setColor(QPalette.WindowText, theme.INK)
    palette.setColor(QPalette.Text, theme.INK)
    palette.setColor(QPalette.ToolTipBase, theme.VOID)
    palette.setColor(QPalette.ToolTipText, theme.INK)
    palette.setColor(QPalette.Highlight, theme.ACCENT)
    palette.setColor(QPalette.HighlightedText, theme.INK)
    app.setPalette(palette)


def main() -> int:
    log, log_path = logs.setup()

    app = QApplication(sys.argv)
    app.setApplicationName("nexon")
    app.setApplicationDisplayName("nexon")
    _apply_palette(app)
    app.setFont(theme.font(theme.BODY))

    hub = vision.get_hub(show_window=False)
    window = MainWindow(hub)
    window.resize(1180, 760)
    window.show()

    def start_camera():
        try:
            with logs.mute_stdout(log_path):   # the SDK's startup banner is not conversation
                hub.ensure_started()
        except Exception:  # noqa: BLE001 — the window stays usable without a camera
            log.exception("ui: camera failed to start")

    threading.Thread(target=start_camera, name="nexon-ui-camera", daemon=True).start()

    try:
        return app.exec()
    finally:
        # closeEvent handles the normal path; this covers a crash or a killed event loop.
        # Drop any armed arc before anything else, and independently of the vision
        # teardown — an exception closing the camera must never leave a live arc armed.
        from nexon.controller import get_controller
        try:
            get_controller().shutdown()
        finally:
            try:
                from nexon.agent import tools
                tools.shutdown()
            except Exception:  # noqa: BLE001 — tools may never have been imported
                pass
            vision.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
