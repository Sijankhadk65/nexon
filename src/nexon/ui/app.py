"""`nexon-ui` — the operator console.

Runs the window against the shared VisionHub and Controller. The hub is created with
show_window=False: the OpenCV preview must not be built alongside Qt, because
opencv-python is itself linked against Qt and two Qt instances in one process crash in
ways that look random. This window replaces that preview.

The camera takes seconds to open, so it is started on a worker thread — the window comes
up immediately and fills in when frames arrive.
"""

import sys
import threading

from PySide6.QtWidgets import QApplication

from nexon import logs
from nexon.perception import vision
from nexon.ui.window import MainWindow


def main() -> int:
    log, log_path = logs.setup()

    app = QApplication(sys.argv)
    app.setApplicationName("nex-ON")

    hub = vision.get_hub(show_window=False)
    window = MainWindow(hub)
    window.resize(1180, 680)
    window.show()

    def start_camera():
        try:
            hub.ensure_started()
        except Exception:  # noqa: BLE001 — the window stays usable without a camera
            log.exception("ui: camera failed to start")

    threading.Thread(target=start_camera, name="nexon-ui-camera", daemon=True).start()

    try:
        return app.exec()
    finally:
        # closeEvent handles the normal path; this covers a crash or a killed event loop.
        from nexon.controller import get_controller
        get_controller().shutdown()
        vision.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
