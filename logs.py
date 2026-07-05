"""Split output: the terminal shows only the conversation; everything else goes to
a timestamped log file.

The screen (stdout) is kept for the user↔nexon conversation. `setup()` redirects
the process's stderr into Log/nexon_<timestamp>.txt so library chatter (Qt fonts,
ALSA, Hugging Face) and status messages are captured, not shown. App events —
conversation turns, tool calls, errors — are written to the same file through a
timestamped logger. `mute_stdout()` briefly redirects stdout too, used only around
the heavy imports so the SDK's one-time "load extensions" banner also lands in the
log rather than on screen.
"""

import datetime
import logging
import os
import sys
from contextlib import contextmanager

LOG_DIR = "Log"


def setup(log_dir: str = LOG_DIR):
    """Create a timestamped log file, send stderr into it, and return (logger, path)."""
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(log_dir, f"nexon_{ts}.txt")

    # Redirect the real stderr fd so C-level noise (Qt/ALSA/HF) and any
    # print(file=sys.stderr) are captured in the log instead of on screen.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    os.dup2(fd, 2)
    os.close(fd)

    logging.basicConfig(
        stream=sys.stderr,  # now points at the log file
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("nexon")
    logger.info("=== nexon session started (log file: %s) ===", path)
    return logger, path


@contextmanager
def mute_stdout(path: str):
    """Temporarily redirect stdout into the log file (e.g. around noisy imports)."""
    sys.stdout.flush()
    saved = os.dup(1)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    os.dup2(fd, 1)
    os.close(fd)
    try:
        yield
    finally:
        sys.stdout.flush()
        os.dup2(saved, 1)
        os.close(saved)
