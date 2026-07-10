"""Filesystem anchors, resolved once so no module has to guess from its own location.

Every runtime artifact (the saved seam, the AOI box, the camera->base extrinsic) and the
vendored Fairino SDK live at fixed places in the repo, NOT next to the module that reads
them. Resolving them here means a module can be moved between subpackages without
stranding its data.

PROJECT_ROOT assumes an editable install (src/nexon/paths.py -> repo root), which is what
`uv sync` produces. Inside a built wheel there is no repo root; set NEXON_DATA_DIR /
NEXON_LOG_DIR to override.
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Tracked runtime artifacts: seam.json, seam_aoi.json, T_base_cam.npy (+ .meta.json).
DATA_DIR = Path(os.environ.get("NEXON_DATA_DIR", PROJECT_ROOT / "data"))

# Vendored Fairino SDK — robot.py puts this on sys.path to `import Robot`.
SDK_DIR = PROJECT_ROOT / "fairino_sdk" / "linux" / "fairino"

# Timestamped stderr logs. Anchored to the repo so the launch directory can't move them.
LOG_DIR = Path(os.environ.get("NEXON_LOG_DIR", PROJECT_ROOT / "Log"))
