"""LangChain tools the Claude orchestrator can call — the robot's senses and actions.

Right now there's one: vision. `detect_objects` grabs a live frame from the Gemini
336L and runs the configured open-vocabulary detector, so Claude can decide *when*
to look and *what* to look for (in plain language) mid-conversation.

The camera and detector are expensive to spin up, so they're created lazily on the
first tool call and kept warm for the rest of the session (the camera keeps
streaming, so auto-exposure stays converged between glances). Call `shutdown()` on
exit to release the device.

Pick the detector backend with the NEXON_DETECTOR env var (default: grounding-dino).
"""

import json
import os
import sys

from langchain_core.tools import tool

from camera import OrbbecCamera
from detector import load_detector

DETECTOR_BACKEND = os.environ.get("NEXON_DETECTOR", "grounding-dino")

# Frames to pull on first open so auto-exposure settles before the first glance.
_WARMUP_FRAMES = 30
# Frames to drain per glance so we analyse a current image, not a buffered one.
_FRESH_FRAMES = 5

_camera: OrbbecCamera | None = None
_detector = None


def _ensure_vision():
    """Open camera + detector once; reuse for the session. Raises on hardware failure."""
    global _camera, _detector
    if _camera is None:
        print("[vision: starting Gemini 336L…]", file=sys.stderr)
        cam = OrbbecCamera()
        cam.start()
        for _ in range(_WARMUP_FRAMES):
            cam.read()
        _camera = cam
    if _detector is None:
        print(f"[vision: loading detector '{DETECTOR_BACKEND}' (first run downloads the model)…]",
              file=sys.stderr)
        _detector = load_detector(DETECTOR_BACKEND)
    return _camera, _detector


def _current_frame(cam: OrbbecCamera):
    """Return a fresh frame, skipping any buffered ones."""
    frame = None
    for _ in range(_FRESH_FRAMES):
        frame = cam.read()
    return frame


@tool(parse_docstring=True)
def detect_objects(targets: list[str], min_confidence: float = 0.4) -> str:
    """Look through the robot's camera right now and locate specific objects.

    Use this whenever you need to see the physical scene — e.g. the user asks what
    you can see, where something is, or to identify parts (like "metal tube" or
    "flange"). It captures a live frame and returns each match with its pixel
    location. Coordinates are pixels: (0,0) is top-left, x grows right, y grows down.
    These are 2D pixel boxes, not real-world sizes.

    Args:
        targets: The things to look for, in plain language, e.g. ["metal tube", "flange"].
        min_confidence: Minimum confidence 0-1 to report a match (default 0.4). Raise
            it to cut false positives in a cluttered scene.
    """
    try:
        cam, det = _ensure_vision()
    except Exception as exc:  # noqa: BLE001 — surface hardware errors to Claude, don't crash the chat
        return json.dumps({"error": f"camera/detector unavailable: {exc}"})

    frame = _current_frame(cam)
    if frame is None:
        return json.dumps({"error": "no frame captured from camera"})

    detections = det.detect(frame, targets, min_confidence=min_confidence)
    h, w = frame.shape[:2]
    return json.dumps(
        {
            "image_size": {"width": w, "height": h},
            "detections": [d.as_dict() for d in detections],
        }
    )


# All tools exposed to the orchestrator. Add future robot tools here.
ALL_TOOLS = [detect_objects]


def shutdown():
    """Release the camera at the end of the session."""
    global _camera, _detector
    if _camera is not None:
        _camera.stop()
        _camera = None
    _detector = None
