"""LangChain tools the Claude orchestrator can call — the robot's senses and actions.

Right now there's one: vision. `detect_objects` looks through the Gemini 336L and
runs the configured open-vocabulary detector, optionally measuring real-world size
from the depth sensor. The camera, detector, and live preview window are owned by
the shared VisionHub (see vision.py) so the chat and the on-screen overlay share a
single camera. Call `shutdown()` on exit to release the device.
"""

import json

from langchain_core.tools import tool

import vision


@tool(parse_docstring=True)
def detect_objects(
    targets: list[str], min_confidence: float = 0.4, measure: bool = False
) -> str:
    """Look through the robot's camera right now and locate specific objects.

    Use this whenever you need to see the physical scene — e.g. the user asks what
    you can see, where something is, or to identify parts (like "metal tube" or
    "flange"). It captures a live frame and returns each match with its pixel
    location. Coordinates are pixels: (0,0) is top-left, x grows right, y grows down.

    Set measure=True to also get each object's real-world size from the depth camera.
    Then every detection includes "dimensions_mm" with length, width, and distance in
    millimeters (length is the longest side, width the perpendicular one). Use this to
    answer questions about a part's physical size. These are approximate (the camera
    sees only the facing surface).

    Args:
        targets: The things to look for, in plain language, e.g. ["metal tube", "flange"].
        min_confidence: Minimum confidence 0-1 to report a match (default 0.4). Raise
            it to cut false positives in a cluttered scene.
        measure: If true, measure each object's real-world dimensions in millimeters
            using the depth sensor. Use when the user asks about size/length/width.
    """
    try:
        result = vision.get_hub().look(targets, min_confidence=min_confidence, measure=measure)
    except Exception as exc:  # noqa: BLE001 — surface hardware errors to Claude, don't crash the chat
        return json.dumps({"error": f"vision unavailable: {exc}"})
    return json.dumps(result)


# All tools exposed to the orchestrator. Add future robot tools here.
ALL_TOOLS = [detect_objects]


def shutdown():
    """Release the camera/preview at the end of the session."""
    vision.shutdown()
