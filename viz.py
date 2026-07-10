"""Shared drawing helpers for detection/measurement overlays.

Used by both the standalone viewer (view.py) and the live window inside the chat
(vision.py), so the on-screen look is identical wherever frames are shown.
"""

import cv2
import numpy as np

BOX_COLOR = (0, 255, 0)
DIM_COLOR = (0, 220, 255)  # amber, for the measured-dimensions line


def draw_detections(bgr: np.ndarray, detections) -> np.ndarray:
    """Draw boxes, labels, and (if measured) dimensions onto a copy of the frame."""
    out = bgr.copy()
    for d in detections:
        x1, y1, x2, y2 = d.box
        cv2.rectangle(out, (x1, y1), (x2, y2), BOX_COLOR, 2)
        _label(out, f"{d.label} {d.confidence:.2f}", (x1, y1), BOX_COLOR)

        if d.dimensions_mm:
            dim = d.dimensions_mm
            text = f"L{dim['length']:.0f} W{dim['width']:.0f}  {dim['distance']:.0f}mm"
            cv2.putText(out, text, (x1 + 2, y2 - 6), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, DIM_COLOR, 2, cv2.LINE_AA)
    return out


def _label(img, text, origin, color):
    """Draw text with a filled background so it stays readable over any scene."""
    x, y = origin
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    y = max(y, th + 6)
    cv2.rectangle(img, (x, y - th - base - 4), (x + tw + 4, y), color, -1)
    cv2.putText(img, text, (x + 2, y - base - 1), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 0, 0), 2, cv2.LINE_AA)


def colorize_depth(depth_mm: np.ndarray, near=200.0, far=3000.0) -> np.ndarray:
    """Colorized depth for visual sanity-checking; invalid (0) pixels stay black."""
    valid = depth_mm > 0
    norm = np.clip((depth_mm - near) / (far - near), 0, 1)
    vis = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    vis[~valid] = (0, 0, 0)
    return vis
