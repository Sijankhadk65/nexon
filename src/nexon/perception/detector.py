"""Swappable object-detection backends for the nexon robot orchestrator.

nexon orchestrates different robots; each needs to *see* differently, so the
detector is an interface, not a fixed model. Claude (the orchestrator) calls one
tool — "find these things" — and never knows which model answers. Swap the backend
by name/config: open-vocab for bootstrapping a new domain (no training data), a
fine-tuned model once you've labelled parts, or Claude's own vision as a fallback.

First trial domain: a welding cobot identifying welding operands (metal tubes,
flanges, plates). Those aren't in any stock closed-vocabulary model, so the first
backend is **open-vocabulary** (Grounding DINO): Claude passes targets as plain
text ("metal tube. flange.") and gets precise pixel boxes back.

Boxes are pixel coordinates only. Real-world dimensions (length/width in mm) come
later from fusing these boxes/masks with the 336L depth stream + camera intrinsics
— `Detection` leaves room for that (`mask`, `dimensions_mm`) without a redesign.

Backends are registered in `_BACKENDS`; construct one with `load_detector(name)`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass
class Detection:
    """One detected object. Coordinates are pixels in the source frame (x1,y1)-(x2,y2)."""

    label: str
    confidence: float
    box: tuple[int, int, int, int]  # (x1, y1, x2, y2)

    # Reserved for the segmentation + depth-fusion layers to come. Kept optional
    # so today's box-only backends don't have to populate them.
    mask: np.ndarray | None = None
    dimensions_mm: dict[str, float] | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def center(self) -> tuple[int, int]:
        x1, y1, x2, y2 = self.box
        return (x1 + x2) // 2, (y1 + y2) // 2

    @property
    def pixel_size(self) -> tuple[int, int]:
        """(width, height) of the box in pixels — not real-world size."""
        x1, y1, x2, y2 = self.box
        return x2 - x1, y2 - y1

    def as_dict(self) -> dict:
        """JSON-friendly view for handing back to Claude as a tool result."""
        d = {
            "label": self.label,
            "confidence": round(self.confidence, 3),
            "box": [int(v) for v in self.box],
            "center": list(self.center),
        }
        if self.dimensions_mm:
            d["dimensions_mm"] = self.dimensions_mm
        return d


@runtime_checkable
class Detector(Protocol):
    """Any detection backend. Implementations are interchangeable behind this method.

    `targets` is a list of natural-language things to look for. Open-vocabulary
    backends use it directly; closed-vocabulary ones may ignore it (they only know
    their trained classes) or use it to filter. `None` means "whatever you know."
    """

    name: str

    def detect(
        self,
        image_bgr: np.ndarray,
        targets: list[str] | None = None,
        *,
        min_confidence: float = 0.3,
    ) -> list[Detection]: ...


# --- backend registry -------------------------------------------------------

_BACKENDS: dict[str, type] = {}


def register(name: str):
    """Class decorator: make a Detector implementation loadable by `name`."""

    def wrap(cls):
        cls.name = name
        _BACKENDS[name] = cls
        return cls

    return wrap


def available_backends() -> list[str]:
    return sorted(_BACKENDS)


def load_detector(name: str = "grounding-dino", **kwargs) -> Detector:
    """Instantiate a registered backend. Heavy deps load inside the backend, not here."""
    if name not in _BACKENDS:
        raise ValueError(
            f"Unknown detector {name!r}. Available: {available_backends()}"
        )
    return _BACKENDS[name](**kwargs)


# --- Grounding DINO: open-vocabulary, permissive (Apache-2.0) ----------------


@register("grounding-dino")
class GroundingDinoDetector:
    """Zero-shot, text-prompted detection via HuggingFace Grounding DINO.

    Detects arbitrary objects named in `targets` with no training — ideal for
    bootstrapping the welding domain before a labelled dataset exists. `-tiny`
    is the dev default; use `-base` on the real (GPU-equipped) welding cell.

    Torch/transformers are imported lazily so simply importing this module (and
    the rest of nexon) stays cheap; the model also downloads on first construction.
    """

    def __init__(
        self,
        model_id: str = "IDEA-Research/grounding-dino-tiny",
        device: str | None = None,
        default_targets: list[str] | None = None,
    ):
        import logging as _logging

        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        from transformers.utils import logging as hf_logging

        # Keep the terminal clean: only surface real errors from these libraries.
        hf_logging.set_verbosity_error()
        _logging.getLogger("huggingface_hub").setLevel(_logging.ERROR)

        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.default_targets = default_targets or ["object"]
        self._torch = torch
        self._processor = AutoProcessor.from_pretrained(model_id)
        self._model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(
            self.device
        )
        self._model.eval()

    @staticmethod
    def _build_prompt(targets: list[str]) -> str:
        # Grounding DINO wants lowercase phrases, each terminated by a period.
        return " ".join(f"{t.strip().lower().rstrip('.')}." for t in targets if t.strip())

    def detect(
        self,
        image_bgr: np.ndarray,
        targets: list[str] | None = None,
        *,
        min_confidence: float = 0.3,
    ) -> list[Detection]:
        from PIL import Image

        targets = targets or self.default_targets
        prompt = self._build_prompt(targets)
        rgb = image_bgr[:, :, ::-1]  # BGR -> RGB
        image = Image.fromarray(np.ascontiguousarray(rgb))

        inputs = self._processor(images=image, text=prompt, return_tensors="pt").to(
            self.device
        )
        with self._torch.no_grad():
            outputs = self._model(**inputs)

        h, w = image_bgr.shape[:2]
        results = self._post_process(inputs, outputs, min_confidence, (h, w))

        # Newer transformers return string phrases under "text_labels"; "labels"
        # is deprecated (and will become integer ids). Prefer the former.
        labels = results.get("text_labels")
        if labels is None:
            labels = results["labels"]

        detections: list[Detection] = []
        for box, score, label in zip(results["boxes"], results["scores"], labels):
            x1, y1, x2, y2 = (int(v) for v in box.tolist())
            text = label if isinstance(label, str) else str(label)
            detections.append(
                Detection(
                    label=text.strip(" ."),
                    confidence=float(score),
                    box=(x1, y1, x2, y2),
                )
            )
        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections

    def _post_process(self, inputs, outputs, threshold, target_size):
        """Call transformers' grounded post-processor across API versions."""
        import warnings

        pp = self._processor.post_process_grounded_object_detection
        common = dict(
            outputs=outputs,
            input_ids=inputs.input_ids,
            target_sizes=[target_size],
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)  # labels->text_labels notice
            # Newer transformers renamed box_threshold -> threshold; try both.
            try:
                return pp(**common, threshold=threshold, text_threshold=0.25)[0]
            except TypeError:
                return pp(**common, box_threshold=threshold, text_threshold=0.25)[0]


# --- optional demo ----------------------------------------------------------


def _draw(image_bgr: np.ndarray, detections: list[Detection]) -> np.ndarray:
    import cv2

    out = image_bgr.copy()
    for d in detections:
        x1, y1, x2, y2 = d.box
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            out,
            f"{d.label} {d.confidence:.2f}",
            (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
    return out


def main():
    """Grab one frame from the 336L and run detection against CLI-supplied targets.

    Usage:  uv run python -m nexon.perception.detector "metal tube" "flange" "bottle"
    """
    import sys

    import cv2

    from nexon.perception.camera import OrbbecCamera

    targets = sys.argv[1:] or ["bottle", "person", "laptop", "cup"]
    print(f"loading detector (first run downloads the model)…")
    detector = load_detector("grounding-dino")
    print(f"targets: {targets}")

    with OrbbecCamera() as cam:
        # Let auto-exposure settle before the frame we actually analyse.
        frame = None
        for _ in range(30):
            frame = cam.read()
    if frame is None:
        print("no frame captured", file=sys.stderr)
        return

    detections = detector.detect(frame, targets, min_confidence=0.3)
    print(f"\n{len(detections)} detection(s):")
    for d in detections:
        print(f"  {d.as_dict()}")

    cv2.imwrite("detections.jpg", _draw(frame, detections))
    print("annotated frame -> detections.jpg")


if __name__ == "__main__":
    main()
