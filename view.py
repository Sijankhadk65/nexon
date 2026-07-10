"""Live visual check: see what the camera detects and measures, in a window.

A standalone verification tool — run it *instead of* main.py, since only one
process can open the camera at a time. It shows the RGB stream with detection
boxes, labels, and (when measuring) real-world dimensions overlaid, so you can
confirm the boxes land on the right things and the millimetre numbers make sense.

Detection runs in a background thread, so the video stays smooth (~30 fps) even
though the model takes ~1–2 s per pass on this CPU — boxes just refresh each time
a pass finishes.

    uv run python view.py "metal tube" "flange"          # measure these, live
    uv run python view.py --no-measure "person" "cup"     # boxes only (faster)
    uv run python view.py --min-confidence 0.5 "bottle"

Keys:  q / Esc quit    m toggle measurement    d toggle depth view    s snapshot

Headless (no display): analyses one frame and saves it to view.jpg instead.
"""

import argparse
import os
import sys
import threading
import time

import cv2

from camera import OrbbecCamera
from detector import load_detector
from dimensioner import measure_all
from viz import colorize_depth, draw_detections


def _has_display() -> bool:
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def _run_headless(cam, detector, targets, min_conf, measure):
    """No window: analyse one settled frame and save an annotated JPG."""
    print("[no display — analysing one frame -> view.jpg]")
    cap = None
    for _ in range(30):  # let auto-exposure settle
        cap = cam.capture()
    if cap is None:
        print("no frame captured", file=sys.stderr)
        return
    dets = detector.detect(cap.bgr, targets, min_confidence=min_conf)
    if measure and cap.depth_mm is not None:
        measure_all(dets, cap.depth_mm, cap.intrinsics)
    cv2.imwrite("view.jpg", draw_detections(cap.bgr, dets))
    for d in dets:
        print(f"  {d.label} {d.confidence:.2f}  {d.dimensions_mm or ''}")
    print("saved view.jpg")


def _run_windowed(cam, detector, targets, min_conf, measure):
    """Smooth live window; detection runs in a worker thread."""
    state = {"cap": None, "dets": [], "measure": measure, "run": True}
    lock = threading.Lock()

    def worker():
        while state["run"]:
            with lock:
                cap, do_measure = state["cap"], state["measure"]
            if cap is None:
                time.sleep(0.01)
                continue
            dets = detector.detect(cap.bgr, targets, min_confidence=min_conf)
            if do_measure and cap.depth_mm is not None:
                measure_all(dets, cap.depth_mm, cap.intrinsics)
            with lock:
                state["dets"] = dets

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    show_depth = False
    print(f"targets: {targets} | measure: {measure}")
    print("keys: q/Esc quit | m measure | d depth | s snapshot")
    try:
        while True:
            cap = cam.capture()
            if cap is None:
                continue
            with lock:
                state["cap"] = cap
                dets, do_measure = state["dets"], state["measure"]

            frame = draw_detections(cap.bgr, dets)
            status = f"measure:{'on' if do_measure else 'off'}  dets:{len(dets)}"
            cv2.putText(frame, status, (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow("nexon vision", frame)
            if show_depth and cap.depth_mm is not None:
                cv2.imshow("depth", colorize_depth(cap.depth_mm))

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("m"):
                with lock:
                    state["measure"] = not state["measure"]
            if key == ord("d"):
                show_depth = not show_depth
                if not show_depth:
                    cv2.destroyWindow("depth")
            if key == ord("s"):
                name = f"snapshot_{int(time.time())}.jpg"
                cv2.imwrite(name, frame)
                print(f"saved {name}")
    finally:
        state["run"] = False
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Live detection+measurement viewer")
    parser.add_argument("targets", nargs="*", default=["person", "laptop", "cup", "bottle"],
                        help="objects to look for (natural language)")
    parser.add_argument("--no-measure", dest="measure", action="store_false",
                        help="skip depth measurement (boxes only, faster)")
    parser.add_argument("--min-confidence", type=float, default=0.4)
    parser.add_argument("--backend", default=os.environ.get("NEXON_DETECTOR", "grounding-dino"))
    args = parser.parse_args()
    targets = args.targets or ["person", "laptop", "cup", "bottle"]

    print(f"loading detector '{args.backend}' (first run downloads the model)…")
    detector = load_detector(args.backend)

    cam = OrbbecCamera(with_depth=True)
    cam.start()
    try:
        if _has_display():
            _run_windowed(cam, detector, targets, args.min_confidence, args.measure)
        else:
            _run_headless(cam, detector, targets, args.min_confidence, args.measure)
    finally:
        cam.stop()


if __name__ == "__main__":
    main()
