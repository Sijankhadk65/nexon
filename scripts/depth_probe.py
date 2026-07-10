"""Depth-noise probe for the Gemini 336L — read-only, never touches the robot.

Settles the "is the depth good enough for ~1 mm welding" question with data instead
of opinion. Open a live view, click a spot on your work surface (put a matte target
at the REAL welding distance), and it samples the depth there over many frames and
reports how much it wiggles:

  * temporal std of the depth (mm)     — the random noise that averaging beats down
  * min / max / peak-to-peak (mm)      — worst-case excursion
  * hole rate                          — how often the sensor returns no depth there
  * 3D (X, Y, Z) std in the camera frame, deprojected with the real intrinsics
  * averaged std = std / sqrt(N)       — what a line-fit over N samples can achieve

How to read it: the single-shot std is your per-point noise; the averaged std is
roughly what a seam-line fit over many pixels/frames gets you. Compare BOTH to 1 mm.
Sample a flat matte patch (never an edge — edges give flying pixels), and let the
sensor warm up a couple of minutes first (depth drifts until it's thermally stable).

Run:   uv run python scripts/depth_probe.py
Keys:  click = set the probe pixel   SPACE / g = run a measurement burst
       + / - = grow / shrink the median patch   q / Esc = quit

Headless (no display): measures once at the image centre (or --pixel) and prints.
"""

import argparse
import os
import sys

import cv2
import numpy as np

from nexon.perception.camera import OrbbecCamera

WINDOW = "depth_probe"


def _has_display() -> bool:
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def patch_median(depth_mm, u, v, patch):
    """Median of the valid (nonzero) depths in an NxN window around (u,v).

    Returns (median_mm, hole_fraction): median is 0.0 if the whole patch is invalid;
    hole_fraction is the share of the window with no depth (0 = fully filled).
    """
    h, w = depth_mm.shape
    u = int(np.clip(u, 0, w - 1))
    v = int(np.clip(v, 0, h - 1))
    r = patch // 2
    win = depth_mm[max(0, v - r):v + r + 1, max(0, u - r):u + r + 1]
    valid = win[win > 0]
    hole_frac = 1.0 - valid.size / win.size if win.size else 1.0
    med = float(np.median(valid)) if valid.size else 0.0
    return med, hole_frac


def compute_stats(depths, hole_fracs, misses, u, v, patch, intr):
    """Turn collected samples into a stats dict, or None if too few valid depths."""
    if len(depths) < 2:
        return None
    d = np.asarray(depths)
    # Fixed pixel, varying z: lateral coords scale with z, so their spread reflects
    # how the deprojected 3D point wiggles under depth noise (Z is the dominant term).
    xc = (u - intr.cx) * d / intr.fx
    yc = (v - intr.cy) * d / intr.fy
    return {
        "pixel": (u, v),
        "patch": patch,
        "n_valid": int(d.size),
        "misses": int(misses),
        "mean_mm": float(d.mean()),
        "std_mm": float(d.std()),
        "min_mm": float(d.min()),
        "max_mm": float(d.max()),
        "p2p_mm": float(np.ptp(d)),
        "std_x_mm": float(xc.std()),
        "std_y_mm": float(yc.std()),
        "std_z_mm": float(d.std()),
        "avg_std_mm": float(d.std() / np.sqrt(d.size)),
        "hole_rate": float(np.mean(hole_fracs)) if hole_fracs else 1.0,
    }


def run_burst(cam, u, v, patch, intr, frames):
    """Blocking collect of `frames` samples at (u,v) — used for the headless path."""
    depths, hole_fracs, misses = [], [], 0
    for _ in range(frames):
        cap = cam.capture()
        if cap is None or cap.depth_mm is None:
            misses += 1
            continue
        z, hf = patch_median(cap.depth_mm, u, v, patch)
        hole_fracs.append(hf)
        if z <= 0:
            misses += 1
            continue
        depths.append(z)
    return compute_stats(depths, hole_fracs, misses, u, v, patch, intr)


def verdict(std_mm, hole_rate):
    """One-line read on whether this depth clears a ~1 mm welding bar."""
    if hole_rate > 0.2:
        return f"HIGH HOLE RATE ({hole_rate:.0%}) — bad spot (edge/reflective?) or poor depth here."
    if std_mm < 0.5:
        return "EXCELLENT — per-point noise already sub-0.5 mm; depth alone is fine."
    if std_mm < 2.0:
        return "GOOD — a seam-line fit (many points/frames) gets you well under 1 mm."
    if std_mm < 5.0:
        return "MARGINAL — needs heavy averaging; keep the plane as a cross-check."
    return "NOISY — single overhead depth won't hit 1 mm; lean on the plane constraint."


def print_stats(s):
    if s is None:
        print("  not enough valid depth samples (all holes?) — try another spot.")
        return
    u, v = s["pixel"]
    print(f"\n  pixel ({u},{v}), {s['patch']}x{s['patch']} median patch, "
          f"{s['n_valid']} valid / {s['misses']} missed")
    print(f"  depth   mean {s['mean_mm']:.1f} mm   std {s['std_mm']:.3f} mm   "
          f"min {s['min_mm']:.1f}   max {s['max_mm']:.1f}   p2p {s['p2p_mm']:.2f} mm")
    print(f"  3D std  X {s['std_x_mm']:.3f}   Y {s['std_y_mm']:.3f}   "
          f"Z {s['std_z_mm']:.3f} mm  (camera frame)")
    print(f"  after averaging {s['n_valid']} frames: std ~ {s['avg_std_mm']:.3f} mm")
    print(f"  hole rate {s['hole_rate']:.1%}")
    print(f"  -> {verdict(s['std_mm'], s['hole_rate'])}\n")


def live(cam, intr, patch0, frames):
    """Interactive: click to aim, SPACE to measure, +/- to resize the patch.

    The measurement burst is accumulated INSIDE this display loop (one sample per
    iteration) instead of a blocking sub-loop, so the GUI keeps getting pumped —
    otherwise the window freezes for the whole burst and Wayland/xcb kills it.
    """
    state = {"px": None, "patch": patch0, "last": None,
             "collecting": False, "burst_px": None,
             "depths": [], "holes": [], "misses": 0}

    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN and not state["collecting"]:
            state["px"] = (x, y)

    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    print("Aim at a flat matte patch at your welding distance. "
          "Click it, then press SPACE to measure. q/Esc to quit.")

    # On this Qt/Wayland build the window handle isn't realized until the first
    # imshow+waitKey, so setMouseCallback beforehand raises "NULL window handler".
    # Attach it inside the loop, once, right after the first successful render.
    callback_set = False
    while True:
        cap = cam.capture()
        if cap is None:
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):  # keep the GUI alive on misses
                break
            continue
        img = cap.bgr.copy()
        h, w = img.shape[:2]
        if state["px"] is None:
            state["px"] = (w // 2, h // 2)

        # Accumulate one burst sample per frame while collecting.
        if state["collecting"]:
            bu, bv = state["burst_px"]
            if cap.depth_mm is None:
                state["misses"] += 1
            else:
                z, hf = patch_median(cap.depth_mm, bu, bv, state["patch"])
                state["holes"].append(hf)
                if z > 0:
                    state["depths"].append(z)
                else:
                    state["misses"] += 1
            if len(state["depths"]) + state["misses"] >= frames:
                s = compute_stats(state["depths"], state["holes"], state["misses"],
                                  bu, bv, state["patch"], intr)
                print_stats(s)
                state["last"] = s
                state["collecting"] = False

        u, v = state["px"]
        live_txt = "no depth"
        if cap.depth_mm is not None:
            z, hf = patch_median(cap.depth_mm, u, v, state["patch"])
            live_txt = f"{z:.1f} mm (holes {hf:.0%})" if z > 0 else "hole"

        r = state["patch"] // 2
        cv2.rectangle(img, (u - r, v - r), (u + r, v + r), (0, 255, 0), 1)
        cv2.drawMarker(img, (u, v), (0, 255, 0), cv2.MARKER_CROSS, 14, 1)
        cv2.putText(img, f"({u},{v}) patch {state['patch']}  depth {live_txt}",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(img, "click=aim  SPACE=measure  +/-=patch  q=quit",
                    (10, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        if state["collecting"]:
            got = len(state["depths"]) + state["misses"]
            cv2.putText(img, f"collecting {got}/{frames}...",
                        (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
        elif state["last"] is not None:
            s = state["last"]
            cv2.putText(img, f"last: std {s['std_mm']:.2f}mm  avg {s['avg_std_mm']:.2f}mm  "
                             f"holes {s['hole_rate']:.0%}",
                        (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

        cv2.imshow(WINDOW, img)
        key = cv2.waitKey(1) & 0xFF
        if not callback_set:
            cv2.setMouseCallback(WINDOW, on_mouse)  # window is realized now
            callback_set = True
        if key in (ord("q"), 27):
            break
        if key in (ord(" "), ord("g")) and not state["collecting"]:
            state.update(collecting=True, burst_px=state["px"],
                         depths=[], holes=[], misses=0)
            print(f"measuring {frames} frames at {state['px']}...")
        elif key in (ord("+"), ord("=")) and not state["collecting"]:
            state["patch"] = min(state["patch"] + 2, 51)
        elif key in (ord("-"), ord("_")) and not state["collecting"]:
            state["patch"] = max(state["patch"] - 2, 1)

    cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description="Depth-noise probe for the Gemini 336L (read-only).")
    ap.add_argument("--frames", type=int, default=200, help="Samples per measurement burst (default 200).")
    ap.add_argument("--patch", type=int, default=5, help="NxN median depth patch (default 5).")
    ap.add_argument("--pixel", type=int, nargs=2, metavar=("U", "V"),
                    help="Headless: pixel to sample (default image centre).")
    args = ap.parse_args()

    try:
        cam = OrbbecCamera(with_depth=True)
    except RuntimeError as exc:
        print(f"[camera error: {exc}]", file=sys.stderr)
        return

    info = cam.device_info
    print(f"camera: {info.get_name()} (serial {info.get_serial_number()}) "
          f"— color+depth {cam.width}x{cam.height}@{cam.fps}")

    with cam:
        # Warm-up / auto-exposure settle, and grab intrinsics from a real frame.
        cap = None
        for _ in range(30):
            cap = cam.capture()
            if cap is not None:
                break
        if cap is None:
            print("[no frames from camera]", file=sys.stderr)
            return
        intr = cap.intrinsics
        print(f"intrinsics: fx={intr.fx:.1f} fy={intr.fy:.1f} "
              f"cx={intr.cx:.1f} cy={intr.cy:.1f}")

        if _has_display():
            live(cam, intr, args.patch, args.frames)
        else:
            h, w = cap.bgr.shape[:2]
            u, v = args.pixel if args.pixel else (w // 2, h // 2)
            print(f"[headless] measuring {args.frames} frames at ({u},{v})...")
            print_stats(run_burst(cam, u, v, args.patch, intr, args.frames))


if __name__ == "__main__":
    main()
