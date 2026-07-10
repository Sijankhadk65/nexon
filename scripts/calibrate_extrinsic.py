"""3D extrinsic calibration: camera-frame XYZ -> robot base XYZ (T_base_cam).

Fits the full 3D rigid transform that turns the 336L's depth-deprojected points into
robot base coordinates, so a detected pixel + its depth becomes a reachable base-frame
target at ANY height. Depth on this rig was measured clean (see depth_probe.py), so the
Umeyama fit of touched-vs-seen points is the simplest accurate route.

WORKFLOW (collect >= 6 points; spread them in X, Y AND HEIGHT):
  1. Put a small visible marker where the depth camera sees it cleanly.
  2. Jog the robot (pendant) so the TCP tip TOUCHES the marker. Press 'r' to record the
     base XYZ (GetActualTCPPose — needs work object 0).
  3. Retract the arm so it doesn't hide the marker (clean depth there). CLICK the marker
     centre -> camera-frame XYZ from aligned depth + intrinsics.
  4. Press 'n' (or SPACE) to bank the pair. Move the marker (change its HEIGHT for some
     points!) and repeat.
  5. Press 'g' to solve + save. Points must NOT be coplanar, or the Z fit is unconstrained.

Run:   uv run python scripts/calibrate_extrinsic.py
Keys:  r=record pose   click=record cam XYZ   n/SPACE=bank   u=undo
       g=solve+gate+save   q/ESC=quit (auto-solves if >=3)

Output: T_base_cam.npy (4x4, cam XYZ mm -> base XYZ mm) + T_base_cam.meta.json (residuals).
The 'g' gate flags points worse than RESID_GATE_MM so you can drop and refit before saving.
"""

import json
import sys

import cv2 as cv
import numpy as np

from nexon import robot
from nexon.perception.camera import OrbbecCamera

DEPTH_PATCH = 5          # NxN median depth patch around a click (robust to holes)
MIN_POINTS = 6           # recommended; need >=3 non-coplanar to solve
RESID_GATE_MM = 4.0      # flag points whose residual exceeds this before saving
MIN_SOLVE = 4            # never drop below this many points (keep the fit redundant)

OUT_FILE = robot.EXTRINSIC_FILE
META_FILE = OUT_FILE.with_suffix(".meta.json")
WINDOW = "calibrate_extrinsic"


def sample_depth_mm(depth_mm, uv, patch=DEPTH_PATCH):
    """Median of the valid (nonzero) depths in an NxN window around (u,v), or 0.0."""
    h, w = depth_mm.shape
    u, v = int(round(uv[0])), int(round(uv[1]))
    r = patch // 2
    win = depth_mm[max(0, v - r):v + r + 1, max(0, u - r):u + r + 1]
    valid = win[win > 0]
    return float(np.median(valid)) if valid.size else 0.0


def pixel_to_camera(uv, z_mm, intr):
    """Pixel + depth -> camera-frame XYZ (mm) via the color intrinsics."""
    x, y, z = intr.deproject(uv[0], uv[1], z_mm)
    return np.array([float(x), float(y), float(z)])


def umeyama(src, dst, with_scale=True):
    """Least-squares similarity: find R, t, c so dst ~= c*R@src + t (Umeyama 1991)."""
    src = np.asarray(src, float)
    dst = np.asarray(dst, float)
    n = len(src)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    Xs, Xd = src - mu_s, dst - mu_d
    Sigma = (Xd.T @ Xs) / n
    U, D, Vt = np.linalg.svd(Sigma)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:   # reflection fix
        S[2, 2] = -1.0
    R = U @ S @ Vt
    c = (np.trace(np.diag(D) @ S) / (Xs ** 2).sum() * n) if with_scale else 1.0
    t = mu_d - c * R @ mu_s
    return R, t, c


def fit_transform(cam_pts, base_pts):
    """Fit T_base_cam (4x4) via Umeyama. Returns (T, per_point_resid_mm, scale) or Nones."""
    cam = np.array(cam_pts, float)
    base = np.array(base_pts, float)
    if len(cam) < 3:
        return None, None, None
    R, t, c = umeyama(cam, base, with_scale=True)
    M = c * R                                # fold scale into the linear block
    err = np.linalg.norm((M @ cam.T).T + t - base, axis=1)
    T = np.eye(4)
    T[:3, :3] = M
    T[:3, 3] = t
    return T, err, c


def report(err, c, base_pts):
    """Print per-point residuals + coplanar / high-residual warnings."""
    base = np.array(base_pts, float)
    print("\nPer-point residuals (mm):",
          ", ".join(f"#{i}={e:.1f}" for i, e in enumerate(err)))
    print(f"mean {err.mean():.2f} mm   max {err.max():.2f} mm   scale {c:.4f}")
    if float(np.ptp(base[:, 2])) < 20:
        print(f"WARNING: base Z spread only {np.ptp(base[:, 2]):.0f} mm — points nearly "
              "coplanar, so the height fit is weak. Re-take at different heights.")
    if err.mean() > 5:
        print("Mean residual > 5 mm: add more/better-spread points or re-touch sloppy ones.")


def save_transform(T, c, err, base_pts):
    """Save T_base_cam.npy + a quality sidecar with the residuals."""
    base = np.array(base_pts, float)
    np.save(OUT_FILE, T)
    META_FILE.write_text(json.dumps({
        "n_points": int(len(base)),
        "scale": float(c),
        "mean_resid_mm": float(err.mean()),
        "max_resid_mm": float(err.max()),
        "base_z_spread_mm": float(np.ptp(base[:, 2])),
    }, indent=2))
    print(f"Saved {OUT_FILE.resolve()} (+ {META_FILE.name})")
    print("T_base_cam (cam XYZ -> base XYZ) =\n", np.round(T, 3))


def solve_and_report(cam_pts, base_pts):
    """Fit, report, and save without the interactive gate (used on quit)."""
    T, err, c = fit_transform(cam_pts, base_pts)
    if T is None:
        print(f"need >=3 points (have {len(cam_pts)}).")
        return None
    report(err, c, base_pts)
    save_transform(T, c, err, base_pts)
    return T


def gate_and_save(cam_pts, base_pts, raw_px):
    """Fit, then let the user drop high-residual points before saving (the 'g' key).

    Mutates the three lists in place so the on-screen markers stay in sync. Returns
    True if a transform was saved. NOTE: this blocks on input() while you decide — the
    preview window pauses during the prompt, which is expected.
    """
    while True:
        T, err, c = fit_transform(cam_pts, base_pts)
        if T is None:
            print(f"need >=3 points (have {len(cam_pts)}).")
            return False
        report(err, c, base_pts)
        over = [i for i, e in enumerate(err) if e > RESID_GATE_MM]
        if not over:
            save_transform(T, c, err, base_pts)
            return True

        print(f"\n{len(over)} point(s) over the {RESID_GATE_MM:.0f} mm gate: "
              + ", ".join(f"#{i}={err[i]:.1f}" for i in over))
        if len(cam_pts) <= MIN_SOLVE:
            print(f"  at the {MIN_SOLVE}-point floor — can't drop more; save or cancel.")
        try:
            ans = input("drop which? [index / w=worst / s=save as-is / c=cancel]: ").strip().lower()
        except EOFError:
            ans = "s"

        if ans in ("c", "cancel"):
            print("cancelled; nothing saved.")
            return False
        if ans in ("s", "save", ""):
            save_transform(T, c, err, base_pts)
            return True
        if ans in ("w", "worst"):
            idx = int(np.argmax(err))
        elif ans.isdigit() and int(ans) < len(cam_pts):
            idx = int(ans)
        else:
            print("  unrecognised input.")
            continue
        if len(cam_pts) <= MIN_SOLVE:
            print(f"  refusing to drop below {MIN_SOLVE} points.")
            continue
        print(f"  dropped point #{idx} (resid {err[idx]:.1f} mm). Refitting...")
        cam_pts.pop(idx)
        base_pts.pop(idx)
        raw_px.pop(idx)


def main():
    rob = robot.connect_readonly()

    try:
        cam = OrbbecCamera(with_depth=True)
    except RuntimeError as exc:
        print(f"[camera error: {exc}]", file=sys.stderr)
        rob.CloseRPC()
        return

    cam_pts, base_pts, raw_px = [], [], []
    pending_cam = None       # camera XYZ from the last click
    pending_px = None        # last clicked pixel (for drawing)
    pending_base = None      # base XYZ from the last 'r'
    latest = {"depth": None, "intr": None}

    def on_mouse(event, x, y, _flags, _param):
        nonlocal pending_cam, pending_px
        if event != cv.EVENT_LBUTTONDOWN:
            return
        depth = latest["depth"]
        if depth is None:
            return
        z = sample_depth_mm(depth, (x, y))
        if z <= 0:
            print(f"({x},{y}): no depth there (hole) — click on the marker again.")
            return
        pending_cam = pixel_to_camera((x, y), z, latest["intr"])
        pending_px = (x, y)
        print(f"camera XYZ set: ({pending_cam[0]:.1f}, {pending_cam[1]:.1f}, "
              f"{pending_cam[2]:.1f}) mm   [press n to bank once pose is recorded]")

    cv.namedWindow(WINDOW, cv.WINDOW_AUTOSIZE)
    print(f"\nNeed {MIN_POINTS}+ points spread in X, Y AND height. "
          "r=pose  click=camXYZ  n=bank  u=undo  g=solve  q=quit\n")

    solved = False
    callback_set = False  # attach mouse cb after first render (Qt/Wayland handle timing)
    try:
        with cam:
            while True:
                capf = cam.capture()
                if capf is None:
                    if cv.waitKey(1) & 0xFF in (ord("q"), 27):
                        break
                    continue
                latest["depth"] = capf.depth_mm
                latest["intr"] = capf.intrinsics

                view = capf.bgr.copy()
                if pending_px:
                    cv.drawMarker(view, pending_px, (0, 255, 255), cv.MARKER_CROSS, 16, 2)
                for raw in raw_px:
                    cv.drawMarker(view, raw, (0, 255, 0), cv.MARKER_TILTED_CROSS, 12, 1)
                cv.putText(view, f"banked: {len(base_pts)}/{MIN_POINTS}   "
                           f"cam:{'set' if pending_cam is not None else '-'}  "
                           f"pose:{'set' if pending_base else '-'}",
                           (10, 25), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv.putText(view, "r=pose click=camXYZ n=bank u=undo g=solve q=quit",
                           (10, view.shape[0] - 14), cv.FONT_HERSHEY_SIMPLEX, 0.5,
                           (200, 200, 200), 1)
                cv.imshow(WINDOW, view)

                key = cv.waitKey(1) & 0xFF
                if not callback_set:
                    cv.setMouseCallback(WINDOW, on_mouse)  # window realized now
                    callback_set = True

                if key == ord("r"):
                    pose = rob.GetActualTCPPose()[1]
                    pending_base = (pose[0], pose[1], pose[2])
                    print(f"pose set: base ({pose[0]:.1f}, {pose[1]:.1f}, {pose[2]:.1f}) mm")
                elif key in (ord("n"), ord(" ")):
                    if pending_cam is None or pending_base is None:
                        print("need BOTH a clicked camera XYZ and a recorded pose first.")
                    else:
                        cam_pts.append(pending_cam)
                        base_pts.append(pending_base)
                        raw_px.append(pending_px)
                        print(f"banked pair {len(base_pts)}: "
                              f"cam{tuple(round(float(v), 1) for v in pending_cam)} -> "
                              f"base{tuple(round(v, 1) for v in pending_base)}")
                        pending_cam = pending_px = pending_base = None
                elif key == ord("u"):
                    if base_pts:
                        cam_pts.pop(); base_pts.pop(); raw_px.pop()
                        print(f"undone. now {len(base_pts)}.")
                elif key == ord("g"):
                    solved = gate_and_save(cam_pts, base_pts, raw_px)
                elif key in (ord("q"), 27):
                    break
    finally:
        cv.destroyAllWindows()
        rob.CloseRPC()

    if len(base_pts) >= 3 and not solved:
        print("\nSolving on exit...")
        solve_and_report(cam_pts, base_pts)


if __name__ == "__main__":
    main()
