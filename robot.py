"""Fairino robot motion runtime: connect/enable + straight-line and joint moves.

This is the robot's "hands" — the counterpart to vision.py's "eyes". It wraps the
Fairino Python SDK (vendored in fairino_sdk/) with the proven connect → enable →
clear-faults sequence and a small set of motion primitives (linear MoveL, torch-down
IK solving, tool-frame left/right, joint MoveJ). The LangChain tools in tools.py are
thin wrappers over these.

Every public call follows the same safe pattern as the CLI it came from: connect →
Mode(0)+RobotEnable(1)+ResetAllError() → run ONE move → CloseRPC(). No connection is
left open, so calls are safe to repeat from the chat/voice loop.

Coordinates are mm in the active work frame; angles are degrees (Fairino RPY).
Velocity is a percentage of max, kept low by default for safety. Set NEXON_ROBOT_IP
to point at a different controller.
"""

import logging
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

# Vendored Fairino SDK: fairino_sdk/linux/fairino/Robot.py. Add it to the path by
# absolute location so imports work regardless of the process' working directory.
_SDK_DIR = Path(__file__).resolve().parent / "fairino_sdk" / "linux" / "fairino"
if str(_SDK_DIR) not in sys.path:
    sys.path.insert(0, str(_SDK_DIR))
import Robot  # noqa: E402 — must follow the sys.path insert above

log = logging.getLogger("nexon")

# --- Defaults ---
ROBOT_IP = os.environ.get("NEXON_ROBOT_IP", "192.168.58.2")
MOVE_VEL = 20.0  # Default speed as a percentage of max — keep low for safety.

# Current commanded speed applied to EVERY move. A single mutable value that
# set_velocity() updates and all motion primitives read, so a "go faster/slower"
# request persists across subsequent moves. Percentage of max; clamped to a safe range.
VEL_MIN, VEL_MAX = 1.0, 100.0
CURRENT_VEL = MOVE_VEL


def set_velocity(vel):
    """Set the speed (percentage of max) used by all subsequent moves.

    Clamps to [VEL_MIN, VEL_MAX] and returns the value actually stored. Only takes
    effect while VEL_MODE == "percentage".
    """
    global CURRENT_VEL
    CURRENT_VEL = float(max(VEL_MIN, min(VEL_MAX, vel)))
    log.info("robot: velocity set to %.1f%%", CURRENT_VEL)
    return CURRENT_VEL


# Velocity mode: "physical" (linear moves travel at PHYSICAL_VEL mm/s — the default)
# or "percentage" (speed is 0-100% of max). Every LINEAR move reads VEL_MODE and
# applies the matching MoveL parameters (see _movel_speed_kwargs). Joint moves (MoveJ)
# are angular, so mm/s has no meaning there — they always use the percentage speed
# regardless of mode.
VEL_MODE = "physical"
VEL_MODES = ("percentage", "physical")

# Physical linear speed / acceleration used when VEL_MODE == "physical". In the SDK's
# physical mode (velAccParamMode=1) the real mm/s speed goes in MoveL's `ovl` and the
# mm/s^2 acceleration in `oacc`. On this firmware `vel` AND `acc` MUST also be set to
# 100 (full scale) or the move is rejected with err 183 — leaving them at their
# defaults (vel=20, acc=0) is what caused the physical move to fail. See red_line_viewer
# ._movel in the farino_app reference.
PHYSICAL_VEL_MIN, PHYSICAL_VEL_MAX = 1.0, 250.0
PHYSICAL_VEL = 30.0   # linear TCP speed, mm/s
PHYSICAL_ACC = 200.0  # linear acceleration, mm/s^2 (must be > 0)


def set_velocity_mode(mode):
    """Switch how speed is interpreted: "percentage" (0-100% of max) or "physical" (mm/s).

    Returns the mode actually stored. Raises ValueError on an unknown mode.
    """
    global VEL_MODE
    if mode not in VEL_MODES:
        raise ValueError(f"mode must be one of {VEL_MODES}, got {mode!r}")
    VEL_MODE = mode
    log.info("robot: velocity mode set to %s", VEL_MODE)
    return VEL_MODE


def set_physical_velocity(vel_mm_s):
    """Set the linear speed (mm/s) used when VEL_MODE == "physical".

    Clamps to [PHYSICAL_VEL_MIN, PHYSICAL_VEL_MAX] and returns the value stored. Only
    takes effect for linear moves while the mode is "physical".
    """
    global PHYSICAL_VEL
    PHYSICAL_VEL = float(max(PHYSICAL_VEL_MIN, min(PHYSICAL_VEL_MAX, vel_mm_s)))
    log.info("robot: physical velocity set to %.1f mm/s", PHYSICAL_VEL)
    return PHYSICAL_VEL


def _movel_speed_kwargs():
    """MoveL keyword args for the active velocity mode — checked on every linear move.

    "physical" -> vel/acc pinned to 100 (full scale, required on this firmware or the
    move errors 183) with the real speed in ovl (mm/s) and acceleration in oacc (mm/s^2);
    "percentage" -> vel=CURRENT_VEL (0-100).
    """
    if VEL_MODE == "physical":
        return {"vel": 100.0, "acc": 100.0, "ovl": PHYSICAL_VEL,
                "oacc": PHYSICAL_ACC, "velAccParamMode": 1}
    return {"vel": CURRENT_VEL}


# --------------------------------------------------------------------------- #
# Axis movement locks
# --------------------------------------------------------------------------- #
# Per-axis enable flags for LINEAR (Cartesian) moves. When an axis is disabled, any
# requested motion along that base-frame axis is suppressed — its target coordinate is
# held at the current value so the TCP cannot travel in it. Enforced by linear_move, so
# it applies to every linear move (move_to / move_relative / move_lateral). Joint moves
# (MoveJ) are angular and are not constrained by these Cartesian locks.
AXIS_ENABLED = {"x": True, "y": True, "z": True}
_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


def set_axis_enabled(axis, enabled):
    """Enable or disable linear movement along a base-frame axis ("x", "y", or "z").

    Returns the stored flag. Raises ValueError on an unknown axis.
    """
    key = str(axis).lower()
    if key not in AXIS_ENABLED:
        raise ValueError(f"axis must be one of {tuple(AXIS_ENABLED)}, got {axis!r}")
    AXIS_ENABLED[key] = bool(enabled)
    log.info("robot: axis %s movement %s", key, "enabled" if AXIS_ENABLED[key] else "disabled")
    return AXIS_ENABLED[key]


def locked_axes():
    """List of base-frame axes currently disabled for linear moves (e.g. ["z"])."""
    return [ax for ax in ("x", "y", "z") if not AXIS_ENABLED[ax]]

# Start / home configuration in joint angles [j1..j6] (degrees).
START_JOINTS = [-90.0, -120.0, 85.0, -85.0, -90.0, 0.0]

# Which TOOL-frame axis the torch/wire points along. Flip to +Z if the tool
# points up instead of down.
TORCH_AXIS = np.array([0.0, 0.0, -1.0])


# --------------------------------------------------------------------------- #
# Connection
# --------------------------------------------------------------------------- #
def connect_and_enable(ip=ROBOT_IP):
    """Connect, enable, and clear faults. Returns (robot, tool, user).

    The Mode(0)+RobotEnable(1)+ResetAllError() sequence is required before any
    motion (otherwise MoveL/MoveJ fail with err 154), and motion must reference the
    active TCP / work-object numbers (otherwise err 14).
    """
    robot = Robot.RPC(ip)
    time.sleep(0.5)

    robot.Mode(0)
    time.sleep(0.5)
    robot.RobotEnable(1)
    time.sleep(1.0)
    robot.ResetAllError()
    time.sleep(0.5)

    tool = robot.GetActualTCPNum()[1]
    user = robot.GetActualWObjNum()[1]
    log.info("robot: connected %s | active tool=%s user=%s", ip, tool, user)
    return robot, tool, user


def connect_readonly(ip=ROBOT_IP):
    """Open an RPC connection for READING pose only — no enable, no fault reset.

    Used by extrinsic calibration, where you jog the arm with the pendant and this
    just reads GetActualTCPPose. Enabling here (as connect_and_enable does) would fight
    the pendant. Warns if the active work object isn't 0, since then the TCP pose is
    not in the base frame. Returns the raw RPC handle.
    """
    rob = Robot.RPC(ip)
    time.sleep(0.5)
    wobj = rob.GetActualWObjNum()[1]
    if wobj != 0:
        log.warning("robot: active WObj=%s (not 0) — TCP pose won't be base-frame; set WObj 0", wobj)
    return rob


# --------------------------------------------------------------------------- #
# Orientation / IK helpers (torch-down solving)
# --------------------------------------------------------------------------- #
def _orientation_candidates():
    """Yield (rx, ry, rz) orientations to try, tool-down first then a coarse sweep."""
    for rz in (0.0, 90.0, 180.0, -90.0):
        yield (180.0, 0.0, rz)
    for rx in range(-180, 181, 90):
        for ry in range(-90, 91, 45):
            for rz in range(-180, 181, 90):
                yield (float(rx), float(ry), float(rz))


def _rpy_to_R(rpy):
    """Fairino RPY degrees [rx, ry, rz] -> 3x3 rotation matrix (R = Rz*Ry*Rx)."""
    rx, ry, rz = (math.radians(a) for a in rpy)
    Rx = np.array([[1, 0, 0], [0, math.cos(rx), -math.sin(rx)], [0, math.sin(rx), math.cos(rx)]])
    Ry = np.array([[math.cos(ry), 0, math.sin(ry)], [0, 1, 0], [-math.sin(ry), 0, math.cos(ry)]])
    Rz = np.array([[math.cos(rz), -math.sin(rz), 0], [math.sin(rz), math.cos(rz), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def torch_dir_in_base(rpy):
    """Unit vector the torch points along, in the base frame, for the given RPY."""
    return _rpy_to_R(rpy) @ TORCH_AXIS


def solve_torch_down_rpy(robot, x, y, z, ref_joints, min_down=0.7):
    """Return a torch-DOWN [rx, ry, rz] reachable at (x, y, z), nearest ref, or None."""
    best = None
    for rx, ry, rz in _orientation_candidates():
        rpy = [float(rx), float(ry), float(rz)]
        if torch_dir_in_base(rpy)[2] > -min_down:  # not pointing down enough
            continue
        err, joints = robot.GetInverseKinRef(0, [x, y, z, *rpy], ref_joints)
        if err != 0 or joints is None:
            continue
        travel = sum(abs(a - b) for a, b in zip(joints, ref_joints))
        if best is None or travel < best[0]:
            best = (travel, rpy)
    return None if best is None else best[1]


# --------------------------------------------------------------------------- #
# Motion primitives
# --------------------------------------------------------------------------- #
def linear_move(robot, tool, user, x, y, z, rx=None, ry=None, rz=None, dry_run=False):
    """Straight-line MoveL to (x, y, z) in the base frame, with flexible orientation.

    Pass rx/ry/rz (degrees) to reorient the tool to that RPY along the line; omit
    them to hold the current orientation. Speed is taken from the active velocity mode
    (percentage of max, or physical mm/s — see _movel_speed_kwargs). dry_run IK-checks
    the target and returns 0 without moving. Returns the SDK error code (0 = success).
    """
    start_pose = robot.GetActualTCPPose()[1]  # [x, y, z, rx, ry, rz]

    target = list(start_pose)
    target[0], target[1], target[2] = x, y, z
    if rx is not None:
        target[3] = rx
    if ry is not None:
        target[4] = ry
    if rz is not None:
        target[5] = rz

    # Enforce per-axis locks: hold any disabled base-frame axis at its current value so
    # the requested motion in that axis is suppressed. Done before the IK check/move so
    # both operate on the actual (clamped) target.
    blocked = locked_axes()
    for ax in blocked:
        target[_AXIS_INDEX[ax]] = start_pose[_AXIS_INDEX[ax]]
    if blocked:
        log.info("robot: axis lock active %s — those axes held fixed", blocked)

    speed = _movel_speed_kwargs()
    log.info("robot: MoveL target %s | %s=%s (dry_run=%s)",
             [round(v, 1) for v in target], VEL_MODE, speed, dry_run)

    if dry_run:
        ref = robot.GetActualJointPosDegree()[1]
        err, joints = robot.GetInverseKinRef(0, target, ref)
        if err != 0 or joints is None:
            log.info("robot: DRY RUN target has NO IK solution (err %s)", err)
            return err or -1
        log.info("robot: DRY RUN target reachable, joints %s", [round(v, 1) for v in joints])
        return 0

    # If locks (or a zero request) leave no effective motion, skip the MoveL — commanding
    # a zero-distance move is pointless and can be rejected by the controller.
    if all(abs(a - b) < 1e-3 for a, b in zip(target, start_pose)):
        log.warning("robot: no effective motion (locks %s) — skipping MoveL", blocked)
        return 0

    ret = robot.MoveL(desc_pos=target, tool=tool, user=user, **speed)
    if ret != 0:
        log.warning("robot: MoveL failed (err %s) — target/path unreachable or singular", ret)
    return ret


def linear_move_torch_down(robot, tool, user, x, y, z, dry_run=False):
    """MoveL to (x, y, z) with an auto-solved, reachable torch-DOWN orientation.

    Returns the SDK error code, or -1 if no torch-down orientation is reachable.
    """
    ref_joints = robot.GetActualJointPosDegree()[1]
    rpy = solve_torch_down_rpy(robot, x, y, z, ref_joints)
    if rpy is None:
        log.warning("robot: no torch-DOWN orientation reachable at (%s, %s, %s)", x, y, z)
        return -1
    log.info("robot: torch-down RPY %s", [round(a, 1) for a in rpy])
    return linear_move(robot, tool, user, x, y, z,
                       rx=rpy[0], ry=rpy[1], rz=rpy[2], dry_run=dry_run)


def _nearest_single_axis_orientation(robot, x, y, z, ref_rpy, ref_joints,
                                     step=5.0, max_deg=180.0):
    """Find a reachable orientation at (x,y,z) that changes ONE axis of ref_rpy.

    Sweeps roll, then pitch, then yaw, smallest magnitude first (so the least
    reorientation that reaches the target wins). Returns the [rx, ry, rz] or None.
    """
    mag = step
    while mag <= max_deg:
        for axis in range(3):          # 0=roll(rx), 1=pitch(ry), 2=yaw(rz)
            for sign in (1.0, -1.0):
                rpy = list(ref_rpy)
                rpy[axis] = ref_rpy[axis] + sign * mag
                err, joints = robot.GetInverseKinRef(0, [x, y, z, *rpy], ref_joints)
                if err == 0 and joints is not None:
                    return rpy
        mag += step
    return None


def linear_move_keep_orientation(robot, tool, user, x, y, z, dry_run=False):
    """MoveL to (x, y, z) preferring the CURRENT tool orientation — the least-erratic path.

    Strategy, in order (the movement preference):
      1. Keep the current orientation and move in a straight line (pure translation).
      2. If that target has no IK solution, reorient about ONE base axis at a time
         (roll, then pitch, then yaw), smallest change first, and use the nearest
         reachable orientation — then translate.

    Position is the base-frame TCP target. Returns the SDK error code, or -1 if nothing
    reachable was found even after single-axis reorientation.
    """
    start = robot.GetActualTCPPose()[1]
    ref_rpy = list(start[3:6])
    ref_joints = robot.GetActualJointPosDegree()[1]

    # 1. Current orientation — pure translation, no reorientation.
    err, _ = robot.GetInverseKinRef(0, [x, y, z, *ref_rpy], ref_joints)
    if err == 0:
        log.info("robot: keeping current orientation %s", [round(a, 1) for a in ref_rpy])
        return linear_move(robot, tool, user, x, y, z, dry_run=dry_run)

    # 2. Nearest single-axis reorientation that reaches the target.
    rpy = _nearest_single_axis_orientation(robot, x, y, z, ref_rpy, ref_joints)
    if rpy is None:
        log.warning("robot: (%.0f, %.0f, %.0f) unreachable even after single-axis reorient",
                    x, y, z)
        return -1
    log.info("robot: reoriented to %s (from %s) to reach target",
             [round(a, 1) for a in rpy], [round(a, 1) for a in ref_rpy])
    return linear_move(robot, tool, user, x, y, z,
                       rx=rpy[0], ry=rpy[1], rz=rpy[2], dry_run=dry_run)


# Fixed BASE-frame directions for intuitive operator commands. These do NOT depend on the
# tool's orientation: left/right run along base X, front/back along base Y — so they match
# the operator's convention (right=+X, left=-X, forward/front=+Y, back=-Y) regardless of how
# the tool is tilted. Base X/Y moves are horizontal, so height (Z) is never changed.
BASE_DIRS = {
    "right":    np.array([1.0, 0.0, 0.0]),
    "left":     np.array([-1.0, 0.0, 0.0]),
    "forward":  np.array([0.0, 1.0, 0.0]),
    "front":    np.array([0.0, 1.0, 0.0]),
    "back":     np.array([0.0, -1.0, 0.0]),
    "backward": np.array([0.0, -1.0, 0.0]),
}


def move_base_direction(robot, tool, user, direction, distance, dry_run=False):
    """MoveL `distance` mm along a fixed BASE-frame `direction`.

    right=+X, left=-X, forward/front=+Y, back/backward=-Y. Straight-line base-frame move with
    orientation preserved (height unchanged). Returns the SDK error code. Raises ValueError on
    an unknown direction.
    """
    key = str(direction).lower()
    if key not in BASE_DIRS:
        raise ValueError(f"direction must be one of {sorted(BASE_DIRS)}, got {direction!r}")
    start = robot.GetActualTCPPose()[1]
    dx, dy, dz = BASE_DIRS[key] * distance
    log.info("robot: %s %s mm (base frame) -> delta [%.1f, %.1f, %.1f]",
             key, distance, dx, dy, dz)
    return linear_move(robot, tool, user,
                       start[0] + dx, start[1] + dy, start[2] + dz,
                       dry_run=dry_run)


# --------------------------------------------------------------------------- #
# Camera -> base extrinsic (hand-eye TF)
# --------------------------------------------------------------------------- #
# 4x4 transform mapping camera-frame XYZ (mm) -> robot base XYZ (mm), produced by
# calibrate_extrinsic.py (Umeyama fit of touched base points vs. depth-deprojected
# camera points). Loaded lazily and cached so runtime tools can turn a detected
# pixel + depth into a base-frame target.
EXTRINSIC_FILE = Path(__file__).resolve().parent / "T_base_cam.npy"
_T_base_cam = None


def load_extrinsic(path=EXTRINSIC_FILE):
    """Load and cache the 4x4 camera->base transform. Raises FileNotFoundError if unset."""
    global _T_base_cam
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"no extrinsic at {p} — run calibrate_extrinsic.py first")
    _T_base_cam = np.load(p)
    log.info("robot: loaded extrinsic %s", p.name)
    return _T_base_cam


def get_extrinsic():
    """Return the cached camera->base transform, loading it on first use."""
    if _T_base_cam is None:
        load_extrinsic()
    return _T_base_cam


def cam_to_base(p_cam, T=None):
    """Camera-frame XYZ (mm) -> base-frame XYZ (mm) via T_base_cam. Returns a length-3 array."""
    M = get_extrinsic() if T is None else np.asarray(T)
    return (M @ np.array([p_cam[0], p_cam[1], p_cam[2], 1.0]))[:3]


def pixel_to_base(u, v, z_mm, intr, T=None):
    """Detected pixel (u,v) + its depth z_mm (mm) -> base-frame XYZ (mm).

    `intr` is any object with a .deproject(u, v, z) method (camera.CameraIntrinsics):
    pixel+depth -> camera-frame XYZ, which cam_to_base then maps into the base frame.
    """
    x, y, z = intr.deproject(u, v, z_mm)
    return cam_to_base((float(x), float(y), float(z)), T)
