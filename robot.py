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


# OPERATION speed mode: "physical" (working strokes travel at PHYSICAL_VEL mm/s — the
# default) or "percentage" (0-100% of max via CURRENT_VEL). This is the "doing work" speed
# and is applied ONLY to the working strokes — the seam traverse, the red-line traverse, and
# the descent onto a red dot (the moves that pass operation=True). EVERY OTHER move uses the
# TRANSPORT_VEL transportation speed instead (see below), so a fast physical weld speed never
# leaks into jogs / positioning. Joint moves (MoveJ) are angular, so mm/s has no meaning
# there — they use a percentage regardless of mode.
VEL_MODE = "physical"
VEL_MODES = ("percentage", "physical")

# Physical linear speed / acceleration used when VEL_MODE == "physical". In the SDK's
# physical mode (velAccParamMode=1) the real mm/s speed goes in MoveL's `ovl` and the
# mm/s^2 acceleration in `oacc`. On this firmware `vel` AND `acc` MUST also be set to
# 100 (full scale) or the move is rejected with err 183 — leaving them at their
# defaults (vel=20, acc=0) is what caused the physical move to fail. See red_line_viewer
# ._movel in the farino_app reference.
PHYSICAL_VEL_MIN, PHYSICAL_VEL_MAX = 1.0, 250.0
PHYSICAL_VEL = 3.0   # linear TCP speed, mm/s
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


# --------------------------------------------------------------------------- #
# Transportation speed (positioning / jog moves)
# --------------------------------------------------------------------------- #
# The OPERATION speed above is the "doing work" speed and is applied ONLY to the working
# strokes (seam traverse, red-line traverse, red-dot descent). EVERY OTHER linear move —
# jogs (move_to / move_relative / move_direction) and the hover / approach / descend /
# retract positioning legs — uses this TRANSPORTATION speed instead: ALWAYS a percentage of
# max, low by default, so repositioning never inherits a fast physical weld speed. Joint
# moves (home / joints) use it too.
TRANSPORT_VEL = 10.0   # percentage of max for positioning / jog moves


def set_transport_velocity(vel):
    """Set the TRANSPORTATION speed (percentage of max) for jog / positioning moves.

    Clamps to [VEL_MIN, VEL_MAX] and returns the stored value. This is separate from the
    operation speed (set_velocity / set_physical_velocity), which only drives the working
    strokes; transportation is always a percentage.
    """
    global TRANSPORT_VEL
    TRANSPORT_VEL = float(max(VEL_MIN, min(VEL_MAX, vel)))
    log.info("robot: transport velocity set to %.1f%%", TRANSPORT_VEL)
    return TRANSPORT_VEL


# --------------------------------------------------------------------------- #
# Weave / oscillation (welding weave overlaid on the traverse)
# --------------------------------------------------------------------------- #
# A weave overlays a side-to-side oscillation on the straight traverse MoveL:
# WeaveSetPara -> WeaveStart -> traverse -> WeaveEnd, so the tool swings as it advances
# along the seam (the motion a real weld weave makes). Ported from red_line_viewer's
# weave_start/weave_end in the farino_app reference. The swing frequency is DERIVED so a
# whole number of cycles fit the path at the physical travel speed — freq = count * speed
# / path_len — so weave needs a KNOWN mm/s speed (VEL_MODE == "physical"); it is skipped in
# percentage mode. Weave is MOTION ONLY: it fires no arc, exactly like the follow_* traces.
WEAVE_ENABLED = False
WEAVE_NUM = 0          # WeaveSetPara config slot on the controller
WEAVE_TYPE = 0         # weaveType: 0=planar triangle, 4=planar sine, ... (see WeaveSetPara)
WEAVE_RANGE = 4.0      # swing amplitude (total side-to-side width), mm
# Weave spacing along the seam, set two mutually-exclusive ways (WEAVE_SPACING picks which):
#   "cycles" -- WEAVE_COUNT full oscillations fitted over the WHOLE seam (count scales with
#               seam length: freq = count * speed / path_len).
#   "pitch"  -- WEAVE_PITCH mm advanced per oscillation, a fixed spacing INDEPENDENT of seam
#               length (freq = speed / pitch); the usual weld spec (mm per weave).
WEAVE_SPACING = "cycles"
WEAVE_COUNT = 6.0      # oscillations over the whole seam, used when WEAVE_SPACING == "cycles"
WEAVE_PITCH = 8.0      # mm advanced per oscillation, used when WEAVE_SPACING == "pitch"
WEAVE_RANGE_MIN, WEAVE_RANGE_MAX = 0.1, 30.0
WEAVE_COUNT_MIN, WEAVE_COUNT_MAX = 1.0, 200.0
WEAVE_PITCH_MIN, WEAVE_PITCH_MAX = 0.5, 100.0

# Human-friendly weave shape names -> Fairino weaveType codes (a useful subset of the
# controller's patterns; see WeaveSetPara docs in the vendored SDK).
WEAVE_PATTERNS = {
    "triangle": 0,           # planar triangular (zig-zag)
    "sine": 4,               # planar sine (smooth)
    "vertical_triangle": 6,  # vertical triangular
    "circle_cw": 2,          # clockwise circular
    "circle_ccw": 3,         # counter-clockwise circular
}


def weave_settings():
    """Current weave configuration as a plain dict.

    Includes the active `spacing` mode plus BOTH the cycles and pitch_mm values (only the one
    named by `spacing` drives the swing; the other is the last value you set for that mode).
    """
    name = next((k for k, v in WEAVE_PATTERNS.items() if v == WEAVE_TYPE), str(WEAVE_TYPE))
    return {"enabled": WEAVE_ENABLED, "pattern": name, "weave_type": WEAVE_TYPE,
            "amplitude_mm": WEAVE_RANGE, "spacing": WEAVE_SPACING,
            "cycles": WEAVE_COUNT, "pitch_mm": WEAVE_PITCH}


def configure_weave(enabled=None, pattern=None, amplitude_mm=None, cycles=None, pitch_mm=None):
    """Enable/disable the weave and set its shape. Only the given fields change.

    pattern is a name from WEAVE_PATTERNS ("triangle", "sine", ...); amplitude_mm/cycles/
    pitch_mm are clamped to their safe ranges. `cycles` and `pitch_mm` are two mutually
    exclusive ways to space the weave along the seam and setting one selects that spacing
    mode (see WEAVE_SPACING): cycles = a fixed number of oscillations over the whole seam;
    pitch_mm = a fixed mm advanced per oscillation, independent of seam length. If both are
    passed, pitch_mm wins. Returns the resulting weave_settings(). Raises ValueError on an
    unknown pattern name.
    """
    global WEAVE_ENABLED, WEAVE_TYPE, WEAVE_RANGE, WEAVE_COUNT, WEAVE_PITCH, WEAVE_SPACING
    if enabled is not None:
        WEAVE_ENABLED = bool(enabled)
    if pattern is not None:
        key = str(pattern).lower()
        if key not in WEAVE_PATTERNS:
            raise ValueError(f"pattern must be one of {sorted(WEAVE_PATTERNS)}, got {pattern!r}")
        WEAVE_TYPE = WEAVE_PATTERNS[key]
    if amplitude_mm is not None:
        WEAVE_RANGE = float(max(WEAVE_RANGE_MIN, min(WEAVE_RANGE_MAX, amplitude_mm)))
    # Spacing: cycles-over-seam vs fixed mm-pitch. Setting either selects its mode; pitch_mm
    # takes precedence when both are given in one call.
    if cycles is not None:
        WEAVE_COUNT = float(max(WEAVE_COUNT_MIN, min(WEAVE_COUNT_MAX, cycles)))
        WEAVE_SPACING = "cycles"
    if pitch_mm is not None:
        WEAVE_PITCH = float(max(WEAVE_PITCH_MIN, min(WEAVE_PITCH_MAX, pitch_mm)))
        WEAVE_SPACING = "pitch"
    spacing = (f"pitch={WEAVE_PITCH:.1f}mm/cycle" if WEAVE_SPACING == "pitch"
               else f"cycles={WEAVE_COUNT:.0f}")
    log.info("robot: weave %s | type=%d amplitude=%.1fmm %s",
             "ON" if WEAVE_ENABLED else "OFF", WEAVE_TYPE, WEAVE_RANGE, spacing)
    return weave_settings()


# --------------------------------------------------------------------------- #
# Arc welding (arc struck during the seam trace)
# --------------------------------------------------------------------------- #
# Ported from red_line_viewer.weld_start/weld_end in the farino_app reference. When welding is
# enabled, the seam trace becomes a weld pass: descend to the lead-in -> ARCStart -> move to
# P1 -> traverse to P2 (the weld stroke; weave rides it if enabled) -> ARCEnd -> retract. Two
# gates for safety:
#   WELD_ENABLED -- run the arc sequence at all during the trace.
#   WELD_LIVE    -- energize a REAL arc. When False it is a DRY WELD: the motion is IDENTICAL
#                   and the arc steps are logged, but nothing is energized (no arc/gas/current).
# Neither persists across process restarts (both default False), so a real arc must be armed
# deliberately each session. Current/voltage normally come from the WebApp welding process
# (WELD_ARC_NUM) over AO0/AO1; set WELD_CURRENT/WELD_VOLTAGE only to override.
WELD_ENABLED = False
WELD_LIVE = False          # False = dry weld (identical motion, nothing energized)
WELD_IO = 0                # ioType: 0=controller IO, 1=extended IO
WELD_ARC_NUM = 1           # WebApp welding process number (ARCStart arcNum)
WELD_CURRENT = None        # A via AO0; None = use the WebApp process
WELD_VOLTAGE = None        # V via AO1; None = use the WebApp process
WELD_GAS = False           # open shielding gas (default off, for gasless flux-cored)
ARC_TIMEOUT_MS = 10000     # ARCStart/ARCEnd strike/extinguish timeout (ms)


def weld_settings():
    """Current weld configuration as a plain dict (enabled/live/io/arc_num/current/voltage/gas)."""
    return {"enabled": WELD_ENABLED, "live": WELD_LIVE, "io": WELD_IO,
            "arc_num": WELD_ARC_NUM, "current_a": WELD_CURRENT, "voltage_v": WELD_VOLTAGE,
            "gas": WELD_GAS}


def configure_weld(enabled=None, live=None, io=None, arc_num=None, current=None, voltage=None,
                   gas=None):
    """Enable/disable welding and set the arc parameters. Only the given fields change.

    enabled turns the arc sequence on for the seam trace; live=True energizes a REAL arc (vs a
    dry weld — identical motion, nothing energized). current/voltage are overrides (None = use
    the WebApp WELD_ARC_NUM process). Returns the resulting weld_settings(). Enabling a live arc
    logs a prominent warning.
    """
    global WELD_ENABLED, WELD_LIVE, WELD_IO, WELD_ARC_NUM, WELD_CURRENT, WELD_VOLTAGE, WELD_GAS
    if enabled is not None:
        WELD_ENABLED = bool(enabled)
    if live is not None:
        WELD_LIVE = bool(live)
        if WELD_LIVE:
            log.warning("robot: LIVE ARC ARMED — the next seam trace will strike a REAL arc")
    if io is not None:
        WELD_IO = int(io)
    if arc_num is not None:
        WELD_ARC_NUM = int(arc_num)
    if current is not None:
        WELD_CURRENT = float(current)
    if voltage is not None:
        WELD_VOLTAGE = float(voltage)
    if gas is not None:
        WELD_GAS = bool(gas)
    log.info("robot: weld %s | %s io=%d arc_num=%d current=%s voltage=%s gas=%s",
             "ON" if WELD_ENABLED else "OFF", "LIVE" if WELD_LIVE else "dry",
             WELD_IO, WELD_ARC_NUM, WELD_CURRENT, WELD_VOLTAGE, WELD_GAS)
    return weld_settings()


def arc_start(robot):
    """Strike the welding arc for the stroke. Returns True if OK (mirrors weld_start in the reference).

    DRY unless WELD_LIVE: when live is False this logs the steps but energizes nothing, so the
    motion is identical to a real pass with no arc/gas/current. Current/voltage normally come
    from the WebApp welding process (WELD_ARC_NUM) over AO0/AO1; only set here if overridden.
    """
    if not WELD_LIVE:
        log.info("robot: [dry weld] would set current/voltage, open gas, ARCStart (nothing energized)")
        return True
    if WELD_CURRENT is not None:
        robot.WeldingSetCurrent(WELD_IO, WELD_CURRENT, 0, 0)   # AO0 = current
    if WELD_VOLTAGE is not None:
        robot.WeldingSetVoltage(WELD_IO, WELD_VOLTAGE, 1, 0)   # AO1 = voltage
    if WELD_GAS:
        log.info("robot: gas ON")
        robot.SetAspirated(WELD_IO, 1)
    rc = robot.ARCStart(WELD_IO, WELD_ARC_NUM, ARC_TIMEOUT_MS)
    log.info("robot: ARCStart(io=%d arc_num=%d) -> %s", WELD_IO, WELD_ARC_NUM, rc)
    return rc == 0


def arc_end(robot):
    """End the welding arc and shut gas (mirrors weld_end). Safe to call even if never struck."""
    if not WELD_LIVE:
        log.info("robot: [dry weld] would ARCEnd + close gas")
        return
    rc = robot.ARCEnd(WELD_IO, WELD_ARC_NUM, ARC_TIMEOUT_MS)
    log.info("robot: ARCEnd(io=%d arc_num=%d) -> %s", WELD_IO, WELD_ARC_NUM, rc)
    if WELD_GAS:
        robot.SetAspirated(WELD_IO, 0)
        log.info("robot: gas OFF")


def _movel_speed_kwargs(operation=False):
    """MoveL keyword args for a linear move — TRANSPORTATION speed unless operation=True.

    operation=False (default): the TRANSPORTATION speed — always percentage, vel=TRANSPORT_VEL
    (0-100). Used for jogs and the hover / descend / retract positioning legs.

    operation=True: the OPERATION speed in the active mode, used only for working strokes
    (seam/line traverse, red-dot descent). "physical" -> vel/acc pinned to 100 (full scale,
    required on this firmware or the move errors 183) with the real speed in ovl (mm/s) and
    acceleration in oacc (mm/s^2); "percentage" -> vel=CURRENT_VEL (0-100).
    """
    if not operation:
        return {"vel": TRANSPORT_VEL}
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
def linear_move(robot, tool, user, x, y, z, rx=None, ry=None, rz=None, dry_run=False,
                operation=False):
    """Straight-line MoveL to (x, y, z) in the base frame, with flexible orientation.

    Pass rx/ry/rz (degrees) to reorient the tool to that RPY along the line; omit
    them to hold the current orientation. Speed comes from _movel_speed_kwargs: the
    TRANSPORTATION speed by default (positioning/jogs), or the OPERATION speed when
    operation=True (working strokes only — seam/line traverse, red-dot descent). dry_run
    IK-checks the target and returns 0 without moving. Returns the SDK error code (0 = success).
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

    speed = _movel_speed_kwargs(operation)
    log.info("robot: MoveL target %s | %s speed=%s (dry_run=%s)",
             [round(v, 1) for v in target],
             ("operation/" + VEL_MODE) if operation else "transport", speed, dry_run)

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


def weave_plan(path_len_mm, speed_mms):
    """Resolve the active spacing to concrete numbers for a given seam length + speed.

    Pure computation (no robot). Returns a dict {freq_hz, cycles, pitch_mm} where — whichever
    spacing mode is set — the OTHER quantity is derived from the seam length, so the automatic
    value is visible before/without moving:
      * "pitch"  -- pitch is fixed; cycles = path_len / pitch; freq = speed / pitch.
      * "cycles" -- cycles is fixed; pitch = path_len / cycles; freq = cycles * speed / len.
    Returns None if the seam length or speed is non-positive (no derivable swing).
    """
    if path_len_mm < 1e-3 or not speed_mms or speed_mms <= 0:
        return None
    if WEAVE_SPACING == "pitch":
        pitch = WEAVE_PITCH
        cycles = path_len_mm / pitch
        freq = speed_mms / pitch
    else:
        cycles = WEAVE_COUNT
        pitch = path_len_mm / cycles
        freq = cycles * speed_mms / path_len_mm
    return {"freq_hz": freq, "cycles": cycles, "pitch_mm": pitch}


def weave_start(robot, path_len_mm, speed_mms):
    """Configure + begin the weave oscillation for the upcoming traverse. Returns True if oscillating.

    Derives the swing frequency from the active spacing mode (see weave_plan), then
    WeaveSetPara + WeaveStart. Requires a physical travel speed (mm/s) and a non-zero path —
    returns False (no oscillation) otherwise. Once started, the oscillation rides every
    subsequent MoveL until weave_end. Mirrors red_line_viewer.weave_start in the farino_app
    reference.
    """
    plan = weave_plan(path_len_mm, speed_mms)
    if plan is None:
        log.warning("robot: weave skipped — needs a physical mm/s speed and a non-zero path")
        return False
    rc = robot.WeaveSetPara(WEAVE_NUM, WEAVE_TYPE, plan["freq_hz"], 0, WEAVE_RANGE,
                            0, 0, 0, 0, 0, 0, 0)
    log.info("robot: WeaveSetPara(num=%d type=%d freq=%.3fHz range=%.1fmm) -> %s "
             "(%s: %.1f cycles @ %.2f mm/cycle over %.0f mm at %.0f mm/s)",
             WEAVE_NUM, WEAVE_TYPE, plan["freq_hz"], WEAVE_RANGE, rc, WEAVE_SPACING,
             plan["cycles"], plan["pitch_mm"], path_len_mm, speed_mms)
    if rc != 0:
        return False
    rc = robot.WeaveStart(WEAVE_NUM)
    log.info("robot: WeaveStart(%d) -> %s", WEAVE_NUM, rc)
    return rc == 0


def weave_end(robot):
    """End the weave oscillation. Safe to call even if it never started. Returns the SDK code."""
    rc = robot.WeaveEnd(WEAVE_NUM)
    log.info("robot: WeaveEnd(%d) -> %s", WEAVE_NUM, rc)
    return rc


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
