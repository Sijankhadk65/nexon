"""LangChain tools the Claude orchestrator can call — the robot's senses and actions.

Two families:

  - vision (`detect_objects`): looks through the Gemini 336L and runs the configured
    open-vocabulary detector, optionally measuring real-world size from the depth
    sensor. The camera/detector/preview are owned by the shared VisionHub (vision.py).
  - motion (`get_robot_pose`, `robot_go_home`, `robot_move_to`, `robot_move_relative`,
    `robot_move_direction`, `robot_move_joints`): drives the Fairino arm via robot.py.
    Each opens a fresh connection, runs one move, and closes it — safe to repeat.
  - speed (`get_velocity_mode`, `set_robot_velocity`, `set_velocity_mode`,
    `set_physical_velocity`, `set_transport_velocity`): TWO shared speeds in robot.py, both
    persistent. The OPERATION speed (percentage or physical mm/s per the operation mode)
    drives ONLY the working strokes — the seam traverse, the red-line traverse, and the
    red-dot descent. The TRANSPORTATION speed (always a percentage, default 10%) drives every
    other move — jogs, joint moves, and the hover/descend/retract positioning legs — so a
    fast weld speed never leaks into repositioning.
  - axis locks (`set_axis_movement`): per-axis X/Y/Z enable flags in robot.py. Each linear
    move checks them and holds any locked axis fixed, so motion can be restricted to
    chosen axes.
  - weave (`set_weave`, `get_weave_settings`): a welding weave (side-to-side oscillation)
    overlaid on the follow_seam / follow_red_line traverse — WeaveSetPara -> WeaveStart ->
    traverse -> WeaveEnd in robot.py. Config (on/off, pattern, amplitude, and spacing as
    either a fixed mm pitch or a cycle count over the seam) persists; the swing rides the
    physical mm/s speed (needs the physical velocity mode). Motion only.
  - eye-to-hand (`move_to_detection`): detect an object, map its pixel+depth to a base-frame
    XYZ via the calibrated camera->base extrinsic (robot.pixel_to_base / cam_to_base), and
    hover the tool over it. Needs T_base_cam.npy from calibrate_extrinsic.py.
  - red marker (`find_red_marker`, `find_red_markers`, `move_to_red_marker`): classical
    color segmentation (marker.py) to see and go to red dot(s)/marker(s) — the neural
    detector can't see a small color blob, so anything red goes through these, not
    detect_objects. move_to_red_marker visits every marker in view, nearest first.
  - red line (`detect_red_line`, `detect_red_lines`, `follow_red_line`): the color analog of
    the seam tools — segments line-shaped RED region(s) (marker.py, no AOI), samples each
    centerline into ordered waypoints, and traces the CURVE (hover -> descend -> traverse each
    waypoint -> retract). detect_red_lines lists every line; follow_red_line traces them all
    along a greedy nearest route. Use for drawn red line(s) / red tape, straight or curved.
  - seam (`set_seam_aoi`, `detect_seam`, `follow_seam`, `follow_saved_seam`): set the scan
    region (AOI) in pixels, then geometrically detect a bare-metal seam (the joint
    between two parts) in the DEPTH map within a configured AOI (seam.py), map both endpoints
    to the base frame, and trace it (approach a LEAD-IN point just before the seam start ->
    move to P1 -> traverse to P2 -> retract), held a fixed standoff ABOVE the seam. follow_seam
    is motion only UNLESS welding is enabled (set_weld), in which case it strikes the arc at the
    lead-in and welds P1->P2 (a DRY weld — identical motion, nothing energized — unless a live
    arc was armed).
  - weld (`set_weld`, `arm_live_arc`, `disarm_live_arc`, `get_weld_settings`): arc-welding
    config in robot.py (ARCStart/ARCEnd, ported from red_line_viewer). `set_weld` enables the
    arc sequence as a DRY weld; striking a REAL arc additionally needs `arm_live_arc`, which
    a human operator must approve at the machine — the agent cannot consent for itself. Both
    default off and never persist across restarts. Only the seam traces weld; follow_red_line
    never does. Every tool that moves or reconfigures the arm goes through nexon.controller,
    which serialises motion and refuses mode changes mid-pass.
    follow_saved_seam traces a seam previously captured with 'w' in `uv run python -m nexon.perception.seam`
    (seam.json) instead of detecting live — a repeatable pass, valid while the camera hasn't
    moved.

Every move accepts dry_run=True to plan + IK-check the target WITHOUT moving the arm;
prefer it first when a target might be out of reach. Call `shutdown()` on exit to
release the camera.
"""

import functools
import json

import numpy as np
from langchain_core.tools import tool

from nexon import robot
from nexon.controller import Busy, Denied, NotArmable, get_controller
from nexon.perception import seam

# Seam trace defaults, shared by detect_seam (preview) and follow_seam / follow_saved_seam so
# the previewed lead-in / standoff match what the trace actually does.
SEAM_STANDOFF_MM = 10.0   # height held ABOVE the detected seam surface for the whole trace
SEAM_LEAD_IN_MM = 2.0     # -Y base-frame offset of the lead-in point before the seam start (very close to P1)
from nexon.perception import vision


# --------------------------------------------------------------------------- #
# Everything that changes or moves the machine goes through the Controller
# --------------------------------------------------------------------------- #
# The agent is one of TWO clients now — a human at the UI is the other — so no tool may
# touch robot.py's module globals directly. The controller is the single owner: it
# serialises motion, locks out mode changes for the duration of a pass, and is the only
# thing that can arm a live arc (and only with a human's consent).
#
# Its exceptions are turned into ordinary {"error": ...} results rather than being allowed
# to escape. A refusal — "the operator declined", "a motion is already running" — is
# something the model should read, explain, and work around; an exception escaping a tool
# would instead kill the agent turn.

def _refusal(exc, **extra) -> str:
    return json.dumps({"error": str(exc), **extra})


def _serialized(fn):
    """Run this tool's body on the controller's motion worker, one pass at a time.

    Blocks the agent until the pass finishes, which is what a synchronous tool call wants.
    Busy means another motion (or the UI) already holds the arm, and comes back as a plain
    error the model can act on.
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return get_controller().run_motion(fn, *args, **kwargs).result()
        except Busy as exc:
            return _refusal(exc, busy=True)
        except Exception as exc:  # noqa: BLE001 — a raising pass must not kill the turn
            return _refusal(exc)
    return wrapper


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


@tool(parse_docstring=True)
def get_robot_pose() -> str:
    """Read the robot arm's current position. Read-only — never moves the arm.

    Use this to answer "where is the arm/tool right now" or before planning a move.
    Returns the Cartesian TCP pose [x, y, z, rx, ry, rz] (mm / degrees) in the active
    work frame, the six joint angles (degrees), and the active tool/work-object numbers.
    """
    try:
        rob, tool, user = robot.connect_and_enable()
        pose = rob.GetActualTCPPose()[1]
        joints = rob.GetActualJointPosDegree()[1]
        rob.CloseRPC()
    except Exception as exc:  # noqa: BLE001 — surface hardware errors to Claude, don't crash the chat
        return json.dumps({"error": f"robot unavailable: {exc}"})
    return json.dumps({
        "tcp_pose": [round(v, 2) for v in pose],
        "joints": [round(v, 2) for v in joints],
        "tool": tool,
        "user": user,
        "operation_velocity_mode": robot.VEL_MODE,
        "operation_velocity": robot.PHYSICAL_VEL if robot.VEL_MODE == "physical" else robot.CURRENT_VEL,
        "operation_velocity_unit": "mm/s" if robot.VEL_MODE == "physical" else "%",
        "transport_velocity_pct": robot.TRANSPORT_VEL,
        "locked_axes": robot.locked_axes(),
    })


@tool(parse_docstring=True)
@_serialized
def robot_go_home(dry_run: bool = False) -> str:
    """Move the arm to its safe home configuration via MoveJ (joint-space, no IK needed).

    Use this to park the arm or recover to a known-good pose. Home joints are
    [-90, -120, 85, -85, -90, 0] degrees. Runs at the TRANSPORTATION speed (set with
    set_transport_velocity) — homing is positioning, not a working stroke.

    Args:
        dry_run: If true, report the planned move without moving the arm.
    """
    try:
        rob, tool, user = robot.connect_and_enable()
        if dry_run:
            rob.CloseRPC()
            return json.dumps({"dry_run": True, "home_joints": robot.START_JOINTS})
        ret = rob.MoveJ(robot.START_JOINTS, tool, user, vel=robot.TRANSPORT_VEL)
        final = rob.GetActualTCPPose()[1]
        rob.CloseRPC()
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"error": f"robot unavailable: {exc}"})
    return json.dumps({"success": ret == 0, "result": ret,
                       "final_pose": [round(v, 1) for v in final]})


@tool(parse_docstring=True)
@_serialized
def robot_move_to(
    x: float,
    y: float,
    z: float,
    rx: float | None = None,
    ry: float | None = None,
    rz: float | None = None,
    down: bool = False,
    dry_run: bool = False,
) -> str:
    """Move the arm in a straight line (MoveL) to an absolute X, Y, Z (mm) position.

    Coordinates are in the active work frame. By default the current tool orientation is
    KEPT and the move is a pure straight-line translation (least erratic); if that target
    is unreachable that way, the tool is reoriented about a single base axis (roll, then
    pitch, then yaw, smallest change first) to reach it. Runs at the TRANSPORTATION speed
    (a jog, set with set_transport_velocity — NOT the operation/weld speed).
    When a target might be out of reach, call once with dry_run=True first — it IK-checks
    the target (and the reorientation fallback) and reports reachability without moving.

    Args:
        x: Target X in mm (active work frame).
        y: Target Y in mm.
        z: Target Z (height) in mm.
        rx: Optional target roll (deg) to force a specific orientation; omit to keep
            current (with the single-axis reorient fallback). Setting any of rx/ry/rz
            disables the fallback and commands exactly that orientation.
        ry: Optional target pitch (deg); omit to keep current.
        rz: Optional target yaw (deg); omit to keep current.
        down: If true, auto-solve a reachable tool-DOWN orientation at the target
            instead of keeping the current one. Ignores rx/ry/rz.
        dry_run: If true, IK-check the target and report reachability without moving.
    """
    try:
        rob, tool, user = robot.connect_and_enable()
        if down:
            ret = robot.linear_move_torch_down(rob, tool, user, x, y, z, dry_run=dry_run)
        elif rx is None and ry is None and rz is None:
            # Keep current orientation; reorient one axis only if otherwise unreachable.
            ret = robot.linear_move_keep_orientation(rob, tool, user, x, y, z, dry_run=dry_run)
        else:
            # Explicit orientation requested — command exactly that, no fallback.
            ret = robot.linear_move(rob, tool, user, x, y, z, rx=rx, ry=ry, rz=rz,
                                    dry_run=dry_run)
        result = {"dry_run": dry_run, "result": ret, "success": ret == 0,
                  "locked_axes": robot.locked_axes()}
        if not dry_run:
            result["final_pose"] = [round(v, 1) for v in rob.GetActualTCPPose()[1]]
        rob.CloseRPC()
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"error": f"robot unavailable: {exc}"})
    return json.dumps(result)


@tool(parse_docstring=True)
@_serialized
def robot_move_relative(
    dx: float = 0.0,
    dy: float = 0.0,
    dz: float = 0.0,
    dry_run: bool = False,
) -> str:
    """Move the arm in a straight line (MoveL) relative to where it is now, by dx/dy/dz (mm).

    Use this for "move up 20 mm" or an explicit axis offset. Orientation is preserved, and
    the move runs at the TRANSPORTATION speed (a jog, set with set_transport_velocity). For
    "move left/right/forward/back" prefer robot_move_direction, which maps those words to the
    fixed base axes (right=+X, forward=+Y).

    Args:
        dx: Offset along base X in mm (default 0).
        dy: Offset along base Y in mm (default 0).
        dz: Offset along base Z (height) in mm (default 0).
        dry_run: If true, IK-check the destination without moving.
    """
    try:
        rob, tool, user = robot.connect_and_enable()
        start = rob.GetActualTCPPose()[1]
        x, y, z = start[0] + dx, start[1] + dy, start[2] + dz
        ret = robot.linear_move(rob, tool, user, x, y, z, dry_run=dry_run)
        result = {"dry_run": dry_run, "result": ret, "success": ret == 0,
                  "start_pose": [round(v, 1) for v in start],
                  "locked_axes": robot.locked_axes()}
        if not dry_run:
            result["final_pose"] = [round(v, 1) for v in rob.GetActualTCPPose()[1]]
        rob.CloseRPC()
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"error": f"robot unavailable: {exc}"})
    return json.dumps(result)


@tool(parse_docstring=True)
@_serialized
def robot_move_direction(
    direction: str,
    distance_mm: float,
    dry_run: bool = False,
) -> str:
    """Move the arm left/right/forward/back along the fixed BASE axes, straight (MoveL).

    USE THIS for any "move left/right" or "move forward/back (front/behind)" request. The
    mapping is fixed in the robot base frame, matching the operator's convention:
      right = +X, left = -X, forward/front = +Y, back/backward = -Y.
    Left/right move along base X, front/back along base Y; height (Z) is never changed. The
    move is a straight base-frame line with orientation preserved, at the TRANSPORTATION
    speed (a jog, set with set_transport_velocity). For up/down or an explicit axis offset use
    robot_move_relative.

    Args:
        direction: "left", "right", "forward" (or "front"), or "back" (or "backward").
        distance_mm: How far to travel, in mm.
        dry_run: If true, IK-check the destination without moving.
    """
    if str(direction).lower() not in robot.BASE_DIRS:
        return json.dumps({"error": f"direction must be one of "
                           f"{sorted(robot.BASE_DIRS)}, got {direction!r}"})
    try:
        rob, tool, user = robot.connect_and_enable()
        start = rob.GetActualTCPPose()[1]
        ret = robot.move_base_direction(rob, tool, user, direction, distance_mm,
                                        dry_run=dry_run)
        result = {"direction": direction, "distance_mm": distance_mm,
                  "dry_run": dry_run, "result": ret, "success": ret == 0,
                  "start_pose": [round(v, 1) for v in start],
                  "locked_axes": robot.locked_axes()}
        if not dry_run:
            result["final_pose"] = [round(v, 1) for v in rob.GetActualTCPPose()[1]]
        rob.CloseRPC()
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"error": f"robot unavailable: {exc}"})
    return json.dumps(result)


@tool(parse_docstring=True)
@_serialized
def robot_move_joints(
    j1: float, j2: float, j3: float, j4: float, j5: float, j6: float,
    dry_run: bool = False,
) -> str:
    """Move the arm to six absolute joint angles (deg) via MoveJ (joint-space, no IK needed).

    Use this when the user specifies joint angles directly, or to reach a known joint
    configuration. Runs at the TRANSPORTATION speed (set with set_transport_velocity). For
    Cartesian targets use robot_move_to instead.

    Args:
        j1: Joint 1 angle in degrees.
        j2: Joint 2 angle in degrees.
        j3: Joint 3 angle in degrees.
        j4: Joint 4 angle in degrees.
        j5: Joint 5 angle in degrees.
        j6: Joint 6 angle in degrees.
        dry_run: If true, report the planned move without moving the arm.
    """
    target = [j1, j2, j3, j4, j5, j6]
    try:
        rob, tool, user = robot.connect_and_enable()
        if dry_run:
            rob.CloseRPC()
            return json.dumps({"dry_run": True, "target_joints": target})
        ret = rob.MoveJ(target, tool, user, vel=robot.TRANSPORT_VEL)
        final = rob.GetActualTCPPose()[1]
        rob.CloseRPC()
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"error": f"robot unavailable: {exc}"})
    return json.dumps({"success": ret == 0, "result": ret,
                       "final_pose": [round(v, 1) for v in final]})


@tool(parse_docstring=True)
def set_robot_velocity(velocity: float) -> str:
    """Set the OPERATION speed as a PERCENTAGE (the working-stroke speed).

    Sets the operation speed used only for the working strokes — the seam traverse, the
    red-line traverse, and the descent onto a red dot — while the operation mode is
    "percentage". Use this when the user asks to change the WORK/weld speed as a percentage
    ("weld at 40%"). It does NOT affect jogs or positioning moves — those use the separate
    transportation speed (set_transport_velocity). The value persists; it does not move the
    arm. To set the operation speed in mm/s instead, use set_velocity_mode("physical") +
    set_physical_velocity.

    Args:
        velocity: Operation speed as a percentage of max, clamped to 1–100. Lower is safer.
    """
    try:
        state = get_controller().set_velocity(velocity)
    except (TypeError, ValueError) as exc:
        return json.dumps({"error": f"invalid velocity: {exc}"})
    except Busy as exc:
        return _refusal(exc, busy=True)
    return json.dumps({"success": True, "operation_velocity": state.velocity["percentage"],
                       "mode": state.velocity["mode"]})


@tool(parse_docstring=True)
def get_velocity_mode() -> str:
    """Read the arm's OPERATION mode/speed and the TRANSPORTATION speed. Read-only — never moves.

    Use this whenever the user asks about the current speed or mode ("what mode are we in",
    "what speed is it set to"). Two speeds are reported: the OPERATION speed (mode + value,
    used only for the seam/line traverse and red-dot descent) and the TRANSPORTATION speed
    (a percentage, used for every jog / positioning move).
    """
    return json.dumps({
        "operation_mode": robot.VEL_MODE,
        "operation_velocity": robot.PHYSICAL_VEL if robot.VEL_MODE == "physical" else robot.CURRENT_VEL,
        "operation_unit": "mm/s" if robot.VEL_MODE == "physical" else "%",
        "operation_percentage_velocity": robot.CURRENT_VEL,
        "operation_physical_velocity_mm_s": robot.PHYSICAL_VEL,
        "transport_velocity_pct": robot.TRANSPORT_VEL,
    })


@tool(parse_docstring=True)
def set_velocity_mode(mode: str) -> str:
    """Switch how the OPERATION speed is interpreted: percentage of max, or physical mm/s.

    Use "percentage" for a 0–100% operation speed (set with set_robot_velocity) or "physical"
    to make the working strokes travel at a real speed in mm/s (set with set_physical_velocity).
    This affects ONLY the operation speed (the seam/line traverse and red-dot descent); jogs
    and positioning always use the transportation speed (a percentage). The mode persists.
    NOTE: joint moves (robot_move_joints, robot_go_home) are angular and always use the
    transportation percentage. Does not move the arm.

    Args:
        mode: Either "percentage" or "physical".
    """
    try:
        state = get_controller().set_velocity_mode(mode)
        stored = state.velocity["mode"]
    except (TypeError, ValueError) as exc:
        return json.dumps({"error": f"invalid mode: {exc}"})
    except Busy as exc:
        return _refusal(exc, busy=True)
    active = robot.PHYSICAL_VEL if stored == "physical" else robot.CURRENT_VEL
    unit = "mm/s" if stored == "physical" else "%"
    return json.dumps({"success": True, "operation_mode": stored,
                       "operation_velocity": active, "unit": unit})


@tool(parse_docstring=True)
def set_physical_velocity(velocity_mm_s: float) -> str:
    """Set the OPERATION speed in mm/s (used when the operation mode is physical).

    Use this when the user asks for a real WORK/weld travel speed ("weld at 30 mm/s"). The
    value persists and applies only to the working strokes — the seam traverse, the red-line
    traverse, and the red-dot descent — while the operation mode is "physical"; call
    set_velocity_mode("physical") to actually use it. It does NOT affect jogs or positioning
    (those use set_transport_velocity), and does not move the arm by itself.

    Args:
        velocity_mm_s: Operation TCP speed in millimetres per second, clamped to 1–250.
            Lower is safer.
    """
    try:
        state = get_controller().set_physical_velocity(velocity_mm_s)
    except (TypeError, ValueError) as exc:
        return json.dumps({"error": f"invalid velocity: {exc}"})
    except Busy as exc:
        return _refusal(exc, busy=True)
    return json.dumps({"success": True,
                       "operation_physical_velocity_mm_s": state.velocity["physical_mm_s"],
                       "mode": state.velocity["mode"]})


@tool(parse_docstring=True)
def set_transport_velocity(velocity: float) -> str:
    """Set the TRANSPORTATION speed (percentage) for jog / positioning moves.

    This is the speed for every NON-working move: jogs (robot_move_to, robot_move_relative,
    robot_move_direction), joint moves (robot_go_home, robot_move_joints), and the hover /
    approach / descend / retract positioning legs of the follow/visit tools. It is ALWAYS a
    percentage of max and defaults low (10%) for safety, so repositioning never runs at the
    operation (weld) speed. Separate from the operation speed (set_robot_velocity /
    set_physical_velocity), which only drives the seam/line traverse and red-dot descent. The
    value persists; it does not move the arm.

    Args:
        velocity: Transportation speed as a percentage of max, clamped to 1–100. Lower is safer.
    """
    try:
        state = get_controller().set_transport_velocity(velocity)
        stored = state.velocity["transport_pct"]
    except (TypeError, ValueError) as exc:
        return json.dumps({"error": f"invalid velocity: {exc}"})
    except Busy as exc:
        return _refusal(exc, busy=True)
    return json.dumps({"success": True, "transport_velocity_pct": stored})


@tool(parse_docstring=True)
def set_weave(enabled: bool, pattern: str = "", amplitude_mm: float = 0.0,
              cycles: float = 0.0, pitch_mm: float = 0.0) -> str:
    """Turn the welding WEAVE (side-to-side oscillation) on or off and set its shape.

    When weave is ON, the traverse of follow_seam / follow_red_line is overlaid with a
    Fairino weave oscillation, so the tool swings side to side as it advances along the seam —
    the motion a real weld weave makes. The spacing of the swings is set two mutually
    exclusive ways: `pitch_mm` (a fixed mm advanced per weave — constant regardless of seam
    length, the usual weld spec) OR `cycles` (a fixed number of swings fitted over the whole
    seam). Setting one selects that mode; if both are given, pitch_mm wins. The swing
    frequency is derived from that at the current PHYSICAL travel speed, so weave needs the
    velocity mode set to "physical" (mm/s) — it is skipped with a note in percentage mode.
    Weave is MOTION ONLY: it still fires no arc. Settings persist for the session and do not
    move the arm by themselves.

    Args:
        enabled: True to weave on the next traverse, False to go back to a straight traverse.
        pattern: Weave shape — "triangle" (planar zig-zag, default) or "sine" (smooth); also
            "vertical_triangle", "circle_cw", "circle_ccw". Blank keeps the current pattern.
        amplitude_mm: Total side-to-side swing width in mm (e.g. 4), clamped to 0.1–30.
            0 or blank keeps the current width.
        cycles: Spacing as a fixed number of full left-right swings over the WHOLE seam
            (e.g. 6), clamped to 1–200. Selects "cycles" spacing. 0 or blank leaves it unset.
        pitch_mm: Spacing as a fixed distance advanced per swing, in mm (e.g. 8), clamped to
            0.5–100 — constant regardless of seam length. Selects "pitch" spacing. 0 or blank
            leaves it unset.
    """
    try:
        settings = get_controller().set_weave(
            enabled=enabled,
            pattern=pattern or None,
            amplitude_mm=amplitude_mm or None,
            cycles=cycles or None,
            pitch_mm=pitch_mm or None,
        ).weave
    except (TypeError, ValueError) as exc:
        return json.dumps({"error": f"invalid weave setting: {exc}"})
    except Busy as exc:
        return _refusal(exc, busy=True)
    out = {"success": True, **settings}
    if enabled and robot.VEL_MODE != "physical":
        out["note"] = ("velocity mode is percentage — weave needs physical mm/s; call "
                       "set_velocity_mode('physical') so the weave can run on the traverse")
    return json.dumps(out)


@tool(parse_docstring=True)
def get_weave_settings() -> str:
    """Report the current welding WEAVE configuration. Read-only, never moves the arm.

    Returns whether weave is enabled, its pattern, side-to-side amplitude (mm) and the active
    spacing: either a fixed pitch (mm per swing) or a fixed number of cycles over the seam
    (the `spacing` field says which; both stored values are reported). When enabled and the
    velocity mode is physical, weave overlays the follow_seam / follow_red_line traverse.
    """
    return json.dumps(get_controller().snapshot().weave)


@tool(parse_docstring=True)
def set_weld(enabled: bool, arc_num: int = 0, io: int = -1,
             current: float = 0.0, voltage: float = 0.0, gas: bool = False) -> str:
    """Turn ARC WELDING on/off for the seam trace and set the arc parameters.

    When welding is ON, follow_seam / follow_saved_seam become a weld pass: descend to the
    lead-in -> ARCStart -> move to P1 -> traverse to P2 (the weld stroke; the weave rides it if
    enabled) -> ARCEnd -> retract. This tool alone always produces a DRY WELD — the motion is
    IDENTICAL and the arc steps are logged, but NOTHING is energized (no arc/gas/current). Use
    it to rehearse the full pass safely.

    There is no `live` argument. Striking a REAL arc is a separate tool, arm_live_arc, which
    requires a human operator to consent at the machine; you cannot grant that yourself. Any
    call to set_weld also disarms a previously armed arc, so re-arming is always deliberate.

    Current/voltage come from the WebApp welding process (arc_num) over AO0/AO1 unless
    overridden. follow_red_line never welds. Settings persist for the session and never
    survive a restart; this does not move the arm.

    Args:
        enabled: True to run the arc sequence on the next seam trace, False for a plain motion trace.
        arc_num: WebApp welding process number (ARCStart arcNum) that sets current/voltage. 0 or blank keeps the current value.
        io: ioType for weld signals — 0=controller IO, 1=extended IO. -1 or blank keeps the current value.
        current: OVERRIDE welding current in amps (via AO0). 0 or blank = use the arc_num process.
        voltage: OVERRIDE welding voltage in volts (via AO1). 0 or blank = use the arc_num process.
        gas: True to open shielding gas during the stroke; default False (gasless flux-cored).
    """
    try:
        state = get_controller().set_weld(
            enabled=enabled,
            arc_num=arc_num or None,
            io=None if io < 0 else io,
            current=current or None,
            voltage=voltage or None,
            gas=gas,
        )
    except (TypeError, ValueError) as exc:
        return json.dumps({"error": f"invalid weld setting: {exc}"})
    except Busy as exc:
        return _refusal(exc, busy=True)
    out = {"success": True, **state.weld}
    if state.dry_weld:
        out["note"] = ("DRY weld — the next seam trace runs the full arc sequence with "
                       "nothing energized. Use arm_live_arc to strike a real arc.")
    return json.dumps(out)


@tool(parse_docstring=True)
def arm_live_arc(reason: str) -> str:
    """Ask the human operator for permission to strike a REAL welding arc. SAFETY-CRITICAL.

    Call this ONLY when the user has clearly asked to actually weld. It does not arm anything
    by itself: it puts the request to the operator at the machine, who must consent before the
    arc is armed. You cannot approve it, and there is no argument that bypasses the prompt —
    this blocks until a person answers, and a refusal comes back as an ordinary error.

    Requires welding to be enabled first (set_weld(enabled=True)), so that the pass has been
    rehearsed as a dry weld. Once armed, the NEXT follow_seam / follow_saved_seam strikes a
    real arc. Arming does not move the arm, and cannot be done while a motion is running.
    Use disarm_live_arc to stand down, and get_weld_settings to check the state.

    Args:
        reason: What you intend to weld, in one line — shown to the operator so they know
            what they are approving (e.g. "weld the 85 mm butt joint on the saved seam").
    """
    try:
        state = get_controller().arm_live_arc(reason)
    except Busy as exc:
        return _refusal(exc, busy=True)
    except NotArmable as exc:
        return _refusal(exc, armed=False)
    except Denied as exc:
        return _refusal(exc, armed=False, denied=True)
    return json.dumps({"success": True, "armed": state.live_armed, **state.weld,
                       "warning": "LIVE ARC ARMED — the next seam trace strikes a REAL arc."})


@tool(parse_docstring=True)
def disarm_live_arc() -> str:
    """Stand down a live arc. Always safe, always allowed, and safe to call twice.

    Leaves welding enabled, so the next seam trace is a DRY weld (identical motion, nothing
    energized) rather than silently becoming a motion-only pass. Works even while a motion is
    running. Call this whenever the user changes their mind, expresses doubt, or the plan
    changes after arming.
    """
    state = get_controller().disarm()
    return json.dumps({"success": True, "armed": state.live_armed, **state.weld})


@tool(parse_docstring=True)
def get_weld_settings() -> str:
    """Report the current ARC WELDING configuration. Read-only, never moves the arm.

    Returns whether welding is enabled, whether a live arc is ARMED (a real arc, requiring
    operator consent via arm_live_arc) or the pass is a dry weld, the arc parameters (io,
    arc_num, current/voltage overrides, gas), and whether a motion is currently running.
    """
    state = get_controller().snapshot()
    return json.dumps({**state.weld, "armed": state.live_armed,
                       "dry_weld": state.dry_weld, "busy": state.busy})


@tool(parse_docstring=True)
def set_axis_movement(axis: str, enabled: bool) -> str:
    """Lock or unlock the arm's movement along a base-frame axis (X, Y, or Z).

    Use this to restrict motion to certain axes: set enabled=False to prevent the tool
    from moving along an axis ("don't move in Z", "lock the X axis"), or enabled=True to
    allow it again. The flag persists and is checked on every linear move (robot_move_to,
    robot_move_relative, robot_move_direction) — any requested motion along a locked axis is
    suppressed, holding that coordinate fixed while the other axes still move. Joint moves
    (robot_move_joints, robot_go_home) are angular and are NOT affected. Does not move the
    arm by itself.

    Args:
        axis: Which base-frame axis to change: "x", "y", or "z".
        enabled: True to allow movement along the axis, False to lock it.
    """
    try:
        get_controller().set_axis_enabled(axis, enabled)
        stored = robot.AXIS_ENABLED[str(axis).lower()]
    except Busy as exc:
        return _refusal(exc, busy=True)
    except (TypeError, ValueError, AttributeError) as exc:
        return json.dumps({"error": f"invalid axis: {exc}"})
    return json.dumps({"success": True, "axis": str(axis).lower(), "enabled": stored,
                       "locked_axes": robot.locked_axes()})


@tool(parse_docstring=True)
@_serialized
def move_to_detection(
    target: str,
    hover_mm: float = 100.0,
    descend_mm: float = 0.0,
    min_confidence: float = 0.4,
    dry_run: bool = False,
) -> str:
    """Find something with the camera and move the tool over its real 3D position.

    Detects `target`, converts the detected pixel + depth into a base-frame XYZ using
    the calibrated camera->base transform, then moves the tool to hover `hover_mm` above
    that point, and optionally descends `descend_mm` after. Motion KEEPS the current tool
    orientation (a straight translation, least erratic); only if the target is unreachable
    that way does it reorient about a single axis. This is the "go to where the camera
    sees it" action. Requires a saved extrinsic (run calibrate_extrinsic.py first).
    Runs at the TRANSPORTATION speed (positioning); axis locks apply.

    ALWAYS prefer dry_run=True first: it locates the object and IK-checks the hover
    target without moving. Then run with dry_run=False.

    Args:
        target: What to look for, in plain language (e.g. "metal tube", "seam marker").
        hover_mm: Height to hover above the detected point, in mm (default 100).
        descend_mm: After hovering, descend this many mm straight down (default 0 = hover only).
        min_confidence: Minimum detection confidence 0-1 (default 0.4).
        dry_run: If true, locate + IK-check the hover without moving the arm.
    """
    try:
        loc = vision.get_hub().locate(target, min_confidence=min_confidence)
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"error": f"vision unavailable: {exc}"})
    if "error" in loc:
        return json.dumps(loc)
    meta = {"target": target, "confidence": loc["confidence"], "center_px": loc["center"]}
    return _hover_over_cam_xyz(loc["cam_xyz_mm"], hover_mm, descend_mm, dry_run, meta)


def _hover_over_cam_xyz(cam_xyz, hover_mm, descend_mm, dry_run, meta, operation=False):
    """Map a camera-frame XYZ to the base frame and hover the tool over it, then descend.

    Shared by move_to_detection and move_to_red_marker. Uses the keep-current-orientation
    strategy (least erratic) and returns a JSON string result (with `meta` merged in). The
    hover is always TRANSPORTATION speed; with operation=True the DESCENT onto the target
    runs at the OPERATION speed (used for the red-dot placement).
    """
    try:
        base = robot.cam_to_base(cam_xyz)
    except FileNotFoundError as exc:
        return json.dumps({"error": str(exc)})
    bx, by, bz = float(base[0]), float(base[1]), float(base[2])
    hover_z = bz + hover_mm
    result = dict(meta)
    result.update({
        "base_xyz_mm": [round(bx, 1), round(by, 1), round(bz, 1)],
        "hover_xyz_mm": [round(bx, 1), round(by, 1), round(hover_z, 1)],
        "descend_mm": descend_mm,
        "dry_run": dry_run,
    })
    try:
        rob, tool, user = robot.connect_and_enable()
        ret = robot.linear_move_keep_orientation(rob, tool, user, bx, by, hover_z, dry_run=dry_run)
        result["hover_result"] = ret
        result["hover_success"] = ret == 0
        if not dry_run and ret == 0 and descend_mm > 0:
            ret2 = robot.linear_move(rob, tool, user, bx, by, hover_z - descend_mm,
                                     operation=operation)
            result["descend_result"] = ret2
            result["descend_success"] = ret2 == 0
        if not dry_run:
            result["final_pose"] = [round(v, 1) for v in rob.GetActualTCPPose()[1]]
        rob.CloseRPC()
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"error": f"robot unavailable: {exc}"})
    return json.dumps(result)


def _trace_polyline_base(points, hover_mm, dry_run, meta, standoff_mm=0.0, lead_in=None,
                         allow_weld=False):
    """Trace an ordered polyline of base-frame points: hover->descend->traverse each->retract.

    Shared by follow_seam and follow_red_line. `points` is an ordered list of length-3
    base-frame XYZ (mm), head->tail (>=2; a straight line is just two). Approaches above the
    entry, descends to it, does a straight MoveL to every subsequent point in turn (following a
    curve as a chain of short segments), then retracts above the last. Keeps the current tool
    orientation (reorienting one axis only on the approach if it's unreachable). MOTION ONLY —
    never fires an arc/weld output. Returns a JSON string (with `meta` merged in).

    standoff_mm raises the WHOLE trace this far above each point's detected Z, so the tool
    follows the seam at a constant clearance instead of touching it. lead_in, if given, is a
    base-frame XYZ reached FIRST (before the first seam point) at the same standoff height —
    the pre-seam approach point (later the arc-set point). Its own standoff is added too, so
    pass it at the seam Z. Only the seam legs (points[0]->points[-1]) are the operation stroke;
    the lead-in->first-point move is positioning (transport speed).
    """
    sz = float(standoff_mm)
    pts = [[float(p[0]), float(p[1]), float(p[2]) + sz] for p in points]  # trace standoff above
    first, last = pts[0], pts[-1]
    waypoints = []
    if lead_in is not None:
        li = [float(lead_in[0]), float(lead_in[1]), float(lead_in[2]) + sz]
        waypoints.append(("hover_lead_in", li[0], li[1], li[2] + hover_mm))
        waypoints.append(("descend_lead_in", li[0], li[1], li[2]))
        waypoints.append(("move_to_p1", first[0], first[1], first[2]))
    else:
        waypoints.append(("hover_p1", first[0], first[1], first[2] + hover_mm))
        waypoints.append(("descend_p1", first[0], first[1], first[2]))
    for i, p in enumerate(pts[1:], start=1):
        waypoints.append((f"traverse_{i}", p[0], p[1], p[2]))
    waypoints.append(("retract", last[0], last[1], last[2] + hover_mm))
    length = float(sum(np.linalg.norm(np.array(pts[i + 1]) - np.array(pts[i]))
                       for i in range(len(pts) - 1)))
    result = dict(meta)
    result.update({
        "p1_base_mm": [round(v, 1) for v in first],
        "p2_base_mm": [round(v, 1) for v in last],
        "num_waypoints": len(pts),
        "length_mm": round(length, 1),
        "hover_mm": hover_mm, "standoff_mm": round(sz, 1),
        "dry_run": dry_run, "motion_only": True, "steps": [],
    })
    if lead_in is not None:
        result["lead_in_base_mm"] = [round(li[0], 1), round(li[1], 1), round(li[2], 1)]
    # Weave: overlay a side-to-side oscillation on the TRAVERSE legs (not the hover/descend/
    # retract). It rides a physical mm/s speed so the cycle count holds, so it only applies in
    # physical velocity mode and never on a dry run (which just IK-checks the path). The weave
    # is ALWAYS ended in the finally below, even if a leg errors, so it's never left running.
    want_weave = robot.WEAVE_ENABLED and not dry_run
    speed_mms = robot.PHYSICAL_VEL if robot.VEL_MODE == "physical" else None
    if want_weave and speed_mms is None:
        result["weave"] = {"applied": False,
                           "note": "weave needs physical velocity mode (mm/s) — "
                                   "call set_velocity_mode('physical')"}
        want_weave = False
    # Weld: strike the arc once the tool has descended to the working height (at the lead-in if
    # there is one, else at P1), keep it lit through the move-to-P1 + P1->P2 stroke, and end it
    # before the retract. Only seam traces pass allow_weld; never on a dry run. The arc is
    # ALWAYS ended in the finally, even on error, so it's never left struck. WELD_LIVE gates a
    # REAL arc vs a dry weld (identical motion, nothing energized).
    want_weld = allow_weld and robot.WELD_ENABLED and not dry_run
    result["motion_only"] = not want_weld
    try:
        rob, tool, user = robot.connect_and_enable()
        ok = True
        weaving = False
        arced = False
        try:
            for i, (label, x, y, z) in enumerate(waypoints):
                # Strike the arc before the first working leg (move-to-P1, else the first
                # traverse) — i.e. once we're at the lead-in / P1 working height. Abort the
                # pass if it fails to establish (no stroke).
                if want_weld and not arced and (label == "move_to_p1"
                                                or label.startswith("traverse")):
                    arced = robot.arc_start(rob)
                    result["weld"] = {"struck": arced, **robot.weld_settings()}
                    if not arced:
                        result["weld"]["note"] = "arc did not establish — aborted before the stroke"
                        ok = False
                        break
                # Begin the weave right before the first traverse leg (after descending to P1)
                # so the oscillation rides the whole seam-following traverse; end it after the
                # last traverse leg, before retracting straight up.
                if want_weave and not weaving and label.startswith("traverse"):
                    weaving = robot.weave_start(rob, length, speed_mms)
                    result["weave"] = {"applied": weaving, "path_len_mm": round(length, 1),
                                       "speed_mm_s": speed_mms, **robot.weave_settings()}
                    # Surface the derived spacing for THIS seam: in cycles mode the pitch is
                    # automatic (pitch = length / cycles), in pitch mode the cycle count is.
                    plan = robot.weave_plan(length, speed_mms)
                    if plan is not None:
                        result["weave"].update({
                            "effective_cycles": round(plan["cycles"], 1),
                            "effective_pitch_mm": round(plan["pitch_mm"], 2),
                            "weave_freq_hz": round(plan["freq_hz"], 3)})
                # End the stroke overlays (weave, then arc) after the last traverse leg, before
                # retracting straight up.
                if label == "retract":
                    if weaving:
                        robot.weave_end(rob)
                        weaving = False
                    if arced:
                        robot.arc_end(rob)
                        arced = False
                if i == 0:  # first waypoint (hover) may reorient a single axis if unreachable
                    ret = robot.linear_move_keep_orientation(rob, tool, user, x, y, z, dry_run=dry_run)
                else:       # rest are pure translations along the line, orientation held
                    # Only the traverse legs are the working stroke (seam/line) -> OPERATION
                    # speed; descend-to-P1 and retract are positioning -> transportation.
                    ret = robot.linear_move(rob, tool, user, x, y, z, dry_run=dry_run,
                                            operation=label.startswith("traverse"))
                result["steps"].append({"step": label,
                                        "target_mm": [round(x, 1), round(y, 1), round(z, 1)],
                                        "result": ret, "success": ret == 0})
                if ret != 0:
                    ok = False
                    break
        finally:
            if weaving:
                robot.weave_end(rob)
            if arced:
                robot.arc_end(rob)
        result["success"] = ok
        if not dry_run:
            result["final_pose"] = [round(v, 1) for v in rob.GetActualTCPPose()[1]]
        rob.CloseRPC()
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"error": f"robot unavailable: {exc}"})
    return json.dumps(result)


@tool(parse_docstring=True)
def find_red_marker() -> str:
    """Look for a RED marker (e.g. a red dot on paper) via color. Read-only, never moves.

    Use this whenever the user asks about a red dot / red marker / red sticker. It uses
    color segmentation, NOT the object detector — the detector (detect_objects) cannot see
    a small featureless color blob, so use this instead for anything red. Returns whether
    one was found, its sub-pixel pixel centre, and its base-frame XYZ if the camera is
    calibrated (T_base_cam.npy present).
    """
    try:
        loc = vision.get_hub().locate_marker()
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"error": f"vision unavailable: {exc}"})
    if "error" in loc:
        return json.dumps({"found": False, **loc})
    out = {"found": True, "center_px": loc["center"], "radius_px": loc["radius_px"]}
    try:
        base = robot.cam_to_base(loc["cam_xyz_mm"])
        out["base_xyz_mm"] = [round(float(base[0]), 1), round(float(base[1]), 1),
                              round(float(base[2]), 1)]
    except FileNotFoundError:
        out["base_xyz_mm"] = None
        out["note"] = "camera not calibrated (no T_base_cam.npy) — pixel only"
    return json.dumps(out)


@tool(parse_docstring=True)
def find_red_markers() -> str:
    """Look for ALL RED markers (e.g. red dots on paper) via color. Read-only, never moves.

    Like find_red_marker but reports EVERY red marker in view, not just one — use this when
    the user asks how many red markers there are, or to list/enumerate them. Uses color
    segmentation, NOT the object detector. Returns the count and, for each marker (largest
    first), its sub-pixel pixel centre and base-frame XYZ if the camera is calibrated
    (T_base_cam.npy present). A marker over a depth hole is listed with base_xyz_mm = null.
    """
    try:
        res = vision.get_hub().locate_markers()
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"error": f"vision unavailable: {exc}"})
    if "error" in res:
        return json.dumps({"found": False, "count": 0, **res})

    calibrated = True
    markers = []
    for m in res["markers"]:
        entry = {"center_px": m["center"], "radius_px": m["radius_px"]}
        if m.get("cam_xyz_mm") is None:
            entry["base_xyz_mm"] = None
            entry["note"] = m.get("note", "no depth there — reposition")
        else:
            try:
                base = robot.cam_to_base(m["cam_xyz_mm"])
                entry["base_xyz_mm"] = [round(float(base[0]), 1), round(float(base[1]), 1),
                                        round(float(base[2]), 1)]
            except FileNotFoundError:
                entry["base_xyz_mm"] = None
                calibrated = False
        markers.append(entry)

    out = {"found": True, "count": len(markers), "markers": markers}
    if not calibrated:
        out["note"] = "camera not calibrated (no T_base_cam.npy) — pixels only"
    return json.dumps(out)


def _order_markers_greedy_nearest(markers):
    """Order markers into a greedy nearest-neighbour ROUTE from the current TCP position.

    Starting at the live tool position, repeatedly pick the closest not-yet-visited marker
    (in base-frame XYZ), then advance the reference point TO that marker and pick the next
    closest from there — a nearest-neighbour tour, which is a shorter path than sorting once
    from the start. Markers over a depth hole (no cam_xyz_mm) have no 3D position, so they
    keep their relative order and go last (they're skipped anyway). Falls back to the input
    order (largest-first) if the extrinsic is missing or the pose can't be read — the visit
    loop then still runs, just in the original order.
    """
    located = [m for m in markers if m.get("cam_xyz_mm") is not None]
    holes = [m for m in markers if m.get("cam_xyz_mm") is None]
    if len(located) <= 1:
        return located + holes
    try:
        rob = robot.connect_readonly()
        try:
            tcp = rob.GetActualTCPPose()[1]
        finally:
            rob.CloseRPC()
        cur = [float(tcp[0]), float(tcp[1]), float(tcp[2])]
        base = [[float(c) for c in robot.cam_to_base(m["cam_xyz_mm"])] for m in located]
    except Exception:  # noqa: BLE001 — not calibrated / robot unavailable: keep largest-first
        return located + holes

    remaining = set(range(len(located)))
    ordered = []
    while remaining:
        j = min(remaining, key=lambda k: sum((base[k][i] - cur[i]) ** 2 for i in range(3)))
        ordered.append(located[j])
        remaining.discard(j)
        cur = base[j]  # advance along the route so the next pick is relative to here
    return ordered + holes


def _order_lines_greedy_nearest(lines):
    """Greedy nearest-neighbour ROUTE over multiple base-frame polylines, entering each from
    its nearer end.

    `lines` is a list of ordered base-frame point lists (each length-3 XYZ). Starting at the
    live tool position, repeatedly pick the line whose closest endpoint is nearest, orient it
    so the tool ENTERS at that endpoint (reversing the polyline if the far end is closer),
    trace to the other end, then continue from there. Returns the reordered, possibly-reversed
    polylines (as plain float lists). Falls back to the input order (longest first, unflipped)
    if the TCP pose can't be read.
    """
    pts_lists = [[[float(c) for c in p] for p in ln] for ln in lines]
    if len(pts_lists) <= 1:
        return pts_lists
    try:
        rob = robot.connect_readonly()
        try:
            tcp = rob.GetActualTCPPose()[1]
        finally:
            rob.CloseRPC()
        cur = [float(tcp[0]), float(tcp[1]), float(tcp[2])]
    except Exception:  # noqa: BLE001 — robot unavailable: keep longest-first, unflipped
        return pts_lists

    def d2(a, b):
        return sum((a[i] - b[i]) ** 2 for i in range(3))

    remaining = set(range(len(pts_lists)))
    ordered = []
    while remaining:
        # pick the line whose nearer endpoint is closest to the current tool position
        k = min(remaining, key=lambda k: min(d2(cur, pts_lists[k][0]), d2(cur, pts_lists[k][-1])))
        pts = pts_lists[k]
        if d2(cur, pts[-1]) < d2(cur, pts[0]):  # far end is closer -> enter from the tail
            pts = pts[::-1]
        ordered.append(pts)
        remaining.discard(k)
        cur = pts[-1]  # advance to this line's exit endpoint
    return ordered


@tool(parse_docstring=True)
@_serialized
def move_to_red_marker(
    hover_mm: float = 100.0,
    descend_mm: float = 0.0,
    dry_run: bool = False,
) -> str:
    """Find EVERY red marker with the camera and visit each one in turn.

    Locates all red markers by color (see find_red_markers), maps each one's pixel + depth
    to a base-frame XYZ via the calibrated camera->base transform, then visits them along a
    greedy NEAREST-NEIGHBOUR route (from the current tool position, always going to the
    closest not-yet-visited marker next): hovers `hover_mm` above each and optionally
    descends `descend_mm`. Motion keeps the current tool
    orientation (least erratic), reorienting one axis only if unreachable. Requires a saved
    extrinsic (calibrate_extrinsic.py). The hover between markers runs at the TRANSPORTATION
    speed; the descent onto each dot runs at the OPERATION speed. Axis locks apply. If only
    one marker is in view this simply visits it.

    ALWAYS prefer dry_run=True first: it locates every marker and IK-checks each hover without
    moving (all markers are checked so you see which are reachable). Then run with
    dry_run=False, which stops the sequence at the first marker whose hover fails.

    Args:
        hover_mm: Height to hover above each marker, in mm (default 100).
        descend_mm: At each marker, after hovering, descend this many mm straight down (default 0 = hover only).
        dry_run: If true, locate + IK-check every hover without moving the arm.
    """
    try:
        res = vision.get_hub().locate_markers()
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"error": f"vision unavailable: {exc}"})
    if "error" in res:
        return json.dumps(res)

    markers = _order_markers_greedy_nearest(res["markers"])
    visits = []
    stopped = False
    for i, m in enumerate(markers):
        meta = {"marker": "red", "marker_index": i + 1,
                "center_px": m["center"], "radius_px": m["radius_px"]}
        if m.get("cam_xyz_mm") is None:
            visits.append({**meta, "skipped": m.get("note", "no depth there — reposition")})
            continue
        # The descent onto the dot is the red-dot working stroke -> OPERATION speed; the
        # hover between markers stays transportation.
        visit = json.loads(_hover_over_cam_xyz(
            m["cam_xyz_mm"], hover_mm, descend_mm, dry_run, meta, operation=True))
        visits.append(visit)
        # On a real run, halt the sequence if a hover errors or IK-fails — don't keep
        # commanding motions after a failure. In dry_run, check every marker regardless.
        if not dry_run and ("error" in visit or not visit.get("hover_success")):
            stopped = True
            break

    out = {
        "marker": "red",
        "count": len(markers),
        "visited": len(visits),
        "dry_run": dry_run,
        "visits": visits,
    }
    if stopped:
        out["note"] = "stopped early: a marker's hover failed (see its visit entry)"
    return json.dumps(out)


@tool(parse_docstring=True)
def detect_red_line(num_samples: int = 12) -> str:
    """Find a RED LINE (a red-marked seam/path) by color and report its waypoints. Read-only.

    The color analog of detect_seam: segments the most line-shaped RED region in the frame —
    no AOI needed, the same color cue as find_red_marker but for an ELONGATED mark (red tape
    or a drawn red line, not a dot) — and samples its centerline into up to `num_samples`
    ordered waypoints, so a CURVED line is captured, not just its endpoints. Returns the
    waypoints as pixels and, if the camera is calibrated (T_base_cam.npy), as base-frame XYZ
    with the line's traced 3D length. Use this to see the red path before follow_red_line.

    Args:
        num_samples: How many centerline points to sample along the line (default 12; more = finer curve).
    """
    try:
        loc = vision.get_hub().locate_red_line(num_samples=num_samples)
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"error": f"vision unavailable: {exc}"})
    if "error" in loc:
        return json.dumps({"found": False, **loc})
    wps = loc["waypoints"]
    out = {"found": True, "num_waypoints": len(wps), "length_px": loc["length_px"],
           "waypoints_px": [w["px"] for w in wps]}
    try:
        base = [robot.cam_to_base(w["cam_xyz_mm"]) for w in wps]
        out["waypoints_base_mm"] = [[round(float(v), 1) for v in b] for b in base]
        out["length_mm"] = round(float(sum(np.linalg.norm(base[i + 1] - base[i])
                                            for i in range(len(base) - 1))), 1)
    except FileNotFoundError:
        out["note"] = "camera not calibrated (no T_base_cam.npy) — pixels only"
    return json.dumps(out)


@tool(parse_docstring=True)
def detect_red_lines(num_samples: int = 12) -> str:
    """Find ALL RED LINES by color and report each one's waypoints. Read-only, never moves.

    Like detect_red_line but reports EVERY red line in view (longest first), not just one — use
    this when the user asks how many red lines/paths there are, or to list them before tracing
    with follow_red_line. Each line's centerline is sampled into up to `num_samples` ordered
    waypoints (so curves are captured). Returns the count and, per line, its waypoints as pixels
    and — if calibrated (T_base_cam.npy) — as base-frame XYZ with the line's traced 3D length.

    Args:
        num_samples: How many centerline points to sample along each line (default 12; more = finer curve).
    """
    try:
        res = vision.get_hub().locate_red_lines(num_samples=num_samples)
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"error": f"vision unavailable: {exc}"})
    if "error" in res:
        return json.dumps({"found": False, "count": 0, **res})

    calibrated = True
    lines = []
    for ln in res["lines"]:
        wps = ln["waypoints"]
        entry = {"num_waypoints": len(wps), "length_px": ln["length_px"],
                 "waypoints_px": [w["px"] for w in wps]}
        try:
            base = [robot.cam_to_base(w["cam_xyz_mm"]) for w in wps]
            entry["waypoints_base_mm"] = [[round(float(v), 1) for v in b] for b in base]
            entry["length_mm"] = round(float(sum(np.linalg.norm(base[i + 1] - base[i])
                                                 for i in range(len(base) - 1))), 1)
        except FileNotFoundError:
            calibrated = False
        lines.append(entry)

    out = {"found": True, "count": len(lines), "lines": lines}
    if not calibrated:
        out["note"] = "camera not calibrated (no T_base_cam.npy) — pixels only"
    return json.dumps(out)


@tool(parse_docstring=True)
@_serialized
def follow_red_line(hover_mm: float = 60.0, num_samples: int = 12, dry_run: bool = False) -> str:
    """Trace EVERY red line in view, one after another — MOTION ONLY, never fires an arc/weld.

    The color analog of follow_seam, but handles MULTIPLE curved lines: locates all red lines
    by color (see detect_red_lines), samples each one's centerline into ordered waypoints, maps
    them to the base frame, then traces the lines along a greedy nearest-neighbour route —
    entering each line at whichever endpoint is closer. Per line it runs: hover above the entry
    point -> descend -> straight MoveL through every waypoint (tracing the curve) -> retract
    above the exit. The traverse runs at the OPERATION speed/mode (use physical mm/s for a
    real travel speed); hover/descend/retract use the transportation speed. If the weave is
    enabled (set_weave) and the operation mode is physical, each traverse is overlaid with a
    side-to-side weave oscillation. Keeps the current tool orientation (reorienting one axis
    only if an approach is unreachable); axis locks apply.
    Does NOT weld (the weave is motion only). Requires a saved extrinsic. If only one line is
    in view this simply traces it.

    ALWAYS prefer dry_run=True first: it IK-checks every waypoint of every line without moving.
    On a real run the sequence stops at the first line whose trace fails.

    Args:
        hover_mm: Approach/retract height above each line's endpoints, in mm (default 60).
        num_samples: How many centerline points to trace along each line (default 12; more = finer curve).
        dry_run: If true, IK-check all waypoints without moving the arm.
    """
    try:
        res = vision.get_hub().locate_red_lines(num_samples=num_samples)
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"error": f"vision unavailable: {exc}"})
    if "error" in res:
        return json.dumps(res)
    try:
        lines = [[robot.cam_to_base(w["cam_xyz_mm"]) for w in ln["waypoints"]]
                 for ln in res["lines"]]
    except FileNotFoundError as exc:
        return json.dumps({"error": str(exc)})

    ordered = _order_lines_greedy_nearest(lines)
    traces = []
    stopped = False
    for i, pts in enumerate(ordered):
        trace = json.loads(_trace_polyline_base(pts, hover_mm, dry_run,
                                                {"line": "red", "line_index": i + 1}))
        traces.append(trace)
        # On a real run, halt if a line's trace errors or IK-fails — don't keep commanding
        # motion after a failure. In dry_run, check every line regardless.
        if not dry_run and ("error" in trace or not trace.get("success")):
            stopped = True
            break

    out = {"line": "red", "count": len(lines), "traced": len(traces),
           "dry_run": dry_run, "traces": traces}
    if stopped:
        out["note"] = "stopped early: a line's trace failed (see its entry)"
    return json.dumps(out)


@tool(parse_docstring=True)
def set_seam_aoi(x1: int, y1: int, x2: int, y2: int) -> str:
    """Set the seam detector's area of interest (AOI) box, in image pixels.

    The AOI is the region detect_seam / follow_seam scan for the joint. Give a TIGHT box
    with its LONG side ALONG the seam (a loose box pulls in clutter and detection fails).
    Coordinates are pixels in the camera image: (0,0) is top-left, x grows right, y grows
    down. The box is saved (persists across runs) and clamped to the frame. After setting,
    this reports the frame size and whether a seam is detected there now, so you can adjust.
    If you don't know the pixel range, the returned frame_size gives it.

    Args:
        x1: Left edge X of the box, in pixels.
        y1: Top edge Y of the box, in pixels.
        x2: Right edge X of the box, in pixels.
        y2: Bottom edge Y of the box, in pixels.
    """
    ax1, ay1 = int(min(x1, x2)), int(min(y1, y2))
    ax2, ay2 = int(max(x1, x2)), int(max(y1, y2))
    hub = vision.get_hub()
    try:
        size = hub.frame_size()
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"error": f"vision unavailable: {exc}"})
    if size is not None:
        w, h = size
        ax1, ax2 = max(0, min(ax1, w - 1)), max(1, min(ax2, w))
        ay1, ay2 = max(0, min(ay1, h - 1)), max(1, min(ay2, h))
    if ax2 - ax1 < 5 or ay2 - ay1 < 5:
        return json.dumps({"error": "AOI too small — give a wider box"})

    seam.save_aoi((ax1, ay1, ax2, ay2))
    out = {"success": True, "aoi": [ax1, ay1, ax2, ay2]}
    if size is not None:
        out["frame_size"] = {"width": size[0], "height": size[1]}
    loc = hub.locate_seam()  # instant feedback: does a seam show up in this AOI?
    if "error" in loc:
        out["seam_detected"] = False
        out["detect_note"] = loc["error"]
    else:
        out["seam_detected"] = True
        out["seam_px"] = {"p1": loc["p1_px"], "p2": loc["p2_px"], "resid_px": loc.get("resid_px")}
    return json.dumps(out)


@tool(parse_docstring=True)
def detect_seam() -> str:
    """Find the seam (joint between two parts) with the camera and report its endpoints. Read-only.

    Detects the bare-metal seam geometrically in the DEPTH map within the configured area
    of interest (AOI) — the joint shows as a gap/step/crease in the surface. Returns both
    endpoints as pixels and — if the camera is calibrated (T_base_cam.npy) — as base-frame
    XYZ with the seam's 3D length. Also previews how follow_seam would trace it: the LEAD-IN
    point (P1 offset -2 mm in base Y, the pre-seam / future arc-set point) and the standoff
    height (10 mm) the trace is held above the surface. p1/p2 base XYZ are the raw on-surface
    readings; the trace runs standoff above them. Needs an AOI set via `uv run python -m nexon.perception.seam`.
    Use this to see where the seam is (and where the lead-in lands) before tracing it.
    """
    try:
        loc = vision.get_hub().locate_seam()
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"error": f"vision unavailable: {exc}"})
    if "error" in loc:
        return json.dumps({"found": False, **loc})
    out = {"found": True, "p1_px": loc["p1_px"], "p2_px": loc["p2_px"],
           "length_px": loc["length_px"]}
    try:
        b1 = robot.cam_to_base(loc["p1_cam_xyz_mm"])
        b2 = robot.cam_to_base(loc["p2_cam_xyz_mm"])
        out["p1_base_mm"] = [round(float(v), 1) for v in b1]
        out["p2_base_mm"] = [round(float(v), 1) for v in b2]
        out["length_mm"] = round(float(np.linalg.norm(b2 - b1)), 1)
        # Preview the follow_seam trace geometry: lead-in point + standoff height. The trace
        # holds SEAM_STANDOFF_MM above the surface, and starts SEAM_LEAD_IN_MM behind P1 in Y.
        out["standoff_mm"] = SEAM_STANDOFF_MM
        out["lead_in_base_mm"] = [round(float(b1[0]), 1),
                                  round(float(b1[1]) - SEAM_LEAD_IN_MM, 1),
                                  round(float(b1[2]) + SEAM_STANDOFF_MM, 1)]
        out["trace_p1_base_mm"] = [round(float(b1[0]), 1), round(float(b1[1]), 1),
                                   round(float(b1[2]) + SEAM_STANDOFF_MM, 1)]
        out["trace_p2_base_mm"] = [round(float(b2[0]), 1), round(float(b2[1]), 1),
                                   round(float(b2[2]) + SEAM_STANDOFF_MM, 1)]
    except FileNotFoundError:
        out["note"] = "camera not calibrated (no T_base_cam.npy) — pixels only"
    return json.dumps(out)


@tool(parse_docstring=True)
@_serialized
def follow_seam(hover_mm: float = 60.0, standoff_mm: float = SEAM_STANDOFF_MM,
                lead_in_mm: float = SEAM_LEAD_IN_MM, dry_run: bool = False) -> str:
    """Trace the seam (joint between two parts). Motion only UNLESS welding is enabled (set_weld).

    Locates the seam (geometrically, in depth within the AOI), maps both endpoints to the base
    frame via the calibrated camera->base transform, then runs: hover above the lead-in ->
    descend to the LEAD-IN point (P1 offset -lead_in_mm in base Y, the pre-seam / arc-set
    point) -> move to P1 -> straight traverse to P2 -> retract. The WHOLE trace is held
    `standoff_mm` above the seam's detected surface (so the tool follows the seam at a constant
    clearance, never touching it). The P1->P2 traverse runs at the OPERATION speed/mode (use
    physical mm/s for a real travel speed); the lead-in approach and positioning use the
    transportation speed. If the weave is enabled (set_weave) and the operation mode is
    physical, the P1->P2 traverse is overlaid with a side-to-side weave oscillation. Keeps the
    current tool orientation (reorienting one axis only if a waypoint is unreachable); axis
    locks apply. Requires a saved extrinsic.

    WELDING: if enabled via set_weld, this becomes a weld pass — the arc is struck at the
    lead-in, kept lit through move-to-P1 and the P1->P2 stroke, and ended before the retract. It
    is a DRY WELD (identical motion, nothing energized) unless a human operator armed a REAL
    arc via arm_live_arc. A dry_run never welds (it only IK-checks).

    ALWAYS prefer dry_run=True first: it IK-checks every waypoint without moving.

    Args:
        hover_mm: Approach/retract height above the lead-in / seam, in mm (default 60).
        standoff_mm: Height held ABOVE the detected seam for the whole trace, in mm (default 10).
        lead_in_mm: How far before the seam start to place the lead-in point, as a -Y base-frame
            offset from P1, in mm (default 2, very close to P1). 0 disables the lead-in.
        dry_run: If true, IK-check all waypoints without moving the arm.
    """
    try:
        loc = vision.get_hub().locate_seam()
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"error": f"vision unavailable: {exc}"})
    if "error" in loc:
        return json.dumps(loc)
    try:
        b1 = robot.cam_to_base(loc["p1_cam_xyz_mm"])
        b2 = robot.cam_to_base(loc["p2_cam_xyz_mm"])
    except FileNotFoundError as exc:
        return json.dumps({"error": str(exc)})
    # Lead-in: a point before the seam start, offset -lead_in_mm in base Y (later the arc-set
    # point). Same Z as P1; the standoff is added inside _trace_polyline_base.
    lead_in = None if lead_in_mm <= 0 else [b1[0], b1[1] - float(lead_in_mm), b1[2]]
    return _trace_polyline_base([b1, b2], hover_mm, dry_run, {},
                                standoff_mm=standoff_mm, lead_in=lead_in, allow_weld=True)


@tool(parse_docstring=True)
@_serialized
def follow_saved_seam(hover_mm: float = 60.0, standoff_mm: float = SEAM_STANDOFF_MM,
                      lead_in_mm: float = SEAM_LEAD_IN_MM, dry_run: bool = False) -> str:
    """Trace the SAVED seam captured in the seam.py preview. Motion only UNLESS welding is enabled.

    Loads the seam saved with 'w' in `uv run python -m nexon.perception.seam` (seam.json) instead of detecting
    live, maps its endpoints to the base frame via the calibrated camera->base transform, then
    runs the same lead-in -> move to P1 -> traverse to P2 -> retract path as follow_seam, held
    `standoff_mm` above the seam (a weave overlays the P1->P2 traverse if enabled via set_weave).
    If welding is enabled (set_weld) this becomes a weld pass just like follow_seam — the arc is
    struck at the lead-in and ended before the retract (a DRY weld unless arm_live_arc was
    approved by the operator).
    Use this to re-run a seam you captured earlier without re-detecting — the camera must NOT
    have moved since it was saved, as the endpoints are stored in the camera frame. Requires a
    saved seam (seam.json) and a saved extrinsic (T_base_cam.npy).

    ALWAYS prefer dry_run=True first: it IK-checks every waypoint without moving.

    Args:
        hover_mm: Approach/retract height above the lead-in / seam, in mm (default 60).
        standoff_mm: Height held ABOVE the detected seam for the whole trace, in mm (default 10).
        lead_in_mm: How far before the seam start to place the lead-in point, as a -Y base-frame
            offset from P1, in mm (default 2, very close to P1). 0 disables the lead-in.
        dry_run: If true, IK-check all waypoints without moving the arm.
    """
    rec = seam.load_seam()
    if rec is None:
        return json.dumps({"error": "no saved seam — capture one by pressing 'w' in "
                                    "'uv run python -m nexon.perception.seam'"})
    try:
        b1 = robot.cam_to_base(rec["p1_cam_xyz_mm"])
        b2 = robot.cam_to_base(rec["p2_cam_xyz_mm"])
    except FileNotFoundError as exc:
        return json.dumps({"error": str(exc)})
    except (KeyError, TypeError):
        return json.dumps({"error": "saved seam is missing camera-frame endpoints — "
                                    "re-save it with 'w' in seam.py"})
    lead_in = None if lead_in_mm <= 0 else [b1[0], b1[1] - float(lead_in_mm), b1[2]]
    return _trace_polyline_base([b1, b2], hover_mm, dry_run,
                                {"source": "saved_seam", "saved_at": rec.get("saved_at"),
                                 "length_mm": rec.get("length_mm")},
                                standoff_mm=standoff_mm, lead_in=lead_in, allow_weld=True)


# All tools exposed to the orchestrator. Add future robot tools here.
ALL_TOOLS = [
    detect_objects,
    find_red_marker,
    find_red_markers,
    detect_red_line,
    detect_red_lines,
    set_seam_aoi,
    detect_seam,
    get_robot_pose,
    get_velocity_mode,
    set_robot_velocity,
    set_velocity_mode,
    set_physical_velocity,
    set_transport_velocity,
    set_weave,
    get_weave_settings,
    set_weld,
    get_weld_settings,
    arm_live_arc,
    disarm_live_arc,
    set_axis_movement,
    robot_go_home,
    robot_move_to,
    robot_move_relative,
    robot_move_direction,
    robot_move_joints,
    move_to_detection,
    move_to_red_marker,
    follow_red_line,
    follow_seam,
    follow_saved_seam,
]


def shutdown():
    """Release the camera/preview at the end of the session."""
    vision.shutdown()
