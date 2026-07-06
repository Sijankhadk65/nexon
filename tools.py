"""LangChain tools the Claude orchestrator can call — the robot's senses and actions.

Two families:

  - vision (`detect_objects`): looks through the Gemini 336L and runs the configured
    open-vocabulary detector, optionally measuring real-world size from the depth
    sensor. The camera/detector/preview are owned by the shared VisionHub (vision.py).
  - motion (`get_robot_pose`, `robot_go_home`, `robot_move_to`, `robot_move_relative`,
    `robot_move_lateral`, `robot_move_joints`): drives the Fairino arm via robot.py.
    Each opens a fresh connection, runs one move, and closes it — safe to repeat.
  - speed (`get_velocity_mode`, `set_robot_velocity`, `set_velocity_mode`,
    `set_physical_velocity`): a single shared velocity in robot.py that every move reads,
    so speed changes persist. Speed is interpreted either as a physical mm/s (the default
    mode) or as a percentage of max, depending on the velocity mode; each linear move
    checks the mode and applies the matching parameters.
  - axis locks (`set_axis_movement`): per-axis X/Y/Z enable flags in robot.py. Each linear
    move checks them and holds any locked axis fixed, so motion can be restricted to
    chosen axes.

Every move accepts dry_run=True to plan + IK-check the target WITHOUT moving the arm;
prefer it first when a target might be out of reach. Call `shutdown()` on exit to
release the camera.
"""

import json

from langchain_core.tools import tool

import robot
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
        "velocity_mode": robot.VEL_MODE,
        "velocity": robot.PHYSICAL_VEL if robot.VEL_MODE == "physical" else robot.CURRENT_VEL,
        "velocity_unit": "mm/s" if robot.VEL_MODE == "physical" else "%",
        "locked_axes": robot.locked_axes(),
    })


@tool(parse_docstring=True)
def robot_go_home(dry_run: bool = False) -> str:
    """Move the arm to its safe home configuration via MoveJ (joint-space, no IK needed).

    Use this to park the arm or recover to a known-good pose. Home joints are
    [-90, -120, 85, -85, -90, 0] degrees. Runs at the current velocity (set with
    set_robot_velocity).

    Args:
        dry_run: If true, report the planned move without moving the arm.
    """
    try:
        rob, tool, user = robot.connect_and_enable()
        if dry_run:
            rob.CloseRPC()
            return json.dumps({"dry_run": True, "home_joints": robot.START_JOINTS})
        ret = rob.MoveJ(robot.START_JOINTS, tool, user, vel=robot.CURRENT_VEL)
        final = rob.GetActualTCPPose()[1]
        rob.CloseRPC()
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"error": f"robot unavailable: {exc}"})
    return json.dumps({"success": ret == 0, "result": ret,
                       "final_pose": [round(v, 1) for v in final]})


@tool(parse_docstring=True)
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

    Coordinates are in the active work frame. By default the current tool orientation
    is kept (safest). Runs at the current velocity (set with set_robot_velocity). When
    a target might be out of reach, call once with dry_run=True first — it IK-checks the
    target and reports whether it is reachable without moving.

    Args:
        x: Target X in mm (active work frame).
        y: Target Y in mm.
        z: Target Z (height) in mm.
        rx: Optional target roll (deg) to reorient the tool along the line; omit to keep current.
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
        else:
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
def robot_move_relative(
    dx: float = 0.0,
    dy: float = 0.0,
    dz: float = 0.0,
    dry_run: bool = False,
) -> str:
    """Move the arm in a straight line (MoveL) relative to where it is now, by dx/dy/dz (mm).

    Use this for "move up 20 mm", "back 50 mm", etc. Orientation is preserved, and the
    move runs at the current velocity (set with set_robot_velocity). For "move
    left/right" prefer robot_move_lateral, which respects the tool's tilt.

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
def robot_move_lateral(
    direction: str,
    distance_mm: float,
    keep_z: bool = True,
    dry_run: bool = False,
) -> str:
    """Move the arm left or right relative to the TOOL, in a straight line (MoveL).

    USE THIS for any "move left" / "move right" request. "Right" is the tool's +X
    axis, "left" its -X, resolved in the current tool frame (so it tracks the tool's
    tilt) and executed as a base-frame straight line with orientation preserved — you
    do not need to work out which base axis is "right". Runs at the current velocity
    (set with set_robot_velocity).

    Args:
        direction: "left" or "right".
        distance_mm: How far to travel, in mm.
        keep_z: If true (default) keep the move horizontal (height unchanged) even
            when the tool is tilted. Set false to follow the tool's tilt (may change Z).
        dry_run: If true, IK-check the destination without moving.
    """
    if direction not in robot.TOOL_DIRS:
        return json.dumps({"error": f"direction must be 'left' or 'right', got {direction!r}"})
    try:
        rob, tool, user = robot.connect_and_enable()
        start = rob.GetActualTCPPose()[1]
        ret = robot.move_tool_direction(rob, tool, user, direction, distance_mm,
                                        dry_run=dry_run, keep_z=keep_z)
        result = {"direction": direction, "distance_mm": distance_mm, "keep_z": keep_z,
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
def robot_move_joints(
    j1: float, j2: float, j3: float, j4: float, j5: float, j6: float,
    dry_run: bool = False,
) -> str:
    """Move the arm to six absolute joint angles (deg) via MoveJ (joint-space, no IK needed).

    Use this when the user specifies joint angles directly, or to reach a known joint
    configuration. Runs at the current velocity (set with set_robot_velocity). For
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
        ret = rob.MoveJ(target, tool, user, vel=robot.CURRENT_VEL)
        final = rob.GetActualTCPPose()[1]
        rob.CloseRPC()
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"error": f"robot unavailable: {exc}"})
    return json.dumps({"success": ret == 0, "result": ret,
                       "final_pose": [round(v, 1) for v in final]})


@tool(parse_docstring=True)
def set_robot_velocity(velocity: float) -> str:
    """Set the arm's PERCENTAGE movement speed for all subsequent moves.

    Use this whenever the user asks to change speed as a percentage ("go faster", "slow
    down", "move at 50%"). The value persists — every later move runs at this speed
    while the velocity mode is "percentage". It does not move the arm by itself. To
    command a speed in mm/s instead, use set_velocity_mode("physical") +
    set_physical_velocity.

    Args:
        velocity: Speed as a percentage of max, clamped to 1–100. Lower is safer.
    """
    try:
        stored = robot.set_velocity(velocity)
    except (TypeError, ValueError) as exc:
        return json.dumps({"error": f"invalid velocity: {exc}"})
    return json.dumps({"success": True, "velocity": stored, "mode": robot.VEL_MODE})


@tool(parse_docstring=True)
def get_velocity_mode() -> str:
    """Read the arm's current velocity mode and speed settings. Read-only — never moves.

    Use this whenever the user asks about the current speed or mode ("what mode are we
    in", "what speed is it set to"). Returns the active mode ("percentage" or
    "physical"), the speed that mode uses (with its unit), and both stored values.
    """
    return json.dumps({
        "mode": robot.VEL_MODE,
        "active_velocity": robot.PHYSICAL_VEL if robot.VEL_MODE == "physical" else robot.CURRENT_VEL,
        "unit": "mm/s" if robot.VEL_MODE == "physical" else "%",
        "percentage_velocity": robot.CURRENT_VEL,
        "physical_velocity_mm_s": robot.PHYSICAL_VEL,
    })


@tool(parse_docstring=True)
def set_velocity_mode(mode: str) -> str:
    """Switch how the arm's speed is interpreted: percentage of max, or physical mm/s.

    Use "percentage" for a 0–100% speed (set with set_robot_velocity) or "physical" to
    make linear moves travel at a real speed in mm/s (set with set_physical_velocity).
    The mode persists and every later linear move checks it. NOTE: joint moves
    (robot_move_joints, robot_go_home) are angular and always use the percentage speed
    regardless of mode. Does not move the arm.

    Args:
        mode: Either "percentage" or "physical".
    """
    try:
        stored = robot.set_velocity_mode(mode)
    except (TypeError, ValueError) as exc:
        return json.dumps({"error": f"invalid mode: {exc}"})
    active = robot.PHYSICAL_VEL if stored == "physical" else robot.CURRENT_VEL
    unit = "mm/s" if stored == "physical" else "%"
    return json.dumps({"success": True, "mode": stored, "active_velocity": active, "unit": unit})


@tool(parse_docstring=True)
def set_physical_velocity(velocity_mm_s: float) -> str:
    """Set the arm's PHYSICAL linear speed in mm/s (used when the velocity mode is physical).

    Use this when the user asks for a real travel speed ("move at 30 mm/s"). The value
    persists and applies to linear moves (robot_move_to, robot_move_relative,
    robot_move_lateral) while the velocity mode is "physical" — call
    set_velocity_mode("physical") to actually use it. It does not move the arm by itself.

    Args:
        velocity_mm_s: Linear TCP speed in millimetres per second, clamped to 1–250.
            Lower is safer.
    """
    try:
        stored = robot.set_physical_velocity(velocity_mm_s)
    except (TypeError, ValueError) as exc:
        return json.dumps({"error": f"invalid velocity: {exc}"})
    return json.dumps({"success": True, "physical_velocity_mm_s": stored, "mode": robot.VEL_MODE})


@tool(parse_docstring=True)
def set_axis_movement(axis: str, enabled: bool) -> str:
    """Lock or unlock the arm's movement along a base-frame axis (X, Y, or Z).

    Use this to restrict motion to certain axes: set enabled=False to prevent the tool
    from moving along an axis ("don't move in Z", "lock the X axis"), or enabled=True to
    allow it again. The flag persists and is checked on every linear move (robot_move_to,
    robot_move_relative, robot_move_lateral) — any requested motion along a locked axis is
    suppressed, holding that coordinate fixed while the other axes still move. Joint moves
    (robot_move_joints, robot_go_home) are angular and are NOT affected. Does not move the
    arm by itself.

    Args:
        axis: Which base-frame axis to change: "x", "y", or "z".
        enabled: True to allow movement along the axis, False to lock it.
    """
    try:
        stored = robot.set_axis_enabled(axis, enabled)
    except (TypeError, ValueError, AttributeError) as exc:
        return json.dumps({"error": f"invalid axis: {exc}"})
    return json.dumps({"success": True, "axis": str(axis).lower(), "enabled": stored,
                       "locked_axes": robot.locked_axes()})


# All tools exposed to the orchestrator. Add future robot tools here.
ALL_TOOLS = [
    detect_objects,
    get_robot_pose,
    get_velocity_mode,
    set_robot_velocity,
    set_velocity_mode,
    set_physical_velocity,
    set_axis_movement,
    robot_go_home,
    robot_move_to,
    robot_move_relative,
    robot_move_lateral,
    robot_move_joints,
]


def shutdown():
    """Release the camera/preview at the end of the session."""
    vision.shutdown()
