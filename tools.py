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
  - eye-to-hand (`move_to_detection`): detect an object, map its pixel+depth to a base-frame
    XYZ via the calibrated camera->base extrinsic (robot.pixel_to_base / cam_to_base), and
    hover the tool over it. Needs T_base_cam.npy from calibrate_extrinsic.py.
  - red marker (`find_red_marker`, `find_red_markers`, `move_to_red_marker`): classical
    color segmentation (marker.py) to see and go to red dot(s)/marker(s) — the neural
    detector can't see a small color blob, so anything red goes through these, not
    detect_objects. move_to_red_marker visits every marker in view, nearest first.
  - seam (`set_seam_aoi`, `detect_seam`, `follow_seam`): set the scan region (AOI) in pixels,
    then geometrically detect a bare-metal seam (the joint
    between two parts) in the DEPTH map within a configured AOI (seam.py), map both endpoints
    to the base frame, and trace it (hover -> descend -> traverse -> retract). follow_seam is
    MOTION ONLY — it never fires an arc/weld output; it's a safe dry run of the path.

Every move accepts dry_run=True to plan + IK-check the target WITHOUT moving the arm;
prefer it first when a target might be out of reach. Call `shutdown()` on exit to
release the camera.
"""

import json

import numpy as np
from langchain_core.tools import tool

import robot
import seam
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

    Coordinates are in the active work frame. By default the current tool orientation is
    KEPT and the move is a pure straight-line translation (least erratic); if that target
    is unreachable that way, the tool is reoriented about a single base axis (roll, then
    pitch, then yaw, smallest change first) to reach it. Runs at the current velocity.
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


@tool(parse_docstring=True)
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
    Speed/mode and axis locks apply.

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


def _hover_over_cam_xyz(cam_xyz, hover_mm, descend_mm, dry_run, meta):
    """Map a camera-frame XYZ to the base frame and hover the tool over it, then descend.

    Shared by move_to_detection and move_to_red_marker. Uses the keep-current-orientation
    strategy (least erratic) and returns a JSON string result (with `meta` merged in).
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
            ret2 = robot.linear_move(rob, tool, user, bx, by, hover_z - descend_mm)
            result["descend_result"] = ret2
            result["descend_success"] = ret2 == 0
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


@tool(parse_docstring=True)
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
    extrinsic (calibrate_extrinsic.py). Speed/mode and axis locks apply. If only one marker
    is in view this simply visits it.

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
        visit = json.loads(_hover_over_cam_xyz(
            m["cam_xyz_mm"], hover_mm, descend_mm, dry_run, meta))
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
    XYZ with the seam's 3D length. Needs an AOI set via `uv run python seam.py`. Use this to
    see where the weld seam is before tracing it with follow_seam.
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
    except FileNotFoundError:
        out["note"] = "camera not calibrated (no T_base_cam.npy) — pixels only"
    return json.dumps(out)


@tool(parse_docstring=True)
def follow_seam(hover_mm: float = 60.0, dry_run: bool = False) -> str:
    """Trace the seam (joint between two parts) — MOTION ONLY, never fires an arc or weld output.

    Locates the seam (geometrically, in depth within the AOI), maps both endpoints to the base frame via the calibrated
    camera->base transform, then runs: hover above P1 -> descend to P1 -> straight traverse
    to P2 -> retract above P2. The traverse is a straight MoveL at the current velocity/mode
    (use physical mm/s for a real weld-travel speed). Keeps the current tool orientation
    (reorienting one axis only if a waypoint is unreachable); axis locks apply. This is a
    safe dry run of the weld path — it does NOT weld. Requires a saved extrinsic.

    ALWAYS prefer dry_run=True first: it IK-checks every waypoint without moving.

    Args:
        hover_mm: Approach/retract height above the seam endpoints, in mm (default 60).
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
    p1 = [float(b1[0]), float(b1[1]), float(b1[2])]
    p2 = [float(b2[0]), float(b2[1]), float(b2[2])]
    waypoints = [
        ("hover_p1", p1[0], p1[1], p1[2] + hover_mm),
        ("descend_p1", p1[0], p1[1], p1[2]),
        ("traverse_p2", p2[0], p2[1], p2[2]),
        ("retract_p2", p2[0], p2[1], p2[2] + hover_mm),
    ]
    result = {
        "p1_base_mm": [round(v, 1) for v in p1],
        "p2_base_mm": [round(v, 1) for v in p2],
        "length_mm": round(float(np.linalg.norm(b2 - b1)), 1),
        "hover_mm": hover_mm, "dry_run": dry_run, "motion_only": True, "steps": [],
    }
    try:
        rob, tool, user = robot.connect_and_enable()
        ok = True
        for i, (label, x, y, z) in enumerate(waypoints):
            if i == 0:  # first waypoint may reorient a single axis if unreachable
                ret = robot.linear_move_keep_orientation(rob, tool, user, x, y, z, dry_run=dry_run)
            else:       # rest are pure translations along the seam, orientation held
                ret = robot.linear_move(rob, tool, user, x, y, z, dry_run=dry_run)
            result["steps"].append({"step": label,
                                    "target_mm": [round(x, 1), round(y, 1), round(z, 1)],
                                    "result": ret, "success": ret == 0})
            if ret != 0:
                ok = False
                break
        result["success"] = ok
        if not dry_run:
            result["final_pose"] = [round(v, 1) for v in rob.GetActualTCPPose()[1]]
        rob.CloseRPC()
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"error": f"robot unavailable: {exc}"})
    return json.dumps(result)


# All tools exposed to the orchestrator. Add future robot tools here.
ALL_TOOLS = [
    detect_objects,
    find_red_marker,
    find_red_markers,
    set_seam_aoi,
    detect_seam,
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
    move_to_detection,
    move_to_red_marker,
    follow_seam,
]


def shutdown():
    """Release the camera/preview at the end of the session."""
    vision.shutdown()
