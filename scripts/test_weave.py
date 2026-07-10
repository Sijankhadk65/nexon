"""Drive nexon's welding WEAVE as a dry-run swing (no arc) to check the pattern.

A weave weld = a normal straight MoveL with an oscillation overlaid on it:

    WeaveSetPara(num, type, freq, ...)   # configure the swing
    MoveL(seam_start)                    # get to the seam start FIRST
    WeaveStart(num)                      # begin oscillating
    MoveL(seam_end)                      # <-- the weave rides on THIS move
    WeaveEnd(num)
    # for a real weld you'd wrap the travel move in ARCStart(...) / ARCEnd(...)

KEY INSIGHT -- the weave COUNT depends on the travel speed:
    cycles = freq(Hz) * travel_time(s) = freq * path_len / speed
    => freq = cycles * speed / path_len
So a predictable number of weaves needs a KNOWN physical travel speed (mm/s). nexon's
robot.weave_start derives that frequency for you and robot.set_physical_velocity + the
"physical" velocity mode make the travel MoveL run at the real mm/s (velAccParamMode=1).

This exercises the SAME robot.py path the follow_seam / follow_red_line tools use
(configure_weave -> weave_start -> physical MoveL -> weave_end), so if the pattern looks
right here it will look right on a real seam trace. It strikes NO arc and engages NO
welder I/O -- it's a pure dry-run swing: the TCP physically traces the weave with nothing
energized. It does NOT need the camera or a saved extrinsic.

weaveType / pattern names (robot.WEAVE_PATTERNS):
    triangle (0)   sine (4)   vertical_triangle (6)   circle_cw (2)   circle_ccw (3)

Run from the nexon dir:

    uv run python scripts/test_weave.py           # DRY RUN (prints the plan only)
    uv run python scripts/test_weave.py --live    # swing the arm, no arc
    uv run python scripts/test_weave.py --live --pattern sine --range 8 --cycles 12 --speed 15
    uv run python scripts/test_weave.py --live --dist 150 --axis y
"""

import argparse
import logging
import sys
import time

from nexon import robot


def main():
    ap = argparse.ArgumentParser(
        description="Run nexon's weave as a dry-run swing (no arc).")
    ap.add_argument("--ip", default=robot.ROBOT_IP,
                    help=f"Robot IP (default {robot.ROBOT_IP})")
    ap.add_argument("--live", action="store_true",
                    help="Actually move (default is a dry run that only prints the plan)")
    ap.add_argument("--pattern", default="triangle",
                    choices=sorted(robot.WEAVE_PATTERNS),
                    help="Weave shape (default triangle)")
    ap.add_argument("--range", type=float, default=8.0,
                    help="Weave amplitude (total side-to-side swing width) in mm (default 8)")
    ap.add_argument("--cycles", type=float, default=10.0,
                    help="Spacing as a fixed number of weave cycles across the path (default 10)")
    ap.add_argument("--pitch", type=float, default=None,
                    help="Spacing as a fixed mm advanced per weave cycle (overrides --cycles); "
                         "constant regardless of seam length")
    ap.add_argument("--speed", type=float, default=15.0,
                    help="Travel speed in mm/s along the seam (default 15)")
    ap.add_argument("--accel", type=float, default=robot.PHYSICAL_ACC,
                    help=f"Travel acceleration in mm/s^2 (default {robot.PHYSICAL_ACC:.0f})")
    ap.add_argument("--dist", type=float, default=120.0,
                    help="Seam length in mm (default 120)")
    ap.add_argument("--axis", choices=("x", "y"), default="x",
                    help="Base axis to travel along (default x)")
    ap.add_argument("--no-home", action="store_true",
                    help="Skip the MoveJ to START_JOINTS (weave from wherever it is)")
    args = ap.parse_args()

    if args.dist <= 0:
        ap.error("--dist must be > 0")

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Configure the weave through robot.py exactly as the set_weave tool does, and pin the
    # travel to a physical mm/s so the derived swing frequency holds (weave_start needs a real
    # speed). --pitch selects fixed mm-per-cycle spacing; otherwise --cycles over the seam.
    settings = robot.configure_weave(enabled=True, pattern=args.pattern,
                                     amplitude_mm=args.range, cycles=args.cycles,
                                     pitch_mm=args.pitch)
    robot.set_velocity_mode("physical")
    robot.set_physical_velocity(args.speed)
    robot.PHYSICAL_ACC = float(args.accel)

    # Same frequency + cycle-count derivation weave_start does, for the printed plan.
    if settings["spacing"] == "pitch":
        weave_freq = args.speed / settings["pitch_mm"]
        eff_cycles = args.dist / settings["pitch_mm"]
        spacing_desc = (f"pitch {settings['pitch_mm']:.1f} mm/cycle "
                        f"(~{eff_cycles:.1f} cycles over {args.dist:.0f} mm)")
        freq_expr = f"{args.speed} / {settings['pitch_mm']:.1f}"
    else:
        eff_cycles = settings["cycles"]
        weave_freq = eff_cycles * args.speed / args.dist
        spacing_desc = f"{eff_cycles:.0f} cycles over the seam"
        freq_expr = f"{eff_cycles:.0f} * {args.speed} / {args.dist}"
    print(f"Weave pattern: {settings['pattern']} (type {settings['weave_type']}), "
          f"{spacing_desc}, range {settings['amplitude_mm']:.1f} mm")
    print(f"Travel: {args.dist} mm along +{args.axis} at {args.speed} mm/s "
          f"({args.accel:.0f} mm/s^2)")
    print(f"=> weaveFrequency = {freq_expr} = {weave_freq:.3f} Hz")
    print("Swing control: WeaveStart/WeaveEnd (no arc, no welder I/O)")

    if not args.live:
        print("\nDRY RUN -- the sequence that would run is:")
        spacing_kw = (f"pitch_mm={settings['pitch_mm']:.1f}" if settings["spacing"] == "pitch"
                      else f"cycles={settings['cycles']:.0f}")
        print(f"    configure_weave(pattern={args.pattern!r}, amplitude_mm={args.range}, "
              f"{spacing_kw})")
        print("    connect_and_enable()")
        if not args.no_home:
            print("    MoveJ(START_JOINTS)            # to a repeatable start")
        print(f"    weave_start(path_len_mm={args.dist}, speed_mms={args.speed})")
        print(f"        -> WeaveSetPara(0, {settings['weave_type']}, {weave_freq:.3f}, 0, "
              f"{args.range}, 0...) + WeaveStart(0)")
        print(f"    linear_move(start +{args.dist}mm {args.axis})  # physical mm/s, weave rides it")
        print("    weave_end()                    # WeaveEnd(0)")
        print("\nPass --live to actually swing the arm.")
        return

    try:
        rob, tool, user = robot.connect_and_enable(args.ip)
    except Exception as exc:  # noqa: BLE001
        print(f"robot unavailable: {exc}", file=sys.stderr)
        return 1

    weaving = False
    try:
        # Go to a repeatable start pose (joint move, so mm/s doesn't apply here).
        if not args.no_home:
            print("MoveJ to START_JOINTS:",
                  rob.MoveJ(robot.START_JOINTS, tool, user, vel=robot.CURRENT_VEL))
            time.sleep(0.3)

        start_pose = rob.GetActualTCPPose()[1]
        print("Start pose:", [round(v, 1) for v in start_pose])

        # Seam end = start translated along the chosen base axis (Z, orientation held).
        end = list(start_pose)
        end[0 if args.axis == "x" else 1] += args.dist
        print("Seam end:", [round(v, 1) for v in end])

        # Begin oscillating, travel at the KNOWN physical speed (the weave rides this
        # MoveL), then end -- always ending the weave in the finally, even on error.
        weaving = robot.weave_start(rob, args.dist, args.speed)
        if not weaving:
            print("weave did not start; aborting (no traverse).")
            return 1

        t0 = time.monotonic()
        ret = robot.linear_move(rob, tool, user, end[0], end[1], end[2])
        elapsed = time.monotonic() - t0
        print("MoveL (weaving, no arc):", ret)

        if ret == 0:
            got = weave_freq * elapsed
            print(f"\nTravelled {args.dist} mm in {elapsed:.2f} s "
                  f"(~{args.dist / max(elapsed, 1e-6):.1f} mm/s) -> ~{got:.1f} weave cycles "
                  f"(target {eff_cycles:.1f}).")
        else:
            print(f"\nMove rejected (err {ret}). In physical mode vel/acc must be 100 with "
                  "real ovl/oacc; check the target is reachable.")
    finally:
        if weaving:
            robot.weave_end(rob)
        rob.CloseRPC()


if __name__ == "__main__":
    sys.exit(main() or 0)
