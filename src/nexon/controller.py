"""Single owner of the machine's mutable state, sitting between the clients and robot.py.

robot.py keeps its configuration in module globals — velocity, weave, weld, axis locks —
mutated in place by set_velocity / configure_weave / configure_weld / set_axis_enabled.
That is survivable with exactly one client. It stops being survivable the moment there are
two: Claude (through agent/tools.py) and a human at the UI, both able to change the machine
while a trace is running. `_trace_polyline_base` reads WELD_ENABLED once, at the top of the
pass, so a toggle flipped mid-trace is a live race on whether an arc gets struck.

So every read and every write goes through one Controller instance:

  * snapshot()  -- one consistent read of the whole machine, cheap and non-blocking, so a
                   UI can call it on every repaint. It performs NO robot I/O; use
                   read_pose() for that.
  * mutators    -- arm_live_arc / disarm / set_weld / set_weave / set_velocity_* /
                   set_axis_enabled. Nothing else in the process assigns those globals.
  * run_motion  -- any pass (a trace, a jog) runs on ONE worker thread and returns a Future,
                   so at most one motion is ever in flight and a caller can choose whether
                   to block. The controller does not know what a motion is; agent.tools
                   owns them, which is why there is no import cycle here.
  * subscribe() -- called with the new snapshot after every transition, so a UI can light
                   its arc indicator when the AGENT arms the arc, not just when the human
                   does.

Two rules make the two-client story safe, and they live here rather than in each caller:

  1. Mode changes are rejected while a motion is in flight (Busy). One worker thread means
     one pass at a time; rejecting config changes means the pass runs with the settings it
     was planned with.
  2. disarm() is always allowed — while busy, from an error path, from a window-close
     handler — and is idempotent. There is no state in which the arc cannot be shut off.

Arming a live arc additionally requires passing LIVE_ARC_CONFIRMATION verbatim. A boolean
is too easy to pass by accident from a UI toggle or a mis-parsed tool argument; energizing
a real welding arc should not be reachable by a stray `True`.

The method list here is deliberately the shape of a service definition. If nexon is ever
split into a hardware daemon and a separate GUI process, this becomes the .proto and the
callers do not change.
"""

import logging
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field

from nexon import robot

log = logging.getLogger("nexon")

# What a human types at the console authorizer to consent to a real arc. It is NOT a
# password and it is NOT a gate against the agent — the agent never reaches this code path.
LIVE_ARC_CONFIRMATION = "ARM LIVE ARC"


class Busy(RuntimeError):
    """A mode change was attempted while a motion was in flight."""


class NotArmable(RuntimeError):
    """arm_live_arc() was called without welding enabled, or with no authorizer registered."""


class Denied(RuntimeError):
    """A human refused the live arc, or no one answered in time."""


@dataclass(frozen=True)
class MachineState:
    """An immutable, consistent read of the machine. Cheap: no robot I/O, no locks held.

    `live_armed` is the one flag a UI must render unmissably — when it is True the next
    seam trace strikes a REAL arc.
    """

    weld: dict = field(default_factory=dict)
    weave: dict = field(default_factory=dict)
    velocity: dict = field(default_factory=dict)
    locked_axes: tuple = ()
    busy: bool = False

    @property
    def live_armed(self) -> bool:
        return bool(self.weld.get("enabled") and self.weld.get("live"))

    @property
    def dry_weld(self) -> bool:
        """Welding enabled but not energized — the trace moves identically, nothing fires."""
        return bool(self.weld.get("enabled") and not self.weld.get("live"))


class Controller:
    def __init__(self):
        # Re-entrant: a mutator calls _snapshot_locked() while already holding the lock.
        self._lock = threading.RLock()
        # One worker => one motion at a time. Serialising passes is a safety property,
        # not a performance choice; do not raise max_workers.
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="nexon-motion")
        self._busy = False
        self._subscribers = []
        # Set by whatever surface a HUMAN is sitting at — the console prompt today, a Qt
        # dialog later. Never set by the agent. None means no one can consent, so no live
        # arc: the controller fails closed.
        self._authorizer = None

    # ---------------------------------------------------------------- reads

    def snapshot(self) -> MachineState:
        with self._lock:
            return self._snapshot_locked()

    def _snapshot_locked(self) -> MachineState:
        return MachineState(
            weld=dict(robot.weld_settings()),
            weave=dict(robot.weave_settings()),
            velocity={"mode": robot.VEL_MODE,
                      "percentage": robot.CURRENT_VEL,
                      "physical_mm_s": robot.PHYSICAL_VEL,
                      "transport_pct": robot.TRANSPORT_VEL},
            locked_axes=tuple(robot.locked_axes()),
            busy=self._busy,
        )

    def read_pose(self) -> Future:
        """TCP pose in the base frame. Talks to the arm, so it runs on the worker."""
        return self._submit(self._read_pose_blocking)

    @staticmethod
    def _read_pose_blocking():
        rob = robot.connect_readonly()
        return [round(v, 1) for v in rob.GetActualTCPPose()[1]]

    # -------------------------------------------------------- notification

    def subscribe(self, fn) -> None:
        """Register fn(state) to be called after every state transition."""
        with self._lock:
            self._subscribers.append(fn)

    def _notify(self, state: MachineState) -> None:
        # Called with the lock RELEASED: a subscriber that calls back into the controller
        # (a UI re-reading state) must not deadlock, and a slow one must not hold the
        # machine. A raising subscriber is logged and ignored — a broken UI widget can
        # never wedge robot control.
        for fn in list(self._subscribers):
            try:
                fn(state)
            except Exception:  # noqa: BLE001
                log.exception("controller: subscriber %r raised", fn)

    def _commit(self, state: MachineState) -> MachineState:
        self._notify(state)
        return state

    def _require_idle(self, what: str) -> None:
        if self._busy:
            raise Busy(f"cannot {what} while a motion is running")

    # ------------------------------------------------------------- welding

    def set_weld(self, *, enabled=None, io=None, arc_num=None, current=None,
                 voltage=None, gas=None) -> MachineState:
        """Configure welding. Note there is NO `live` parameter — see arm_live_arc().

        Enabling welding here only makes the next seam trace run the arc sequence as a DRY
        weld: identical motion, arc steps logged, nothing energized.

        ANY call disarms a live arc. Otherwise WELD_LIVE, which configure_weld leaves alone,
        would survive an enabled=False/True cycle and re-arm a real arc with no one having
        called arm_live_arc() — and changing current, voltage or gas under an already-lit
        arc is not something to do implicitly either. Re-arming is always explicit.
        """
        with self._lock:
            self._require_idle("change weld settings")
            was_armed = robot.WELD_LIVE
            robot.configure_weld(enabled=enabled, io=io, arc_num=arc_num,
                                 current=current, voltage=voltage, gas=gas, live=False)
            state = self._snapshot_locked()
        if was_armed:
            log.warning("controller: live arc DISARMED by a weld reconfiguration")
        return self._commit(state)

    def set_arc_authorizer(self, fn) -> None:
        """Register the HUMAN surface that consents to a live arc: fn(reason) -> bool.

        Called by app.py / the UI at startup, never by the agent. It may block for as long
        as it takes a person to answer; the controller holds no lock while it runs.
        """
        with self._lock:
            self._authorizer = fn

    def arm_live_arc(self, reason: str = "") -> MachineState:
        """Ask a human to arm a REAL arc, and arm it if they consent.

        The agent may CALL this — deciding a weld is warranted is exactly the brain's job —
        but it cannot SATISFY it. Consent comes from the authorizer, a callable owned by
        whatever surface a person is actually sitting at. There is no argument to this
        method that grants permission, so no token the model can read out of a docstring
        and pass back. With no authorizer registered, arming fails closed.

        `reason` is shown to the human, so the agent must say what it intends to weld.

        Raises NotArmable (welding not enabled, or nobody can consent), Denied (the human
        said no), or Busy (a motion is running — arming mid-pass would change what an
        already-planned trace does).
        """
        with self._lock:
            self._require_idle("arm a live arc")
            if not robot.WELD_ENABLED:
                raise NotArmable("enable welding first (set_weld(enabled=True)); "
                                 "rehearse the pass as a dry weld before arming")
            authorizer = self._authorizer
        if authorizer is None:
            raise NotArmable("no human authorizer is registered — a live arc cannot be "
                             "armed by software alone")

        # Asked with the lock RELEASED: a person may take a minute, and disarm() must stay
        # callable throughout. That means the machine can change while we wait, so every
        # precondition is re-checked below before anything is energized.
        log.warning("controller: requesting human consent for a LIVE ARC — %s", reason or "(no reason given)")
        try:
            granted = bool(authorizer(reason))
        except Exception as exc:  # noqa: BLE001 — a broken prompt must not arm an arc
            log.exception("controller: arc authorizer raised; treating as DENIED")
            raise Denied(f"authorizer failed: {exc}") from exc
        if not granted:
            log.warning("controller: live arc DENIED by the operator")
            raise Denied("the operator declined the live arc")

        with self._lock:
            self._require_idle("arm a live arc")          # a trace may have started
            if not robot.WELD_ENABLED:                    # someone may have disabled welding
                raise NotArmable("welding was disabled while consent was pending")
            robot.configure_weld(live=True)
            state = self._snapshot_locked()
        log.warning("controller: LIVE ARC ARMED (%s)", reason or "no reason given")
        return self._commit(state)

    def disarm(self) -> MachineState:
        """Drop the live arc. Always permitted — including while busy — and idempotent.

        This is what a window-close handler, an error path, or a panic button calls. It
        leaves `enabled` alone: the trace stays a dry weld rather than silently becoming a
        motion-only pass, which is the more conservative surprise.
        """
        with self._lock:
            was = robot.WELD_LIVE
            robot.configure_weld(live=False)
            state = self._snapshot_locked()
        if was:
            log.warning("controller: live arc DISARMED")
        return self._commit(state)

    # -------------------------------------------------------------- weave

    def set_weave(self, *, enabled=None, pattern=None, amplitude_mm=None,
                  cycles=None, pitch_mm=None) -> MachineState:
        with self._lock:
            self._require_idle("change weave settings")
            robot.configure_weave(enabled=enabled, pattern=pattern,
                                  amplitude_mm=amplitude_mm, cycles=cycles,
                                  pitch_mm=pitch_mm)
            state = self._snapshot_locked()
        return self._commit(state)

    # ------------------------------------------------------- speed & locks

    def set_velocity_mode(self, mode: str) -> MachineState:
        return self._apply("change velocity mode", robot.set_velocity_mode, mode)

    def set_velocity(self, pct: float) -> MachineState:
        return self._apply("change velocity", robot.set_velocity, pct)

    def set_physical_velocity(self, mm_s: float) -> MachineState:
        return self._apply("change velocity", robot.set_physical_velocity, mm_s)

    def set_transport_velocity(self, pct: float) -> MachineState:
        return self._apply("change transport velocity", robot.set_transport_velocity, pct)

    def set_axis_enabled(self, axis: str, enabled: bool) -> MachineState:
        return self._apply("change axis locks", robot.set_axis_enabled, axis, enabled)

    def _apply(self, what, fn, *args) -> MachineState:
        with self._lock:
            self._require_idle(what)
            fn(*args)                      # raises ValueError on a bad axis/mode
            state = self._snapshot_locked()
        return self._commit(state)

    # -------------------------------------------------------------- motion

    def run_motion(self, fn, *args, **kwargs) -> Future:
        """Run `fn` on the motion worker. Raises Busy if a motion is already in flight.

        The controller stays ignorant of what a motion IS — agent.tools owns the traces and
        the jogs — so there is no import cycle between them. What the controller guarantees
        is that exactly one runs at a time, that mode changes are locked out for its
        duration (so a pass executes with the settings it was planned with), and that
        `busy` is published to subscribers at both edges.

        Callers who need the result block on the Future. Never call .result() FROM the
        worker thread: it would be waiting on itself. Busy makes that hard to reach, but
        do not go looking for it.
        """
        return self._submit(lambda: fn(*args, **kwargs))

    def _submit(self, fn) -> Future:
        with self._lock:
            if self._busy:
                raise Busy("a motion is already running")
            self._busy = True
            state = self._snapshot_locked()
        self._commit(state)

        fut = self._worker.submit(fn)
        fut.add_done_callback(self._motion_finished)
        return fut

    def _motion_finished(self, _fut) -> None:
        with self._lock:
            self._busy = False
            state = self._snapshot_locked()
        self._commit(state)

    # ------------------------------------------------------------ lifecycle

    def shutdown(self, wait: bool = True) -> None:
        """Disarm, then stop the worker. Safe to call twice; call it from window-close."""
        try:
            self.disarm()
        finally:
            self._worker.shutdown(wait=wait)


def console_arc_authorizer(reason: str) -> bool:
    """Ask at the terminal. Requires LIVE_ARC_CONFIRMATION typed verbatim; anything else is no.

    The typed phrase is not a secret — it exists so consent cannot be given by hitting
    Enter on autopilot. Register with `controller.set_arc_authorizer(console_arc_authorizer)`.
    A Qt dialog replaces this later; the controller does not care which asked.
    """
    print("\n" + "=" * 68)
    print("  LIVE ARC REQUESTED — the next seam trace will strike a REAL arc.")
    if reason:
        print(f"  Reason: {reason}")
    print(f"  Type {LIVE_ARC_CONFIRMATION!r} to consent, anything else to decline.")
    print("=" * 68)
    try:
        return input("  > ").strip() == LIVE_ARC_CONFIRMATION
    except (EOFError, KeyboardInterrupt):
        return False


_controller = None
_controller_lock = threading.Lock()


def get_controller() -> Controller:
    """The process-wide Controller. Both the agent and the UI must share one instance."""
    global _controller
    with _controller_lock:
        if _controller is None:
            _controller = Controller()
        return _controller
