"""A small finite-state machine for nexon's conversational loop.

Each turn walks a fixed cycle:

    IDLE -> LISTENING -> WAITING -> RESPONSE -> IDLE

  IDLE       Nothing in flight; waiting at the prompt for the user to act.
  LISTENING  Capturing the user's utterance — the mic recording (push-to-talk)
             and its transcription. Typed input skips straight to WAITING.
  WAITING    Input is in hand and a request is in flight, but no tokens have come
             back yet. Re-entered before each post-tool round of a turn, so it also
             covers "the model is deciding what to say next after a tool result."
  RESPONSE   nexon is answering. A superstate spanning three overlapping phases:
               THINKING  generating the reply (tokens streaming in)
               SPEAKING  synthesized audio queued/playing
               TOOL      a tool call is running
             These overlap by design — sentence N plays while N+1 is generated — so
             they are phases of one state, not separate states. `phase` marks the
             most recent activity to begin, for the logs; it is not a strict order.

The machine is deliberately small: it holds the current state, checks that each
transition is one the loop actually makes (logging a warning rather than crashing on
a surprise), and logs every change. That buys clean, greppable session logs today and
one seam — the optional `on_change` hook — to hang side effects on later: a terminal
status line, an LED on the robot, or gating the mic during RESPONSE for barge-in.
"""

import logging
from enum import Enum

log = logging.getLogger("nexon")


class State(Enum):
    IDLE = "IDLE"
    LISTENING = "LISTENING"
    WAITING = "WAITING"
    RESPONSE = "RESPONSE"


class Phase(Enum):
    """The current activity within RESPONSE; None in every other state."""

    THINKING = "THINKING"
    SPEAKING = "SPEAKING"
    TOOL = "TOOL"


# Transitions the loop actually makes. LISTENING is optional (typed input goes
# IDLE -> WAITING); RESPONSE -> WAITING covers the next round of a multi-tool turn.
# Any state may fall back to IDLE — an error, interrupt, or empty utterance ends the
# turn early — so IDLE is always permitted and is not listed per-state.
_ALLOWED = {
    State.IDLE: {State.LISTENING, State.WAITING},
    State.LISTENING: {State.WAITING},
    State.WAITING: {State.RESPONSE},
    State.RESPONSE: {State.WAITING},
}


class Machine:
    """Tracks nexon's conversational state and logs every transition.

    `on_change(state, phase)` is an optional callback fired after each change; keep
    it fast and non-raising (a raise is caught and logged, never propagated) since it
    runs inline in the conversation loop.
    """

    def __init__(self, on_change=None):
        self._state = State.IDLE
        self._phase: Phase | None = None
        self._on_change = on_change
        log.info("FSM: start in %s", self._state.value)

    @property
    def state(self) -> State:
        return self._state

    @property
    def phase(self) -> Phase | None:
        return self._phase

    @property
    def label(self) -> str:
        """Human-readable current state, e.g. "IDLE" or "RESPONSE/THINKING"."""
        if self._phase is not None:
            return f"{self._state.value}/{self._phase.value}"
        return self._state.value

    def to(self, state: State) -> None:
        """Move to `state`. No-op if already there; logs the change (or a warning)."""
        if state is self._state:
            return
        if state is State.IDLE or state in _ALLOWED[self._state]:
            log.info("FSM: %s -> %s", self._state.value, state.value)
        else:
            # A surprise transition is a bug worth seeing in the log, not a reason to
            # kill a live conversation — take it, but flag it.
            log.warning("FSM: unexpected %s -> %s", self._state.value, state.value)
        self._state = state
        self._phase = None  # phase only has meaning within a state
        self._notify()

    def phase_to(self, phase: Phase) -> None:
        """Note the current activity within RESPONSE. Ignored in any other state."""
        if self._state is not State.RESPONSE or phase is self._phase:
            return
        self._phase = phase
        log.info("FSM: RESPONSE/%s", phase.value)
        self._notify()

    def _notify(self) -> None:
        if self._on_change is None:
            return
        try:
            self._on_change(self._state, self._phase)
        except Exception:  # noqa: BLE001 — a status hook must never break the loop
            log.exception("FSM: on_change hook failed")
