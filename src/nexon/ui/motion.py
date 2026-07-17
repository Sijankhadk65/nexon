"""Springs, and the two things they buy that a fixed-duration curve cannot.

A `QPropertyAnimation` with an easing curve is a recording: it interpolates from A to B over
a fixed time, and if the target changes halfway it must either finish first or restart from
a standstill. Both look wrong. A spring is not a recording — it is a rule about how a value
chases a target — so:

  * RETARGETING IS FREE. Change `target` mid-flight and the motion continues from wherever
    the value is, at whatever speed it was already moving. No jump, no dead stop. This is
    what makes an interface feel grabbable: a message can arrive while the transcript is
    still settling from the last one, and the scroll simply bends toward the new bottom.

  * VELOCITY SURVIVES. The spring integrates a velocity, so a reversal blends through zero
    instead of hard-cutting from +v to −v. Replacing one animation with another at a
    reversal is what produces the "brick wall" feeling; carrying velocity is the fix.

The parameters are Apple's two, not the physics triplet, because mass/stiffness/damping are
three numbers to describe two behaviours:

  damping   1.0 settles with no overshoot; below 1.0 it overshoots and oscillates.
  response  seconds to reach the target — NOT a duration. A spring has no duration; how
            long it takes to settle emerges from the parameters and the current velocity.

Default to damping 1.0 everywhere. Bounce (≈0.8) is earned only when the motion continues
something the operator's own hand started — a flick, a throw, a drag release. Nothing in
this window is dragged, so nothing in this window bounces: overshoot on a message that
merely arrived reads as a bug, not as physics.

REDUCED MOTION. `reduced()` is checked here rather than by every caller, and it does not
mean "no feedback". It means the value crosses to its target through opacity or lands
immediately, rather than travelling. `spring_opacity` keeps working — a cross-fade is
exactly the non-vestibular equivalent — while `Spring` snaps and `Pulse` holds still.
"""

import math

from PySide6.QtCore import QAbstractAnimation, QEasingCurve, QObject, QVariantAnimation, Signal
from PySide6.QtWidgets import QGraphicsOpacityEffect

from nexon.ui import theme

# Integrated at a fixed 240 Hz regardless of the display's rate, so the motion is identical
# on a 60 Hz panel and a 144 Hz one — only the sampling of it differs.
_SUBSTEP = 1.0 / 240.0
# A frame that took longer than this (a stalled camera, a slow tool result landing) is
# clamped rather than integrated: a 400 ms dt would fling the spring across the screen.
_MAX_FRAME_MS = 64.0

# Below both of these, the motion is under a pixel and under a pixel per second. Snap.
_EPS_VALUE = 0.4
_EPS_VELOCITY = 0.8


def reduced() -> bool:
    return theme.reduced_motion()


class Spring(QAbstractAnimation):
    """A damped harmonic oscillator chasing `target`, emitting `value_changed` per frame.

    Driven by Qt's animation timer, so it is synchronised with the compositor rather than a
    wall-clock QTimer. `duration()` is -1: it runs until it settles, which is the honest
    answer — the settle time is an output of the physics, not an input.
    """

    value_changed = Signal(float)
    settled = Signal()

    def __init__(self, value: float = 0.0, damping: float = 1.0,
                 response: float = 0.35, parent: QObject | None = None):
        super().__init__(parent)
        self._value = float(value)
        self._target = float(value)
        self._velocity = 0.0
        self._damping = float(damping)
        self._response = max(1e-3, float(response))
        self._last_ms = 0.0

    # ---------------------------------------------------------------- state

    @property
    def value(self) -> float:
        return self._value

    @property
    def velocity(self) -> float:
        return self._velocity

    @property
    def target(self) -> float:
        return self._target

    def set_value(self, value: float) -> None:
        """Teleport. Kills velocity — use only to initialise, never to interrupt."""
        self.stop()
        self._value = self._target = float(value)
        self._velocity = 0.0
        self.value_changed.emit(self._value)

    def retarget(self, target: float, velocity: float | None = None) -> None:
        """Chase a new target from the CURRENT value, keeping the current velocity.

        This is the whole point of the class. `velocity` overrides the carried velocity,
        which is what a gesture hands off at release — nothing here does that yet, but the
        seam is where it belongs.
        """
        self._target = float(target)
        if velocity is not None:
            self._velocity = float(velocity)

        if reduced():
            # Land immediately, but still emit: the value must arrive, only the travel is
            # suppressed. Callers that fade instead of moving use spring_opacity.
            self._value = self._target
            self._velocity = 0.0
            self.stop()
            self.value_changed.emit(self._value)
            self.settled.emit()
            return

        if self._settled():
            self._value = self._target
            self.value_changed.emit(self._value)
            self.settled.emit()
            return
        if self.state() != QAbstractAnimation.Running:
            self._last_ms = 0.0
            self.start()

    # ------------------------------------------------------------- physics

    def _settled(self) -> bool:
        return (abs(self._target - self._value) < _EPS_VALUE
                and abs(self._velocity) < _EPS_VELOCITY)

    def _integrate(self, h: float) -> None:
        omega = 2.0 * math.pi / self._response
        # Semi-implicit Euler: acceleration from the current position, then position from
        # the NEW velocity. Explicit Euler adds energy and a critically damped spring
        # would visibly overshoot — which is the one thing damping 1.0 promises not to do.
        accel = (-2.0 * self._damping * omega * self._velocity
                 - omega * omega * (self._value - self._target))
        self._velocity += accel * h
        self._value += self._velocity * h

    def duration(self) -> int:
        return -1

    def updateCurrentTime(self, current_ms: int) -> None:
        dt_ms = min(float(current_ms) - self._last_ms, _MAX_FRAME_MS)
        self._last_ms = float(current_ms)
        if dt_ms <= 0.0:
            return

        dt = dt_ms / 1000.0
        steps = max(1, math.ceil(dt / _SUBSTEP))
        h = dt / steps
        for _ in range(steps):
            self._integrate(h)

        if self._settled():
            self._value = self._target
            self._velocity = 0.0
            self.value_changed.emit(self._value)
            self.stop()
            self.settled.emit()
            return
        self.value_changed.emit(self._value)


class Pulse(QVariantAnimation):
    """A slow, symmetric breath between two values. Reserved for the live-arc indicator.

    A looping animation is a cost: it draws the eye forever and can be nauseating. It is
    justified here and nowhere else, because a live arc is the one state whose cost of
    going unnoticed is a burn. Even so, the colour and the words carry the state on their
    own — the pulse only makes them harder to ignore, and it stops entirely under reduced
    motion, where a moving indicator would be exactly the wrong kind of insistent.
    """

    def __init__(self, low: float = 0.35, high: float = 1.0, period_ms: int = 1500,
                 parent: QObject | None = None):
        super().__init__(parent)
        self.setStartValue(low)
        self.setEndValue(high)
        self.setDuration(period_ms // 2)
        self.setEasingCurve(QEasingCurve.InOutSine)   # no corners at the turnaround
        self.setLoopCount(-1)
        self._low, self._high = low, high
        self.finished.connect(self._reverse)

    def _reverse(self) -> None:
        self.setDirection(QAbstractAnimation.Backward
                          if self.direction() == QAbstractAnimation.Forward
                          else QAbstractAnimation.Forward)

    def begin(self) -> None:
        if reduced():
            self.valueChanged.emit(self._high)   # hold at full strength, motionless
            return
        if self.state() != QAbstractAnimation.Running:
            self.start()

    def end(self) -> None:
        self.stop()
        self.valueChanged.emit(self._low)


def spring_opacity(widget, to: float, damping: float = 1.0, response: float = 0.30):
    """Fade `widget` toward `to`, starting from its CURRENT on-screen opacity.

    Reading the presentation value rather than the last commanded one is what lets a
    half-faded widget be re-targeted without a jump. The effect is cached on the widget so
    repeated calls drive one effect instead of stacking them, and the spring is parented to
    the widget so it dies with it.
    """
    effect = widget.graphicsEffect()
    if not isinstance(effect, QGraphicsOpacityEffect):
        effect = QGraphicsOpacityEffect(widget)
        effect.setOpacity(0.0)
        widget.setGraphicsEffect(effect)

    spring = getattr(widget, "_opacity_spring", None)
    if spring is None:
        spring = Spring(effect.opacity(), damping, response, parent=widget)
        spring.value_changed.connect(effect.setOpacity)
        widget._opacity_spring = spring          # noqa: SLF001 — cache on the widget

    spring.retarget(max(0.0, min(1.0, to)))
    return spring
