"""The transcript: what was said, by whom, and what the machine is doing about it.

Two voices, told apart by material rather than by a label. The operator's words get a
saturated fill; nexon's get a dark, near-opaque one. This all lives on the window's solid
column, never over the camera — prose on a moving weld scene cannot be read at any blur
radius.

EVERY SIZE HERE IS MEASURED, NOT HINTED. A bubble is added straight to the column with an
alignment flag; there is no per-message row wrapper, because a nested layout is one more
thing that can hold a stale height while the bubble inside it grows on every streamed token,
and Qt clips a child to its parent. And `QLabel.sizeHint()` on wrapped text is a guess: it
picks a width, returns the height THAT width would need, and is then laid out at another
width entirely. So `Bubble._remeasure` wraps the text with QFontMetrics against the width
the bubble is actually allowed and fixes both labels to the result. Nothing is left to guess.

SCROLLING IS A CONVERSATION, NOT A COMMAND. New messages retarget a spring at the bottom of
the range; the spring is already moving, so it bends toward the new bottom rather than
restarting. Touch the wheel and the spring is abandoned mid-flight — input is never locked
out during a transition — and the view unpins. Scroll back to the bottom and it re-pins,
so following the conversation is the default and reading back is always available.

WAITING IS SHOWN, NOT HIDDEN. An empty nexon bubble carries a typing indicator from the
moment the request goes out. When a tool runs, the bubble says which faculty is engaged —
"looking through the camera", "moving the arm" — never the tool's name, its arguments, or
its JSON. Tool traffic is Claude's private working state and belongs in the log; what the
operator needs is an honest, ongoing answer to "what is it doing right now".
"""

import math

from PySide6.QtCore import QEasingCurve, QRect, QRectF, Qt, QTimer, QVariantAnimation
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QLabel, QScrollArea, QSizePolicy, QVBoxLayout, QWidget

from nexon.ui import motion, theme

# A bubble stops short of the column's full width so the two voices stay visibly staggered —
# left and right — rather than becoming one ragged block of text.
BUBBLE_MAX_FRACTION = 0.84
BUBBLE_MIN_WIDTH = 240

# How close to the bottom still counts as "at the bottom" — a few pixels of slack so a
# rounding error never silently unpins the view.
PIN_SLACK = 28

# Named for the faculty engaged, not the function called. The operator is watching an arm,
# not reading a stack trace. Anything unlisted falls back to a plain "working".
TOOL_PHRASES = {
    "detect_objects": "looking through the camera",
    "detect_seam": "looking for the seam",
    "set_seam_aoi": "framing the seam",
    "follow_seam": "tracing the seam",
    "find_red_marker": "looking for the marker",
    "find_red_markers": "looking for the markers",
    "move_to_red_marker": "moving to the marker",
    "detect_red_line": "looking for the line",
    "detect_red_lines": "looking for the lines",
    "follow_red_line": "tracing the line",
    "move_to_detection": "moving to what it sees",
    "robot_move_to": "moving the arm",
    "robot_move_relative": "moving the arm",
    "robot_move_direction": "moving the arm",
    "robot_move_joints": "moving the arm",
    "robot_go_home": "parking the arm",
    "get_robot_pose": "checking where the arm is",
    "arm_live_arc": "asking to arm the arc",
    "disarm_live_arc": "standing the arc down",
    "set_weld": "setting up the weld",
    "get_weld_settings": "checking the weld settings",
    "set_weave": "setting the weave",
    "get_weave_settings": "checking the weave",
    "set_axis_movement": "locking an axis",
}


def tool_phrase(name: str) -> str:
    return TOOL_PHRASES.get(name, "working")


class TypingIndicator(QWidget):
    """Three dots breathing in sequence, while nexon has said nothing yet.

    Under reduced motion the dots hold still at a readable opacity: the information is
    "a reply is coming", and that survives losing the animation.
    """

    DOTS = 3
    DOT = 5
    GAP = 5

    def __init__(self, parent=None):
        super().__init__(parent)
        self._phase = 0.0
        self.setFixedSize(self.DOTS * self.DOT + (self.DOTS - 1) * self.GAP, self.DOT * 2)

        self._anim = QVariantAnimation(self)
        self._anim.setStartValue(0.0)
        self._anim.setEndValue(1.0)
        self._anim.setDuration(1100)
        self._anim.setLoopCount(-1)
        self._anim.setEasingCurve(QEasingCurve.Linear)
        self._anim.valueChanged.connect(self._advance)
        if not motion.reduced():
            self._anim.start()

    def _advance(self, value) -> None:
        self._phase = float(value)
        self.update()

    def stop(self) -> None:
        self._anim.stop()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(Qt.NoPen)
        y = self.height() / 2 - self.DOT / 2

        for i in range(self.DOTS):
            if motion.reduced():
                alpha = 0.55
            else:
                # Each dot lags the last by a third of the cycle; a raised cosine keeps the
                # brightening and dimming symmetric, with no corner at the turn.
                offset = (self._phase - i / self.DOTS) % 1.0
                alpha = 0.25 + 0.55 * (0.5 - 0.5 * math.cos(2 * math.pi * offset))
            color = QColor(theme.INK)
            color.setAlphaF(alpha)
            painter.setBrush(color)
            painter.drawEllipse(QRectF(i * (self.DOT + self.GAP), y, self.DOT, self.DOT))


class Bubble(QWidget):
    """One utterance. `role` is 'user' or 'nexon'; nexon's grows as tokens arrive.

    Every child is given an explicitly MEASURED size. `QLabel.sizeHint()` on a word-wrapped
    label is a heuristic — it guesses a width, returns the height that width would need, and
    is then laid out at some other width entirely. The result is a bubble whose last line is
    clipped. `_remeasure` wraps the text with QFontMetrics against the width the bubble is
    actually allowed, and fixes both labels to what it finds, so the layout has nothing left
    to guess at.
    """

    def __init__(self, role: str, text: str = "", parent=None):
        super().__init__(parent)
        self._role = role
        self._text = text
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        pad_x, pad_y = theme.em(1.05), theme.em(0.72)
        self._pad_x = pad_x
        self._max_text = BUBBLE_MIN_WIDTH
        column = QVBoxLayout(self)
        column.setContentsMargins(pad_x, pad_y, pad_x, pad_y)
        column.setSpacing(theme.em(0.35))

        # Only nexon has an activity line, and only while it is working.
        self._activity: QLabel | None = None
        self._typing: TypingIndicator | None = None
        if role == "nexon":
            self._typing = TypingIndicator(self)
            column.addWidget(self._typing, alignment=Qt.AlignLeft)

        self._body = QLabel(text, self)
        self._body.setWordWrap(True)
        self._body.setFont(theme.font(theme.BODY))
        self._body.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self._body.setStyleSheet(
            f"color:{theme.rgba(theme.USER_INK if role == 'user' else theme.NEXON_INK)};"
            "background:transparent;")
        self._body.setVisible(bool(text))
        column.addWidget(self._body)

        self._remeasure()
        motion.spring_opacity(self, 1.0, response=0.32)

    # ---------------------------------------------------------------- content

    @property
    def text(self) -> str:
        return self._text

    def append(self, text: str) -> None:
        """Add streamed tokens. The first one retires the typing indicator."""
        if not text:
            return
        self._text += text
        if self._typing is not None:
            self._typing.stop()
            self._typing.hide()
            self._typing = None
        self.set_activity(None)
        self._body.setVisible(True)
        self._body.setText(self._text)
        self._remeasure()

    def set_activity(self, phrase: str | None) -> None:
        """Say what nexon is doing, above whatever it has already said."""
        if phrase is None:
            if self._activity is not None:
                self._activity.hide()
                self._remeasure()
            return

        if self._activity is None:
            self._activity = QLabel(self)
            self._activity.setFont(theme.font(theme.CAPTION))
            self._activity.setStyleSheet(
                f"color:{theme.rgba(theme.INK_SECONDARY)}; background:transparent;")
            # Above the text, below the typing dots: it describes what is producing the
            # text that is about to appear beneath it.
            self.layout().insertWidget(0, self._activity)
        self._activity.setText(f"{phrase}…")
        self._activity.show()
        self._remeasure()

    def finish(self) -> None:
        """The turn is over: no more tokens, nothing in flight."""
        if self._typing is not None:
            self._typing.stop()
            self._typing.hide()
            self._typing = None
        self.set_activity(None)
        if not self._text:
            self.hide()          # nothing was ever said; leave no empty shell behind

    # ------------------------------------------------------------------ measure

    def set_max_width(self, width: int) -> None:
        """The widest this bubble may become, in pixels, including its padding."""
        self._max_text = max(80, width - 2 * self._pad_x)
        self._remeasure()

    def _remeasure(self) -> None:
        """Fix both labels to the size the text actually needs at the allowed width.

        Ends in updateGeometry(), NOT adjustSize(). adjustSize() resizes this widget on the
        spot and tells no one, so the row's QHBoxLayout keeps the height it computed when the
        bubble was empty — and then vertically centres a bubble taller than its row, handing
        it a NEGATIVE y. That is a streaming reply drawing upward over the message above it.
        updateGeometry() invalidates the row, the column, and the scroll range instead, which
        is what has to happen on every token anyway.
        """
        avail = self._max_text
        flags = int(Qt.TextWordWrap) | int(Qt.AlignLeft)

        if self._text:
            metrics = self._body.fontMetrics()
            wrapped = metrics.boundingRect(QRect(0, 0, avail, 1 << 22), flags, self._text)
            self._body.setFixedSize(min(avail, max(1, wrapped.width())),
                                    max(metrics.height(), wrapped.height()))

        if self._activity is not None and self._activity.isVisible():
            metrics = self._activity.fontMetrics()
            self._activity.setFixedSize(
                min(avail, metrics.horizontalAdvance(self._activity.text())),
                metrics.height())

        self.updateGeometry()

    # ------------------------------------------------------------------ paint

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        radius = theme.RADIUS_BUBBLE
        path = QPainterPath()
        path.addRoundedRect(QRectF(self.rect()), radius, radius)

        if self._role == "user":
            painter.fillPath(path, theme.surface(theme.USER_FILL))
            return

        painter.fillPath(path, theme.surface(theme.NEXON_FILL))
        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(theme.stroke(), 1.0))
        painter.drawPath(path)


class Notice(QLabel):
    """A centred, quiet line for things that are not speech: errors, resets, refusals.

    Unlike a bubble, a notice takes the column's full width, so the layout can give it a
    width first and ask its height second. That is what heightForWidth is for, and it is
    the only case here where QLabel's own wrapping can be trusted.
    """

    def __init__(self, text: str, color: QColor | None = None, parent=None):
        super().__init__(text, parent)
        self.setAlignment(Qt.AlignCenter)
        self.setWordWrap(True)
        self.setFont(theme.font(theme.CAPTION))
        self.setStyleSheet(
            f"color:{theme.rgba(color or theme.INK_TERTIARY)}; background:transparent;")
        policy = self.sizePolicy()
        policy.setHeightForWidth(True)
        self.setSizePolicy(policy)
        motion.spring_opacity(self, 1.0, response=0.32)


class Transcript(QScrollArea):
    """The scrolling column of bubbles, floating over the camera."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QScrollArea.NoFrame)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.viewport().setAutoFillBackground(False)
        self.viewport().setStyleSheet("background:transparent;")
        self.setStyleSheet(f"""
            QScrollArea {{ background:transparent; border:none; }}
            QScrollBar:vertical {{
                background:transparent; width:{theme.em(0.5)}px; margin:0;
            }}
            QScrollBar::handle:vertical {{
                background:{theme.rgba(theme.INK_TERTIARY)};
                border-radius:{theme.em(0.25)}px; min-height:{theme.em(2.5)}px;
            }}
            QScrollBar::handle:vertical:hover {{
                background:{theme.rgba(theme.INK_SECONDARY)};
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height:0; }}
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
                background:transparent;
            }}
        """)

        self._content = QWidget()
        self._content.setAttribute(Qt.WA_TranslucentBackground)
        self._column = QVBoxLayout(self._content)
        self._column.setSpacing(theme.em(0.6))
        # Pushes the first messages to the BOTTOM, next to the composer, so a new
        # conversation begins where the operator is already looking.
        self._column.addStretch(1)
        self.setWidget(self._content)

        self._pinned = True
        self._scroll = motion.Spring(0.0, damping=1.0, response=0.38, parent=self)
        self._scroll.value_changed.connect(
            lambda v: self.verticalScrollBar().setValue(int(round(v))))

        # A bubble grows as tokens land and as it word-wraps; the range therefore changes
        # constantly during a reply, and each change re-aims the spring at the new bottom.
        self.verticalScrollBar().rangeChanged.connect(self._follow)

    # ------------------------------------------------------------------ messages

    def _add(self, widget: QWidget, align) -> QWidget:
        """Put `widget` straight into the column, aligned left or right.

        There is deliberately NO per-message row wrapper. A wrapper's nested QHBoxLayout is
        one more layout that can hold a stale height while the bubble inside it grows on
        every streamed token — and because Qt clips a child to its parent, a bubble taller
        than its stale row gets its top sliced off, or is centred to a negative y and drawn
        over the message above. QVBoxLayout aligns its own items, so the wrapper bought
        nothing and cost a failure mode.
        """
        if align in (Qt.AlignLeft, Qt.AlignRight):
            self._column.addWidget(widget, 0, align | Qt.AlignTop)
        else:
            self._column.addWidget(widget)        # a notice spans the column and wraps
        return widget

    def add_user(self, text: str) -> Bubble:
        bubble = Bubble("user", text)
        self._add(bubble, Qt.AlignRight)
        self._pinned = True          # they just acted; take them to their own message
        self._resize_bubbles()
        self._follow()
        return bubble

    def begin_nexon(self) -> Bubble:
        bubble = Bubble("nexon")
        self._add(bubble, Qt.AlignLeft)
        self._resize_bubbles()
        self._follow()
        return bubble

    def add_notice(self, text: str, color: QColor | None = None) -> Notice:
        notice = Notice(text, color)
        self._add(notice, Qt.AlignCenter)
        self._follow()
        return notice

    def remove(self, widget: QWidget) -> None:
        """Retire a message or notice, leaving no gap where it was."""
        self._column.removeWidget(widget)
        widget.deleteLater()

    def clear(self) -> None:
        while self._column.count() > 1:          # keep the leading stretch
            item = self._column.takeAt(1)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
        self._pinned = True

    # ------------------------------------------------------------------ scrolling

    def _follow(self, *_args) -> None:
        if not self._pinned:
            return
        bar = self.verticalScrollBar()
        # Retarget rather than restart: the spring keeps whatever velocity it had, so a
        # reply that streams in over several seconds scrolls as one continuous motion.
        self._scroll.retarget(float(bar.maximum()))

    def _at_bottom(self) -> bool:
        bar = self.verticalScrollBar()
        return bar.value() >= bar.maximum() - PIN_SLACK

    def wheelEvent(self, event) -> None:
        # The user grabbed a moving view. Abandon the animation at its current value —
        # never make them wait for it to land before their scroll takes effect.
        self._scroll.stop()
        super().wheelEvent(event)
        self._pinned = self._at_bottom()
        self._scroll.set_value(float(self.verticalScrollBar().value()))

    def mousePressEvent(self, event) -> None:
        self._scroll.stop()
        super().mousePressEvent(event)

    # -------------------------------------------------------------------- layout

    def _resize_bubbles(self) -> None:
        usable = max(120, self.viewport().width() - 2 * self._column.contentsMargins().left())
        width = max(BUBBLE_MIN_WIDTH, int(usable * BUBBLE_MAX_FRACTION))
        for bubble in self._content.findChildren(Bubble):
            bubble.set_max_width(min(width, usable))

    def set_chrome_insets(self, top: int, bottom: int, side: int | None = None) -> None:
        """Padding around the scrolling content. The column supplies most of the side gap."""
        side = theme.em(0.3) if side is None else side
        self._column.setContentsMargins(side, top, side, bottom)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._resize_bubbles()
        # A resize re-wraps every bubble and moves the bottom; if we were following the
        # conversation, keep following it. Deferred so the layout has settled first.
        if self._pinned:
            QTimer.singleShot(0, lambda: self._scroll.set_value(
                float(self.verticalScrollBar().maximum())))
