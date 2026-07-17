"""The operator window: a conversation beside the camera, in front of the machine.

The camera and the conversation sit side by side, in a splitter the operator can drag. The
first version of this window floated the transcript over a full-bleed camera feed, and it
was beautiful against a synthetic test frame and unreadable against a real one — a weld cell
is a moving, high-contrast field, and no amount of blur or scrim makes prose sit safely on
top of it. So the video keeps its own half, undimmed, and the conversation gets a solid
column. Legibility is not a thing to trade for depth.

What DOES float, still, is the chrome that belongs to the camera: the arc panel and the
action pills sit on the video as translucent glass, because they describe what you are
looking at. The conversation's own chrome sits on the column with it.


Three threading rules hold this together, and breaking any of them hangs the arm:

  1. NOTHING that touches the controller or the session runs on the GUI thread.
     Session.send blocks for the length of a reply — streaming, tools, an entire weld pass.
     arm_live_arc blocks for a human. Both go through `_run_async`.
  2. Callbacks arrive on whatever thread caused them — the motion worker, the session
     worker, this window's own workers. Controller.subscribe and Session's event callback
     are each funnelled through a Qt signal, which marshals them onto the GUI thread before
     any widget is touched.
  3. The video widget paints on its own timer and never blocks on the camera.

The window is a CLIENT of the controller, exactly like the agent is. It reads state only
from the snapshots it is handed, never from robot.py's globals — so when Claude arms the
arc mid-conversation, this window lights up without being told about Claude at all. What is
new is that Claude is now IN this process: `Session` runs on a worker thread here rather
than in a second `uv run nexon` with its own Controller and its own copy of robot.py's
globals. One process, one owner of the machine's state. The subscribe/notify machinery that
lights the arc indicator "when the agent arms the arc" only ever worked within a process;
now the agent is in it.

THE SPLIT IS DRAGGABLE. Neither half may be collapsed and both have a floor, so no drag can
leave the operator without a camera or without a way to read a reply. Where an operator wants
the divider between those bounds is not something this file should have an opinion about.
"""

import json
import threading

from PySide6.QtCore import QRect, Qt, Signal
from PySide6.QtGui import QAction, QActionGroup, QKeySequence, QPainter
from PySide6.QtWidgets import (QHBoxLayout, QLabel, QMainWindow, QMenu, QMessageBox,
                               QSplitter, QVBoxLayout, QWidget)

from nexon.controller import get_controller
from nexon.session import LANGUAGES, Session, SessionError
from nexon.ui import theme
from nexon.ui.authorizer import ArcAuthorizer
from nexon.ui.composer import Composer
from nexon.ui.controls import IconButton
from nexon.ui.conversation import Transcript, tool_phrase
from nexon.ui.settings_dialog import SettingsDialog
from nexon.ui.status import ActionBar, MachinePanel
from nexon.ui.video import VideoView
from nexon.voice.wake import WakeWord, earcon

# Breathing room inside each half.
MARGIN = 1.0

# How the window divides. The camera earns the larger share — it is what the conversation is
# about — but the column never drops below a width prose can be read at.
STAGE_SHARE, COLUMN_SHARE = 62, 38
COLUMN_MIN = 380
STAGE_MIN = 420


class Stage(QWidget):
    """The camera, with the machine's own chrome floating on it.

    The arc panel and the action pills live here rather than in the column because they
    describe what is on this half of the screen. A control belongs next to the thing it
    affects, and these affect the arm you are watching.
    """

    def __init__(self, hub, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(STAGE_MIN)
        self.video = VideoView(hub, self)
        self.machine = MachinePanel(self)
        self.actions = ActionBar(self)

        # Every pane of glass samples the CAMERA, never the pane beneath it. The panels are
        # siblings of the video rather than its children, so the mapping goes through global
        # coordinates — mapTo() only walks up to an ancestor.
        def backdrop(widget: QWidget, rect: QRect):
            origin = self.video.mapFromGlobal(widget.mapToGlobal(rect.topLeft()))
            return self.video.backdrop(QRect(origin, rect.size()))

        self.machine.set_backdrop_source(backdrop)
        self.video.painted.connect(self.machine.update)   # the glass is as fresh as the frame

    def place(self) -> None:
        """Position the floating clusters. Called on resize, and when one changes size."""
        margin = theme.em(MARGIN)
        self.video.setGeometry(0, 0, self.width(), self.height())

        self.machine.adjustSize()
        self.machine.move(margin, margin)

        self.actions.adjustSize()
        self.actions.move(margin, self.height() - margin - self.actions.height())

        for widget in (self.machine, self.actions):
            widget.raise_()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.place()


class Column(QWidget):
    """The conversation's half: solid, quiet, and the only place prose is allowed to live."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(COLUMN_MIN)
        self.setAutoFillBackground(False)

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), theme.COLUMN)
        # A hairline where the column meets the camera: one pixel is all the separation two
        # surfaces of such different character need.
        painter.fillRect(0, 0, 1, self.height(), theme.HAIRLINE)


class MainWindow(QMainWindow):
    # Controller transitions, session events and worker results all arrive off-thread;
    # these carry them home to the GUI thread.
    _state_changed = Signal(object)
    _session_event = Signal(str, object)
    _result = Signal(str, object)
    _booted = Signal(object)              # None on success, else the Exception
    _transcribed = Signal(str)
    _wake_fired = Signal()                # the wake word was heard (off the wake thread)
    _hf_transcribed = Signal(str, bool)   # a hands-free utterance (text, was-a-follow-up)
    _hands_free_ready = Signal(bool)      # the wake model finished loading (ok?)

    def __init__(self, hub):
        super().__init__()
        self.setWindowTitle("nexon")
        self._hub = hub
        self._ctl = get_controller()
        self._pending = None              # the nexon bubble currently being written into
        self._ready = False
        self._booting = None              # the "starting…" notice, until it is replaced
        self._boot_error = None           # the last failure notice, cleared if a retry works

        # Hands-free: an always-on wake word that opens the mic without a button. Off until
        # the operator turns it on; `_wake` is built at boot (only if voice exists at all).
        self._wake: WakeWord | None = None
        self._hands_free = False
        self._capturing = False           # a wake-driven VAD capture is in flight
        self._capturing_barge = False     # ...and it was opened by a barge-in, not the wake word

        # A human, and only a human, consents to a real arc. Registering this is what makes
        # arm_live_arc possible at all — with no authorizer the controller fails closed and
        # the agent simply cannot weld for real.
        self._authorizer = ArcAuthorizer(self)
        self._ctl.set_arc_authorizer(self._authorizer.ask)

        self._session = Session(on_event=lambda kind, payload:
                                self._session_event.emit(kind, payload))

        self._build()

        # Queued by default, because the emitters are other threads.
        self._state_changed.connect(self._render_state)
        self._session_event.connect(self._on_session_event)
        self._result.connect(self._render_result)
        self._booted.connect(self._on_booted)
        self._transcribed.connect(self._on_transcribed)
        self._wake_fired.connect(self._on_wake_fired)
        self._hf_transcribed.connect(self._on_hf_transcribed)
        self._hands_free_ready.connect(self._on_hands_free_ready)

        self._transcript.set_chrome_insets(top=theme.em(0.5), bottom=theme.em(0.5))
        self._ctl.subscribe(self._state_changed.emit)
        self._render_state(self._ctl.snapshot())

        # Building the model imports torch and transformers. Seconds, not milliseconds.
        self._booting = self._transcript.add_notice("starting nexon…")
        self._composer.setEnabled(False)
        threading.Thread(target=self._boot, name="nexon-ui-boot", daemon=True).start()

    # ------------------------------------------------------------------ layout

    def _build(self) -> None:
        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)   # neither half may be dragged out of existence
        splitter.setHandleWidth(1)
        splitter.setStyleSheet(
            f"QSplitter::handle {{ background:{theme.rgba(theme.HAIRLINE)}; }}")
        self.setCentralWidget(splitter)
        self._splitter = splitter

        self._stage = Stage(self._hub, splitter)
        self._video = self._stage.video
        self._machine = self._stage.machine
        self._actions = self._stage.actions

        column = Column(splitter)
        self._column = column
        self._transcript = Transcript(column)
        self._composer = Composer(column)
        self._more = IconButton("more", column)
        self._more.setToolTip("More")

        title = QLabel("nexon", column)
        title.setFont(theme.font(theme.CAPTION, weight=600, caps=True))
        title.setStyleSheet(f"color:{theme.rgba(theme.INK_TERTIARY)}; background:transparent;")

        margin = theme.em(MARGIN)
        header = QHBoxLayout()
        header.setContentsMargins(0, margin, 0, 0)
        header.addWidget(title)
        header.addStretch(1)
        header.addWidget(self._more)

        stack = QVBoxLayout(column)
        stack.setContentsMargins(margin, 0, margin, margin)
        stack.setSpacing(theme.em(0.5))
        stack.addLayout(header)
        stack.addWidget(self._transcript, stretch=1)
        stack.addWidget(self._composer)

        splitter.addWidget(self._stage)
        splitter.addWidget(column)
        splitter.setStretchFactor(0, STAGE_SHARE)
        splitter.setStretchFactor(1, COLUMN_SHARE)

        self._composer.submitted.connect(self._on_submit)
        self._composer.stop_requested.connect(self._on_stop)
        self._composer.stop_listening.connect(self._disable_hands_free)
        self._composer.hold_started.connect(self._on_hold_started)
        self._composer.hold_ended.connect(self._on_hold_ended)

        self._machine.weld_toggled.connect(self._on_weld_toggled)
        self._machine.arm_requested.connect(self._on_arm)
        self._machine.disarm_requested.connect(self._on_disarm)

        self._actions.detect_seam.connect(self._on_detect_seam)
        self._actions.dry_trace.connect(self._on_dry_trace)
        self._actions.depth_toggled.connect(self._video.set_depth)

        self._more.clicked.connect(self._on_more)
        self._build_shortcuts()

        # The camera earns the larger share; both halves have a floor (see Stage/Column).
        splitter.setSizes([STAGE_SHARE * 10, COLUMN_SHARE * 10])

    def _build_shortcuts(self) -> None:
        # No menu bar: the overflow button carries these. The shortcuts still exist,
        # because the people who use them daily should not have to open a menu.
        settings = QAction("&Settings…", self)
        settings.setShortcut(QKeySequence.Preferences)
        settings.triggered.connect(self._on_settings)

        quit_action = QAction("&Quit", self)
        quit_action.setShortcut(QKeySequence.Quit)
        quit_action.triggered.connect(self.close)

        self.addAction(settings)
        self.addAction(quit_action)
        self._settings_action = settings
        self._quit_action = quit_action

    def _layout(self) -> None:
        """Re-place the camera's floating chrome. The splitter owns everything else."""
        self._stage.place()

    # ------------------------------------------------------------- threading

    def _run_async(self, label, fn, *args, **kwargs) -> None:
        """Run a blocking controller/tool call off the GUI thread; report via _result.

        Every controller entry point can block — on the motion worker, or on a human
        answering the arc dialog. Calling one from a button handler would freeze the
        window, and in the arm case would deadlock against the dialog it is waiting for.
        """
        def work():
            try:
                self._result.emit(label, fn(*args, **kwargs))
            except Exception as exc:  # noqa: BLE001 — surfaced in the transcript
                self._result.emit(label, exc)

        threading.Thread(target=work, name=f"nexon-ui-{label}", daemon=True).start()

    def _boot(self) -> None:
        try:
            self._session.start()
            # Hands-free rides the same mic and the same transcriber as push-to-talk, so it
            # only exists when voice does. The model itself is loaded lazily, the first time
            # the operator turns hands-free on — booting stays fast for those who never do.
            voice = self._session.voice
            if voice is not None:
                self._wake = WakeWord(device=voice.device)
            self._booted.emit(None)
        except Exception as exc:  # noqa: BLE001 — reported in the transcript
            self._booted.emit(exc)

    def _on_booted(self, error) -> None:
        if self._booting is not None:
            self._transcript.remove(self._booting)
            self._booting = None

        if error is not None:
            self._ready = False
            self._composer.setEnabled(False)
            self._boot_error = self._transcript.add_notice(str(error), theme.WARN)
            if isinstance(error, SessionError):
                self._on_settings()      # the one failure they can fix from here
            return

        # A retry worked. Leaving "No Anthropic API key" on screen above a working
        # conversation would be a lie about the current state.
        if self._boot_error is not None:
            self._transcript.remove(self._boot_error)
            self._boot_error = None

        self._ready = True
        self._composer.setEnabled(True)
        self._composer.set_voice_available(self._session.listening_available)
        self._composer.focus()

    # ---------------------------------------------------------------- speaking

    def _on_submit(self, text: str) -> None:
        if not self._ready:
            return
        if self._wake is not None:      # the mic is about to be busy replying; stand the wake word down
            self._wake.disarm()
        self._transcript.add_user(text)
        self._pending = self._transcript.begin_nexon()
        self._composer.set_busy(True)
        threading.Thread(target=self._turn, args=(text,),
                         name="nexon-ui-turn", daemon=True).start()

    def _turn(self, text: str) -> None:
        try:
            self._session.send(text)      # blocks; emits its own events, including turn_end
        except Exception as exc:  # noqa: BLE001 — send() raised before it could report
            self._session_event.emit("error", {"message": str(exc)})
            self._session_event.emit("turn_end", {"ok": False, "interrupted": False})

    def _on_stop(self) -> None:
        self._session.interrupt()

    def _on_session_event(self, kind: str, payload: dict) -> None:
        if kind == "token" and self._pending is not None:
            self._pending.append(payload["text"])
        elif kind == "tool" and self._pending is not None:
            self._pending.set_activity(tool_phrase(payload["name"]))
        elif kind == "error":
            self._transcript.add_notice(payload["message"], theme.WARN)
        elif kind == "turn_end":
            interrupted = payload.get("interrupted")
            barged = payload.get("barged")
            if self._pending is not None:
                had_text = bool(self._pending.text)
                self._pending.finish()
                self._pending = None
                if interrupted and not barged and not had_text:
                    self._transcript.add_notice("stopped")
            self._composer.set_busy(False)
            # Barge-in: the user talked over the reply, so they're already mid-sentence —
            # capture it now (VAD-endpointed), whether or not hands-free is on. No wake word,
            # no button. `listening_available` is guaranteed here (barge only fires with voice).
            if barged and self._session.listening_available:
                self._start_capture(followup=True, barge=True)
            # Hands-free keeps the floor: after a reply, open a short VAD window so a
            # follow-up needs no wake word. On silence the capture re-arms the wake word.
            # A plain stop breaks the loop back to the wake word instead of listening on.
            elif self._hands_free and not interrupted:
                self._start_capture(followup=True)
            else:
                self._composer.focus()
                self._ensure_wake()

    # ----------------------------------------------------------------- listening

    def _on_hold_started(self) -> None:
        # Spawning arecord is a fork+exec, not a blocking device open, so it is safe here —
        # and it must be, because the release that ends it may arrive milliseconds later.
        voice = self._session.voice
        if voice is None or not self._ready:
            return
        if self._wake is not None:      # push-to-talk takes the mic; the wake word must let go first
            self._wake.disarm()
        if not voice.start_recording():
            return
        self._composer.set_listening(True)

    def _on_hold_ended(self) -> None:
        voice = self._session.voice
        if voice is None or not voice.recording:
            return
        self._composer.set_listening(False)
        self._composer.set_transcribing(True)

        def work():
            self._transcribed.emit(voice.stop_and_transcribe())   # a network round trip

        threading.Thread(target=work, name="nexon-ui-stt", daemon=True).start()

    def _on_transcribed(self, text: str) -> None:
        self._composer.set_transcribing(False)
        if not text:
            self._transcript.add_notice("heard nothing")
            self._ensure_wake()        # a dud push-to-talk shouldn't leave hands-free deaf
            return
        self._on_submit(text)

    # ------------------------------------------------------------- hands-free

    def _ensure_wake(self) -> None:
        """Arm the wake word exactly when we're idle and hands-free; disarm otherwise.

        The single gate for the mic between turns. arm()/disarm() are idempotent, so this is
        safe to call liberally — it never opens a second arecord on a mic that's already busy
        recording an utterance (`_capturing`), replying (`_pending`), or held for push-to-talk.
        """
        if self._wake is None:
            return
        if self._hands_free and self._ready and self._pending is None and not self._capturing:
            self._wake.arm(self._wake_fired.emit)
        else:
            self._wake.disarm()

    def _on_wake_fired(self) -> None:
        """The wake word was heard (marshalled from the wake thread). Open the mic."""
        if not self._hands_free or not self._ready or self._pending is not None:
            return
        self._start_capture(followup=False)

    def _start_capture(self, followup: bool, barge: bool = False) -> None:
        """Take the mic off the wake word and capture one utterance (VAD-endpointed).

        `barge` marks a capture opened because the user talked over a reply — it is delivered
        even when hands-free is off, since the operator plainly meant to speak.
        """
        if self._wake is not None:
            self._wake.disarm()
        self._capturing = True
        self._capturing_barge = barge
        self._composer.set_listening(True)
        threading.Thread(target=self._capture_worker, args=(followup,),
                         name="nexon-ui-wake", daemon=True).start()

    def _capture_worker(self, followup: bool) -> None:
        voice = self._session.voice
        if voice is None:
            self._hf_transcribed.emit("", followup)
            return
        if not followup:
            earcon()      # audible "listening" ack; blocks ~120 ms, then the mic opens clean
        # Blocks until the user stops talking, or returns "" if they never start (silence).
        self._hf_transcribed.emit(voice.listen_until_silence(), followup)

    def _on_hf_transcribed(self, text: str, followup: bool) -> None:
        self._capturing = False
        barge = self._capturing_barge
        self._capturing_barge = False
        self._composer.set_listening(False)
        # A barge-in capture is delivered even with hands-free off — the operator meant to
        # speak. A plain hands-free capture is dropped if hands-free was switched off mid-listen.
        if not self._hands_free and not barge:
            return
        if text:
            self._on_submit(text)      # starts a turn; turn_end opens the next follow-up window
        elif self._hands_free:
            self._ensure_wake()        # silence — back to waiting for the wake word
        else:
            self._composer.focus()     # a lone barge-in false-trigger — back to rest

    def _on_hands_free(self, on: bool) -> None:
        """Toggle hands-free. Turning it on loads the wake model off-thread first."""
        if not on:
            self._disable_hands_free()
            return
        if self._wake is None:
            return
        self._transcript.add_notice("enabling hands-free…")

        def work():
            self._hands_free_ready.emit(self._wake.available)   # may download+build the model

        threading.Thread(target=work, name="nexon-ui-wake-load", daemon=True).start()

    def _on_hands_free_ready(self, ok: bool) -> None:
        if not ok:
            self._transcript.add_notice(
                "hands-free unavailable — run `uv sync` to install the wake word, "
                "or set NEXON_WAKE_MODEL", theme.WARN)
            return
        self._hands_free = True
        self._composer.set_hands_free(True)
        self._composer.set_listening_active(True)   # the stop button now has a loop to stop
        self._transcript.add_notice("hands-free on — say the wake word to talk")
        self._ensure_wake()

    def _disable_hands_free(self) -> None:
        """Turn hands-free off and stop whatever the mic is doing right now.

        The one exit from a hands-free session, reachable from the stop button and the menu
        toggle alike: it aborts an in-flight VAD capture, disarms the wake word, and drops the
        composer back to rest — so the operator is never stuck in a listen loop with no way out.
        A reply already in flight is left alone; the stop button targets that separately.
        """
        self._hands_free = False
        self._composer.set_hands_free(False)
        self._composer.set_listening_active(False)
        voice = self._session.voice
        if voice is not None:
            voice.cancel_listening()   # unblock a follow-up window that's mid-capture
        if self._capturing:
            self._composer.set_listening(False)
        self._ensure_wake()            # disarms (hands_free is now False)

    # ---------------------------------------------------------------- actions

    def _on_weld_toggled(self, enabled: bool) -> None:
        self._run_async("welding", self._ctl.set_weld, enabled=enabled)

    def _on_arm(self) -> None:
        # Off-thread on purpose: arm_live_arc blocks until the dialog is answered, and the
        # dialog needs this thread to run its event loop.
        self._run_async("arm", self._ctl.arm_live_arc, "armed from the operator console")

    def _on_disarm(self) -> None:
        self._run_async("disarm", self._ctl.disarm)

    def _on_detect_seam(self) -> None:
        from nexon.agent import tools
        self._run_async("detect seam", tools.detect_seam.invoke, {})

    def _on_dry_trace(self) -> None:
        from nexon.agent import tools
        self._run_async("dry trace", tools.follow_seam.invoke, {"dry_run": True})

    def _on_more(self) -> None:
        menu = QMenu(self)
        menu.addAction(self._settings_action)

        clear = menu.addAction("Clear conversation")
        clear.setEnabled(self._ready and self._pending is None)
        clear.triggered.connect(self._on_clear)

        speak = menu.addAction("Speak replies")
        speak.setCheckable(True)
        speak.setChecked(self._session.tts_enabled and self._session.speaking_available)
        speak.setEnabled(self._session.speaking_available)
        if not self._session.speaking_available:
            speak.setToolTip("Set an ElevenLabs API key in Settings")
        speak.toggled.connect(self._on_speak_toggled)

        # Hands-free needs the same voice stack as talking at all, plus a loadable wake model.
        hands_free = menu.addAction("Hands-free (wake word)")
        hands_free.setCheckable(True)
        hands_free.setChecked(self._hands_free)
        hands_free.setEnabled(
            self._ready and self._session.listening_available and self._wake is not None)
        if self._wake is None:
            hands_free.setToolTip("Set an ElevenLabs API key in Settings to talk")
        hands_free.toggled.connect(self._on_hands_free)

        # Changing the language rewrites the system prompt and re-aims the transcriber, so
        # it is not something to do to a reply that is already half-written.
        languages = menu.addMenu("Language")
        languages.setEnabled(self._ready and self._pending is None)
        group = QActionGroup(languages)
        group.setExclusive(True)
        for code, name in list(LANGUAGES.items()) + [("auto", "Auto-detect")]:
            action = languages.addAction(name)
            action.setCheckable(True)
            action.setChecked(self._session.language == code)
            action.triggered.connect(lambda _checked, c=code: self._on_language(c))
            group.addAction(action)

        menu.addSeparator()
        menu.addAction(self._quit_action)

        # Anchored to the button that opened it: a menu that appears somewhere else has
        # thrown away the only spatial cue the operator had.
        menu.exec(self._more.mapToGlobal(self._more.rect().bottomLeft()))

    def _on_clear(self) -> None:
        try:
            self._session.reset()
        except RuntimeError as exc:
            self._transcript.add_notice(str(exc), theme.WARN)
            return
        self._transcript.clear()

    def _on_speak_toggled(self, on: bool) -> None:
        self._session.tts_enabled = on

    def _on_language(self, code: str) -> None:
        self._session.language = code
        self._transcript.add_notice(f"language: {LANGUAGES.get(code, 'auto-detect')}")

    def _on_settings(self) -> None:
        # Modal and on the GUI thread: the dialog touches no robot state, and its one
        # blocking call (listing voices) runs on its own worker.
        if not SettingsDialog(self).exec():
            return
        if self._ready:
            self._transcript.add_notice("settings saved — restart nexon to apply")
            return
        # It never started, most likely for want of the key they just typed. Try again.
        self._booting = self._transcript.add_notice("starting nexon…")
        threading.Thread(target=self._boot, name="nexon-ui-boot", daemon=True).start()

    # ---------------------------------------------------------------- rendering

    def _render_state(self, state) -> None:
        self._machine.render_state(state)
        self._actions.render_state(state)
        self._layout()               # the arc panel changes size with the arc's state

    def _render_result(self, label: str, result) -> None:
        """A direct control's outcome. Not speech — nexon did not say it, so it is a notice."""
        if isinstance(result, Exception):
            self._transcript.add_notice(f"{label}: {result}", theme.WARN)
            return
        if isinstance(result, str):          # tools return JSON strings
            try:
                payload = json.loads(result)
            except ValueError:
                self._transcript.add_notice(f"{label}: {result[:120]}")
                return
            if "error" in payload:
                self._transcript.add_notice(f"{label}: {payload['error']}", theme.WARN)
                return
        self._transcript.add_notice(f"{label}: ok")

    # ---------------------------------------------------------------- teardown

    def closeEvent(self, event) -> None:
        state = self._ctl.snapshot()
        if state.busy:
            QMessageBox.warning(self, "Motion running",
                                "A pass is still running. Wait for it to finish.")
            event.ignore()
            return
        # Stop the reply and the speech first, then disarm before anything else can fail.
        # Both are idempotent.
        if self._wake is not None:
            self._wake.disarm()        # release the mic so no arecord outlives the window
        self._session.shutdown()
        self._ctl.shutdown()
        self._hub.stop()
        event.accept()
