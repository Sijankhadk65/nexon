"""The conversation with Claude, with no opinion about how it is displayed.

This is `app.py`'s old terminal loop with `print()` and `input()` taken out of it. What is
left is a Session: it owns the message history, the bound tools, the speech synthesizer and
the state machine, and it reports what happens through one callback. A frontend supplies
that callback and decides whether an event becomes a line on a terminal or a bubble in a
window. Nothing here imports Qt.

The reason for the split is not tidiness. The old design had the agent in one process and
the operator console in another, which meant TWO `Controller` singletons and two copies of
robot.py's module globals driving one arm — so a `disarm` clicked in the console mutated a
different `WELD_LIVE` than the one the agent's trace was about to read. Everything the
controller's docstring promises about single ownership holds only within a process. Putting
the conversation here lets the window run it in-process, which is what makes that promise
true again.

THREADING. `send()` blocks: it streams a reply, runs tools, waits on motion. Call it on a
worker thread. The callback is therefore invoked on that worker thread, and a GUI frontend
must marshal it — `ui/window.py` does so through a Qt signal. `interrupt()` is the one
method meant to be called from another thread while `send()` runs; it stops speech at once
and unwinds the turn.

TOOL TRAFFIC IS NOT CONVERSATION. Tool calls and their JSON results are Claude's private
working state and never enter the transcript; they go to the log. A frontend is told only
that a tool named X is running, so it can say something honest about what the machine is
doing without spilling coordinates into a conversation.
"""

import logging
import os
import queue
import subprocess
import threading

from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from nexon import settings
from nexon.voice import fsm

log = logging.getLogger("nexon")

MODEL = "claude-opus-4-8"
MAX_TOKENS = 4096
BASE_SYSTEM_PROMPT = (
    "You are nexon, an assistant that orchestrates a robot arm equipped with a camera. "
    "You have a detect_objects tool that looks through the robot's camera to find "
    "objects you name. Use it whenever the user asks what you can see, where "
    "something is, or to identify physical parts. It uses a neural object detector, which "
    "CANNOT see a small featureless color blob like a red dot/marker — for anything RED, "
    "use find_red_marker (to see one), find_red_markers (to see/count all of them) and "
    "move_to_red_marker (to visit each one), which use color detection instead. For a weld "
    "SEAM (the joint between two parts), the detector scans a "
    "preset area of interest (AOI); set it with set_seam_aoi (a tight pixel box with its long "
    "side along the seam), then use detect_seam (to see its endpoints, the lead-in and standoff) "
    "and follow_seam (to trace it). follow_seam traces the seam held a fixed standoff above it, "
    "approaching a lead-in point just before the start. It is MOTION ONLY unless welding is "
    "enabled with set_weld: then it strikes the arc at the lead-in and welds P1->P2 — but that "
    "is a DRY weld (identical motion, nothing energized). Striking a REAL arc needs "
    "arm_live_arc, which asks the human operator at the machine to consent; you cannot approve "
    "it yourself, and it blocks until they answer. Only call it when the user has clearly asked "
    "to actually weld, and always rehearse with a dry pass first. If they decline, say so and "
    "carry on — do not retry. disarm_live_arc stands down at any time and is always safe. "
    "Read the weld state with get_weld_settings. Only one motion runs at a time: if a tool "
    "reports busy, the arm is mid-pass — wait, don't retry. For RED line(s)/tape (red-marked "
    "paths, not a metal joint), use detect_red_line (one line), detect_red_lines (see/count all "
    "of them) and follow_red_line (traces every line by color, no AOI needed) — motion "
    "only, never welds. "
    "You can also move the arm: get_robot_pose reads its current position; "
    "robot_move_to goes to an absolute X/Y/Z (mm); robot_move_relative nudges by an "
    "offset; robot_move_direction moves left/right (base X) or forward/back (base Y) — "
    "right=+X, forward=+Y; robot_move_joints "
    "sets joint angles; and robot_go_home parks it. move_to_detection finds an object "
    "with the camera and moves the tool over its real 3D position (via the calibrated "
    "camera-to-base transform) — use it for 'go to'/'move to' the thing you see. There are TWO "
    "speeds. The OPERATION speed is the working-stroke speed and applies ONLY to the seam "
    "traverse, the red-line traverse, and the descent onto a red dot; it has two modes — a "
    "physical mm/s (the default, set with set_physical_velocity) or a percentage of max (set "
    "with set_robot_velocity), switched with set_velocity_mode. The TRANSPORTATION speed (a "
    "percentage, default 10%, set with set_transport_velocity) applies to EVERYTHING ELSE — "
    "jogs, joint moves, and hover/descend/retract positioning. Read both with get_velocity_mode. "
    "To honor 'weld/trace at 30 mm/s', set the physical velocity AND make sure the operation "
    "mode is physical; for a work percentage use percentage mode; for jog speed use "
    "set_transport_velocity. Joint moves always use the transportation percentage. You can also lock individual base-frame "
    "axes with set_axis_movement (e.g. disable Z so the tool can't change height); a "
    "locked axis is held fixed on every linear move while the others still move. Keep the "
    "speed low for safety, and if a target might be unreachable, first call the move with "
    "dry_run=True to IK-check it, then move for real once it reports reachable. "
    "Keep replies very short — one or two sentences, spoken plainly. State only the result "
    "or the single most important detail; do not explain your steps, list tool calls, or read "
    "out raw coordinates unless asked."
)

# Supported forced languages (ISO code -> name). "auto" lets Scribe detect per
# utterance — convenient, but background speech in another language can hijack it,
# so a single forced language is the robust default.
LANGUAGES = {"en": "English", "hi": "Hindi", "de": "German"}
DEFAULT_LANG = os.environ.get("NEXON_LANG", "en").lower()

# Cap tool-call rounds per turn so a misbehaving loop can't run away.
MAX_TOOL_ITERS = 5

# eleven_flash_v2_5 is ElevenLabs' lowest-latency model (~75ms). mp3_44100_128
# plays cleanly through mpv, and concatenated MP3 chunks stream without gaps.
ELEVEN_MODEL = "eleven_flash_v2_5"
OUTPUT_FORMAT = "mp3_44100_128"
VOICE_SETTINGS = VoiceSettings(stability=0.5, similarity_boost=0.75)

# Don't flush a sentence shorter than this (chars, stripped) so decimals ("3.5"),
# abbreviations ("Dr."), and one-word fragments don't become tiny separate requests.
MIN_SENTENCE_CHARS = 12

# Sentinel pushed onto the audio queue to signal end-of-response to the TTS thread.
_DONE = object()


class SessionError(RuntimeError):
    """The session cannot start — no API key, or the model could not be built."""


def system_prompt_for(lang: str) -> str:
    """Base prompt plus a directive to always answer in the forced language."""
    name = LANGUAGES.get(lang)
    if name:
        return (f"{BASE_SYSTEM_PROMPT} Always respond in {name}, regardless of the "
                f"language of the input.")
    return BASE_SYSTEM_PROMPT  # "auto" — no language constraint


def stt_lang(lang: str) -> str | None:
    """The ISO code to force on Scribe, or None to auto-detect."""
    return lang if lang in LANGUAGES else None


def chunk_text(chunk) -> str:
    """Extract plain text from a streamed LangChain message chunk."""
    content = chunk.content
    if isinstance(content, str):
        return content
    parts = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "".join(parts)


def build_model() -> ChatAnthropic:
    return ChatAnthropic(model=MODEL, max_tokens=MAX_TOKENS)


def _el_detail(exc) -> str:
    """Compact ElevenLabs ApiError summary (status + code); falls back to repr."""
    body = getattr(exc, "body", None)
    detail = body.get("detail") if isinstance(body, dict) else None
    if isinstance(detail, dict):
        return f"{getattr(exc, 'status_code', '?')} {detail.get('status') or detail.get('code')}"
    return repr(exc)


def _el_voice_not_found(exc) -> bool:
    """True only if the error means the voice id itself doesn't exist (vs a permission/other error)."""
    if getattr(exc, "status_code", None) == 404:
        return True
    body = getattr(exc, "body", None)
    detail = body.get("detail") if isinstance(body, dict) else None
    if isinstance(detail, dict):
        return "voice_not_found" in (detail.get("status"), detail.get("code"))
    return False


def resolve_voice_id(client: ElevenLabs) -> str | None:
    """Pick a voice: a set ELEVENLABS_VOICE_ID (trusted), else the first voice on the account.

    A set voice id is TRUSTED unless we can prove it's wrong: we try to fetch it but only
    OVERRIDE it when the voice genuinely doesn't exist (404). If validation simply can't run —
    e.g. the key lacks the 'voices_read' permission — we keep the env voice, because the actual
    TTS (text_to_speech) only needs the 'text_to_speech' permission and may still work. Only
    when no voice id is given do we list account voices (which does require 'voices_read').
    """
    env_voice = os.environ.get("ELEVENLABS_VOICE_ID")
    if env_voice:
        try:
            client.voices.get(env_voice)
            return env_voice
        except Exception as exc:  # noqa: BLE001
            if _el_voice_not_found(exc):
                log.warning("tts: ELEVENLABS_VOICE_ID '%s' does not exist on this account; "
                            "falling back to an available voice", env_voice)
            else:  # couldn't validate (e.g. key missing 'voices_read') — trust it, TTS may still work
                log.warning("tts: could not validate ELEVENLABS_VOICE_ID (%s); using it anyway "
                            "— grant the key 'voices_read' to silence this", _el_detail(exc))
                return env_voice
    try:
        voices = client.voices.get_all().voices
        if voices:
            return voices[0].voice_id
    except Exception as exc:  # noqa: BLE001
        log.warning("tts: could not list voices (%s) — grant the key 'voices_read', or set "
                    "ELEVENLABS_VOICE_ID", _el_detail(exc))
    return None


class Speaker:
    """Buffers streamed text into sentences, synthesizes each, and plays via mpv.

    Per assistant turn: `speak()` starts a background playback thread, `push()`
    feeds tokens (flushing whole sentences to the synth queue as they complete),
    and `finish()` flushes the tail and blocks until playback ends. `stop()` cancels
    a turn mid-flight (barge-in): it kills mpv so audio cuts off at once rather than
    draining the buffer. mpv is managed here directly (not via elevenlabs' stream()
    helper) precisely so there's a process handle to kill.
    """

    def __init__(self, client: ElevenLabs, voice_id: str):
        self.client = client
        self.voice_id = voice_id
        self._q: queue.Queue | None = None
        self._buf = ""
        self._thread: threading.Thread | None = None
        self._mpv: subprocess.Popen | None = None
        self._stop = threading.Event()

    @staticmethod
    def _last_boundary(buf: str) -> int | None:
        """Index just past the last sentence-ender that's safe to flush, else None.

        A boundary counts only if followed by whitespace (so "3.5" isn't split)
        and the sentence so far is at least MIN_SENTENCE_CHARS. A newline always
        counts. A boundary at the very end of the buffer waits for the next token.
        """
        last = None
        for i, ch in enumerate(buf):
            if ch not in ".!?\n":
                continue
            at_end = i + 1 >= len(buf)
            if ch == "\n":
                last = i + 1
            elif not at_end and buf[i + 1].isspace():
                if len(buf[: i + 1].strip()) >= MIN_SENTENCE_CHARS:
                    last = i + 1
        return last

    def _audio_gen(self):
        """Yield audio bytes for each queued sentence, in order, into one mpv."""
        while True:
            sentence = self._q.get()
            if sentence is _DONE or self._stop.is_set():
                return
            try:
                audio = self.client.text_to_speech.convert(
                    self.voice_id,
                    text=sentence,
                    model_id=ELEVEN_MODEL,
                    output_format=OUTPUT_FORMAT,
                    voice_settings=VOICE_SETTINGS,
                )
                for chunk in audio:
                    if self._stop.is_set():  # barge-in during synthesis
                        return
                    yield chunk
            except Exception as exc:  # noqa: BLE001
                log.warning("tts error: %s", exc)
                # Stop synthesizing but drain the queue so finish() won't block.
                while self._q.get() is not _DONE:
                    pass
                return

    def _run(self):
        """Own an mpv process and pump synthesized audio into it until done/stopped."""
        try:
            self._mpv = subprocess.Popen(
                ["mpv", "--no-cache", "--no-terminal", "--", "fd://0"],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            log.warning("tts: mpv not found — install mpv to hear replies")
            while self._q.get() is not _DONE:  # drain so finish() won't block
                pass
            return

        try:
            for chunk in self._audio_gen():
                if self._stop.is_set():
                    break
                self._mpv.stdin.write(chunk)
                self._mpv.stdin.flush()
        except (BrokenPipeError, ValueError):
            pass  # mpv was killed (stop()) mid-write
        except Exception as exc:  # noqa: BLE001
            log.warning("tts playback error: %s", exc)

        mpv = self._mpv
        self._mpv = None
        if mpv is None:
            return
        if self._stop.is_set():
            mpv.kill()  # cut off audio already buffered inside mpv
        else:
            try:
                mpv.stdin.close()
            except OSError:
                pass
            mpv.wait()  # let it finish playing what's buffered

    def speak(self):
        self._q = queue.Queue()
        self._buf = ""
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    @property
    def interrupted(self) -> bool:
        return self._stop.is_set()

    def stop(self):
        """Cancel playback immediately (barge-in). Safe to call any time."""
        if self._q is None:
            return
        self._stop.set()
        self._q.put(_DONE)  # unblock a get() waiting for the next sentence
        mpv = self._mpv
        if mpv is not None:
            mpv.kill()

    def push(self, token: str):
        if self._q is None or not token:
            return
        self._buf += token
        idx = self._last_boundary(self._buf)
        if idx is not None:
            chunk, self._buf = self._buf[:idx], self._buf[idx:]
            if chunk.strip():
                self._q.put(chunk)

    def flush(self):
        """Speak whatever is buffered right now, without ending the turn.

        Called at the end of each response so a complete-but-unterminated sentence
        (e.g. a preamble like "Let me take a look.") is spoken immediately, rather
        than waiting for the next response — which may be many seconds away across a
        tool call.
        """
        if self._q is not None and self._buf.strip():
            self._q.put(self._buf)
            self._buf = ""

    def finish(self):
        # If stopped (barge-in), the tail is abandoned and the queue may be drained
        # already — just join the thread and reset. Otherwise flush the last sentence.
        if self._q is not None and not self._stop.is_set():
            if self._buf.strip():
                self._q.put(self._buf)
            self._q.put(_DONE)
        self._buf = ""
        if self._thread is not None:
            self._thread.join()
        self._q = None
        self._thread = None
        self._mpv = None


class Session:
    """One conversation: history, tools, voice, and the state machine that sequences them.

    Events, all delivered on whichever thread caused them:

        state     (state=fsm.State, phase=fsm.Phase|None)  the machine moved
        token     (text=str)                               a fragment of the reply
        tool      (name=str)                               a tool started running
        error     (message=str)                            the turn failed; it was rolled back
        turn_end  (ok=bool, interrupted=bool, barged=bool) the turn is over, either way;
                                                           barged=True means the user cut it
                                                           off by talking over the reply

    A frontend that renders `token` as it arrives gets streaming for free. One that ignores
    every event except `turn_end` still works.
    """

    def __init__(self, on_event=None, lang: str = DEFAULT_LANG):
        self._on_event = on_event or (lambda kind, payload: None)
        self._lang = lang if (lang in LANGUAGES or lang == "auto") else "en"

        self._agent = None
        self._tools_by_name: dict = {}
        self._messages: list = []
        self._speaker: Speaker | None = None
        self._el: ElevenLabs | None = None
        self.voice = None                       # stt.VoiceInput, or None
        self._listener = None                   # barge.Listener while barge-in is available

        self.tts_enabled = True                 # honoured only if a speaker exists
        self._interrupt = threading.Event()
        self._barged = threading.Event()        # this turn's interrupt came from a barge-in
        self._turn_lock = threading.Lock()      # one turn at a time, always
        self._machine = fsm.Machine(on_change=self._on_fsm_change)

    # ------------------------------------------------------------------ lifecycle

    def start(self) -> None:
        """Build the model, bind the tools, and open the voice clients. Slow; not on a GUI thread.

        Raises SessionError when there is no Anthropic key — the one failure the operator
        must be told about in words, because nothing else in the window will work without it.
        Voice is different: its absence degrades the session to text, and only logs.
        """
        # Fills in keys saved from the settings dialog. An explicit `export` still wins.
        # Must run before any SDK client is built: langchain and elevenlabs read the
        # environment directly at construction.
        settings.apply_to_env()

        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise SessionError(
                "No Anthropic API key. Add one in Settings, or export ANTHROPIC_API_KEY.")

        from nexon.agent import tools

        self._tools_by_name = {t.name: t for t in tools.ALL_TOOLS}
        self._agent = build_model().bind_tools(tools.ALL_TOOLS)
        self._messages = [SystemMessage(content=system_prompt_for(self._lang))]

        # ElevenLabs powers both voice out (TTS) and voice in (Scribe STT), so one client
        # is shared. Without the key, nexon runs text-only in both directions.
        el_key = os.environ.get("ELEVENLABS_API_KEY")
        if el_key:
            from nexon.voice import stt

            # The key MUST be passed explicitly: elevenlabs 2.56 does NOT fall back to the
            # ELEVENLABS_API_KEY environment variable — a bare ElevenLabs() sends no auth
            # header at all, and every call 401s with "neither header nor xi-api-key
            # received". The SDK named the parameter after the env var but never wired the
            # two together.
            self._el = ElevenLabs(api_key=el_key)
            voice_id = resolve_voice_id(self._el)
            if voice_id:
                self._speaker = Speaker(self._el, voice_id)
                log.info("tts: using voice %s", voice_id)
            else:
                log.warning("tts: no voice available (set ELEVENLABS_VOICE_ID); text-only")
            # Reuse the one client so STT rides the same explicit-key fix.
            self.voice = stt.VoiceInput(client=self._el, language=stt_lang(self._lang))
        else:
            log.info("voice: ELEVENLABS_API_KEY not set — running text-only")

        # Barge-in: while nexon speaks, a background mic watcher lets the operator cut in by
        # talking over the reply. Always on when nexon can both hear and speak — no toggle;
        # the FSM hook arms it only during RESPONSE and frees the mic the instant a turn ends.
        if self._speaker is not None and self.voice is not None:
            from nexon.voice import barge

            self._listener = barge.Listener(device=self.voice.device)
            log.info("barge-in: enabled (talk over a reply to interrupt)")

    def shutdown(self) -> None:
        self.interrupt()
        if self._listener is not None:
            self._listener.disarm()

    # --------------------------------------------------------------------- state

    @property
    def speaking_available(self) -> bool:
        return self._speaker is not None

    @property
    def listening_available(self) -> bool:
        return self.voice is not None

    @property
    def language(self) -> str:
        return self._lang

    @language.setter
    def language(self, lang: str) -> None:
        if lang not in LANGUAGES and lang != "auto":
            raise ValueError(f"unsupported language {lang!r}")
        self._lang = lang
        if self._messages:
            self._messages[0] = SystemMessage(content=system_prompt_for(lang))
        if self.voice is not None:
            self.voice.language = stt_lang(lang)
        log.info("language set to %s", lang)

    def reset(self) -> None:
        """Forget the conversation, keep the machine. Rejected mid-turn."""
        if self._turn_lock.locked():
            raise RuntimeError("a reply is still in flight")
        self._messages = [SystemMessage(content=system_prompt_for(self._lang))]
        log.info("history cleared")

    def interrupt(self) -> None:
        """Stop the reply and the speech now. Callable from any thread, at any time."""
        self._interrupt.set()
        if self._speaker is not None:
            self._speaker.stop()

    def _emit(self, kind: str, **payload) -> None:
        try:
            self._on_event(kind, payload)
        except Exception:  # noqa: BLE001 — a broken frontend must never wedge a turn
            log.exception("session: event handler raised on %r", kind)

    def _on_fsm_change(self, state, phase) -> None:
        self._update_barge(state)
        self._emit("state", state=state, phase=phase)

    def _update_barge(self, state) -> None:
        """Arm the barge-in mic watcher while nexon is speaking; release it otherwise.

        Live only during RESPONSE, only when audio is actually going out (TTS on), and never
        after this turn has already been interrupted. arm()/disarm() are idempotent and
        disarm kills arecord — so the mic is free again the moment RESPONSE ends. Runs on the
        turn worker (via the FSM hook), so it never blocks the GUI.
        """
        if self._listener is None:
            return
        speaking = self.tts_enabled and self._speaker is not None
        if state is fsm.State.RESPONSE and speaking and not self._interrupt.is_set():
            self._listener.arm(self._on_barge)
        else:
            self._listener.disarm()

    def _on_barge(self) -> None:
        """Fired on the listener thread the moment the user starts talking over the reply."""
        if self._interrupt.is_set():
            return
        self._barged.set()
        log.info("barge-in: user started speaking — cutting off the reply")
        self.interrupt()

    # ---------------------------------------------------------------------- turns

    def send(self, text: str) -> bool:
        """Run one full turn. BLOCKS — stream, tools, motion, speech. Never on a GUI thread.

        Returns True if the turn completed. On failure or interruption the whole turn is
        rolled back out of the history, because a turn can append several AIMessages and
        ToolMessages and a half-turn is not a conversation Claude can continue from.
        """
        if not self._agent:
            raise SessionError("session was never started")
        if not self._turn_lock.acquire(blocking=False):
            raise RuntimeError("a reply is still in flight")

        try:
            log.info("USER: %s", text)
            history_len = len(self._messages)
            self._messages.append(HumanMessage(content=text))

            speaking = self.tts_enabled and self._speaker is not None
            self._interrupt.clear()
            self._barged.clear()            # fresh for this turn; the FSM hook may arm the watcher
            if speaking:
                self._speaker.speak()

            try:
                ok = self._run_turn(speaking)
            finally:
                if speaking:
                    self._speaker.finish()      # flush the tail, wait for playback
                self._machine.to(fsm.State.IDLE)

            interrupted = self._interrupt.is_set()
            if interrupted or not ok:
                del self._messages[history_len:]

            self._emit("turn_end", ok=ok, interrupted=interrupted,
                       barged=self._barged.is_set())
            return ok
        finally:
            self._turn_lock.release()

    def _run_turn(self, speaking: bool) -> bool:
        """Stream Claude's reply, run any tool calls, feed results back, repeat.

        Only the final natural-language prose is emitted as tokens; tool calls and their
        JSON results are Claude's private working state and go to the log. Drives the FSM:
        WAITING while a request is in flight, RESPONSE from the first token, with
        THINKING/SPEAKING/TOOL phases inside it.
        """
        for _ in range(MAX_TOOL_ITERS):
            if self._interrupt.is_set():
                return True

            # Request in flight; no tokens back yet. On the 2nd+ round this is the pause
            # after a tool result while Claude decides what to say next.
            self._machine.to(fsm.State.WAITING)
            gathered = None
            response = []
            try:
                for chunk in self._agent.stream(self._messages):
                    if self._interrupt.is_set():   # user barged in — stop streaming at once
                        return True
                    # First token of this round — the reply is now under way.
                    if self._machine.state is not fsm.State.RESPONSE:
                        self._machine.to(fsm.State.RESPONSE)
                        self._machine.phase_to(fsm.Phase.THINKING)
                    # AIMessageChunks accumulate (text + tool-call fragments) via `+`.
                    gathered = chunk if gathered is None else gathered + chunk
                    text = chunk_text(chunk)
                    if text:
                        response.append(text)
                        self._emit("token", text=text)
                        if speaking:
                            self._speaker.push(text)
            except Exception as exc:  # noqa: BLE001 — surfaced to the frontend and the log
                log.exception("agent turn failed: %s", exc)
                self._emit("error", message=str(exc))
                return False

            if response:
                log.info("NEXON: %s", "".join(response).strip())

            # Speak this response now — critically, the preamble before a tool call, which
            # would otherwise sit unspoken until the post-tool response.
            if speaking:
                self._speaker.flush()
                if response:  # audio is now queued/playing (overlaps the next round)
                    self._machine.phase_to(fsm.Phase.SPEAKING)

            if gathered is None:
                return True
            self._messages.append(gathered)

            tool_calls = getattr(gathered, "tool_calls", None)
            if not tool_calls:
                return True

            # Run each requested tool and hand the results back for the next round.
            self._machine.phase_to(fsm.Phase.TOOL)
            for call in tool_calls:
                log.info("TOOL_CALL: %s(%s)", call["name"], call["args"])
                self._emit("tool", name=call["name"])
                tool_obj = self._tools_by_name.get(call["name"])
                if tool_obj is None:
                    self._messages.append(
                        ToolMessage(content=f"unknown tool {call['name']}",
                                    tool_call_id=call["id"]))
                    continue
                result = tool_obj.invoke(call)   # returns a ToolMessage
                log.info("TOOL_RESULT: %s", getattr(result, "content", result))
                self._messages.append(result)

        return True
