"""A terminal chat with Claude (via LangChain) that speaks replies with ElevenLabs.

Claude's response is streamed token-by-token with LangChain. Those tokens are
buffered into sentences, and each finished sentence is synthesized with ElevenLabs
(`convert`, Flash model) and fed into a single continuous mpv process via the
built-in `stream()` helper. Because sentence N+1 is generated and synthesized
while sentence N is still playing, speech starts fast (after the first sentence)
and long replies play through to the end — with no realtime-WebSocket to drop.

Env vars:
  ANTHROPIC_API_KEY    - required, for Claude
  ELEVENLABS_API_KEY   - optional; without it, runs text-only
  ELEVENLABS_VOICE_ID  - optional; defaults to the first voice on your account
"""

import logging
import os
import queue
import subprocess
import sys
import threading
import warnings

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

import fsm
import logs

# tools/vision/stt are imported inside main() under logs.mute_stdout so their
# SDK/ML startup chatter is captured in the log file, not printed to the screen.

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
    "side along the seam), then use detect_seam (to see its endpoints) and follow_seam (to "
    "trace it) — follow_seam is motion only and never welds. For RED line(s)/tape (red-marked "
    "paths, not a metal joint), use detect_red_line (one line), detect_red_lines (see/count all "
    "of them) and follow_red_line (traces every line by color, no AOI needed) — also motion "
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


def run_agent_turn(agent, messages, tools_by_name, speaker, speaking, machine,
                   interrupt) -> bool:
    """Stream Claude's reply, run any tool calls, feed results back, repeat.

    Only the final natural-language prose is spoken; tool calls and their JSON
    results are Claude's private working state. Appends every AIMessage/ToolMessage
    to `messages`. Drives the FSM: WAITING while a request is in flight, RESPONSE from
    the first token, with THINKING/SPEAKING/TOOL phases inside it. Returns True on
    success; on error the caller rolls back the turn. If `interrupt` (barge-in) is set
    it stops promptly and returns True with nothing appended — the caller sees the set
    event and rolls the turn back.
    """
    for _ in range(MAX_TOOL_ITERS):
        if interrupt.is_set():
            return True
        # Request in flight; no tokens back yet. On the 2nd+ round this is the pause
        # after a tool result while Claude decides what to say next.
        machine.to(fsm.State.WAITING)
        gathered = None
        response = []
        try:
            for chunk in agent.stream(messages):
                if interrupt.is_set():  # user barged in — stop streaming at once
                    return True
                # First token of this round — the reply is now under way.
                if machine.state is not fsm.State.RESPONSE:
                    machine.to(fsm.State.RESPONSE)
                    machine.phase_to(fsm.Phase.THINKING)
                # AIMessageChunks accumulate (text + tool-call fragments) via `+`.
                gathered = chunk if gathered is None else gathered + chunk
                text = chunk_text(chunk)
                if text:
                    print(text, end="", flush=True)  # screen: the conversation
                    response.append(text)
                    if speaking:
                        speaker.push(text)
        except Exception as exc:  # noqa: BLE001
            print(f"\n[error — see log]\n")
            log.exception("agent turn failed: %s", exc)
            return False

        if response:
            log.info("NEXON: %s", "".join(response).strip())

        # Speak this response now — critically, the preamble before a tool call,
        # which would otherwise sit unspoken until the post-tool response.
        if speaking:
            speaker.flush()
            if response:  # audio is now queued/playing (overlaps the next round)
                machine.phase_to(fsm.Phase.SPEAKING)

        if gathered is None:
            return True
        messages.append(gathered)

        tool_calls = getattr(gathered, "tool_calls", None)
        if not tool_calls:
            return True

        # Run each requested tool and hand the results back for the next round.
        # Tool traffic is log-only — the screen stays a clean conversation.
        machine.phase_to(fsm.Phase.TOOL)
        for call in tool_calls:
            log.info("TOOL_CALL: %s(%s)", call["name"], call["args"])
            tool_obj = tools_by_name.get(call["name"])
            if tool_obj is None:
                messages.append(
                    ToolMessage(content=f"unknown tool {call['name']}",
                                tool_call_id=call["id"])
                )
                continue
            result = tool_obj.invoke(call)  # returns a ToolMessage
            log.info("TOOL_RESULT: %s", getattr(result, "content", result))
            messages.append(result)
        print("claude> ", end="", flush=True)

    return True


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
                print(f"[tts: ELEVENLABS_VOICE_ID '{env_voice}' does not exist on this account; "
                      f"falling back to an available voice]", file=sys.stderr)
            else:  # couldn't validate (e.g. key missing 'voices_read') — trust it, TTS may still work
                print(f"[tts: could not validate ELEVENLABS_VOICE_ID ({_el_detail(exc)}); "
                      f"using it anyway — grant the key 'voices_read' to silence this]",
                      file=sys.stderr)
                return env_voice
    try:
        voices = client.voices.get_all().voices
        if voices:
            return voices[0].voice_id
    except Exception as exc:  # noqa: BLE001
        print(f"[tts: could not list voices ({_el_detail(exc)}) — grant the key 'voices_read', "
              f"or set ELEVENLABS_VOICE_ID]", file=sys.stderr)
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
                print(f"\n[tts error: {exc}]", file=sys.stderr)
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
            print("\n[tts: mpv not found — install mpv to hear replies]", file=sys.stderr)
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
            print(f"\n[tts playback error: {exc}]", file=sys.stderr)

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


def main():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        # Printed before logging is set up, so it reaches the screen.
        print(
            "ANTHROPIC_API_KEY is not set.\n"
            "Set it before starting the chat, e.g.:\n"
            "  export ANTHROPIC_API_KEY=sk-ant-...\n",
            file=sys.stderr,
        )
        return

    global log
    log, log_path = logs.setup()  # stderr -> log file; screen keeps only conversation

    # Import the heavy modules with stdout muted so the SDK's "load extensions"
    # banner and any other import chatter land in the log, not on screen.
    with logs.mute_stdout(log_path):
        import barge
        import stt
        import tools
        import vision

    lang = DEFAULT_LANG if DEFAULT_LANG in LANGUAGES or DEFAULT_LANG == "auto" else "en"

    model = build_model()
    tools_by_name = {t.name: t for t in tools.ALL_TOOLS}
    agent = model.bind_tools(tools.ALL_TOOLS)
    messages = [SystemMessage(content=system_prompt_for(lang))]

    # ElevenLabs powers both voice out (TTS) and voice in (Scribe STT), so one
    # client is shared. Without the key, nexon runs text-only in both directions.
    el = None
    speaker = None
    if os.environ.get("ELEVENLABS_API_KEY"):
        el = ElevenLabs()  # reads ELEVENLABS_API_KEY from env
        voice_id = resolve_voice_id(el)
        if voice_id:
            speaker = Speaker(el, voice_id)
            log.info("tts: using voice %s", voice_id)
        else:
            log.warning("tts: no voice available (set ELEVENLABS_VOICE_ID); text-only")
    else:
        log.info("tts: ELEVENLABS_API_KEY not set — running text-only")

    # Open the camera + live preview window now so you can watch what the robot
    # sees during the chat. Best-effort: if the camera isn't ready, the chat still
    # runs and detect_objects will report the error to Claude if it's called.
    show_window = os.environ.get("NEXON_NO_WINDOW") is None
    vision_on = False
    try:
        with logs.mute_stdout(log_path):  # capture SDK stdout during camera start
            vision.get_hub(show_window=show_window).ensure_started()
        vision_on = True
    except Exception as exc:  # noqa: BLE001
        log.warning("vision: camera unavailable (%s); detection will error if used", exc)

    # Voice input (push-to-talk Scribe STT) reuses the ElevenLabs client, so it's
    # only available when the key is set. It's forced to `lang` so background speech
    # in another language can't hijack the transcription.
    voice = stt.VoiceInput(client=el, language=stt_lang(lang)) if el else None
    voice_in = False
    speaking = False  # set per turn; referenced by the FSM hook below

    # Barge-in: interrupt nexon by speaking while it talks. Needs both voice in and
    # out; off unless NEXON_BARGE is truthy, since without echo cancellation an
    # open-speaker setup can hear nexon's own voice (see barge.py).
    can_barge = voice is not None and speaker is not None
    barge_enabled = can_barge and os.environ.get("NEXON_BARGE", "").lower() in ("1", "true", "yes", "on")
    listener = barge.Listener(device=voice.device) if can_barge else None
    interrupt = threading.Event()

    def barge_fired():
        """Called from the listener thread the moment the user starts talking."""
        interrupt.set()
        if speaker is not None:
            speaker.stop()
        log.info("barge-in: user started speaking — cutting off the reply")

    def on_state(state, _phase):
        """Arm the mic watcher only while nexon is actually speaking (RESPONSE)."""
        if listener is None:
            return
        active = barge_enabled and voice_in and speaking
        if state is fsm.State.RESPONSE and active:
            listener.arm(barge_fired)
        else:
            listener.disarm()

    lang_name = LANGUAGES.get(lang, "auto-detect")
    tts_on = speaker is not None
    print(f"\nnexon chat — model: {MODEL} | voice out: {'on' if tts_on else 'off'} "
          f"| vision: {'on' if vision_on else 'off'} | lang: {lang_name}"
          f"{' | barge-in: on' if barge_enabled else ''}")
    if vision_on and show_window:
        print("Live window open — keys there: g pixel grid (read AOI coords), "
              "d depth view, s snapshot, q close window.")
    print("Commands: /lang <en|hi|de|auto>, /voice, /barge, /reset, /mute, /unmute, /exit or /quit.\n")

    # Drives IDLE -> LISTENING -> WAITING -> RESPONSE; its on_change hook gates the
    # barge-in mic (live only during RESPONSE) and is the seam for a status line/LED.
    machine = fsm.Machine(on_change=on_state)

    def capture(vad: bool = False) -> str:
        """One utterance: LISTENING, then transcribe. Push-to-talk stops on Enter;
        vad=True endpoints on silence instead (used after a barge-in)."""
        machine.to(fsm.State.LISTENING)
        if vad:
            print("  listening… [stops after you pause] ", end="", flush=True)
            text = voice.listen_until_silence()
        else:
            print("  recording… [Enter to stop] ", end="", flush=True)
            text = voice.listen()
        print()
        return text

    pending_listen = False  # after a barge-in, jump straight into recording

    while True:
        machine.to(fsm.State.IDLE)

        if pending_listen and voice is not None and voice_in:
            # User barged in — they're already talking, so record now, don't prompt.
            # Endpoint on silence: pressing Enter mid-sentence would be awkward.
            pending_listen = False
            user_input = capture(vad=True)
            if not user_input:
                print("(heard nothing — try again)\n")
                continue
            print(f"you (voice)> {user_input}\n")
            typed = None  # skip the command/prompt path below
        else:
            pending_listen = False
            # Write the prompt to stdout ourselves (not via input()'s prompt arg):
            # readline sends its prompt to stderr, which we've redirected to the log,
            # so an input(prompt) prompt would be invisible on screen.
            prompt = "you [🎤 Enter to talk]> " if voice_in else "you> "
            print(prompt, end="", flush=True)
            try:
                typed = input().strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

        # Commands are always typed (work in either input mode). Skipped on the
        # barge-in fast path, where typed is None and user_input is already set.
        if typed is not None:
            if typed in ("/exit", "/quit"):
                break
            if typed.startswith("/lang"):
                parts = typed.split()
                choice = parts[1].lower() if len(parts) == 2 else ""
                if choice in LANGUAGES or choice == "auto":
                    lang = choice
                    messages[0] = SystemMessage(content=system_prompt_for(lang))
                    if voice is not None:
                        voice.language = stt_lang(lang)
                    name = LANGUAGES.get(lang, "auto-detect")
                    print(f"(language set to {name})\n")
                    log.info("language set to %s", lang)
                else:
                    cur = LANGUAGES.get(lang, "auto-detect")
                    print(f"(usage: /lang <en|hi|de|auto> — current: {cur})\n")
                continue
            if typed == "/voice":
                if voice is None:
                    print("(voice input unavailable — set ELEVENLABS_API_KEY)\n")
                    continue
                voice_in = not voice_in
                if voice_in:
                    print(f"(voice input on — mic: {voice.device or 'system default'}; "
                          f"press Enter to talk)\n")
                else:
                    print("(voice input off)\n")
                continue
            if typed == "/barge":
                if not can_barge:
                    print("(barge-in unavailable — needs both voice in and out)\n")
                    continue
                barge_enabled = not barge_enabled
                if barge_enabled and not voice_in:
                    print("(barge-in on — turn on /voice too; interrupt nexon by "
                          "speaking while it talks)\n")
                else:
                    print(f"(barge-in {'on' if barge_enabled else 'off'})\n")
                log.info("barge-in %s", "on" if barge_enabled else "off")
                continue
            if typed == "/reset":
                messages = [SystemMessage(content=system_prompt_for(lang))]
                print("(history cleared)\n")
                continue
            if typed == "/mute":
                tts_on = False
                print("(voice out off)\n")
                continue
            if typed == "/unmute":
                tts_on = speaker is not None
                print(f"(voice out {'on' if tts_on else 'unavailable'})\n")
                continue

            if voice_in and typed == "":
                # Push-to-talk: record until the next Enter, then transcribe.
                user_input = capture()
                if not user_input:
                    print("(heard nothing — try again)\n")
                    continue
                print(f"you (voice)> {user_input}\n")
            elif typed == "":
                continue
            else:
                user_input = typed

        log.info("USER: %s", user_input)

        # Snapshot history length so a failed/interrupted turn rolls back cleanly —
        # the turn may append several AIMessages and ToolMessages, not just one.
        history_len = len(messages)
        messages.append(HumanMessage(content=user_input))

        speaking = tts_on and speaker is not None
        interrupt.clear()  # fresh for this turn; the FSM hook may arm the mic watcher
        if speaking:
            speaker.speak()

        print("claude> ", end="", flush=True)
        try:
            ok = run_agent_turn(agent, messages, tools_by_name, speaker, speaking,
                                machine, interrupt)
        except KeyboardInterrupt:
            print("\n(interrupted)\n")
            ok = False

        if speaking:
            speaker.finish()  # flush the tail sentence and wait for playback
        machine.to(fsm.State.IDLE)  # release the mic watcher before we prompt/listen

        if interrupt.is_set():
            # Barge-in: drop the cut-off exchange and go capture what the user said.
            del messages[history_len:]
            print("\n(interrupted — listening)\n")
            pending_listen = voice_in
            continue
        if not ok:
            del messages[history_len:]  # discard the whole failed turn
            continue
        print("\n")


def _shutdown():
    try:
        import tools  # imported inside main(); re-import here hits the module cache

        tools.shutdown()
    except Exception:  # noqa: BLE001
        pass


if __name__ == "__main__":
    try:
        main()
    finally:
        _shutdown()
