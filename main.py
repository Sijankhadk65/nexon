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
import sys
import threading
import warnings

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from elevenlabs import VoiceSettings
from elevenlabs import stream as play_audio_stream
from elevenlabs.client import ElevenLabs
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

import logs

# tools/vision/stt are imported inside main() under logs.mute_stdout so their
# SDK/ML startup chatter is captured in the log file, not printed to the screen.

log = logging.getLogger("nexon")

MODEL = "claude-opus-4-8"
MAX_TOKENS = 4096
BASE_SYSTEM_PROMPT = (
    "You are nexon, an assistant that orchestrates a robot equipped with a camera. "
    "You have a detect_objects tool that looks through the robot's camera to find "
    "objects you name. Use it whenever the user asks what you can see, where "
    "something is, or to identify physical parts. Be concise; describe what you find "
    "in natural language rather than reading out raw coordinates."
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


def run_agent_turn(agent, messages, tools_by_name, speaker, speaking) -> bool:
    """Stream Claude's reply, run any tool calls, feed results back, repeat.

    Only the final natural-language prose is spoken; tool calls and their JSON
    results are Claude's private working state. Appends every AIMessage/ToolMessage
    to `messages`. Returns True on success; on error the caller rolls back the turn.
    """
    for _ in range(MAX_TOOL_ITERS):
        gathered = None
        response = []
        try:
            for chunk in agent.stream(messages):
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

        if gathered is None:
            return True
        messages.append(gathered)

        tool_calls = getattr(gathered, "tool_calls", None)
        if not tool_calls:
            return True

        # Run each requested tool and hand the results back for the next round.
        # Tool traffic is log-only — the screen stays a clean conversation.
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


def resolve_voice_id(client: ElevenLabs) -> str | None:
    """Pick a voice: env override, else the first voice on the account."""
    env_voice = os.environ.get("ELEVENLABS_VOICE_ID")
    if env_voice:
        return env_voice
    try:
        voices = client.voices.get_all().voices
        if voices:
            return voices[0].voice_id
    except Exception as exc:  # noqa: BLE001
        print(f"[tts: could not list voices: {exc}]", file=sys.stderr)
    return None


class Speaker:
    """Buffers streamed text into sentences, synthesizes each, and plays via mpv.

    Per assistant turn: `speak()` starts a background playback thread, `push()`
    feeds tokens (flushing whole sentences to the synth queue as they complete),
    and `finish()` flushes the tail and blocks until playback ends.
    """

    def __init__(self, client: ElevenLabs, voice_id: str):
        self.client = client
        self.voice_id = voice_id
        self._q: queue.Queue | None = None
        self._buf = ""
        self._thread: threading.Thread | None = None

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
            if sentence is _DONE:
                return
            try:
                audio = self.client.text_to_speech.convert(
                    self.voice_id,
                    text=sentence,
                    model_id=ELEVEN_MODEL,
                    output_format=OUTPUT_FORMAT,
                    voice_settings=VOICE_SETTINGS,
                )
                yield from audio
            except Exception as exc:  # noqa: BLE001
                print(f"\n[tts error: {exc}]", file=sys.stderr)
                # Stop synthesizing but drain the queue so finish() won't block.
                while self._q.get() is not _DONE:
                    pass
                return

    def _run(self):
        try:
            play_audio_stream(self._audio_gen())  # blocks, feeding mpv
        except Exception as exc:  # noqa: BLE001
            print(f"\n[tts playback error: {exc}]", file=sys.stderr)

    def speak(self):
        self._q = queue.Queue()
        self._buf = ""
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

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
        if self._q is not None:
            if self._buf.strip():
                self._q.put(self._buf)
            self._buf = ""
            self._q.put(_DONE)
        if self._thread is not None:
            self._thread.join()
        self._q = None
        self._thread = None


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

    lang_name = LANGUAGES.get(lang, "auto-detect")
    tts_on = speaker is not None
    print(f"\nnexon chat — model: {MODEL} | voice out: {'on' if tts_on else 'off'} "
          f"| vision: {'on' if vision_on else 'off'} | lang: {lang_name}")
    if vision_on and show_window:
        print("Live window open — keys there: d depth view, s snapshot, q close window.")
    print("Commands: /lang <en|hi|de|auto>, /voice, /reset, /mute, /unmute, /exit or /quit.\n")

    while True:
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

        # Commands are always typed (work in either input mode).
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
            print("  recording… [Enter to stop] ", end="", flush=True)
            user_input = voice.listen()
            print()
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
        if speaking:
            speaker.speak()

        print("claude> ", end="", flush=True)
        try:
            ok = run_agent_turn(agent, messages, tools_by_name, speaker, speaking)
        except KeyboardInterrupt:
            print("\n(interrupted)\n")
            ok = False

        if speaking:
            speaker.finish()  # flush the tail sentence and wait for playback

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
