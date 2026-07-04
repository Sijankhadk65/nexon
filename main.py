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

import os
import queue
import sys
import threading

from elevenlabs import VoiceSettings
from elevenlabs import stream as play_audio_stream
from elevenlabs.client import ElevenLabs
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

MODEL = "claude-opus-4-8"
MAX_TOKENS = 4096
SYSTEM_PROMPT = "You are a helpful, concise assistant."

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
        print(
            "ANTHROPIC_API_KEY is not set.\n"
            "Set it before starting the chat, e.g.:\n"
            "  export ANTHROPIC_API_KEY=sk-ant-...\n",
            file=sys.stderr,
        )
        return

    model = build_model()
    messages = [SystemMessage(content=SYSTEM_PROMPT)]

    # Voice is optional — fall back to text-only if ElevenLabs isn't configured.
    speaker = None
    if os.environ.get("ELEVENLABS_API_KEY"):
        el = ElevenLabs()  # reads ELEVENLABS_API_KEY from env
        voice_id = resolve_voice_id(el)
        if voice_id:
            speaker = Speaker(el, voice_id)
        else:
            print(
                "[tts: no voice available — set ELEVENLABS_VOICE_ID to a voice ID "
                "from your dashboard (the key may lack the voices_read permission). "
                "Running text-only.]",
                file=sys.stderr,
            )
    else:
        print("[tts: ELEVENLABS_API_KEY not set — running text-only]", file=sys.stderr)

    tts_on = speaker is not None
    print(f"\nnexon chat — model: {MODEL} | voice: {'on' if tts_on else 'off'}")
    print("Commands: /reset, /mute, /unmute, /exit or /quit.\n")

    while True:
        try:
            user_input = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue
        if user_input in ("/exit", "/quit"):
            break
        if user_input == "/reset":
            messages = [SystemMessage(content=SYSTEM_PROMPT)]
            print("(history cleared)\n")
            continue
        if user_input == "/mute":
            tts_on = False
            print("(voice off)\n")
            continue
        if user_input == "/unmute":
            tts_on = speaker is not None
            print(f"(voice {'on' if tts_on else 'unavailable'})\n")
            continue

        messages.append(HumanMessage(content=user_input))

        speaking = tts_on and speaker is not None
        if speaking:
            speaker.speak()

        print("claude> ", end="", flush=True)
        reply_parts = []
        try:
            for chunk in model.stream(messages):
                text = chunk_text(chunk)
                if text:
                    reply_parts.append(text)
                    print(text, end="", flush=True)
                    if speaking:
                        speaker.push(text)
        except KeyboardInterrupt:
            print("\n(interrupted)\n")
            if speaking:
                speaker.finish()
            messages.pop()
            continue
        except Exception as exc:  # noqa: BLE001
            print(f"\n[error: {exc}]\n")
            if speaking:
                speaker.finish()
            messages.pop()
            continue

        if speaking:
            speaker.finish()  # flush the tail sentence and wait for playback
        print("\n")
        messages.append(AIMessage(content="".join(reply_parts)))


if __name__ == "__main__":
    main()
