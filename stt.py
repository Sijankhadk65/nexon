"""Push-to-talk speech-to-text via ElevenLabs Scribe.

Records the microphone with `arecord` (ALSA — already on the system, no PortAudio
needed) while you hold the floor, then transcribes with ElevenLabs' Scribe model.
One cloud provider for all of nexon's voice (in and out), and strongly multilingual
— German, English, etc. — with robust automatic language detection.

Push-to-talk flow, driven by main.py: recording starts, and `record()` blocks on
Enter to stop — press Enter to begin, speak, press Enter to end.

Config:
  NEXON_STT_MODEL  Scribe model id (default "scribe_v1").
  NEXON_STT_LANG   Pin a language (ISO code, e.g. "de", "en") to skip auto-detect.
  NEXON_MIC        arecord capture device (default: auto-detected USB mic).

Requires ELEVENLABS_API_KEY (the same key used for text-to-speech).
"""

import logging
import os
import re
import subprocess
import tempfile

log = logging.getLogger("nexon")

SAMPLE_RATE = 16000  # mono 16 kHz WAV is plenty for speech and keeps uploads small
STT_MODEL = os.environ.get("NEXON_STT_MODEL", "scribe_v1")
# None => let Scribe auto-detect the language; set NEXON_STT_LANG to pin it.
LANGUAGE = os.environ.get("NEXON_STT_LANG") or None


def default_mic() -> str | None:
    """Pick the capture device for arecord's -D flag.

    Honors NEXON_MIC if set. Otherwise auto-selects a USB mic when present —
    without PulseAudio/PipeWire, ALSA's `default` is the built-in mic, so a USB
    mic must be named explicitly. Returns a `plughw:<card>,0` string (plughw
    resamples if the mic doesn't do 16 kHz natively), or None to let arecord
    use its own default.
    """
    override = os.environ.get("NEXON_MIC")
    if override:
        return override
    try:
        listing = subprocess.run(
            ["arecord", "-l"], capture_output=True, text=True
        ).stdout
    except FileNotFoundError:
        return None
    for line in listing.splitlines():
        # e.g. "card 1: Device [USB PnP Sound Device], device 0: USB Audio [...]"
        m = re.match(r"card (\d+): (\S+) \[([^\]]*)\]", line)
        if m and ("usb" in (m.group(2) + m.group(3)).lower()):
            return f"plughw:{m.group(1)},0"
    return None


class VoiceInput:
    def __init__(self, client=None, model_id: str = STT_MODEL,
                 sample_rate: int = SAMPLE_RATE, device: str | None = None,
                 language: str | None = LANGUAGE):
        self._client = client  # an elevenlabs.ElevenLabs; created lazily if None
        self.model_id = model_id
        self.sample_rate = sample_rate
        self.device = device if device is not None else default_mic()
        self.language = language  # None => auto-detect

    def _ensure_client(self):
        if self._client is None:
            from elevenlabs.client import ElevenLabs

            self._client = ElevenLabs()  # reads ELEVENLABS_API_KEY
        return self._client

    def record(self) -> str | None:
        """Record until the user presses Enter. Returns a WAV path, or None on failure.

        arecord runs as a child process writing a mono 16-kHz WAV; the blocking
        input() is the push-to-talk 'stop' — it returns when you press Enter.
        """
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        cmd = ["arecord", "-q", "-f", "S16_LE", "-r", str(self.sample_rate), "-c", "1"]
        if self.device:  # explicit device (e.g. the USB mic); else arecord's default
            cmd += ["-D", self.device]
        cmd.append(tmp.name)
        try:
            proc = subprocess.Popen(cmd, stderr=subprocess.DEVNULL)
        except FileNotFoundError:
            log.error("voice: 'arecord' not found — install alsa-utils")
            os.unlink(tmp.name)
            return None

        try:
            input()  # push-to-talk: Enter stops recording
        except (EOFError, KeyboardInterrupt):
            pass
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
        return tmp.name

    def transcribe(self, wav_path: str) -> str:
        """Upload the WAV to ElevenLabs Scribe and return the transcript."""
        client = self._ensure_client()
        kwargs = {"model_id": self.model_id}
        if self.language:  # omit entirely to let Scribe auto-detect
            kwargs["language_code"] = self.language
        with open(wav_path, "rb") as audio:
            resp = client.speech_to_text.convert(file=audio, **kwargs)

        text = (getattr(resp, "text", "") or "").strip()
        if not self.language and text:
            lang = getattr(resp, "language_code", None)
            if lang:
                log.info("voice: detected language '%s'", lang)
        return text

    def listen(self) -> str:
        """Capture one push-to-talk utterance and return its transcript (maybe empty)."""
        wav_path = self.record()
        if wav_path is None:
            return ""
        try:
            # Too-short clips (< ~0.3 s) are almost always an accidental double-Enter.
            if os.path.getsize(wav_path) < self.sample_rate * 2 * 0.3:
                return ""
            return self.transcribe(wav_path)
        except Exception as exc:  # noqa: BLE001 — a failed transcription shouldn't crash the chat
            log.warning("voice: transcription failed: %s", exc)
            return ""
        finally:
            try:
                os.unlink(wav_path)
            except OSError:
                pass
