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
import wave

import numpy as np

log = logging.getLogger("nexon")

SAMPLE_RATE = 16000  # mono 16 kHz WAV is plenty for speech and keeps uploads small
STT_MODEL = os.environ.get("NEXON_STT_MODEL", "scribe_v1")
# None => let Scribe auto-detect the language; set NEXON_STT_LANG to pin it.
LANGUAGE = os.environ.get("NEXON_STT_LANG") or None

# Silence-based endpointing (used only for barge-in capture, where pressing Enter to
# stop would be awkward mid-sentence). Record until the user has spoken and then gone
# quiet for VAD_SILENCE_S. Thresholds are int16 RMS per 30 ms frame.
VAD_FRAME_MS = 30
VAD_THRESHOLD = float(os.environ.get("NEXON_VAD_THRESHOLD", "500"))
VAD_SILENCE_S = float(os.environ.get("NEXON_VAD_SILENCE_S", "3.5"))
VAD_MAX_S = float(os.environ.get("NEXON_VAD_MAX_S", "30"))       # hard cap on length
VAD_START_TIMEOUT_S = float(os.environ.get("NEXON_VAD_START_TIMEOUT_S", "8"))  # give up if silent


def _script_mismatch(text: str, lang: str | None) -> bool:
    """True if `text` is clearly not in the forced language's script.

    ElevenLabs' language_code is only a hint, so Scribe can still return Hindi for
    a forced-English session (e.g. someone else speaking Hindi nearby). Hindi uses
    Devanagari while English/German use Latin, so a script check catches that hard
    case. It can't tell English from German (both Latin) — that's fine; the goal is
    to stop off-language audio hijacking the conversation.
    """
    if not lang or not text:
        return False
    devanagari = sum(1 for c in text if "ऀ" <= c <= "ॿ")
    latin = sum(1 for c in text if "a" <= c.lower() <= "z")
    if lang == "hi":
        return devanagari == 0 and latin > 0  # forced Hindi but Latin came back
    if lang in ("en", "de"):
        return devanagari > latin  # forced Latin-script language but got Devanagari
    return False


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
        # Scribe's language_code is only a hint; enforce the forced language by
        # discarding a transcript that came back in the wrong script.
        if _script_mismatch(text, self.language):
            log.info("voice: discarded off-language transcript (forced %s): %r",
                     self.language, text)
            return ""
        if not self.language and text:
            detected = getattr(resp, "language_code", None)
            if detected:
                log.info("voice: detected language '%s'", detected)
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

    def record_until_silence(self, silence_s: float = VAD_SILENCE_S,
                             max_s: float = VAD_MAX_S,
                             start_timeout_s: float = VAD_START_TIMEOUT_S) -> str | None:
        """Record until ~silence_s of quiet follows speech, then stop on its own.

        Unlike record() (which stops on Enter), this reads the mic stream, tracks
        per-frame loudness, and endpoints itself once the user finishes talking —
        used for barge-in, where the user is already mid-sentence. Writes the captured
        PCM to a WAV and returns its path, or None if nothing was said / on failure.
        """
        frame_bytes = int(self.sample_rate * VAD_FRAME_MS / 1000) * 2  # 2 bytes/sample
        frame_s = VAD_FRAME_MS / 1000
        cmd = ["arecord", "-q", "-f", "S16_LE", "-r", str(self.sample_rate),
               "-c", "1", "-t", "raw"]
        if self.device:
            cmd += ["-D", self.device]
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        except FileNotFoundError:
            log.error("voice: 'arecord' not found — install alsa-utils")
            return None

        frames = bytearray()
        started = False       # have we heard speech yet?
        silent = 0.0          # seconds of continuous silence since last speech
        elapsed = 0.0
        try:
            while True:
                buf = proc.stdout.read(frame_bytes)
                if not buf or len(buf) < frame_bytes:
                    break  # stream ended
                frames += buf
                elapsed += frame_s
                samples = np.frombuffer(buf, dtype=np.int16).astype(np.float32)
                rms = float(np.sqrt(np.mean(samples * samples))) if samples.size else 0.0
                if rms >= VAD_THRESHOLD:
                    started = True
                    silent = 0.0
                elif started:
                    silent += frame_s
                    if silent >= silence_s:  # spoke, then went quiet — done
                        break
                if elapsed >= max_s:
                    break
                if not started and elapsed >= start_timeout_s:
                    break  # user never spoke
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()

        if not started:
            return None

        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        with wave.open(tmp.name, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)  # S16_LE
            w.setframerate(self.sample_rate)
            w.writeframes(bytes(frames))
        return tmp.name

    def listen_until_silence(self) -> str:
        """Like listen(), but endpoints on silence instead of Enter (for barge-in)."""
        wav_path = self.record_until_silence()
        if wav_path is None:
            return ""
        try:
            if os.path.getsize(wav_path) < self.sample_rate * 2 * 0.3:
                return ""
            return self.transcribe(wav_path)
        except Exception as exc:  # noqa: BLE001
            log.warning("voice: transcription failed: %s", exc)
            return ""
        finally:
            try:
                os.unlink(wav_path)
            except OSError:
                pass
