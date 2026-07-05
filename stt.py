"""Offline push-to-talk speech-to-text for voice input.

Records the microphone with `arecord` (ALSA — already on the system, no PortAudio
needed) while you hold the floor, then transcribes locally with faster-whisper on
the CPU. No cloud, no per-use cost. The model loads lazily on first use and is
cached afterwards.

Push-to-talk flow, driven by main.py: recording starts, and `record()` blocks on
Enter to stop — so you press Enter to begin, speak, and press Enter to end.

Model size via NEXON_WHISPER_MODEL (default base.en — a good speed/accuracy balance
on CPU; try small.en for more accuracy, tiny.en for more speed).
"""

import os
import re
import subprocess
import sys
import tempfile

SAMPLE_RATE = 16000  # Whisper's native rate; record straight to it
MODEL_SIZE = os.environ.get("NEXON_WHISPER_MODEL", "base.en")


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
    def __init__(self, model_size: str = MODEL_SIZE, sample_rate: int = SAMPLE_RATE,
                 device: str | None = None):
        self.model_size = model_size
        self.sample_rate = sample_rate
        self.device = device if device is not None else default_mic()
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            from faster_whisper import WhisperModel

            print(f"[voice: loading Whisper '{self.model_size}' "
                  f"(first run downloads it)…]", file=sys.stderr)
            # int8 is the fast CPU path; base.en fits comfortably in memory.
            self._model = WhisperModel(self.model_size, device="cpu", compute_type="int8")
        return self._model

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
            print("[voice: 'arecord' not found — install alsa-utils]", file=sys.stderr)
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
        """Transcribe a WAV file to text (faster-whisper decodes it via PyAV)."""
        model = self._ensure_model()
        segments, _ = model.transcribe(wav_path, language="en", beam_size=1)
        return " ".join(seg.text for seg in segments).strip()

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
        finally:
            try:
                os.unlink(wav_path)
            except OSError:
                pass
