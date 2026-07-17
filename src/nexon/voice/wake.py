"""Always-on wake-word detection: notice the operator say "hey nexon".

While the conversation is IDLE, a background thread reads raw PCM from the mic and
runs each 80 ms frame through openWakeWord — a small, local, ONNX model that scores
how much the recent audio sounds like the wake phrase. When the score crosses a
threshold, it fires a callback once and stops; the caller then plays an earcon and
opens the mic for the actual utterance (stt.record_until_silence). Nothing here talks
to the cloud: openWakeWord runs on the CPU, so sitting armed costs no API calls.

This is the twin of `barge.Listener` — same arecord-in-a-thread shape, same arm/disarm
contract — but it watches for a *phrase* (a trained model) rather than for raw energy.
One mic, handed between them: the caller arms the wake word only while IDLE and disarms
it the instant a capture, a reply, or a manual push-to-talk needs the device, so there
is never a second arecord open on it.

THE MODEL IS A STAND-IN UNTIL A REAL ONE IS TRAINED. openWakeWord ships pretrained
phrases (hey_jarvis, alexa, hey_mycroft…) but not "hey nexon" — a custom phrase means
training a small model (openWakeWord's synthetic-data pipeline, run once, offline). So
the default here is `hey_jarvis`, and NEXON_WAKE_MODEL takes either another bundled name
or a path to a trained `hey_nexon.onnx`. Everything else — the gate, the earcon, the
follow-up window — is the same whichever model is loaded.

Config:
  NEXON_WAKE_MODEL      Bundled openWakeWord name, or a path to a .onnx/.tflite model
                        (default "hey_jarvis" — see the note above).
  NEXON_WAKE_THRESHOLD  Score in [0,1] that counts as the phrase (default 0.5; raise it
                        if it fires on the wrong words, lower it if it misses the phrase).
"""

import logging
import os
import subprocess
import tempfile
import threading
import wave

import numpy as np

log = logging.getLogger("nexon")

SAMPLE_RATE = 16000              # openWakeWord is trained on mono 16 kHz
CHUNK_SAMPLES = 1280             # 80 ms @ 16 kHz — the frame size the model expects
CHUNK_BYTES = CHUNK_SAMPLES * 2  # 2 bytes/sample (S16_LE)
MODEL = os.environ.get("NEXON_WAKE_MODEL", "hey_jarvis")
THRESHOLD = float(os.environ.get("NEXON_WAKE_THRESHOLD", "0.5"))
# Ignore the first frames after arming: the mic buffer can still hold the tail of an
# earcon or the reply that just ended, and a stale detection would re-fire at once.
WARMUP_FRAMES = 5  # ~400 ms


def _load_model(spec: str):
    """Build an openWakeWord Model for `spec` (a bundled name or a model file path).

    Raises if openWakeWord/onnxruntime aren't installed or the model can't be built —
    the caller treats any failure as "wake word unavailable" and degrades to push-to-talk.
    """
    import openwakeword
    from openwakeword.model import Model

    # The melspectrogram + embedding feature models (and the bundled wake models) are
    # fetched once into openWakeWord's cache. Best-effort: if the network is down but the
    # files are already there, Model() below still succeeds; if they're missing it raises.
    try:
        openwakeword.utils.download_models()
    except Exception as exc:  # noqa: BLE001
        log.info("wake: could not refresh openWakeWord models (%s); using the cache", exc)

    return Model(wakeword_models=[spec], inference_framework="onnx")


class WakeWord:
    """Arms a local wake-word watcher that fires once when the phrase is heard.

    `arm(on_wake)` starts arecord and detection in a background thread and calls
    `on_wake()` (from that thread) the first time the model's score crosses the
    threshold, then stops. `disarm()` releases the mic. Both are idempotent and safe
    to call repeatedly. `available` reports whether the model could be loaded — check
    it (off the GUI thread; the first call may download and build the model) before
    offering hands-free at all.
    """

    def __init__(self, device: str | None = None, model: str = MODEL,
                 threshold: float = THRESHOLD, sample_rate: int = SAMPLE_RATE):
        self.device = device
        self.spec = model
        self.threshold = threshold
        self.sample_rate = sample_rate
        self._model = None
        self._load_failed = False
        self._proc: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None
        self._armed = threading.Event()
        self._on_wake = None

    # ------------------------------------------------------------------ model

    def _ensure_model(self):
        """Load the model once (lazily). Returns it, or None if it can't be built."""
        if self._model is not None:
            return self._model
        if self._load_failed:
            return None
        try:
            self._model = _load_model(self.spec)
            log.info("wake: loaded model %r (threshold %.2f)", self.spec, self.threshold)
        except Exception as exc:  # noqa: BLE001 — absence degrades to push-to-talk, never crashes
            log.warning("wake: unavailable (%s) — `uv sync` to install openwakeword, or "
                        "set NEXON_WAKE_MODEL to a valid name/path", exc)
            self._load_failed = True
            return None
        return self._model

    @property
    def available(self) -> bool:
        """True if the wake model is loadable. May block on first call (download+build)."""
        return self._ensure_model() is not None

    # ------------------------------------------------------------------ arm/disarm

    def arm(self, on_wake) -> None:
        """Begin listening for the wake phrase. No-op if already armed or unavailable."""
        if self._thread is not None:
            return
        if self._ensure_model() is None:
            return
        self._on_wake = on_wake
        self._armed.set()
        self._thread = threading.Thread(target=self._run, daemon=True, name="wake")
        self._thread.start()

    def disarm(self) -> None:
        """Stop listening and release the mic. No-op if not armed."""
        if self._thread is None:
            return
        self._armed.clear()
        self._stop_proc()          # unblocks the thread's read() with EOF
        self._thread.join(timeout=2)
        self._thread = None
        self._on_wake = None

    def _stop_proc(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            return
        proc.terminate()
        try:
            proc.wait(timeout=1)
        except subprocess.TimeoutExpired:
            proc.kill()

    @staticmethod
    def _read_exact(stream, n: int) -> bytes | None:
        """Read exactly n bytes, or None at EOF — a partial read must not end an always-on loop."""
        buf = b""
        while len(buf) < n:
            chunk = stream.read(n - len(buf))
            if not chunk:
                return None
            buf += chunk
        return buf

    def _run(self) -> None:
        cmd = ["arecord", "-q", "-f", "S16_LE", "-r", str(self.sample_rate),
               "-c", "1", "-t", "raw"]
        if self.device:
            cmd += ["-D", self.device]
        try:
            self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                          stderr=subprocess.DEVNULL)
        except FileNotFoundError:
            log.warning("wake: 'arecord' not found — wake word unavailable this turn")
            return

        warmup = WARMUP_FRAMES
        try:
            while self._armed.is_set():
                buf = self._read_exact(self._proc.stdout, CHUNK_BYTES)
                if buf is None:
                    break  # arecord ended (disarmed)
                if warmup > 0:
                    warmup -= 1
                    continue
                samples = np.frombuffer(buf, dtype=np.int16)
                scores = self._model.predict(samples)
                score = max(scores.values()) if scores else 0.0
                if score >= self.threshold:
                    log.info("wake: phrase detected (score≈%.2f ≥ %.2f)", score, self.threshold)
                    cb = self._on_wake
                    if cb is not None:
                        cb()  # caller disarms us, plays the earcon, and opens the mic
                    return
        finally:
            self._stop_proc()


# ------------------------------------------------------------------ earcon

_EARCON_PATH: str | None = None


def _earcon_wav() -> str:
    """Write a short beep WAV once and return its path (cached for the process)."""
    global _EARCON_PATH
    if _EARCON_PATH and os.path.exists(_EARCON_PATH):
        return _EARCON_PATH
    sr, dur, freq = 16000, 0.12, 880.0
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    # A quick attack/release envelope so the beep doesn't click on/off.
    env = np.clip(np.minimum(t / 0.01, (dur - t) / 0.02), 0.0, 1.0)
    tone = (0.3 * np.sin(2 * np.pi * freq * t) * env * 32767).astype(np.int16)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", prefix="nexon-earcon-", delete=False)
    tmp.close()
    with wave.open(tmp.name, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(tone.tobytes())
    _EARCON_PATH = tmp.name
    return _EARCON_PATH


def earcon() -> None:
    """Play a short 'I'm listening' beep. Best-effort and blocking (~120 ms); never raises.

    Call it on a worker thread (not the GUI thread) just before opening the mic, so the
    beep finishes before recording starts and doesn't land in the captured audio.
    """
    try:
        subprocess.run(["aplay", "-q", _earcon_wav()],
                       stderr=subprocess.DEVNULL, timeout=2)
    except Exception as exc:  # noqa: BLE001 — a missing aplay just means no beep
        log.debug("wake: earcon failed (%s)", exc)
