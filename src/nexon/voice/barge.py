"""Voice barge-in: notice the user starting to talk while nexon is speaking.

While nexon plays a reply (FSM state RESPONSE), a background thread reads raw PCM
from the mic and watches its short-term loudness. When speech-level energy stays up
long enough to be a real interruption — not a transient click — it fires a callback
so the caller can cut off the TTS and go listen. This is the "smart gate": the mic
stays live *during* RESPONSE, but only a sustained human interruption counts, and it
is released (arecord killed) the moment RESPONSE ends.

The honest caveat: there is no acoustic echo cancellation here. If the mic picks up
nexon's own voice out of the speakers, that energy can trip the detector too. It is
reliable with headphones or a close/directional mic; with open speakers, raise
NEXON_BARGE_THRESHOLD until nexon stops interrupting itself. Barge-in is opt-in for
exactly this reason — it never touches the dependable push-to-talk path unless armed.

Config:
  NEXON_BARGE_THRESHOLD   RMS (int16) that counts as speech (default 800; higher =
                          less sensitive — the main knob for a noisy/echoey setup).
  NEXON_BARGE_SUSTAIN_MS  How long energy must stay up to fire (default 300 ms).
"""

import logging
import os
import subprocess
import threading

import numpy as np

log = logging.getLogger("nexon")

SAMPLE_RATE = 16000  # matches stt.py; mono 16-bit
FRAME_MS = 30        # analysis window; 30 ms @ 16 kHz = 480 samples = 960 bytes
THRESHOLD = float(os.environ.get("NEXON_BARGE_THRESHOLD", "800"))
SUSTAIN_MS = int(os.environ.get("NEXON_BARGE_SUSTAIN_MS", "300"))
# arecord's first frames are startup junk (device warm-up); ignore ~150 ms.
WARMUP_MS = 150


class Listener:
    """Arms a mic-energy watcher that fires once when the user starts speaking.

    `arm(on_speech)` starts recording and detection in a background thread and calls
    `on_speech()` (from that thread) the first time sustained speech is seen, then
    stops. `disarm()` releases the mic. Both are safe to call repeatedly.
    """

    def __init__(self, device: str | None = None, threshold: float = THRESHOLD,
                 sustain_ms: int = SUSTAIN_MS, sample_rate: int = SAMPLE_RATE,
                 frame_ms: int = FRAME_MS):
        self.device = device
        self.threshold = threshold
        self.sustain_ms = sustain_ms
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self._frame_bytes = int(sample_rate * frame_ms / 1000) * 2  # 2 bytes/sample
        self._warmup_frames = max(0, WARMUP_MS // frame_ms)
        self._proc: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None
        self._armed = threading.Event()
        self._on_speech = None

    def arm(self, on_speech) -> None:
        """Begin watching for a barge-in. No-op if already armed."""
        if self._thread is not None:
            return
        self._on_speech = on_speech
        self._armed.set()
        self._thread = threading.Thread(target=self._run, daemon=True, name="barge")
        self._thread.start()

    def disarm(self) -> None:
        """Stop watching and release the mic. No-op if not armed."""
        if self._thread is None:
            return
        self._armed.clear()
        self._stop_proc()  # unblocks the thread's blocking read with EOF
        self._thread.join(timeout=2)
        self._thread = None
        self._on_speech = None

    def _stop_proc(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            return
        proc.terminate()
        try:
            proc.wait(timeout=1)
        except subprocess.TimeoutExpired:
            proc.kill()

    def _run(self) -> None:
        cmd = ["arecord", "-q", "-f", "S16_LE", "-r", str(self.sample_rate),
               "-c", "1", "-t", "raw"]
        if self.device:
            cmd += ["-D", self.device]
        try:
            self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                          stderr=subprocess.DEVNULL)
        except FileNotFoundError:
            log.warning("barge: 'arecord' not found — barge-in unavailable this turn")
            return

        needed = max(1, self.sustain_ms // self.frame_ms)  # frames above threshold
        warmup = self._warmup_frames
        voiced = 0
        try:
            while self._armed.is_set():
                buf = self._proc.stdout.read(self._frame_bytes)
                if not buf or len(buf) < self._frame_bytes:
                    break  # arecord ended (disarmed) or short read
                if warmup > 0:
                    warmup -= 1
                    continue
                samples = np.frombuffer(buf, dtype=np.int16).astype(np.float32)
                rms = float(np.sqrt(np.mean(samples * samples))) if samples.size else 0.0
                if rms >= self.threshold:
                    voiced += 1
                    if voiced >= needed:
                        log.info("barge: speech detected (rms≈%.0f ≥ %.0f)",
                                 rms, self.threshold)
                        cb = self._on_speech
                        if cb is not None:
                            cb()  # caller cancels TTS + flags the interrupt
                        return
                else:
                    voiced = 0  # must be *sustained*; a lone spike doesn't count
        finally:
            self._stop_proc()
