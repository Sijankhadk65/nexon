# nexon

A voice-driven, vision-enabled assistant that **orchestrates robots**. Claude is the
brain: you talk to it, it looks through a depth camera, finds and measures objects,
and talks back — in your language of choice.

The first target is a **welding cobot**: identifying welding operands (metal tubes,
flanges, plates) and measuring their real-world dimensions so the robot knows what
it's working with. The design is deliberately modular so the same orchestrator can
drive other robots and swap in different vision models.

```
 ┌─────────────────────────── nexon (main.py) ───────────────────────────┐
 │  you ──voice/text──▶  Claude (orchestrator)  ──speaks──▶ you           │
 │                            │  calls tools                              │
 │                            ▼                                           │
 │                      detect_objects ─────────────▶ VisionHub          │
 │                                                     │                  │
 │   Gemini 336L ──colour+depth──▶ open-vocab detect ──▶ measure (mm)    │
 │                                     (Grounding DINO)   (depth + PCA)   │
 └───────────────────────────────────────────────────────────────────────┘
```

## Features

- **Conversational orchestration** — Claude (Opus 4.8, via LangChain) runs the show,
  deciding when to look and what to look for, mid-conversation.
- **Voice in and out** — push-to-talk speech-to-text (ElevenLabs Scribe) and
  streamed, sentence-buffered text-to-speech (ElevenLabs).
- **Open-vocabulary detection** — name any object in plain language ("metal tube",
  "flange"); no per-class training needed. Backend is swappable.
- **Real-world measurement** — fuses the RGB detection with the aligned depth stream
  and camera intrinsics to estimate an object's length, width, and distance in
  millimetres.
- **Live vision window** — watch exactly what the robot sees and measures, with boxes,
  labels, and dimensions overlaid.
- **Forced language** — lock the whole session to English, Hindi, or German so
  background speech in another language can't hijack the conversation.
- **Clean terminal + full logs** — the screen shows only the conversation; everything
  else (status, tool traces, library noise) goes to a timestamped log file.

## Hardware

- **Orbbec Gemini 336L** depth camera over USB 3.0 (RGB + depth).
- A microphone (a USB mic is auto-detected) for voice input.
- Speakers for voice output (`mpv` is used for playback).
- Runs CPU-only; no GPU required (models run on CPU, so first-look latency is a few
  seconds on modest hardware).

## Requirements

- Python **3.11**
- [`uv`](https://docs.astral.sh/uv/) for dependency management
- System tools: `arecord` (ALSA, for mic capture) and `mpv` (for TTS playback)
- API keys:
  - `ANTHROPIC_API_KEY` — **required** (Claude)
  - `ELEVENLABS_API_KEY` — optional; enables voice in *and* out (text-only without it)

## Install

```bash
uv sync
```

Key Python dependencies (installed automatically): `pyorbbecsdk2` (Orbbec SDK v2),
`langchain-anthropic`, `elevenlabs`, `transformers` + `torch` (Grounding DINO),
`opencv-python`, `numpy`. Vision and STT models download on first use and are cached.

## Run

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export ELEVENLABS_API_KEY=...        # optional, for voice
uv run python main.py
```

Then just talk to it — e.g. *"What can you see?"*, *"Is there a metal tube in front of
you?"*, *"How big is it?"*. Claude looks through the camera, measures if asked, and
replies out loud.

### In-chat commands

| Command | Action |
|---|---|
| `/voice` | Toggle push-to-talk mic input (Enter to start, Enter to stop) |
| `/lang <en\|hi\|de\|auto>` | Force the conversation language (input + reply) |
| `/mute` · `/unmute` | Turn spoken replies off / on |
| `/reset` | Clear conversation history (keeps language) |
| `/exit` · `/quit` | Leave |

### Live vision window keys

`d` toggle depth view · `s` save a snapshot · `q` close the window (chat keeps running).

### Standalone vision viewer

Run detection/measurement without the chat (uses the camera exclusively):

```bash
uv run python view.py "metal tube" "flange"        # measure these, live
uv run python view.py --no-measure "person" "cup"   # boxes only, faster
```

## Configuration (environment variables)

| Variable | Default | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | Claude access (required) |
| `ELEVENLABS_API_KEY` | — | Voice in/out (optional) |
| `ELEVENLABS_VOICE_ID` | first voice | TTS voice |
| `NEXON_LANG` | `en` | Forced language: `en`, `hi`, `de`, or `auto` |
| `NEXON_DETECTOR` | `grounding-dino` | Detection backend |
| `NEXON_MIC` | auto (USB mic) | `arecord -D` capture device |
| `NEXON_STT_MODEL` | `scribe_v1` | ElevenLabs STT model |
| `NEXON_NO_WINDOW` | (unset) | Set to disable the live preview window |

## Project layout

| Module | Responsibility |
|---|---|
| `main.py` | Chat loop, agentic tool loop, TTS, voice input, language, logging setup |
| `camera.py` | Gemini 336L capture: RGB, depth aligned to colour, camera intrinsics |
| `detector.py` | Swappable `Detector` interface + Grounding DINO open-vocab backend |
| `dimensioner.py` | Detection box + depth → real-world dimensions (deprojection + PCA) |
| `vision.py` | `VisionHub`: shared camera/detector + live preview window |
| `tools.py` | The `detect_objects` LangChain tool Claude calls |
| `viz.py` | Shared drawing helpers (boxes, labels, dimensions, depth colormap) |
| `view.py` | Standalone live detection/measurement viewer |
| `stt.py` | Push-to-talk speech-to-text (ElevenLabs Scribe + `arecord`) |
| `logs.py` | Splits output: conversation to screen, everything to a timestamped log |

Logs are written to `Log/nexon_<timestamp>.txt`.

## How measurement works

1. Enable colour + depth on the 336L; software-align depth to the colour frame so
   pixels correspond 1:1.
2. The detector returns a pixel box for the named object.
3. Within that box, the nearest coherent depth band is isolated (dropping the
   background) and deprojected to a 3D point cloud using the camera intrinsics.
4. PCA on the cloud gives the principal axis (length), the perpendicular axis
   (width), and the median distance — all in millimetres.

Measurements are **approximate**: the camera sees only the facing surface, and
accuracy is bounded by how tightly the detection box frames the object.

## Notes & limitations

- Detection/measurement is **on-demand single-frame**, not real-time tracking —
  a good fit for a conversational orchestrator and for on-CPU hardware.
- Measurement accuracy improves with a tighter box; a segmentation mask (planned)
  would tighten it further.
- `NEXON_LANG` forces the reply language (via the system prompt) and the STT
  language. Because ElevenLabs' language hint is soft, a script guard additionally
  discards transcripts in the wrong script (e.g. Devanagari during a forced-English
  session).

## Roadmap

- **Segmentation masks** (e.g. FastSAM) for tighter, more accurate dimensioning.
- **Fine-tuned welding-part detector** once a labelled dataset exists (the detector
  interface already supports swapping backends).
- Additional robot tools beyond vision (the orchestrator/tool structure generalises).
