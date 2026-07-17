"""`nexon` — launches the operator console.

This module used to BE the application: a terminal REPL that owned the message history, the
tool loop, the TTS speaker and the barge-in listener, printing the conversation to stdout.
Two things were wrong with that, and both are structural rather than cosmetic.

  The conversation and the machine lived in different processes. `nexon` ran the agent;
  `nexon-ui` ran the operator console. `controller.get_controller()` is a process-wide
  singleton and robot.py keeps its weld/velocity/weave state in module globals, so running
  both gave two Controllers and two sets of those globals driving one arm. A `disarm`
  clicked in the console mutated a different `WELD_LIVE` than the one the agent's trace was
  about to read, and the whole single-owner argument in controller.py held only within a
  process. It now holds across the application, because there is one.

  The conversation had nowhere to appear. Everything the agent does is physical — it looks
  through a camera, it moves an arm, it strikes an arc — and a terminal can show none of it.

So the loop moved to `nexon.session` (headless: no Qt, no terminal, one event callback) and
the frontend moved to `nexon.ui`, which runs the session on a worker thread with the camera
behind it. Both console scripts land here.

What is not carried over: the `/lang`, `/voice`, `/barge`, `/mute` and `/reset` commands.
Language and voice-out live in the window's overflow menu; barge-in is no longer a toggle —
`nexon.session` arms `nexon.voice.barge` automatically whenever nexon can both hear and speak,
so the operator can always just talk over a reply to cut it off.
"""

from nexon.ui.app import main

__all__ = ["main"]


if __name__ == "__main__":
    raise SystemExit(main())
