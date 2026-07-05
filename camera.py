"""RGB video capture from the Orbbec Gemini 336L for object detection.

Opens the camera's color sensor via the Orbbec SDK v2 (`pyorbbecsdk2`) and yields
continuous frames as BGR NumPy arrays — the layout OpenCV and most object-detection
models expect. The device also exposes depth and IR streams on the same pipeline,
but those aren't wired up yet; this module is RGB-only for now.

Run it directly for a live preview (press q or Esc to quit):
    uv run python camera.py

With no display available (headless/SSH), it prints the measured frame rate and
writes a single snapshot to `frame.jpg` instead of opening a window.
"""

import sys
import time

import cv2
import numpy as np
import pyorbbecsdk as ob

# The 336L color sensor's default mode. MJPG at 1280x720x30 is well-supported and
# keeps USB bandwidth modest; cv2 decodes it straight to BGR. Set WIDTH/HEIGHT to
# None to just take the sensor's default profile.
WIDTH = 1280
HEIGHT = 720
FPS = 30
COLOR_FORMAT = ob.OBFormat.MJPG

# wait_for_frames timeout (ms). A couple of frame intervals is plenty at 30 fps.
FRAME_TIMEOUT_MS = 1000


def frame_to_bgr(frame) -> np.ndarray | None:
    """Convert an Orbbec color frame to a contiguous BGR image, or None if undecodable.

    Handles the formats the 336L color sensor emits: MJPG (decoded by OpenCV) plus
    the raw pixel layouts (RGB/BGR/BGRA/YUYV). Grayscale IR-style formats aren't
    expected on the color stream and return None.
    """
    fmt = frame.get_format()
    w, h = frame.get_width(), frame.get_height()
    data = np.frombuffer(frame.get_data(), dtype=np.uint8)

    if fmt == ob.OBFormat.MJPG:
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    if fmt == ob.OBFormat.RGB:
        return cv2.cvtColor(data.reshape(h, w, 3), cv2.COLOR_RGB2BGR)
    if fmt == ob.OBFormat.BGR:
        return np.ascontiguousarray(data.reshape(h, w, 3))
    if fmt == ob.OBFormat.BGRA:
        return cv2.cvtColor(data.reshape(h, w, 4), cv2.COLOR_BGRA2BGR)
    if fmt == ob.OBFormat.RGBA:
        return cv2.cvtColor(data.reshape(h, w, 4), cv2.COLOR_RGBA2BGR)
    if fmt == ob.OBFormat.YUYV:
        return cv2.cvtColor(data.reshape(h, w, 2), cv2.COLOR_YUV2BGR_YUYV)
    return None


class OrbbecCamera:
    """Continuous RGB frame source for the Gemini 336L.

    Use as a context manager so the pipeline is always stopped cleanly:

        with OrbbecCamera() as cam:
            for bgr in cam.frames():
                run_object_detection(bgr)

    `frames()` yields decoded BGR NumPy arrays and runs until the caller stops
    iterating (e.g. `break`), so downstream object detection sets the pace.
    """

    def __init__(self, width=WIDTH, height=HEIGHT, fps=FPS, color_format=COLOR_FORMAT):
        self.width = width
        self.height = height
        self.fps = fps
        self.color_format = color_format

        ctx = ob.Context()
        ctx.set_logger_level(ob.OBLogLevel.NONE)  # keep the terminal chat-clean
        devices = ctx.query_devices()
        if devices.get_count() == 0:
            raise RuntimeError(
                "No Orbbec device found. Is the Gemini 336L plugged in over USB 3.0?"
            )

        self._pipeline = ob.Pipeline(devices.get_device_by_index(0))
        self.device_info = self._pipeline.get_device().get_device_info()
        self._config = self._build_config()
        self._started = False

    def _build_config(self) -> "ob.Config":
        """Enable the requested color mode, falling back to the sensor default."""
        profiles = self._pipeline.get_stream_profile_list(ob.OBSensorType.COLOR_SENSOR)
        profile = self._select_color_profile(profiles)
        config = ob.Config()
        config.enable_stream(profile)
        # Reflect what we actually got, in case we fell back to the default.
        vsp = profile.as_video_stream_profile()
        self.width, self.height = vsp.get_width(), vsp.get_height()
        self.fps, self.color_format = vsp.get_fps(), vsp.get_format()
        return config

    def _select_color_profile(self, profiles):
        """Find a profile matching the requested WIDTH/HEIGHT/FPS/format, else default."""
        if self.width is not None and self.height is not None:
            for i in range(profiles.get_count()):
                vsp = profiles.get_stream_profile_by_index(i).as_video_stream_profile()
                if (
                    vsp.get_width() == self.width
                    and vsp.get_height() == self.height
                    and vsp.get_fps() == self.fps
                    and vsp.get_format() == self.color_format
                ):
                    return vsp
            print(
                f"[camera: {self.width}x{self.height}@{self.fps} "
                f"{self.color_format} unavailable — using sensor default]",
                file=sys.stderr,
            )
        return profiles.get_default_video_stream_profile()

    def start(self):
        if not self._started:
            self._pipeline.start(self._config)
            self._started = True

    def stop(self):
        if self._started:
            self._pipeline.stop()
            self._started = False

    def read(self) -> np.ndarray | None:
        """Grab one BGR frame, or None if none arrived within the timeout."""
        frames = self._pipeline.wait_for_frames(FRAME_TIMEOUT_MS)
        if frames is None:
            return None
        color = frames.get_color_frame()
        if color is None:
            return None
        return frame_to_bgr(color)

    def frames(self):
        """Yield BGR frames continuously until the caller stops iterating."""
        self.start()
        while True:
            bgr = self.read()
            if bgr is not None:
                yield bgr

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()


def _has_display() -> bool:
    import os

    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def main():
    """Live RGB preview — a stand-in for the object-detection loop to come."""
    try:
        cam = OrbbecCamera()
    except RuntimeError as exc:
        print(f"[camera error: {exc}]", file=sys.stderr)
        return

    info = cam.device_info
    print(
        f"camera: {info.get_name()} (serial {info.get_serial_number()}) "
        f"— color {cam.width}x{cam.height}@{cam.fps} {cam.color_format}"
    )

    show = _has_display()
    if not show:
        print("[no display — measuring FPS for 3s, then saving a settled frame to frame.jpg]")

    frames_seen = 0
    last_bgr = None
    t0 = time.time()
    try:
        with cam:
            for bgr in cam.frames():
                frames_seen += 1
                last_bgr = bgr
                if show:
                    cv2.imshow("Gemini 336L — RGB", bgr)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):  # q or Esc
                        break
                elif time.time() - t0 >= 3.0:
                    # Save the last frame, not the first: auto-exposure needs a
                    # second or two to converge, so an early frame looks black.
                    cv2.imwrite("frame.jpg", last_bgr)
                    break
    except KeyboardInterrupt:
        pass
    finally:
        if show:
            cv2.destroyAllWindows()

    elapsed = time.time() - t0
    if elapsed > 0:
        print(f"captured {frames_seen} frames in {elapsed:.1f}s "
              f"({frames_seen / elapsed:.1f} fps)")


if __name__ == "__main__":
    main()
