"""Design tokens: colour, type, spacing, and the three accessibility signals.

Everything visual in ui/ resolves through this module, so a value can be defended in one
place rather than argued about in six. Three ideas are worth stating outright, because
they explain most of what looks arbitrary below.

TYPE IS SIZE-SPECIFIC. Tracking (letter spacing) is not one number. Large text reads too
loose as it grows, so it gets NEGATIVE tracking; small text — the status strip, the
timestamps — reads too tight, so it gets positive tracking. A single `letter-spacing` for
the whole app is wrong somewhere. `font()` therefore takes a size and derives its own
tracking, and callers never set it by hand.

SIZES ARE RELATIVE TO THE DESKTOP'S FONT. The operator may have scaled their text up, and
a shop-floor screen is read from further away than a laptop. `_base()` reads the platform
font's point size once and every size in the scale is a multiple of it, so a larger system
font enlarges the whole interface instead of overflowing it. Spacing follows through `em()`.

THE THREE PREFERENCES ARE INDEPENDENT. Reduced motion, reduced transparency, and increased
contrast are separate switches and are read separately. Reduced motion never means "no
feedback" — it means a cross-fade instead of a slide, and it is honoured in ui/motion.py.
Reduced transparency makes the glass frosty and opaque rather than removing the panels.
Each is an env var (NEXON_REDUCED_MOTION, NEXON_REDUCED_TRANSPARENCY, NEXON_MORE_CONTRAST);
motion additionally honours GNOME's `enable-animations`, since that is the one the desktop
actually exposes and an operator who turned animations off system-wide meant it here too.
"""

import functools
import os
import subprocess

from PySide6.QtGui import QColor, QFont, QFontDatabase
from PySide6.QtWidgets import QApplication

# --------------------------------------------------------------------------- #
# Colour
#
# A dark palette, because the camera is behind everything and a light interface over a
# moving image is unreadable. Foreground values carry alpha rather than being pre-blended:
# over live video there is no fixed backdrop to blend against.
# --------------------------------------------------------------------------- #

# Behind everything, on the frames before the camera opens.
VOID = QColor("#0B0B0C")

# Text. Not flat grey: over a translucent, changing backdrop, grey text loses contrast as
# the video moves under it. Secondary text stays near-white and sheds ALPHA instead, which
# holds its edge. This is what vibrancy does on Apple's platforms.
INK = QColor(255, 255, 255, 255)
INK_SECONDARY = QColor(235, 235, 245, 160)
INK_TERTIARY = QColor(235, 235, 245, 96)

# Materials. GLASS is the standard floating surface: dark, translucent, blurred backdrop.
# GLASS_DEEP is heavier, for surfaces that must separate a structural region (the arc
# banner) rather than merely float. Never put one directly on the other — see glass.py.
GLASS = QColor(28, 28, 30, 150)
GLASS_DEEP = QColor(18, 18, 20, 205)
GLASS_STROKE = QColor(255, 255, 255, 28)
# The bright top edge is light catching the lip of a real material. It is what makes a
# translucent rectangle read as a pane of glass rather than a grey box.
GLASS_EDGE = QColor(255, 255, 255, 48)

# Speech. The operator's own words are the one place a saturated fill earns its keep: it
# separates two voices at a glance, with no label to read. nexon's bubble is nearly opaque:
# it sits on the solid column, where translucency would buy nothing but a bubble the same
# colour as the surface under it.
USER_FILL = QColor(10, 132, 255, 235)
USER_INK = QColor(255, 255, 255, 255)
NEXON_FILL = QColor(38, 38, 42, 240)
NEXON_INK = QColor(255, 255, 255, 240)

# The composer's fill: a shade lighter than the column, so the field reads as something you
# can type into rather than a hole in the panel.
FIELD = QColor(36, 36, 40, 240)

ACCENT = QColor(10, 132, 255)

# Machine state. Red means energised, and it is used for nothing else anywhere in the UI —
# an operator must never have to ask whether this particular red means an arc.
ARC_LIVE = QColor(255, 69, 58)
ARC_LIVE_DIM = QColor(255, 69, 58, 40)
WELD_DRY = QColor(48, 209, 88)
WARN = QColor(255, 159, 10)

# The veil laid over the camera. Light: the only text on the frame now is the arc panel and
# the action pills, and both carry their own material — the video no longer has to be dimmed
# into a backdrop for a conversation, because the conversation has its own column.
VEIL = QColor(0, 0, 0, 30)
SCRIM_BOTTOM = QColor(0, 0, 0, 70)

# The conversation's column. Solid, not glass: prose over a moving weld scene is unreadable
# at any blur radius, and legibility is not a thing to trade for depth.
COLUMN = QColor(17, 17, 19)
HAIRLINE = QColor(255, 255, 255, 20)


# --------------------------------------------------------------------------- #
# Type
# --------------------------------------------------------------------------- #

# In preference order. The system UI font already ships optical sizing, tracking tables
# and legibility tuning that a webfont would have to reinvent, so we take the best one
# the machine has rather than bundling our own.
_FAMILIES = ("Inter", "SF Pro Text", "Cantarell", "Ubuntu", "Noto Sans", "DejaVu Sans")

# The type scale, as multiples of the desktop's font size.
DISPLAY = 1.55
TITLE = 1.20
BODY = 1.05
CAPTION = 0.86
MICRO = 0.78


@functools.lru_cache(maxsize=1)
def _base() -> float:
    """The desktop's UI font size in points. Requires a live QApplication."""
    app = QApplication.instance()
    size = app.font().pointSizeF() if app else 10.0
    return size if size > 0 else 10.0


@functools.lru_cache(maxsize=1)
def family() -> str:
    available = set(QFontDatabase.families())
    for name in _FAMILIES:
        if name in available:
            return name
    app = QApplication.instance()
    return app.font().family() if app else "sans-serif"


def _tracking(points: float) -> float:
    """Letter spacing in points for text of this size. Negative as text grows.

    Type designed for one size and scaled to another is wrong at both. Below ~11pt the
    letters need air; above ~16pt they need to be pulled together. The two anchors here
    (+0.30pt at micro, −0.45pt at display) are interpolated across the range, which is a
    cheap stand-in for the per-size tracking table a real type family would carry.
    """
    lo, hi = 8.0, 22.0
    t = max(0.0, min(1.0, (points - lo) / (hi - lo)))
    return 0.30 + t * (-0.75)


def _leading(scale: float) -> float:
    """Line height as a multiple of the font size. Inverse to size, as leading should be."""
    if scale >= TITLE:
        return 1.12          # tight: large text is already visually spacious
    if scale >= BODY:
        return 1.42          # comfortable: this is what conversation is read at
    return 1.30


def font(scale: float = BODY, weight: int = QFont.Weight.Normal, caps: bool = False) -> QFont:
    """A font at `scale`, with tracking derived from its resolved size.

    `weight` may be a QFont.Weight or one of its integer values (400, 600, 700…); PySide6
    only accepts the enum, so a plain int is coerced rather than rejected.
    """
    points = _base() * scale
    f = QFont(family())
    f.setPointSizeF(points)
    f.setWeight(QFont.Weight(weight))
    f.setLetterSpacing(QFont.AbsoluteSpacing, _tracking(points))
    if caps:
        f.setCapitalization(QFont.AllUppercase)
        # Capitals have no ascender/descender variety to separate them, so they need
        # more air than the size alone would suggest.
        f.setLetterSpacing(QFont.AbsoluteSpacing, _tracking(points) + 0.6)
    return f


def leading_px(scale: float = BODY) -> int:
    """Line height in device pixels for text at `scale` — for Qt's rich text line-height."""
    return round(_base() * scale * _leading(scale) * 96 / 72)


def em(multiple: float = 1.0) -> int:
    """Spacing in device pixels, expressed in ems of the base font.

    Padding, gaps and radii all go through this, so scaling the desktop's text scales the
    layout with it instead of cramming larger glyphs into fixed boxes.
    """
    return max(1, round(_base() * multiple * 96 / 72))


# Corner radii. Continuous enough at these sizes; larger surfaces get larger radii, which
# is what keeps a big panel and a small pill looking like the same material.
RADIUS_PILL = 999
RADIUS_PANEL = 18
RADIUS_BUBBLE = 18
RADIUS_SHEET = 22


# --------------------------------------------------------------------------- #
# Accessibility preferences
# --------------------------------------------------------------------------- #

def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


@functools.lru_cache(maxsize=1)
def _gnome_animations_disabled() -> bool:
    """True if the desktop has animations switched off. Best-effort and cached.

    GNOME's `enable-animations` is the only reduced-motion signal a Linux desktop reliably
    exposes, and Qt does not surface it. An operator who turned it off system-wide meant it
    here as well. Any failure — no gsettings, no session bus, another desktop — is simply
    "not disabled"; this must never be a reason the window fails to open.
    """
    try:
        out = subprocess.run(
            ["gsettings", "get", "org.gnome.desktop.interface", "enable-animations"],
            capture_output=True, text=True, timeout=1.0,
        ).stdout.strip()
    except Exception:  # noqa: BLE001 — no gsettings, no bus, not GNOME
        return False
    return out == "false"


def reduced_motion() -> bool:
    """Cross-fade instead of moving. Never means 'no feedback' — see ui/motion.py."""
    return _env_flag("NEXON_REDUCED_MOTION") or _gnome_animations_disabled()


def reduced_transparency() -> bool:
    """Frost the glass: raise the fill's opacity and skip the backdrop blur entirely."""
    return _env_flag("NEXON_REDUCED_TRANSPARENCY")


def more_contrast() -> bool:
    """Near-solid surfaces with a defined, contrasting border."""
    return _env_flag("NEXON_MORE_CONTRAST")


def surface(base: QColor) -> QColor:
    """`base`, adjusted for the transparency and contrast preferences."""
    color = QColor(base)
    if more_contrast():
        return QColor(color.red(), color.green(), color.blue(), 255)
    if reduced_transparency():
        # Opaque enough that losing the blur costs no legibility, still dark enough to
        # read as a distinct layer above the video rather than a hole punched in it.
        color.setAlpha(min(255, color.alpha() + 70))
    return color


def stroke(base: QColor = GLASS_STROKE) -> QColor:
    """The panel border. Under increased contrast it becomes a real, visible edge."""
    if more_contrast():
        return QColor(255, 255, 255, 190)
    return base


def rgba(color: QColor) -> str:
    """A Qt-stylesheet rgba() string, for the few widgets styled by stylesheet."""
    return (f"rgba({color.red()},{color.green()},{color.blue()},"
            f"{color.alphaF():.3f})")
