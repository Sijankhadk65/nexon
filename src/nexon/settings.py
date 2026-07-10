"""Persisted configuration: API keys, and the voice to speak with.

Two stores, because two kinds of value:

  * SECRETS (the Anthropic and ElevenLabs keys) go to the OS keyring — gnome-keyring,
    KWallet, whatever the desktop provides. A shop-floor box with no session bus has no
    secret service, so there is a fallback: ~/.config/nexon/secrets.ini, created 0600 and
    never widened. It is a fallback, not a default; the keyring is tried first every time.

  * PLAIN settings (the voice id) go to ~/.config/nexon/settings.ini. Nothing sensitive.

Both live under XDG_CONFIG_HOME, deliberately NOT in the repo's data/ directory — that
directory is git-tracked, and a key written there would be committed.

The environment always WINS. `export ANTHROPIC_API_KEY=...` overrides whatever is stored,
so a CI run, a container, or `env VAR=x uv run nexon` behaves the way anyone would expect,
and the settings dialog can tell the operator when what they typed is being shadowed.
`apply_to_env()` pushes stored values into os.environ for the libraries (langchain,
elevenlabs) that read them straight from there — it never overwrites a value already set.
"""

import configparser
import logging
import os
from pathlib import Path

log = logging.getLogger("nexon")

APP_NAME = "nexon"

# Read from the environment by the libraries themselves, so the names must match theirs.
SECRET_KEYS = ("ANTHROPIC_API_KEY", "ELEVENLABS_API_KEY")
PLAIN_KEYS = ("ELEVENLABS_VOICE_ID",)


def config_dir() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME") or (Path.home() / ".config")
    return Path(base) / APP_NAME


def _settings_file() -> Path:
    return config_dir() / "settings.ini"


def _secrets_file() -> Path:
    return config_dir() / "secrets.ini"


# --------------------------------------------------------------------------- #
# Secrets: keyring first, 0600 file second
# --------------------------------------------------------------------------- #

def _keyring():
    """The usable keyring backend, or None. Never raises — a missing bus is normal."""
    try:
        import keyring
        from keyring.backends import fail

        backend = keyring.get_keyring()
        if isinstance(backend, fail.Keyring):
            return None
        return keyring
    except Exception:  # noqa: BLE001 — no keyring installed, no dbus, locked wallet
        return None


def _read_secrets_file() -> configparser.ConfigParser:
    parser = configparser.ConfigParser()
    path = _secrets_file()
    if path.exists():
        parser.read(path)
    return parser


def _write_secrets_file(parser: configparser.ConfigParser) -> None:
    path = _secrets_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    # Create the file 0600 BEFORE any secret reaches it. Opening then chmod-ing would
    # leave a window where the key is world-readable.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as fh:
        parser.write(fh)
    os.chmod(path, 0o600)   # an existing file may predate this and be too permissive


def backend_name() -> str:
    """Human-readable description of where secrets are being kept right now."""
    if _keyring() is not None:
        import keyring
        return f"system keyring ({keyring.get_keyring().__class__.__name__})"
    return f"file {_secrets_file()} (0600)"


def get_secret(name: str) -> str | None:
    """Stored secret, ignoring the environment. Keyring first, then the fallback file."""
    kr = _keyring()
    if kr is not None:
        try:
            value = kr.get_password(APP_NAME, name)
            if value:
                return value
        except Exception:  # noqa: BLE001 — locked wallet, user cancelled the unlock prompt
            log.warning("settings: keyring read failed for %s; trying the file", name)
    return _read_secrets_file().get("secrets", name, fallback=None) or None


def set_secret(name: str, value: str) -> None:
    """Store (or clear, if value is empty) a secret. Prefers the keyring."""
    value = (value or "").strip()
    kr = _keyring()
    if kr is not None:
        try:
            if value:
                kr.set_password(APP_NAME, name, value)
            else:
                try:
                    kr.delete_password(APP_NAME, name)
                except Exception:  # noqa: BLE001 — nothing stored is not an error
                    pass
            return
        except Exception:  # noqa: BLE001
            log.warning("settings: keyring write failed for %s; falling back to file", name)

    parser = _read_secrets_file()
    if not parser.has_section("secrets"):
        parser.add_section("secrets")
    if value:
        parser.set("secrets", name, value)
    else:
        parser.remove_option("secrets", name)
    _write_secrets_file(parser)


# --------------------------------------------------------------------------- #
# Plain settings
# --------------------------------------------------------------------------- #

def get_plain(name: str) -> str | None:
    parser = configparser.ConfigParser()
    path = _settings_file()
    if path.exists():
        parser.read(path)
    return parser.get("nexon", name, fallback=None) or None


def set_plain(name: str, value: str) -> None:
    parser = configparser.ConfigParser()
    path = _settings_file()
    if path.exists():
        parser.read(path)
    if not parser.has_section("nexon"):
        parser.add_section("nexon")
    value = (value or "").strip()
    if value:
        parser.set("nexon", name, value)
    else:
        parser.remove_option("nexon", name)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        parser.write(fh)


# --------------------------------------------------------------------------- #
# Resolution
# --------------------------------------------------------------------------- #

def get_stored(name: str) -> str | None:
    """Whatever is persisted for `name`, ignoring the environment."""
    if name in SECRET_KEYS:
        return get_secret(name)
    return get_plain(name)


def set_stored(name: str, value: str) -> None:
    if name in SECRET_KEYS:
        set_secret(name, value)
    else:
        set_plain(name, value)


def is_overridden(name: str) -> bool:
    """True when the environment shadows what is stored — the dialog says so out loud."""
    return bool(os.environ.get(name))


def resolve(name: str) -> str | None:
    """Effective value: the environment if set, else what is stored."""
    return os.environ.get(name) or get_stored(name)


def apply_to_env() -> None:
    """Push stored values into os.environ for libraries that read it directly.

    Never overwrites an existing variable, so an explicit `export` always wins. Call this
    once at startup, before anything constructs an SDK client.
    """
    for name in (*SECRET_KEYS, *PLAIN_KEYS):
        if os.environ.get(name):
            continue
        value = get_stored(name)
        if value:
            os.environ[name] = value
