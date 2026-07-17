"""Settings dialog: the Anthropic and ElevenLabs keys, and the TTS voice.

The keys are read once, by `session.start()`, which calls settings.apply_to_env() before it
builds the Anthropic and ElevenLabs clients. Those clients then hold whatever they were
given, so a key saved here takes effect the next time nexon starts — with one exception the
window handles: if the session never started for want of a key, saving one retries the boot
immediately rather than making the operator relaunch.

Fetching the voice list is a network call against ElevenLabs, so it runs on a worker thread
and returns through a signal. Nothing here blocks the GUI thread.

Where the environment already defines a key, that value wins at runtime no matter what is
typed here (see settings.apply_to_env). Rather than silently letting someone save a key that
will never be used, each overridden field says so and is marked read-only.
"""

import threading

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox, QFormLayout, QHBoxLayout,
                               QLabel, QLineEdit, QPushButton, QVBoxLayout)

from nexon import settings

ENV_NOTE = "set in the environment — that value wins; clear it to use this one"


class SettingsDialog(QDialog):
    _voices_loaded = Signal(object)     # list[(name, id)] or an Exception

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("nex-ON settings")
        self.setMinimumWidth(560)

        self._anthropic = QLineEdit(settings.get_stored("ANTHROPIC_API_KEY") or "")
        self._anthropic.setEchoMode(QLineEdit.Password)
        self._anthropic.setPlaceholderText("sk-ant-…")

        self._eleven = QLineEdit(settings.get_stored("ELEVENLABS_API_KEY") or "")
        self._eleven.setEchoMode(QLineEdit.Password)
        self._eleven.setPlaceholderText("optional — enables voice in and out")

        # Editable so a voice id can be pasted without the account ever being listed.
        self._voice = QComboBox()
        self._voice.setEditable(True)
        self._voice.setInsertPolicy(QComboBox.NoInsert)
        self._voice.setCurrentText(settings.get_stored("ELEVENLABS_VOICE_ID") or "")

        self._load = QPushButton("Load voices")
        self._load.clicked.connect(self._on_load_voices)

        self._status = QLabel("")
        self._status.setWordWrap(True)

        form = QFormLayout()
        form.addRow("Anthropic API key", self._with_note(self._anthropic, "ANTHROPIC_API_KEY"))
        form.addRow("ElevenLabs API key", self._with_note(self._eleven, "ELEVENLABS_API_KEY"))

        voice_row = QHBoxLayout()
        voice_row.addWidget(self._voice, stretch=1)
        voice_row.addWidget(self._load)
        form.addRow("Voice", self._wrap(voice_row, "ELEVENLABS_VOICE_ID"))

        where = QLabel(f"Keys are stored in the {settings.backend_name()}.\n"
                       f"Other settings: {settings._settings_file()}")
        where.setWordWrap(True)
        small = QFont()
        small.setPointSizeF(max(7.0, where.font().pointSizeF() - 1.5))
        where.setFont(small)
        where.setStyleSheet("color:#888;")

        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._on_save)
        buttons.rejected.connect(self.reject)

        root = QVBoxLayout(self)
        root.addLayout(form)
        root.addWidget(self._status)
        root.addWidget(where)
        root.addWidget(buttons)

        self._voices_loaded.connect(self._on_voices)

    # ------------------------------------------------------------------ layout

    def _with_note(self, widget, env_name):
        row = QHBoxLayout()
        row.addWidget(widget, stretch=1)
        return self._wrap(row, env_name, widget)

    def _wrap(self, row, env_name, widget=None):
        from PySide6.QtWidgets import QWidget
        holder = QWidget()
        column = QVBoxLayout(holder)
        column.setContentsMargins(0, 0, 0, 0)
        column.addLayout(row)
        if settings.is_overridden(env_name):
            note = QLabel(ENV_NOTE)
            note.setStyleSheet("color:#c47f00;")
            column.addWidget(note)
            if widget is not None:
                widget.setReadOnly(True)
        return holder

    # ------------------------------------------------------------------- voices

    def _on_load_voices(self) -> None:
        key = self._eleven.text().strip() or settings.resolve("ELEVENLABS_API_KEY") or ""
        if not key:
            self._status.setText("Enter an ElevenLabs API key first.")
            return
        self._load.setEnabled(False)
        self._status.setText("Loading voices…")

        def work():
            try:
                from elevenlabs.client import ElevenLabs
                voices = ElevenLabs(api_key=key).voices.get_all().voices
                self._voices_loaded.emit([(v.name, v.voice_id) for v in voices])
            except Exception as exc:  # noqa: BLE001 — reported in the dialog
                self._voices_loaded.emit(exc)

        threading.Thread(target=work, name="nexon-ui-voices", daemon=True).start()

    def _on_voices(self, result) -> None:
        self._load.setEnabled(True)
        if isinstance(result, Exception):
            # Listing needs the 'voices_read' permission; TTS itself does not. A key
            # without it still works, so keep whatever id is typed rather than clearing it.
            self._status.setText(f"Could not list voices: {result}. "
                                 "Grant the key 'voices_read', or paste a voice id.")
            return
        if not result:
            self._status.setText("That key has no voices on its account.")
            return

        current = self._voice.currentText().strip()
        self._voice.clear()
        for name, voice_id in result:
            self._voice.addItem(name, userData=voice_id)

        index = next((i for i, (_, vid) in enumerate(result) if vid == current), -1)
        if index >= 0:
            self._voice.setCurrentIndex(index)
        elif current:
            self._voice.setCurrentText(current)   # an id not on this account; keep it
        self._status.setText(f"Loaded {len(result)} voices.")

    def _selected_voice_id(self) -> str:
        """The id behind the chosen name, or whatever raw text was typed."""
        index = self._voice.currentIndex()
        if index >= 0 and self._voice.itemText(index) == self._voice.currentText():
            return self._voice.itemData(index) or ""
        return self._voice.currentText().strip()

    # -------------------------------------------------------------------- save

    def _on_save(self) -> None:
        try:
            if not settings.is_overridden("ANTHROPIC_API_KEY"):
                settings.set_stored("ANTHROPIC_API_KEY", self._anthropic.text())
            if not settings.is_overridden("ELEVENLABS_API_KEY"):
                settings.set_stored("ELEVENLABS_API_KEY", self._eleven.text())
            if not settings.is_overridden("ELEVENLABS_VOICE_ID"):
                settings.set_stored("ELEVENLABS_VOICE_ID", self._selected_voice_id())
        except Exception as exc:  # noqa: BLE001 — a locked keyring, a read-only home
            self._status.setText(f"Could not save: {exc}")
            return
        self.accept()
