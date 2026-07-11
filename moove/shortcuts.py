# shortcuts.py
"""Customizable keyboard shortcuts: a small registry + JSON persistence, plus
Qt key-event <-> string helpers shared by the canvas and the settings dialog."""
import json
import os

from PyQt6.QtCore import Qt

# (action_id, human label, default key). Order defines the settings-dialog order.
SHORTCUT_DEFS = [
    ("previous_file",     "Previous file",            ","),
    ("next_file",         "Next file",                "."),
    ("play",              "Play current view",        "space"),
    ("undo",              "Undo segmentation",        "ctrl+z"),
    ("redo",              "Redo segmentation",        "ctrl+shift+z"),
    ("edit_none",         "Edit: None / cancel",      "escape"),
    ("edit_new",          "Edit: New segment",        "n"),
    ("edit_delete",       "Edit: Delete segment",     "d"),
    ("edit_move",         "Edit: Move segment",       "m"),
    ("edit_label",        "Edit: Label interactive",  "l"),
    ("toggle_segmented",  "Toggle 'Segmented' flag",  ";"),
    ("toggle_classified", "Toggle 'Classified' flag", "'"),
]

_DEFAULTS = {aid: key for aid, _, key in SHORTCUT_DEFS}
_LABELS = {aid: label for aid, label, _ in SHORTCUT_DEFS}

_SPECIAL = {
    Qt.Key.Key_Left: "left", Qt.Key.Key_Right: "right",
    Qt.Key.Key_Up: "up", Qt.Key.Key_Down: "down",
    Qt.Key.Key_Escape: "escape", Qt.Key.Key_Backspace: "backspace",
    Qt.Key.Key_Delete: "delete", Qt.Key.Key_Return: "enter",
    Qt.Key.Key_Enter: "enter", Qt.Key.Key_Space: "space",
    Qt.Key.Key_Tab: "tab",
}


def qt_key_to_string(e):
    """Normalize a Qt key event to a string like 'ctrl+z', 'left', 'space', 'a'.
    (On macOS Qt maps Cmd -> ControlModifier, so 'ctrl' means Cmd there.)"""
    base = _SPECIAL.get(e.key())
    if base is None:
        txt = e.text()
        if txt and txt.isprintable() and txt.strip():
            base = txt.lower()
    if base is None:
        return None
    mods = []
    m = e.modifiers()
    if m & Qt.KeyboardModifier.ControlModifier:
        mods.append("ctrl")
    if m & Qt.KeyboardModifier.AltModifier:
        mods.append("alt")
    if m & Qt.KeyboardModifier.ShiftModifier:
        mods.append("shift")
    return "+".join(mods + [base])


def pretty_key(key, mac=False):
    """Human-readable label for a key string (for the settings dialog)."""
    if not key:
        return "—"
    parts = key.split("+")
    sym = {
        "ctrl": "⌘" if mac else "Ctrl",
        "shift": "⇧" if mac else "Shift",
        "alt": "⌥" if mac else "Alt",
        "space": "Space", "escape": "Esc", "left": "←", "right": "→",
        "up": "↑", "down": "↓", "enter": "⏎", "backspace": "⌫",
        "delete": "⌦", "tab": "Tab",
    }
    out = [sym.get(p, p.upper() if len(p) == 1 else p.capitalize()) for p in parts]
    return ("" if mac else "+").join(out) if mac else "+".join(out)


class ShortcutManager:
    def __init__(self, config_dir):
        self.path = os.path.join(config_dir, "shortcuts.json")
        self._keys = dict(_DEFAULTS)
        self._load()

    def _load(self):
        try:
            with open(self.path) as f:
                data = json.load(f)
            for aid in self._keys:
                if isinstance(data.get(aid), str):
                    self._keys[aid] = data[aid]
        except Exception:
            pass

    def save(self):
        try:
            with open(self.path, "w") as f:
                json.dump(self._keys, f, indent=2)
        except Exception:
            pass

    def key_for(self, action_id):
        return self._keys.get(action_id)

    def action_for(self, key):
        if not key:
            return None
        for aid, k in self._keys.items():
            if k == key:
                return aid
        return None

    def set(self, action_id, key):
        self._keys[action_id] = key

    def reset(self):
        self._keys = dict(_DEFAULTS)

    def items(self):
        """[(action_id, label, key), ...] in registry order."""
        return [(aid, _LABELS[aid], self._keys[aid]) for aid, _, _ in SHORTCUT_DEFS]
