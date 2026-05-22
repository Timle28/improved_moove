"""GUI tests for MooveMainWindow.

Each test class shares a single MooveMainWindow instance (gui_window fixture)
to avoid macOS GPU context exhaustion that causes segfaults after ~8 windows.
Tests within a class reset state as needed.
"""
import os
import re

import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QPushButton, QRadioButton

from moove.utils import plot_data


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestWindowStartup:
    def test_window_is_visible(self, gui_window):
        assert gui_window.isVisible()

    def test_window_title(self, gui_window):
        assert gui_window.windowTitle() == "MooveGUI"

    def test_bird_combo_populated(self, gui_window):
        items = [gui_window.bird_combo.itemText(i)
                 for i in range(gui_window.bird_combo.count())]
        assert "bird_test" in items

    def test_experiment_combo_populated(self, gui_window):
        items = [gui_window.experiment_combo.itemText(i)
                 for i in range(gui_window.experiment_combo.count())]
        assert "experiment_a" in items

    def test_day_combo_populated(self, gui_window):
        items = [gui_window.day_combo.itemText(i)
                 for i in range(gui_window.day_combo.count())]
        assert "day_1" in items

    def test_file_combo_has_all_bouts(self, gui_window):
        items = [gui_window.file_combo.itemText(i)
                 for i in range(gui_window.file_combo.count())]
        assert "bout_1.wav" in items
        assert "bout_2.wav" in items
        assert "bout_3.wav" in items

    def test_display_dict_loaded(self, gui_window):
        dd = gui_window.app_state.display_dict
        assert dd is not None
        assert "file_name" in dd
        assert "song_data" in dd
        assert "sampling_rate" in dd


class TestDisplayDictConsistency:
    """The invariant len(labels)==len(onsets)==len(offsets) must always hold."""

    def test_initial_load_consistent(self, gui_window):
        dd = gui_window.app_state.display_dict
        if "labels" in dd:
            assert len(dd["labels"]) == len(dd["onsets"]) == len(dd["offsets"])


class TestNavigation:
    def _reset_to_file_0(self, window):
        s = window.app_state
        s.current_file_index = 0
        plot_data(s)

    def test_next_file(self, gui_window, qtbot):
        self._reset_to_file_0(gui_window)
        s = gui_window.app_state

        next_btn = [b for b in gui_window.findChildren(QPushButton)
                    if b.text() == "Next"]
        assert len(next_btn) == 1
        qtbot.mouseClick(next_btn[0], Qt.MouseButton.LeftButton)

        assert s.current_file_index == 1
        assert s.song_files[s.current_file_index] != s.song_files[0]

    def test_previous_file(self, gui_window, qtbot):
        s = gui_window.app_state
        s.current_file_index = 1
        plot_data(s)

        prev_btn = [b for b in gui_window.findChildren(QPushButton)
                    if b.text() == "Previous"]
        assert len(prev_btn) == 1
        qtbot.mouseClick(prev_btn[0], Qt.MouseButton.LeftButton)

        assert s.current_file_index == 0

    def test_next_clamps_at_last(self, gui_window, qtbot):
        """Pressing Next on the last file stays on the last file (clamped)."""
        s = gui_window.app_state
        last_idx = len(s.song_files) - 1
        s.current_file_index = last_idx
        plot_data(s)

        next_btn = [b for b in gui_window.findChildren(QPushButton)
                    if b.text() == "Next"]
        qtbot.mouseClick(next_btn[0], Qt.MouseButton.LeftButton)

        assert s.current_file_index == last_idx

    def test_display_dict_consistent_after_navigation(self, gui_window, qtbot):
        self._reset_to_file_0(gui_window)
        next_btn = [b for b in gui_window.findChildren(QPushButton)
                    if b.text() == "Next"]
        for _ in range(len(gui_window.app_state.song_files)):
            qtbot.mouseClick(next_btn[0], Qt.MouseButton.LeftButton)
            dd = gui_window.app_state.display_dict
            if "labels" in dd:
                assert len(dd["labels"]) == len(dd["onsets"]) == len(dd["offsets"]), \
                    f"Consistency broken for {dd['file_name']}"


class TestCheckboxes:
    def test_segmented_checkbox_writes_rec(self, gui_window, qtbot):
        s = gui_window.app_state
        s.current_file_index = 0
        plot_data(s)

        gui_window.segmented_cb.setChecked(True)

        rec_path = os.path.join(s.data_dir, "bout_1.rec")
        with open(rec_path) as f:
            content = f.read()
        match = re.search(r"Hand Segmented\s*=\s*(\d+)", content)
        assert match is not None
        assert match.group(1) == "1"

        # Uncheck to restore state
        gui_window.segmented_cb.setChecked(False)
        with open(rec_path) as f:
            content = f.read()
        match = re.search(r"Hand Segmented\s*=\s*(\d+)", content)
        assert match.group(1) == "0"

    def test_classified_checkbox_writes_rec(self, gui_window, qtbot):
        s = gui_window.app_state
        s.current_file_index = 0
        plot_data(s)

        gui_window.classified_cb.setChecked(True)

        rec_path = os.path.join(s.data_dir, "bout_1.rec")
        with open(rec_path) as f:
            content = f.read()
        match = re.search(r"Hand Classified\s*=\s*(\d+)", content)
        assert match is not None
        assert match.group(1) == "1"

        # Restore
        gui_window.classified_cb.setChecked(False)


class TestEditTypeRadios:
    def test_default_edit_type_is_none(self, gui_window):
        assert gui_window.app_state.edit_type == "None"

    def test_switch_to_delete_segment(self, gui_window, qtbot):
        radios = {rb.text(): rb
                  for rb in gui_window.findChildren(QRadioButton)}
        qtbot.mouseClick(radios["Delete Segment"], Qt.MouseButton.LeftButton)
        assert gui_window.app_state.edit_type == "Delete Segment"

    def test_switch_to_new_segment(self, gui_window, qtbot):
        radios = {rb.text(): rb
                  for rb in gui_window.findChildren(QRadioButton)}
        qtbot.mouseClick(radios["New Segment"], Qt.MouseButton.LeftButton)
        assert gui_window.app_state.edit_type == "New Segment"

    def test_switch_to_move_segment(self, gui_window, qtbot):
        radios = {rb.text(): rb
                  for rb in gui_window.findChildren(QRadioButton)}
        qtbot.mouseClick(radios["Move Segment"], Qt.MouseButton.LeftButton)
        assert gui_window.app_state.edit_type == "Move Segment"

    def test_switch_back_to_none(self, gui_window, qtbot):
        radios = {rb.text(): rb
                  for rb in gui_window.findChildren(QRadioButton)}
        qtbot.mouseClick(radios["Delete Segment"], Qt.MouseButton.LeftButton)
        qtbot.mouseClick(radios["None"], Qt.MouseButton.LeftButton)
        assert gui_window.app_state.edit_type == "None"
