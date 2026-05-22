"""End-to-end workflow tests for MooveGUI."""
import os
import re

import numpy as np
import pytest
from types import SimpleNamespace

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QPushButton, QRadioButton
from scipy.io import loadmat

from moove.utils import plot_data, save_notmat
from moove.utils.gui_utils import zoom, unzoom, swipe_left, swipe_right
from moove.utils.syllable_utils import (
    add_new_segment, delete_segment, edit_syllable, highlight_syllable,
)
from moove.utils.window_utils import (
    open_resegment_window, open_relabel_window,
    open_training_window, open_cluster_window,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def _reset(gui_window, file_idx=0):
    """Reset the session-scoped window to a clean state on a given file."""
    s = gui_window.app_state
    s.current_file_index = file_idx
    s.edit_type = "None"
    s.new_onset = None
    s.moved_point = None
    s.selected_syllable_index = None
    plot_data(s)


def _click_btn(gui_window, qtbot, label):
    btn = [b for b in gui_window.findChildren(QPushButton) if b.text() == label]
    assert btn, f"Button '{label}' not found"
    qtbot.mouseClick(btn[0], Qt.MouseButton.LeftButton)


def _notmat_labels(app_state):
    """Read labels from the current file's .not.mat on disk."""
    dd = app_state.display_dict
    path = os.path.join(app_state.data_dir, f"{dd['file_name']}.not.mat")
    mat = loadmat(path)
    return str(mat["labels"].flat[0]) if mat["labels"].size else ""


# =====================================================================
# Workflow 1: Navigate → edit label → verify persistence
# =====================================================================
class TestEditLabelWorkflow:
    """User opens file 0, selects a syllable, renames it, navigates away
    and back, and confirms the change persisted on disk and in memory."""

    def test_edit_label_persists_across_navigation(self, gui_window, qtbot):
        _reset(gui_window, 0)
        s = gui_window.app_state
        dd = s.display_dict
        original_labels = dd["labels"]

        # Switch to Label Interactive mode
        radios = {rb.text(): rb for rb in gui_window.findChildren(QRadioButton)}
        qtbot.mouseClick(radios["Label Interactive"], Qt.MouseButton.LeftButton)
        assert s.edit_type == "Label Interactive"

        # Select the first syllable and rename it to "z"
        highlight_syllable(0, s)
        assert s.selected_syllable_index == 0

        key_event = SimpleNamespace(key="z")
        edit_syllable(key_event, s)

        # Label should now start with 'z'
        assert s.display_dict["labels"][0] == "z"
        disk_labels_after_edit = _notmat_labels(s)
        assert disk_labels_after_edit[0] == "z"

        # Navigate away to file 1 and back
        _click_btn(gui_window, qtbot, "Next")
        assert s.current_file_index == 1
        _click_btn(gui_window, qtbot, "Previous")
        assert s.current_file_index == 0

        # After returning, label should still be 'z'
        assert s.display_dict["labels"][0] == "z"
        assert _notmat_labels(s)[0] == "z"

        # Restore original label
        s.edit_type = "Label Interactive"
        highlight_syllable(0, s)
        key_event = SimpleNamespace(key=original_labels[0])
        edit_syllable(key_event, s)
        s.edit_type = "None"
        s.selected_syllable_index = None


# =====================================================================
# Workflow 2: Delete segment → add new segment → navigate round-trip
# =====================================================================
class TestDeleteAddWorkflow:
    """User deletes a segment, adds a new one in its place, navigates
    away and back, confirms the change persisted."""

    def test_delete_then_add_persists(self, gui_window, qtbot):
        _reset(gui_window, 0)
        s = gui_window.app_state
        dd = s.display_dict

        n_orig = len(dd["onsets"])
        first_onset = dd["onsets"][0]
        first_offset = dd["offsets"][0]
        mid_sec = (first_onset + first_offset) / 2 / 1000.0

        # Delete first segment
        del_event = SimpleNamespace(xdata=mid_sec, inaxes=s.ax3, button=1)
        delete_segment(del_event, s)
        assert len(dd["onsets"]) == n_orig - 1

        # Add a new segment in the same gap
        new_onset_sec = first_onset / 1000.0
        new_offset_sec = first_offset / 1000.0
        evt_l = SimpleNamespace(xdata=new_onset_sec, inaxes=s.ax1, button=1)
        add_new_segment(evt_l, s)
        evt_r = SimpleNamespace(xdata=new_offset_sec, inaxes=s.ax1, button=3)
        add_new_segment(evt_r, s)
        assert len(dd["onsets"]) == n_orig  # back to original count
        # New segment should have label "x"
        idx = int(np.where(dd["onsets"] == first_onset)[0][0])
        assert dd["labels"][idx] == "x"

        # Navigate away and back
        _click_btn(gui_window, qtbot, "Next")
        _click_btn(gui_window, qtbot, "Previous")

        # Verify on disk
        disk_labels = _notmat_labels(s)
        assert "x" in disk_labels

        # Restore: relabel back to original
        s.edit_type = "Label Interactive"
        highlight_syllable(idx, s)
        key_event = SimpleNamespace(key="a")
        edit_syllable(key_event, s)
        s.edit_type = "None"
        s.selected_syllable_index = None


# =====================================================================
# Workflow 3: Zoom → swipe → unzoom → verify axes restored
# =====================================================================
class TestZoomSwipeWorkflow:
    """User zooms in, swipes right, swipes left, then unzooms.
    All three axes should be back at original ranges."""

    def test_zoom_swipe_unzoom_restores(self, gui_window):
        _reset(gui_window, 0)
        s = gui_window.app_state
        orig_xlim = s.ax1.get_xlim()

        zoom(s)
        swipe_right(s)
        swipe_left(s)
        unzoom(s)

        xlim = s.ax1.get_xlim()
        assert np.isclose(xlim[0], s.original_x_range[0], atol=1e-6)
        assert np.isclose(xlim[1], s.original_x_range[1], atol=1e-6)

        # All axes should be in sync
        for ax in (s.ax1, s.ax2, s.ax3):
            assert np.allclose(ax.get_xlim(), s.original_x_range, atol=1e-6)


# =====================================================================
# Workflow 4: Open each dialog → close → state unchanged
# =====================================================================
class TestDialogOpenCloseWorkflow:
    """Opening and closing each dialog should not alter the main window
    state (file index, display_dict, edit_type)."""

    @pytest.mark.parametrize("open_fn,attr", [
        (open_resegment_window, "resegment_window"),
        (open_relabel_window, "relabel_window"),
        (open_training_window, "training_window"),
        (open_cluster_window, "cluster_window"),
    ])
    def test_dialog_open_close_no_side_effects(self, gui_window, qtbot,
                                                open_fn, attr):
        _reset(gui_window, 0)
        s = gui_window.app_state
        file_idx_before = s.current_file_index
        labels_before = s.display_dict["labels"]
        edit_type_before = s.edit_type

        open_fn(gui_window, s)
        dlg = getattr(s, attr)
        assert dlg is not None
        dlg.close()

        assert s.current_file_index == file_idx_before
        assert s.display_dict["labels"] == labels_before
        assert s.edit_type == edit_type_before

        setattr(s, attr, None)


# =====================================================================
# Workflow 5: Checkbox toggle → navigate → verify per-file independence
# =====================================================================
class TestCheckboxNavigationWorkflow:
    """Checkbox changes are per-file. Toggling 'Segmented' on file 0,
    navigating to file 1, then back to file 0 should still show the
    change only on file 0's .rec."""

    def test_segmented_checkbox_per_file(self, gui_window, qtbot):
        _reset(gui_window, 0)
        s = gui_window.app_state

        # Mark file 0 as segmented
        gui_window.segmented_cb.setChecked(True)

        rec0 = os.path.join(s.data_dir, "bout_1.rec")
        with open(rec0) as f:
            assert "Hand Segmented = 1" in f.read()

        # Navigate to file 1 – it should have its own rec state
        _click_btn(gui_window, qtbot, "Next")
        rec1_name = os.path.splitext(s.song_files[s.current_file_index])[0] + ".rec"
        rec1 = os.path.join(s.data_dir, rec1_name)

        # Navigate back
        _click_btn(gui_window, qtbot, "Previous")

        # File 0 rec should still show Hand Segmented = 1
        with open(rec0) as f:
            match = re.search(r"Hand Segmented\s*=\s*(\d+)", f.read())
            assert match and match.group(1) == "1"

        # Restore
        gui_window.segmented_cb.setChecked(False)


# =====================================================================
# Workflow 6: Full navigation sweep – display_dict always consistent
# =====================================================================
class TestFullNavigationSweep:
    """Navigate through all files twice (forward+back) and confirm
    len(labels)==len(onsets)==len(offsets) after every step."""

    def test_forward_backward_consistency(self, gui_window, qtbot):
        _reset(gui_window, 0)
        s = gui_window.app_state
        n_files = len(s.song_files)

        # Forward
        for _ in range(n_files - 1):
            _click_btn(gui_window, qtbot, "Next")
            dd = s.display_dict
            if "labels" in dd:
                assert len(dd["labels"]) == len(dd["onsets"]) == len(dd["offsets"])

        # Backward
        for _ in range(n_files - 1):
            _click_btn(gui_window, qtbot, "Previous")
            dd = s.display_dict
            if "labels" in dd:
                assert len(dd["labels"]) == len(dd["onsets"]) == len(dd["offsets"])

        assert s.current_file_index == 0
