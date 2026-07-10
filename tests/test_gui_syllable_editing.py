"""GUI tests for syllable editing functions and GUI utilities."""
import os
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.io import loadmat

from moove.utils.gui_utils import zoom, unzoom, swipe_left, swipe_right
from moove.utils.syllable_utils import valid_move, delete_segment, add_new_segment
from moove.utils import plot_data


# ---------------------------------------------------------------------------
# Helper: reset window to a clean file-0 state
# ---------------------------------------------------------------------------
def _reset_file_0(gui_window):
    s = gui_window.app_state
    s.current_file_index = 0
    s.edit_type = "None"
    s.new_onset = None
    s.moved_point = None
    s.selected_syllable_index = None
    plot_data(s)


# =====================================================================
# valid_move – pure logic, no GUI fixtures required
# =====================================================================
class TestValidMove:
    """Tests for the valid_move() boundary-validation helper."""

    @pytest.fixture()
    def segments(self):
        onsets = np.array([100.0, 300.0, 500.0])
        offsets = np.array([200.0, 400.0, 600.0])
        return onsets, offsets

    # --- Onset moves ---
    def test_onset_move_valid_middle(self, segments):
        onsets, offsets = segments
        # Move onset[1] to 250 ms → between offsets[0]=200 and offsets[1]=400
        assert valid_move(0.250, 1, onsets, offsets, is_onset=True) is True

    def test_onset_move_invalid_middle_past_prev_offset(self, segments):
        onsets, offsets = segments
        # Move onset[1] to 150 ms → before offsets[0]=200 → invalid
        assert valid_move(0.150, 1, onsets, offsets, is_onset=True) is False

    def test_onset_move_first_valid(self, segments):
        onsets, offsets = segments
        # Move onset[0] to 50 ms → still before offsets[0]=200
        assert valid_move(0.050, 0, onsets, offsets, is_onset=True) is True

    def test_onset_move_first_invalid(self, segments):
        onsets, offsets = segments
        # Move onset[0] to 250 ms → past offsets[0]=200 → invalid
        assert valid_move(0.250, 0, onsets, offsets, is_onset=True) is False

    def test_onset_move_last_valid(self, segments):
        onsets, offsets = segments
        # Move onset[2] to 450 ms → after offsets[1]=400
        assert valid_move(0.450, 2, onsets, offsets, is_onset=True) is True

    def test_onset_move_last_invalid(self, segments):
        onsets, offsets = segments
        # Move onset[2] to 350 ms → before offsets[1]=400 → invalid
        assert valid_move(0.350, 2, onsets, offsets, is_onset=True) is False

    # --- Offset moves ---
    def test_offset_move_valid_middle(self, segments):
        onsets, offsets = segments
        # Move offset[1] to 450 ms → between onsets[1]=300 and onsets[2]=500
        assert valid_move(0.450, 1, onsets, offsets, is_onset=False) is True

    def test_offset_move_invalid_middle_past_next_onset(self, segments):
        onsets, offsets = segments
        # Move offset[1] to 550 ms → past onsets[2]=500 → invalid
        assert valid_move(0.550, 1, onsets, offsets, is_onset=False) is False

    def test_offset_move_first_valid(self, segments):
        onsets, offsets = segments
        # Move offset[0] to 250 ms → between onsets[0]=100 and onsets[1]=300
        assert valid_move(0.250, 0, onsets, offsets, is_onset=False) is True

    def test_offset_move_first_invalid(self, segments):
        onsets, offsets = segments
        # Move offset[0] to 350 ms → past onsets[1]=300 → invalid
        assert valid_move(0.350, 0, onsets, offsets, is_onset=False) is False

    def test_offset_move_last_valid(self, segments):
        onsets, offsets = segments
        # Move offset[2] to 550 ms → still after onsets[2]=500
        assert valid_move(0.550, 2, onsets, offsets, is_onset=False) is True

    def test_offset_move_last_invalid(self, segments):
        onsets, offsets = segments
        # Move offset[2] to 450 ms → before onsets[2]=500 → invalid
        assert valid_move(0.450, 2, onsets, offsets, is_onset=False) is False


# =====================================================================
# Zoom / Unzoom / Swipe
# =====================================================================
class TestZoomUnzoom:
    def test_zoom_reduces_x_range(self, gui_window):
        _reset_file_0(gui_window)
        s = gui_window.app_state
        x0, x1 = s.ax1.get_xlim()
        original_width = x1 - x0

        zoom(s)

        new_x0, new_x1 = s.ax1.get_xlim()
        assert (new_x1 - new_x0) < original_width

    def test_unzoom_resets_range(self, gui_window):
        _reset_file_0(gui_window)
        s = gui_window.app_state

        zoom(s)
        unzoom(s)

        x0, x1 = s.ax1.get_xlim()
        assert np.isclose(x0, s.original_x_range[0], atol=1e-6)
        assert np.isclose(x1, s.original_x_range[1], atol=1e-6)

    def test_zoom_preserves_center(self, gui_window):
        _reset_file_0(gui_window)
        s = gui_window.app_state
        x0, x1 = s.ax1.get_xlim()
        orig_center = (x0 + x1) / 2

        zoom(s)

        new_x0, new_x1 = s.ax1.get_xlim()
        new_center = (new_x0 + new_x1) / 2
        assert np.isclose(orig_center, new_center, atol=1e-6)

    def test_zoom_all_axes_sync(self, gui_window):
        _reset_file_0(gui_window)
        s = gui_window.app_state

        zoom(s)

        ax1_lim = s.ax1.get_xlim()
        ax2_lim = s.ax2.get_xlim()
        ax3_lim = s.ax3.get_xlim()
        assert np.allclose(ax1_lim, ax2_lim, atol=1e-6)
        assert np.allclose(ax1_lim, ax3_lim, atol=1e-6)


class TestSwipe:
    def test_swipe_right_shifts_x(self, gui_window):
        _reset_file_0(gui_window)
        s = gui_window.app_state
        # Zoom in first so there is room to swipe: the view clamps to the
        # recording extent, so swiping at full zoom is correctly a no-op.
        zoom(s); zoom(s); zoom(s)
        x0_before, x1_before = s.ax1.get_xlim()

        swipe_right(s)

        x0_after, x1_after = s.ax1.get_xlim()
        assert x0_after > x0_before
        assert x1_after > x1_before

    def test_swipe_left_shifts_x(self, gui_window):
        _reset_file_0(gui_window)
        s = gui_window.app_state
        zoom(s); zoom(s); zoom(s)
        x0_before, x1_before = s.ax1.get_xlim()

        swipe_left(s)

        x0_after, x1_after = s.ax1.get_xlim()
        assert x0_after < x0_before
        assert x1_after < x1_before

    def test_swipe_right_left_returns(self, gui_window):
        """Swipe right then left should approximately restore position."""
        _reset_file_0(gui_window)
        s = gui_window.app_state
        x0_orig, x1_orig = s.ax1.get_xlim()

        swipe_right(s)
        swipe_left(s)

        x0_after, x1_after = s.ax1.get_xlim()
        assert np.isclose(x0_orig, x0_after, atol=1e-3)
        assert np.isclose(x1_orig, x1_after, atol=1e-3)

    def test_swipe_syncs_all_axes(self, gui_window):
        _reset_file_0(gui_window)
        s = gui_window.app_state

        swipe_right(s)

        ax1_lim = s.ax1.get_xlim()
        ax2_lim = s.ax2.get_xlim()
        ax3_lim = s.ax3.get_xlim()
        assert np.allclose(ax1_lim, ax2_lim, atol=1e-6)
        assert np.allclose(ax1_lim, ax3_lim, atol=1e-6)


# =====================================================================
# delete_segment
# =====================================================================
class TestDeleteSegment:
    def test_delete_removes_segment(self, gui_window):
        """Clicking within a segment range should remove it."""
        _reset_file_0(gui_window)
        s = gui_window.app_state
        dd = s.display_dict

        n_before = len(dd["onsets"])
        labels_before = dd["labels"]
        # Click in the middle of the first segment (onset=100ms, offset=200ms)
        mid_sec = (dd["onsets"][0] + dd["offsets"][0]) / 2 / 1000.0

        event = SimpleNamespace(xdata=mid_sec, inaxes=s.ax3, button=1)
        delete_segment(event, s)

        assert len(dd["onsets"]) == n_before - 1
        assert len(dd["offsets"]) == n_before - 1
        assert len(dd["labels"]) == len(labels_before) - 1

    def test_delete_saves_notmat(self, gui_window):
        """After deleting, the .not.mat file should reflect the change."""
        _reset_file_0(gui_window)
        s = gui_window.app_state
        dd = s.display_dict

        notmat_path = os.path.join(s.data_dir, f"{dd['file_name']}.not.mat")
        n_before = len(dd["onsets"])
        mid_sec = (dd["onsets"][0] + dd["offsets"][0]) / 2 / 1000.0

        event = SimpleNamespace(xdata=mid_sec, inaxes=s.ax3, button=1)
        delete_segment(event, s)

        saved = loadmat(notmat_path)
        assert saved["onsets"].shape[0] == n_before - 1

    def test_delete_outside_segment_is_noop(self, gui_window):
        """Click outside any segment should not change anything."""
        _reset_file_0(gui_window)
        s = gui_window.app_state
        dd = s.display_dict

        n_before = len(dd["onsets"])
        # Click at x=0 (before any segment)
        event = SimpleNamespace(xdata=0.001, inaxes=s.ax3, button=1)
        delete_segment(event, s)

        assert len(dd["onsets"]) == n_before


# =====================================================================
# add_new_segment
# =====================================================================
class TestAddNewSegment:
    def test_add_segment_creates_new_entry(self, gui_window):
        """Left-click sets onset, right-click sets offset; new segment appears."""
        _reset_file_0(gui_window)
        s = gui_window.app_state
        dd = s.display_dict
        n_before = len(dd["onsets"])

        # Place new segment in a gap – after the last offset
        last_off_sec = dd["offsets"][-1] / 1000.0
        new_onset_sec = last_off_sec + 0.01
        new_offset_sec = last_off_sec + 0.03

        # Left-click → set onset
        evt_left = SimpleNamespace(xdata=new_onset_sec, inaxes=s.ax1, button=1)
        add_new_segment(evt_left, s)
        assert s.new_onset is not None

        # Right-click → set offset and create
        evt_right = SimpleNamespace(xdata=new_offset_sec, inaxes=s.ax1, button=3)
        add_new_segment(evt_right, s)

        assert len(dd["onsets"]) == n_before + 1
        assert len(dd["offsets"]) == n_before + 1
        assert len(dd["labels"]) == n_before + 1
        assert "x" in dd["labels"]  # new segments get label "x"

    def test_add_segment_saves_notmat(self, gui_window):
        """After adding, the .not.mat file should reflect the change."""
        _reset_file_0(gui_window)
        s = gui_window.app_state
        dd = s.display_dict
        n_before = len(dd["onsets"])

        last_off_sec = dd["offsets"][-1] / 1000.0
        new_onset_sec = last_off_sec + 0.01
        new_offset_sec = last_off_sec + 0.03

        evt_left = SimpleNamespace(xdata=new_onset_sec, inaxes=s.ax1, button=1)
        add_new_segment(evt_left, s)
        evt_right = SimpleNamespace(xdata=new_offset_sec, inaxes=s.ax1, button=3)
        add_new_segment(evt_right, s)

        notmat_path = os.path.join(s.data_dir, f"{dd['file_name']}.not.mat")
        saved = loadmat(notmat_path)
        assert saved["onsets"].shape[0] == n_before + 1

    def test_add_overlapping_segment_rejected(self, gui_window):
        """A segment overlapping an existing one should not be created."""
        _reset_file_0(gui_window)
        s = gui_window.app_state
        dd = s.display_dict
        n_before = len(dd["onsets"])

        # Try to create a segment that overlaps the first one
        overlap_onset_sec = dd["onsets"][0] / 1000.0 - 0.01
        overlap_offset_sec = dd["offsets"][0] / 1000.0 + 0.01

        evt_left = SimpleNamespace(xdata=overlap_onset_sec, inaxes=s.ax1, button=1)
        add_new_segment(evt_left, s)
        evt_right = SimpleNamespace(xdata=overlap_offset_sec, inaxes=s.ax1, button=3)
        add_new_segment(evt_right, s)

        assert len(dd["onsets"]) == n_before  # no change

    def test_add_reversed_onset_offset_rejected(self, gui_window):
        """onset >= offset should be rejected."""
        _reset_file_0(gui_window)
        s = gui_window.app_state
        dd = s.display_dict
        n_before = len(dd["onsets"])

        last_off_sec = dd["offsets"][-1] / 1000.0
        # Set onset AFTER offset
        evt_left = SimpleNamespace(xdata=last_off_sec + 0.05, inaxes=s.ax1, button=1)
        add_new_segment(evt_left, s)
        evt_right = SimpleNamespace(xdata=last_off_sec + 0.02, inaxes=s.ax1, button=3)
        add_new_segment(evt_right, s)

        assert len(dd["onsets"]) == n_before
