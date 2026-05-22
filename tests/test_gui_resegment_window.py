"""GUI tests for the Resegment dialog (window_utils.open_resegment_window).

Tests verify the dialog opens correctly and contains the expected widgets
(Evfuncs panel + Segmentation Network panel, radio buttons, parameter
entries, comboboxes, status/progress).
"""
import pytest
from PyQt6.QtWidgets import (
    QDialog, QRadioButton, QLineEdit, QComboBox, QPushButton,
    QCheckBox, QProgressBar, QLabel, QButtonGroup,
)

from moove.utils.window_utils import open_resegment_window


@pytest.fixture()
def resegment_dlg(gui_window, qtbot):
    """Open the resegment dialog and yield it; close after the test."""
    open_resegment_window(gui_window, gui_window.app_state)
    dlg = gui_window.app_state.resegment_window
    qtbot.addWidget(dlg)
    yield dlg
    dlg.close()
    gui_window.app_state.resegment_window = None


class TestResegmentWindowOpens:
    def test_dialog_is_visible(self, resegment_dlg):
        assert resegment_dlg.isVisible()

    def test_dialog_title(self, resegment_dlg):
        assert resegment_dlg.windowTitle() == "Resegmentation"

    def test_dialog_is_qdialog(self, resegment_dlg):
        assert isinstance(resegment_dlg, QDialog)


class TestResegmentEvfuncsPanel:
    """Left panel – Evfuncs parameter widgets."""

    def test_evfuncs_scope_radios_present(self, resegment_dlg):
        radios = [rb.text() for rb in resegment_dlg.findChildren(QRadioButton)]
        for label in ("Current File", "Current Day", "Current Experiment", "Current Bird"):
            assert radios.count(label) >= 1

    def test_evfuncs_param_entries_present(self, resegment_dlg):
        labels = [lbl.text() for lbl in resegment_dlg.findChildren(QLabel)]
        for param_label in ("Threshold:", "Min Syllable Length:", "Min Silent Duration:",
                            "Frequency Cutoffs:", "Smoothing Window:"):
            assert param_label in labels

    def test_evfuncs_param_entries_prefilled(self, resegment_dlg, gui_window):
        """Entries should be pre-filled from app_state.evfuncs_params."""
        entries = resegment_dlg.findChildren(QLineEdit)
        entry_texts = [e.text() for e in entries]
        s = gui_window.app_state
        assert s.evfuncs_params['threshold'].get() in entry_texts
        assert s.evfuncs_params['min_syl_dur'].get() in entry_texts

    def test_evfuncs_segment_button(self, resegment_dlg):
        buttons = [b for b in resegment_dlg.findChildren(QPushButton)
                   if b.text() == "Segment"]
        assert len(buttons) >= 1


class TestResegmentMLPanel:
    """Right panel – ML segmentation parameter widgets."""

    def test_model_combobox_present(self, resegment_dlg):
        combos = resegment_dlg.findChildren(QComboBox)
        model_combos = [c for c in combos
                        if c.itemText(0) == "Select Trained Segmentation Model"]
        assert len(model_combos) == 1

    def test_overwrite_checkbox(self, resegment_dlg):
        cbs = [cb for cb in resegment_dlg.findChildren(QCheckBox)
               if "Overwrite" in cb.text()]
        assert len(cbs) == 1

    def test_ml_param_entries(self, resegment_dlg):
        labels = [lbl.text() for lbl in resegment_dlg.findChildren(QLabel)]
        for param_label in ("Decision Threshold:", "Onset Window Size:",
                            "Min Syllable Length:", "Min Silent Duration:"):
            assert param_label in labels

    def test_ml_segment_button(self, resegment_dlg):
        buttons = [b for b in resegment_dlg.findChildren(QPushButton)
                   if b.text() == "Segment"]
        # Two "Segment" buttons: evfuncs + ML
        assert len(buttons) == 2


class TestResegmentStatusWidgets:
    def test_status_label_hidden(self, resegment_dlg):
        assert hasattr(resegment_dlg, 'status_label')
        assert not resegment_dlg.status_label.isVisible()

    def test_progressbar_hidden(self, resegment_dlg):
        assert hasattr(resegment_dlg, 'progressbar')
        assert not resegment_dlg.progressbar.isVisible()
