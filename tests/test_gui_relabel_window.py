"""GUI tests for the Relabel dialog (window_utils.open_relabel_window).

Tests verify the dialog opens correctly with scope radio buttons,
model combobox, overwrite checkbox, and action button.
"""
import pytest
from PyQt6.QtWidgets import (
    QDialog, QRadioButton, QComboBox, QPushButton,
    QCheckBox, QProgressBar, QLabel,
)

from moove.utils.window_utils import open_relabel_window


@pytest.fixture()
def relabel_dlg(gui_window, qtbot):
    """Open the relabel dialog and yield it; close after the test."""
    open_relabel_window(gui_window, gui_window.app_state)
    dlg = gui_window.app_state.relabel_window
    qtbot.addWidget(dlg)
    yield dlg
    dlg.close()
    gui_window.app_state.relabel_window = None


class TestRelabelWindowOpens:
    def test_dialog_is_visible(self, relabel_dlg):
        assert relabel_dlg.isVisible()

    def test_dialog_title(self, relabel_dlg):
        assert relabel_dlg.windowTitle() == "Relabel"

    def test_dialog_is_qdialog(self, relabel_dlg):
        assert isinstance(relabel_dlg, QDialog)


class TestRelabelWidgets:
    def test_classification_header(self, relabel_dlg):
        labels = [lbl.text() for lbl in relabel_dlg.findChildren(QLabel)]
        assert any("Classification" in l for l in labels)

    def test_scope_radios_present(self, relabel_dlg):
        radios = [rb.text() for rb in relabel_dlg.findChildren(QRadioButton)]
        for label in ("Current File", "Current Day", "Current Experiment", "Current Bird"):
            assert label in radios

    def test_current_file_selected_by_default(self, relabel_dlg):
        radios = relabel_dlg.findChildren(QRadioButton)
        checked = [rb for rb in radios if rb.isChecked()]
        assert any(rb.text() == "Current File" for rb in checked)

    def test_model_combobox_present(self, relabel_dlg):
        combos = relabel_dlg.findChildren(QComboBox)
        model_combos = [c for c in combos
                        if c.itemText(0) == "Select Trained Classification Model"]
        assert len(model_combos) == 1

    def test_overwrite_checkbox(self, relabel_dlg):
        cbs = [cb for cb in relabel_dlg.findChildren(QCheckBox)
               if "Overwrite" in cb.text()]
        assert len(cbs) == 1

    def test_relabel_button(self, relabel_dlg):
        buttons = [b for b in relabel_dlg.findChildren(QPushButton)
                   if b.text() == "Relabel"]
        assert len(buttons) == 1

    def test_batch_combobox_present(self, relabel_dlg):
        combos = relabel_dlg.findChildren(QComboBox)
        # After update_batch_select_combobox_relabel, first item is "All Files"
        batch_combos = [c for c in combos
                        if c.count() > 0 and c.itemText(0) == "All Files"]
        assert len(batch_combos) == 1


class TestRelabelStatusWidgets:
    def test_status_label_hidden(self, relabel_dlg):
        assert hasattr(relabel_dlg, 'status_label')
        assert not relabel_dlg.status_label.isVisible()

    def test_progressbar_hidden(self, relabel_dlg):
        assert hasattr(relabel_dlg, 'progressbar')
        assert not relabel_dlg.progressbar.isVisible()
