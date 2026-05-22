"""GUI tests for the Training dialog (window_utils.open_training_window).

Tests verify the dialog opens correctly with both panels (Segmentation
and Classification), dataset creation widgets, training parameter entries,
and action buttons.
"""
import pytest
from PyQt6.QtWidgets import (
    QDialog, QRadioButton, QLineEdit, QComboBox, QPushButton,
    QCheckBox, QProgressBar, QLabel,
)

from moove.utils.window_utils import open_training_window


@pytest.fixture()
def training_dlg(gui_window, qtbot):
    """Open the training dialog and yield it; close after the test."""
    open_training_window(gui_window, gui_window.app_state)
    dlg = gui_window.app_state.training_window
    qtbot.addWidget(dlg)
    yield dlg
    dlg.close()
    gui_window.app_state.training_window = None


class TestTrainingWindowOpens:
    def test_dialog_is_visible(self, training_dlg):
        assert training_dlg.isVisible()

    def test_dialog_title(self, training_dlg):
        assert training_dlg.windowTitle() == "Training"

    def test_dialog_is_qdialog(self, training_dlg):
        assert isinstance(training_dlg, QDialog)


class TestTrainingSegmentationPanel:
    """Left panel – segmentation training."""

    def test_segmentation_header(self, training_dlg):
        labels = [lbl.text() for lbl in training_dlg.findChildren(QLabel)]
        assert any("Segmentation" in l for l in labels)

    def test_scope_radios_present(self, training_dlg):
        radios = [rb.text() for rb in training_dlg.findChildren(QRadioButton)]
        for label in ("Current Day", "Current Experiment", "Current Bird"):
            assert label in radios

    def test_use_segmented_only_checkbox(self, training_dlg):
        cbs = [cb for cb in training_dlg.findChildren(QCheckBox)
               if "segmented" in cb.text().lower()]
        assert len(cbs) >= 1

    def test_dataset_name_entry(self, training_dlg):
        entries = training_dlg.findChildren(QLineEdit)
        entry_texts = [e.text() for e in entries]
        assert "edit_seg_dataset_name" in entry_texts

    def test_chunk_size_entry_prefilled(self, training_dlg, gui_window):
        s = gui_window.app_state
        entries = training_dlg.findChildren(QLineEdit)
        entry_texts = [e.text() for e in entries]
        assert s.train_segmentation_params['chunk_size'].get() in entry_texts

    def test_create_dataset_button(self, training_dlg):
        buttons = [b for b in training_dlg.findChildren(QPushButton)
                   if b.text() == "Create Training Dataset"]
        assert len(buttons) >= 1

    def test_start_training_button(self, training_dlg):
        buttons = [b for b in training_dlg.findChildren(QPushButton)
                   if b.text() == "Start Training"]
        assert len(buttons) >= 1

    def test_seg_training_params_labels(self, training_dlg):
        labels = [lbl.text() for lbl in training_dlg.findChildren(QLabel)]
        for param_label in ("Epochs:", "Batch Size:", "Learning Rate:",
                            "Early Stopping Patience:"):
            assert param_label in labels

    def test_seg_dataset_combobox(self, training_dlg):
        """The segmentation dataset combobox attribute should exist on the dialog."""
        assert hasattr(training_dlg, 'training_dataset_combobox_segmentation')
        assert isinstance(training_dlg.training_dataset_combobox_segmentation, QComboBox)

    def test_downsampling_radio(self, training_dlg):
        rbs = [rb for rb in training_dlg.findChildren(QRadioButton)
               if rb.text() == "Downsampling"]
        assert len(rbs) >= 1

    def test_overlap_chunks_checkbox(self, training_dlg):
        cbs = [cb for cb in training_dlg.findChildren(QCheckBox)
               if "overlap" in cb.text().lower()]
        assert len(cbs) == 1


class TestTrainingClassificationPanel:
    """Right panel – classification training."""

    def test_classification_header(self, training_dlg):
        labels = [lbl.text() for lbl in training_dlg.findChildren(QLabel)]
        assert any("Classification" in l for l in labels)

    def test_use_classified_only_checkbox(self, training_dlg):
        cbs = [cb for cb in training_dlg.findChildren(QCheckBox)
               if "classified" in cb.text().lower()]
        assert len(cbs) >= 1

    def test_cls_dataset_name_entry(self, training_dlg):
        entries = training_dlg.findChildren(QLineEdit)
        entry_texts = [e.text() for e in entries]
        assert "edit_class_dataset_name" in entry_texts

    def test_spec_param_entries(self, training_dlg):
        labels = [lbl.text() for lbl in training_dlg.findChildren(QLabel)]
        for param_label in ("Nperseg:", "Noverlap:", "NFFT:", "Frequency Cutoffs:"):
            assert param_label in labels

    def test_two_create_dataset_buttons(self, training_dlg):
        """Both panels have a Create Training Dataset button."""
        buttons = [b for b in training_dlg.findChildren(QPushButton)
                   if b.text() == "Create Training Dataset"]
        assert len(buttons) == 2

    def test_two_start_training_buttons(self, training_dlg):
        """Both panels have a Start Training button."""
        buttons = [b for b in training_dlg.findChildren(QPushButton)
                   if b.text() == "Start Training"]
        assert len(buttons) == 2

    def test_augmentation_button(self, training_dlg):
        """Classification panel has an Augmentation... button."""
        buttons = [b for b in training_dlg.findChildren(QPushButton)
                   if b.text() == "Augmentation..."]
        assert len(buttons) == 1


class TestAugmentationDialog:
    """Tests for the augmentation settings dialog interaction."""

    def test_augmentation_dialog_opens_and_checkbox_toggles(self, training_dlg, gui_window, qtbot):
        """Open augmentation dialog, toggle the enable checkbox, click Cancel."""
        from PyQt6.QtCore import QTimer

        btn = [b for b in training_dlg.findChildren(QPushButton)
               if b.text() == "Augmentation..."][0]

        def interact_with_dialog():
            # Find the augmentation dialog (child of training_dlg)
            dialogs = [w for w in training_dlg.findChildren(QDialog)]
            assert len(dialogs) >= 1, "Augmentation dialog not found"
            adlg = dialogs[-1]
            assert adlg.windowTitle() == "Data Augmentation Settings"

            # Find the Enable checkbox
            cbs = [cb for cb in adlg.findChildren(QCheckBox)
                   if "enable" in cb.text().lower() or "augmentation" in cb.text().lower()]
            assert len(cbs) == 1
            enable_cb = cbs[0]

            # Toggle checkbox
            original = enable_cb.isChecked()
            enable_cb.setChecked(not original)
            assert enable_cb.isChecked() != original

            # Toggle back
            enable_cb.setChecked(original)
            assert enable_cb.isChecked() == original

            # Close via Cancel
            cancel_btns = [b for b in adlg.findChildren(QPushButton)
                           if b.text() == "Cancel"]
            assert len(cancel_btns) == 1
            cancel_btns[0].click()

        QTimer.singleShot(200, interact_with_dialog)
        btn.click()

    def test_augmentation_dialog_apply_saves_params(self, training_dlg, gui_window, qtbot):
        """Open dialog, change params, click OK → params saved to app_state."""
        from PyQt6.QtCore import QTimer

        btn = [b for b in training_dlg.findChildren(QPushButton)
               if b.text() == "Augmentation..."][0]
        app_state = gui_window.app_state

        def interact_with_dialog():
            dialogs = [w for w in training_dlg.findChildren(QDialog)]
            adlg = dialogs[-1]

            # Uncheck Enable
            cbs = [cb for cb in adlg.findChildren(QCheckBox)]
            enable_cb = cbs[0]
            enable_cb.setChecked(False)

            # Click OK
            ok_btns = [b for b in adlg.findChildren(QPushButton)
                       if b.text() == "OK"]
            ok_btns[0].click()

        QTimer.singleShot(200, interact_with_dialog)
        btn.click()

        # After OK, the app_state should reflect the change
        assert app_state.augmentation_params['enabled'].get() is False

        # Restore default
        app_state.augmentation_params['enabled'].set(True)


class TestTrainingStatusWidgets:
    def test_status_label_hidden(self, training_dlg):
        assert hasattr(training_dlg, 'status_label')
        assert not training_dlg.status_label.isVisible()

    def test_progressbar_hidden(self, training_dlg):
        assert hasattr(training_dlg, 'progressbar')
        assert not training_dlg.progressbar.isVisible()
