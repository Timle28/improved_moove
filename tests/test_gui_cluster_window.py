"""GUI tests for the Cluster dialog (window_utils.open_cluster_window).

Tests verify the dialog opens correctly with dataset creation widgets,
UMAP parameter entries, cluster/Dash/replace buttons.
"""
import pytest
from PyQt6.QtWidgets import (
    QDialog, QRadioButton, QLineEdit, QComboBox, QPushButton,
    QCheckBox, QLabel,
)

from moove.utils.window_utils import open_cluster_window


@pytest.fixture()
def cluster_dlg(gui_window, qtbot):
    """Open the cluster dialog and yield it; close after the test."""
    open_cluster_window(gui_window, gui_window.app_state)
    dlg = gui_window.app_state.cluster_window
    qtbot.addWidget(dlg)
    yield dlg
    dlg.close()
    gui_window.app_state.cluster_window = None


class TestClusterWindowOpens:
    def test_dialog_is_visible(self, cluster_dlg):
        assert cluster_dlg.isVisible()

    def test_dialog_title(self, cluster_dlg):
        assert cluster_dlg.windowTitle() == "Cluster"

    def test_dialog_is_qdialog(self, cluster_dlg):
        assert isinstance(cluster_dlg, QDialog)


class TestClusterDatasetCreation:
    def test_scope_radios_present(self, cluster_dlg):
        radios = [rb.text() for rb in cluster_dlg.findChildren(QRadioButton)]
        for label in ("Current Day", "Current Experiment", "Current Bird"):
            assert label in radios

    def test_use_segmented_checkbox(self, cluster_dlg):
        cbs = [cb for cb in cluster_dlg.findChildren(QCheckBox)
               if "segmented" in cb.text().lower()]
        assert len(cbs) == 1

    def test_dataset_name_entry(self, cluster_dlg):
        entries = cluster_dlg.findChildren(QLineEdit)
        entry_texts = [e.text() for e in entries]
        assert "edit_cluster_dataset_name" in entry_texts

    def test_spec_param_entries(self, cluster_dlg):
        labels = [lbl.text() for lbl in cluster_dlg.findChildren(QLabel)]
        for param_label in ("Nperseg:", "Noverlap:", "NFFT:", "Frequency Cutoffs:"):
            assert param_label in labels

    def test_create_cluster_dataset_button(self, cluster_dlg):
        buttons = [b for b in cluster_dlg.findChildren(QPushButton)
                   if b.text() == "Create Cluster Dataset"]
        assert len(buttons) == 1


class TestClusterUMAPParams:
    def test_umap_param_labels(self, cluster_dlg):
        labels = [lbl.text() for lbl in cluster_dlg.findChildren(QLabel)]
        for param_label in ("N_neighbors:", "Min_dist:", "N Syllables:"):
            assert param_label in labels

    def test_umap_params_prefilled(self, cluster_dlg, gui_window):
        s = gui_window.app_state
        entries = cluster_dlg.findChildren(QLineEdit)
        entry_texts = [e.text() for e in entries]
        assert s.umap_k_means_params['n_neighbors'].get() in entry_texts
        assert s.umap_k_means_params['min_dist'].get() in entry_texts
        assert s.umap_k_means_params['n_clusters'].get() in entry_texts

    def test_cluster_dataset_combobox(self, cluster_dlg):
        """The cluster dataset combobox attribute should exist on the dialog."""
        assert hasattr(cluster_dlg, 'cluster_dataset_combobox')
        assert isinstance(cluster_dlg.cluster_dataset_combobox, QComboBox)


class TestClusterActionButtons:
    def test_cluster_syllables_button(self, cluster_dlg):
        buttons = [b for b in cluster_dlg.findChildren(QPushButton)
                   if b.text() == "Cluster Syllables"]
        assert len(buttons) == 1

    def test_open_dash_button(self, cluster_dlg):
        buttons = [b for b in cluster_dlg.findChildren(QPushButton)
                   if b.text() == "Open Dash GUI"]
        assert len(buttons) == 1

    def test_close_dash_button(self, cluster_dlg):
        buttons = [b for b in cluster_dlg.findChildren(QPushButton)
                   if b.text() == "Close Dash GUI"]
        assert len(buttons) == 1

    def test_replace_labels_button(self, cluster_dlg):
        buttons = [b for b in cluster_dlg.findChildren(QPushButton)
                   if b.text() == "Replace Labels"]
        assert len(buttons) == 1
