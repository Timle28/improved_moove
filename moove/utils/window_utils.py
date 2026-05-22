# utils/window_utils.py – PyQt6 dialog windows
import os
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QLineEdit,
    QComboBox, QPushButton, QCheckBox, QRadioButton, QButtonGroup,
    QProgressBar, QWidget, QSizePolicy, QMessageBox,
)
from PyQt6.QtCore import Qt

from moove.app_state import Var
from moove.qt_helpers import show_info


def _btn(text, callback=None):
    """Create a QPushButton with autoDefault disabled (avoids blue highlight on macOS)."""
    b = QPushButton(text)
    b.setAutoDefault(False)
    b.setDefault(False)
    if callback:
        b.clicked.connect(callback)
    return b


def _set_dlg_icon(dlg):
    """Copy the application-level icon onto a dialog so child message boxes inherit it."""
    app = dlg.parent()
    while app is not None and hasattr(app, 'windowIcon'):
        icon = app.windowIcon()
        if not icon.isNull():
            dlg.setWindowIcon(icon)
            return
        app = app.parent() if hasattr(app, 'parent') else None
    from PyQt6.QtWidgets import QApplication
    app_icon = QApplication.instance().windowIcon()
    if not app_icon.isNull():
        dlg.setWindowIcon(app_icon)


def _get_bird_exp_day(app_state):
    """Read current bird/experiment/day from the stored comboboxes."""
    b = app_state.bird_combobox.currentText() if app_state.bird_combobox else ""
    e = app_state.experiment_combobox.currentText() if app_state.experiment_combobox else ""
    d = app_state.day_combobox.currentText() if app_state.day_combobox else ""
    return b, e, d


# ======================================================================
# Resegment Dialog
# ======================================================================
def open_resegment_window(parent, app_state):
    from moove.utils import start_segment_evfuncs, start_segment_files_thread, plot_data

    dlg = QDialog(parent)
    dlg.setWindowTitle("Resegmentation")
    dlg.resize(700, 450)
    _set_dlg_icon(dlg)
    app_state.resegment_window = dlg
    dlg._task_running = False
    dlg._task_cancel_requested = False

    def _resegment_close_event(event):
        if getattr(dlg, '_task_running', False):
            reply = QMessageBox.question(
                dlg,
                "Resegmentation is still running",
                "A resegmentation job is still running. Abort it and close this window?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                dlg._task_cancel_requested = True
                event.accept()
                return
            event.ignore()
            return
        event.accept()

    dlg.closeEvent = _resegment_close_event

    root = QVBoxLayout(dlg)
    root.setContentsMargins(8, 8, 8, 8)
    panels = QHBoxLayout()
    panels.setAlignment(Qt.AlignmentFlag.AlignTop)
    root.addLayout(panels, stretch=1)

    # --- Left: Evfuncs ---
    left = QGridLayout()
    left_w = QWidget()
    left_w.setLayout(left)
    left_w.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
    panels.addWidget(left_w, stretch=1, alignment=Qt.AlignmentFlag.AlignTop)

    row = 0
    left.addWidget(QLabel("<b style='font-size:16px'>Evfuncs</b>"), row, 0, 1, 2, Qt.AlignmentFlag.AlignCenter)
    row += 1

    ev_sel = Var("current_file")
    ev_group = QButtonGroup(dlg)
    for txt, val in [("Current File", "current_file"), ("Current Day", "current_day"),
                     ("Current Experiment", "current_experiment"), ("Current Bird", "current_bird")]:
        rb = QRadioButton(txt)
        ev_group.addButton(rb)
        if val == "current_file":
            rb.setChecked(True)

        def _make_cb(v=val):
            return lambda: (ev_sel.set(v), app_state.update_batch_select_combobox_resegment_ev(v))

        rb.toggled.connect(lambda checked, cb=_make_cb(): cb() if checked else None)
        left.addWidget(rb, row, 0, 1, 2)
        row += 1

    ev_batch_combo = QComboBox()
    ev_batch_combo.addItem("Select Batch File")
    dlg.resegment_batch_combobox_ev = ev_batch_combo
    app_state.update_batch_select_combobox_resegment_ev(ev_sel.get())
    left.addWidget(ev_batch_combo, 2, 1)

    evfuncs_params = [("Threshold:", 'threshold'), ("Min Syllable Length:", 'min_syl_dur'),
                      ("Min Silent Duration:", 'min_silent_dur'), ("Frequency Cutoffs:", 'freq_cutoffs'),
                      ("Smoothing Window:", 'smooth_window')]
    ev_entries = {}
    for label_text, key in evfuncs_params:
        left.addWidget(QLabel(label_text), row, 0)
        entry = QLineEdit(app_state.evfuncs_params[key].get())
        ev_entries[key] = entry
        left.addWidget(entry, row, 1)
        row += 1

    def _do_ev_segment():
        if getattr(dlg, '_task_running', False):
            show_info(dlg, "Info", "A resegmentation job is already running.")
            return
        for k, e in ev_entries.items():
            app_state.evfuncs_params[k].set(e.text())
        b, e, d = _get_bird_exp_day(app_state)
        start_segment_evfuncs(app_state, ev_sel.get(), ev_batch_combo.currentText(), b, e, d)

    btn_ev = _btn("Segment", _do_ev_segment)
    left.addWidget(btn_ev, row, 0, 1, 2)
    row += 1
    left.addWidget(QLabel("<b style='font-size:16px'></b>"), row, 0, 1, 2, Qt.AlignmentFlag.AlignCenter)
    row += 1
    left.addWidget(QLabel("<b style='font-size:16px'>Set Manual Threshold</b>"), row, 0, 1, 2,
                   Qt.AlignmentFlag.AlignCenter)
    row += 1
    left.addWidget(QLabel("New Threshold"), row, 0)
    entry_thres = QLineEdit(app_state.evfuncs_params['threshold'].get())
    left.addWidget(entry_thres, row, 1)
    row += 1

    def _do_get_threshold():
        app_state.evfuncs_params['threshold'].set(entry_thres.text())
        plot_data(app_state)

    btn_ev = _btn("Set Threshold", _do_get_threshold)
    left.addWidget(btn_ev, row, 0, 1, 2)

    # --- Right: Segmentation Network ---
    right = QGridLayout()
    right_w = QWidget()
    right_w.setLayout(right)
    right_w.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
    panels.addWidget(right_w, stretch=1, alignment=Qt.AlignmentFlag.AlignTop)

    row = 0
    right.addWidget(QLabel("<b style='font-size:16px'>Segmentation Network</b>"), row, 0, 1, 2,
                    Qt.AlignmentFlag.AlignCenter)
    row += 1

    sm_sel = Var("current_file")
    sm_group = QButtonGroup(dlg)
    for txt, val in [("Current File", "current_file"), ("Current Day", "current_day"),
                     ("Current Experiment", "current_experiment"), ("Current Bird", "current_bird")]:
        rb = QRadioButton(txt)
        sm_group.addButton(rb)
        if val == "current_file":
            rb.setChecked(True)

        def _make_cb(v=val):
            return lambda: (sm_sel.set(v), app_state.update_batch_select_combobox_resegment(v))

        rb.toggled.connect(lambda checked, cb=_make_cb(): cb() if checked else None)
        right.addWidget(rb, row, 0, 1, 2)
        row += 1

    sm_batch_combo = QComboBox()
    sm_batch_combo.addItem("Select Batch File")
    dlg.resegment_batch_combobox = sm_batch_combo
    app_state.update_batch_select_combobox_resegment(sm_sel.get())
    right.addWidget(sm_batch_combo, 2, 1)

    overwrite_cb = QCheckBox("Overwrite Already Segmented Files")
    right.addWidget(overwrite_cb, row, 0, 1, 2)
    row += 1

    trained_models_path = os.path.join(app_state.config["global_dir"], "trained_models")
    model_files = [f[:-4] for f in os.listdir(trained_models_path) if f.endswith("_seg_model.pth")]
    model_combo = QComboBox()
    model_combo.addItem("Select Trained Segmentation Model")
    model_combo.addItems(model_files)
    right.addWidget(model_combo, row, 0, 1, 2)
    row += 1

    params = [("Decision Threshold:", 'decision_threshold'), ("Onset Window Size:", 'onset_window_size'),
              ("N Onset True:", 'n_onset_true'), ("Offset Window Size:", 'offset_window_size'),
              ("N Offset False:", 'n_offset_false'), ("Min Syllable Length:", 'min_syllable_length'),
              ("Min Silent Duration:", 'min_silent_duration')]
    ml_entries = {}
    for label_text, key in params:
        right.addWidget(QLabel(label_text), row, 0)
        entry = QLineEdit(app_state.mlseg_params[key].get())
        ml_entries[key] = entry
        right.addWidget(entry, row, 1)
        row += 1

    # Status widgets
    dlg.status_label = QLabel("")
    dlg.status_label.setStyleSheet("color: green; font-size: 14px;")
    dlg.status_label.hide()
    dlg.progressbar = QProgressBar()
    dlg.progressbar.hide()

    def _do_ml_segment():
        if getattr(dlg, '_task_running', False):
            show_info(dlg, "Info", "A resegmentation job is already running.")
            return
        for k, e in ml_entries.items():
            app_state.mlseg_params[k].set(e.text())
        sel_model = model_combo.currentText()
        if sel_model == "Select Trained Segmentation Model":
            sel_model = ""
        b, e, d = _get_bird_exp_day(app_state)
        start_segment_files_thread(app_state, sel_model, sm_sel.get(),
                                   overwrite_cb.isChecked(), sm_batch_combo.currentText(),
                                   b, e, d)

    btn_ml = _btn("Segment", _do_ml_segment)
    right.addWidget(btn_ml, row, 0, 1, 2)

    root.addWidget(dlg.status_label)
    root.addWidget(dlg.progressbar)

    dlg.show()


# ======================================================================
# Relabel Dialog
# ======================================================================
def open_relabel_window(parent, app_state):
    from moove.utils import start_classify_files_thread

    dlg = QDialog(parent)
    dlg.setWindowTitle("Relabel")
    dlg.setMinimumWidth(400)
    _set_dlg_icon(dlg)
    app_state.relabel_window = dlg
    dlg._task_running = False
    dlg._task_cancel_requested = False

    def _relabel_close_event(event):
        if getattr(dlg, '_task_running', False):
            reply = QMessageBox.question(
                dlg,
                "Relabeling is still running",
                "A relabeling job is still running. Abort it and close this window?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                dlg._task_cancel_requested = True
                event.accept()
                return
            event.ignore()
            return
        event.accept()

    dlg.closeEvent = _relabel_close_event

    outer = QVBoxLayout(dlg)
    outer.setContentsMargins(8, 8, 8, 8)
    grid = QGridLayout()
    outer.addLayout(grid)
    row = 0
    grid.addWidget(QLabel("<b style='font-size:16px'>Classification Network</b>"), row, 0, 1, 2,
                   Qt.AlignmentFlag.AlignCenter)
    row += 1

    sel = Var("current_file")
    grp = QButtonGroup(dlg)
    for txt, val in [("Current File", "current_file"), ("Current Day", "current_day"),
                     ("Current Experiment", "current_experiment"), ("Current Bird", "current_bird")]:
        rb = QRadioButton(txt)
        grp.addButton(rb)
        if val == "current_file":
            rb.setChecked(True)

        def _make_cb(v=val):
            return lambda: (sel.set(v), app_state.update_batch_select_combobox_relabel(v))

        rb.toggled.connect(lambda checked, cb=_make_cb(): cb() if checked else None)
        grid.addWidget(rb, row, 0, 1, 2)
        row += 1

    batch_combo = QComboBox()
    batch_combo.addItem("Select Batch File")
    dlg.relabel_batch_combobox = batch_combo
    app_state.update_batch_select_combobox_relabel(sel.get())
    grid.addWidget(batch_combo, 2, 1)

    overwrite_cb = QCheckBox("Overwrite Already Classified Files")
    grid.addWidget(overwrite_cb, row, 0, 1, 2)
    row += 1

    trained_models_path = os.path.join(app_state.config["global_dir"], "trained_models")
    model_files = [f[:-4] for f in os.listdir(trained_models_path) if f.endswith("_class_model.pth")]
    model_combo = QComboBox()
    model_combo.addItem("Select Trained Classification Model")
    model_combo.addItems(model_files)
    grid.addWidget(model_combo, row, 0, 1, 2)
    row += 1

    dlg.status_label = QLabel("")
    dlg.status_label.setStyleSheet("color: green; font-size: 14px;")
    dlg.status_label.hide()
    dlg.progressbar = QProgressBar()
    dlg.progressbar.hide()

    def _do_relabel():
        if getattr(dlg, '_task_running', False):
            show_info(dlg, "Info", "A relabeling job is already running.")
            return
        sel_model = model_combo.currentText()
        if sel_model == "Select Trained Classification Model":
            sel_model = ""
        b, e, d = _get_bird_exp_day(app_state)
        start_classify_files_thread(app_state, sel_model, sel.get(),
                                    overwrite_cb.isChecked(), batch_combo.currentText(),
                                    b, e, d)

    btn = _btn("Relabel", _do_relabel)
    grid.addWidget(btn, row, 0, 1, 2)

    outer.addWidget(dlg.status_label)
    outer.addWidget(dlg.progressbar)

    dlg.adjustSize()
    dlg.setFixedHeight(dlg.sizeHint().height())
    dlg.show()


# ======================================================================
# Training Dialog
# ======================================================================
def open_training_window(parent, app_state):
    from moove.utils import (
        start_create_segmentation_training_dataset, start_segmentation_training,
        start_create_classification_training_dataset, start_classification_training,
    )

    dlg = QDialog(parent)
    dlg.setWindowTitle("Training")
    dlg.resize(700, 560)
    _set_dlg_icon(dlg)
    app_state.training_window = dlg
    dlg._training_running = False
    dlg._training_cancel_requested = False
    dlg._task_running = False
    dlg._task_cancel_requested = False

    def _training_close_event(event):
        if getattr(dlg, '_training_running', False) or getattr(dlg, '_task_running', False):
            reply = QMessageBox.question(
                dlg,
                "An operation is still running",
                "A training operation is still running. Abort it and close this window?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                dlg._training_cancel_requested = True
                dlg._task_cancel_requested = True
                event.accept()
                return
            event.ignore()
            return
        event.accept()

    dlg.closeEvent = _training_close_event

    root = QVBoxLayout(dlg)
    root.setContentsMargins(8, 8, 8, 8)
    panels = QHBoxLayout()
    panels.setAlignment(Qt.AlignmentFlag.AlignTop)
    root.addLayout(panels, stretch=1)

    # ---- Left: Segmentation ----
    left = QGridLayout()
    left_w = QWidget()
    left_w.setLayout(left)
    left_w.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
    panels.addWidget(left_w, stretch=1, alignment=Qt.AlignmentFlag.AlignTop)

    row = 0
    left.addWidget(QLabel("<b style='font-size:16px'>Segmentation Network</b>"), row, 0, 1, 2,
                   Qt.AlignmentFlag.AlignCenter)
    row += 1

    use_seg_only = QCheckBox("Use segmented files only")
    left.addWidget(use_seg_only, row, 0, 1, 2)
    row += 1

    seg_sel = Var("current_day")
    seg_grp = QButtonGroup(dlg)
    for txt, val in [("Current Day", "current_day"), ("Current Experiment", "current_experiment"),
                     ("Current Bird", "current_bird")]:
        rb = QRadioButton(txt)
        seg_grp.addButton(rb)
        if val == "current_day":
            rb.setChecked(True)

        def _mc(v=val):
            return lambda: (seg_sel.set(v), app_state.update_batch_select_combobox_segment(v))

        rb.toggled.connect(lambda chk, cb=_mc(): cb() if chk else None)
        left.addWidget(rb, row, 0, 1, 2)
        row += 1

    seg_batch = QComboBox()
    seg_batch.addItem("Select Batch File")
    dlg.training_batch_combobox_segmentation = seg_batch
    app_state.update_batch_select_combobox_segment(seg_sel.get())
    left.addWidget(seg_batch, 3, 1)

    left.addWidget(QLabel("Training Dataset Name:"), row, 0)
    seg_ds_name = QLineEdit("edit_seg_dataset_name")
    left.addWidget(seg_ds_name, row, 1)
    row += 1

    left.addWidget(QLabel("Chunk Size:"), row, 0)
    seg_chunk = QLineEdit(app_state.train_segmentation_params['chunk_size'].get())
    left.addWidget(seg_chunk, row, 1)
    row += 1

    left.addWidget(QLabel("Hist Size:"), row, 0)
    seg_hist = QLineEdit(app_state.train_segmentation_params['hist_size'].get())
    left.addWidget(seg_hist, row, 1)
    row += 1

    seg_overlap = QCheckBox("Overlap chunks")
    seg_overlap.setChecked(app_state.train_segmentation_params['overlap_chunks'].get())
    left.addWidget(seg_overlap, row, 0, 1, 2)
    row += 1

    def _create_seg_ds():
        if getattr(dlg, '_training_running', False) or getattr(dlg, '_task_running', False):
            show_info(dlg, "Info", "A training operation is already running.")
            return
        app_state.train_segmentation_params['chunk_size'].set(seg_chunk.text())
        app_state.train_segmentation_params['hist_size'].set(seg_hist.text())
        app_state.train_segmentation_params['overlap_chunks'].set(seg_overlap.isChecked())
        b, e, d = _get_bird_exp_day(app_state)
        start_create_segmentation_training_dataset(
            app_state, seg_ds_name.text(), use_seg_only.isChecked(),
            seg_sel.get(), seg_batch.currentText(), b, e, d, dlg)

    btn_create_seg = _btn("Create Training Dataset", _create_seg_ds)
    left.addWidget(btn_create_seg, row, 0, 1, 2)
    row += 1

    left.addWidget(QLabel(""), row, 0)
    row += 1

    seg_ds_combo = QComboBox()
    seg_ds_combo.addItem("Select Training Dataset")
    dlg.training_dataset_combobox_segmentation = seg_ds_combo
    app_state.update_segmentation_datasets_combobox()
    left.addWidget(seg_ds_combo, row, 0, 1, 2)
    row += 1

    seg_imbalance_grp = QButtonGroup(dlg)
    seg_rb_none = QRadioButton("None")
    seg_rb_down = QRadioButton("Downsampling")
    seg_rb_weighted = QRadioButton("Weighted BCE")
    for rb in [seg_rb_none, seg_rb_down, seg_rb_weighted]:
        seg_imbalance_grp.addButton(rb)
    _seg_strat = app_state.train_segmentation_params['imbalance_strategy'].get()
    if _seg_strat == 'none':
        seg_rb_none.setChecked(True)
    elif _seg_strat == 'weighted_loss':
        seg_rb_weighted.setChecked(True)
    else:
        seg_rb_down.setChecked(True)
    seg_imbalance_row = QWidget()
    seg_imbalance_layout = QHBoxLayout(seg_imbalance_row)
    seg_imbalance_layout.setContentsMargins(0, 0, 0, 0)
    for rb in [seg_rb_none, seg_rb_down, seg_rb_weighted]:
        seg_imbalance_layout.addWidget(rb)
    left.addWidget(seg_imbalance_row, row, 0, 1, 2)
    row += 1

    seg_t_entries = {}
    for lbl, key in [("Epochs:", 'epochs'), ("Batch Size:", 'batch_size'),
                     ("Learning Rate:", 'learning_rate'), ("Early Stopping Patience:", 'early_stopping_patience')]:
        left.addWidget(QLabel(lbl), row, 0)
        e = QLineEdit(app_state.train_segmentation_params[key].get())
        seg_t_entries[key] = e
        left.addWidget(e, row, 1)
        row += 1

    def _train_seg():
        if getattr(dlg, '_training_running', False) or getattr(dlg, '_task_running', False):
            show_info(dlg, "Info", "A training operation is already running.")
            return
        if seg_rb_none.isChecked():
            _strat = 'none'
        elif seg_rb_weighted.isChecked():
            _strat = 'weighted_loss'
        else:
            _strat = 'downsampling'
        app_state.train_segmentation_params['imbalance_strategy'].set(_strat)
        for k, e in seg_t_entries.items():
            app_state.train_segmentation_params[k].set(e.text())
        start_segmentation_training(dlg, app_state, seg_ds_combo.currentText())

    btn_train_seg = _btn("Start Training", _train_seg)
    left.addWidget(btn_train_seg, row, 0, 1, 2)

    # ---- Right: Classification ----
    right = QGridLayout()
    right_w = QWidget()
    right_w.setLayout(right)
    right_w.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
    panels.addWidget(right_w, stretch=1, alignment=Qt.AlignmentFlag.AlignTop)

    row = 0
    right.addWidget(QLabel("<b style='font-size:16px'>Classification Network</b>"), row, 0, 1, 2,
                    Qt.AlignmentFlag.AlignCenter)
    row += 1

    use_class_only = QCheckBox("Use classified files only")
    right.addWidget(use_class_only, row, 0, 1, 2)
    row += 1

    cls_sel = Var("current_day")
    cls_grp = QButtonGroup(dlg)
    for txt, val in [("Current Day", "current_day"), ("Current Experiment", "current_experiment"),
                     ("Current Bird", "current_bird")]:
        rb = QRadioButton(txt)
        cls_grp.addButton(rb)
        if val == "current_day":
            rb.setChecked(True)

        def _mc(v=val):
            return lambda: (cls_sel.set(v), app_state.update_batch_select_combobox_class(v))

        rb.toggled.connect(lambda chk, cb=_mc(): cb() if chk else None)
        right.addWidget(rb, row, 0, 1, 2)
        row += 1

    cls_batch = QComboBox()
    cls_batch.addItem("Select Batch File")
    dlg.training_batch_combobox_classification = cls_batch
    app_state.update_batch_select_combobox_class(cls_sel.get())
    right.addWidget(cls_batch, 3, 1)

    right.addWidget(QLabel("Training Dataset Name:"), row, 0)
    cls_ds_name = QLineEdit("edit_class_dataset_name")
    right.addWidget(cls_ds_name, row, 1)
    row += 1

    spec_entries = {}
    for lbl, key in [("N Input Chunks / Size:", 'input_length'), ("Nperseg:", 'nperseg'),
                     ("Noverlap:", 'noverlap'), ("NFFT:", 'nfft'), ("Frequency Cutoffs:", 'freq_cutoffs')]:
        right.addWidget(QLabel(lbl), row, 0)
        e = QLineEdit(app_state.spec_params[key].get())
        spec_entries[key] = e
        right.addWidget(e, row, 1)
        row += 1

    def _create_cls_ds():
        if getattr(dlg, '_training_running', False) or getattr(dlg, '_task_running', False):
            show_info(dlg, "Info", "A training operation is already running.")
            return
        for k, e in spec_entries.items():
            app_state.spec_params[k].set(e.text())
        b, e, d = _get_bird_exp_day(app_state)
        start_create_classification_training_dataset(
            app_state, cls_ds_name.text(), use_class_only.isChecked(),
            cls_sel.get(), cls_batch.currentText(), b, e, d, dlg)

    btn_create_cls = _btn("Create Training Dataset", _create_cls_ds)
    right.addWidget(btn_create_cls, row, 0, 1, 2)
    row += 1

    right.addWidget(QLabel(""), row, 0)
    row += 1

    cls_ds_combo = QComboBox()
    cls_ds_combo.addItem("Select Training Dataset")
    dlg.training_dataset_combobox_classification = cls_ds_combo
    app_state.update_classification_datasets_combobox()
    right.addWidget(cls_ds_combo, row, 0, 1, 2)
    row += 1

    cls_imbalance_grp = QButtonGroup(dlg)
    cls_rb_none = QRadioButton("None")
    cls_rb_down = QRadioButton("Downsampling")
    cls_rb_weighted = QRadioButton("Weighted Loss")
    for rb in [cls_rb_none, cls_rb_down, cls_rb_weighted]:
        cls_imbalance_grp.addButton(rb)
    _cls_strat = app_state.train_classification_params['imbalance_strategy'].get()
    if _cls_strat == 'none':
        cls_rb_none.setChecked(True)
    elif _cls_strat == 'weighted_loss':
        cls_rb_weighted.setChecked(True)
    else:
        cls_rb_down.setChecked(True)
    cls_down_row = QHBoxLayout()
    for rb in [cls_rb_none, cls_rb_down, cls_rb_weighted]:
        cls_down_row.addWidget(rb)

    def _open_augmentation_settings():
        aug = app_state.augmentation_params
        adlg = QDialog(dlg)
        adlg.setWindowTitle("Data Augmentation Settings")
        adlg.resize(360, 280)
        _set_dlg_icon(adlg)
        form = QGridLayout(adlg)
        r = 0

        aug_enabled = QCheckBox("Enable augmentation during training")
        aug_enabled.setChecked(aug['enabled'].get())
        form.addWidget(aug_enabled, r, 0, 1, 2)
        r += 1

        entries = {}
        for lbl, key in [("Probability (0-1):", 'probability'),
                         ("Noise Level:", 'noise_level'),
                         ("Freq Mask Width:", 'freq_mask_width'),
                         ("Time Mask Width:", 'time_mask_width'),
                         ("Compression Factor:", 'compression_factor')]:
            form.addWidget(QLabel(lbl), r, 0)
            e = QLineEdit(aug[key].get())
            entries[key] = e
            form.addWidget(e, r, 1)
            r += 1

        def _apply():
            aug['enabled'].set(aug_enabled.isChecked())
            for k, e in entries.items():
                aug[k].set(e.text())
            adlg.accept()

        btn_row = QHBoxLayout()
        btn_row.addWidget(_btn("OK", _apply))
        btn_row.addWidget(_btn("Cancel", adlg.reject))
        form.addLayout(btn_row, r, 0, 1, 2)
        adlg.exec()

    btn_aug = _btn("Augmentation...", _open_augmentation_settings)
    cls_down_row.addWidget(btn_aug)
    right.addLayout(cls_down_row, row, 0, 1, 2)
    row += 1

    cls_t_entries = {}
    for lbl, key in [("Epochs:", 'epochs'), ("Batch Size:", 'batch_size'),
                     ("Learning Rate:", 'learning_rate'), ("Early Stopping Patience:", 'early_stopping_patience')]:
        right.addWidget(QLabel(lbl), row, 0)
        e = QLineEdit(app_state.train_classification_params[key].get())
        cls_t_entries[key] = e
        right.addWidget(e, row, 1)
        row += 1

    def _train_cls():
        if getattr(dlg, '_training_running', False) or getattr(dlg, '_task_running', False):
            show_info(dlg, "Info", "A training operation is already running.")
            return
        if cls_rb_none.isChecked():
            _strat = 'none'
        elif cls_rb_weighted.isChecked():
            _strat = 'weighted_loss'
        else:
            _strat = 'downsampling'
        app_state.train_classification_params['imbalance_strategy'].set(_strat)
        for k, e in cls_t_entries.items():
            app_state.train_classification_params[k].set(e.text())
        b = app_state.bird_combobox.currentText() if app_state.bird_combobox else ""
        start_classification_training(dlg, app_state, cls_ds_combo.currentText(), b)

    btn_train_cls = _btn("Start Training", _train_cls)
    right.addWidget(btn_train_cls, row, 0, 1, 2)

    # Shared status widgets
    dlg.status_label = QLabel("")
    dlg.status_label.setStyleSheet("color: green; font-size: 14px;")
    dlg.status_label.hide()
    dlg.progressbar = QProgressBar()
    dlg.progressbar.hide()

    root.addWidget(dlg.status_label)
    root.addWidget(dlg.progressbar)

    dlg.show()


# ======================================================================
# Cluster Dialog
# ======================================================================
def open_cluster_window(parent, app_state):
    from moove.utils import (
        start_create_cluster_dataset_thread, start_clustering_thread,
        start_dash_app_thread, replace_labels_from_df, stop_dash_app_thread,
        remove_pkl_suffix,
    )

    dlg = QDialog(parent)
    dlg.setWindowTitle("Cluster")
    dlg.resize(400, 580)
    _set_dlg_icon(dlg)
    app_state.cluster_window = dlg
    dlg._task_running = False
    dlg._task_cancel_requested = False

    def _cluster_close_event(event):
        if getattr(dlg, '_task_running', False):
            reply = QMessageBox.question(
                dlg,
                "Cluster operation is still running",
                "A cluster operation is still running. Abort it and close this window?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                dlg._task_cancel_requested = True
                event.accept()
                return
            event.ignore()
            return
        event.accept()

    dlg.closeEvent = _cluster_close_event

    outer = QVBoxLayout(dlg)
    outer.setContentsMargins(8, 8, 8, 8)
    grid = QGridLayout()
    outer.addLayout(grid)
    outer.addStretch()
    row = 0
    grid.addWidget(QLabel("<b style='font-size:16px'>Cluster Operations</b>"), row, 0, 1, 2,
                   Qt.AlignmentFlag.AlignCenter)
    row += 1

    use_seg = QCheckBox("Use segmented files only")
    grid.addWidget(use_seg, row, 0, 1, 2)
    row += 1

    sel = Var("current_day")
    grp = QButtonGroup(dlg)
    for txt, val in [("Current Day", "current_day"), ("Current Experiment", "current_experiment"),
                     ("Current Bird", "current_bird")]:
        rb = QRadioButton(txt)
        grp.addButton(rb)
        if val == "current_day":
            rb.setChecked(True)

        def _mc(v=val):
            return lambda: (sel.set(v), app_state.update_batch_select_combobox_cluster(v))

        rb.toggled.connect(lambda chk, cb=_mc(): cb() if chk else None)
        grid.addWidget(rb, row, 0)
        row += 1

    batch_combo = QComboBox()
    batch_combo.addItem("Select Batch File")
    dlg.cluster_batch_combobox = batch_combo
    app_state.update_batch_select_combobox_cluster(sel.get())
    grid.addWidget(batch_combo, 3, 1)

    grid.addWidget(QLabel("Cluster Dataset Name:"), row, 0)
    ds_name = QLineEdit("edit_cluster_dataset_name")
    grid.addWidget(ds_name, row, 1)
    row += 1

    spec_entries = {}
    for lbl, key in [("Nperseg:", 'nperseg'), ("Noverlap:", 'noverlap'),
                     ("NFFT:", 'nfft'), ("Frequency Cutoffs:", 'freq_cutoffs')]:
        grid.addWidget(QLabel(lbl), row, 0)
        e = QLineEdit(app_state.spec_params[key].get())
        spec_entries[key] = e
        grid.addWidget(e, row, 1)
        row += 1

    def _create_ds():
        if getattr(dlg, '_task_running', False):
            show_info(dlg, "Info", "A cluster job is already running.")
            return
        for k, e in spec_entries.items():
            app_state.spec_params[k].set(e.text())
        b, e, d = _get_bird_exp_day(app_state)
        start_create_cluster_dataset_thread(app_state, ds_name.text(), use_seg.isChecked(),
                                            sel.get(), batch_combo.currentText(), b, e, d, dlg)

    btn_create = _btn("Create Cluster Dataset", _create_ds)
    grid.addWidget(btn_create, row, 0, 1, 2)
    row += 1

    grid.addWidget(QLabel(""), row, 0)
    row += 1

    clus_combo = QComboBox()
    clus_combo.addItem("Select Cluster Dataset")
    dlg.cluster_dataset_combobox = clus_combo
    app_state.update_cluster_datasets_combobox()
    grid.addWidget(clus_combo, row, 0, 1, 2)
    row += 1

    umap_entries = {}
    for lbl, key in [("N_neighbors:", 'n_neighbors'), ("Min_dist:", 'min_dist'),
                     ("N Syllables:", 'n_clusters')]:
        grid.addWidget(QLabel(lbl), row, 0)
        e = QLineEdit(app_state.umap_k_means_params[key].get())
        umap_entries[key] = e
        grid.addWidget(e, row, 1)
        row += 1

    def _cluster():
        if getattr(dlg, '_task_running', False):
            show_info(dlg, "Info", "A cluster job is already running.")
            return
        for k, e in umap_entries.items():
            app_state.umap_k_means_params[k].set(e.text())
        start_clustering_thread(dlg, app_state, remove_pkl_suffix(clus_combo.currentText()))

    btn_cluster = _btn("Cluster Syllables", _cluster)
    grid.addWidget(btn_cluster, row, 0, 1, 2)
    row += 1

    btn_dash = _btn("Open Dash GUI",
                    lambda: start_dash_app_thread(app_state, remove_pkl_suffix(clus_combo.currentText())))
    grid.addWidget(btn_dash, row, 0)

    btn_close_dash = _btn("Close Dash GUI",
                          lambda: stop_dash_app_thread(app_state))
    grid.addWidget(btn_close_dash, row, 1)
    row += 1

    def _replace_labels():
        if getattr(dlg, '_task_running', False):
            show_info(dlg, "Info", "A cluster job is already running.")
            return
        replace_labels_from_df(app_state, remove_pkl_suffix(clus_combo.currentText()), dlg)

    btn_replace = _btn("Replace Labels", _replace_labels)
    grid.addWidget(btn_replace, row, 0, 1, 2)
    row += 1

    dlg.status_label = QLabel("")
    dlg.status_label.setStyleSheet("color: green; font-size: 14px;")
    dlg.status_label.hide()
    dlg.progressbar = QProgressBar()
    dlg.progressbar.hide()
    outer.addWidget(dlg.status_label)
    outer.addWidget(dlg.progressbar)

    dlg.show()
