# moovegui.py – PyQt6 main window
import os
import sys
import configparser
import shutil
import platform
import ctypes
import logging
import threading
import signal
import time
from pathlib import Path

# Import torch early to avoid DLL/OpenMP conflicts on Windows when
# scientific/GUI stacks (e.g., matplotlib) load runtime DLLs first.
import torch

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.widgets import RectangleSelector
from PIL import Image

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QComboBox, QPushButton, QCheckBox, QRadioButton, QButtonGroup,
    QMessageBox, QSizePolicy
)
from PyQt6.QtCore import Qt, QRect, QTimer
from PyQt6.QtGui import QIcon, QPixmap, QPalette

from moove.qt_helpers import QRangeSliderV, RadioAdapter, set_combo_items, invoke_in_main_thread, show_info
from moove.utils import (
    get_display_data, get_directories, read_batch, get_file_data_by_index,
    save_seg_class_recfile, plot_data, select_event, edit_syllable,
    handle_keypress, zoom, unzoom, swipe_left, swipe_right, handle_playback,
    handle_delete, handle_crop, open_resegment_window, update,
    open_cluster_window, open_training_window, open_relabel_window, find_batch_files,
    create_batch_file, unzoom_small
)
from moove.models.ConvMLP import ConvMLP
from moove.models.CNN import CNN
from moove.app_state import AppState

for key in list(mpl.rcParams):
    if key.startswith('keymap.'):
        mpl.rcParams[key] = []

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------
moove_config_dir = os.environ.get('MOOVE_CONFIG_DIR')
if moove_config_dir:
    home_config_dir = os.path.expanduser(moove_config_dir)
else:
    home_config_dir = os.path.join(Path.home(), ".moove")

config_file_path = os.path.join(home_config_dir, 'moove_config.ini')

if not os.path.exists(config_file_path):
    example_config_file_path = os.path.join(os.path.dirname(__file__), 'moove_config.ini.example')
    os.makedirs(home_config_dir, exist_ok=True)
    shutil.copy(example_config_file_path, config_file_path)
    logger.info(f"Created config file at: {config_file_path}")

_config = configparser.ConfigParser()
_config.read(config_file_path)
_global_dir = os.path.expanduser(_config.get("GENERAL", "global_dir"))

for subdir in ["rec_data", "trained_models", "training_data", "cluster_data", "playbacks"]:
    os.makedirs(os.path.join(_global_dir, subdir), exist_ok=True)

package_example_data = os.path.join(os.path.dirname(__file__), "example_data", "bird_x")
target_bird_x_dir = os.path.join(_global_dir, "rec_data", "bird_x")
if not os.path.exists(target_bird_x_dir):
    shutil.copytree(package_example_data, target_bird_x_dir)

package_example_data_WN = os.path.join(os.path.dirname(__file__), "example_data", "white_noise")
target_WN_dir = os.path.join(_global_dir, "playbacks", "white_noise")
if not os.path.exists(target_WN_dir):
    shutil.copytree(package_example_data_WN, target_WN_dir)


def _resolve_icon_path():
    """Return preferred icon path, platform-aware.

    macOS: .icns or high-res PNG (Retina).
    Windows: .ico first.
    Fallback: any available PNG.
    """
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    assets = os.path.normpath(os.path.join(pkg_dir, "..", "assets"))

    if sys.platform == "darwin":
        candidates = [
            os.path.join(assets, "logo_128_white_bg.icns"),
            os.path.join(assets, "logo_white_bg.png"),
            os.path.join(assets, "logo_128_white_bg.png"),
            os.path.join(assets, "logo.png"),
        ]
    else:
        candidates = [
            os.path.join(assets, "logo_128_white_bg_scaled.ico"),
            os.path.join(assets, "logo_128_white_bg.png"),
            os.path.join(assets, "logo.png"),
        ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


class MooveMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MooveGUI")
        self._set_icon()

        # App state
        self.app_state = AppState(_global_dir)
        self.app_state.load_state()
        self._apply_config()

        # Derive colours from palette
        bg = self.palette().color(QPalette.ColorRole.Window)
        brightness = (bg.red() * 299 + bg.green() * 587 + bg.blue() * 114) / 1000
        self.app_state.text_color = "#ffffff" if brightness < 128 else "#000000"
        self.app_state.bg_color = bg.name()

        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(2)

        self._build_top_bar(main_layout)
        self._build_plot_area(main_layout)
        self._build_button_bar(main_layout)
        self._build_radio_bar(main_layout)
        self._connect_canvas_events()

        plot_data(self.app_state)
        self.app_state.init_flag = True

    def restore_last_window_geometry(self):
        """Restore window position/size from app state if it is still on any screen."""
        geom = self.app_state.window_geometry
        if not isinstance(geom, dict):
            return False

        try:
            x = int(geom.get("x"))
            y = int(geom.get("y"))
            width = int(geom.get("width"))
            height = int(geom.get("height"))
        except (TypeError, ValueError):
            return False

        if width <= 0 or height <= 0:
            return False

        target = QRect(x, y, width, height)
        for screen in QApplication.screens():
            if screen.availableGeometry().intersects(target):
                self.setGeometry(target)
                return True
        return False

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------
    def _apply_config(self):
        s = self.app_state
        s.config["global_dir"] = _global_dir
        s.config["rec_data"] = os.path.join(_global_dir, "rec_data")
        s.config["lower_spec_plot"] = int(_config.get('GUI', 'lower_spec_plot'))
        s.config["upper_spec_plot"] = int(_config.get('GUI', 'upper_spec_plot'))
        s.config["vmin_range_slider"] = float(_config.get('GUI', 'vmin_range_slider'))
        s.config["vmax_range_slider"] = float(_config.get('GUI', 'vmax_range_slider'))
        s.config["spec_nperseg"] = int(_config.get('GUI', 'spec_nperseg'))
        s.config["spec_noverlap"] = int(_config.get('GUI', 'spec_noverlap'))
        s.config["spec_nfft"] = int(_config.get('GUI', 'spec_nfft'))
        s.config["performance"] = str(_config.get('GUI', 'performance'))

    # ------------------------------------------------------------------
    # Icon
    # ------------------------------------------------------------------
    def _set_icon(self):
        try:
            icon_path = _resolve_icon_path()
            if icon_path:
                icon = QIcon(icon_path)
                self.setWindowIcon(icon)
                QApplication.instance().setWindowIcon(icon)
            if sys.platform == 'win32':
                ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID('moove.gui')
        except Exception as e:
            logger.warning(f"Could not set window icon: {e}")

    def _apply_native_windows_icon(self):
        """Apply taskbar icon via WinAPI as fallback when Qt icon is ignored."""
        if sys.platform != "win32":
            return

        icon_path = _resolve_icon_path()
        if not icon_path:
            return

        try:
            hwnd = int(self.winId())
            user32 = ctypes.windll.user32

            IMAGE_ICON = 1
            LR_LOADFROMFILE = 0x0010
            LR_DEFAULTSIZE = 0x0040
            WM_SETICON = 0x0080
            ICON_SMALL = 0
            ICON_BIG = 1

            hicon = user32.LoadImageW(None, icon_path, IMAGE_ICON, 0, 0, LR_LOADFROMFILE | LR_DEFAULTSIZE)
            if hicon:
                user32.SendMessageW(hwnd, WM_SETICON, ICON_SMALL, hicon)
                user32.SendMessageW(hwnd, WM_SETICON, ICON_BIG, hicon)
        except Exception as e:
            logger.warning(f"Could not apply native Windows icon: {e}")

    # ------------------------------------------------------------------
    # Top bar (comboboxes + checkboxes)
    # ------------------------------------------------------------------
    def _build_top_bar(self, parent_layout):
        s = self.app_state
        bar = QHBoxLayout()

        # Determine restored path parts
        path_parts = None
        if s.data_dir:
            selected_day_path = s.data_dir
            path_parts = Path(selected_day_path).parts
            if path_parts[-4] != Path(s.config['rec_data']).name:
                s.data_dir = None
                path_parts = None

        # Bird
        birds = sorted(get_directories(s.config['rec_data']))
        self.bird_combo = QComboBox()
        self.bird_combo.setMinimumWidth(120)
        self.bird_combo.addItems(birds)
        s.bird_combobox = self.bird_combo
        if path_parts and path_parts[-3] in birds:
            self.bird_combo.setCurrentText(path_parts[-3])
        self.bird_combo.currentTextChanged.connect(self._on_bird_changed)
        bar.addWidget(self.bird_combo)

        # Experiment
        self.experiment_combo = QComboBox()
        self.experiment_combo.setMinimumWidth(180)
        s.experiment_combobox = self.experiment_combo
        self._populate_experiments()
        if path_parts:
            exps = [self.experiment_combo.itemText(i) for i in range(self.experiment_combo.count())]
            if path_parts[-2] in exps:
                self.experiment_combo.setCurrentText(path_parts[-2])
        self.experiment_combo.currentTextChanged.connect(self._on_experiment_changed)
        bar.addWidget(self.experiment_combo)

        # Day
        self.day_combo = QComboBox()
        self.day_combo.setMinimumWidth(120)
        s.day_combobox = self.day_combo
        self._populate_days()
        if path_parts:
            days_list = [self.day_combo.itemText(i) for i in range(self.day_combo.count())]
            if path_parts[-1] in days_list:
                self.day_combo.setCurrentText(path_parts[-1])
            else:
                self._set_data_dir_from_combos()
        else:
            self._set_data_dir_from_combos()
        self.day_combo.currentTextChanged.connect(lambda: self._on_day_changed())
        bar.addWidget(self.day_combo)

        # Load batch & song files
        selected_day_path = s.data_dir or os.path.join(
            s.config['rec_data'], self.bird_combo.currentText(),
            self.experiment_combo.currentText(), self.day_combo.currentText())
        s.data_dir = selected_day_path

        batch_files = find_batch_files(selected_day_path)
        if s.current_batch_file in batch_files:
            s.song_files = read_batch(selected_day_path, s.current_batch_file)
        else:
            s.current_batch_file = "batch.txt"
            s.song_files = read_batch(selected_day_path)
        if s.current_file_index is None:
            s.current_file_index = 0

        # Refresh batch files on startup
        valid_files = sorted(
            f for f in os.listdir(s.data_dir) if f.endswith('.wav') or f.endswith('.cbin'))
        for batch in batch_files:
            bp = os.path.join(s.data_dir, batch)
            if batch == 'batch.txt':
                with open(bp, 'w') as fh:
                    fh.write('\n'.join(valid_files))
            else:
                with open(bp, 'r') as fh:
                    keep = fh.read().splitlines()
                with open(bp, 'w') as fh:
                    fh.write('\n'.join(f for f in keep if f in valid_files))
        s.song_files = read_batch(s.data_dir, s.current_batch_file)

        # If the previously stored file is gone, default to the first file of the day.
        if s.song_files:
            if s.current_file_index is None or s.current_file_index < 0 or s.current_file_index >= len(s.song_files):
                s.current_file_index = 0
            else:
                selected_name = s.song_files[s.current_file_index]
                selected_path = os.path.join(s.data_dir, selected_name)
                if not os.path.exists(selected_path):
                    s.current_file_index = 0

        # File combobox
        self.file_combo = QComboBox()
        self.file_combo.setMinimumWidth(260)
        self.file_combo.addItems(s.song_files)
        if s.song_files:
            self.file_combo.setCurrentText(s.song_files[s.current_file_index])
        s.combobox = self.file_combo
        self.file_combo.currentTextChanged.connect(self._on_file_changed)
        bar.addWidget(self.file_combo)

        # Batch combobox
        self.batch_combo = QComboBox()
        self.batch_combo.setMinimumWidth(180)
        self.batch_combo.addItems(batch_files)
        self.batch_combo.setCurrentText(s.current_batch_file)
        s.batch_combobox = self.batch_combo
        self.batch_combo.currentTextChanged.connect(self._on_batch_changed)
        bar.addWidget(self.batch_combo)

        bar.addStretch()

        # Segmented / Classified checkboxes
        self.segmented_cb = QCheckBox("Segmented")
        self.classified_cb = QCheckBox("Classified")
        s.segmented_checkbox = self.segmented_cb
        s.classified_checkbox = self.classified_cb
        self.segmented_cb.stateChanged.connect(self._on_checkbox_toggle)
        self.classified_cb.stateChanged.connect(self._on_checkbox_toggle)
        bar.addWidget(self.segmented_cb)
        bar.addSpacing(8)
        bar.addWidget(self.classified_cb)

        parent_layout.addLayout(bar)

    # Plot area (matplotlib canvas + range slider)
    def _build_plot_area(self, parent_layout):
        s = self.app_state
        plot_row = QHBoxLayout()

        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(
            3, 1, figsize=(9, 5.5),
            gridspec_kw={'height_ratios': [6, 1, 6]}, sharex=True)

        self.canvas = FigureCanvasQTAgg(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.canvas.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        s.set_axes(self.ax1, self.ax2, self.ax3)
        s.set_canvas(self.canvas)
        if s.song_files:
            try:
                s.display_dict = get_display_data(
                    get_file_data_by_index(s.data_dir, s.song_files, s.current_file_index, s),
                    s.config)
            except Exception as exc:
                s.logger.warning("Initial file load failed, will use plot fallback handling: %s", exc)
                s.display_dict = None
        else:
            s.display_dict = None
        s.ax3_background = s.canvas.copy_from_bbox(s.ax3.bbox)

        plot_row.addWidget(self.canvas, stretch=1)

        # Range slider
        vmin_cfg = s.config['vmin_range_slider']
        vmax_cfg = s.config['vmax_range_slider']
        if (s.current_vmin is not None and s.current_vmax is not None
                and s.current_vmin > vmin_cfg and s.current_vmax < vmax_cfg):
            init_bot, init_top = s.current_vmin, s.current_vmax
        else:
            dist = (vmax_cfg - vmin_cfg) / 4
            init_bot, init_top = vmin_cfg + dist, vmax_cfg - dist

        self.range_slider = QRangeSliderV(vmin_cfg, vmax_cfg, init_bot, init_top)
        self.range_slider.setFixedWidth(100)
        self.range_slider.valuesChanged.connect(self._on_slider_changed)
        plot_row.addWidget(self.range_slider)

        parent_layout.addLayout(plot_row, stretch=1)

        # Rectangle selectors
        self.rect_sel_ax1 = RectangleSelector(
            self.ax1, self._on_rect_select, useblit=True, button=[1],
            minspanx=30, minspany=30, spancoords='pixels', interactive=False,
            state_modifier_keys={"rotate": ""})
        self.rect_sel_ax3 = RectangleSelector(
            self.ax3, self._on_rect_select, useblit=True, button=[1],
            minspanx=30, minspany=30, spancoords='pixels', interactive=False,
            state_modifier_keys={"rotate": ""})

    # ------------------------------------------------------------------
    # Button bar
    # ------------------------------------------------------------------
    def _build_button_bar(self, parent_layout):
        s = self.app_state
        bar = QHBoxLayout()
        bar.setContentsMargins(0, 6, 0, 0)

        refresh_text = "↻" if platform.system() == 'Darwin' else "⟳"
        btn = lambda text, cb: self._make_btn(text, cb, bar)

        bar.addStretch()
        btn(refresh_text, lambda: update(s))
        btn("Previous", lambda: (s.change_file(-1), plot_data(s)))
        btn("Next", lambda: (s.change_file(1), plot_data(s)))
        btn("  <  ", lambda: swipe_left(s))
        btn("  >  ", lambda: swipe_right(s))
        btn("🏠", lambda: unzoom(s))
        btn("Zoom", lambda: zoom(s))
        btn("Unzoom", lambda: unzoom_small(s))
        btn("Crop", lambda: handle_crop(s))
        btn("Delete", lambda: handle_delete(s))
        btn("Play", lambda: handle_playback(s))
        btn("Resegment", lambda: open_resegment_window(self, s))
        btn("Relabel", lambda: open_relabel_window(self, s))
        btn("Training", lambda: open_training_window(self, s))
        btn("Cluster", lambda: open_cluster_window(self, s))
        bar.addStretch()

        parent_layout.addLayout(bar)

    @staticmethod
    def _make_btn(text, callback, layout):
        b = QPushButton(text)
        b.clicked.connect(callback)
        layout.addWidget(b)
        return b

    # ------------------------------------------------------------------
    # Radio buttons for edit mode
    # ------------------------------------------------------------------
    def _build_radio_bar(self, parent_layout):
        bar = QHBoxLayout()
        bar.addStretch()
        self.edit_group = QButtonGroup(self)
        options = [("None", 1), ("New Segment", 2), ("Delete Segment", 3),
                   ("Move Segment", 4), ("Label Interactive", 5)]
        for txt, val in options:
            rb = QRadioButton(txt)
            self.edit_group.addButton(rb, val)
            bar.addWidget(rb)
            if val == 1:
                rb.setChecked(True)
        self.edit_group.idToggled.connect(self._on_edit_type_toggled)
        bar.addStretch()
        parent_layout.addLayout(bar)

        self.radio_adapter = RadioAdapter(self.edit_group)
        self.app_state.reset_edit_type_gui = lambda: self.radio_adapter.set("1")

    # ------------------------------------------------------------------
    # Canvas events
    # ------------------------------------------------------------------
    def _connect_canvas_events(self):
        s = self.app_state
        self.canvas.mpl_connect('key_press_event',
                                lambda ev: handle_keypress(ev, s, self.radio_adapter))
        self.canvas.mpl_connect('button_press_event', lambda ev: select_event(ev, s))
        self.canvas.mpl_connect('key_press_event', lambda ev: edit_syllable(ev, s))

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------
    def _on_bird_changed(self):
        self._populate_experiments()
        self._populate_days()
        self._on_day_changed()

    def _on_experiment_changed(self):
        self._populate_days()
        self._on_day_changed()

    def _on_day_changed(self):
        s = self.app_state
        if not s.init_flag:
            return
        self._set_data_dir_from_combos()
        s.current_file_index = 0

        batch_files = find_batch_files(s.data_dir)
        set_combo_items(self.batch_combo, batch_files, "batch.txt")
        if not batch_files:
            create_batch_file(s.data_dir)
        s.current_batch_file = "batch.txt"

        s.song_files = read_batch(s.data_dir, s.current_batch_file)
        set_combo_items(self.file_combo, s.song_files,
                        s.song_files[s.current_file_index] if s.song_files else None)
        if s.song_files:
            plot_data(s)
        else:
            for ax in [s.ax1, s.ax2, s.ax3]:
                ax.clear()
            s.canvas.draw()

    def _on_file_changed(self):
        self.app_state.selected_syllable_index = None
        s = self.app_state
        selected = self.file_combo.currentText()
        if selected and selected in s.song_files:
            s.current_file_index = s.song_files.index(selected)
            plot_data(s)

    def _on_batch_changed(self):
        s = self.app_state
        selected = self.batch_combo.currentText()
        if not selected:
            return
        s.current_batch_file = selected
        s.song_files = read_batch(s.data_dir, selected)
        s.current_file_index = 0 if s.song_files else None
        set_combo_items(self.file_combo, s.song_files,
                        s.song_files[0] if s.song_files else "")
        if s.song_files:
            plot_data(s)
        else:
            for ax in [s.ax1, s.ax2, s.ax3]:
                ax.clear()
            s.canvas.draw()

    def _on_slider_changed(self, vmin, vmax):
        s = self.app_state
        s.current_vmin = vmin
        s.current_vmax = vmax
        s.redraw_spectrogram(vmin, vmax)

    def _on_checkbox_toggle(self):
        s = self.app_state
        seg = "1" if self.segmented_cb.isChecked() else "0"
        cla = "1" if self.classified_cb.isChecked() else "0"
        s.segmented_var.set(seg)
        s.classified_var.set(cla)
        fp = get_file_data_by_index(s.data_dir, s.song_files, s.current_file_index, s)
        rec_path = os.path.splitext(fp["file_path"])[0] + ".rec"
        saved = save_seg_class_recfile(rec_path, seg, cla)
        if not saved:
            show_info(self, "Error", f"Could not save status because rec file is missing:\n{rec_path}")

    def _on_rect_select(self, eclick, erelease):
        if abs(eclick.xdata - erelease.xdata) * 1000 < 5:
            return

        def _set(axis, ec, er):
            if ec.ydata > er.ydata:
                ec.ydata, er.ydata = er.ydata, ec.ydata
            if ec.xdata > er.xdata:
                ec.xdata, er.xdata = er.xdata, ec.xdata
            axis.set_xlim(ec.xdata, er.xdata)

        if eclick.inaxes == self.ax1:
            _set(self.ax1, eclick, erelease)
        elif eclick.inaxes == self.ax3:
            _set(self.ax3, eclick, erelease)
        self.canvas.draw()

    _EDIT_MAP = {1: "None", 2: "New Segment", 3: "Delete Segment",
                 4: "Move Segment", 5: "Label Interactive"}

    def _on_edit_type_toggled(self, btn_id, checked):
        if not checked:
            return
        value = self._EDIT_MAP.get(btn_id, "None")
        self.app_state.edit_type = value
        # cursor = Qt.CursorShape.CrossCursor if value != "None" else Qt.CursorShape.ArrowCursor
        cursor = (
            Qt.CursorShape.ArrowCursor
            if value in ("None", "Label Interactive")
            else Qt.CursorShape.CrossCursor
        )
        self.canvas.setCursor(cursor)
        if value == "None" and self.app_state.selected_syllable_index is not None:
            if len(self.app_state.ax2.texts) <= self.app_state.selected_syllable_index:
                try:
                    self.app_state.ax2.texts[self.app_state.selected_syllable_index].set_color('black')
                except IndexError:
                    return  # better/other solution than changing 'if' statement
            self.app_state.selected_syllable_index = None
            self.canvas.draw_idle()
        elif value == 'Label Interactive':
            self.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _populate_experiments(self):
        bird = self.bird_combo.currentText()
        if not bird:
            return
        exps = sorted(get_directories(os.path.join(self.app_state.config['rec_data'], bird)))
        set_combo_items(self.experiment_combo, exps, exps[0] if exps else None)

    def _populate_days(self):
        bird = self.bird_combo.currentText()
        exp = self.experiment_combo.currentText()
        if not bird or not exp:
            return
        days = sorted(get_directories(
            os.path.join(self.app_state.config['rec_data'], bird, exp)))
        set_combo_items(self.day_combo, days, days[0] if days else None)

    def _set_data_dir_from_combos(self):
        self.app_state.data_dir = os.path.join(
            self.app_state.config['rec_data'],
            self.bird_combo.currentText(),
            self.experiment_combo.currentText(),
            self.day_combo.currentText())

    # ------------------------------------------------------------------
    # Close
    # ------------------------------------------------------------------
    def closeEvent(self, event):
        s = self.app_state
        g = self.geometry()
        s.window_geometry = {
            "x": g.x(),
            "y": g.y(),
            "width": g.width(),
            "height": g.height(),
        }
        with s.thread_lock:
            active = len(s.active_threads)
        if active > 0:
            reply = QMessageBox.question(
                self, "Active Threads",
                f"{active} thread(s) still running. Close anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.No:
                event.ignore()
                return
            s.shutdown_all_threads()

        s.save_state()
        logger.info("Application state saved")

        # Timeout safety net
        def _force():
            time.sleep(3.0)
            os._exit(1)

        threading.Thread(target=_force, daemon=True).start()

        event.accept()
        QApplication.instance().quit()


def main():
    if sys.platform == "win32":
        try:
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("moove.gui")
        except Exception as e:
            logger.warning(f"Could not set Windows AppUserModelID early: {e}")

    app = QApplication(sys.argv)
    app.setApplicationName("MooveGUI")
    app.setOrganizationName("Moove")

    icon_path = _resolve_icon_path()

    if icon_path:
        app.setWindowIcon(QIcon(icon_path))
        # On macOS, also set the Dock / NSApplication icon so that
        # QMessageBox and other system dialogs show the Moove logo
        # instead of the default Python rocket.
        if sys.platform == "darwin":
            try:
                from AppKit import NSApplication, NSImage
                ns_image = NSImage.alloc().initByReferencingFile_(icon_path)
                NSApplication.sharedApplication().setApplicationIconImage_(ns_image)
            except Exception:
                pass

    window = MooveMainWindow()
    if not window.restore_last_window_geometry():
        window.resize(1200, 600)
    window.show()
    QTimer.singleShot(0, window._apply_native_windows_icon)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
