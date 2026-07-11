# app_state.py
import logging
import os
import threading
import json
import re


class Var:
    """Drop-in replacement for tk.StringVar with .get()/.set() interface."""
    def __init__(self, value=""):
        self._value = str(value)

    def get(self):
        return self._value

    def set(self, value):
        self._value = str(value)

    def __repr__(self):
        return f"Var({self._value!r})"


class BoolVar:
    """Drop-in replacement for tk.BooleanVar with .get()/.set() interface."""
    def __init__(self, value=False):
        self._value = bool(value)

    def get(self):
        return self._value

    def set(self, value):
        self._value = bool(value)

    def __repr__(self):
        return f"BoolVar({self._value!r})"


def _set_combobox_items(combobox, items):
    """Set items on a QComboBox, preserving current text if possible."""
    if combobox is None:
        return
    current = combobox.currentText()
    combobox.clear()
    combobox.addItems(items)
    idx = combobox.findText(current)
    if idx >= 0:
        combobox.setCurrentIndex(idx)


class AppState:
    def __init__(self, global_dir):
        self.text_color = None
        self.bg_color = None
        self.dash_thread = None
        self.server = None
        self.server_thread = None
        self.stop_event = threading.Event()
        self.spec = None
        self.spec_full_data = None
        self.spec_low_data = None
        self.canvas = None
        self.ax1 = None
        self.ax2 = None
        self.ax3 = None
        self.ax2_background = None
        self.ax3_background = None
        self.last_ax3_marker = None
        self.display_dict = None
        self.edit_type = "None"
        self.moved_point = None
        self.new_onset = None
        self.selected_syllable_index = None
        # Segmentation undo/redo history (per loaded file).
        self.seg_history = []
        self.seg_history_index = -1
        self.data_dir = ""
        self.current_file_index = 0
        self.last_file_delta = 0
        self.song_files = []
        self.current_batch_file = "batch.txt"
        self.window_geometry = None
        self.original_x_range = None
        self.original_y_range_ax1 = None
        self.original_y_range_ax2 = None
        self.original_y_range_ax3 = None
        self.current_vmin = None
        self.current_vmax = None
        self.combobox = None
        self.batch_combobox = None
        self.bird_combobox = None
        self.experiment_combobox = None
        self.day_combobox = None
        self.segmented_checkbox = None
        self.classified_checkbox = None
        self.shortcuts = None
        self.reset_edit_type_gui = None
        self.config = {
            'global_dir': global_dir,
            'rec_data': Var(value="rec_data"),
            'lower_spec_plot': Var(value="500"),
            'upper_spec_plot': Var(value="12500"),
            'vmin_range_slider': Var(value="-100"),
            'vmax_range_slider': Var(value="-10"),
            'spec_nfft': Var(value="1024"),
            'spec_noverlap': Var(value="896"),
            'spec_nperseg': Var(value="1024"),
            'performance': Var(value="fast"),
        }
        self.evfuncs_params = {
            'threshold': Var(value="-50"),
            'min_syl_dur': Var(value="0.03"),
            'min_silent_dur': Var(value="0.005"),
            'freq_cutoffs': Var(value="500,10000"),
            'smooth_window': Var(value="2"),
        }
        self.mlseg_params = {
            'decision_threshold': Var(value="0.5"),
            'onset_window_size': Var(value="5"),
            'n_onset_true': Var(value="3"),
            'offset_window_size': Var(value="5"),
            'n_offset_false': Var(value="4"),
            'min_syllable_length': Var(value="0.03"),
            'min_silent_duration': Var(value="0.005"),
        }
        self.spec_params = {
            'nperseg': Var(value="64"),
            'noverlap': Var(value="32"),
            'nfft': Var(value="128"),
            'freq_cutoffs': Var(value="0,22050"),
            'input_length': Var(value="21,64"),
        }
        self.umap_k_means_params = {
            'n_neighbors': Var(value="15"),
            'min_dist': Var(value="0.1"),
            'n_clusters': Var(value="10"),
        }
        self.train_classification_params = {
            'epochs': Var(value="1000"),
            'batch_size': Var(value="64"),
            'learning_rate': Var(value="0.001"),
            'early_stopping_patience': Var(value="5"),
            'imbalance_strategy': Var(value="weighted_loss"),
        }
        self.augmentation_params = {
            'enabled': BoolVar(value=True),
            'probability': Var(value="0.2"),
            'noise_level': Var(value="0.0001"),
            'freq_mask_width': Var(value="10"),
            'time_mask_width': Var(value="10"),
            'compression_factor': Var(value="0.5"),
        }
        self.train_segmentation_params = {
            'hist_size': Var(value="3"),
            'chunk_size': Var(value="64"),
            'overlap_chunks': BoolVar(value=False),
            'epochs': Var(value="1000"),
            'batch_size': Var(value="64"),
            'learning_rate': Var(value="0.001"),
            'early_stopping_patience': Var(value="5"),
            'imbalance_strategy': Var(value="weighted_loss"),
        }
        self.segmented_var = Var(value="0")
        self.classified_var = Var(value="0")
        self.resegment_window = None
        self.training_window = None
        self.cluster_window = None
        self.relabel_window = None
        self.current_segmentation_model = None
        self.current_classification_model = None
        self.init_flag = False
        self.update_timer = None

        self.active_threads = []
        self.thread_lock = threading.Lock()

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def save_state(self, filename="app_state.json"):
        """Saves the current state of the GUI in the app_state.json file"""
        global_dir = self.config['global_dir']
        filepath = os.path.join(global_dir, filename)

        try:
            os.makedirs(global_dir, exist_ok=True)
        except Exception as e:
            self.logger.error(f"Failed to create directory {global_dir}: {e}")
            return

        try:
            state_dict = {
                'current_vmin': self.current_vmin,
                'current_vmax': self.current_vmax,
                'data_dir': self.data_dir,
                'song_files': self.song_files,
                'current_file_index': self.current_file_index,
                'current_batch_file': self.current_batch_file,
                'window_geometry': self.window_geometry,
                'evfuncs_params': {key: value.get() for key, value in self.evfuncs_params.items()},
                'mlseg_params': {key: value.get() for key, value in self.mlseg_params.items()},
                'spec_params': {key: value.get() for key, value in self.spec_params.items()},
                'umap_k_means_params': {key: value.get() for key, value in self.umap_k_means_params.items()},
                'train_segmentation_params': {key: value.get() for key, value in self.train_segmentation_params.items()},
                'train_classification_params': {key: value.get() for key, value in self.train_classification_params.items()},
                'augmentation_params': {key: value.get() for key, value in self.augmentation_params.items()},
            }

            with open(filepath, 'w') as f:
                json.dump(state_dict, f)

        except Exception as e:
            self.logger.error(f"Error saving app state to {filepath}: {e}")

    def load_state(self, filename="app_state.json"):
        """Loads the previous state of the GUI saved in the app_state.json"""
        global_dir = self.config['global_dir']
        filepath = os.path.join(global_dir, filename)

        if not os.path.exists(filepath):
            self.logger.warning(f"No state file found at: {filepath}")
            return

        try:
            with open(filepath, 'r') as f:
                state_dict = json.load(f)
        except json.JSONDecodeError as e:
            self.logger.error(f"Error reading state file: {e}")
            return

        self.current_vmin = state_dict.get('current_vmin')
        self.current_vmax = state_dict.get('current_vmax')
        self.data_dir = state_dict.get('data_dir')
        self.song_files = state_dict.get('song_files')
        self.current_file_index = state_dict.get('current_file_index')
        self.current_batch_file = state_dict.get('current_batch_file', 'batch')
        self.window_geometry = state_dict.get('window_geometry')

        for key, value in state_dict.get('evfuncs_params', {}).items():
            if key in self.evfuncs_params:
                self.evfuncs_params[key].set(value)
        for key, value in state_dict.get('mlseg_params', {}).items():
            if key in self.mlseg_params:
                self.mlseg_params[key].set(value)
        for key, value in state_dict.get('spec_params', {}).items():
            if key in self.spec_params:
                self.spec_params[key].set(value)
        for key, value in state_dict.get('umap_k_means_params', {}).items():
            if key in self.umap_k_means_params:
                self.umap_k_means_params[key].set(value)
        for key, value in state_dict.get('train_segmentation_params', {}).items():
            if key in self.train_segmentation_params:
                self.train_segmentation_params[key].set(value)
        for key, value in state_dict.get('train_classification_params', {}).items():
            if key in self.train_classification_params:
                self.train_classification_params[key].set(value)
        for key, value in state_dict.get('augmentation_params', {}).items():
            if key in self.augmentation_params:
                self.augmentation_params[key].set(value)

    def set_canvas(self, canvas):
        self.canvas = canvas

    def draw_canvas(self):
        if self.canvas:
            # draw_idle() is non-blocking and coalesces a burst of redraws into a
            # single render, which keeps zoom/pan/slider interaction responsive.
            # (The old _update_ax2_background recapture was removed: the cached
            # ax2_background was never restored anywhere, so it was pure overhead
            # on every draw.)
            self.canvas.draw_idle()

    def redraw_spectrogram(self, vmin, vmax):
        if self.canvas is not None and hasattr(self.canvas, 'set_levels'):
            self.canvas.set_levels(vmin, vmax)

    def set_axes(self, ax1, ax2, ax3):
        self.ax1 = ax1
        self.ax2 = ax2
        self.ax3 = ax3
        for ax in [self.ax1, self.ax2, self.ax3]:
            ax.set_clip_on(True)
            for line in ax.lines:
                line.set_clip_on(True)

    def get_axes(self):
        return self.ax1, self.ax2, self.ax3

    def set_original_x_range(self, original_x_range):
        self.original_x_range = original_x_range

    def set_original_y_range_ax1(self, original_y_range_ax1, original_y_range_ax2, original_y_range_ax3):
        self.original_y_range_ax1 = original_y_range_ax1
        self.original_y_range_ax2 = original_y_range_ax2
        self.original_y_range_ax3 = original_y_range_ax3

    def get_data_dir(self):
        return self.data_dir

    def change_file(self, delta):
        self.last_file_delta = delta
        current_file_index = self.current_file_index
        current_file_index += delta
        current_file_index = max(0, min(len(self.song_files) - 1, current_file_index))
        self.current_file_index = current_file_index
        if self.combobox is not None:
            self.combobox.setCurrentText(self.song_files[current_file_index])
        self.selected_syllable_index = None
        self.edit_type = "None"
        if self.reset_edit_type_gui:
            self.reset_edit_type_gui()

    def _get_batch_files(self, select_path="current_day"):
        """Collect batch file names based on selection scope."""
        birds = os.path.abspath(os.path.join(self.data_dir, "..", ".."))
        experiments = os.path.join(self.data_dir, "..")
        day = self.data_dir
        batch_files = []
        if select_path == "current_day":
            batch_files = [f for f in os.listdir(day) if re.match('.*batch.*', f)]
        elif select_path == "current_experiment":
            batch_files = [f for f in os.listdir(experiments) if re.match('.*batch.*', f)]
        elif select_path == "current_bird":
            batch_files = [f for f in os.listdir(birds) if re.match('.*batch.*', f)]
        return ["All Files"] + batch_files

    def update_classification_datasets_combobox(self):
        training_data_folder = os.path.join(self.config['global_dir'], "training_data")
        datasets = [f for f in os.listdir(training_data_folder) if f.endswith("_class.pkl")]
        combo = getattr(self.training_window, 'training_dataset_combobox_classification', None)
        _set_combobox_items(combo, datasets)

    def update_segmentation_datasets_combobox(self):
        training_data_folder = os.path.join(self.config['global_dir'], "training_data")
        datasets = [f for f in os.listdir(training_data_folder) if f.endswith("_seg.pkl")]
        combo = getattr(self.training_window, 'training_dataset_combobox_segmentation', None)
        _set_combobox_items(combo, datasets)

    def update_cluster_datasets_combobox(self):
        cluster_data_folder = os.path.join(self.config['global_dir'], "cluster_data")
        datasets = [f for f in os.listdir(cluster_data_folder) if f.endswith(".pkl")]
        combo = getattr(self.cluster_window, 'cluster_dataset_combobox', None)
        _set_combobox_items(combo, datasets)

    def update_batch_select_combobox_resegment_ev(self, select_path="current_day"):
        batch_files = self._get_batch_files(select_path)
        combo = getattr(self.resegment_window, 'resegment_batch_combobox_ev', None)
        if combo is not None:
            combo.clear()
            combo.addItems(batch_files)
            combo.setCurrentText("All Files")

    def update_batch_select_combobox_resegment(self, select_path="current_day"):
        batch_files = self._get_batch_files(select_path)
        combo = getattr(self.resegment_window, 'resegment_batch_combobox', None)
        if combo is not None:
            combo.clear()
            combo.addItems(batch_files)
            combo.setCurrentText("All Files")

    def update_batch_select_combobox_relabel(self, select_path="current_day"):
        batch_files = self._get_batch_files(select_path)
        combo = getattr(self.relabel_window, 'relabel_batch_combobox', None)
        if combo is not None:
            combo.clear()
            combo.addItems(batch_files)
            combo.setCurrentText("All Files")

    def update_batch_select_combobox_class(self, select_path="current_day"):
        batch_files = self._get_batch_files(select_path)
        combo = getattr(self.training_window, 'training_batch_combobox_classification', None)
        if combo is not None:
            combo.clear()
            combo.addItems(batch_files)
            combo.setCurrentText("All Files")

    def update_batch_select_combobox_segment(self, select_path="current_day"):
        batch_files = self._get_batch_files(select_path)
        combo = getattr(self.training_window, 'training_batch_combobox_segmentation', None)
        if combo is not None:
            combo.clear()
            combo.addItems(batch_files)
            combo.setCurrentText("All Files")

    def update_batch_select_combobox_cluster(self, select_path="current_day"):
        batch_files = self._get_batch_files(select_path)
        combo = getattr(self.cluster_window, 'cluster_batch_combobox', None)
        if combo is not None:
            combo.clear()
            combo.addItems(batch_files)
            combo.setCurrentText("All Files")

    def add_thread(self, thread):
        with self.thread_lock:
            self.active_threads.append(thread)

    def remove_thread(self, thread):
        with self.thread_lock:
            if thread in self.active_threads:
                self.active_threads.remove(thread)

    def shutdown_all_threads(self):
        """Gracefully shutdown all active threads"""
        with self.thread_lock:
            active_count = len(self.active_threads)

        if active_count > 0:
            self.logger.info(f"Shutting down {active_count} active threads...")

            if hasattr(self, 'server') and self.server:
                try:
                    self.server.shutdown()
                    self.server = None
                    self.logger.debug("Dash server shutdown")
                except Exception as e:
                    self.logger.debug(f"Error shutting down server: {e}")

            with self.thread_lock:
                threads_to_join = self.active_threads.copy()

            for thread in threads_to_join:
                if thread.is_alive():
                    try:
                        thread.join(timeout=1.0)
                        if thread.is_alive():
                            self.logger.debug(f"Thread {thread.name} did not join within timeout")
                        else:
                            self.logger.debug(f"Thread {thread.name} joined successfully")
                    except Exception as e:
                        self.logger.debug(f"Error joining thread {thread.name}: {e}")

            with self.thread_lock:
                remaining_threads = [t for t in self.active_threads if t.is_alive()]

            if remaining_threads:
                self.logger.info(f"Force terminating {len(remaining_threads)} stubborn threads")
                for thread in remaining_threads:
                    if thread.is_alive() and hasattr(thread, 'ident') and thread.ident:
                        try:
                            import ctypes
                            res = ctypes.pythonapi.PyThreadState_SetAsyncExc(
                                ctypes.c_long(thread.ident),
                                ctypes.py_object(SystemExit)
                            )
                            if res > 1:
                                ctypes.pythonapi.PyThreadState_SetAsyncExc(thread.ident, None)
                            self.logger.debug(f"Force terminated thread {thread.name}")
                        except Exception as e:
                            self.logger.debug(f"Could not force terminate thread {thread.name}: {e}")
                            try:
                                thread._stop()
                            except:
                                pass

            with self.thread_lock:
                self.active_threads.clear()

            self.logger.info("Thread shutdown completed")

    def reset_edit_type(self):
        """Reset the edit type to 'None' and update the GUI if available"""
        self.edit_type = "None"
        if self.reset_edit_type_gui:
            self.reset_edit_type_gui()
