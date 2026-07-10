import logging
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import os
import traceback
from matplotlib.collections import LineCollection
from moove.qt_helpers import show_info
from moove.utils.audio_utils import (decibel)
from moove.utils.movefuncs_utils import (load_recfile, ensure_recfile_exists_and_has_flags)

plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 14,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14
})


_AMP_ENVELOPE_BUCKETS = 4000


def plot_amplitude_line(ax3, display_dict):
    """Plot the amplitude trace, min/max-decimated for large signals.

    A full song has hundreds of thousands of samples but the plot is only ~1-2k
    pixels wide, so drawing every sample is wasted work. We collapse the signal
    into ~4000 buckets and draw each bucket's min and max, which keeps every
    peak/trough visible (important for reading the threshold) while making the
    line cheap to redraw."""
    amp = np.asarray(display_dict["amplitude"], dtype=float)
    sr = display_dict["sampling_rate"]
    n = amp.shape[0]
    x = np.arange(n) / sr
    if n > _AMP_ENVELOPE_BUCKETS * 2:
        step = n // _AMP_ENVELOPE_BUCKETS
        trimmed = (n // step) * step
        a = amp[:trimmed].reshape(-1, step)
        xb = x[:trimmed:step]
        mn = a.min(axis=1)
        mx = a.max(axis=1)
        xs = np.repeat(xb, 2)
        ys = np.empty(xs.shape[0], dtype=float)
        ys[0::2] = mn
        ys[1::2] = mx
        ax3.plot(xs, ys, color='#2653c5')
    else:
        ax3.plot(x, amp, color='#2653c5')


def draw_segment_markers(ax3, display_dict, app_state):
    """Draw onset/offset bars + boundary markers at the threshold height.

    Uses one LineCollection for all bars and one Line2D for all '+' markers
    (2 artists total) instead of ~3 artists per syllable, so the canvas redraws
    far faster during zoom/pan when many syllables are present."""
    onsets = display_dict.get("onsets")
    offsets = display_dict.get("offsets")
    if onsets is None or offsets is None:
        return
    n = min(len(onsets), len(offsets))
    if n == 0:
        return
    try:
        height = float(app_state.evfuncs_params['threshold'].get())
    except (ValueError, KeyError, TypeError):
        return
    on = np.asarray(onsets[:n], dtype=float) / 1000.0
    off = np.asarray(offsets[:n], dtype=float) / 1000.0
    bars = [[(on[i], height), (off[i], height)] for i in range(n)]
    ax3.add_collection(LineCollection(bars, colors='black', linewidths=1.5))
    xs = np.concatenate([on, off])
    ax3.plot(xs, np.full(xs.shape, height), linestyle='None', marker='+',
             color='black', markersize=10, markeredgewidth=1.5)


def draw_threshold_line(ax3, app_state):
    """Draw the current segmentation threshold as a full-width line on ax3.

    Lets the user see where the threshold sits against the amplitude trace
    (the hover readout helps pick a new value to set it to)."""
    try:
        thr = float(app_state.evfuncs_params['threshold'].get())
    except (ValueError, KeyError, TypeError):
        return
    ax3.axhline(thr, color='#d62728', linestyle='--', linewidth=1.0, alpha=0.8)


def update_plots(display_dict, app_state, filepath):
    """Render a newly loaded file. Delegates to the pyqtgraph canvas."""
    vmin, vmax = app_state.current_vmin, app_state.current_vmax
    app_state.canvas.render(display_dict, app_state, vmin, vmax)


def update_ax2_ax3(ax2, ax3, display_dict, app_state):
    """Redraw amplitude + labels + segments after an edit (keeps x range)."""
    app_state.canvas.update_amp_labels(display_dict, app_state)


def update_ax2(ax2, display_dict, app_state):
    """Redraw only the syllable labels."""
    app_state.canvas._draw_labels(display_dict)


def plot_data(app_state):
    """Plot new data and update the application state."""
    from moove.utils.file_utils import get_file_data_by_index, get_display_data

    def _load_checkbox_flags(rec_path):
        try:
            rec_data = load_recfile(rec_path)
            return int(rec_data.get("hand_segmented", 0)), int(rec_data.get("hand_classified", 0))
        except Exception as exc:
            app_state.logger.warning("Could not load recfile '%s': %s. Falling back to unchecked state.", rec_path, exc)
            return 0, 0

    def _sync_checkbox_widgets(segmented, classified):
        if app_state.segmented_checkbox is not None:
            # app_state.segmented_checkbox.blockSignals(True)
            app_state.segmented_checkbox.setChecked(bool(segmented))
            # app_state.segmented_checkbox.blockSignals(False)
        if app_state.classified_checkbox is not None:
            # app_state.classified_checkbox.blockSignals(True)
            app_state.classified_checkbox.setChecked(bool(classified))
            # app_state.classified_checkbox.blockSignals(False)

    try:
        file_path = get_file_data_by_index(app_state.data_dir, app_state.song_files, app_state.current_file_index, app_state)
        ensure_recfile_exists_and_has_flags(file_path["file_path"])
        app_state.display_dict = get_display_data(file_path, app_state.config)
        from moove.utils.syllable_utils import init_seg_history
        init_seg_history(app_state)
        app_state._file_skip_attempted = False
        app_state._startup_default_attempted = False
        app_state._default_first_attempted = False
        
        update_plots(app_state.display_dict, app_state, file_path)
        ax1, ax2, ax3 = app_state.get_axes()
        ax1.set_navigate(False)

        # Set original axis ranges for zooming and unzooming
        original_x_range = (ax1.get_xlim()[0], ax1.get_xlim()[1])
        app_state.set_original_x_range(original_x_range)
        original_y_range_ax1 = (ax1.get_ylim()[0], ax1.get_ylim()[1])
        original_y_range_ax2 = (ax2.get_ylim()[0], ax2.get_ylim()[1])
        original_y_range_ax3 = (ax3.get_ylim()[0], ax3.get_ylim()[1])
        app_state.set_original_y_range_ax1(original_y_range_ax1, original_y_range_ax2, original_y_range_ax3)

        # Load and set segmented/classified state for checkboxes.
        hand_segmented, hand_classified = _load_checkbox_flags(os.path.splitext(file_path["file_path"])[0] + ".rec")
        app_state.segmented_var.set(str(hand_segmented))
        app_state.classified_var.set(str(hand_classified))
        _sync_checkbox_widgets(hand_segmented, hand_classified)
        app_state.edit_type = "None"

        app_state.logger.debug("Recfile loaded and checkboxes updated for file: %s", file_path["file_name"])

        # Store the last valid file path for fallback
        if hasattr(app_state, 'last_valid_file_path'):
            app_state.last_valid_file_path = file_path["file_path"]
        else:
            app_state.last_valid_file_path = file_path["file_path"]
        
        app_state.reset_edit_type_gui()

        app_state.draw_canvas()
        
    except Exception as exc:
        # Log the REAL error so it is visible in the console / log file.
        app_state.logger.error("plot_data failed: %s", exc)
        app_state.logger.debug("Full traceback:\n%s", traceback.format_exc())

        is_missing_file = isinstance(exc, FileNotFoundError) or "does not exist" in str(exc)

        # Startup context: if stored file is missing, default to first file of current day.
        if is_missing_file and not app_state.init_flag and not getattr(app_state, "_startup_default_attempted", False):
            app_state._startup_default_attempted = True
            try:
                existing_files = []
                for name in app_state.song_files:
                    candidate = os.path.join(app_state.data_dir, name)
                    if not os.path.isabs(candidate):
                        candidate = os.path.join(os.getcwd(), candidate)
                    if os.path.exists(candidate):
                        existing_files.append(name)

                if existing_files:
                    app_state.song_files = existing_files
                    app_state.current_file_index = 0
                    if app_state.combobox is not None:
                        app_state.combobox.blockSignals(True)
                        app_state.combobox.clear()
                        app_state.combobox.addItems(app_state.song_files)
                        app_state.combobox.setCurrentIndex(0)
                        app_state.combobox.blockSignals(False)
                    show_info(None, "Warning",
                              "Das zuletzt geoeffnete File wurde nicht gefunden. "
                              "Es wird mit dem ersten verfuegbaren File fortgefahren.")
                    print(f"Startup fallback: defaulted to first existing batch file: {app_state.song_files[0]}")
                    plot_data(app_state)
                    return
            except Exception as startup_error:
                app_state.logger.error("Startup default fallback failed: %s", startup_error)

        # File navigation context: if previous/next failed, move one more in same direction.
        nav_delta = getattr(app_state, "last_file_delta", 0)
        if is_missing_file and app_state.init_flag and nav_delta in (-1, 1) and not getattr(app_state, "_file_skip_attempted", False):
            app_state._file_skip_attempted = True
            try:
                app_state.change_file(nav_delta)
                plot_data(app_state)
                return
            except Exception as nav_error:
                app_state.logger.error("Navigation fallback failed: %s", nav_error)

        # If there's an error with the current file, try to fall back to the last valid file
        if hasattr(app_state, 'last_valid_file_path') and app_state.last_valid_file_path:
            try:
                # Try to plot the last valid file instead
                fallback_file_data = {"file_name": os.path.basename(app_state.last_valid_file_path), 
                                    "file_path": app_state.last_valid_file_path}
                ensure_recfile_exists_and_has_flags(fallback_file_data["file_path"])
                app_state.display_dict = get_display_data(fallback_file_data, app_state.config)
                
                update_plots(app_state.display_dict, app_state, fallback_file_data)
                ax1, ax2, ax3 = app_state.get_axes()
                ax1.set_navigate(False)

                # Set original axis ranges for zooming and unzooming
                original_x_range = (ax1.get_xlim()[0], ax1.get_xlim()[1])
                app_state.set_original_x_range(original_x_range)
                original_y_range_ax1 = (ax1.get_ylim()[0], ax1.get_ylim()[1])
                original_y_range_ax2 = (ax2.get_ylim()[0], ax2.get_ylim()[1])
                original_y_range_ax3 = (ax3.get_ylim()[0], ax3.get_ylim()[1])
                app_state.set_original_y_range_ax1(original_y_range_ax1, original_y_range_ax2, original_y_range_ax3)

                # Load and set segmented/classified state for checkboxes.
                hand_segmented, hand_classified = _load_checkbox_flags(os.path.splitext(fallback_file_data["file_path"])[0] + ".rec")
                app_state.segmented_var.set(str(hand_segmented))
                app_state.classified_var.set(str(hand_classified))
                _sync_checkbox_widgets(hand_segmented, hand_classified)
                app_state.edit_type = "None"

                app_state.reset_edit_type_gui()

                app_state.logger.debug("Successfully fell back to last valid file: %s", fallback_file_data["file_name"])
                app_state.draw_canvas()
                
            except Exception as fallback_error:
                app_state.logger.error("Fallback plot also failed: %s", fallback_error)
        else:
            app_state.logger.error("No fallback file available.")
            app_state.segmented_var.set("0")
            app_state.classified_var.set("0")
            _sync_checkbox_widgets(0, 0)
