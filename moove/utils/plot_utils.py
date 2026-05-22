import logging
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import os
import traceback
from moove.qt_helpers import show_info
from moove.utils.audio_utils import (decibel)
from moove.utils.movefuncs_utils import (load_recfile, ensure_recfile_exists_and_has_flags)

plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 14,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14
})


def update_plots(display_dict, app_state, filepath):
    """Update plots with new display data."""
    ax1, ax2, ax3 = app_state.get_axes()

    for ax in (ax1, ax2, ax3): 
        ax.tick_params(axis='both', colors=app_state.text_color) 

    plt.tight_layout() 
    plt.subplots_adjust(left=0.129, right=0.999, top=0.92, bottom=0.1, hspace=0.1) 

    vmin, vmax = app_state.current_vmin, app_state.current_vmax

    # Filter frequencies
    freqs = display_dict["freqs"]
    valid_freqs = (freqs >= app_state.config['lower_spec_plot']) & (freqs <= app_state.config['upper_spec_plot'])
    filtered_freqs = freqs[valid_freqs]
    filtered_spectrogram_data = display_dict["spectrogram_data"][valid_freqs, :]
    # calculates the power into amplitudes
    amplitude_spec = np.sqrt(filtered_spectrogram_data)
    # amplitude -> dB
    db_spec = decibel(amplitude_spec)

    feedback_time = []
    # Load info from .rec file
    catch = load_recfile(os.path.splitext(filepath["file_path"])[0] + ".rec")["catch_song"]
    feedback_time = load_recfile(os.path.splitext(filepath["file_path"])[0] + ".rec")["feedback_info"]

    # Update ax1 (Spectrogram)
    ax1.clear()

    performance_mode = app_state.config['performance']
    
    if performance_mode == "fast":
        # faster, less detailed drawings
        extent = [display_dict["times"].min(), display_dict["times"].max(),
                filtered_freqs.min(), filtered_freqs.max()]

        app_state.spec = ax1.imshow(db_spec, aspect='auto', origin='lower',
                                    extent=extent, cmap='jet', vmin=vmin, vmax=vmax,
                                    interpolation='none')
    else:
        # slower, more detailed drawing
        app_state.spec = ax1.pcolormesh(display_dict["times"], filtered_freqs, db_spec,
                                        shading='gouraud', cmap='jet', vmin=vmin, vmax=vmax)

    # get axis position in figure coordinates
    pos = ax2.get_position()
    fig = ax2.figure
    # remove all previous figure level text
    for txt in list(fig.texts):
        txt.remove()
    fig.patches = [p for p in fig.patches if not (isinstance(p, patches.Rectangle) and getattr(p, 'is_box', False))]

    # Mark catch trials in GUI
    if catch == 1:
        ax1.text(0.0, 1.0, "catch", color='crimson', fontsize=17, transform=ax1.transAxes)
        rect = patches.Rectangle((pos.x0, pos.y0), pos.width, pos.height, linewidth=1,
                                 edgecolor='crimson', facecolor='none', transform=fig.transFigure, zorder=100)
        rect.is_box = True
        fig.patches.append(rect)

    # Mark feedback timings
    feedback_timings = []
    if feedback_time:
        for i in range(len(feedback_time)):
            feedback_timings.append(feedback_time[i][0])
        for t in feedback_timings:
            ax1.text(t, -0.01, '^', ha='center', va='top', color='red', fontsize=24,
                     transform=ax1.get_xaxis_transform(), zorder=1000000)

    ax1.set_title(display_dict["file_name"], color=app_state.text_color)
    ax1.set_ylabel('Frequency (Hz)', color=app_state.text_color)

    # Update ax2 (Label visualization)
    ax2.clear()
    if "labels" in display_dict:
        min_len = min(len(display_dict["onsets"]), len(display_dict["offsets"]), len(display_dict["labels"]))
        for i, label in zip(range(min_len), display_dict["labels"]):
            label_position = (display_dict["onsets"][i] + display_dict["offsets"][i]) / (2 * 1000)
            ax2.text(label_position, 0.5, label, ha='center', va='center', clip_on=True)

        app_state.logger.debug("display_dict onsets: %s", display_dict["onsets"])
        app_state.logger.debug("display_dict offsets: %s", display_dict["offsets"])
        app_state.logger.debug("display_dict labels: %s", display_dict["labels"])

    ax2.set_yticks([])
    ax2.set_xlim(display_dict["times"][0], display_dict["times"][-1])

    # Update ax3 (Amplitude plot)
    ax3.clear()
    ax3.plot(np.arange(len(display_dict["song_data"])) / display_dict["sampling_rate"], display_dict["amplitude"],
             color='#2653c5')
    ax3.set_ylabel('Amplitude (dB)', color=app_state.text_color)
    ax3.set_xlabel('Time (s)', color=app_state.text_color)
    ax3.set_ylim(display_dict["amplitude"].min(), display_dict["amplitude"].max())

    # Mark onsets and offsets
    if "onsets" in display_dict and "offsets" in display_dict:
        bar_height = float(app_state.evfuncs_params['threshold'].get())
        for onset, offset in zip(display_dict["onsets"], display_dict["offsets"]):
            onset_sec = onset / 1000
            offset_sec = offset / 1000
            ax3.hlines(bar_height, onset_sec, offset_sec, color='black', linewidth=1.5)
            ax3.plot(onset_sec, bar_height, marker='+', color='black', markersize=10, markeredgewidth=1.5)
            ax3.plot(offset_sec, bar_height, marker='+', color='black', markersize=10, markeredgewidth=1.5)

    app_state.set_axes(ax1, ax2, ax3)

    fig.patch.set_facecolor(app_state.bg_color) 

    fig.align_ylabels() 
    app_state.draw_canvas()


def update_ax2_ax3(ax2, ax3, display_dict, app_state):
    """Update the second and third axes with new display data."""
    app_state.logger.debug("Updating ax2 and ax3")

    x_lim = ax2.get_xlim()
    app_state.logger.debug("Current x-axis limits: %s", x_lim)

    ax2.clear()
    # Update ax2 with labels
    if "labels" in display_dict:
        min_len = min(len(display_dict["onsets"]), len(display_dict["offsets"]), len(display_dict["labels"]))
        for i, label in zip(range(min_len), display_dict["labels"]):
            label_position = (display_dict["onsets"][i] + display_dict["offsets"][i]) / (2 * 1000)
            ax2.text(label_position, 0.5, label, ha='center', va='center', clip_on=True)
    ax2.set_yticks([])
    ax2.set_xlim(x_lim)

    ax3.clear()
    ax3.plot(np.arange(len(display_dict["song_data"])) / display_dict["sampling_rate"], display_dict["amplitude"],
             color='#2653c5')
    ax3.set_ylabel('Amplitude (dB)', color=app_state.text_color)
    ax3.set_xlabel('Time (s)', color=app_state.text_color)
    ax3.set_ylim(display_dict["amplitude"].min(), display_dict["amplitude"].max())
    ax3.set_xlim(x_lim)

    # Mark onsets and offsets
    if "onsets" in display_dict and "offsets" in display_dict:
        bar_height = float(app_state.evfuncs_params['threshold'].get())
        for onset, offset in zip(display_dict["onsets"], display_dict["offsets"]):
            onset_sec = onset / 1000
            offset_sec = offset / 1000
            ax3.hlines(bar_height, onset_sec, offset_sec, color='black', linewidth=1.5)
            ax3.plot(onset_sec, bar_height, marker='+', color='black', markersize=10, markeredgewidth=1.5)
            ax3.plot(offset_sec, bar_height, marker='+', color='black', markersize=10, markeredgewidth=1.5)

    app_state.draw_canvas()


def update_ax2(ax2, display_dict, app_state):
    """Update the second axis with new display data."""
    x_lim = ax2.get_xlim()
    y_lim = ax2.get_ylim()
    ax2.clear()
    ax2.set_yticks([])
    ax2.set_xlim(x_lim)

    # Cover up old labels temporarily
    ax2.add_patch(
        plt.Rectangle(
            (x_lim[0], y_lim[0]),
            x_lim[1] - x_lim[0],
            y_lim[1] - y_lim[0],
            color='white', zorder=0
        )
    )

    # write new labels to axis
    if "labels" in display_dict:
        min_len = min(len(display_dict["onsets"]), len(display_dict["offsets"]), len(display_dict["labels"]))
        for i in range(min_len):
            label_position = (display_dict["onsets"][i] + display_dict["offsets"][i]) / 2000
            ax2.text(label_position, 0.5, display_dict["labels"][i], ha='center', va='center', clip_on=True, zorder=1)

    for artist in ax2.texts + ax2.patches:
        ax2.draw_artist(artist)
    app_state.canvas.blit(ax2.bbox)


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
