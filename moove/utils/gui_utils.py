# utils/gui_utils.py
import os

from moove.qt_helpers import set_combo_items


def zoom(app_state):
    """Zoom in by reducing the x-axis range by 30%."""
    x_start, x_end = app_state.ax1.get_xlim()
    x_center = (x_start + x_end) / 2
    x_diff = x_end - x_start
    new_diff = x_diff * 0.7

    app_state.ax1.set_xlim(x_center - new_diff / 2, x_center + new_diff / 2)
    app_state.ax2.set_xlim(x_center - new_diff / 2, x_center + new_diff / 2)
    app_state.ax3.set_xlim(x_center - new_diff / 2, x_center + new_diff / 2)

    app_state.logger.debug("Zoomed in to new x-axis range: (%f, %f)", x_center - new_diff / 2, x_center + new_diff / 2)
    app_state.draw_canvas()


def unzoom(app_state):
    """Reset zoom to the original x and y axis ranges."""
    app_state.ax1.set_xlim(app_state.original_x_range[0], app_state.original_x_range[1])
    app_state.ax1.set_ylim(app_state.original_y_range_ax1[0], app_state.original_y_range_ax1[1])

    app_state.ax2.set_xlim(app_state.original_x_range[0], app_state.original_x_range[1])
    app_state.ax2.set_ylim(app_state.original_y_range_ax2[0], app_state.original_y_range_ax2[1])

    app_state.ax3.set_xlim(app_state.original_x_range[0], app_state.original_x_range[1])
    app_state.ax3.set_ylim(app_state.original_y_range_ax3[0], app_state.original_y_range_ax3[1])

    app_state.logger.debug("Reset zoom to original ranges.")
    app_state.draw_canvas()


def unzoom_small(app_state):
    """Zoom out by inreasing the x-axis range by 30%."""
    x_start, x_end = app_state.ax1.get_xlim()
    x_center = (x_start + x_end) / 2
    x_diff = x_end - x_start
    new_diff = x_diff * 1.3

    app_state.ax1.set_xlim(x_center - new_diff / 2, x_center + new_diff / 2)
    app_state.ax2.set_xlim(x_center - new_diff / 2, x_center + new_diff / 2)
    app_state.ax3.set_xlim(x_center - new_diff / 2, x_center + new_diff / 2)

    app_state.logger.debug("Zoomed in to new x-axis range: (%f, %f)", x_center - new_diff / 2, x_center + new_diff / 2)
    app_state.draw_canvas()


def swipe_left(app_state):
    """Swipe view to the left by moving the x-axis range left by 10%."""
    x_start, x_end = app_state.ax1.get_xlim()
    x_diff = x_end - x_start
    new_start = x_start - x_diff * 0.9
    new_end = x_end - x_diff * 0.9

    app_state.ax1.set_xlim(new_start, new_end)
    app_state.ax2.set_xlim(new_start, new_end)
    app_state.ax3.set_xlim(new_start, new_end)

    app_state.logger.debug("Swiped left to new x-axis range: (%f, %f)", new_start, new_end)
    app_state.draw_canvas()


def swipe_right(app_state):
    """Swipe view to the right by moving the x-axis range right by 10%."""
    x_start, x_end = app_state.ax1.get_xlim()
    x_diff = x_end - x_start
    new_start = x_start + x_diff * 0.9
    new_end = x_end + x_diff * 0.9

    app_state.ax1.set_xlim(new_start, new_end)
    app_state.ax2.set_xlim(new_start, new_end)
    app_state.ax3.set_xlim(new_start, new_end)

    app_state.logger.debug("Swiped right to new x-axis range: (%f, %f)", new_start, new_end)
    app_state.draw_canvas()


def update(app_state):
    """Update all batch files and update the GUI."""
    from moove.utils.file_utils import find_batch_files, read_batch, create_batch_file
    from moove.utils.plot_utils import plot_data

    previous_batch = app_state.current_batch_file
    previous_file = None
    if app_state.song_files and app_state.current_file_index is not None and app_state.current_file_index < len(app_state.song_files):
        previous_file = app_state.song_files[app_state.current_file_index]

    batch_files = find_batch_files(app_state.data_dir)
    valid_files = [f for f in os.listdir(app_state.data_dir) if f.endswith('.wav') or f.endswith('.cbin')]

    if not batch_files:
        create_batch_file(os.path.join(app_state.data_dir))
    for batch in batch_files:
        batch_path = os.path.join(app_state.data_dir, batch)
        if batch == 'batch.txt':
            with open(batch_path, 'w') as f:
                f.write('\n'.join(valid_files))
        else:
            with open(batch_path, 'r') as batch_open:
                keep_files = batch_open.read().splitlines()
            filtered_files = [f for f in keep_files if f in valid_files]
            with open(batch_path, 'w') as f:
                f.write('\n'.join(filtered_files))

    app_state.logger.info("Batch files have been updated.")

    if previous_batch in batch_files:
        app_state.current_batch_file = previous_batch
    elif "batch.txt" in batch_files:
        app_state.current_batch_file = "batch.txt"
    else:
        app_state.current_batch_file = batch_files[0] if batch_files else ""

    set_combo_items(app_state.batch_combobox, batch_files, app_state.current_batch_file)
    app_state.song_files = read_batch(app_state.data_dir, app_state.current_batch_file) if app_state.current_batch_file else []

    if previous_file in app_state.song_files:
        app_state.current_file_index = app_state.song_files.index(previous_file)
    else:
        app_state.current_file_index = 0 if app_state.song_files else None

    set_combo_items(app_state.combobox, app_state.song_files,
                    app_state.song_files[app_state.current_file_index] if app_state.song_files else "")

    if app_state.song_files:
        plot_data(app_state)
        app_state.logger.info("Plots have been updated.")
    else:
        for ax in [app_state.ax1, app_state.ax2, app_state.ax3]:
            ax.clear()
        app_state.draw_canvas()
