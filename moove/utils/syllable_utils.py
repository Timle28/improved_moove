# utils/syllable_utils.py
import numpy as np
import os
from matplotlib.patches import Rectangle

_SEG_HISTORY_LIMIT = 200


def _seg_snapshot(display_dict):
    """Copy the current segmentation (onsets/offsets/labels) for the history."""
    return {
        'onsets': np.array(display_dict.get('onsets', []), copy=True),
        'offsets': np.array(display_dict.get('offsets', []), copy=True),
        'labels': str(display_dict.get('labels', '')),
    }


def init_seg_history(app_state):
    """Start a fresh undo/redo history for the currently loaded file."""
    dd = app_state.display_dict
    if dd is None or 'onsets' not in dd:
        app_state.seg_history = []
        app_state.seg_history_index = -1
        return
    app_state.seg_history = [_seg_snapshot(dd)]
    app_state.seg_history_index = 0


def record_seg_state(app_state):
    """Push the current segmentation onto the history (call after each edit)."""
    dd = app_state.display_dict
    if dd is None:
        return
    if app_state.seg_history_index < 0:
        init_seg_history(app_state)
        return
    # Drop any redo branch, then append the new state.
    history = app_state.seg_history[:app_state.seg_history_index + 1]
    history.append(_seg_snapshot(dd))
    if len(history) > _SEG_HISTORY_LIMIT:
        history = history[-_SEG_HISTORY_LIMIT:]
    app_state.seg_history = history
    app_state.seg_history_index = len(history) - 1


def _apply_seg_snapshot(app_state, snap):
    from moove.utils.plot_utils import update_ax2_ax3
    from moove.utils import save_notmat

    dd = app_state.display_dict
    if dd is None:
        return
    dd['onsets'] = np.array(snap['onsets'], copy=True)
    dd['offsets'] = np.array(snap['offsets'], copy=True)
    dd['labels'] = snap['labels']
    # Selection may now point past the end; clear it to stay safe.
    app_state.selected_syllable_index = None
    save_notmat(os.path.join(app_state.data_dir, f"{dd['file_name']}.not.mat"), dd)
    update_ax2_ax3(app_state.ax2, app_state.ax3, dd, app_state)


def undo_segmentation(app_state):
    """Revert to the previous segmentation state (Ctrl/Cmd+Z)."""
    if app_state.seg_history_index <= 0:
        app_state.logger.debug("Undo: nothing to undo.")
        return
    app_state.seg_history_index -= 1
    _apply_seg_snapshot(app_state, app_state.seg_history[app_state.seg_history_index])
    app_state.logger.debug("Undo -> history index %d", app_state.seg_history_index)


def redo_segmentation(app_state):
    """Re-apply the next segmentation state (Ctrl/Cmd+Shift+Z)."""
    if app_state.seg_history_index >= len(app_state.seg_history) - 1:
        app_state.logger.debug("Redo: nothing to redo.")
        return
    app_state.seg_history_index += 1
    _apply_seg_snapshot(app_state, app_state.seg_history[app_state.seg_history_index])
    app_state.logger.debug("Redo -> history index %d", app_state.seg_history_index)


def set_threshold_from_click(event, app_state):
    """Double-click on the amplitude plot to set the segmentation threshold."""
    from moove.utils.plot_utils import update_ax2_ax3

    if not getattr(event, 'dblclick', False):
        return
    if event.inaxes is not app_state.ax3 or event.ydata is None:
        return
    value = round(float(event.ydata), 1)
    try:
        app_state.evfuncs_params['threshold'].set(str(value))
    except (KeyError, AttributeError):
        return
    if app_state.display_dict is not None:
        update_ax2_ax3(app_state.ax2, app_state.ax3, app_state.display_dict, app_state)
    app_state.logger.info("Threshold set to %.1f dB via double-click.", value)


def add_new_segment(event, app_state):
    """Add a new segment based on user input from a mouse event."""
    from moove.utils.plot_utils import update_ax2_ax3
    from moove.utils import save_notmat

    display_dict = app_state.display_dict
    if display_dict is None:
        return
    
    if event.inaxes in {app_state.ax3, app_state.ax1} and event.button == 1:  # Left-click
        app_state.new_onset = event.xdata * 1000  # Convert to milliseconds
    elif event.inaxes in {app_state.ax3, app_state.ax1} and event.button == 3 and app_state.new_onset:  # Right-click
        new_offset = event.xdata * 1000

        # Ensure new onset is before new offset
        if app_state.new_onset >= new_offset:
            app_state.new_onset = None
            return

        onsets, offsets = display_dict["onsets"], display_dict["offsets"]

        # Check if the new segment overlaps with existing segments
        if any((app_state.new_onset < onset < new_offset) or (app_state.new_onset > onset and app_state.new_onset < offset) for onset, offset in zip(onsets, offsets)):
            app_state.new_onset = None
            return

        onsets = np.append(onsets, app_state.new_onset)
        offsets = np.append(offsets, new_offset)
        onsets.sort()
        offsets.sort()
        display_dict["onsets"], display_dict["offsets"] = onsets, offsets

        labels = list(display_dict["labels"])
        onset_index = np.where(onsets == app_state.new_onset)[0][0]
        new_labels = labels[:onset_index] + ["x"] + labels[onset_index:]
        display_dict["labels"] = ''.join(new_labels)

        app_state.new_onset = None
        save_notmat(os.path.join(app_state.data_dir, f"{display_dict['file_name']}.not.mat"), display_dict)
        record_seg_state(app_state)
        update_ax2_ax3(app_state.ax2, app_state.ax3, display_dict, app_state)

    else:
        if app_state.new_onset:
            app_state.new_onset = None


def _label_index_at(app_state, x_sec):
    """Index of the syllable whose [onset, offset] spans x_sec, else nearest."""
    dd = app_state.display_dict
    if dd is None:
        return None
    on = np.asarray(dd.get("onsets", []), dtype=float) / 1000.0
    off = np.asarray(dd.get("offsets", []), dtype=float) / 1000.0
    n = min(len(on), len(off))
    if n == 0:
        return None
    for i in range(n):
        if on[i] <= x_sec <= off[i]:
            return i
    centers = (on[:n] + off[:n]) / 2.0
    return int(np.argmin(np.abs(centers - x_sec)))


def select_event(event, app_state):
    """Handle syllable selection based on event type."""
    if app_state.edit_type == "Label Interactive" and event.inaxes == app_state.ax2:
        idx = _label_index_at(app_state, event.xdata)
        if idx is not None:
            highlight_syllable(idx, app_state)
    elif app_state.edit_type == "New Segment":
        add_new_segment(event, app_state)
    elif app_state.edit_type == "Delete Segment":
        delete_segment(event, app_state)
    elif app_state.edit_type == "Move Segment":
        move_segment(event, app_state)


def _redraw_ax2_labels(app_state):
    """Label color/text changes auto-render with pyqtgraph; nothing to do."""
    return


def highlight_syllable(idx, app_state):
    """Highlight the selected syllable label in red."""
    canvas = app_state.canvas
    n = canvas.label_count()
    if idx < 0 or idx >= n:
        return

    prev_idx = app_state.selected_syllable_index
    if prev_idx is not None and 0 <= prev_idx < n:
        canvas.set_label_color(prev_idx, '#000000')

    app_state.selected_syllable_index = idx
    canvas.set_label_color(idx, '#d62728')


def edit_syllable(event, app_state):
    """Edit the selected syllable label."""
    from moove.utils import save_notmat
    import os

    display_dict = app_state.display_dict
    if app_state.edit_type == "None":
        return

    num_labels = len(display_dict["labels"]) if display_dict else 0

    if event.key in ('left', 'right') and num_labels > 0:
        if app_state.selected_syllable_index is None:
            new_idx = 0 if event.key == 'right' else num_labels - 1
        elif event.key == 'left':
            new_idx = (app_state.selected_syllable_index - 1) % num_labels
        else:
            new_idx = (app_state.selected_syllable_index + 1) % num_labels
        highlight_syllable(new_idx, app_state)
        return

    if app_state.selected_syllable_index is None or num_labels == 0:
        return

    if len(event.key) == 1 and ((event.key.isalpha() and event.key.islower()) or event.key.isdigit()):
        labels = list(display_dict["labels"])
        labels[app_state.selected_syllable_index] = event.key
        display_dict["labels"] = ''.join(labels)

        # Update only the changed label item to keep interaction responsive.
        app_state.canvas.set_label_text(app_state.selected_syllable_index, event.key)

        save_notmat(
            os.path.join(app_state.data_dir, f"{display_dict['file_name']}.not.mat"),
            display_dict
        )
        record_seg_state(app_state)
        next_idx = (app_state.selected_syllable_index + 1) % num_labels
        highlight_syllable(next_idx, app_state)


def delete_segment(event, app_state):
    """Delete a segment based on the user's click event."""
    from moove.utils.plot_utils import update_ax2_ax3
    from moove.utils import save_notmat

    display_dict = app_state.display_dict
    if display_dict is None:
        return

    onsets, offsets = display_dict["onsets"], display_dict["offsets"]
    if event.xdata:
        delete_point = event.xdata * 1000  # Convert to milliseconds
    else:
        return

    for i in range(len(onsets)):
        if onsets[i] <= delete_point <= offsets[i]:  # Check if click is within segment range
            # Remove the segment and corresponding label
            onsets = np.delete(onsets, i)
            offsets = np.delete(offsets, i)
            labels = list(display_dict["labels"])
            del labels[i]
            display_dict["labels"], display_dict["onsets"], display_dict["offsets"] = ''.join(labels), onsets, offsets

            save_notmat(os.path.join(app_state.data_dir, f"{display_dict['file_name']}.not.mat"), display_dict)
            record_seg_state(app_state)
            update_ax2_ax3(app_state.ax2, app_state.ax3, display_dict, app_state)
            break


def move_segment(event, app_state):
    """Move an onset or offset marker based on a click event."""
    from moove.utils.plot_utils import update_ax2_ax3
    from moove.utils import save_notmat

    display_dict = app_state.display_dict
    if display_dict is None:
        return

    onsets, offsets = display_dict["onsets"], display_dict["offsets"]

    if not app_state.moved_point:
        if event.inaxes == app_state.ax3 and event.button == 1:  # Left-click
            click_x = event.xdata
            tolerance = 0.1

            closest_marker, min_distance, marker_type, marker_index = None, float('inf'), None, None

            # Check onsets
            for i, onset in enumerate(onsets):
                onset_sec = onset / 1000
                distance = abs(click_x - onset_sec)
                if distance < min_distance and distance < tolerance:
                    min_distance = distance
                    closest_marker, marker_type, marker_index = onset_sec, 'onset', i

            # Check offsets
            for i, offset in enumerate(offsets):
                offset_sec = offset / 1000
                distance = abs(click_x - offset_sec)
                if distance < min_distance and distance < tolerance:
                    min_distance = distance
                    closest_marker, marker_type, marker_index = offset_sec, 'offset', i

            if closest_marker is not None:
                app_state.moved_point = (marker_type, marker_index)
                marker_height = float(app_state.evfuncs_params['threshold'].get())
                app_state.canvas.mark_selected(closest_marker, marker_height)

    else:
        if event.inaxes == app_state.ax3 and event.button == 3:  # Right-click
            new_x = event.xdata
            marker_type, marker_index = app_state.moved_point

            # Validation for moving onset and offset markers
            if marker_type == 'onset' and not valid_move(new_x, marker_index, onsets, offsets, True):
                return
            elif marker_type == 'offset' and not valid_move(new_x, marker_index, onsets, offsets, False):
                return

            if marker_type == 'onset':
                onsets[marker_index] = new_x * 1000
            elif marker_type == 'offset':
                offsets[marker_index] = new_x * 1000

            display_dict["onsets"], display_dict["offsets"] = onsets, offsets
            app_state.moved_point = None
            save_notmat(os.path.join(app_state.data_dir, f"{display_dict['file_name']}.not.mat"), display_dict)
            record_seg_state(app_state)
            update_ax2_ax3(app_state.ax2, app_state.ax3, display_dict, app_state)
        else:
            app_state.moved_point = None
            update_ax2_ax3(app_state.ax2, app_state.ax3, display_dict, app_state)


def valid_move(new_x, marker_index, onsets, offsets, is_onset):
    """Check if a move is valid based on current onsets and offsets."""
    if is_onset:
        if 0 < marker_index < len(onsets) - 1 and not (offsets[marker_index - 1] / 1000 < new_x < offsets[marker_index] / 1000):
            return False
        if marker_index == 0 and new_x >= (offsets[marker_index] / 1000):
            return False
        if marker_index == len(onsets) - 1 and new_x <= (offsets[marker_index - 1] / 1000):
            return False
    else:
        if 0 < marker_index < len(offsets) - 1 and not (onsets[marker_index] / 1000 < new_x < onsets[marker_index + 1] / 1000):
            return False
        if marker_index == 0 and new_x >= (onsets[marker_index + 1] / 1000):
            return False
        if marker_index == len(offsets) - 1 and new_x <= (onsets[marker_index] / 1000):
            return False
    return True


def mark_selected_marker(ax3, time, marker_height):
    """Deprecated: selection markers are drawn via canvas.mark_selected()."""
    return


def _undo_redo_action(key):
    """Map a matplotlib key string to 'undo'/'redo', or None.

    Accepts Ctrl and Cmd (super) so it works on macOS and elsewhere:
    Ctrl/Cmd+Z = undo, Ctrl/Cmd+Shift+Z and Ctrl/Cmd+Y = redo.
    """
    if not key:
        return None
    parts = key.split('+')
    base = parts[-1]
    mods = set(parts[:-1])
    has_cmd = bool(mods & {'ctrl', 'control', 'cmd', 'super'})
    if not has_cmd:
        return None
    if base == 'z':
        return 'redo' if 'shift' in mods else 'undo'
    if base == 'y':
        return 'redo'
    return None


def handle_keypress(event, app_state, v):
    """Handle keypress events to set edit types and update the selection bar."""
    app_state.logger.debug("Key pressed: %s", event.key)

    action = _undo_redo_action(event.key)
    if action == 'undo':
        undo_segmentation(app_state)
        return
    if action == 'redo':
        redo_segmentation(app_state)
        return

    # shortcuts
    edit_type = app_state.edit_type
    if event.key == 'escape':
        edit_type = "None"
        v.set("1")
        if app_state.selected_syllable_index is not None:
            idx = app_state.selected_syllable_index
            app_state.canvas.set_label_color(idx, '#000000')
            app_state.selected_syllable_index = None
    elif edit_type != "Label Interactive":
        if event.key == 'm':
            edit_type = "Move Segment"
            v.set("4")
        elif event.key == 'n':
            edit_type = "New Segment"
            v.set("2")
        elif event.key == 'd':
            edit_type = "Delete Segment"
            v.set("3")
        elif event.key == 'l':
            edit_type = "Label Interactive"
            v.set("5")

    app_state.logger.debug("Edit type set to: %s", edit_type)
    app_state.edit_type = edit_type
