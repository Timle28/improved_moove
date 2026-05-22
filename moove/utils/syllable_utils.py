# utils/syllable_utils.py
import numpy as np
import os
from matplotlib.patches import Rectangle


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
        update_ax2_ax3(app_state.ax2, app_state.ax3, display_dict, app_state)

    else:
        if app_state.new_onset:
            app_state.new_onset = None


def select_event(event, app_state):
    """Handle syllable selection based on event type."""
    if app_state.edit_type == "Label Interactive" and event.inaxes == app_state.ax2:
        for i, text in enumerate(app_state.ax2.texts):
            if text.get_window_extent().contains(event.x, event.y):
                highlight_syllable(i, app_state)
                break
    elif app_state.edit_type == "New Segment":
        add_new_segment(event, app_state)
    elif app_state.edit_type == "Delete Segment":
        delete_segment(event, app_state)
    elif app_state.edit_type == "Move Segment":
        move_segment(event, app_state)


def _redraw_ax2_labels(app_state):
    """Fast redraw for ax2 labels without restoring cached backgrounds."""
    if app_state.ax2 is None or app_state.canvas is None:
        return

    ax2 = app_state.ax2

    # Temporary white overlay to cover old glyph pixels before redrawing labels.
    wipe = Rectangle(
        (0.0, 0.0), 1.0, 1.0,
        transform=ax2.transAxes,
        facecolor='white',
        edgecolor='none',
        zorder=0,
        clip_on=False,
    )
    ax2.draw_artist(wipe)

    for text in ax2.texts:
        ax2.draw_artist(text)

    app_state.canvas.blit(ax2.bbox)


def highlight_syllable(idx, app_state):
    """Highlight the selected syllable."""
    if idx < 0 or idx >= len(app_state.ax2.texts):
        return

    prev_idx = app_state.selected_syllable_index
    if prev_idx is not None and 0 <= prev_idx < len(app_state.ax2.texts):
        app_state.ax2.texts[prev_idx].set_color('black')
 
    app_state.selected_syllable_index = idx
    app_state.ax2.texts[idx].set_color('red')

    _redraw_ax2_labels(app_state)


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

        # Update only the changed text artist to keep interaction responsive.
        app_state.ax2.texts[app_state.selected_syllable_index].set_text(event.key)

        save_notmat(
            os.path.join(app_state.data_dir, f"{display_dict['file_name']}.not.mat"),
            display_dict
        )
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
                mark_selected_marker(app_state.ax3, closest_marker, marker_height)

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
    """Highlight the selected marker in the plot."""
    ax3.plot(time, marker_height, marker='+', color='red', markersize=10, markeredgewidth=1.5)
    ax3.figure.canvas.draw()


def handle_keypress(event, app_state, v):
    """Handle keypress events to set edit types and update the selection bar."""
    app_state.logger.debug("Key pressed: %s", event.key)

    # shortcuts
    edit_type = app_state.edit_type
    if event.key == 'escape':
        edit_type = "None"
        v.set("1")
        if app_state.selected_syllable_index is not None:
            idx = app_state.selected_syllable_index
            app_state.ax2.texts[idx].set_color('black')
            app_state.selected_syllable_index = None
            _redraw_ax2_labels(app_state)
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
