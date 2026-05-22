# utils/segment_utils.py
import evfuncs
import matplotlib
import numpy as np
import os
import threading
import torch
import re

from moove.qt_helpers import invoke_in_main_thread, show_info
from moove.utils.movefuncs_utils import (
    create_recfile_for_existing_audio,
    ensure_hand_segmented_and_classified_lines,
)

matplotlib.use('Agg')


def _set_resegment_running(app_state, running):
    """Store running state for the resegment dialog."""
    win = getattr(app_state, 'resegment_window', None)
    if win is not None:
        win._task_running = bool(running)
        if running:
            win._task_cancel_requested = False


def _resegment_cancel_requested(app_state):
    """Return True if user requested cancellation via dialog close."""
    win = getattr(app_state, 'resegment_window', None)
    return bool(win is not None and getattr(win, '_task_cancel_requested', False))


def _set_training_task_running(app_state, running):
    """Store running state for training dataset creation dialog tasks."""
    win = getattr(app_state, 'training_window', None)
    if win is not None:
        win._task_running = bool(running)
        if running:
            win._task_cancel_requested = False


def _training_task_cancel_requested(app_state):
    """Return True if user requested cancellation for training dialog task."""
    win = getattr(app_state, 'training_window', None)
    return bool(win is not None and getattr(win, '_task_cancel_requested', False))


def load_segmentation_checkmarks(all_files):
    """Return files that are not yet marked as hand-segmented."""
    unsegmented_files = []
    for file in all_files:
        recfile_path = os.path.splitext(file)[0] + ".rec"
        if not os.path.exists(recfile_path):
            # Missing recfile should not block relabeling.
            create_recfile_for_existing_audio(file)
            unsegmented_files.append(file)
            continue
        with open(recfile_path, "r") as f:
            content = f.read()

        content = ensure_hand_segmented_and_classified_lines(recfile_path, content)

        hand_segmented_pattern = r"Hand Segmented = (\d+)"
        hand_segmented_match = re.search(hand_segmented_pattern, content)

        if hand_segmented_match.group(1) == '0':
            unsegmented_files.append(file)

    return unsegmented_files


def start_segment_evfuncs(app_state, selection, batch_file, bird, experiment, day):
    """Start the segmentation process using Evfuncs in a separate thread."""
    from moove.utils import get_files_for_day, get_files_for_experiment, get_files_for_bird, get_file_data_by_index

    files = []
    if selection == "current_day":
        files = get_files_for_day(app_state, bird, experiment, day, batch_file)
    elif selection == "current_experiment":
        files = get_files_for_experiment(app_state, bird, experiment, batch_file)
    elif selection == "current_bird":
        files = get_files_for_bird(app_state, bird, batch_file)
    elif selection == "current_file":
        files = [get_file_data_by_index(app_state.data_dir, app_state.song_files, app_state.current_file_index, app_state)["file_path"]]

    win = app_state.resegment_window
    if getattr(win, '_task_running', False):
        show_info(win, "Info", "A resegmentation job is already running.")
        return

    progressbar = win.progressbar
    progressbar.setMaximum(len(files))
    progressbar.setValue(0)
    progressbar.show()
    _set_resegment_running(app_state, True)

    def thread_wrapper():
        current_thread = threading.current_thread()
        try:
            segment_evfuncs(app_state, progressbar, files)
        finally:
            _set_resegment_running(app_state, False)
            app_state.remove_thread(current_thread)

    thread = threading.Thread(target=thread_wrapper, name="SegmentEvfuncsThread")
    app_state.add_thread(thread)
    thread.start()


def segment_evfuncs(app_state, progressbar, files):
    """Perform segmentation using Evfuncs for each selected file."""
    from moove.utils import get_display_data, plot_data, save_notmat, decibel

    original_data_dir = app_state.data_dir
    original_song_files = app_state.song_files.copy() if app_state.song_files else []
    original_current_file_index = app_state.current_file_index
    failed_count = 0
    cancelled = False

    for i, file_i in enumerate(files):
        if _resegment_cancel_requested(app_state):
            cancelled = True
            break
        try:
            invoke_in_main_thread(progressbar.setValue, i)

            file_data = get_display_data({"file_name": os.path.basename(file_i), "file_path": file_i}, app_state.config)
            app_state.data_dir = os.path.dirname(file_i)

            params = app_state.evfuncs_params
            threshold = float(params['threshold'].get())
            min_syl_dur = float(params['min_syl_dur'].get())
            min_silent_dur = float(params['min_silent_dur'].get())
            freq_cutoffs = tuple(map(int, params['freq_cutoffs'].get().split(',')))
            smooth_window = int(params['smooth_window'].get())

            sampling_rate = int(file_data["sampling_rate"])
            rawsong = file_data["song_data"]
            smooth = evfuncs.smooth_data(rawsong, sampling_rate, freq_cutoffs, smooth_window)
            db_smooth = decibel(smooth)
            onsets, offsets = evfuncs.segment_song(db_smooth, sampling_rate, threshold, min_syl_dur, min_silent_dur)

            if onsets is not None and offsets is not None:
                onsets = np.multiply(onsets, 1000)
                offsets = np.multiply(offsets, 1000)
                file_data.update({"onsets": onsets, "offsets": offsets, "labels": "x" * len(onsets)})
            else:
                file_data.update({"onsets": np.array([]), "offsets": np.array([]), "labels": ""})

            save_notmat(os.path.join(app_state.data_dir, file_data["file_name"] + ".not.mat"), file_data)
        except Exception as e:
            if isinstance(e, FileNotFoundError):
                app_state.logger.warning("Skipped missing file during Evfuncs segmentation: %s", file_i)
            else:
                app_state.logger.warning("Skipped file during Evfuncs segmentation: %s (%s)", file_i, e)
            print(f"Skipped file: {file_i}")
            failed_count += 1
            continue

    app_state.data_dir = original_data_dir
    app_state.song_files = original_song_files
    app_state.current_file_index = original_current_file_index

    if cancelled:
        invoke_in_main_thread(progressbar.hide)
        invoke_in_main_thread(lambda: show_info(
            app_state.resegment_window, "Info", "Resegmentation aborted."))
        return

    invoke_in_main_thread(progressbar.setValue, len(files))
    invoke_in_main_thread(plot_data, app_state)
    invoke_in_main_thread(progressbar.hide)
    invoke_in_main_thread(lambda: show_info(
        app_state.resegment_window, "Info",
        f"Segmentation with Evfuncs completed!\nFailed files: {failed_count}"))


def segment_ml(
    model, metadata, device, raw_song_data, sampling_rate=44100,
    chunk_size=64, decision_threshold=0.5, hist_size=3, onset_window_size=5,
    n_onset_true=3, offset_window_size=5, n_offset_false=4, min_silent_duration=0.03,
    min_syllable_length=0.005):
    """Perform segmentation using a machine learning model."""
    from moove.utils.audio_utils import index_to_seconds, seconds_to_chunk_index

    mean_loaded = torch.tensor(metadata['mean']).to(device)
    std_loaded = torch.tensor(metadata['std']).to(device)
    seg_input_size = hist_size

    y_pred_list, onset_flag = [], False
    onset_idxs, offset_idxs, raw_audio_data_list = [], [], []

    for i in range(0, len(raw_song_data), chunk_size):
        raw_audio_data = raw_song_data[i:i + chunk_size]
        raw_audio_data_list.append(raw_audio_data)
        if len(raw_audio_data) < chunk_size:
            break

        if i >= hist_size * chunk_size:
            X = torch.tensor(np.concatenate(raw_audio_data_list[-seg_input_size:]).astype(np.float32)).unsqueeze(0).to(device)
            X = (X - mean_loaded) / std_loaded
            y_pred = torch.sigmoid(model(X)).item()
            y_pred_list.append(1 if y_pred > decision_threshold else 0)

            sub_y_onset = y_pred_list[-onset_window_size:]
            sub_y_offset = y_pred_list[-offset_window_size:]
            if not onset_flag and sub_y_onset.count(1) >= n_onset_true:
                if 0 in sub_y_onset[::-1]:
                    onset_idxs.append(len(y_pred_list) - sub_y_onset[::-1].index(0) - 1)
                    onset_flag = True
            elif onset_flag and sub_y_offset.count(0) >= n_offset_false:
                if 1 in sub_y_offset[::-1]:
                    offset_idxs.append(len(y_pred_list) - sub_y_offset[::-1].index(1) - 1)
                    onset_flag = False

    segments_2_remove = []
    min_dist = seconds_to_chunk_index(min_syllable_length, chunk_size, sampling_rate)
    min_dist2 = seconds_to_chunk_index(min_silent_duration, chunk_size, sampling_rate)

    if len(onset_idxs) == len(offset_idxs) + 1:
        onset_idxs.pop()
    elif len(onset_idxs) != len(offset_idxs):
        raise ValueError("Mismatch in onset and offset counts; cannot proceed.")

    for i, (oi, fi) in enumerate(zip(onset_idxs, offset_idxs)):
        if fi - oi < min_dist or (i < len(onset_idxs) - 1 and onset_idxs[i + 1] - fi < min_dist2):
            segments_2_remove.append(i)

    onset_idxs = [onset_idxs[i] for i in range(len(onset_idxs)) if i not in segments_2_remove]
    offset_idxs = [offset_idxs[i] for i in range(len(offset_idxs)) if i not in segments_2_remove]

    onsets = [index_to_seconds(idx, chunk_size, sampling_rate) for idx in onset_idxs]
    offsets = [index_to_seconds(idx, chunk_size, sampling_rate) for idx in offset_idxs]
    return onsets, offsets


def create_segmentation_training_dataset(app_state, progressbar, dataset_name, all_files, parent):
    """Create a segmentation training dataset from multiple files."""
    from moove.utils import get_display_data, save_features, plot_data, extract_raw_audio

    if len(all_files) == 0:
        invoke_in_main_thread(progressbar.hide)
        invoke_in_main_thread(lambda: show_info(
            parent, "Error", "Not enough files given! You need at least 1 file to create a dataset."))
        return

    chunk_size = int(app_state.train_segmentation_params['chunk_size'].get())
    hist_size = int(app_state.train_segmentation_params['hist_size'].get()) + 1
    overlap_chunks = app_state.train_segmentation_params['overlap_chunks'].get()

    all_feature_arrays = []
    cancelled = False

    def generate_concatenated_chunks_with_labels(arr, hist_size, overlap_chunks=False):
        """Concatenate consecutive chunk windows with labels.

        arr: (n_chunks, chunk_size+2) — col 0 = file_index, cols 1:-1 = features, col -1 = label
        Returns: (n_windows, 1 + hist_size*chunk_size + 1)
        """
        n = len(arr)
        if n < hist_size:
            return np.empty((0, arr.shape[1]), dtype=np.float32)
        step_size = 1 if overlap_chunks else hist_size
        features = arr[:, 1:-1]
        labels = arr[:, -1]
        file_idx = arr[0, 0]
        chunk_width = features.shape[1]
        concat_width = hist_size * chunk_width

        indices = list(range(0, n - (hist_size - 1), step_size))
        num_windows = len(indices)
        if num_windows == 0:
            return np.empty((0, 1 + concat_width + 1), dtype=np.float32)

        result = np.empty((num_windows, 1 + concat_width + 1), dtype=np.float32)
        result[:, 0] = file_idx
        for j, i in enumerate(indices):
            result[j, 1:-1] = features[i:i + hist_size].ravel()
            result[j, -1] = labels[i]
        return result

    invoke_in_main_thread(progressbar.hide)

    def _show_looking():
        if hasattr(app_state.training_window, 'status_label'):
            app_state.training_window.status_label.setText("Looking for segments...")
            app_state.training_window.status_label.show()
    invoke_in_main_thread(_show_looking)

    def get_onset_offset_info(file_path):
        notmat_file = file_path + ".not.mat"
        if os.path.exists(notmat_file):
            notmat_dict = evfuncs.load_notmat(notmat_file)
            return {"onsets": notmat_dict.get("onsets", []), "offsets": notmat_dict.get("offsets", [])}
        return {"onsets": [], "offsets": []}

    num_segs = 0
    for fp in all_files:
        if _training_task_cancel_requested(app_state):
            cancelled = True
            break
        info = get_onset_offset_info(fp)
        if len(info["onsets"]) > 0 and len(info["offsets"]) > 0:
            num_segs += min(len(info["onsets"]), len(info["offsets"]))

    if cancelled:
        invoke_in_main_thread(progressbar.hide)
        invoke_in_main_thread(lambda: (
            app_state.training_window.status_label.hide() if hasattr(app_state.training_window, 'status_label') else None,
            show_info(app_state.training_window, "Info", "Segmentation dataset creation aborted.")))
        return

    if num_segs == 0:
        invoke_in_main_thread(lambda: (
            app_state.training_window.status_label.hide() if hasattr(app_state.training_window, 'status_label') else None,
            show_info(app_state.training_window, "Error", "No segments found in the given files.")))
        return

    def _hide_show_progress():
        if hasattr(app_state.training_window, 'status_label'):
            app_state.training_window.status_label.hide()
        progressbar.show()
    invoke_in_main_thread(_hide_show_progress)

    for file_index, file_path in enumerate(all_files):
        if _training_task_cancel_requested(app_state):
            cancelled = True
            break
        try:
            display_data = get_display_data({"file_name": os.path.basename(file_path), "file_path": file_path}, app_state.config)
        except Exception as e:
            if isinstance(e, FileNotFoundError):
                app_state.logger.warning("Skipped missing file in seg dataset creation: %s", file_path)
            else:
                app_state.logger.warning("Skipped file in seg dataset creation: %s (%s)", file_path, e)
            print(f"Skipped file: {file_path}")
            continue
        sampling_rate = int(display_data["sampling_rate"])
        rawsong = display_data["song_data"]
        onsets = (np.array(display_data["onsets"]) * sampling_rate / 1000).astype(int)
        offsets = (np.array(display_data["offsets"]) * sampling_rate / 1000).astype(int)
        invoke_in_main_thread(progressbar.setValue, file_index + 1)

        audio_features = extract_raw_audio(rawsong, chunk_size)
        # audio_features is (chunk_size, num_full_chunks) — transpose to (num_full_chunks, chunk_size)
        chunks = np.asarray(audio_features, dtype=np.float32).T
        n_chunks = chunks.shape[0]
        labels = np.zeros(n_chunks, dtype=np.float32)

        for i, start_idx in enumerate(range(0, len(rawsong) - chunk_size + 1, chunk_size)):
            end_idx = start_idx + chunk_size
            if i < n_chunks and any(onset <= end_idx and start_idx <= offset for onset, offset in zip(onsets, offsets)):
                labels[i] = 1

        # Build [file_index, features..., label] per chunk via vectorized hstack
        file_indices = np.full((n_chunks, 1), file_index, dtype=np.float32)
        labels_col = labels.reshape(-1, 1)
        file_features = np.hstack([file_indices, chunks, labels_col])

        concatenated = generate_concatenated_chunks_with_labels(file_features, hist_size, overlap_chunks)
        if concatenated.size > 0:
            all_feature_arrays.append(concatenated)

    if cancelled:
        invoke_in_main_thread(progressbar.hide)
        invoke_in_main_thread(lambda: show_info(
            app_state.training_window, "Info", "Segmentation dataset creation aborted."))
        return

    if len(all_feature_arrays) == 0:
        invoke_in_main_thread(progressbar.hide)
        invoke_in_main_thread(lambda: show_info(
            app_state.training_window, "Error", "No valid files could be processed for segmentation dataset creation."))
        return

    feature_array = np.vstack(all_feature_arrays)
    del all_feature_arrays  # free memory before save
    save_features(app_state, dataset_name, feature_array, chunk_size=chunk_size, hist_size=hist_size, num_syls=num_segs)

    app_state.update_segmentation_datasets_combobox()
    invoke_in_main_thread(progressbar.setValue, len(all_files))
    invoke_in_main_thread(plot_data, app_state)
    invoke_in_main_thread(progressbar.hide)

    invoke_in_main_thread(lambda: show_info(
        app_state.training_window, "Info",
        f"The segmentation training dataset has been created successfully!\n"
        f"Features and metadata saved as {dataset_name}_seg.pkl"))
    invoke_in_main_thread(lambda: app_state.change_file(0))


def segment_files_ml(app_state, progressbar, all_files, model, metadata, device):
    """Segment files using a machine learning model in a threaded process."""
    from moove.utils import get_display_data, plot_data, save_notmat

    if not all_files:
        app_state.logger.warning("segment_files_ml: no files to process (list is empty).")
        invoke_in_main_thread(progressbar.hide)
        invoke_in_main_thread(lambda: show_info(
            app_state.resegment_window, "Info",
            "No files to segment. (All files may already be marked as segmented.)"))
        return

    original_data_dir = app_state.data_dir
    original_song_files = app_state.song_files.copy() if app_state.song_files else []
    original_current_file_index = app_state.current_file_index
    failed_count = 0
    cancelled = False

    hist_size = int(metadata['hist_size'])
    chunk_size = int(metadata['chunk_size'])

    for i, file_path in enumerate(all_files):
        if _resegment_cancel_requested(app_state):
            cancelled = True
            break
        try:
            invoke_in_main_thread(progressbar.setValue, i)
            display_data = get_display_data({"file_name": os.path.basename(file_path), "file_path": file_path}, app_state.config)
            app_state.data_dir = os.path.dirname(file_path)

            params = app_state.mlseg_params
            onsets, offsets = segment_ml(
                model, metadata, device, display_data["song_data"], int(display_data["sampling_rate"]),
                chunk_size, float(params['decision_threshold'].get()), hist_size,
                int(params['onset_window_size'].get()), int(params['n_onset_true'].get()),
                int(params['offset_window_size'].get()), int(params['n_offset_false'].get()),
                float(params['min_silent_duration'].get()), float(params['min_syllable_length'].get()))

            onsets_ms = np.array(onsets) * 1000
            offsets_ms = np.array(offsets) * 1000
            display_data.update({
                "onsets": onsets_ms,
                "offsets": offsets_ms,
                "labels": "x" * len(onsets)
            })
            notmat_path = os.path.join(app_state.data_dir, display_data["file_name"] + ".not.mat")
            save_notmat(notmat_path, display_data)
            # app_state.logger.info(f"ML segmentation: saved {len(onsets)} segments to {notmat_path}")
        except Exception as e:
            if isinstance(e, FileNotFoundError):
                app_state.logger.warning("Skipped missing file during ML segmentation: %s", file_path)
            else:
                app_state.logger.warning("Skipped file during ML segmentation: %s (%s)", file_path, e)
            print(f"Skipped file: {file_path}")
            failed_count += 1
            continue

    app_state.data_dir = original_data_dir
    app_state.song_files = original_song_files
    app_state.current_file_index = original_current_file_index

    if cancelled:
        invoke_in_main_thread(progressbar.hide)
        invoke_in_main_thread(lambda: show_info(
            app_state.resegment_window, "Info", "Resegmentation aborted."))
        return

    invoke_in_main_thread(progressbar.hide)
    invoke_in_main_thread(plot_data, app_state)
    invoke_in_main_thread(lambda: show_info(
        app_state.resegment_window, "Info",
        f"Segmentation completed!\nFailed files: {failed_count}"))


def start_segment_files_thread(app_state, segmentation_model_name, selection, checkbox_ow, batch_file, bird, experiment, day):
    """Start a threaded process to segment files based on a selected model and criteria."""
    from moove.utils import (get_files_for_day, get_files_for_experiment, get_files_for_bird, get_file_data_by_index,
                             _load_checkpoint_with_compat)

    files = []
    if selection == "current_day":
        files = get_files_for_day(app_state, bird, experiment, day, batch_file)
    elif selection == "current_experiment":
        files = get_files_for_experiment(app_state, bird, experiment, batch_file)
    elif selection == "current_bird":
        files = get_files_for_bird(app_state, bird, batch_file)
    elif selection == "current_file":
        files = [get_file_data_by_index(app_state.data_dir, app_state.song_files, app_state.current_file_index, app_state)["file_path"]]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if segmentation_model_name is None or segmentation_model_name == "":
        show_info(app_state.relabel_window, "Error", "Please select a trained classification model to proceed.")
        return
    model_path = os.path.join(app_state.config['global_dir'], 'trained_models', f'{segmentation_model_name}.pth')

    try:
        checkpoint = _load_checkpoint_with_compat(model_path, device, app_state)
    except Exception as e:
        app_state.logger.error("Could not load checkpoint '%s': %s", model_path, e)
        show_info(app_state.resegment_window, "Error",
                  "Selected classification model could not be loaded."
                  "Please verify model format / torch compatibility.\n\n"
                  f"Details: {e}")
        return

    if not isinstance(checkpoint, dict) or 'model' not in checkpoint or 'metadata' not in checkpoint:
        show_info(app_state.relabel_window, "Error",
                  "Selected classification model has an unsupported checkpoint structure.")
        return

    model, metadata = checkpoint['model'], checkpoint['metadata']
    model.to(device)
    model.eval()

    if not checkbox_ow:
        files = load_segmentation_checkmarks(files)

    win = app_state.resegment_window
    if getattr(win, '_task_running', False):
        show_info(win, "Info", "A resegmentation job is already running.")
        return

    progressbar = win.progressbar
    progressbar.setMaximum(len(files))
    progressbar.setValue(0)
    progressbar.show()
    _set_resegment_running(app_state, True)

    def thread_wrapper():
        current_thread = threading.current_thread()
        try:
            segment_files_ml(app_state, progressbar, files, model, metadata, device)
        finally:
            _set_resegment_running(app_state, False)
            app_state.remove_thread(current_thread)

    thread = threading.Thread(target=thread_wrapper, name="SegmentMLThread")
    app_state.add_thread(thread)
    thread.start()


def start_create_segmentation_training_dataset(app_state, dataset_name, use_selected_files, selection, batch_file, bird, experiment, day, parent):
    """Start a threaded process to create a segmentation training dataset."""
    from moove.utils import get_files_for_day, get_files_for_experiment, get_files_for_bird, filter_segmented_files

    files = []
    if selection == "current_day":
        files = get_files_for_day(app_state, bird, experiment, day, batch_file)
    elif selection == "current_experiment":
        files = get_files_for_experiment(app_state, bird, experiment, batch_file)
    elif selection == "current_bird":
        files = get_files_for_bird(app_state, bird, batch_file)

    if use_selected_files:
        files = filter_segmented_files(files)

    if len(files) == 0:
        show_info(parent, "Error", "No files found for the current selection/filter.")
        return

    dataset_name = str(dataset_name)
    if len(dataset_name) < 1:
        show_info(parent, "Error", "Dataset name not valid! A dataset name needs to contain at least one character.")
        return

    win = app_state.training_window
    if getattr(win, '_training_running', False) or getattr(win, '_task_running', False):
        show_info(parent, "Info", "A training operation is already running.")
        return

    progressbar = win.progressbar
    progressbar.setMaximum(len(files))
    progressbar.setValue(0)
    progressbar.show()
    _set_training_task_running(app_state, True)

    def thread_wrapper():
        current_thread = threading.current_thread()
        try:
            create_segmentation_training_dataset(app_state, progressbar, dataset_name, files, parent)
        finally:
            _set_training_task_running(app_state, False)
            app_state.remove_thread(current_thread)

    thread = threading.Thread(target=thread_wrapper, name="CreateSegDatasetThread")
    app_state.add_thread(thread)
    thread.start()
