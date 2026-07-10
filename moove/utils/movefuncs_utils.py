# utils/movefuncs_utils.py
import datetime
import numpy as np
import os
import re
import scipy.io.wavfile as wav
import threading
from jinja2 import Template
from pathlib import Path
from scipy.io import savemat

from PyQt6.QtWidgets import QMessageBox

from moove.qt_helpers import invoke_in_main_thread, set_combo_items

try:
    import sounddevice as sd
except Exception:
    sd = None


def _mat_scalar_or_empty(value):
    """Return float64 scalar for scalar-like values, else an empty float64 array."""
    if value is None:
        return np.array([], dtype=np.float64)
    arr = np.asarray(value)
    if arr.size == 0:
        return np.array([], dtype=np.float64)
    return np.float64(arr.flat[0])


def save_cbin(filepath, data, sample_freq):
    """Writes data to a .cbin file."""
    data = data.astype('>i2')
    data.tofile(filepath)


def save_notmat(filename, notmat_dict):
    """Saves dictionary as a .not.mat file with all numeric fields as float64, using empty arrays if fields are missing."""
    filename = Path(filename)

    if not str(filename).endswith(".not.mat"):
        raise ValueError(
            f"Filename should have extension .not.mat but extension was: {filename.suffix}"
        )

    onsets = notmat_dict['onsets'].astype(np.float64)
    offsets = notmat_dict['offsets'].astype(np.float64)

    header = notmat_dict.get('__header__', 'MATLAB 5.0 MAT-file')
    version = notmat_dict.get('__version__', '1.0')
    globals_ = notmat_dict.get('__globals__', [])
    fs = notmat_dict.get('Fs', notmat_dict.get('sampling_rate', 0))

    save_dict = {
        '__header__': header,
        '__version__': version,
        '__globals__': globals_,
        'Fs': np.float64(fs),
        'fname': notmat_dict['file_name'],
        'labels': notmat_dict['labels'],
        'onsets': onsets.reshape(-1, 1),
        'offsets': offsets.reshape(-1, 1),
        'min_int': _mat_scalar_or_empty(notmat_dict.get('min_int')),
        'min_dur': _mat_scalar_or_empty(notmat_dict.get('min_dur')),
        'threshold': _mat_scalar_or_empty(notmat_dict.get('threshold')),
        'sm_win': _mat_scalar_or_empty(notmat_dict.get('sm_win'))
    }

    header = save_dict['__header__']
    save_dict['__header__'] = header.encode('latin-1') if isinstance(header, str) else header

    savemat(filename, save_dict, do_compression=True)


def ensure_hand_segmented_and_classified_lines(recfile_path, content):
    """Ensure 'Hand Segmented' and 'Hand Classified' lines exist in the recfile."""
    has_hand_segmented = re.search(r"Hand Segmented = (\d+)", content) is not None
    has_hand_classified = re.search(r"Hand Classified = (\d+)", content) is not None
    if has_hand_segmented and has_hand_classified:
        return content

    lines = content.splitlines()
    changed = False

    if not has_hand_segmented:
        insert_idx = None
        for idx, line in enumerate(lines):
            if line.startswith("Hand Classified ="):
                insert_idx = idx
                break

        if insert_idx is None:
            for idx, line in enumerate(lines):
                if line.startswith("Catch Song ="):
                    insert_idx = idx + 1
                    break

        if insert_idx is not None:
            lines.insert(insert_idx, "Hand Segmented = 0")
            changed = True

    if not has_hand_classified:
        insert_idx = None
        for idx, line in enumerate(lines):
            if line.startswith("Hand Segmented ="):
                insert_idx = idx + 1
                break

        if insert_idx is None:
            for idx, line in enumerate(lines):
                if line.startswith("Catch Song ="):
                    insert_idx = idx + 1
                    break

        if insert_idx is not None:
            lines.insert(insert_idx, "Hand Classified = 0")
            changed = True

    if not changed:
        return content

    updated_content = "\n".join(lines)
    if content.endswith("\n"):
        updated_content += "\n"

    with open(recfile_path, "w") as f:
        f.write(updated_content)

    print(f"Added missing rec file entries: {recfile_path}")

    return updated_content


def load_recfile(file_path):
    '''Loads a .rec file and returns its contents as a dictionary.'''
    with open(file_path, "r") as f:
        content = f.read()

    date_pattern = r"File created: (.+)"
    begin_rec_pattern = r"begin rec = (\d+) ms"
    trig_time_pattern = r"trig time  = (\d+(\.\d+)?) ms"
    rec_end_pattern = r"rec end = (\d+) ms"
    adfreq_pattern = r"ADFREQ =\s+(\d+)"
    chans_pattern = r"Chans = (\d+)"
    samples_pattern = r"Samples = (\d+)"
    catch_song_pattern = r"Catch Song = (\d+)"
    hand_segmented_pattern = r"Hand Segmented = (\d+)"
    hand_classified_pattern = r"Hand Classified = (\d+)"
    t_before_pattern = r"T Before = ([\d\.]+)"
    t_after_pattern = r"T After = ([\d\.]+)"
    feedback_pattern = r"([\d\.]+E\+?\d+) msec: (FB|catch) # ([A-Za-z0-9_\.\\/:]+) : Templ = (\d+)"

    date_match = re.search(date_pattern, content)
    begin_rec_match = re.search(begin_rec_pattern, content)
    trig_time_match = re.search(trig_time_pattern, content)
    rec_end_match = re.search(rec_end_pattern, content)
    adfreq_match = re.search(adfreq_pattern, content)
    chans_match = re.search(chans_pattern, content)
    samples_match = re.search(samples_pattern, content)
    catch_song_match = re.search(catch_song_pattern, content)
    hand_segmented_match = re.search(hand_segmented_pattern, content)
    hand_classified_match = re.search(hand_classified_pattern, content)
    t_before_match = re.search(t_before_pattern, content)
    t_after_match = re.search(t_after_pattern, content)
    feedback_matches = re.findall(feedback_pattern, content)

    feedback_info = []
    for match in feedback_matches:
        feedback_time = float(match[0]) / 1000
        trig_pulse = match[2]
        templ = int(match[3])
        feedback_info.append((feedback_time, trig_pulse, templ))

    recfile_dict = {
        "file_created": date_match.group(1),
        "begin_rec": int(begin_rec_match.group(1)),
        "trig_time": int(float(trig_time_match.group(1))),
        "rec_end": int(rec_end_match.group(1)),
        "adfreq": int(adfreq_match.group(1)),
        "chans": int(chans_match.group(1)),
        "samples": int(samples_match.group(1)),
        "catch_song": int(catch_song_match.group(1)),
        "hand_segmented": int(hand_segmented_match.group(1)),
        "hand_classified": int(hand_classified_match.group(1)),
        "t_before": float(t_before_match.group(1)),
        "t_after": float(t_after_match.group(1)),
        "feedback_info": feedback_info
    }

    return recfile_dict


def save_recfile(file_path, recfile_dict):
    '''Saves dictionary as a .rec file.'''
    template_path = Path(__file__).parent.parent / 'templates' / 'recfile_template.jinja'
    template_string = template_path.read_text(encoding='utf-8')
    template = Template(template_string)
    output = template.render(recfile_dict)

    with open(file_path, 'w') as f:
        f.write(output)


def create_recfile_for_existing_audio(
    wav_path,
    notmat_path=None,
    t_before=2.0000000000E+00,
    t_after=1.0000000000E+00,
    catch_song=0,
    hand_segmented=0,
    hand_classified=0,
    overwrite=False,
):
    """Create a .rec file for an existing audio file."""
    wav_path = Path(wav_path)
    if not wav_path.exists():
        print(f"Audio file not found: {wav_path}")
        return None

    suffix = wav_path.suffix.lower()
    if suffix == ".wav":
        sampling_rate, song_data = wav.read(str(wav_path))
    elif suffix == ".cbin":
        import evfuncs
        song_data, sampling_rate = evfuncs.load_cbin(str(wav_path))
    else:
        raise ValueError(f"Unsupported audio format for rec creation: {wav_path.suffix}")

    chans = 1 if song_data.ndim == 1 else song_data.shape[1]
    total_samples = len(song_data)
    duration_s = total_samples / sampling_rate

    if t_after is None:
        t_after = duration_s

    if notmat_path is not None:
        try:
            import evfuncs
            notmat = evfuncs.load_notmat(str(notmat_path))
            notmat_fs = float(notmat.get("Fs", sampling_rate))
            if abs(notmat_fs - sampling_rate) > 1:
                import warnings
                warnings.warn(
                    f"Sampling rate mismatch: wav={sampling_rate} Hz, "
                    f"notmat Fs={notmat_fs} Hz"
                )
        except Exception:
            pass

    file_created = (
        datetime.datetime.now().strftime("%a, %b %d, %Y, %H:%M:%S") + ".0"
    )

    recfile_dict = {
        "file_created": file_created,
        "begin_rec": 0,
        "trig_time": int(t_before * 1000),
        "rec_end": int(duration_s * 1000),
        "adfreq": sampling_rate,
        "chans": chans,
        "samples": total_samples,
        "catch_song": catch_song,
        "hand_segmented": hand_segmented,
        "hand_classified": hand_classified,
        "t_before": t_before,
        "t_after": t_after,
        "feedback_info": [],
    }

    rec_path = wav_path.with_suffix(".rec")
    if rec_path.exists() and not overwrite:
        raise FileExistsError(
            f"Rec file already exists: {rec_path}. "
            "Pass overwrite=True to replace it."
        )

    save_recfile(str(rec_path), recfile_dict)
    print(f"Created rec file: {rec_path}")
    return str(rec_path)


def ensure_recfile_exists_and_has_flags(file_path):
    """Ensure the recfile exists and has 'Hand Segmented'/'Hand Classified' lines."""
    file_path = Path(file_path)
    recfile_path = file_path.with_suffix(".rec")

    if not recfile_path.exists():
        create_recfile_for_existing_audio(file_path)

    with open(recfile_path, "r") as f:
        content = f.read()

    return ensure_hand_segmented_and_classified_lines(str(recfile_path), content)


def extract_raw_audio(full_audio_data, chunk_size):
    '''Extracts raw audio data as a (chunk_size, n_chunks) array.'''
    full_audio_data = np.asarray(full_audio_data, dtype=np.float32)
    num_full_chunks = len(full_audio_data) // chunk_size
    trimmed = full_audio_data[:num_full_chunks * chunk_size]
    return trimmed.reshape(num_full_chunks, chunk_size).T


def play_sound(display_dict, ax1):
    '''Plays the sound of the displayed data.'''
    import logging
    log = logging.getLogger("moove")
    if sd is None:
        log.error("Playback unavailable: sounddevice (sd) failed to import.")
        return

    x_start, x_end = ax1.get_xlim()
    sr = display_dict["sampling_rate"]
    x1_border = max(0, int(x_start * sr))
    x2_border = int(x_end * sr)

    sound = display_dict["song_data"][x1_border:x2_border]
    if len(sound) == 0:
        log.warning("Playback: empty selection (%.3f-%.3f s); nothing to play.", x_start, x_end)
        return
    log.info("Playback: %d samples (%.3f-%.3f s) at %d Hz.", len(sound), x_start, x_end, sr)

    def play():
        try:
            sd.play(sound, samplerate=sr)
            sd.wait()
        except Exception as exc:
            log.error("Playback failed in sd.play: %s", exc, exc_info=True)

    play_sound_thread = threading.Thread(target=play)
    play_sound_thread.start()


def handle_playback(app_state):
    '''Function to handle the playback of the displayed data.'''
    display_dict = app_state.display_dict
    ax1 = app_state.ax1

    def thread_wrapper():
        current_thread = threading.current_thread()
        try:
            play_sound(display_dict, ax1)
        finally:
            app_state.remove_thread(current_thread)

    play_sound_thread = threading.Thread(target=thread_wrapper, name="PlaybackThread")
    app_state.add_thread(play_sound_thread)
    play_sound_thread.start()


def confirm_delete(app_state):
    ''' Confirm the deletion of the displayed file.'''
    from moove.utils.plot_utils import plot_data
    current_file = app_state.song_files[app_state.current_file_index]
    working_dir = os.getcwd()

    if current_file[-5:] == ".cbin":
        if os.path.exists(os.path.join(working_dir, app_state.data_dir, current_file + ".not.mat")):
            os.remove(os.path.join(working_dir, app_state.data_dir, current_file + ".not.mat"))
        if os.path.exists(os.path.join(working_dir, app_state.data_dir, current_file[:-5] + ".rec")):
            os.remove(os.path.join(working_dir, app_state.data_dir, current_file[:-5] + ".rec"))
        if os.path.exists(os.path.join(working_dir, app_state.data_dir, current_file)):
            os.remove(os.path.join(working_dir, app_state.data_dir, current_file))
    elif current_file[-4:] == ".wav":
        if os.path.exists(os.path.join(working_dir, app_state.data_dir, current_file + ".not.mat")):
            os.remove(os.path.join(working_dir, app_state.data_dir, current_file + ".not.mat"))
        if os.path.exists(os.path.join(working_dir, app_state.data_dir, current_file[:-4] + ".rec")):
            os.remove(os.path.join(working_dir, app_state.data_dir, current_file[:-4] + ".rec"))
        if os.path.exists(os.path.join(working_dir, app_state.data_dir, current_file)):
            os.remove(os.path.join(working_dir, app_state.data_dir, current_file))
    else:
        app_state.logger.warning("Not supported file format")
        return

    del app_state.song_files[app_state.current_file_index]

    with open(os.path.join(app_state.data_dir, app_state.current_batch_file), "w") as file:
        for song in app_state.song_files:
            file.write(song + '\n')

    set_combo_items(app_state.combobox, app_state.song_files)

    if app_state.current_file_index >= len(app_state.song_files):
        app_state.change_file(-1)
        plot_data(app_state)
    else:
        app_state.change_file(0)
        plot_data(app_state)


def handle_delete(app_state):
    '''Function to handle the deletion of the displayed file.'''
    current_file = app_state.song_files[app_state.current_file_index]

    result = QMessageBox.question(
        None, "Delete Options",
        f"How do you want to remove '{current_file}'?\n\nYes = Delete file from disk\nNo = Remove from batch only",
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel
    )

    if result == QMessageBox.StandardButton.Yes:
        confirm_delete(app_state)
    elif result == QMessageBox.StandardButton.No:
        remove_from_batch_only(app_state)


def remove_from_batch_only(app_state):
    '''Remove file from current batch but keep it on disk.'''
    from moove.utils.plot_utils import plot_data

    del app_state.song_files[app_state.current_file_index]

    with open(os.path.join(app_state.data_dir, app_state.current_batch_file), "w") as file:
        for song in app_state.song_files:
            file.write(song + '\n')

    set_combo_items(app_state.combobox, app_state.song_files)

    if app_state.current_file_index >= len(app_state.song_files):
        app_state.change_file(-1)
        plot_data(app_state)
    else:
        app_state.change_file(0)
        plot_data(app_state)


def crop_not_mat(file_path, display_dict, x1_border, x2_border):
    '''Function to crop a .not.mat file.'''
    onsets = display_dict["onsets"]
    offsets = display_dict["offsets"]
    labels = display_dict["labels"]

    x1_border_ms = x1_border * 1000
    x2_border_ms = x2_border * 1000

    onset_index = next((i for i, onset in enumerate(onsets) if onset >= x1_border_ms), len(onsets))
    offset_index = next((i for i, offset in enumerate(offsets) if offset > x2_border_ms), len(offsets))

    cropped_onsets = onsets[onset_index:offset_index]
    cropped_offsets = offsets[onset_index:offset_index]
    cropped_labels = labels[onset_index:offset_index]

    display_dict["onsets"] = np.subtract(cropped_onsets, x1_border_ms)
    display_dict["offsets"] = np.subtract(cropped_offsets, x1_border_ms)
    display_dict["labels"] = cropped_labels

    save_notmat(file_path, display_dict)


def crop_rec_file(file_path, display_dict, x1_border, x2_border, len_cropped_song):
    '''Function to crop a .rec file.'''
    import datetime
    recfile_dict = load_recfile(file_path)

    recfile_dict["file_created"] = datetime.datetime.now().strftime("%a, %b %d, %Y, %H:%M:%S")
    recfile_dict["rec_end"] = int(np.round((x2_border - x1_border) * 1000))
    recfile_dict["samples"] = int(len_cropped_song)

    for feedbackinfo_idx in range(len(recfile_dict["feedback_info"])):
        new_feedback_triggertime = (recfile_dict["feedback_info"][feedbackinfo_idx][0] - x1_border) * 1000
        coeff, exp = "{:.6E}".format(new_feedback_triggertime).split("E")
        if recfile_dict["catch_song"] == 1:
            recfile_dict["feedback_info"][feedbackinfo_idx] = (f"{coeff}E{str(int(exp))}",
                                                               f"catch # catch_file.wav : Templ = {0}")
        elif recfile_dict["catch_song"] == 0:
            recfile_dict["feedback_info"][feedbackinfo_idx] = (f"{coeff}E{str(int(exp))}",
                                                               f"FB # {recfile_dict['feedback_info'][feedbackinfo_idx][1]} : Templ = {0}")

    save_recfile(file_path, recfile_dict)


def confirm_crop(app_state):
    '''Function to confirm the cropping of the displayed data.'''
    from moove.utils.file_utils import get_file_data_by_index, get_display_data
    from moove.utils.plot_utils import plot_data

    file_path = get_file_data_by_index(app_state.data_dir, app_state.song_files, app_state.current_file_index, app_state)
    display_dict = get_display_data(file_path, app_state.config)

    ax1 = app_state.ax1

    x_start, x_end = ax1.get_xlim()
    x1_border = int(x_start * display_dict["sampling_rate"])
    x2_border = int(x_end * display_dict["sampling_rate"])

    cropped_song_data = display_dict["song_data"][x1_border:x2_border]

    file_name = display_dict["file_name"]
    file_extension = os.path.splitext(file_name)[1].lower()
    file_path = os.path.join(app_state.data_dir, file_name)

    if file_extension == ".cbin":
        save_cbin(file_path, cropped_song_data, display_dict["sampling_rate"])
        if os.path.exists(file_path + ".not.mat"):
            crop_not_mat(os.path.join(app_state.data_dir, file_name + ".not.mat"), display_dict, x_start, x_end)
        if os.path.exists(os.path.join(app_state.data_dir, file_name[:-5] + ".rec")):
            crop_rec_file(os.path.join(app_state.data_dir, file_name[:-5] + ".rec"), display_dict, x_start, x_end, len(cropped_song_data))
    elif file_extension == ".wav":
        wav.write(file_path, display_dict["sampling_rate"], cropped_song_data)
        if os.path.exists(file_path + ".not.mat"):
            crop_not_mat(os.path.join(app_state.data_dir, file_name + ".not.mat"), display_dict, x_start, x_end)
        if os.path.exists(os.path.join(app_state.data_dir, file_name[:-4] + ".rec")):
            crop_rec_file(os.path.join(app_state.data_dir, file_name[:-4] + ".rec"), display_dict, x_start, x_end, len(cropped_song_data))

    file_path = get_file_data_by_index(app_state.data_dir, app_state.song_files, app_state.current_file_index, app_state)
    display_dict = get_display_data(file_path, app_state.config)

    app_state.change_file(0)
    plot_data(app_state)


def handle_crop(app_state):
    '''Function to handle the cropping of the displayed data'''
    result = QMessageBox.question(
        None, "Confirm", "Are you sure you want to crop to the displayed area?",
        QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel
    )
    if result == QMessageBox.StandardButton.Ok:
        confirm_crop(app_state)
