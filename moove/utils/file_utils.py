# utils/file_utils.py
import evfuncs
import os
import re
import pickle
import threading
from collections import OrderedDict
from scipy.io import wavfile as wav
from scipy.signal import spectrogram

from moove.qt_helpers import invoke_in_main_thread, show_info
from moove.utils.audio_utils import decibel

# Cache of the expensive WAV-derived data (spectrogram + smoothed amplitude),
# keyed by (file_path, mtime). Segmentation (.not.mat) is always read fresh, so
# edits/resegmentation stay correct. Makes Next/Previous and revisits fast.
_WAV_CACHE = OrderedDict()
_WAV_CACHE_MAX = 32
_WAV_CACHE_LOCK = threading.Lock()


def _compute_wav_derived(file_name, file_path, spec_config):
    if file_name.endswith(".cbin"):
        song_data, sampling_rate = evfuncs.load_cbin(file_path)
    elif file_name.endswith(".wav"):
        sampling_rate, song_data = wav.read(file_path)
    else:
        raise ValueError("Unsupported file format")

    smoothed_song_data = evfuncs.smooth_data(song_data, sampling_rate, (500, 10000), 2)
    freqs, times, spectrogram_data = spectrogram(
        song_data, fs=sampling_rate,
        nperseg=spec_config["spec_nperseg"], noverlap=spec_config["spec_noverlap"],
        nfft=spec_config["spec_nfft"])
    return {
        "sampling_rate": sampling_rate,
        "song_data": song_data,
        "freqs": freqs,
        "times": times,
        "spectrogram_data": spectrogram_data,
        "amplitude": decibel(smoothed_song_data),
    }


def get_directories(path):
    """Return a list of directories in the specified path."""
    return [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]


def create_batch_file(data_dir):
    """Create a default batch.txt file."""
    valid_files = [f for f in os.listdir(data_dir) if f.endswith('.wav') or f.endswith('.cbin')]
    batch_path = os.path.join(data_dir, 'batch.txt')
    with open(batch_path, 'w') as batch_file:
        for audio_file in valid_files:
            batch_file.write(f"{audio_file}\n")
    print(f"Default batch.txt file created with {len(valid_files)} entries.")


def find_batch_files(day_path):
    """Find all files in the day directory that match the pattern '.*batch*.' with allowed extensions only"""
    allowed_extensions = ['', '.txt', '.keep']

    all_files = os.listdir(day_path)
    batch_files = []

    for f in all_files:
        if re.search(r'batch', f, re.IGNORECASE):
            _, ext = os.path.splitext(f)
            if ext in allowed_extensions:
                batch_files.append(f)

    if not batch_files:
        create_batch_file(day_path)
        batch_files = ['batch.txt']
    if 'batch.txt' not in batch_files:
        create_batch_file(day_path)
        batch_files.insert(0, 'batch.txt')
    return batch_files


def read_batch(day_path, batch_file="batch.txt"):
    """Read and return the contents of the specified batch file as a list of lines."""
    file_path = os.path.join(day_path, batch_file)
    try:
        with open(file_path, "r") as file:
            content = file.readlines()
    except FileNotFoundError:
        return []
    return [line.strip() for line in content]


def remove_line(file_path, rm_line):
    """Remove line from a given batch file."""
    with open(file_path, 'r') as file:
        lines = file.readlines()
    lines = [line for line in lines if line.strip() != rm_line]
    with open(file_path, 'w') as file:
        file.writelines(lines)


def get_file_data_by_index(path, song_files, file_index, app_state):
    """Retrieve file data by index from the app state.
       Delete entry from song list and all batch files if song file is missing."""
    if not song_files:
        raise FileNotFoundError("No files available in the selected batch.")

    if file_index is None or file_index < 0 or file_index >= len(song_files):
        app_state.current_file_index = 0
        current_file = song_files[0]
    else:
        current_file = song_files[file_index]

    file_path = os.path.join(os.getcwd(), path, current_file)
    file_data_dict = {"file_name": current_file, "file_path": file_path}
    return file_data_dict


def get_display_data(file_data_dict, config):
    """Generate display data including spectrogram and amplitude for a given file."""
    dspec_config = {
        "spec_nperseg": 1024,
        "spec_noverlap": 896,
        "spec_nfft": 1024,
    }

    if config is not None:
        spectrogram_config = {**dspec_config, **config}
    else:
        spectrogram_config = dspec_config

    file_name = file_data_dict["file_name"]
    file_path = file_data_dict["file_path"].replace("wsl$", "wsl.localhost")
    notmat_dict = {}

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file does not exist: {file_path}")

    try:
        mtime = os.path.getmtime(file_path)
    except OSError:
        mtime = 0
    cache_key = (file_path, mtime)

    with _WAV_CACHE_LOCK:
        base = _WAV_CACHE.get(cache_key)
        if base is not None:
            _WAV_CACHE.move_to_end(cache_key)

    if base is None:
        base = _compute_wav_derived(file_name, file_path, spectrogram_config)
        with _WAV_CACHE_LOCK:
            _WAV_CACHE[cache_key] = base
            while len(_WAV_CACHE) > _WAV_CACHE_MAX:
                _WAV_CACHE.popitem(last=False)

    # Shallow copy so per-file edits (onsets/offsets/labels) never mutate the
    # cached WAV arrays.
    display_dict = dict(base)
    display_dict["file_name"] = file_name

    if os.path.exists(file_path + ".not.mat"):
        notmat_dict = evfuncs.load_notmat(file_path + ".not.mat")
    if notmat_dict:
        display_dict.update(notmat_dict)

    return display_dict


def prefetch_wav_data(file_paths, config=None):
    """Warm the WAV cache for the given files in a background thread (so the
    next Next/Previous is instant). Cheap no-op for already-cached files."""
    dspec = {"spec_nperseg": 1024, "spec_noverlap": 896, "spec_nfft": 1024}
    spec_config = {**dspec, **config} if config else dspec

    def _work():
        for fp in file_paths:
            if not fp or not os.path.exists(fp):
                continue
            try:
                mtime = os.path.getmtime(fp)
            except OSError:
                mtime = 0
            key = (fp, mtime)
            with _WAV_CACHE_LOCK:
                if key in _WAV_CACHE:
                    continue
            try:
                data = _compute_wav_derived(os.path.basename(fp), fp, spec_config)
            except Exception:
                continue
            with _WAV_CACHE_LOCK:
                _WAV_CACHE[key] = data
                while len(_WAV_CACHE) > _WAV_CACHE_MAX:
                    _WAV_CACHE.popitem(last=False)

    threading.Thread(target=_work, name="PrefetchWavThread", daemon=True).start()


def save_seg_class_recfile(filepath, segmented, classified):
    """Save or update recfile for a specific file"""
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            if line.strip().startswith('Hand Segmented ='):
                new_lines.append(f'Hand Segmented = {segmented}\n')
            elif line.strip().startswith('Hand Classified ='):
                new_lines.append(f'Hand Classified = {classified}\n')
            else:
                new_lines.append(line)

        with open(filepath, 'w') as f:
            f.writelines(new_lines)
        return True

    except FileNotFoundError:
        return False


def get_files_for_day(app_state, bird, experiment, day, batch_file="batch.txt"):
    """Return files for a specific day (and batch) within the experiment and bird directory."""
    day_path = os.path.join(app_state.config['rec_data'], bird, experiment, day)
    if batch_file == "All Files":
        files = [f for f in os.listdir(day_path)
                 if f.lower().endswith('.wav') or f.lower().endswith('.cbin')]
    else:
        files = read_batch(day_path, batch_file)
    return [os.path.join(day_path, f) for f in files]


def get_files_for_experiment(app_state, bird, experiment, batch_file="batch.txt"):
    """Return all files for a specific experiment (and batch) of the specified bird."""
    experiment_path = os.path.join(app_state.config['rec_data'], bird, experiment)
    all_files = []

    days = get_directories(experiment_path)
    for day in days:
        day_path = os.path.join(app_state.config['rec_data'], bird, experiment, day)

        if batch_file == "All Files":
            files_in_day = [f for f in os.listdir(day_path)
                            if f.lower().endswith('.wav') or f.lower().endswith('.cbin')]
            for file in files_in_day:
                full_file_path = os.path.join(day_path, file)
                all_files.append(full_file_path)
        else:
            files_in_batch = read_batch(experiment_path, batch_file)
            files_in_day = [f for f in os.listdir(day_path)
                            if f.lower().endswith('.wav') or f.lower().endswith('.cbin')]
            for file in files_in_day:
                if file in files_in_batch:
                    full_file_path = os.path.join(day_path, file)
                    all_files.append(full_file_path)

    return all_files


def get_files_for_bird(app_state, bird, batch_file="batch.txt"):
    """Return all files for a specific bird (and batch) from bird folder."""
    bird_path = os.path.join(app_state.config['rec_data'], bird)
    all_files = []

    experiments = get_directories(bird_path)
    for experiment in experiments:
        experiment_path = os.path.join(bird_path, experiment)
        days = get_directories(experiment_path)
        for day in days:
            day_path = os.path.join(experiment_path, day)

            if batch_file == "All Files":
                files_in_day = [f for f in os.listdir(day_path)
                                if f.lower().endswith('.wav') or f.lower().endswith('.cbin')]
                for file in files_in_day:
                    full_file_path = os.path.join(day_path, file)
                    all_files.append(full_file_path)
            else:
                files_in_batch = read_batch(bird_path, batch_file)
                files_in_day = [f for f in os.listdir(day_path)
                                if f.lower().endswith('.wav') or f.lower().endswith('.cbin')]
                for file in files_in_day:
                    if file in files_in_batch:
                        full_file_path = os.path.join(day_path, file)
                        all_files.append(full_file_path)

    return all_files


def filter_segmented_files(files):
    """Filter and return files that are segmented according to the recfile."""
    segment_those_files = []
    for file in files:
        recfile_path = os.path.splitext(file)[0] + ".rec"
        if not os.path.exists(recfile_path):
            continue
        with open(recfile_path, "r") as f:
            content = f.read()

        hand_segmented_pattern = r"Hand Segmented = (\d+)"
        hand_segmented_match = re.search(hand_segmented_pattern, content)

        if hand_segmented_match and hand_segmented_match.group(1) == '1':
            segment_those_files.append(file)

    print(f"Total segmented files found: {len(segment_those_files)}")
    return segment_those_files


def filter_classified_files(files):
    """Filter and return files that are classified according to the recdata."""
    classify_those_files = []
    for file in files:
        recfile_path = os.path.splitext(file)[0] + ".rec"
        if not os.path.exists(recfile_path):
            continue
        with open(recfile_path, "r") as f:
            content = f.read()

        hand_segmented_pattern = r"Hand Classified = (\d+)"
        hand_segmented_match = re.search(hand_segmented_pattern, content)

        if hand_segmented_match and hand_segmented_match.group(1) == '1':
            classify_those_files.append(file)

    print(f"Total classified files found: {len(classify_those_files)}")
    return classify_those_files


def save_features(app_state, dataset_name, arr, chunk_size=64, hist_size=3, num_files=None, num_syls=None):
    """Save feature arrays to a dataset with the specified chunk and history size metadata."""
    metadata = {
        'chunk_size': chunk_size,
        'hist_size': hist_size
    }

    save_path = os.path.join(app_state.config['global_dir'], 'training_data', f'{dataset_name}_seg.pkl')
    if num_files is None or num_files <= 1:
        with open(save_path, 'wb') as f:
            pickle.dump({'features': arr, 'metadata': metadata, 'syllables': num_syls}, f)
    else:
        split_size = arr.shape[0] // num_files
        for i in range(num_files):
            start_idx = i * split_size
            end_idx = (i + 1) * split_size if i < num_files - 1 else arr.shape[0]
            with open(f'training_data/{dataset_name}_{i}_seg.pkl', 'wb') as f:
                pickle.dump({'features': arr[start_idx:end_idx], 'metadata': metadata, 'syllables': num_syls}, f)
    # info box combined with box in segment_util.py line 278
    # if num_files is None or num_files <= 1:
    #     invoke_in_main_thread(lambda: show_info(
    #         None, "Info", f"Features and metadata saved as {dataset_name}_seg.pkl"))
    # else:
    #     invoke_in_main_thread(lambda: show_info(
    #         None, "Info",
    #         f"Features and metadata saved as {dataset_name}_0_seg.pkl to {dataset_name}_{num_files-1}_seg.pkl"))


def remove_pkl_suffix(filename):
    """ Remove the suffix from a filename if it exists. """
    return filename.rsplit('.pkl', 1)[0] if filename.endswith('.pkl') else filename
