# important first import
import torch
import torch.nn.functional as F

import os
import re
import sys
import glob
import configparser
import numpy as np
import datetime
import logging
import random
import threading
import time
import shutil
import sounddevice as sd
from scipy.signal import lfilter, butter, lfilter_zi, spectrogram
from scipy.ndimage import uniform_filter1d
from scipy.io import wavfile
from pathlib import Path
from moove.utils.movefuncs_utils import save_notmat, save_recfile
from moove.utils.training_utils import _load_checkpoint_with_version_fallback

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# Create / read config file
# Allow custom config directory via environment variable, fallback to ~/.moove
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

config = configparser.ConfigParser()
config.read(config_file_path)
global_dir = config.get("GENERAL", "global_dir")
global_dir = os.path.expanduser(global_dir)

os.makedirs(os.path.join(global_dir, "rec_data"), exist_ok=True)
os.makedirs(os.path.join(global_dir, "playbacks"), exist_ok=True)

# Variables needed regardless of realtime_classification
# section TAF
bird_name = config.get('TAF', 'bird_name')
experiment_name = config.get('TAF', 'experiment_name')
frame_rate = int(config.get('TAF', 'frame_rate'))
chunk_size = int(config.get('TAF', 'chunk_size'))
t_before = float(config.get('TAF', 't_before'))  # in seconds
t_after = float(config.get('TAF', 't_after'))  # in seconds
min_bout_duration = float(config.get('TAF', 'min_bout_duration'))
memory_cleanup_interval = int(config.get('TAF', 'memory_cleanup_interval'))
# input channel
input_channel_str = config.get('TAF', 'input_channel')
if ',' in input_channel_str:
    config_input_channel = [int(x.strip()) for x in input_channel_str.split(',')]
else:
    config_input_channel = [int(input_channel_str)]

# section bird
data_output_folder_path = os.path.join(global_dir, 'rec_data')
bout_threshold_db = int(config.get(bird_name, 'bout_threshold_db'))
window_size = int(config.get(bird_name, 'window_size'))
bandpass_lowcut = int(config.get(bird_name, 'bandpass_lowcut'))
bandpass_highcut = int(config.get(bird_name, 'bandpass_highcut'))
bandpass_order = int(config.get(bird_name, 'bandpass_order'))

# bout detection (threshold parameters)
realtime_classification = config.getboolean(bird_name, 'realtime_classification')

# white noise
targeting = config.getboolean(bird_name, 'targeting')
catch_trial_probability = float(config.get(bird_name, 'catch_trial_probability'))
white_noise_duration = float(config.get(bird_name, 'white_noise_duration'))
trigger_time_offset = float(config.get(bird_name, 'trigger_time_offset'))

# sliding interval algorithm for segmentation network
min_silent_duration = float(config.get(bird_name, 'min_silent_duration'))
min_syllable_length = float(config.get(bird_name, 'min_syllable_length'))

if realtime_classification:
    # Load and initialize models
    seg_model_name = config.get(bird_name, 'segmentation_model_name')
    class_model_name = config.get(bird_name, 'classification_model_name')

    decision_threshold = float(config.get(bird_name, 'decision_threshold'))
    onset_window_size = int(config.get(bird_name, 'onset_window_size'))
    n_onset_true = int(config.get(bird_name, 'n_onset_true'))
    offset_window_size = int(config.get(bird_name, 'offset_window_size'))
    n_offset_false = int(config.get(bird_name, 'n_offset_false'))
    

    # target string is read in as the regular expression
    targeted_sequence_str = config.get(bird_name, 'targeted_sequence')
    if targeted_sequence_str.lower() != 'none':
        targeted_sequence_list = [snippet.strip() for snippet in targeted_sequence_str.split(',')]
    else:
        targeted_sequence_list = None


    # path to playback stimuli
    playback_stim_path = os.path.expanduser(config.get(bird_name, 'playback_dir'))
    # Find all wav files in playback stim path
    if config.get(bird_name, 'computer_generated_white_noise') == 'True':
        computer_generated_white_noise = True
    elif config.get(bird_name, 'computer_generated_white_noise') == 'False':
        computer_generated_white_noise = False
        playback_files = sorted(glob.glob(os.path.join(playback_stim_path, '*.wav')))
        playback_sounds = {}
        for wav_file in playback_files:
            samplerate_PB, sound_wav = wavfile.read(wav_file)
            playback_sounds[os.path.basename(wav_file)] = (sound_wav / np.max(sound_wav), samplerate_PB)

    # model paths
    model_dir_path = os.path.join(global_dir, "trained_models")
    # Remove file extension if present
    if seg_model_name.endswith('.pth'):
        seg_model_name = seg_model_name[:-4]
    if class_model_name.endswith('.pth'):
        class_model_name = class_model_name[:-4]

    # Load segmentation model
    checkpoint = _load_checkpoint_with_version_fallback(os.path.join(model_dir_path, f'{seg_model_name}.pth'))
    seg_model = checkpoint['model']
    metadata = checkpoint['metadata']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seg_model.to(device)
    seg_model.eval()

    # Load classification model
    checkpoint = _load_checkpoint_with_version_fallback(os.path.join(model_dir_path, f'{class_model_name}.pth'))
    class_model = checkpoint['model']
    metadata.update(checkpoint['metadata'])

    logger.info("Model metadata:")
    logger.info(metadata)

    class_model.to(device)
    class_model.eval()

    # Parse input_length and chunk_size from metadata
    input_length_str = metadata['input_length']
    input_length, chunk_size_from_metadata = map(int, input_length_str.split(','))

    # Overwrite chunk_size with the one from metadata
    chunk_size = metadata['chunk_size']

    # Check if chunk sizes match
    if chunk_size != chunk_size_from_metadata:
        logger.error("Chunk size from metadata does not match the chunk size.")
        sys.exit(1)

    hist_size = metadata['hist_size']
    mean_loaded = torch.tensor(metadata['mean']).to(device)
    std_loaded = torch.tensor(metadata['std']).to(device)
    nperseg = metadata['nperseg']
    noverlap = metadata['noverlap']
    nfft = metadata['nfft']
    lowcut = int(metadata['lowcut'])
    highcut = int(metadata['highcut'])
    int_to_label = metadata['int_to_label']
    input_chunks = input_length
    seg_input_size = hist_size
else:
    input_length = None
    input_chunks = None
    device = None
    targeted_sequence = None
    targeted_sequence_list = None
    lowcut = None
    highcut = None
    int_to_label = None
    seg_input_size = None
    hist_size = None


def seconds_to_index(seconds, chunk_size, sample_rate):
    """Converts seconds to an index based on chunk size and sample rate."""
    index_size = int((seconds * sample_rate) // chunk_size)
    return index_size


def index_to_seconds(index, chunk_size, sample_rate):
    """Converts an index to seconds based on chunk size and sample rate."""
    seconds = (index * chunk_size) / sample_rate
    return seconds


def calculate_db(waveform):
    """Calculates the decibel value of a waveform."""
    rms = np.sqrt(np.mean(waveform ** 2))
    rms = max(rms, 1e-10)
    decibel = 20 * np.log10(rms)
    return decibel


def play_white_noise(duration_ms):
    """Plays white noise for the specified duration in milliseconds."""
    global is_playing_white_noise, white_noise_index, white_noise
    is_playing_white_noise = True
    white_noise_index = 0
    white_noise = generate_white_noise(duration_ms, frame_rate)


def generate_white_noise(duration_ms, frame_rate, dtype=np.float32):
    """Generates white noise for the specified duration in milliseconds."""
    num_samples = int(frame_rate * duration_ms / 1000)
    return (np.random.randn(num_samples)).astype(dtype)


def play_playback_file(key):
    global is_playing_playback_file, playback_sound_index, playback_sound
    is_playing_playback_file = True
    playback_sound, sr = playback_sounds[key]
    playback_sound_index = 0


def apply_butter_bandpass_filter(data, numerator_coeffs, denominator_coeffs, zi):
    """Applies a Butterworth bandpass filter to the data."""
    y, zf = lfilter(numerator_coeffs, denominator_coeffs, data, zi=zi)
    return y, zf


def millisecond_to_fixed_notation(ms):
    """Formats milliseconds to scientific notation string."""
    coeff, exp = "{:.6E}".format(ms).split("E")
    exp = str(int(exp))  # removes leading zero
    return f"{coeff}E{exp}"


def clean_lists(lists, n):
    """Cleans lists by keeping only the last n elements."""
    for i in range(len(lists)):
        try:
            lists[i][:] = lists[i][-n:]
        except Exception as e:
            logger.debug(e)
            continue


def butter_bandpass_coeffs(lowcut, highcut, fs, order=5):
    """Calculates Butterworth bandpass filter coefficients."""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    numerator_coeffs, denominator_coeffs = butter(order, [low, high], btype='band')
    return numerator_coeffs, denominator_coeffs


def normalize_spectrogram(spectrogram):
    """Normalizes the spectrogram by subtracting mean and dividing by standard deviation."""
    mean = spectrogram.mean()
    std = spectrogram.std()
    normalized_spectrogram = (spectrogram - mean) / std if std != 0 else spectrogram
    return normalized_spectrogram


def check_targeted_sequence(lst, targeted_sequence):
    """Checks if the last elements of lst match the targeted_sequence."""
    recorded_sequence = ''.join(lst)
    return re.search(targeted_sequence, recorded_sequence)


def daily_initialization(data_output_folder_path, experiment_name, bird_name):
    """Initializes daily folders and returns the path to the day folder."""
    if not os.path.exists(data_output_folder_path):
        os.makedirs(data_output_folder_path)

    bird_folder = os.path.join(data_output_folder_path, bird_name)
    if not os.path.exists(bird_folder):
        os.makedirs(bird_folder)

    experiment_folder = os.path.join(bird_folder, experiment_name)

    day_folder = os.path.join(experiment_folder, datetime.datetime.now().strftime("%y%m%d"))
    if not os.path.exists(day_folder):
        os.makedirs(day_folder)

    # Create empty file "batch.txt" in day_folder (if it does not exist)
    batch_path = os.path.join(day_folder, "batch.txt")
    if not os.path.exists(batch_path):
        with open(batch_path, 'w') as f:
            f.write("")
    return day_folder


def save_bout(raw_audio_chunks, bout_indexes_waited, bout_recdt, wn_recfile_dict, onsets, offsets, pred_syl_list):
    """Saves the bout to disk and creates associated files."""
    save_path = daily_initialization(data_output_folder_path, experiment_name, bird_name)
    add_index = int(seconds_to_index(t_before, chunk_size, frame_rate))
    logger.info("Bout indexes waited: %s", bout_indexes_waited)
    logger.info("Add index: %s", add_index)
    logger.debug("Type of raw_audio_chunks: %s", type(raw_audio_chunks))
    logger.debug("Length of raw_audio_chunks: %s", len(raw_audio_chunks))

    # Check if we need to add empty chunks
    required_chunks = bout_indexes_waited + add_index
    if required_chunks > len(raw_audio_chunks):
        missing_chunks = required_chunks - len(raw_audio_chunks)
        logger.info(f"Adding {missing_chunks} empty chunks to fill the gap")
        
        for _ in range(missing_chunks):
            empty_chunk = np.zeros(chunk_size, dtype=np.float32)
            raw_audio_chunks.append(empty_chunk)

    raw_audio2save = np.concatenate(raw_audio_chunks[-(bout_indexes_waited + add_index):])

    logger.info("Length of audio in seconds: %.4f", len(raw_audio2save) / frame_rate)
    logger.debug("%s", len(raw_audio2save) / chunk_size)
    logger.debug("%s", seconds_to_index(min_bout_duration, chunk_size, frame_rate))

    if (len(raw_audio2save) / chunk_size) <= seconds_to_index(min_bout_duration, chunk_size, frame_rate):
        logger.info("Not enough data to save")
        return

    logger.info("Saving bout")
    batch_path = os.path.join(save_path, "batch.txt")

    with open(batch_path, "r") as file:
        lines = file.readlines()

    file_length_lines = len(lines)
    bout_recdt_formatted = bout_recdt.strftime("%y%m%d_%H%M%S")

    # Create file name
    file_name = f"{bird_name}_{bout_recdt_formatted}.{file_length_lines}.wav"

    # Add file name to batch
    with open(batch_path, "a") as file:
        file.write(file_name + "\n")

    wavfile.write(os.path.join(save_path, file_name), frame_rate, raw_audio2save)

    # Create feedback info from wn_recfile_dict
    feedback_info = []
    for trigger_time, wn_info in wn_recfile_dict.items():
        if trigger_time != "catch_song":  # skip metadata key
            feedback_info.append((trigger_time, wn_info))

    # Save recfile
    channels = 1  # Adjust if necessary
    template_vars = {
        'file_created': bout_recdt.strftime("%a, %b %d, %Y, %H:%M:%S") + f".{int(file_length_lines)}",
        'begin_rec': 0,
        'trig_time': t_before * 1000,  # Convert to milliseconds
        'rec_end': int((len(raw_audio2save) / frame_rate) * 1000),
        'adfreq': frame_rate,
        'chans': channels,
        'samples': len(raw_audio2save),
        'catch_song': wn_recfile_dict.get("catch_song", False),
        'hand_segmented': 0,
        'hand_classified': 0,
        't_before': "{:.10E}".format(t_before),
        't_after': "{:.10E}".format(t_after),
        'feedback_info': feedback_info,
    }

    save_recfile(os.path.join(save_path, file_name.replace(".wav", ".rec")), template_vars)

    logger.info("Recfile saved!")

    # Convert predicted syllable list to string
    pred_syl_str = ''.join(map(str, pred_syl_list))

    notmat_dict = {
        '__header__': b'MATLAB 5.0 MAT-file, Platform: PCWIN64, Created on: MOOVETAF',
        '__version__': '1.0',
        '__globals__': [],
        'Fs': frame_rate,
        'file_name': file_name,
        'labels': pred_syl_str,
        'onsets': np.array(onsets).reshape(-1,1),
        'offsets': np.array(offsets).reshape(-1,1),
        'min_int': min_silent_duration * 100,
        'min_dur': min_syllable_length * 100,
        'threshold': 10 ** ((bout_threshold_db - 10) / 20),
        'sm_win': 2
    }
    logger.debug(notmat_dict)
    save_notmat(os.path.join(save_path, file_name.replace(".wav", ".wav.not.mat")), notmat_dict)

    logger.info("Notmat file saved!")


# Initialize global variables
raw_audio_chunks = []
db_values_list = []
bout_flag = False
bout_index2wait = 0
bout_indexes_waited = 0
bout_recdt = ""
onsets = []
offsets = []
onset_flag = False
offset_pending = False
offset_detected_time = 0  # Store when offset was first detected
waited_class_time = 0
class_flag = False
pred_syl_list = []
pred_syl_list_for_playback = []
wn_recfile_dict = {}
not_catch_trial_flag = False
no_classify_flag_wn = False
no_classify_flag_wn_idx2wait = 0
missing_y_pred_flag = False
min_silent_index2wait = 0
min_silent_waited = True
initialization_complete = False  # Flag to track initialization

if realtime_classification:
    y_pred_list = [0] * hist_size
    seg_input_size = hist_size
else:
    y_pred_list = []
    seg_input_size = None
offset_pending = False


def stream_callback(indata, outdata, frames, time_info, status):
    """Callback function for processing live audio stream."""
    global raw_audio_chunks
    global db_values_list
    global bout_flag
    global bout_index2wait
    global bout_indexes_waited
    global bout_recdt
    global y_pred_list
    global onsets
    global offsets
    global onset_flag
    global waited_class_time
    global class_flag
    global pred_syl_list
    global pred_syl_list_for_playback
    global wn_recfile_dict
    global not_catch_trial_flag
    global no_classify_flag_wn
    global no_classify_flag_wn_idx2wait
    global missing_y_pred_flag
    global min_silent_index2wait
    global min_silent_waited
    global seg_input_size
    global initialization_complete
    global offset_detected_time
    global targeted_sequence_list, targeted_sequence

    global is_playing_white_noise, white_noise_index, white_noise
    global is_playing_playback_file, playback_sound_index, playback_sound
    global bandpass_numerator_coeffs, bandpass_denominator_coeffs, zi
    global frame_rate, channels, input_chunks, config_input_channel
    global lowcut, highcut
    global int_to_label
    global device
    global offset_pending

    input_channels = channels[0]

    if len(config_input_channel) == 1:
        audio_data = indata[:, config_input_channel[0]]
    elif input_channels > 1 and len(config_input_channel) == 2:
        audio_data = (indata[:, config_input_channel[0]] + indata[:, config_input_channel[1]]) / 2
    else:
        logger.error("Too many input channels given. Only 1 or 2 channels are allowed.\nPlease quit the program!!!")
        

    # multiply audio_signal to get signal range between -1 and 1
    audio_data = audio_data * 1e2
    audio_data = np.asarray(audio_data, dtype=np.float32)
    raw_audio_chunks.append(audio_data.copy())

    # Check if we have enough data for initialization (t_before seconds)
    required_chunks = int(seconds_to_index(t_before, chunk_size, frame_rate))
    if not initialization_complete and len(raw_audio_chunks) >= required_chunks:
        initialization_complete = True
        logger.info(f"Initialization complete. Collected {len(raw_audio_chunks)} chunks ({len(raw_audio_chunks) * chunk_size / frame_rate:.2f} seconds)")

    # Apply bandpass filter
    filtered_data, zi = apply_butter_bandpass_filter(
        audio_data,
        bandpass_numerator_coeffs,
        bandpass_denominator_coeffs,
        zi
    )

    db_value = calculate_db(filtered_data)
    db_values_list.append(db_value)

    if len(db_values_list) <= window_size:
        smoothed_db = bout_threshold_db - 10  # Arbitrary value below threshold
    else:
        smoothed_db = uniform_filter1d(
            db_values_list[-window_size:],
            size=window_size,
            mode='constant',
            origin=0
        )[-1]

    # Check threshold and start recording (only after initialization is complete)
    if smoothed_db > bout_threshold_db and not bout_flag and initialization_complete:
        bout_index2wait = int(seconds_to_index(t_after, chunk_size, frame_rate))
        recdt = datetime.datetime.now() - datetime.timedelta(seconds=t_before)
        bout_recdt = recdt
        bout_flag = True
        wn_recfile_dict = {}
        missing_y_pred_flag = False

        if random.random() >= catch_trial_probability:
            not_catch_trial_flag = True
            wn_recfile_dict["catch_song"] = 0
        else:
            not_catch_trial_flag = False
            wn_recfile_dict["catch_song"] = 1

        # chooses the target sequence for the upcoming bout from the target sequence list
        # if several options then choose one randomly
        if targeted_sequence_list is None:
            targeted_sequence = None
        elif len(targeted_sequence_list) > 1:
            targeted_sequence = np.random.choice(targeted_sequence_list)
        elif len(targeted_sequence_list) == 1:
            targeted_sequence = targeted_sequence_list[0]
        else:
            logger.error("Invalid or missing target sequence")

        logger.info("Not catch trial flag: %s", not_catch_trial_flag)
        logger.info("Threshold triggered")

    # Handle no classification during white noise playback
    if no_classify_flag_wn:
        no_classify_flag_wn_idx2wait -= 1
        if no_classify_flag_wn_idx2wait == 0:
            no_classify_flag_wn = False

    # Memory cleanup
    if len(db_values_list) > (memory_cleanup_interval * frame_rate / chunk_size) and not bout_flag:
        n = int(seconds_to_index(5, chunk_size, frame_rate))
        lists = [raw_audio_chunks, db_values_list, y_pred_list, pred_syl_list]
        clean_lists(lists, n)
        logger.debug("Cleaned list entries")

    # Handle min_silent_duration if classification is enabled
    if realtime_classification:
        if not min_silent_waited:
            min_silent_index2wait -= 1
            if min_silent_index2wait <= 0:
                min_silent_waited = True

    # Determine if classification is allowed
    classification_allowed = realtime_classification and min_silent_waited and not no_classify_flag_wn

    # If bout_flag is set, wait for "t_after" seconds of silence
    if bout_flag:
        bout_indexes_waited += 1
        if smoothed_db > bout_threshold_db:
            bout_index2wait = int(seconds_to_index(t_after, chunk_size, frame_rate))

        if bout_index2wait == 0:
            # Before saving, drop incomplete trailing entries (e.g., onset without offset/label).
            # This keeps not.mat arrays aligned and prevents one-extra-onset files.
            while len(onsets) > len(offsets):
                onsets.pop()
                logger.debug("Dropped dangling onset before save")

            while len(offsets) > len(onsets):
                offsets.pop()
                logger.debug("Dropped dangling offset before save")

            complete_syllables = min(len(onsets), len(offsets), len(pred_syl_list))
            while len(onsets) > complete_syllables:
                onsets.pop()
            while len(offsets) > complete_syllables:
                offsets.pop()
            while len(pred_syl_list) > complete_syllables:
                pred_syl_list.pop()

            bout_data_to_save = raw_audio_chunks.copy()
            bout_indexes_waited_copy = bout_indexes_waited
            bout_rec_dt_copy = bout_recdt
            onsets_copy = onsets.copy()
            offsets_copy = offsets.copy()
            pred_syl_list_copy = pred_syl_list.copy()
            wn_recfile_dict_copy = wn_recfile_dict.copy()
            # Start a thread to save the bout
            save_thread = threading.Thread(target=save_bout, args=(
            bout_data_to_save, bout_indexes_waited_copy, bout_rec_dt_copy, wn_recfile_dict_copy, onsets_copy,
            offsets_copy, pred_syl_list_copy))
            save_thread.start()

            # Clean up lists and variables
            n = int(seconds_to_index(t_before, chunk_size, frame_rate))
            onsets = []
            offsets = []
            raw_audio_chunks = raw_audio_chunks[-n:]
            bout_indexes_waited = 0
            y_pred_list = y_pred_list[-n:] if realtime_classification else []
            onset_flag = False
            class_flag = False
            waited_class_time = 0
            missing_y_pred_flag = False
            offset_pending = False
            offset_detected_time = 0
            bout_flag = False
            pred_syl_list = []
            pred_syl_list_for_playback = []
            min_silent_waited = True

        bout_index2wait -= 1

        if classification_allowed:
            # Data preparation for segmentation
            if len(raw_audio_chunks) >= seg_input_size:
                X = torch.tensor(np.concatenate(raw_audio_chunks[-seg_input_size:])).float().to(device)
                X = X.unsqueeze(0)
                X = (X - mean_loaded) / std_loaded
                y = seg_model(X)
                y = torch.sigmoid(y)

                y_pred = torch.where(y > decision_threshold, torch.tensor(1.0, device=device),
                                     torch.tensor(0.0, device=device))
                y_pred_list.append(int(y_pred.item()))

                # Onset detection
                sub_y = y_pred_list[-onset_window_size:]
                count = sub_y.count(1)
                
                # New onset can only be detected if:
                # 1. No onset is currently active (onset_flag = False), OR
                # 2. Current onset has offset condition met (enough zeros for offset)
                zeros_in_window = len(sub_y) - count  # Number of zeros in window
                can_detect_new_onset = (not onset_flag) or (onset_flag and zeros_in_window >= n_offset_false)
                
                if can_detect_new_onset and count >= n_onset_true:
                    # If there's an active onset with offset condition met, process the offset first
                    if onset_flag and zeros_in_window >= n_offset_false:
                        logger.debug("Processing offset for current syllable before new onset")
                        
                        # Handle pending offset or immediate offset
                        if offset_pending:
                            # Use the originally detected time
                            offset_time_index = offset_detected_time
                        else:
                            # Calculate current offset time
                            offset_time_index = bout_indexes_waited
                        
                        # Calculate offset time
                        sub_y_long_rev = y_pred_list[::-1]
                        try:
                            idx = sub_y_long_rev.index(1)
                        except ValueError:
                            idx = 0
                        offset_time = ((index_to_seconds(offset_time_index - idx, chunk_size,
                                                         frame_rate)) * 1000) + (t_before * 1000)
                        offsets.append(offset_time)
                        min_silent_index2wait = int(seconds_to_index(min_silent_duration, chunk_size, frame_rate))
                        min_silent_waited = False
                        
                        # Check syllable duration
                        last_duration = offsets[-1] - onsets[-1]
                        len_offset = len(offsets)
                        len_ypred = len(pred_syl_list)
                        if last_duration < (min_syllable_length * 1000):
                            onsets.pop()
                            offsets.pop()
                            if class_flag:
                                # Short syllable is discarded while classification is pending.
                                # Abort current classification to avoid adding a stale label.
                                class_flag = False
                                waited_class_time = 0
                                offset_pending = False
                                offset_detected_time = 0
                                missing_y_pred_flag = False
                                logger.debug("Discarded short syllable and aborted pending classification")
                            elif len_ypred == len_offset:
                                pred_syl_list.pop()
                                pred_syl_list_for_playback.pop()
                            elif len_ypred == (len_offset - 1):
                                # No label exists yet for this discarded syllable.
                                logger.debug("Discarded short syllable had no label to remove")
                            else:
                                logger.warning(
                                    "Mismatch detected during short-syllable discard; resyncing state "
                                    "(onsets=%s offsets=%s labels=%s)",
                                    len(onsets), len(offsets), len(pred_syl_list)
                                )
                                complete_syllables = min(len(onsets), len(offsets), len(pred_syl_list))
                                onsets = onsets[:complete_syllables]
                                offsets = offsets[:complete_syllables]
                                pred_syl_list = pred_syl_list[:complete_syllables]
                                pred_syl_list_for_playback = pred_syl_list.copy()
                                class_flag = False
                                waited_class_time = 0
                                missing_y_pred_flag = False
                                offset_pending = False
                                offset_detected_time = 0
                        
                        # Reset flags for new onset and offset
                        onset_flag = False
                        offset_pending = False
                        offset_detected_time = 0
                    
                    # Now detect the new onset
                    if count >= n_onset_true:
                        sub_y_rev = sub_y[::-1]
                        try:
                            idx_10 = sub_y_rev.index(0)
                        except ValueError:
                            idx_10 = 0
                        onset_time = ((index_to_seconds(bout_indexes_waited - idx_10, chunk_size, frame_rate)) * 1000) + (
                                    t_before * 1000)
                        onsets.append(onset_time)
                        onset_flag = True
                        class_flag = True
                        waited_class_time = input_chunks - idx_10

                # Offset detection
                elif onset_flag and zeros_in_window >= n_offset_false:
                    if class_flag:
                        # Classification is still pending; set offset_pending
                        if not offset_pending:
                            # Store the time when offset was first detected
                            offset_detected_time = bout_indexes_waited
                        offset_pending = True
                    else:
                        # Proceed to record offset
                        sub_y_long_rev = y_pred_list[::-1]
                        try:
                            idx = sub_y_long_rev.index(1)
                        except ValueError:
                            idx = 0
                        offset_time = ((index_to_seconds(bout_indexes_waited - idx, chunk_size, frame_rate)) * 1000) + (
                                    t_before * 1000)
                        offsets.append(offset_time)
                        min_silent_index2wait = int(seconds_to_index(min_silent_duration, chunk_size, frame_rate))
                        min_silent_waited = False
                        logger.debug("min_silent_index2wait: %s", min_silent_index2wait)

                        last_duration = offsets[-1] - onsets[-1]
                        len_offset = len(offsets)
                        len_ypred = len(pred_syl_list)
                        if last_duration < (min_syllable_length * 1000):
                            # Remove the last onset, offset, and corresponding pred_syl_list entry
                            onsets.pop()
                            offsets.pop()
                            if class_flag:
                                # Short syllable is discarded while classification is pending.
                                # Abort current classification to avoid adding a stale label.
                                class_flag = False
                                waited_class_time = 0
                                offset_pending = False
                                offset_detected_time = 0
                                missing_y_pred_flag = False
                                logger.debug("Discarded short syllable and aborted pending classification")
                            elif len_ypred == len_offset:
                                pred_syl_list.pop()
                                pred_syl_list_for_playback.pop()
                            elif len_ypred == (len_offset - 1):
                                # No label exists yet for this discarded syllable.
                                logger.debug("Discarded short syllable had no label to remove")
                            else:
                                logger.warning(
                                    "Mismatch detected during short-syllable discard; resyncing state "
                                    "(onsets=%s offsets=%s labels=%s)",
                                    len(onsets), len(offsets), len(pred_syl_list)
                                )
                                complete_syllables = min(len(onsets), len(offsets), len(pred_syl_list))
                                onsets = onsets[:complete_syllables]
                                offsets = offsets[:complete_syllables]
                                pred_syl_list = pred_syl_list[:complete_syllables]
                                pred_syl_list_for_playback = pred_syl_list.copy()
                                class_flag = False
                                waited_class_time = 0
                                missing_y_pred_flag = False
                                offset_pending = False
                                offset_detected_time = 0
                        onset_flag = False
                        offset_pending = False
                        offset_detected_time = 0

                # Classification
                if class_flag:
                    waited_class_time -= 1
                    if waited_class_time == 0:
                        class_flag = False
                        if len(raw_audio_chunks) >= input_chunks:
                            class_data = np.concatenate(raw_audio_chunks[-input_chunks:])
                            f, t_spec, Sxx_taf = spectrogram(
                                class_data, fs=frame_rate, nperseg=nperseg, noverlap=noverlap, nfft=nfft
                            )
                            # Ensure f is a numpy array
                            f = np.array(f)
                            # Frequency selection
                            freq_mask = (f >= lowcut) & (f <= highcut)
                            Sxx_taf = Sxx_taf[freq_mask, :]
                            Sxx_normalized = normalize_spectrogram(Sxx_taf)
                            input_data = torch.tensor(Sxx_normalized).float().unsqueeze(0).unsqueeze(0).to(device)
                            # Apply padding
                            input_tensor = F.pad(input_data, (0, 1, 0, 1))

                            # Classification
                            output = class_model(input_tensor)
                            predicted_class = int_to_label[torch.argmax(output).item()]
                            if not missing_y_pred_flag:
                                pred_syl_list.append(
                                    predicted_class)  # this is the list of syllables that go into the not.mat file
                                pred_syl_list_for_playback.append(
                                    predicted_class)  # this is the list of syllables that are compared with the target sequence
                            else:
                                missing_y_pred_flag = False
                                
                            # if targeting is True, check if the last sequence is equal to the targeted sequence
                            if targeting:
                                # Check if last sequence is equal to targeted sequence for potential playback
                                if targeted_sequence and check_targeted_sequence(pred_syl_list_for_playback,
                                                                                targeted_sequence):
                                    trigger_time = index_to_seconds(bout_indexes_waited, chunk_size, frame_rate) * 1000 + (
                                                t_before * 1000)
                                    formatted_time = millisecond_to_fixed_notation(trigger_time)

                                    # not_catch trial: choose feedback and put time in rec file,
                                    # catch trial: writes theoretical playback time into rec file
                                    if not_catch_trial_flag:
                                        if computer_generated_white_noise:
                                            # playback of computer generated white noise
                                            play_white_noise(
                                                white_noise_duration * 1000)
                                            file_path = 'computer_generated_WN'
                                            sound_duration = white_noise_duration
                                            logger.info(f"WN is {np.round(sound_duration * 1000, 0)}ms long")
                                        elif not computer_generated_white_noise:
                                            # playback of random stimulus in the playback folder of bird from the config
                                            # file path
                                            file_path = random.choice(list(playback_sounds.keys()))
                                            play_playback_file(file_path)
                                            sound_duration = np.round(len(playback_sound) / frame_rate, 3)
                                            logger.info(f"Playback sound is {np.round(sound_duration * 1000, 0)}ms long")
        
                                        # add the stimulus as entry '-' to the list so after the playback the list has an entry
                                        # of the playback given. It should not compare the last predicted syllables with
                                        # the target sequence as if nothing happened and play back the stimulus again
                                        pred_syl_list_for_playback.append('-')
                                        no_classify_flag_wn = True  # wait to classify during playback or white noise
                                        template_value_wn = 0  # Adjust as necessary
                                        wn_recfile_dict[formatted_time] = f"FB # {file_path} : Templ = {template_value_wn}"
                                        # no classification during playback or white noise
                                        no_classify_flag_wn_idx2wait = int(
                                            seconds_to_index(sound_duration + trigger_time_offset, chunk_size, frame_rate))
                                    # when catch trials, still writes theoretical playback time into rec file
                                    elif not not_catch_trial_flag:
                                        template_value_catch = 0  # Adjust as necessary
                                        wn_recfile_dict[
                                            formatted_time] = f"catch # catch_file.wav : Templ = {template_value_catch}"

                        # After classification, check if an offset is pending
                        if offset_pending:
                            # Process the pending offset - offset condition was met when detected
                            # Use the originally detected time, not current time
                            sub_y_long_rev = y_pred_list[::-1]
                            try:
                                idx = sub_y_long_rev.index(1)
                            except ValueError:
                                idx = 0
                            offset_time = ((index_to_seconds(offset_detected_time - idx, chunk_size,
                                                             frame_rate)) * 1000) + (t_before * 1000)
                            offsets.append(offset_time)
                            min_silent_index2wait = int(seconds_to_index(min_silent_duration, chunk_size, frame_rate))
                            min_silent_waited = False
                            logger.debug("min_silent_index2wait: %s", min_silent_index2wait)

                            last_duration = offsets[-1] - onsets[-1]
                            len_offset = len(offsets)
                            len_ypred = len(pred_syl_list)
                            if last_duration < (min_syllable_length * 1000):
                                # Remove the last onset, offset, and corresponding pred_syl_list entry
                                onsets.pop()
                                offsets.pop()
                                if class_flag:
                                    # Short syllable is discarded while classification is pending.
                                    # Abort current classification to avoid adding a stale label.
                                    class_flag = False
                                    waited_class_time = 0
                                    offset_pending = False
                                    offset_detected_time = 0
                                    missing_y_pred_flag = False
                                    logger.debug("Discarded short syllable and aborted pending classification")
                                elif len_ypred == len_offset:
                                    pred_syl_list.pop()
                                    pred_syl_list_for_playback.pop()
                                elif len_ypred == (len_offset - 1):
                                    # No label exists yet for this discarded syllable.
                                    logger.debug("Discarded short syllable had no label to remove")
                                else:
                                    logger.warning(
                                        "Mismatch detected during short-syllable discard; resyncing state "
                                        "(onsets=%s offsets=%s labels=%s)",
                                        len(onsets), len(offsets), len(pred_syl_list)
                                    )
                                    complete_syllables = min(len(onsets), len(offsets), len(pred_syl_list))
                                    onsets = onsets[:complete_syllables]
                                    offsets = offsets[:complete_syllables]
                                    pred_syl_list = pred_syl_list[:complete_syllables]
                                    pred_syl_list_for_playback = pred_syl_list.copy()
                                    class_flag = False
                                    waited_class_time = 0
                                    missing_y_pred_flag = False
                                    offset_pending = False
                                    offset_detected_time = 0
                            onset_flag = False
                            offset_pending = False
                            offset_detected_time = 0

    # Output handling (Playback)
    if is_playing_white_noise:
        # Calculate how many white noise samples are left to play
        samples_remaining = len(white_noise) - white_noise_index
        if samples_remaining >= frames:
            # If enough samples remain, fill the output buffer with the next chunk of white noise
            outdata[:] = white_noise[white_noise_index:white_noise_index + frames].reshape(-1, 1)
            white_noise_index += frames  # Move index forward by the number of frames just played
            if white_noise_index >= len(white_noise):
                is_playing_white_noise = False
                white_noise_index = 0
        else:
            # If not enough samples remain to fill the whole buffer:
            # - Fill part of the buffer with the remaining white noise
            # - Fill the rest with zeros (silence)
            outdata[:samples_remaining] = white_noise[white_noise_index:].reshape(-1, 1)
            outdata[samples_remaining:frames - samples_remaining] = 0
            # White noise playback is finished, reset state for future use
            is_playing_white_noise = False
            white_noise_index = 0

    elif is_playing_playback_file:
        # Calculate how many playback sound samples are left to play
        samples_remaining = len(playback_sound) - playback_sound_index
        if samples_remaining >= frames:
            # If enough samples remain, fill the output buffer with the next chunk of playback sound
            outdata[:] = playback_sound[playback_sound_index:playback_sound_index + frames].reshape(-1, 1)
            playback_sound_index += frames  # Move index forward by the number of frames just played
            if playback_sound_index >= len(playback_sound):
                # playback is finished, reset state for future use
                is_playing_playback_file = False
                playback_sound_index = 0
        else:
            # If not enough samples remain to fill the whole buffer:
            # - Fill part of the buffer with the remaining playback sound
            # - Fill the rest with zeros (silence)
            outdata[:samples_remaining] = playback_sound[playback_sound_index:].reshape(-1, 1)
            outdata[samples_remaining:] = 0
            # playback is finished, reset state for future use
            is_playing_playback_file = False
            playback_sound_index = 0
    else:
        outdata.fill(0)


frame_rate, channels, input_chunks = None, None, None
bandpass_numerator_coeffs, bandpass_denominator_coeffs, zi = None, None, None


def list_audio_devices():
    """Lists available audio devices."""
    print(sd.query_devices())


def select_input_output_devices():
    """Prompts user to select input and output audio devices."""
    list_audio_devices()
    input_device_index = int(input("Select the desired input device (Microphone): "))
    output_device_index = int(input("Select the desired output device (Speaker): "))

    # Get information about the input device
    input_device_info = sd.query_devices(input_device_index, 'input')
    output_device_info = sd.query_devices(output_device_index, 'output')

    # Set global variables for frame rate and channels
    global frame_rate, channels, input_chunks
    frame_rate = int(input_device_info['default_samplerate'])
    input_channels = int(input_device_info['max_input_channels'])
    output_channels = int(output_device_info['max_output_channels'])
    channels = (input_channels, output_channels)
    input_chunks = input_length

    return input_device_index, output_device_index


def setup_audio_stream(input_device_index, output_device_index):
    """Sets up the audio stream for recording and playback."""
    global frame_rate, channels, bandpass_numerator_coeffs, bandpass_denominator_coeffs, zi, white_noise, audio_buffer

    stream = sd.Stream(
        device=(input_device_index, output_device_index),
        samplerate=frame_rate,
        blocksize=chunk_size,
        dtype=np.float32,
        channels=channels,
        callback=stream_callback
    )

    bandpass_numerator_coeffs, bandpass_denominator_coeffs = butter_bandpass_coeffs(bandpass_lowcut, bandpass_highcut,
                                                                                    frame_rate, bandpass_order)
    zi = lfilter_zi(bandpass_numerator_coeffs, bandpass_denominator_coeffs)

    num_samples = int(frame_rate * white_noise_duration)
    white_noise = np.random.randn(num_samples).astype(np.float32)
    audio_buffer = np.zeros((frame_rate,), dtype=np.float32)

    return stream


def process_live_audio():
    """Processes live audio input and handles recording and playback."""
    try:
        # Create output folder
        daily_initialization(data_output_folder_path, experiment_name, bird_name)

        # Select input and output devices
        input_device_index, output_device_index = select_input_output_devices()
        logger.info(f"Selected input device: {input_device_index}")
        logger.info(f"Selected output device: {output_device_index}")
        logger.info(f"Frame rate: {frame_rate}")
        logger.info(f"Number of channels: {channels}")

        stream = setup_audio_stream(input_device_index, output_device_index)
        stream.start()

        logger.info("Audio recording started")
        while True:
            time.sleep(0.1)

    except KeyboardInterrupt:
        stream.stop()
        stream.close()
        logger.info("Audio recording stopped")


# Global buffer and playback status
is_playing_white_noise = False
is_playing_playback_file = False
white_noise_index = 0
playback_sound_index = 0
white_noise = None
playback_sound = None


# Start processing live audio
def main():
    process_live_audio()


if __name__ == "__main__":
    main()
