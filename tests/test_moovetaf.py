"""Tests for MooveTAF – the real-time recording / bout-detection script.

moovetaf.py reads config and sets globals at import time.
Tests cover pure functions and stream_callback bout-detection logic.
"""
import configparser
import datetime
import os
import shutil
import textwrap
import threading
import time

import numpy as np
import pytest
from scipy.io import loadmat
from scipy.io.wavfile import write as wav_write


# =====================================================================
# Helpers – import pure functions directly from the module.
# We need to set env var MOOVE_CONFIG_DIR *before* importing moovetaf
# so the module reads our test config instead of ~/.moove.
# =====================================================================

@pytest.fixture(scope="session")
def taf_config_dir(tmp_path_factory):
    """Create a temporary moove config + directory tree for TAF tests."""
    root = tmp_path_factory.mktemp("moove_taf")

    for d in ("rec_data", "trained_models", "playbacks"):
        (root / d).mkdir(parents=True, exist_ok=True)

    cfg = root / "moove_config.ini"
    cfg.write_text(textwrap.dedent(f"""\
        [GENERAL]
        global_dir = {root}

        [GUI]
        upper_spec_plot = 12500
        lower_spec_plot = 500
        vmin_range_slider = -140
        vmax_range_slider = -10
        spec_nperseg = 1024
        spec_noverlap = 896
        spec_nfft = 1024
        performance = fast

        [TAF]
        bird_name = bird_test
        experiment_name = experiment_a
        frame_rate = 44100
        chunk_size = 64
        t_before = 2
        t_after = 1
        min_bout_duration = 6
        memory_cleanup_interval = 60
        input_channel = 0

        [bird_test]
        bout_threshold_db = -25
        window_size = 10
        bandpass_lowcut = 500
        bandpass_highcut = 15000
        bandpass_order = 2
        realtime_classification = False
        segmentation_model_name = dummy_seg_model.pth
        classification_model_name = dummy_class_model.pth
        targeting = False
        targeted_sequence = none
        catch_trial_probability = 0.1
        white_noise_duration = 0.08
        computer_generated_white_noise = True
        playback_dir = {root / "playbacks"}
        trigger_time_offset = 0.005
        decision_threshold = 0.5
        onset_window_size = 5
        n_onset_true = 3
        offset_window_size = 5
        n_offset_false = 4
        min_syllable_length = 0.03
        min_silent_duration = 0.005
    """))

    return root


@pytest.fixture(scope="session")
def taf_module(taf_config_dir, monkeypatch_session):
    """Import moovetaf once with our test config.

    Uses a session-scoped monkeypatch to set MOOVE_CONFIG_DIR before import.
    """
    monkeypatch_session.setenv("MOOVE_CONFIG_DIR", str(taf_config_dir))
    import importlib
    import moove.moovetaf as taf
    importlib.reload(taf)  # force re-read of config
    # moovetaf.py line ~905 overwrites frame_rate/channels/input_chunks with None
    # (they're meant to be set by select_input_output_devices at runtime).
    # Restore the config values so tests can use them.
    import configparser as _cp
    _cfg = _cp.ConfigParser()
    _cfg.read(os.path.join(str(taf_config_dir), "moove_config.ini"))
    taf.frame_rate = int(_cfg.get("TAF", "frame_rate"))
    return taf


@pytest.fixture(scope="session")
def monkeypatch_session():
    """Session-scoped monkeypatch (pytest only provides function-scoped)."""
    from _pytest.monkeypatch import MonkeyPatch
    mp = MonkeyPatch()
    yield mp
    mp.undo()


# =====================================================================
# Layer 1 – Pure function tests
# =====================================================================

class TestSecondsIndexConversions:
    def test_seconds_to_index_basic(self, taf_module):
        # 1 second at chunk_size=64, sr=44100  →  44100//64 = 689
        idx = taf_module.seconds_to_index(1.0, 64, 44100)
        assert idx == 689

    def test_index_to_seconds_basic(self, taf_module):
        secs = taf_module.index_to_seconds(689, 64, 44100)
        assert abs(secs - (689 * 64 / 44100)) < 1e-9

    def test_roundtrip(self, taf_module):
        idx = taf_module.seconds_to_index(2.5, 64, 44100)
        secs = taf_module.index_to_seconds(idx, 64, 44100)
        assert abs(secs - 2.5) < (64 / 44100)  # within one chunk


class TestCalculateDB:
    def test_silence_is_very_low(self, taf_module):
        silence = np.zeros(1024, dtype=np.float32)
        db = taf_module.calculate_db(silence)
        assert db <= -180  # essentially -inf, clamped by 1e-10 guard

    def test_full_scale_sine(self, taf_module):
        t = np.linspace(0, 1, 44100, endpoint=False)
        sine = np.sin(2 * np.pi * 1000 * t).astype(np.float32)
        db = taf_module.calculate_db(sine)
        # RMS of sine = 1/sqrt(2), dB ≈ -3.01
        assert -4.0 < db < -2.0

    def test_positive_for_loud_signal(self, taf_module):
        loud = np.ones(1024, dtype=np.float32) * 10.0
        db = taf_module.calculate_db(loud)
        assert db > 0


class TestGenerateWhiteNoise:
    def test_correct_length(self, taf_module):
        wn = taf_module.generate_white_noise(100, 44100)  # 100 ms
        expected = int(44100 * 100 / 1000)
        assert len(wn) == expected

    def test_dtype_float32(self, taf_module):
        wn = taf_module.generate_white_noise(50, 44100)
        assert wn.dtype == np.float32

    def test_not_all_zeros(self, taf_module):
        wn = taf_module.generate_white_noise(100, 44100)
        assert np.any(wn != 0)


class TestButterworthBandpass:
    def test_coeffs_are_arrays(self, taf_module):
        b, a = taf_module.butter_bandpass_coeffs(500, 15000, 44100, order=2)
        assert len(b) > 0 and len(a) > 0

    def test_filter_application(self, taf_module):
        from scipy.signal import lfilter_zi
        b, a = taf_module.butter_bandpass_coeffs(500, 15000, 44100, order=2)
        zi = lfilter_zi(b, a)
        data = np.random.randn(1024).astype(np.float32)
        filtered, zf = taf_module.apply_butter_bandpass_filter(data, b, a, zi)
        assert filtered.shape == data.shape
        assert zf.shape == zi.shape


class TestNormalizeSpectrogram:
    def test_zero_mean(self, taf_module):
        spec = np.random.randn(64, 64).astype(np.float32)
        normed = taf_module.normalize_spectrogram(spec)
        assert abs(normed.mean()) < 1e-5

    def test_constant_input(self, taf_module):
        spec = np.ones((10, 10), dtype=np.float32) * 5.0
        normed = taf_module.normalize_spectrogram(spec)
        # std=0 → should return original
        assert np.allclose(normed, spec)


class TestCheckTargetedSequence:
    def test_match_found(self, taf_module):
        lst = ["a", "b", "c", "a"]
        assert taf_module.check_targeted_sequence(lst, r"ca$")

    def test_no_match(self, taf_module):
        lst = ["a", "b", "c"]
        assert not taf_module.check_targeted_sequence(lst, r"za$")

    def test_regex_pattern(self, taf_module):
        lst = ["j", "c", "a"]
        assert taf_module.check_targeted_sequence(lst, r"[jc]a$")


class TestCleanLists:
    def test_keeps_last_n(self, taf_module):
        a = list(range(100))
        b = list(range(100))
        taf_module.clean_lists([a, b], 10)
        assert len(a) == 10
        assert a == list(range(90, 100))
        assert len(b) == 10

    def test_shorter_than_n(self, taf_module):
        a = [1, 2, 3]
        taf_module.clean_lists([a], 10)
        assert a == [1, 2, 3]


class TestMillisecondFixedNotation:
    def test_format(self, taf_module):
        result = taf_module.millisecond_to_fixed_notation(1234.5)
        assert "E" in result
        # Should be reconstructable as a float
        assert abs(float(result) - 1234.5) < 0.01


class TestDailyInitialization:
    def test_creates_day_folder(self, taf_module, tmp_path):
        day_path = taf_module.daily_initialization(
            str(tmp_path), "exp_a", "bird_x"
        )
        assert os.path.isdir(day_path)
        # batch.txt created
        assert os.path.isfile(os.path.join(day_path, "batch.txt"))

    def test_idempotent(self, taf_module, tmp_path):
        p1 = taf_module.daily_initialization(str(tmp_path), "exp_a", "bird_x")
        p2 = taf_module.daily_initialization(str(tmp_path), "exp_a", "bird_x")
        assert p1 == p2


# =====================================================================
# Layer 2 – save_bout integration
# =====================================================================

class TestSaveBout:
    def test_saves_wav_rec_notmat(self, taf_module, taf_config_dir):
        """Simulate saving a bout and verify all output files."""
        sr = 44100
        chunk = 64
        t_before = 2.0
        t_after = 1.0
        duration_s = 8.0  # > min_bout_duration (6s)

        n_chunks = int(duration_s * sr / chunk)
        raw_chunks = [np.random.randn(chunk).astype(np.float32)
                      for _ in range(n_chunks)]

        bout_idx_waited = n_chunks - int(t_before * sr / chunk)
        bout_dt = datetime.datetime(2026, 3, 13, 10, 30, 0)

        wn_dict = {"catch_song": 0}
        onsets = [2100.0, 2500.0]
        offsets = [2300.0, 2700.0]
        pred_syls = ["a", "b"]

        taf_module.save_bout(
            raw_chunks, bout_idx_waited, bout_dt, wn_dict,
            onsets, offsets, pred_syls,
        )

        # Find the day folder
        day_folder = taf_module.daily_initialization(
            taf_module.data_output_folder_path,
            taf_module.experiment_name,
            taf_module.bird_name,
        )

        wav_files = [f for f in os.listdir(day_folder) if f.endswith(".wav")]
        assert len(wav_files) >= 1

        wav_name = wav_files[0]
        base = wav_name.replace(".wav", "")

        # .rec file exists
        assert os.path.isfile(os.path.join(day_folder, base + ".rec"))

        # .not.mat file exists
        notmat_path = os.path.join(day_folder, wav_name + ".not.mat")
        assert os.path.isfile(notmat_path)

        # Verify notmat contents
        mat = loadmat(notmat_path)
        assert mat["onsets"].shape[0] == 2
        assert mat["offsets"].shape[0] == 2

        # batch.txt should contain the file name
        with open(os.path.join(day_folder, "batch.txt")) as f:
            assert wav_name in f.read()


# =====================================================================
# Layer 2 – stream_callback with simulated audio
# =====================================================================

class TestStreamCallbackBoutDetection:
    """Feed synthetic audio chunks through stream_callback to exercise
    bout detection logic (realtime_classification=False path)."""

    def _make_outdata(self, chunk_size):
        return np.zeros((chunk_size, 1), dtype=np.float32)

    def _make_loud_chunk(self, chunk_size, sr, freq=1000.0, amplitude=0.5):
        """Generate a chunk containing a sine wave within the bandpass range."""
        t = np.arange(chunk_size, dtype=np.float32) / sr
        signal = (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)
        return signal.reshape(-1, 1)

    def _reset_globals(self, taf):
        """Reset the mutable global state between tests."""
        taf.raw_audio_chunks = []
        taf.db_values_list = []
        taf.bout_flag = False
        taf.bout_index2wait = 0
        taf.bout_indexes_waited = 0
        taf.bout_recdt = ""
        taf.onsets = []
        taf.offsets = []
        taf.onset_flag = False
        taf.offset_pending = False
        taf.offset_detected_time = 0
        taf.waited_class_time = 0
        taf.class_flag = False
        taf.pred_syl_list = []
        taf.pred_syl_list_for_playback = []
        taf.wn_recfile_dict = {}
        taf.not_catch_trial_flag = False
        taf.no_classify_flag_wn = False
        taf.no_classify_flag_wn_idx2wait = 0
        taf.missing_y_pred_flag = False
        taf.min_silent_index2wait = 0
        taf.min_silent_waited = True
        taf.initialization_complete = False
        taf.y_pred_list = []
        taf.is_playing_white_noise = False
        taf.is_playing_playback_file = False

    def _setup_stream_globals(self, taf):
        """Set up the globals that setup_audio_stream normally configures."""
        from scipy.signal import lfilter_zi
        sr = taf.frame_rate
        taf.channels = (1, 1)
        taf.input_chunks = None
        b, a = taf.butter_bandpass_coeffs(
            taf.bandpass_lowcut, taf.bandpass_highcut, sr, taf.bandpass_order
        )
        taf.bandpass_numerator_coeffs = b
        taf.bandpass_denominator_coeffs = a
        taf.zi = lfilter_zi(b, a)
        taf.white_noise = np.zeros(int(sr * taf.white_noise_duration), dtype=np.float32)
        taf.config_input_channel = [0]

    def test_silence_no_bout(self, taf_module):
        """Feeding silence should never trigger bout_flag."""
        taf = taf_module
        self._reset_globals(taf)
        self._setup_stream_globals(taf)

        chunk_size = taf.chunk_size
        silence = np.zeros((chunk_size, 1), dtype=np.float32)
        outdata = self._make_outdata(chunk_size)

        for _ in range(500):
            taf.stream_callback(
                silence.copy(), outdata, chunk_size, None, None
            )

        assert not taf.bout_flag

    def test_loud_signal_triggers_bout(self, taf_module):
        """A loud burst should trigger bout_flag after initialization."""
        taf = taf_module
        self._reset_globals(taf)
        self._setup_stream_globals(taf)

        chunk_size = taf.chunk_size
        outdata = self._make_outdata(chunk_size)

        # Phase 1: silence for initialization (t_before = 2s)
        silence = np.zeros((chunk_size, 1), dtype=np.float32)
        n_init = int(taf.seconds_to_index(taf.t_before, chunk_size, taf.frame_rate)) + 10
        for _ in range(n_init):
            taf.stream_callback(silence.copy(), outdata, chunk_size, None, None)

        assert taf.initialization_complete

        # Phase 2: loud signal (sine wave in bandpass range)
        for i in range(taf.window_size + 5):
            loud = self._make_loud_chunk(chunk_size, taf.frame_rate)
            taf.stream_callback(loud.copy(), outdata, chunk_size, None, None)

        assert taf.bout_flag

    def test_bout_saved_after_silence(self, taf_module, taf_config_dir):
        """After bout trigger + sufficient data + silence, save_bout is called."""
        taf = taf_module
        self._reset_globals(taf)
        self._setup_stream_globals(taf)

        chunk_size = taf.chunk_size
        outdata = self._make_outdata(chunk_size)

        # Phase 1: silence for initialization
        silence = np.zeros((chunk_size, 1), dtype=np.float32)
        n_init = int(taf.seconds_to_index(taf.t_before, chunk_size, taf.frame_rate)) + 10
        for _ in range(n_init):
            taf.stream_callback(silence.copy(), outdata, chunk_size, None, None)

        # Phase 2: loud signal for > min_bout_duration (6s)
        n_loud = int(taf.seconds_to_index(7.0, chunk_size, taf.frame_rate))
        for i in range(n_loud):
            loud = self._make_loud_chunk(chunk_size, taf.frame_rate)
            taf.stream_callback(loud.copy(), outdata, chunk_size, None, None)

        assert taf.bout_flag

        # Phase 3: silence for t_after (1s) → should trigger save
        n_silence = int(taf.seconds_to_index(taf.t_after + 0.5, chunk_size, taf.frame_rate))
        for _ in range(n_silence):
            taf.stream_callback(silence.copy(), outdata, chunk_size, None, None)

        # bout_flag should be cleared after save
        assert not taf.bout_flag

        # Give save thread time to finish
        time.sleep(0.5)

        # Check that files were written
        day_folder = taf.daily_initialization(
            taf.data_output_folder_path, taf.experiment_name, taf.bird_name
        )
        wav_files = [f for f in os.listdir(day_folder) if f.endswith(".wav")]
        assert len(wav_files) >= 1

    def test_memory_cleanup_during_silence(self, taf_module):
        """Prolonged silence should trigger list cleanup."""
        taf = taf_module
        self._reset_globals(taf)
        self._setup_stream_globals(taf)

        chunk_size = taf.chunk_size
        outdata = self._make_outdata(chunk_size)
        silence = np.zeros((chunk_size, 1), dtype=np.float32)

        # Feed enough silence to exceed memory_cleanup_interval
        n_chunks = int(taf.memory_cleanup_interval * taf.frame_rate / chunk_size) + 100
        for _ in range(n_chunks):
            taf.stream_callback(silence.copy(), outdata, chunk_size, None, None)

        # raw_audio_chunks should have been trimmed
        max_expected = int(taf.seconds_to_index(5, chunk_size, taf.frame_rate)) + 100
        assert len(taf.raw_audio_chunks) < n_chunks

    def test_initialization_requires_t_before_chunks(self, taf_module):
        """initialization_complete should only be True after t_before seconds."""
        taf = taf_module
        self._reset_globals(taf)
        self._setup_stream_globals(taf)

        chunk_size = taf.chunk_size
        outdata = self._make_outdata(chunk_size)
        silence = np.zeros((chunk_size, 1), dtype=np.float32)

        # Feed just 10 chunks (way less than t_before=2s)
        for _ in range(10):
            taf.stream_callback(silence.copy(), outdata, chunk_size, None, None)

        assert not taf.initialization_complete

        # Now feed enough
        n_needed = int(taf.seconds_to_index(taf.t_before, chunk_size, taf.frame_rate))
        for _ in range(n_needed):
            taf.stream_callback(silence.copy(), outdata, chunk_size, None, None)

        assert taf.initialization_complete
