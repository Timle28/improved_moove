"""Shared fixtures for GUI tests.

Generates a small synthetic test environment with multiple bout files
(WAV + .rec + .not.mat) inside a ``bird_test/experiment_a/day_1`` tree
so that the GUI can start without touching real user data.
"""
import configparser
import os
import shutil
import textwrap

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.io import savemat
from scipy.io.wavfile import write as wav_write


# ---------------------------------------------------------------------------
# Synthetic test-data helpers
# ---------------------------------------------------------------------------

SAMPLE_RATE = 44100


def _make_sine_wav(path, freq=3000, duration_s=1.0, sr=SAMPLE_RATE):
    """Write a short WAV file containing a pure sine tone."""
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    data = (np.sin(2 * np.pi * freq * t) * 16384).astype(np.int16)
    wav_write(str(path), sr, data)
    return data


def _make_rec(path, sr=SAMPLE_RATE, n_samples=44100, hand_seg=0, hand_cls=0):
    """Write a minimal .rec file."""
    content = textwrap.dedent(f"""\
        File created: 2026-01-01

            begin rec = 0 ms
            trig time  = 2000 ms
            rec end = {int(n_samples / sr * 1000)} ms

        ADFREQ = {sr}
        Chans = 1
        Samples = {n_samples}
        Catch Song = 0
        Hand Segmented = {hand_seg}
        Hand Classified = {hand_cls}
        T Before = 2.0
        T After = 1.0
        Feedback information:
    """)
    with open(path, "w") as f:
        f.write(content)


def _make_notmat(path, onsets_ms, offsets_ms, labels, sr=SAMPLE_RATE):
    """Write a .not.mat file with given segments."""
    save_dict = {
        "__header__": b"MATLAB 5.0 MAT-file",
        "__version__": "1.0",
        "__globals__": [],
        "Fs": np.float64(sr),
        "fname": os.path.basename(path).replace(".not.mat", ""),
        "labels": labels,
        "onsets": np.array(onsets_ms, dtype=np.float64).reshape(-1, 1),
        "offsets": np.array(offsets_ms, dtype=np.float64).reshape(-1, 1),
        "min_int": np.array([], dtype=np.float64),
        "min_dur": np.array([], dtype=np.float64),
        "threshold": np.float64(-50),
        "sm_win": np.float64(2),
    }
    savemat(str(path), save_dict, do_compression=True)


# ---------------------------------------------------------------------------
# Bout specs:  (name, duration_s, segments, hand_seg, hand_cls)
# ---------------------------------------------------------------------------

BOUT_SPECS = [
    {
        "name": "bout_1.wav",
        "duration_s": 1.0,
        "freq": 3000,
        "segments": [(100, 200, "a"), (350, 450, "b"), (600, 700, "a")],
        "hand_seg": 0,
        "hand_cls": 0,
    },
    {
        "name": "bout_2.wav",
        "duration_s": 1.5,
        "freq": 4000,
        "segments": [(80, 180, "c"), (300, 420, "d")],
        "hand_seg": 1,
        "hand_cls": 0,
    },
    {
        "name": "bout_3.wav",
        "duration_s": 0.8,
        "freq": 2500,
        "segments": [(50, 150, "a"), (200, 350, "b"), (400, 500, "c"), (600, 700, "a")],
        "hand_seg": 1,
        "hand_cls": 1,
    },
]


def _populate_day(day_dir):
    """Create WAV + .rec + .not.mat for every bout spec, plus batch.txt."""
    os.makedirs(day_dir, exist_ok=True)
    filenames = []
    for spec in BOUT_SPECS:
        wav_path = os.path.join(day_dir, spec["name"])
        n_samples = int(spec["duration_s"] * SAMPLE_RATE)
        _make_sine_wav(wav_path, freq=spec["freq"], duration_s=spec["duration_s"])

        rec_path = os.path.splitext(wav_path)[0] + ".rec"
        _make_rec(rec_path, n_samples=n_samples,
                  hand_seg=spec["hand_seg"], hand_cls=spec["hand_cls"])

        onsets = [s[0] for s in spec["segments"]]
        offsets = [s[1] for s in spec["segments"]]
        labels = "".join(s[2] for s in spec["segments"])
        _make_notmat(wav_path + ".not.mat", onsets, offsets, labels)

        filenames.append(spec["name"])

    batch_path = os.path.join(day_dir, "batch.txt")
    with open(batch_path, "w") as f:
        f.write("\n".join(sorted(filenames)))


# ---------------------------------------------------------------------------
# pytest fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def _test_data_root(tmp_path_factory):
    """Build the rec_data tree once per test session (read-only template)."""
    root = tmp_path_factory.mktemp("moove_test")

    rec_data = root / "rec_data" / "bird_test" / "experiment_a" / "day_1"
    _populate_day(str(rec_data))

    # Create other required sub-directories
    for d in ("trained_models", "training_data", "cluster_data", "playbacks"):
        (root / d).mkdir(parents=True, exist_ok=True)

    # Write a minimal config
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
    """))

    return root


@pytest.fixture()
def test_env(tmp_path, _test_data_root):
    """Per-test copy of the test data so each test starts clean.

    Returns a dict with:
        global_dir  – root of the .moove tree  (str)
        rec_data    – …/rec_data               (str)
        day_dir     – …/rec_data/bird_test/experiment_a/day_1  (str)
        config_path – path to moove_config.ini  (str)
    """
    dest = tmp_path / "moove_home"
    shutil.copytree(str(_test_data_root), str(dest))

    return {
        "global_dir": str(dest),
        "rec_data": str(dest / "rec_data"),
        "day_dir": str(dest / "rec_data" / "bird_test" / "experiment_a" / "day_1"),
        "config_path": str(dest / "moove_config.ini"),
    }


@pytest.fixture(scope="session")
def gui_window(_test_data_root, qapp):
    """Single MooveMainWindow for the entire test session.

    macOS limits the number of concurrent Metal/OpenGL contexts.
    FigureCanvasQTAgg allocates one per window and they are not properly
    freed even after close()/deleteLater(), leading to segfaults after
    ~4-8 windows.  Sharing one window across all GUI tests avoids this.
    Tests that modify state (navigation, checkboxes, radios) must reset
    it before or after their assertions.
    """
    from unittest.mock import patch
    import moove.moovegui as moovegui_mod
    from moove.moovegui import MooveMainWindow

    cfg = configparser.ConfigParser()
    cfg.read(str(_test_data_root / "moove_config.ini"))

    with patch.object(moovegui_mod, "_global_dir", str(_test_data_root)), \
         patch.object(moovegui_mod, "_config", cfg):
        window = MooveMainWindow()
    window.show()
    qapp.processEvents()

    yield window

    window.close()
