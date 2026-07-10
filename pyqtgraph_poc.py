"""Throwaway pyqtgraph proof-of-concept for moove plot smoothness.

Loads one of your real recordings the same way the app does, and renders the
spectrogram + amplitude with pyqtgraph instead of matplotlib so you can feel the
pan/zoom smoothness before we commit to a full migration.

Run:
    uv run python pyqtgraph_poc.py            # uses the last file from app_state.json
    uv run python pyqtgraph_poc.py /path/to/file.wav

Interactions (pyqtgraph built-ins, GPU-composited -- no re-rasterizing per frame):
    * drag            -> pan
    * scroll wheel    -> zoom (x+y); right-drag -> zoom
    * drag the RED line on the amplitude plot -> set threshold (live readout)
    * the title bar shows a live FPS counter during interaction.

Nothing here touches your data or the app; it's read-only.
"""
import json
import os
import sys
import time

import numpy as np
import matplotlib.cm as cm
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

from moove.utils.file_utils import get_display_data
from moove.utils.audio_utils import decibel

LOWER_HZ, UPPER_HZ = 500, 12500


def resolve_file():
    if len(sys.argv) > 1:
        fp = os.path.abspath(sys.argv[1])
        return os.path.basename(fp), fp
    state = json.load(open(os.path.expanduser("~/.moove/app_state.json")))
    ddir, sf = state["data_dir"], state["song_files"]
    idx = min(state.get("current_file_index", 0), len(sf) - 1)
    return sf[idx], os.path.join(ddir, sf[idx])


def load(file_name, file_path):
    dd = get_display_data({"file_name": file_name, "file_path": file_path}, None)
    freqs = dd["freqs"]
    valid = (freqs >= LOWER_HZ) & (freqs <= UPPER_HZ)
    db = decibel(np.sqrt(dd["spectrogram_data"][valid, :]))
    f = freqs[valid]
    t = dd["times"]
    amp = np.asarray(dd["amplitude"], dtype=float)
    x = np.arange(amp.shape[0]) / dd["sampling_rate"]
    return db, (float(t[0]), float(t[-1])), (float(f.min()), float(f.max())), x, amp


class FPSLayout(pg.GraphicsLayoutWidget):
    """GraphicsLayoutWidget that counts real repaints to report interaction FPS."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._frames = 0

    def paintEvent(self, ev):
        super().paintEvent(ev)
        self._frames += 1

    def pop_fps(self, dt):
        fps = self._frames / dt if dt else 0
        self._frames = 0
        return fps


def main():
    file_name, file_path = resolve_file()
    db, (t0, t1), (f0, f1), x, amp = load(file_name, file_path)

    try:
        state = json.load(open(os.path.expanduser("~/.moove/app_state.json")))
        vmin = state.get("current_vmin") or -60.0
        vmax = state.get("current_vmax") or -10.0
    except Exception:
        vmin, vmax = float(np.percentile(db, 5)), float(np.percentile(db, 99))

    pg.setConfigOptions(imageAxisOrder="row-major", antialias=False)
    app = pg.mkQApp("moove pyqtgraph POC")

    win = FPSLayout()
    win.resize(1600, 900)
    win.setWindowTitle(f"moove pyqtgraph POC — {file_name}")

    # --- spectrogram ---------------------------------------------------------
    p1 = win.addPlot(row=0, col=0)
    p1.setLabel("left", "Frequency (Hz)")
    p1.getViewBox().setMouseEnabled(x=True, y=True)
    img = pg.ImageItem()
    img.setImage(db, levels=(vmin, vmax))
    lut = (cm.jet(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.ubyte)
    img.setLookupTable(lut)
    img.setRect(QtCore.QRectF(t0, f0, t1 - t0, f1 - f0))
    p1.addItem(img)
    p1.setTitle("Spectrogram — drag=pan, wheel=zoom")

    # --- amplitude -----------------------------------------------------------
    p2 = win.addPlot(row=1, col=0)
    p2.setLabel("left", "Amplitude (dB)")
    p2.setLabel("bottom", "Time (s)")
    p2.setXLink(p1)                       # share x with the spectrogram
    curve = p2.plot(x, amp, pen=pg.mkPen("#2653c5"))
    curve.setDownsampling(auto=True, method="peak")  # full 483k pts, drawn cheap
    curve.setClipToView(True)
    p2.setTitle("Amplitude — drag the RED line to set threshold")

    thr = pg.InfiniteLine(
        angle=0, movable=True, pos=(vmin + vmax) / 2,
        pen=pg.mkPen("#d62728", width=2), hoverPen=pg.mkPen("#ff0000", width=3))
    pg.InfLineLabel(thr, text="{value:.1f} dB", position=0.04,
                    color="#d62728", fill=(255, 255, 255, 200))
    p2.addItem(thr)

    # set sensible initial ranges (full file)
    p1.setXRange(t0, t1, padding=0)
    p1.setYRange(f0, f1, padding=0)
    p2.setYRange(float(amp.min()), float(amp.max()), padding=0.02)

    win.show()

    # --- live FPS in the title bar ------------------------------------------
    last = {"t": time.perf_counter()}

    def tick():
        now = time.perf_counter()
        fps = win.pop_fps(now - last["t"])
        last["t"] = now
        win.setWindowTitle(
            f"moove pyqtgraph POC — {file_name} — {fps:4.0f} fps during interaction "
            f"(spec {db.shape[1]}x{db.shape[0]}, amp {amp.shape[0]:,} pts)")

    timer = QtCore.QTimer()
    timer.timeout.connect(tick)
    timer.start(500)

    app.exec()


if __name__ == "__main__":
    main()
