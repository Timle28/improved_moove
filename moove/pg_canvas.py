# pg_canvas.py
"""pyqtgraph rendering canvas for moove's spectrogram + amplitude plots.

Replaces the matplotlib FigureCanvas. pyqtgraph composites the spectrogram image
and amplitude curve on the GPU-backed QGraphicsView, so pan/zoom is smooth and
nothing has to be hidden during motion.

Interaction (matches the requested scheme):
  * plain mouse wheel        -> nothing
  * Shift + wheel            -> pan time (x)
  * Cmd(mac)/Ctrl(win)+wheel -> zoom time (x), anchored at cursor
  * Option(Alt) + wheel      -> zoom y of the hovered plot, anchored at cursor
  * two-finger trackpad drag -> pan (x, and y of the hovered plot)
  * pinch                    -> zoom time (x), anchored at cursor
  * left-drag (None mode)    -> rubber-band box-zoom
  * clicks (edit modes)      -> segment editing (handled by syllable_utils)
  * drag the threshold line / double-click amplitude -> set threshold

The three ViewBoxes are exposed as ax1/ax2/ax3 adapters with matplotlib-style
get_xlim/set_xlim/get_ylim/set_ylim so the rest of the app is unchanged.
"""
import numpy as np
import matplotlib.cm as cm
import pyqtgraph as pg
from pyqtgraph import Point
from pyqtgraph.Qt import QtCore
from PyQt6.QtCore import Qt, QEvent
from PyQt6.QtGui import QNativeGestureEvent
from PyQt6.QtWidgets import QApplication

# White background + black axes/text, matching the original matplotlib GUI
# (pyqtgraph otherwise defaults to a black background).
pg.setConfigOptions(imageAxisOrder="row-major", antialias=False,
                    background="w", foreground="k")

_JET_LUT = (cm.jet(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.ubyte)
_AMP_COLOR = "#2653c5"
_SEG_COLOR = "#000000"
_THR_COLOR = "#d62728"
_HOVER_COLOR = "#000000"

_SCROLL_ZOOM = 1.1          # zoom factor per wheel/pinch notch (gentle)
_WHEEL_PAN_FRAC = 0.2       # Shift+wheel pan, fraction of view width per notch


class _AxisAdapter:
    """matplotlib-style get/set xlim/ylim over pyqtgraph ViewBoxes.

    X always goes through the shared master viewbox so the three (x-linked)
    plots stay bit-identical in x; Y uses each plot's own viewbox."""

    def __init__(self, xvb, yvb):
        self.xvb = xvb
        self.yvb = yvb

    def get_xlim(self):
        (x0, x1), _ = self.xvb.viewRange()
        return x0, x1

    def set_xlim(self, x0, x1):
        self.xvb.setXRange(x0, x1, padding=0)

    def get_ylim(self):
        _, (y0, y1) = self.yvb.viewRange()
        return y0, y1

    def set_ylim(self, y0, y1):
        self.yvb.setYRange(y0, y1, padding=0)

    # no-ops so matplotlib-era call sites keep working
    def set_navigate(self, *a):
        pass

    def clear(self):
        pass


class _EditViewBox(pg.ViewBox):
    """ViewBox that defers wheel handling to the canvas and routes mouse
    clicks/drags to the app's editing logic. Left-drag in 'None' edit mode is a
    rubber-band box-zoom (pyqtgraph RectMode)."""

    def __init__(self, canvas, axis_id, **kw):
        super().__init__(**kw)
        self._canvas = canvas
        self.axis_id = axis_id          # 'spec' | 'label' | 'amp'
        self.setMenuEnabled(False)
        self.setMouseMode(pg.ViewBox.RectMode)

    def wheelEvent(self, ev, axis=None):
        ev.ignore()                     # the GraphicsView owns wheel/gesture

    def mouseDragEvent(self, ev, axis=None):
        if self._canvas.app_state.edit_type == "None":
            super().mouseDragEvent(ev, axis)   # box-zoom
        else:
            ev.ignore()                  # edit modes use clicks, not drags

    def mouseClickEvent(self, ev):
        if self._canvas._on_view_click(self, ev):
            ev.accept()
        else:
            ev.ignore()


class MoovePgCanvas(pg.GraphicsLayoutWidget):
    def __init__(self, app_state, parent=None):
        super().__init__(parent)
        self.app_state = app_state
        self.setBackground("w")
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._radio_adapter = None
        # Height the segment bars/crosses are pinned to: the threshold the file
        # was last *segmented* at. Stays fixed when the live threshold (red line)
        # is moved by double-click/drag, so you can compare the two.
        self._seg_threshold = None

        self.vb_spec = _EditViewBox(self, "spec")
        self.vb_lbl = _EditViewBox(self, "label")
        self.vb_amp = _EditViewBox(self, "amp")

        # Row 0: file-name title (own row, so it doesn't shrink the spectrogram).
        self.title = pg.LabelItem(justify="center", color="k", size="24pt")
        self.addItem(self.title, row=0, col=0)
        self.p_spec = self.addPlot(row=1, col=0, viewBox=self.vb_spec)
        self.p_lbl = self.addPlot(row=2, col=0, viewBox=self.vb_lbl)
        self.p_amp = self.addPlot(row=3, col=0, viewBox=self.vb_amp)
        # Equal stretch for spectrogram and amplitude; amplitude carries the
        # bottom time axis, so give it a touch more to even out the plot areas.
        self.ci.layout.setRowStretchFactor(1, 20)
        self.ci.layout.setRowStretchFactor(2, 1)
        self.ci.layout.setRowStretchFactor(3, 27)
        # Keep the middle label row a short strip regardless of window height.
        self.ci.layout.setRowMaximumHeight(2, 34)

        for p in (self.p_spec, self.p_lbl, self.p_amp):
            p.showGrid(x=False, y=False)
            p.setMenuEnabled(False)
            # Full rectangular frame around each plot, like matplotlib's default
            # spines (this is the box that "windows" the label letters).
            p.getViewBox().setBorder(pg.mkPen("k", width=1))
        _lbl_style = {"color": "#000000", "font-size": "20pt"}
        self.p_spec.setLabel("left", "Frequency (Hz)", **_lbl_style)
        self.p_amp.setLabel("left", "Amplitude (dB)", **_lbl_style)
        self.p_amp.setLabel("bottom", "Time (s)", **_lbl_style)
        self.p_lbl.getAxis("left").setStyle(showValues=False)
        self.p_lbl.getAxis("left").setWidth(self.p_spec.getAxis("left").width())
        self.p_lbl.setYRange(0, 1, padding=0)
        self.p_lbl.hideAxis("bottom")
        self.p_spec.hideAxis("bottom")

        self.p_lbl.setXLink(self.p_spec)
        self.p_amp.setXLink(self.p_spec)

        # Manual ranges only: auto-range would re-fit (with padding) after every
        # setData/paint and undo our explicit zoom/pan ranges.
        for vb in (self.vb_spec, self.vb_lbl, self.vb_amp):
            vb.disableAutoRange()
            vb.setDefaultPadding(0.0)

        # ---- items ----
        self.img = pg.ImageItem()
        self.img.setLookupTable(_JET_LUT)
        self.p_spec.addItem(self.img)
        self.amp_curve = self.p_amp.plot(pen=pg.mkPen(_AMP_COLOR, width=2.5))
        self.amp_curve.setDownsampling(auto=True, method="peak")
        self.amp_curve.setClipToView(True)
        self.seg_bars = self.p_amp.plot(pen=pg.mkPen(_SEG_COLOR, width=1.5), connect="finite")
        self.seg_markers = pg.ScatterPlotItem(symbol="+", size=10, pen=pg.mkPen(_SEG_COLOR), brush=pg.mkBrush(_SEG_COLOR))
        self.p_amp.addItem(self.seg_markers)
        self.sel_marker = pg.ScatterPlotItem(symbol="+", size=12, pen=pg.mkPen(_THR_COLOR, width=2), brush=pg.mkBrush(_THR_COLOR))
        self.p_amp.addItem(self.sel_marker)

        self.thr_line = pg.InfiniteLine(angle=0, movable=True,
                                        pen=pg.mkPen(_THR_COLOR, width=1.5, style=Qt.PenStyle.DashLine),
                                        hoverPen=pg.mkPen(_THR_COLOR, width=2.5))
        self.p_amp.addItem(self.thr_line)
        self.thr_line.sigPositionChangeFinished.connect(self._on_threshold_dragged)


        # hover guide on the amplitude plot
        self.hover_line = pg.InfiniteLine(angle=0, movable=False,
                                          pen=pg.mkPen(_HOVER_COLOR, width=0.8))
        self.hover_label = pg.TextItem(color=_HOVER_COLOR, anchor=(1, 1))
        self.hover_line.setVisible(False)
        self.hover_label.setVisible(False)
        self.p_amp.addItem(self.hover_line, ignoreBounds=True)
        self.p_amp.addItem(self.hover_label, ignoreBounds=True)
        self.scene().sigMouseMoved.connect(self._on_mouse_moved)
        self.scene().sigMouseClicked.connect(self._on_scene_click)

        self.label_items = []
        self.ax1 = _AxisAdapter(self.vb_spec, self.vb_spec)
        self.ax2 = _AxisAdapter(self.vb_spec, self.vb_lbl)
        self.ax3 = _AxisAdapter(self.vb_spec, self.vb_amp)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def render(self, display_dict, app_state, vmin, vmax):
        """Full render for a newly loaded file (replaces update_plots)."""
        from moove.utils.audio_utils import decibel

        def _num(v):
            return float(v.get() if hasattr(v, "get") else v)

        freqs = display_dict["freqs"]
        lo = _num(app_state.config['lower_spec_plot'])
        hi = _num(app_state.config['upper_spec_plot'])
        valid = (freqs >= lo) & (freqs <= hi)
        f = freqs[valid]
        db = decibel(np.sqrt(display_dict["spectrogram_data"][valid, :]))
        t = display_dict["times"]
        t0, t1 = float(t[0]), float(t[-1])
        f0, f1 = float(f.min()), float(f.max())

        # Float images REQUIRE levels in pyqtgraph; fall back to percentiles.
        if vmin is None or vmax is None:
            vmin = float(np.percentile(db, 2))
            vmax = float(np.percentile(db, 99))
        self.img.setImage(db, levels=(vmin, vmax), autoLevels=False)
        self.img.setRect(QtCore.QRectF(t0, f0, t1 - t0, f1 - f0))

        amp = np.asarray(display_dict["amplitude"], dtype=float)
        x = np.arange(amp.shape[0]) / display_dict["sampling_rate"]
        self.amp_curve.setData(x, amp)
        amin, amax = float(amp.min()), float(amp.max())

        self.title.setText(str(display_dict.get("file_name", "")), color="k", size="24pt")

        # axis ranges + clamp limits
        self.vb_spec.setLimits(xMin=t0, xMax=t1, yMin=f0, yMax=f1)
        self.vb_amp.setLimits(xMin=t0, xMax=t1, yMin=amin, yMax=amax)
        self.vb_spec.setRange(xRange=(t0, t1), yRange=(f0, f1), padding=0)
        self.vb_amp.setYRange(amin, amax, padding=0.02)

        app_state.set_original_x_range((t0, t1))
        app_state.set_original_y_range_ax1((f0, f1), (0, 1), (amin, amax))

        # Pin the segment bars/crosses to this file's segmentation threshold.
        self._seg_threshold = self._seg_height(display_dict, app_state)
        self._draw_threshold(app_state)
        self._draw_segments(display_dict, app_state)
        self._draw_labels(display_dict)

    def update_amp_labels(self, display_dict, app_state):
        """Redraw amplitude + labels + segments, keeping the x range
        (replaces update_ax2_ax3, called after edits/undo)."""
        amp = np.asarray(display_dict["amplitude"], dtype=float)
        x = np.arange(amp.shape[0]) / display_dict["sampling_rate"]
        self.amp_curve.setData(x, amp)
        self.clear_selected()
        self._draw_threshold(app_state)
        self._draw_segments(display_dict, app_state)
        self._draw_labels(display_dict)

    def _threshold_value(self, app_state):
        try:
            return float(app_state.evfuncs_params['threshold'].get())
        except (ValueError, KeyError, TypeError, AttributeError):
            return None

    def _seg_height(self, display_dict, app_state):
        """Threshold the file was segmented at: the value stored in the .not.mat
        if present, else the current live threshold (snapshotted at load time)."""
        t = display_dict.get("threshold")
        try:
            arr = np.ravel(np.asarray(t, dtype=float))
            if arr.size and np.isfinite(arr[0]):
                return float(arr[0])
        except (TypeError, ValueError):
            pass
        return self._threshold_value(app_state)

    def _draw_threshold(self, app_state):
        v = self._threshold_value(app_state)
        if v is not None:
            self.thr_line.blockSignals(True)
            self.thr_line.setValue(v)
            self.thr_line.blockSignals(False)

    def _draw_segments(self, display_dict, app_state):
        onsets = display_dict.get("onsets")
        offsets = display_dict.get("offsets")
        # Pinned to the segmentation threshold, NOT the live (red-line) value.
        h = self._seg_threshold
        if h is None:
            h = self._threshold_value(app_state)
        n = 0 if onsets is None or offsets is None else min(len(onsets), len(offsets))
        if n == 0 or h is None:
            self.seg_bars.setData([], [])
            self.seg_markers.setData([], [])
            return
        on = np.asarray(onsets[:n], dtype=float) / 1000.0
        off = np.asarray(offsets[:n], dtype=float) / 1000.0
        xs, ys = [], []
        for i in range(n):
            xs += [on[i], off[i], np.nan]
            ys += [h, h, np.nan]
        self.seg_bars.setData(xs, ys)
        bx = np.concatenate([on, off])
        self.seg_markers.setData(bx, np.full(bx.shape, h))

    def _draw_labels(self, display_dict):
        for it in self.label_items:
            self.p_lbl.removeItem(it)
        self.label_items = []
        if "labels" not in display_dict:
            return
        onsets = display_dict.get("onsets", [])
        offsets = display_dict.get("offsets", [])
        labels = display_dict["labels"]
        n = min(len(onsets), len(offsets), len(labels))
        for i in range(n):
            cx = (onsets[i] + offsets[i]) / 2000.0
            it = pg.TextItem(text=str(labels[i]), color="#000000", anchor=(0.5, 0.5))
            it.setPos(cx, 0.5)
            self.p_lbl.addItem(it)
            self.label_items.append(it)

    def set_levels(self, vmin, vmax):
        self.img.setLevels((vmin, vmax))

    # no-ops for the old matplotlib canvas API
    def draw(self):
        pass

    def draw_idle(self):
        pass

    def set_radio_adapter(self, adapter):
        self._radio_adapter = adapter

    # label highlighting (used by syllable editing) ---------------------
    def set_label_color(self, idx, color):
        if 0 <= idx < len(self.label_items):
            self.label_items[idx].setColor(color)

    def set_label_text(self, idx, text):
        if 0 <= idx < len(self.label_items):
            self.label_items[idx].setText(str(text))

    def label_count(self):
        return len(self.label_items)

    def mark_selected(self, time_s, height):
        self.sel_marker.setData([time_s], [height])

    def clear_selected(self):
        self.sel_marker.setData([], [])

    # ------------------------------------------------------------------
    # Threshold drag / set
    # ------------------------------------------------------------------
    def _on_threshold_dragged(self):
        # Only the red line moves; segment bars stay pinned to _seg_threshold.
        value = round(float(self.thr_line.value()), 1)
        try:
            self.app_state.evfuncs_params['threshold'].set(str(value))
        except (KeyError, AttributeError):
            return
        self.app_state.logger.info("Threshold set to %.1f dB (drag).", value)

    # ------------------------------------------------------------------
    # Hover readout (amplitude plot)
    # ------------------------------------------------------------------
    def _on_mouse_moved(self, scene_pos):
        if not self.vb_amp.sceneBoundingRect().contains(scene_pos):
            if self.hover_line.isVisible():
                self.hover_line.setVisible(False)
                self.hover_label.setVisible(False)
            return
        p = self.vb_amp.mapSceneToView(scene_pos)
        self.hover_line.setPos(p.y())
        self.hover_line.setVisible(True)
        (x0, x1), _ = self.vb_amp.viewRange()
        self.hover_label.setText(f"{p.y():.1f} dB")
        self.hover_label.setPos(x1, p.y())
        self.hover_label.setVisible(True)

    # ------------------------------------------------------------------
    # Mouse clicks -> editing
    # ------------------------------------------------------------------
    def _on_scene_click(self, ev):
        # double-click on the amplitude plot sets the threshold
        if ev.double() and self.vb_amp.sceneBoundingRect().contains(ev.scenePos()):
            p = self.vb_amp.mapSceneToView(ev.scenePos())
            value = round(float(p.y()), 1)
            try:
                self.app_state.evfuncs_params['threshold'].set(str(value))
            except (KeyError, AttributeError):
                return
            # Move only the red line; keep the segment bars pinned where they
            # were segmented, so both levels are visible for comparison.
            if self.app_state.display_dict is not None:
                self._draw_threshold(self.app_state)
            self.app_state.logger.info("Threshold set to %.1f dB (double-click).", value)

    def _on_view_click(self, vb, ev):
        """Translate a pyqtgraph click into the app's edit handlers.
        Returns True if handled."""
        from moove.utils.syllable_utils import select_event

        s = self.app_state
        if s.edit_type == "None":
            return False
        p = vb.mapSceneToView(ev.scenePos())
        btn = 1 if ev.button() == Qt.MouseButton.LeftButton else (
              3 if ev.button() == Qt.MouseButton.RightButton else 0)
        if btn == 0:
            return False
        inaxes = {"spec": s.ax1, "label": s.ax2, "amp": s.ax3}[vb.axis_id]
        adapter = _ClickEvent(xdata=p.x(), ydata=p.y(), inaxes=inaxes, button=btn,
                              x=ev.scenePos().x(), y=ev.scenePos().y())
        select_event(adapter, s)
        return True

    # ------------------------------------------------------------------
    # Wheel + native gesture -> pan / zoom (the requested scheme)
    # ------------------------------------------------------------------
    def wheelEvent(self, e):
        # macOS often drops the Option/Alt modifier from the wheel event itself,
        # so OR in the live keyboard state to detect it reliably.
        mods = e.modifiers() | QApplication.queryKeyboardModifiers()
        pd = e.pixelDelta()
        ad = e.angleDelta()
        is_trackpad = not pd.isNull()
        raw = (pd.y() or pd.x()) if is_trackpad else (ad.y() or ad.x())
        zoom_steps = raw / 120.0    # original zoom direction
        pan_steps = -raw / 120.0    # reverted scroll/pan direction
        if not raw:
            e.accept()
            return
        if mods & Qt.KeyboardModifier.ControlModifier:      # Cmd/Ctrl -> zoom x
            self._zoom_x(self._scroll_factor(zoom_steps), e.position())
        elif mods & Qt.KeyboardModifier.AltModifier:        # Option/Alt -> zoom y
            self._zoom_y(self._scroll_factor(zoom_steps), e.position())
        elif mods & Qt.KeyboardModifier.ShiftModifier:      # Shift -> pan x
            self._pan_x_frac(pan_steps * _WHEEL_PAN_FRAC)
        elif is_trackpad:                                   # two-finger -> pan
            self._pan_pixels(pd.x(), pd.y(), e.position())
        # plain mouse wheel: nothing
        e.accept()

    def event(self, e):
        if e.type() == QEvent.Type.NativeGesture and isinstance(e, QNativeGestureEvent):
            if e.gestureType() == Qt.NativeGestureType.ZoomNativeGesture:
                factor = 1.0 / (1.0 + e.value())
                self._zoom_x(factor, e.position())
                return True
        return super().event(e)

    # ------------------------------------------------------------------
    # Keyboard -> app handlers (mode shortcuts, undo/redo, label editing)
    # ------------------------------------------------------------------
    def keyPressEvent(self, e):
        if e.key() == Qt.Key.Key_Space:          # space -> play current view
            from moove.utils.movefuncs_utils import handle_playback
            handle_playback(self.app_state)
            e.accept()
            return
        key = self._mpl_key(e)
        if key is None:
            super().keyPressEvent(e)
            return
        from moove.utils.syllable_utils import handle_keypress, edit_syllable
        adapter = _KeyEvent(key)
        handle_keypress(adapter, self.app_state, self._radio_adapter)
        edit_syllable(adapter, self.app_state)
        e.accept()

    _SPECIAL_KEYS = None

    def _mpl_key(self, e):
        if MoovePgCanvas._SPECIAL_KEYS is None:
            MoovePgCanvas._SPECIAL_KEYS = {
                Qt.Key.Key_Left: "left", Qt.Key.Key_Right: "right",
                Qt.Key.Key_Up: "up", Qt.Key.Key_Down: "down",
                Qt.Key.Key_Escape: "escape", Qt.Key.Key_Backspace: "backspace",
                Qt.Key.Key_Delete: "delete", Qt.Key.Key_Return: "enter",
                Qt.Key.Key_Enter: "enter",
            }
        base = self._SPECIAL_KEYS.get(e.key())
        if base is None:
            txt = e.text()
            if txt and txt.isprintable() and txt.strip():
                base = txt.lower()
        if base is None:
            return None
        mods = []
        m = e.modifiers()
        if m & Qt.KeyboardModifier.ControlModifier:   # Cmd on macOS / Ctrl on Win
            mods.append("ctrl")
        if m & Qt.KeyboardModifier.AltModifier:
            mods.append("alt")
        if m & Qt.KeyboardModifier.ShiftModifier:
            mods.append("shift")
        return "+".join(mods + [base])

    @staticmethod
    def _scroll_factor(steps):
        return _SCROLL_ZOOM ** -steps

    def _scene_view_x(self, widget_pos):
        scene_pos = self.mapToScene(widget_pos.toPoint())
        return self.vb_spec.mapSceneToView(scene_pos).x()

    def _zoom_x(self, factor, widget_pos):
        cx = self._scene_view_x(widget_pos)
        self.vb_spec.scaleBy(s=[factor, 1.0], center=Point(cx, 0.0))

    def _vb_under(self, widget_pos):
        """Spectrogram or amplitude viewbox under the cursor, else None."""
        sp = self.mapToScene(widget_pos.toPoint())
        if self.vb_spec.sceneBoundingRect().contains(sp):
            return self.vb_spec
        if self.vb_amp.sceneBoundingRect().contains(sp):
            return self.vb_amp
        return None

    def _zoom_y(self, factor, widget_pos):
        vb = self._vb_under(widget_pos)
        if vb is None:
            return
        cy = vb.mapSceneToView(self.mapToScene(widget_pos.toPoint())).y()
        vb.scaleBy(s=[1.0, factor], center=Point(0.0, cy))

    def _pan_x_frac(self, frac):
        (x0, x1), _ = self.vb_spec.viewRange()
        self.vb_spec.translateBy(x=frac * (x1 - x0))

    def _pan_pixels(self, dx_px, dy_px, widget_pos):
        if dx_px:
            xscale = self.vb_spec.viewPixelSize()[0]
            self.vb_spec.translateBy(x=-dx_px * xscale)
        if dy_px:
            scene_pos = self.mapToScene(widget_pos.toPoint())
            vb = self.vb_amp if self.vb_amp.sceneBoundingRect().contains(scene_pos) else None
            if vb is not None:
                yscale = vb.viewPixelSize()[1]
                vb.translateBy(y=dy_px * yscale)


class _ClickEvent:
    """Lightweight stand-in for a matplotlib MouseEvent, for edit handlers."""
    __slots__ = ("xdata", "ydata", "inaxes", "button", "x", "y", "key", "dblclick")

    def __init__(self, xdata, ydata, inaxes, button, x, y):
        self.xdata = xdata
        self.ydata = ydata
        self.inaxes = inaxes
        self.button = button
        self.x = x
        self.y = y
        self.key = None
        self.dblclick = False


class _KeyEvent:
    """Lightweight stand-in for a matplotlib KeyEvent."""
    __slots__ = ("key", "inaxes")

    def __init__(self, key):
        self.key = key
        self.inaxes = None
