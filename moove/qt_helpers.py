# qt_helpers.py
"""PyQt6 helper utilities: thread-safe invoker, custom range slider, combo helpers."""
import functools
from PyQt6.QtCore import QObject, QEvent, QTimer, pyqtSignal, pyqtSlot, Qt
from PyQt6.QtGui import QPainter, QColor, QFont, QFontMetrics
from PyQt6.QtWidgets import QWidget, QApplication, QMessageBox
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg


class _Invoker(QObject):
    """Singleton helper that executes callables on the main GUI thread.

    Must be created on the main thread, and uses QueuedConnection so that
    emits from worker threads are dispatched to the main event loop.
    """
    _call = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        app = QApplication.instance()
        if app is not None:
            self.moveToThread(app.thread())
        self._call.connect(self._execute, Qt.ConnectionType.QueuedConnection)

    @pyqtSlot(object)
    def _execute(self, fn):
        fn()


_invoker_instance = None


def invoke_in_main_thread(fn, *args, **kwargs):
    """Schedule *fn* to run on the main (GUI) thread. Thread-safe."""
    global _invoker_instance
    if _invoker_instance is None:
        _invoker_instance = _Invoker()
    if args or kwargs:
        fn = functools.partial(fn, *args, **kwargs)
    _invoker_instance._call.emit(fn)


def _app_icon_pixmap(size=64):
    """Return the application icon as a QPixmap, or None."""
    app = QApplication.instance()
    if app is None:
        return None
    icon = app.windowIcon()
    if icon.isNull():
        return None
    return icon.pixmap(size, size)


def show_info(parent, title, message):
    """Show a QMessageBox.information with the moove icon (not the Python rocket)."""
    if parent is not None and not parent.isVisible():
        parent = None
    box = QMessageBox(QMessageBox.Icon.NoIcon, title, message,
                      QMessageBox.StandardButton.Ok, parent)
    pix = _app_icon_pixmap()
    if pix is not None:
        box.setWindowIcon(QApplication.instance().windowIcon())
        box.setIconPixmap(pix)
    box.exec()


def show_confirm_action_window(parent, title, message):
    """Show a confirmation dialog. Returns True if user clicks OK, False otherwise."""
    if parent is not None and not parent.isVisible():
        parent = None

    box = QMessageBox(
        QMessageBox.Icon.Question,
        title,
        message,
        QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel,
        parent
    )

    box.setDefaultButton(QMessageBox.StandardButton.Ok)

    pix = _app_icon_pixmap()
    if pix is not None:
        box.setWindowIcon(QApplication.instance().windowIcon())
        box.setIconPixmap(pix)

    result = box.exec()

    return result == QMessageBox.StandardButton.Ok


def set_combo_items(combo, items, current_text=None):
    """Replace all items in a QComboBox, optionally selecting one."""
    if combo is None:
        return
    combo.blockSignals(True)
    combo.clear()
    combo.addItems(items)
    if current_text is not None:
        idx = combo.findText(current_text)
        if idx >= 0:
            combo.setCurrentIndex(idx)
        else:
            combo.setCurrentText(current_text)
    combo.blockSignals(False)


class RadioAdapter:
    """Wraps a QButtonGroup to provide a .get()/.set() interface."""
    def __init__(self, button_group):
        self._bg = button_group

    def set(self, value):
        btn = self._bg.button(int(value))
        if btn:
            btn.setChecked(True)

    def get(self):
        checked = self._bg.checkedButton()
        if checked:
            return str(self._bg.id(checked))
        return "1"


class QRangeSliderV(QWidget):
    """Vertical dual-handle range slider for spectrogram vmin/vmax control."""
    valuesChanged = pyqtSignal(float, float)

    _TRACK_WIDTH = 6
    _HANDLE_RADIUS = 9
    _MARGIN = 14

    def __init__(self, min_val, max_val, bottom_val, top_val, parent=None):
        super().__init__(parent)
        self._min = float(min_val)
        self._max = float(max_val)
        self._bottom = float(bottom_val)
        self._top = float(top_val)
        self._dragging = None
        self.setMinimumWidth(90)
        self.setMinimumHeight(100)
        self._font = QFont("Arial", 12, QFont.Weight.Bold)

    # -- public API --
    def bottom(self):
        return self._bottom

    def top(self):
        return self._top

    def setBottom(self, v):
        v = max(self._min, min(v, self._top))
        if v != self._bottom:
            self._bottom = v
            self.update()
            self.valuesChanged.emit(self._bottom, self._top)

    def setTop(self, v):
        v = min(self._max, max(v, self._bottom))
        if v != self._top:
            self._top = v
            self.update()
            self.valuesChanged.emit(self._bottom, self._top)

    # -- coordinate helpers --
    def _val_to_y(self, val):
        """Map a value to a y pixel coordinate (top of widget = max, bottom = min)."""
        h = self.height() - 2 * self._MARGIN
        if self._max == self._min:
            return self._MARGIN
        frac = (val - self._min) / (self._max - self._min)
        return self._MARGIN + h * (1.0 - frac)

    def _y_to_val(self, y):
        h = self.height() - 2 * self._MARGIN
        if h <= 0:
            return self._min
        frac = 1.0 - (y - self._MARGIN) / h
        return self._min + frac * (self._max - self._min)

    # -- painting --
    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setFont(self._font)
        fm = QFontMetrics(self._font)
        cx = self.width() // 2

        # track
        track_top = self._MARGIN
        track_bot = self.height() - self._MARGIN
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(200, 200, 200))
        p.drawRoundedRect(cx - self._TRACK_WIDTH // 2, track_top,
                          self._TRACK_WIDTH, track_bot - track_top,
                          self._TRACK_WIDTH // 2, self._TRACK_WIDTH // 2)

        # selected range highlight
        y_bot = self._val_to_y(self._bottom)
        y_top = self._val_to_y(self._top)
        p.setBrush(QColor(80, 130, 200))
        p.drawRect(cx - self._TRACK_WIDTH // 2, int(y_top),
                   self._TRACK_WIDTH, int(y_bot - y_top))

        # handles
        for y_pos, color in [(y_bot, QColor(60, 60, 200)), (y_top, QColor(200, 60, 60))]:
            p.setBrush(color)
            p.setPen(QColor(40, 40, 40))
            p.drawEllipse(int(cx - self._HANDLE_RADIUS), int(y_pos - self._HANDLE_RADIUS),
                          self._HANDLE_RADIUS * 2, self._HANDLE_RADIUS * 2)

        # value labels (right of handles, vertically centered on dot)
        p.setPen(QColor(0, 0, 0))
        label_x = cx + self._HANDLE_RADIUS + 6
        text_y_offset = fm.ascent() // 2 - 1
        p.drawText(label_x, int(y_top + text_y_offset), f"{self._top:.0f}")
        p.drawText(label_x, int(y_bot + text_y_offset), f"{self._bottom:.0f}")

        p.end()

    # -- mouse interaction --
    def mousePressEvent(self, event):
        y = event.position().y()
        y_bot = self._val_to_y(self._bottom)
        y_top = self._val_to_y(self._top)
        dist_bot = abs(y - y_bot)
        dist_top = abs(y - y_top)
        threshold = self._HANDLE_RADIUS * 2.5
        if dist_bot < threshold or dist_top < threshold:
            self._dragging = 'bottom' if dist_bot < dist_top else 'top'

    def mouseMoveEvent(self, event):
        if self._dragging is None:
            return
        val = self._y_to_val(event.position().y())
        val = max(self._min, min(self._max, val))
        if self._dragging == 'bottom':
            val = max(self._min, min(val, self._top))
            if val != self._bottom:
                self._bottom = val
                self.update()
                self.valuesChanged.emit(self._bottom, self._top)
        else:
            val = min(self._max, max(val, self._bottom))
            if val != self._top:
                self._top = val
                self.update()
                self.valuesChanged.emit(self._bottom, self._top)

    def mouseReleaseEvent(self, event):
        self._dragging = None


class GestureCanvas(FigureCanvasQTAgg):
    """matplotlib canvas with native trackpad pinch-zoom and two-finger pan.

    Gestures (all clamped to the data extent stored on app_state):
      * pinch                -> zoom time (x) on the spectrogram and amplitude
                               plots together, anchored under the cursor.
      * Shift + pinch        -> zoom y of the plot under the cursor.
      * two-finger drag        -> pan: horizontal = time (x), vertical = y of the
                                 plot under the cursor.
      * plain mouse wheel       -> nothing.
      * Shift + mouse wheel     -> pan time (x) left/right.
      * Cmd/Ctrl + mouse wheel  -> zoom time (x), anchored under the cursor.

    (Cmd on macOS and Ctrl on Windows both arrive as Qt ControlModifier.)
    Existing buttons and the rectangle box-zoom keep working unchanged.
    """

    # Zoom per scroll notch (a notch is delta == 120). Multiplicative, so the
    # absolute step scales with the current view width -> consistent feel at any
    # zoom level. Kept gentle ("short distance") on purpose.
    _SCROLL_ZOOM = 1.1
    # Pan per Shift+wheel notch, as a fraction of the current view width
    # (normalized, so it stays gentle and proportional at every zoom level).
    _WHEEL_PAN_FRAC = 0.2
    # While interacting, render at this fraction of the real device-pixel-ratio
    # (lower = faster but blurrier during motion; snaps crisp when it settles).
    _COARSE_DPR_SCALE = 0.5

    def __init__(self, figure, app_state):
        super().__init__(figure)
        self._app_state = app_state
        self._coarse = False
        self._full_dpr = self.device_pixel_ratio
        # After a continuous gesture (pinch / two-finger pan) settles, restore
        # the full-resolution spectrogram and do one crisp redraw.
        self._hidden_artists = []
        self._settle = QTimer(self)
        self._settle.setSingleShot(True)
        self._settle.setInterval(160)
        self._settle.timeout.connect(self._exit_coarse)

    # -- Qt event hooks ------------------------------------------------------
    def event(self, e):
        if e.type() == QEvent.Type.NativeGesture:
            if self._handle_native_gesture(e):
                return True
        return super().event(e)

    def wheelEvent(self, e):
        pos = e.position()
        mods = e.modifiers()
        pd = e.pixelDelta()
        ad = e.angleDelta()
        # Modifiers decide the action FIRST, before trackpad-vs-mouse -- otherwise
        # Cmd+two-finger-scroll (which carries a pixelDelta) gets swallowed by the
        # trackpad-pan branch and never zooms.
        is_trackpad = not pd.isNull()
        raw = (pd.y() or pd.x()) if is_trackpad else (ad.y() or ad.x())
        steps = raw / 120.0
        if not steps:
            e.accept()
            return
        if mods & Qt.KeyboardModifier.ControlModifier:    # Cmd (mac) / Ctrl (win) -> zoom x
            self._zoom_x(self._SCROLL_ZOOM ** -steps, pos, coarse=True)
        elif mods & Qt.KeyboardModifier.ShiftModifier:    # Shift -> pan x
            self._pan_x(steps * self._WHEEL_PAN_FRAC, coarse=True)
        elif is_trackpad:                                 # plain two-finger -> pan
            self._pan(pd.x(), pd.y(), pos)
        # plain mouse wheel: do nothing
        e.accept()

    def _handle_native_gesture(self, e):
        if e.gestureType() != Qt.NativeGestureType.ZoomNativeGesture:
            return False
        scale = 1.0 / (1.0 + e.value())  # value > 0 => zoom in => narrower view
        # Continuous gesture -> render coarse while moving, crisp once settled.
        if QApplication.keyboardModifiers() & Qt.KeyboardModifier.ShiftModifier:
            self._zoom_y(scale, e.position(), coarse=True)
        else:
            self._zoom_x(scale, e.position(), coarse=True)
        return True

    # -- gesture actions -----------------------------------------------------
    def _zoom_x(self, scale, pos, coarse=False):
        s = self._app_state
        ax = s.ax1
        if ax is None or not self._gestures_ready():
            return
        x0, x1 = ax.get_xlim()
        if x1 == x0:
            return
        left, _ = self._axes_frac(pos, ax)   # cursor fraction across the axis
        xdata = x0 + left * (x1 - x0)
        new_w = (x1 - x0) * scale
        lo, hi = self._clamp(xdata - left * new_w, xdata - left * new_w + new_w,
                             s.original_x_range)
        for a in (s.ax1, s.ax2, s.ax3):
            if a is not None:
                a.set_xlim(lo, hi)
        self._live_draw(coarse)

    def _zoom_y(self, scale, pos, coarse=False):
        ax, bounds = self._hovered_axis(pos)
        if ax is None:
            return
        y0, y1 = ax.get_ylim()
        if y1 == y0:
            return
        _, bottom = self._axes_frac(pos, ax)
        ydata = y0 + bottom * (y1 - y0)
        new_h = (y1 - y0) * scale
        lo, hi = self._clamp(ydata - bottom * new_h, ydata - bottom * new_h + new_h,
                             bounds)
        ax.set_ylim(lo, hi)
        self._live_draw(coarse)

    def _pan_x(self, frac, coarse=False):
        """Pan time (x) by *frac* of the current view width (Shift+wheel)."""
        s = self._app_state
        ax = s.ax1
        if ax is None or not self._gestures_ready():
            return
        x0, x1 = ax.get_xlim()
        shift = frac * (x1 - x0)
        lo, hi = self._clamp(x0 + shift, x1 + shift, s.original_x_range)
        for a in (s.ax1, s.ax2, s.ax3):
            if a is not None:
                a.set_xlim(lo, hi)
        self._live_draw(coarse)

    def _pan(self, dx_px, dy_px, pos):
        # Continuous trackpad gesture -> render coarse while moving.
        s = self._app_state
        ax1 = s.ax1
        if ax1 is None or not self._gestures_ready():
            return
        # logical widget size (DPR-independent), pixelDelta is in logical px too
        w = max(self.width(), 1)
        h = max(self.height(), 1)
        # horizontal -> time (x) on all linked axes
        if dx_px:
            x0, x1 = ax1.get_xlim()
            ax_w = ax1.get_position().width * w
            shift = dx_px * (x1 - x0) / max(ax_w, 1)  # content follows fingers
            lo, hi = self._clamp(x0 - shift, x1 - shift, s.original_x_range)
            for a in (s.ax1, s.ax2, s.ax3):
                if a is not None:
                    a.set_xlim(lo, hi)
        # vertical -> y of the hovered plot
        if dy_px:
            ax, bounds = self._hovered_axis(pos)
            if ax is not None:
                y0, y1 = ax.get_ylim()
                ax_h = ax.get_position().height * h
                shift = dy_px * (y1 - y0) / max(ax_h, 1)
                lo, hi = self._clamp(y0 + shift, y1 + shift, bounds)
                ax.set_ylim(lo, hi)
        self._live_draw(coarse=True)

    # -- helpers -------------------------------------------------------------
    def _gestures_ready(self):
        return self._app_state.original_x_range is not None

    def _axes_frac(self, pos, ax):
        """Cursor as (fx, fy) fractions within *ax*.

        Uses the logical widget size and the axis' figure-fraction position, so
        it is independent of the device-pixel-ratio -- which we lower mid-gesture
        for speed. (Doing this via physical pixels / figure.bbox would make the
        zoom anchor drift whenever the DPR changes.)"""
        w = max(self.width(), 1)
        h = max(self.height(), 1)
        bb = ax.get_position()
        fx = (pos.x() / w - bb.x0) / max(bb.width, 1e-9)
        fy = ((1.0 - pos.y() / h) - bb.y0) / max(bb.height, 1e-9)
        return fx, fy

    def _hovered_axis(self, pos):
        """Return (axis, original_y_bounds) for the plot under the cursor."""
        s = self._app_state
        fy_fig = 1.0 - pos.y() / max(self.height(), 1)
        for ax, bounds in ((s.ax1, s.original_y_range_ax1),
                           (s.ax3, s.original_y_range_ax3)):
            if ax is None:
                continue
            bb = ax.get_position()
            if bb.y0 <= fy_fig <= bb.y1:
                return ax, bounds
        return None, None

    @staticmethod
    def _clamp(lo, hi, bounds):
        """Keep [lo, hi] inside *bounds*, preserving width when possible."""
        if bounds is None:
            return lo, hi
        bmin, bmax = (bounds[0], bounds[1]) if bounds[0] <= bounds[1] else (bounds[1], bounds[0])
        flip = lo > hi
        if flip:
            lo, hi = hi, lo
        width = hi - lo
        if width >= bmax - bmin:
            lo, hi = bmin, bmax
        elif lo < bmin:
            lo, hi = bmin, bmin + width
        elif hi > bmax:
            lo, hi = bmax - width, bmax
        return (hi, lo) if flip else (lo, hi)

    def _live_draw(self, coarse=False):
        # Single coalesced async redraw per frame. draw_idle() collapses a burst
        # of scroll/gesture events into one render at the next event-loop tick,
        # which is what keeps zoom/pan fast. No synchronous draw, no cache work.
        if coarse:
            self._enter_coarse()
            self._settle.start()  # restore full-res once the gesture settles
        self.draw_idle()

    def _enter_coarse(self):
        """Render fast during a live gesture. Three levers, biggest first:
          1. hide the expensive overlays (amplitude line, segment bars, labels)
             -- at a large window these dominate the frame time;
          2. drop the device-pixel-ratio (output-pixel cost);
          3. swap in a time-decimated spectrogram.
        All reversed in _exit_coarse, which then does one crisp redraw."""
        if self._coarse:
            return
        self._coarse = True
        self._hide_overlays()
        self._full_dpr = self.device_pixel_ratio
        try:
            self._set_device_pixel_ratio(self._full_dpr * self._COARSE_DPR_SCALE)
        except Exception:
            pass
        s = self._app_state
        if s.spec is not None and s.spec_low_data is not None:
            try:
                s.spec.set_data(s.spec_low_data)
            except Exception:
                pass

    def _exit_coarse(self):
        """Restore full resolution and do one crisp redraw."""
        if not self._coarse:
            return
        self._coarse = False
        try:
            self._set_device_pixel_ratio(self._full_dpr)
        except Exception:
            pass
        s = self._app_state
        if s.spec is not None and s.spec_full_data is not None:
            try:
                s.spec.set_data(s.spec_full_data)
            except Exception:
                pass
        self._show_overlays()
        self.draw_idle()

    def _hide_overlays(self):
        """Hide the amplitude line, segment bars/markers and syllable labels
        during a gesture (the spectrogram stays visible). Skips animated artists
        (the hover guide) so we don't disturb them."""
        s = self._app_state
        cands = []
        if s.ax2 is not None:
            cands += list(s.ax2.texts)
        if s.ax3 is not None:
            cands += [ln for ln in s.ax3.lines if not ln.get_animated()]
            cands += list(s.ax3.collections)
        self._hidden_artists = [a for a in cands if a.get_visible()]
        for a in self._hidden_artists:
            a.set_visible(False)

    def _show_overlays(self):
        for a in self._hidden_artists:
            a.set_visible(True)
        self._hidden_artists = []


class AmplitudeHover:
    """Hover readout for the amplitude plot.

    Draws a horizontal guide line that follows the cursor across the amplitude
    axis with a live "xx.x dB" label, so the user can hover to read off a good
    segmentation threshold. Uses blitting so it stays smooth over the (static)
    spectrogram. Re-creates its artists automatically after ax.clear().
    """

    def __init__(self, canvas, ax):
        self.canvas = canvas
        self.ax = ax
        self._bg = None
        self.line = None
        self.label = None
        self._ensure_artists()
        canvas.mpl_connect('draw_event', self._on_draw)
        canvas.mpl_connect('motion_notify_event', self._on_move)
        canvas.mpl_connect('axes_leave_event', self._on_leave)

    def _ensure_artists(self):
        """(Re)create the guide artists if a plot redraw cleared the axis."""
        if self.line is None or self.line not in self.ax.lines:
            self.line = self.ax.axhline(
                self.ax.get_ylim()[0], color='#555555', linewidth=0.8,
                linestyle='-', alpha=0.9, animated=True, visible=False)
        if self.label is None or self.label not in self.ax.texts:
            self.label = self.ax.text(
                0.995, 0, '', transform=self.ax.get_yaxis_transform(),
                ha='right', va='bottom', fontsize=9, color='#333333',
                animated=True, visible=False,
                bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='#555555', alpha=0.85))

    def _on_draw(self, _event):
        # Just invalidate the cached background; recapture it lazily on the next
        # hover. Capturing here would add a copy_from_bbox to *every* canvas
        # draw (e.g. each zoom frame), which is exactly the overhead we avoid.
        self._ensure_artists()
        self._bg = None

    def _capture_bg(self):
        # The guide artists are animated=True, so a normal draw never renders
        # them -> the current buffer is already a clean background.
        try:
            self._bg = self.canvas.copy_from_bbox(self.ax.bbox)
        except Exception:
            self._bg = None
        return self._bg

    def _blit_clear(self):
        if self._bg is not None:
            self.canvas.restore_region(self._bg)
            self.canvas.blit(self.ax.bbox)

    def _on_leave(self, event):
        if event.inaxes is self.ax:
            self.line.set_visible(False)
            self.label.set_visible(False)
            self._blit_clear()

    def _on_move(self, event):
        # Only when hovering the amplitude axis with no button held (so we don't
        # fight the rectangle box-zoom, which does its own blitting).
        if event.inaxes is not self.ax or event.button is not None:
            return
        if event.ydata is None:
            return
        if self._bg is None and self._capture_bg() is None:
            return
        self.canvas.restore_region(self._bg)
        self.line.set_ydata([event.ydata, event.ydata])
        self.line.set_visible(True)
        self.label.set_position((0.995, event.ydata))
        self.label.set_text(f'{event.ydata:.1f} dB')
        self.label.set_visible(True)
        self.ax.draw_artist(self.line)
        self.ax.draw_artist(self.label)
        self.canvas.blit(self.ax.bbox)
