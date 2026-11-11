"""Live webcam motion capture inside the RGB MoCap window (macOS/Linux port).

The capture worker (live/capture.py in the separate .venv-live runtime) tracks
face, body and hands with MediaPipe and sends this window two datagrams per
frame: b"P" + JSON packet and b"F" + JPEG preview. This window is the hub:

* Live: preview, Start/Stop, Pause (freeze the avatar), forward to Blender.
* Record: a take is saved as take.jsonl (one packet per frame) and frames.bin
  (the JPEG frames; each packet stores its offset), so any frame loads instantly.
* Takes: Play/Pause/Stop, scrub, speed, loop, In/Out trim, Bake to Blender
  (the Blender receiver keys the trimmed take into an Action), Delete.

The GUI never imports MediaPipe or Torch.
"""
from __future__ import annotations

import bisect
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QProcess, QTimer, Qt, QUrl
from PySide6.QtGui import QDesktopServices, QImage, QPixmap
from PySide6.QtNetwork import QHostAddress, QUdpSocket
from PySide6.QtWidgets import (QComboBox, QDockWidget, QFileDialog, QHBoxLayout, QLabel, QMessageBox,
                               QPushButton, QSizePolicy, QSlider, QVBoxLayout, QWidget, QCheckBox)

from .paths import ROOT, PROJECTS

APP_PORT = 49171
BLENDER_PORT = 49170
TAKES = PROJECTS / 'live_takes'


def list_cameras():
    """Video devices by name (AVFoundation order on macOS, which OpenCV shares)."""
    if sys.platform == 'darwin' and shutil.which('ffmpeg'):
        try:
            out = subprocess.run(['ffmpeg', '-hide_banner', '-f', 'avfoundation', '-list_devices', 'true', '-i', ''],
                                 capture_output=True, text=True, timeout=10).stderr
        except (OSError, subprocess.SubprocessError):
            out = ''
        video = out.split('audio devices')[0]
        names = re.findall(r'\[(\d+)\] (.+)', video)
        cams = [(int(i), n.strip()) for i, n in names if 'Capture screen' not in n]
        if cams:
            return cams
    return [(i, f'Camera {i}') for i in range(4)]


def blender_path():
    explicit = os.environ.get('RGBMOCAP_BLENDER')
    if explicit:
        return Path(explicit)
    from .platform_compat import blender_candidates
    found = [p for p in blender_candidates() if p.is_file()]
    which = shutil.which('blender')
    return found[-1] if found else Path(which) if which else None


class Take:
    def __init__(self, folder: Path):
        self.folder = folder
        self.packets = [json.loads(line) for line in (folder / 'take.jsonl').open(encoding='utf-8')]
        self.times = [p['t'] - self.packets[0]['t'] for p in self.packets] if self.packets else []
        meta = folder / 'take.json'
        self.meta = json.loads(meta.read_text()) if meta.exists() else {}
        self.frames = (folder / 'frames.bin').open('rb')

    @property
    def duration(self):
        return self.times[-1] if self.times else 0.0

    def jpeg(self, index):
        offset, length = self.packets[index].get('jpg', (None, None))
        if offset is None:
            return None
        self.frames.seek(offset)
        return self.frames.read(length)

    def save_meta(self):
        (self.folder / 'take.json').write_text(json.dumps(self.meta, indent=1))

    def close(self):
        self.frames.close()


class Recorder:
    def __init__(self):
        TAKES.mkdir(parents=True, exist_ok=True)
        self.folder = TAKES / datetime.now().strftime('take_%Y%m%d_%H%M%S')
        self.folder.mkdir()
        self.lines = (self.folder / 'take.jsonl').open('w', encoding='utf-8')
        self.frames = (self.folder / 'frames.bin').open('wb')
        self.count = 0; self.started = time.monotonic()

    def add(self, packet, jpeg):
        offset = self.frames.tell(); self.frames.write(jpeg)
        self.lines.write(json.dumps(dict(packet, jpg=[offset, len(jpeg)]), separators=(',', ':')) + '\n')
        self.count += 1

    def finish(self):
        self.lines.close(); self.frames.close()
        (self.folder / 'take.json').write_text(json.dumps(
            {'created': datetime.now().isoformat(timespec='seconds'), 'frames': self.count,
             'seconds': round(time.monotonic() - self.started, 2)}, indent=1))
        return self.folder


class LivePanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.process = None; self.paused = False; self.recorder = None
        self.packet = None; self.take = None; self.playing = False; self.play_pos = 0.0; self.play_clock = None
        self.received = 0; self.fps = 0.0; self._fps_mark = (time.monotonic(), 0)

        self.socket = QUdpSocket(self)
        self.socket.bind(QHostAddress.LocalHost, APP_PORT)
        self.socket.setSocketOption(QUdpSocket.ReceiveBufferSizeSocketOption, 4 << 20)
        self.socket.readyRead.connect(self.read_datagrams)
        self.out = QUdpSocket(self)

        col = QVBoxLayout(self); col.setContentsMargins(6, 6, 6, 6)
        # live row
        row = QHBoxLayout(); col.addLayout(row)
        self.camera = QComboBox(); self.camera.setToolTip('Camera')
        for index, name in list_cameras():
            self.camera.addItem(name, index)
        self.start_button = QPushButton('Start Camera'); self.start_button.clicked.connect(self.toggle_capture)
        self.pause_button = QPushButton('Pause'); self.pause_button.setCheckable(True)
        self.pause_button.setToolTip('Freeze the avatar while you reposition; preview keeps running')
        self.pause_button.toggled.connect(self.set_paused)
        self.record_button = QPushButton('● Record'); self.record_button.setCheckable(True)
        self.record_button.toggled.connect(self.toggle_record)
        self.blender_button = QPushButton('Open Avatar in Blender…')
        self.blender_button.setToolTip('Open a VRM avatar (default: Seed-san) in Blender with the live receiver')
        self.blender_button.clicked.connect(self.open_blender)
        self.send_blender = QCheckBox('Send to Blender'); self.send_blender.setChecked(True)
        for w in (self.camera, self.start_button, self.pause_button, self.record_button, self.send_blender, self.blender_button):
            row.addWidget(w)
        row.addStretch(1)
        self.status = QLabel('Choose a camera and press Start Camera.'); self.status.setObjectName('muted')
        col.addWidget(self.status)
        # preview
        self.preview = QLabel(); self.preview.setAlignment(Qt.AlignCenter); self.preview.setMinimumHeight(240)
        self.preview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.preview.setStyleSheet('background:#111; color:#888;'); self.preview.setText('No camera')
        col.addWidget(self.preview, 1)
        # takes / transport
        row = QHBoxLayout(); col.addLayout(row)
        self.takes = QComboBox(); self.takes.setMinimumWidth(220); self.takes.currentIndexChanged.connect(self.load_take)
        self.play_button = QPushButton('▶ Play'); self.play_button.clicked.connect(self.toggle_play)
        self.stop_button = QPushButton('■ Stop'); self.stop_button.clicked.connect(self.stop_play)
        self.speed = QComboBox()
        for s in ('0.25x', '0.5x', '1x', '1.5x', '2x'):
            self.speed.addItem(s, float(s[:-1]))
        self.speed.setCurrentIndex(2)
        self.loop = QCheckBox('Loop'); self.loop.setChecked(True)
        for w in (QLabel('Take'), self.takes, self.play_button, self.stop_button, self.speed, self.loop):
            row.addWidget(w)
        row.addStretch(1)
        row = QHBoxLayout(); col.addLayout(row)
        self.slider = QSlider(Qt.Horizontal); self.slider.valueChanged.connect(self.scrub)
        self.time_label = QLabel('0.00 / 0.00 s'); self.time_label.setMinimumWidth(110)
        row.addWidget(self.slider, 1); row.addWidget(self.time_label)
        row = QHBoxLayout(); col.addLayout(row)
        self.in_button = QPushButton('Set In'); self.in_button.clicked.connect(lambda: self.set_trim('in'))
        self.out_button = QPushButton('Set Out'); self.out_button.clicked.connect(lambda: self.set_trim('out'))
        self.clear_button = QPushButton('Clear Trim'); self.clear_button.clicked.connect(lambda: self.set_trim(None))
        self.bake_button = QPushButton('Bake to Blender'); self.bake_button.clicked.connect(self.bake)
        self.bake_button.setToolTip('Key the trimmed take onto the Blender avatar as an Action')
        self.delete_button = QPushButton('Delete Take'); self.delete_button.clicked.connect(self.delete_take)
        self.folder_button = QPushButton('Open Folder'); self.folder_button.clicked.connect(
            lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(TAKES))))
        self.trim_label = QLabel(''); self.trim_label.setObjectName('muted')
        for w in (self.in_button, self.out_button, self.clear_button, self.trim_label):
            row.addWidget(w)
        row.addStretch(1)
        for w in (self.bake_button, self.delete_button, self.folder_button):
            row.addWidget(w)

        self.timer = QTimer(self); self.timer.setTimerType(Qt.PreciseTimer)
        self.timer.timeout.connect(self.tick); self.timer.start(16)
        self.refresh_takes()
        self.update_buttons()

    # ---- live capture ---------------------------------------------------------------
    def capture_python(self):
        from .platform_compat import venv_python
        return venv_python(ROOT / '.venv-live')

    def toggle_capture(self):
        if self.process is not None:
            return self.stop_capture()
        python = self.capture_python()
        if not python.is_file():
            QMessageBox.warning(self, 'Live capture', 'The live runtime is missing. Run tools/setup_macos.sh.')
            return
        self.stop_play()
        self.process = QProcess(self)
        self.process.setWorkingDirectory(str(ROOT))
        self.process.setProcessChannelMode(QProcess.MergedChannels)
        self.process.readyReadStandardOutput.connect(self.read_process)
        self.process.finished.connect(self.capture_finished)
        self.process.start(str(python), ['-u', str(ROOT / 'live/capture.py'), '--camera', str(self.camera.currentData()),
                                         '--app', '--port', str(APP_PORT), '--no-preview'])
        self.status.setText('Starting camera and tracking models… (macOS may ask for camera access)')
        self.update_buttons()

    def stop_capture(self):
        if self.record_button.isChecked():
            self.record_button.setChecked(False)
        if self.process is not None:
            self.process.terminate()
            if not self.process.waitForFinished(3000):
                self.process.kill()
        self.process = None
        self.status.setText('Camera stopped.')
        self.update_buttons()

    def read_process(self):
        text = bytes(self.process.readAllStandardOutput()).decode(errors='replace')
        for line in text.splitlines():
            if 'Cannot open the camera' in line or 'Traceback' in line or 'Error' in line:
                self.status.setText(line.strip()[:200])

    def capture_finished(self, code, _status):
        if self.process is not None and code != 0:
            self.status.setText(f'Capture stopped (exit {code}). If this was the first run, allow camera access '
                                'in System Settings > Privacy & Security > Camera and press Start Camera again.')
        self.process = None
        self.update_buttons()

    def set_paused(self, paused):
        self.paused = paused
        self.pause_button.setText('Resume' if paused else 'Pause')

    def read_datagrams(self):
        while self.socket.hasPendingDatagrams():
            datagram = self.socket.receiveDatagram()
            data = bytes(datagram.data())
            if not data:
                continue
            if data[:1] == b'P':
                self.packet = json.loads(data[1:])
                self.received += 1
                if not self.paused and not self.playing:
                    self.send(data[1:])
            elif data[:1] == b'F' and not self.playing:
                self.show_jpeg(data[1:])
                if self.recorder is not None and self.packet is not None and not self.paused:
                    self.recorder.add(self.packet, data[1:])

    def send(self, payload: bytes):
        if self.send_blender.isChecked():
            self.out.writeDatagram(payload, QHostAddress.LocalHost, BLENDER_PORT)

    def show_jpeg(self, jpeg):
        image = QImage.fromData(jpeg, 'JPG')
        if image.isNull():
            return
        pixmap = QPixmap.fromImage(image).scaled(self.preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.preview.setPixmap(pixmap)

    # ---- recording --------------------------------------------------------------------
    def toggle_record(self, on):
        if on:
            if self.process is None:
                self.record_button.setChecked(False)
                self.status.setText('Start the camera first.'); return
            self.recorder = Recorder()
            self.record_button.setText('■ Stop Recording')
        elif self.recorder is not None:
            folder = self.recorder.finish(); self.recorder = None
            self.record_button.setText('● Record')
            self.refresh_takes(select=folder.name)
        self.update_buttons()

    # ---- takes and playback -------------------------------------------------------------
    def refresh_takes(self, select=None):
        self.takes.blockSignals(True); self.takes.clear()
        folders = sorted(TAKES.glob('take_*'), reverse=True) if TAKES.exists() else []
        for f in folders:
            if (f / 'take.jsonl').exists():
                meta = json.loads((f / 'take.json').read_text()) if (f / 'take.json').exists() else {}
                self.takes.addItem(f"{f.name[5:]}  ({meta.get('seconds', '?')} s)", str(f))
        self.takes.blockSignals(False)
        if select:
            i = self.takes.findData(str(TAKES / select))
            if i >= 0:
                self.takes.setCurrentIndex(i)
        self.load_take()

    def load_take(self, *_):
        self.stop_play(reset=False)
        if self.take:
            self.take.close()
        self.take = None
        folder = self.takes.currentData()
        if folder:
            try:
                self.take = Take(Path(folder))
            except (OSError, ValueError) as error:
                self.status.setText(f'Cannot load take: {error}')
        n = len(self.take.packets) if self.take else 0
        self.slider.blockSignals(True); self.slider.setRange(0, max(0, n - 1)); self.slider.setValue(0); self.slider.blockSignals(False)
        self.play_pos = 0.0
        self.show_trim(); self.show_frame(0)
        self.update_buttons()

    def trim_range(self):
        n = len(self.take.packets)
        first = int(self.take.meta.get('in', 0)); last = int(self.take.meta.get('out', n - 1))
        return max(0, min(first, n - 1)), max(0, min(last, n - 1))

    def set_trim(self, which):
        if not self.take:
            return
        if which is None:
            self.take.meta.pop('in', None); self.take.meta.pop('out', None)
        else:
            self.take.meta[which] = self.slider.value()
            first, last = self.trim_range()
            if first > last:
                self.take.meta['in' if which == 'out' else 'out'] = self.slider.value()
        self.take.save_meta(); self.show_trim()

    def show_trim(self):
        if not self.take:
            self.trim_label.setText(''); return
        first, last = self.trim_range()
        self.trim_label.setText(f'Trim {self.take.times[first]:.2f}–{self.take.times[last]:.2f} s'
                                if ('in' in self.take.meta or 'out' in self.take.meta) else 'No trim')

    def toggle_play(self):
        if not self.take:
            return
        if self.playing:
            self.playing = False; self.play_button.setText('▶ Play'); return
        first, last = self.trim_range()
        if not first <= self.slider.value() < last:
            self.play_pos = self.take.times[first]
        else:
            self.play_pos = self.take.times[self.slider.value()]
        self.playing = True; self.play_clock = time.monotonic(); self.play_button.setText('❚❚ Pause')

    def stop_play(self, reset=True):
        self.playing = False; self.play_button.setText('▶ Play')
        if reset and self.take:
            first, _ = self.trim_range(); self.slider.setValue(first)

    def scrub(self, index):
        if self.take and not self.playing:
            self.play_pos = self.take.times[index]
            self.show_frame(index)

    def show_frame(self, index):
        if not self.take or not self.take.packets:
            self.time_label.setText('0.00 / 0.00 s'); return
        index = max(0, min(index, len(self.take.packets) - 1))
        jpeg = self.take.jpeg(index)
        if jpeg:
            self.show_jpeg(jpeg)
        packet = {k: v for k, v in self.take.packets[index].items() if k != 'jpg'}
        self.send(json.dumps(packet, separators=(',', ':')).encode())
        self.time_label.setText(f'{self.take.times[index]:.2f} / {self.take.duration:.2f} s')

    def tick(self):
        now = time.monotonic()
        if now - self._fps_mark[0] >= 1.0:
            self.fps = (self.received - self._fps_mark[1]) / (now - self._fps_mark[0])
            self._fps_mark = (now, self.received)
            if self.process is not None:
                state = 'PAUSED · ' if self.paused else ''
                rec = f' · ● REC {self.recorder.count} frames' if self.recorder else ''
                face = 'face ✓' if self.packet and 'face' in self.packet else 'face –'
                body = 'body ✓' if self.packet and 'pose' in self.packet else 'body –'
                hands = 'hands ' + ('/'.join(sorted((self.packet or {}).get('hands', {}))) or '–')
                self.status.setText(f'{state}{self.fps:4.1f} fps · {face} · {body} · {hands}{rec}')
        if self.playing and self.take:
            first, last = self.trim_range()
            self.play_pos += (now - self.play_clock) * float(self.speed.currentData())
            self.play_clock = now
            if self.play_pos > self.take.times[last]:
                if self.loop.isChecked():
                    self.play_pos = self.take.times[first]
                else:
                    self.playing = False; self.play_button.setText('▶ Play'); self.play_pos = self.take.times[last]
            index = min(last, bisect.bisect_right(self.take.times, self.play_pos) - 1)
            if index != self.slider.value():
                self.slider.blockSignals(True); self.slider.setValue(index); self.slider.blockSignals(False)
                self.show_frame(index)

    def bake(self):
        if not self.take:
            return
        first, last = self.trim_range()
        command = {'cmd': 'bake_take', 'path': str(self.take.folder / 'take.jsonl'), 'in': first, 'out': last}
        self.out.writeDatagram(json.dumps(command).encode(), QHostAddress.LocalHost, BLENDER_PORT)
        self.status.setText(f'Sent {last - first + 1} frames to Blender to bake (Action "RGBMoCap_Live_Take"). '
                            'Blender must be open with the live receiver.')

    def delete_take(self):
        if not self.take:
            return
        folder = self.take.folder
        if QMessageBox.question(self, 'Delete take', f'Delete {folder.name} permanently?') != QMessageBox.Yes:
            return
        self.take.close(); self.take = None
        shutil.rmtree(folder, ignore_errors=True)
        self.refresh_takes()

    # ---- Blender -------------------------------------------------------------------------
    def open_blender(self):
        blender = blender_path()
        if blender is None:
            QMessageBox.warning(self, 'Blender', 'Blender was not found. Install it in /Applications or set RGBMOCAP_BLENDER.')
            return
        default = ROOT / 'live/avatars/Seed-san.vrm'
        vrm, _ = QFileDialog.getOpenFileName(self, 'Avatar (VRM 1.0)', str(default), 'VRM avatars (*.vrm)')
        if not vrm:
            return
        vrm = Path(vrm); blend = ROOT / 'build/avatars' / (vrm.stem + '.blend')
        blend.parent.mkdir(parents=True, exist_ok=True)
        if not blend.exists() or vrm.stat().st_mtime > blend.stat().st_mtime:
            self.status.setText('Preparing avatar for Blender…')
            done = subprocess.run([str(blender), '-b', '--factory-startup', '--python-exit-code', '1', '--python',
                                   str(ROOT / 'live/blender/prepare_vrm.py'), '--', str(vrm), str(blend)],
                                  capture_output=True, text=True)
            if done.returncode != 0 or not blend.exists():
                self.status.setText('Avatar preparation failed: ' + (done.stdout + done.stderr)[-300:]); return
        subprocess.Popen([str(blender), str(blend), '--python', str(ROOT / 'live/blender/start_live.py')],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        self.status.setText(f'Opening {vrm.stem} in Blender with the live receiver…')

    def update_buttons(self):
        live = self.process is not None
        has_take = self.take is not None and bool(self.take.packets)
        self.start_button.setText('Stop Camera' if live else 'Start Camera')
        self.camera.setEnabled(not live)
        self.pause_button.setEnabled(live); self.record_button.setEnabled(live)
        for w in (self.play_button, self.stop_button, self.slider, self.in_button, self.out_button,
                  self.clear_button, self.bake_button, self.delete_button):
            w.setEnabled(has_take and not self.record_button.isChecked())

    def shutdown(self):
        self.stop_capture()
        if self.take:
            self.take.close()


class LiveWorkspace:
    """MainWindow mixin: Live menu toggles a dockable Live Capture panel."""

    def toggle_live(self):
        dock = getattr(self, 'live_dock', None)
        if dock is None:
            self.live_panel = LivePanel(self)
            dock = self.live_dock = QDockWidget('Live Capture', self)
            dock.setObjectName('liveCaptureDock')
            dock.setWidget(self.live_panel)
            dock.setAllowedAreas(Qt.BottomDockWidgetArea | Qt.RightDockWidgetArea | Qt.LeftDockWidgetArea)
            self.addDockWidget(Qt.BottomDockWidgetArea, dock)
            from PySide6.QtWidgets import QApplication
            QApplication.instance().aboutToQuit.connect(self.live_panel.shutdown)
            dock.show(); dock.raise_()
            return
        dock.setVisible(not dock.isVisible())
        if dock.isVisible():
            dock.raise_()
