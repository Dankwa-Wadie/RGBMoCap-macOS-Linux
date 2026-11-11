"""Drive the real app: Live panel -> start camera -> record -> stop -> play -> trim -> screenshot."""
import sys, time, json
from pathlib import Path
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer
out = Path(sys.argv[1])
app = QApplication(sys.argv[:1])
from mocap.theme import configure_application
from mocap.gui import MainWindow
configure_application(app)
w = MainWindow(None); w.resize(1600, 1100); w.show()
steps = []; t0 = time.monotonic()
def at(sec, fn): steps.append((sec, fn))
def log(*a): print(f'[{time.monotonic()-t0:5.1f}s]', *a, flush=True)
at(1, lambda: (w.toggle_live(), log('panel open; cameras:', [w.live_panel.camera.itemText(i) for i in range(w.live_panel.camera.count())])))
at(2, lambda: (w.live_panel.camera.setCurrentIndex(0), w.live_panel.start_button.click(), log('start camera')))
at(10, lambda: (log('status:', w.live_panel.status.text()), w.live_panel.record_button.setChecked(True), log('recording')))
at(14.5, lambda: (w.live_panel.record_button.setChecked(False), log('stopped; takes:', w.live_panel.takes.count(), w.live_panel.takes.currentText())))
at(15, lambda: (w.live_panel.start_button.click(), log('camera stopped')))
at(16, lambda: (w.live_panel.toggle_play(), log('play; frames in take:', len(w.live_panel.take.packets))))
at(18, lambda: log('playing at', w.live_panel.time_label.text()))
at(18.2, lambda: (w.live_panel.toggle_play(), w.live_panel.slider.setValue(len(w.live_panel.take.packets)//4), w.live_panel.set_trim('in'),
                  w.live_panel.slider.setValue(3*len(w.live_panel.take.packets)//4), w.live_panel.set_trim('out'), log('trim:', w.live_panel.trim_label.text())))
at(19, lambda: (w.grab().save(str(out)), log('screenshot saved')))
at(19.5, app.quit)
def run():
    now = time.monotonic() - t0
    while steps and steps[0][0] <= now:
        steps.pop(0)[1]()
timer = QTimer(); timer.timeout.connect(run); timer.start(50)
sys.exit(app.exec())
