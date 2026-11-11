"""Drive the real GUI: open a video, run a GUI action, wait for its workers, screenshot.

usage: gui_stage_check.py VIDEO ACTION OUT.png [timeout_s]
ACTION is a MainWindow method name (find_people, run_full) or stages:a,b,c.
"""
import sys, time, json
from pathlib import Path
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer

video, action, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
limit = float(sys.argv[4]) if len(sys.argv) > 4 else 900
app = QApplication(sys.argv[:1])
# Modal dialogs would block this driver: log them and dismiss instead.
from PySide6.QtWidgets import QDialog, QMessageBox
def _report(self, *a, **k):
    text = getattr(self, 'text', lambda: '')() if isinstance(self, QMessageBox) else ''
    print('DIALOG', type(self).__name__, repr(self.windowTitle()), text[:300], flush=True); return 0
QDialog.exec = _report
QMessageBox.exec = _report
from mocap.theme import configure_application
from mocap.gui import MainWindow
configure_application(app)
w = MainWindow(video); w.show()
state = dict(started=None, t0=time.monotonic(), last='')

def tick():
    if state['started'] is None:
        if time.monotonic() - state['t0'] < 3: return
        if action.startswith('stages:'):
            w.input_jobs(action.split(':', 1)[1].split(','))  # the GUI's own job path, chosen stages
        else:
            getattr(w, action)()
        state['started'] = time.monotonic(); return
    text = w.status.text()
    if text != state['last']:
        print(f"[{time.monotonic()-state['started']:6.1f}s] {text}", flush=True); state['last'] = text
    done = not w.busy and time.monotonic() - state['started'] > 2
    if done or time.monotonic() - state['started'] > limit:
        if not done: w.cancel_job()
        w.grab().save(str(out))
        log = w.log.toPlainText() if hasattr(w.log, 'toPlainText') else ''
        Path(out.with_suffix('.log')).write_text(log)
        print(json.dumps(dict(finished=done, failed=bool(getattr(w, 'failed', False)), status=text,
                              seconds=round(time.monotonic()-state['started'], 1))), flush=True)
        app.exit(0 if done and not getattr(w, 'failed', False) else 1)

timer = QTimer(); timer.timeout.connect(tick); timer.start(250)
sys.exit(app.exec())
