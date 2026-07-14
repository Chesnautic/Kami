"""Kami.pyw — double-clickable Windows launcher (runs via pythonw.exe in
source form, or as Kami.exe when packaged with PyInstaller -- no console
window either way). This is also the single entry point PyInstaller builds
against, so it does double duty:

  - normal launch                  -> opens the GUI
  - launched with --render-worker  -> runs the render pipeline instead

That second mode exists because the GUI kicks off rendering in a separate
process (so the UI stays responsive and a render can be canceled cleanly).
In a normal Python install that second process is just
`python render.py --config ...`. But a packaged .exe has no separate
Python interpreter to shell out to -- so gui.py instead re-launches this
same exe with --render-worker, and this file dispatches straight to
render.main() instead of opening a second GUI window. See RENDER_WORKER_FLAG
in gui.py for the launcher side of this.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

RENDER_WORKER_FLAG = "--render-worker"

if __name__ == "__main__":
    if RENDER_WORKER_FLAG in sys.argv:
        sys.argv.remove(RENDER_WORKER_FLAG)
        from render import main as render_main
        raise SystemExit(render_main())
    else:
        from gui import main as gui_main
        gui_main()
