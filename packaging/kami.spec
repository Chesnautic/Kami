# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for Kami (Y2K Chaotic Music Visualizer).

Builds a one-folder ("onedir") Windows app at dist/Kami/Kami.exe. Onedir
(rather than onefile) is deliberate: Kami.exe re-launches itself as a
background worker process for every render (see Kami.pyw /
gui._render_subprocess_cmd), and onedir means that re-launch is instant
instead of re-extracting a whole onefile bundle to a temp dir each time.

The installer (packaging/installer.iss) just packages this entire
dist/Kami/ folder into Program Files, plus a bundled ffmpeg.exe copied in
alongside it by the CI workflow before the installer is built.

Usage (normally run by .github/workflows/build-windows.yml on Windows, not
locally on Linux/Mac -- PyInstaller builds for whatever OS it runs on):
    pyinstaller packaging/kami.spec --noconfirm --distpath dist --workpath build
"""
import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

PROJECT_ROOT = os.path.abspath(os.path.join(SPECPATH, os.pardir))

hiddenimports = (
    collect_submodules("scipy.io")
    + collect_submodules("scipy.fft")
    + collect_submodules("scipy.special")
)

datas = [
    (os.path.join(PROJECT_ROOT, "kami.ico"), "."),
    (os.path.join(PROJECT_ROOT, "kami_icon_1024.png"), "."),
]
datas += collect_data_files("scipy", include_py_files=False)

a = Analysis(
    [os.path.join(PROJECT_ROOT, "Kami.pyw")],
    pathex=[PROJECT_ROOT],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["matplotlib", "pytest", "IPython", "notebook"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Kami",
    icon=os.path.join(PROJECT_ROOT, "kami.ico"),
    debug=False,
    strip=False,
    upx=False,
    console=False,          # no console window for the GUI launch
    disable_windowed_traceback=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="Kami",
)
