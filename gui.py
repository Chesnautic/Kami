#!/usr/bin/env python3
"""
gui.py — Kami: the Y2K Chaotic Music Visualizer desktop control panel.

A native Tkinter window (no extra GUI dependencies) for picking a WAV
file, customizing colors, tuning how strongly the visuals react to
bass/mid/treble/beats, choosing which patterns are in play, and watching a
live animated preview before committing to a full render.

Run:
    python3 gui.py
    (or double-click Kami.pyw / the Kami desktop shortcut on Windows)

(Needs a Tk-enabled Python — this ships with the standard python.org
installers on Windows/Mac. On Debian/Ubuntu Linux you may need:
    sudo apt install python3-tk
)
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import tkinter as tk
from tkinter import ttk, filedialog, colorchooser, messagebox

import numpy as np
from PIL import ImageTk

from audio_analysis import analyze, Features, wav_duration, load_waveform_preview, trim_wav
from palettes import PALETTES, DEFAULT_PALETTE, build_custom_palette, palette_to_hex_fields
from patterns import PATTERN_REGISTRY, PATTERN_NAMES, SCENE_PACKS
from controls import Controls

try:
    import winsound
    _HAS_WINSOUND = True
except ImportError:
    winsound = None
    _HAS_WINSOUND = False  # not on Windows -- snippet preview playback is disabled

PACK_TITLES = {
    "waveforms": "Pack 1 — Waveforms",
    "cars": "Pack 2 — Pixel Cars",
    "space_sunsets": "Pack 3 — Sunsets & Space",
    "retro_y2k": "Pack 4 — Retro Y2K",
}

SNIPPET_PREVIEW_SECONDS = 4.0  # length of a "click to preview here" snippet


def _format_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    m, s = divmod(int(round(seconds)), 60)
    return f"{m}:{s:02d}"

PREVIEW_W, PREVIEW_H = 480, 270
PREVIEW_FPS = 18
N_GRADIENT_SWATCHES = 5
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RENDER_SCRIPT = os.path.join(SCRIPT_DIR, "render.py")
RENDER_WORKER_FLAG = "--render-worker"  # keep in sync with Kami.pyw


def _render_subprocess_cmd(cfg_path: str) -> list[str]:
    """Build the command that runs a render as a separate process.

    In a normal Python install, sys.executable is python(w).exe and we
    shell out to render.py directly. In a packaged .exe (PyInstaller),
    there is no separate Python interpreter to point at -- sys.executable
    IS the app itself -- so instead we re-launch this same exe with
    --render-worker, which Kami.pyw's entry point dispatches straight to
    render.main() instead of opening a second GUI window."""
    if getattr(sys, "frozen", False):
        return [sys.executable, RENDER_WORKER_FLAG, "--config", cfg_path]
    return [sys.executable, "-u", RENDER_SCRIPT, "--config", cfg_path]

BG = "#0b0713"
PANEL_BG = "#171025"
FG = "#f0eaff"
ACCENT = "#ff2fb0"


def _style_scale(scale: ttk.Scale):
    scale.configure(length=200)


class VisualizerGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("Kami — Y2K Chaotic Music Visualizer")
        root.geometry("1180x780")
        root.minsize(1000, 680)
        root.configure(bg=BG)
        self._set_window_icon(root)

        self._init_style()

        # ---- state ---------------------------------------------------
        self.wav_path: str | None = None
        self.features: Features | None = None
        self._analysis_token = 0

        # ---- snippet picker (trim which part of the WAV to render) ----
        self.wav_duration = 0.0
        self.snippet_start = 0.0
        self.snippet_end = 0.0  # 0.0 == "not set yet" until duration is known
        self.waveform_env = None  # coarse amplitude envelope, np.ndarray or None
        self._snippet_drag = None  # "start" | "end" | None, which handle is being dragged
        self._snippet_temp_wav: str | None = None  # scratch file for scrub-preview playback
        self._is_playing = False

        self.palette_preset = tk.StringVar(value=DEFAULT_PALETTE)
        preset_hex = palette_to_hex_fields(PALETTES[DEFAULT_PALETTE])
        self.bg_color = tk.StringVar(value=preset_hex["bg"])
        self.accent_color = tk.StringVar(value=preset_hex["accent"])
        self.glow_color = tk.StringVar(value=preset_hex["glow"])
        self.color_vars = [tk.StringVar(value=c) for c in preset_hex["colors"][:N_GRADIENT_SWATCHES]]
        while len(self.color_vars) < N_GRADIENT_SWATCHES:
            self.color_vars.append(tk.StringVar(value="#ffffff"))
        self.color_buttons: list[tk.Button] = []

        self.chaos = tk.DoubleVar(value=0.65)
        self.bass_gain = tk.DoubleVar(value=1.0)
        self.mid_gain = tk.DoubleVar(value=1.0)
        self.treble_gain = tk.DoubleVar(value=1.0)
        self.onset_gain = tk.DoubleVar(value=1.0)
        self.glow_strength = tk.DoubleVar(value=1.0)
        self.particle_density = tk.DoubleVar(value=1.0)
        self.switch_speed = tk.DoubleVar(value=1.0)

        self.pattern_enabled = {name: tk.BooleanVar(value=True) for name in PATTERN_NAMES}
        self.preview_pattern = tk.StringVar(value=PATTERN_NAMES[0])

        self.resolution = tk.StringVar(value="1280x720")
        self.fps = tk.IntVar(value=30)
        self.seed_var = tk.StringVar(value="")
        self.out_path_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="Pick a WAV file to get started.")

        self._render_proc: subprocess.Popen | None = None

        # ---- live preview machinery -----------------------------------
        self.preview_states: dict[str, dict] = {name: {} for name in PATTERN_NAMES}
        self.preview_local_t: dict[str, float] = {name: 0.0 for name in PATTERN_NAMES}
        self.preview_rng = np.random.default_rng(1234)
        self.preview_time = 0.0
        self._last_fake_beat_idx = -1
        self.preview_photo = None

        self._preview_after_id = None
        self._build_layout()
        self._wire_color_reset_traces()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._tick_preview()

    def _on_close(self):
        if self._preview_after_id is not None:
            try:
                self.root.after_cancel(self._preview_after_id)
            except Exception:
                pass
        if self._render_proc and self._render_proc.poll() is None:
            try:
                self._render_proc.terminate()
            except Exception:
                pass
        self._stop_playback()
        if self._snippet_temp_wav:
            try:
                os.remove(self._snippet_temp_wav)
            except OSError:
                pass
        self.root.destroy()

    # ------------------------------------------------------------------
    # window chrome
    # ------------------------------------------------------------------
    def _set_window_icon(self, root: tk.Tk):
        ico_path = os.path.join(SCRIPT_DIR, "kami.ico")
        png_path = os.path.join(SCRIPT_DIR, "kami_icon_1024.png")
        try:
            if os.path.exists(ico_path):
                root.iconbitmap(ico_path)  # title bar + taskbar icon on Windows
                return
        except tk.TclError:
            pass
        try:
            if os.path.exists(png_path):
                from PIL import Image
                img = Image.open(png_path)
                img.thumbnail((256, 256))
                self._icon_photo = ImageTk.PhotoImage(img)  # keep a reference
                root.iconphoto(True, self._icon_photo)
        except Exception:
            pass  # icon is cosmetic — never block the app from launching

    # ------------------------------------------------------------------
    # styling / layout
    # ------------------------------------------------------------------
    def _init_style(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TNotebook", background=BG, borderwidth=0)
        style.configure("TNotebook.Tab", background=PANEL_BG, foreground=FG, padding=(14, 8))
        style.map("TNotebook.Tab", background=[("selected", ACCENT)], foreground=[("selected", "#150015")])
        style.configure("TFrame", background=PANEL_BG)
        style.configure("TLabel", background=PANEL_BG, foreground=FG)
        style.configure("TCheckbutton", background=PANEL_BG, foreground=FG)
        style.configure("TButton", background=ACCENT, foreground="#150015", padding=6)
        style.configure("Horizontal.TScale", background=PANEL_BG)
        style.configure("TCombobox", fieldbackground=PANEL_BG)
        style.configure("TProgressbar", background=ACCENT)

    def _build_layout(self):
        # top: file picker row
        top = tk.Frame(self.root, bg=BG)
        top.pack(fill="x", padx=14, pady=(12, 6))

        tk.Button(top, text="Browse WAV...", command=self._pick_wav,
                  bg=ACCENT, fg="#150015", relief="flat", padx=10, pady=4).pack(side="left")
        self.wav_label = tk.Label(top, text="(no file selected)", bg=BG, fg=FG, anchor="w")
        self.wav_label.pack(side="left", padx=10)

        self._build_snippet_picker(self.root)

        # main split: notebook (left) + preview (right)
        main = tk.Frame(self.root, bg=BG)
        main.pack(fill="both", expand=True, padx=14, pady=6)

        left = tk.Frame(main, bg=BG, width=560)
        left.pack(side="left", fill="both", expand=False)
        right = tk.Frame(main, bg=BG)
        right.pack(side="left", fill="both", expand=True, padx=(14, 0))

        notebook = ttk.Notebook(left)
        notebook.pack(fill="both", expand=True)

        colors_tab = ttk.Frame(notebook)
        reactivity_tab = ttk.Frame(notebook)
        patterns_tab = ttk.Frame(notebook)
        output_tab = ttk.Frame(notebook)
        notebook.add(colors_tab, text="Colors")
        notebook.add(reactivity_tab, text="Reactivity")
        notebook.add(patterns_tab, text="Patterns")
        notebook.add(output_tab, text="Output")

        self._build_colors_tab(colors_tab)
        self._build_reactivity_tab(reactivity_tab)
        self._build_patterns_tab(patterns_tab)
        self._build_output_tab(output_tab)

        self._build_preview_panel(right)

        # bottom: render controls
        bottom = tk.Frame(self.root, bg=BG)
        bottom.pack(fill="x", padx=14, pady=(6, 12))

        self.render_button = tk.Button(bottom, text="Render Full Video", command=self._start_render,
                                        bg=ACCENT, fg="#150015", relief="flat", padx=14, pady=8,
                                        font=("TkDefaultFont", 11, "bold"))
        self.render_button.pack(side="left")
        self.cancel_button = tk.Button(bottom, text="Cancel", command=self._cancel_render,
                                        bg="#3a2440", fg=FG, relief="flat", padx=10, pady=8, state="disabled")
        self.cancel_button.pack(side="left", padx=8)

        self.progress = ttk.Progressbar(bottom, orient="horizontal", mode="determinate", maximum=100)
        self.progress.pack(side="left", fill="x", expand=True, padx=12)

        status = tk.Label(self.root, textvariable=self.status_var, bg=BG, fg="#b9a8d9", anchor="w")
        status.pack(fill="x", padx=16, pady=(0, 10))

    def _build_colors_tab(self, parent):
        pad = dict(padx=10, pady=6)

        row = ttk.Frame(parent)
        row.pack(fill="x", **pad)
        ttk.Label(row, text="Preset palette:").pack(side="left")
        combo = ttk.Combobox(row, textvariable=self.palette_preset, values=list(PALETTES.keys()),
                              state="readonly", width=16)
        combo.pack(side="left", padx=8)
        combo.bind("<<ComboboxSelected>>", self._apply_preset_to_swatches)

        ttk.Label(parent, text="Individual colors override the preset — click a swatch to pick a color.",
                  wraplength=480).pack(anchor="w", **pad)

        swatch_row = ttk.Frame(parent)
        swatch_row.pack(fill="x", **pad)
        self._add_swatch(swatch_row, "Background", self.bg_color)
        self._add_swatch(swatch_row, "Accent", self.accent_color)
        self._add_swatch(swatch_row, "Glow", self.glow_color)

        ttk.Label(parent, text="Gradient colors (used by tunnels / plasma / mandalas / bars):").pack(
            anchor="w", **pad)
        grad_row = ttk.Frame(parent)
        grad_row.pack(fill="x", **pad)
        for i, var in enumerate(self.color_vars):
            self._add_swatch(grad_row, f"#{i+1}", var)

    def _add_swatch(self, parent, label, var: tk.StringVar):
        col = ttk.Frame(parent)
        col.pack(side="left", padx=6)
        ttk.Label(col, text=label).pack()
        btn = tk.Button(col, bg=var.get(), width=6, height=2, relief="flat")
        btn.configure(command=lambda v=var, b=btn: self._pick_color(v, b))
        btn.pack()
        self.color_buttons.append(btn)
        return btn

    def _pick_color(self, var: tk.StringVar, button: tk.Button):
        rgb, hexstr = colorchooser.askcolor(color=var.get(), title="Pick a color")
        if hexstr:
            var.set(hexstr)
            button.configure(bg=hexstr)
            self._reset_preview_state()

    def _apply_preset_to_swatches(self, *_):
        fields = palette_to_hex_fields(PALETTES[self.palette_preset.get()])
        self.bg_color.set(fields["bg"])
        self.accent_color.set(fields["accent"])
        self.glow_color.set(fields["glow"])
        for var, hexval in zip(self.color_vars, fields["colors"]):
            var.set(hexval)
        for btn, var in zip(self.color_buttons, [self.bg_color, self.accent_color, self.glow_color] + self.color_vars):
            btn.configure(bg=var.get())
        self._reset_preview_state()

    def _wire_color_reset_traces(self):
        # bg/accent/glow/gradient colors get cached (LUTs, fixed particle
        # colors) inside pattern state dicts — clear that state whenever a
        # color actually changes so the live preview picks up the new hues.
        for var in [self.bg_color, self.accent_color, self.glow_color, *self.color_vars]:
            var.trace_add("write", lambda *_: self._reset_preview_state())

    def _build_reactivity_tab(self, parent):
        sliders = [
            ("Chaos", self.chaos, 0.0, 1.0,
             "Overall randomness: shorter pattern segments, more wobble/glitch."),
            ("Bass sensitivity", self.bass_gain, 0.0, 3.0, "How strongly low end drives pulses/height."),
            ("Mid sensitivity", self.mid_gain, 0.0, 3.0, "How strongly mids drive rotation/speed."),
            ("Treble sensitivity", self.treble_gain, 0.0, 3.0, "How strongly highs drive sparkle/detail."),
            ("Beat sensitivity", self.onset_gain, 0.0, 3.0, "How strongly transients/beats spike the visuals."),
            ("Glow strength", self.glow_strength, 0.0, 2.5, "Neon blur/bloom intensity."),
            ("Particle density", self.particle_density, 0.0, 3.0, "Particle / spark / glitch-band counts."),
            ("Switch speed", self.switch_speed, 0.25, 3.0, "How fast the director cuts between patterns."),
        ]
        for label, var, lo, hi, help_text in sliders:
            self._add_slider(parent, label, var, lo, hi, help_text)

    def _add_slider(self, parent, label, var: tk.DoubleVar, lo, hi, help_text):
        frame = ttk.Frame(parent)
        frame.pack(fill="x", padx=12, pady=6)
        header = ttk.Frame(frame)
        header.pack(fill="x")
        ttk.Label(header, text=label, font=("TkDefaultFont", 10, "bold")).pack(side="left")
        value_lbl = ttk.Label(header, text=f"{var.get():.2f}")
        value_lbl.pack(side="right")

        def on_change(v, var=var, value_lbl=value_lbl):
            value_lbl.configure(text=f"{float(v):.2f}")

        scale = ttk.Scale(frame, from_=lo, to=hi, orient="horizontal", variable=var, command=on_change)
        scale.pack(fill="x")
        ttk.Label(frame, text=help_text, foreground="#8f7fae", wraplength=480).pack(anchor="w")

    def _build_patterns_tab(self, parent):
        header = ttk.Frame(parent)
        header.pack(fill="x", padx=12, pady=(10, 4))
        ttk.Label(header, text="Patterns in the auto-cycle pool for the final render (4 packs, 30 scenes):",
                  font=("TkDefaultFont", 10, "bold")).pack(side="left")

        all_row = ttk.Frame(parent)
        all_row.pack(fill="x", padx=12, pady=(0, 6))
        ttk.Button(all_row, text="Select all", command=lambda: self._set_all_patterns(True)).pack(side="left")
        ttk.Button(all_row, text="Select none", command=lambda: self._set_all_patterns(False)).pack(
            side="left", padx=6)

        # scrollable area — 30 checkboxes across 4 packs won't fit in a fixed
        # frame, so this pack list scrolls independently of the rest of the tab.
        outer = ttk.Frame(parent)
        outer.pack(fill="both", expand=True, padx=6, pady=4)
        canvas = tk.Canvas(outer, bg=PANEL_BG, highlightthickness=0)
        vscroll = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        scroll_frame = ttk.Frame(canvas)
        scroll_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=vscroll.set)
        canvas.pack(side="left", fill="both", expand=True)
        vscroll.pack(side="right", fill="y")

        def _on_mousewheel(event):
            delta = event.delta
            if delta:
                canvas.yview_scroll(int(-delta / 120) or (-1 if delta > 0 else 1), "units")
            elif getattr(event, "num", None) in (4, 5):
                canvas.yview_scroll(-1 if event.num == 4 else 1, "units")

        def _bind_wheel(_e=None):
            for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
                canvas.bind_all(seq, _on_mousewheel)

        def _unbind_wheel(_e=None):
            for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
                canvas.unbind_all(seq)

        # only hijack the scroll wheel while the cursor is actually over this
        # canvas, so it doesn't fight with scrolling elsewhere in the app.
        canvas.bind("<Enter>", _bind_wheel)
        canvas.bind("<Leave>", _unbind_wheel)

        n_cols = 2
        for pack_key, names in SCENE_PACKS.items():
            pack_frame = ttk.Frame(scroll_frame)
            pack_frame.pack(fill="x", padx=6, pady=(8, 2), anchor="w")

            pack_header = ttk.Frame(pack_frame)
            pack_header.pack(fill="x")
            ttk.Label(pack_header, text=PACK_TITLES.get(pack_key, pack_key),
                      font=("TkDefaultFont", 10, "bold")).pack(side="left")
            ttk.Button(pack_header, text="all", width=4,
                       command=lambda ns=names: self._set_pattern_group(ns, True)).pack(side="left", padx=(10, 2))
            ttk.Button(pack_header, text="none", width=4,
                       command=lambda ns=names: self._set_pattern_group(ns, False)).pack(side="left")

            grid = ttk.Frame(pack_frame)
            grid.pack(fill="x", pady=(2, 0))
            for i, name in enumerate(names):
                cb = ttk.Checkbutton(grid, text=name.replace("_", " "), variable=self.pattern_enabled[name])
                cb.grid(row=i // n_cols, column=i % n_cols, sticky="w", padx=6, pady=2)

            ttk.Separator(scroll_frame, orient="horizontal").pack(fill="x", padx=6, pady=(6, 0))

    def _set_all_patterns(self, enabled: bool):
        for var in self.pattern_enabled.values():
            var.set(enabled)

    def _set_pattern_group(self, names, enabled: bool):
        for name in names:
            self.pattern_enabled[name].set(enabled)

    def _build_output_tab(self, parent):
        out_grid = ttk.Frame(parent)
        out_grid.pack(fill="x", padx=12, pady=12)

        ttk.Label(out_grid, text="Resolution:").grid(row=0, column=0, sticky="w", pady=4)
        ttk.Combobox(out_grid, textvariable=self.resolution, state="readonly", width=14,
                     values=["1920x1080", "1280x720", "854x480", "640x360"]).grid(row=0, column=1, sticky="w")

        ttk.Label(out_grid, text="FPS:").grid(row=1, column=0, sticky="w", pady=4)
        ttk.Spinbox(out_grid, from_=12, to=60, textvariable=self.fps, width=6).grid(row=1, column=1, sticky="w")

        ttk.Label(out_grid, text="Seed (blank = random):").grid(row=2, column=0, sticky="w", pady=4)
        ttk.Entry(out_grid, textvariable=self.seed_var, width=14).grid(row=2, column=1, sticky="w")

        ttk.Label(out_grid, text="Output file:").grid(row=3, column=0, sticky="w", pady=4)
        out_row = ttk.Frame(out_grid)
        out_row.grid(row=3, column=1, sticky="w")
        ttk.Entry(out_row, textvariable=self.out_path_var, width=28).pack(side="left")
        ttk.Button(out_row, text="...", width=3, command=self._pick_output).pack(side="left", padx=4)

    def _build_snippet_picker(self, parent):
        frame = tk.Frame(parent, bg=BG)
        frame.pack(fill="x", padx=14, pady=(0, 8))

        header = tk.Frame(frame, bg=BG)
        header.pack(fill="x")
        tk.Label(header, text="Snippet to render:", bg=BG, fg=FG,
                 font=("TkDefaultFont", 9, "bold")).pack(side="left")
        self.snippet_label = tk.Label(header, text="(load a WAV first)", bg=BG, fg="#8f7fae")
        self.snippet_label.pack(side="left", padx=10)

        btn_row = tk.Frame(header, bg=BG)
        btn_row.pack(side="right")
        self.play_selection_btn = tk.Button(btn_row, text="▶ Play selection", command=self._play_selection,
                                             bg="#3a2440", fg=FG, relief="flat", padx=8, pady=2, state="disabled")
        self.play_selection_btn.pack(side="left", padx=3)
        self.stop_playback_btn = tk.Button(btn_row, text="⏹ Stop", command=self._stop_playback,
                                            bg="#3a2440", fg=FG, relief="flat", padx=8, pady=2, state="disabled")
        self.stop_playback_btn.pack(side="left", padx=3)
        self.reset_snippet_btn = tk.Button(btn_row, text="Reset to full track", command=self._reset_snippet,
                                            bg="#3a2440", fg=FG, relief="flat", padx=8, pady=2, state="disabled")
        self.reset_snippet_btn.pack(side="left", padx=3)

        self.snippet_canvas = tk.Canvas(frame, height=54, bg="#120c1e", highlightthickness=1,
                                         highlightbackground="#3a2440")
        self.snippet_canvas.pack(fill="x", pady=(4, 0))
        self.snippet_canvas.bind("<Configure>", lambda e: self._draw_waveform())
        self.snippet_canvas.bind("<Button-1>", self._on_snippet_press)
        self.snippet_canvas.bind("<B1-Motion>", self._on_snippet_drag)
        self.snippet_canvas.bind("<ButtonRelease-1>", self._on_snippet_release)

        note_text = ("Drag the green/red handles to pick which part of the song gets rendered "
                     "(default is the whole track). Click anywhere else on the waveform to "
                     "preview a few seconds from that point.")
        if not _HAS_WINSOUND:
            note_text += " Playback preview needs Windows -- trimming itself still works without it."
        tk.Label(frame, text=note_text, bg=BG, fg="#8f7fae", wraplength=1100, justify="left",
                 font=("TkDefaultFont", 8)).pack(anchor="w", pady=(2, 0))

    def _build_preview_panel(self, parent):
        header = ttk.Frame(parent)
        header.pack(fill="x")
        ttk.Label(header, text="Live preview:", font=("TkDefaultFont", 10, "bold")).pack(side="left")
        combo = ttk.Combobox(header, textvariable=self.preview_pattern, values=PATTERN_NAMES,
                              state="readonly", width=18)
        combo.pack(side="left", padx=8)

        self.preview_canvas = tk.Canvas(parent, width=PREVIEW_W, height=PREVIEW_H,
                                         bg="black", highlightthickness=2, highlightbackground=ACCENT)
        self.preview_canvas.pack(pady=10)
        self.preview_image_id = self.preview_canvas.create_image(0, 0, anchor="nw")

        note = ("Preview uses your real audio once a WAV is analyzed (loops through it); "
                "before that it reacts to a simulated 120bpm pulse so you can dial in colors "
                "and intensity right away.")
        ttk.Label(parent, text=note, wraplength=440, foreground="#8f7fae").pack(anchor="w", pady=(0, 6))

    # ------------------------------------------------------------------
    # WAV loading / analysis
    # ------------------------------------------------------------------
    def _pick_wav(self):
        path = filedialog.askopenfilename(title="Choose a WAV file",
                                           filetypes=[("WAV audio", "*.wav"), ("All files", "*.*")])
        if not path:
            return
        self.wav_path = path
        self.wav_label.configure(text=os.path.basename(path))
        if not self.out_path_var.get().strip():
            self.out_path_var.set(path.rsplit(".", 1)[0] + "_y2k.mp4")
        self.status_var.set("Analyzing audio...")
        self._stop_playback()
        self.wav_duration = 0.0
        self.waveform_env = None
        self.snippet_start = 0.0
        self.snippet_end = 0.0
        self.play_selection_btn.configure(state="disabled")
        self.reset_snippet_btn.configure(state="disabled")
        self._draw_waveform()
        self._analyze_wav_async()
        self._load_waveform_async()

    def _load_waveform_async(self):
        # decoupled from the heavier _analyze_wav_async STFT pass, so the
        # snippet picker (and "listen while you choose" preview) becomes
        # usable right away instead of waiting on full beat/drop analysis
        path = self.wav_path
        token = self._analysis_token

        def worker():
            try:
                dur = wav_duration(path)
                env = load_waveform_preview(path, n_points=1200)
            except Exception as e:
                self.root.after(0, lambda: self.status_var.set(f"Couldn't load waveform preview: {e}"))
                return
            if token != self._analysis_token:
                return  # a newer file was picked while this was loading

            def apply():
                self.wav_duration = dur
                self.waveform_env = env
                self.snippet_start = 0.0
                self.snippet_end = dur
                self.play_selection_btn.configure(state="normal" if _HAS_WINSOUND else "disabled")
                self.reset_snippet_btn.configure(state="normal")
                self._draw_waveform()
            self.root.after(0, apply)

        threading.Thread(target=worker, daemon=True).start()

    # ------------------------------------------------------------------
    # snippet picker: waveform drawing, drag handles, scrub playback
    # ------------------------------------------------------------------
    def _snippet_time_to_x(self, t: float) -> int:
        cw = max(1, self.snippet_canvas.winfo_width())
        if self.wav_duration <= 0:
            return 0
        return int(np.clip(t / self.wav_duration, 0.0, 1.0) * cw)

    def _snippet_x_to_time(self, x: int) -> float:
        cw = max(1, self.snippet_canvas.winfo_width())
        if self.wav_duration <= 0:
            return 0.0
        return float(np.clip(x / cw, 0.0, 1.0)) * self.wav_duration

    def _draw_waveform(self):
        c = self.snippet_canvas
        c.delete("all")
        cw, ch = c.winfo_width(), c.winfo_height()
        if cw < 10 or ch < 10:
            return
        mid = ch / 2
        if self.waveform_env is not None and self.wav_duration > 0:
            env = self.waveform_env
            n = len(env)
            for x in range(cw):
                idx = min(n - 1, int(x / cw * n))
                hgt = float(env[idx]) * (ch * 0.45)
                c.create_line(x, mid - hgt, x, mid + hgt, fill="#8f6fae")
            sx, ex = self._snippet_time_to_x(self.snippet_start), self._snippet_time_to_x(self.snippet_end)
            if sx > 0:
                c.create_rectangle(0, 0, sx, ch, fill="#000000", stipple="gray50", outline="")
            if ex < cw:
                c.create_rectangle(ex, 0, cw, ch, fill="#000000", stipple="gray50", outline="")
            c.create_line(sx, 0, sx, ch, fill="#39ff88", width=2)
            c.create_line(ex, 0, ex, ch, fill="#ff4466", width=2)
        else:
            c.create_line(0, mid, cw, mid, fill="#3a2440")
            c.create_text(cw / 2, ch / 2, text="(load a WAV to see its waveform)", fill="#5a4a6a")
        self._update_snippet_label()

    def _update_snippet_label(self):
        if self.wav_duration <= 0:
            self.snippet_label.configure(text="(load a WAV first)")
            return
        dur = self.snippet_end - self.snippet_start
        self.snippet_label.configure(
            text=f"{_format_time(self.snippet_start)} → {_format_time(self.snippet_end)}  "
                 f"(selected {_format_time(dur)} of {_format_time(self.wav_duration)})")

    def _on_snippet_press(self, event):
        if self.wav_duration <= 0:
            return
        sx, ex = self._snippet_time_to_x(self.snippet_start), self._snippet_time_to_x(self.snippet_end)
        if abs(event.x - sx) <= 6:
            self._snippet_drag = "start"
        elif abs(event.x - ex) <= 6:
            self._snippet_drag = "end"
        else:
            self._snippet_drag = None
            self._play_preview_at(self._snippet_x_to_time(event.x))

    def _on_snippet_drag(self, event):
        if not self._snippet_drag or self.wav_duration <= 0:
            return
        t = self._snippet_x_to_time(event.x)
        if self._snippet_drag == "start":
            self.snippet_start = max(0.0, min(t, self.snippet_end - 0.2))
        else:
            self.snippet_end = min(self.wav_duration, max(t, self.snippet_start + 0.2))
        self._draw_waveform()

    def _on_snippet_release(self, event):
        self._snippet_drag = None

    def _reset_snippet(self):
        self.snippet_start = 0.0
        self.snippet_end = self.wav_duration
        self._draw_waveform()

    def _play_selection(self):
        if not _HAS_WINSOUND or not self.wav_path or self.wav_duration <= 0:
            return
        self._play_range(self.snippet_start, self.snippet_end)

    def _play_preview_at(self, t: float):
        if not _HAS_WINSOUND or not self.wav_path or self.wav_duration <= 0:
            return
        self._play_range(t, min(self.wav_duration, t + SNIPPET_PREVIEW_SECONDS))

    def _play_range(self, start: float, end: float):
        wav_path = self.wav_path

        def worker():
            try:
                old = self._snippet_temp_wav
                fd, path = tempfile.mkstemp(suffix=".wav", prefix="y2k_preview_")
                os.close(fd)
                trim_wav(wav_path, start, end, path)
                self._snippet_temp_wav = path
                winsound.PlaySound(path, winsound.SND_FILENAME | winsound.SND_ASYNC)
                if old:
                    try:
                        os.remove(old)
                    except OSError:
                        pass
            except Exception as e:
                self.root.after(0, lambda: self.status_var.set(f"Playback error: {e}"))
                return
            self.root.after(0, lambda: self._set_playing_state(True))

        threading.Thread(target=worker, daemon=True).start()

    def _stop_playback(self):
        if _HAS_WINSOUND:
            try:
                winsound.PlaySound(None, 0)
            except Exception:
                pass
        self._set_playing_state(False)

    def _set_playing_state(self, playing: bool):
        self._is_playing = playing
        if hasattr(self, "stop_playback_btn"):
            self.stop_playback_btn.configure(state="normal" if playing else "disabled")

    def _analyze_wav_async(self):
        self._analysis_token += 1
        token = self._analysis_token
        path = self.wav_path

        def worker():
            try:
                feat = analyze(path, fps=30.0)
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Analysis failed", str(e)))
                self.root.after(0, lambda: self.status_var.set("Audio analysis failed."))
                return
            if token != self._analysis_token:
                return  # a newer file was picked while this was analyzing

            def apply():
                self.features = feat
                self.preview_time = 0.0
                self.status_var.set(
                    f"Analyzed {os.path.basename(path)}: {feat.duration:.1f}s, "
                    f"{int(feat.is_beat.sum())} beats, {int(feat.is_drop.sum())} drops.")
            self.root.after(0, apply)

        threading.Thread(target=worker, daemon=True).start()

    def _pick_output(self):
        path = filedialog.asksaveasfilename(title="Save video as", defaultextension=".mp4",
                                             filetypes=[("MP4 video", "*.mp4")])
        if path:
            self.out_path_var.set(path)

    # ------------------------------------------------------------------
    # live preview loop
    # ------------------------------------------------------------------
    def _reset_preview_state(self):
        self.preview_states = {name: {} for name in PATTERN_NAMES}

    def _current_controls(self) -> Controls:
        return Controls(
            chaos=self.chaos.get(),
            bass_gain=self.bass_gain.get(),
            mid_gain=self.mid_gain.get(),
            treble_gain=self.treble_gain.get(),
            onset_gain=self.onset_gain.get(),
            glow_strength=self.glow_strength.get(),
            particle_density=self.particle_density.get(),
            switch_speed=self.switch_speed.get(),
        ).clamp()

    def _current_palette(self) -> dict:
        return build_custom_palette(
            base=self.palette_preset.get(),
            bg=self.bg_color.get(), accent=self.accent_color.get(), glow=self.glow_color.get(),
            colors=[v.get() for v in self.color_vars],
        )

    def _fake_feat(self, t: float) -> dict:
        bpm = 120.0
        beat_dur = 60.0 / bpm
        beat_idx = int(t / beat_dur)
        is_beat = beat_idx != self._last_fake_beat_idx
        self._last_fake_beat_idx = beat_idx
        phase = (t % beat_dur) / beat_dur
        bass = float(np.exp(-phase * 9.0))
        mid = float(0.4 + 0.3 * np.sin(t * 1.3))
        treble = float(0.3 + 0.3 * np.sin(t * 2.7 + 1.0))
        rms = float(np.clip(0.3 + 0.5 * bass, 0, 1))
        onset = 1.0 if is_beat else float(np.clip(bass * 0.3, 0, 1))
        is_drop = (t % 8.0) < 0.25
        return dict(rms=rms, bass=np.clip(bass, 0, 1), mid=np.clip(mid, 0, 1),
                    treble=np.clip(treble, 0, 1), onset=onset, is_beat=is_beat, is_drop=is_drop)

    def _tick_preview(self):
        dt = 1.0 / PREVIEW_FPS
        self.preview_time += dt
        pattern = self.preview_pattern.get()

        if self.features is not None:
            idx = int(self.preview_time * self.features.fps) % self.features.n_frames
            feat = self.features[idx]
        else:
            feat = self._fake_feat(self.preview_time)

        ctrl = self._current_controls()
        feat = ctrl.apply_to_feature(feat)
        pal = self._current_palette()

        self.preview_local_t[pattern] = self.preview_local_t.get(pattern, 0.0) + dt
        state = self.preview_states.setdefault(pattern, {})
        fn = PATTERN_REGISTRY[pattern]
        try:
            img = fn(PREVIEW_W, PREVIEW_H, feat, self.preview_local_t[pattern],
                      self.preview_rng, pal, ctrl.as_dict(), state)
            self.preview_photo = ImageTk.PhotoImage(img)
            self.preview_canvas.itemconfig(self.preview_image_id, image=self.preview_photo)
        except Exception as e:
            self.status_var.set(f"Preview error in {pattern}: {e}")

        self._preview_after_id = self.root.after(int(1000 / PREVIEW_FPS), self._tick_preview)

    # ------------------------------------------------------------------
    # rendering
    # ------------------------------------------------------------------
    def _start_render(self):
        if not self.wav_path:
            messagebox.showwarning("No file", "Pick a WAV file first.")
            return
        pool = [name for name, var in self.pattern_enabled.items() if var.get()]
        if not pool:
            messagebox.showwarning("No patterns", "Enable at least one pattern.")
            return

        out_path = self.out_path_var.get().strip() or (self.wav_path.rsplit(".", 1)[0] + "_y2k.mp4")
        seed = None
        if self.seed_var.get().strip():
            try:
                seed = int(self.seed_var.get().strip())
            except ValueError:
                messagebox.showwarning("Bad seed", "Seed must be an integer, or leave it blank.")
                return

        controls = self._current_controls()
        cfg = dict(
            wav=self.wav_path, out=out_path,
            resolution=self.resolution.get(), fps=self.fps.get(),
            seed=seed, palette=self.palette_preset.get(),
            bg_color=self.bg_color.get(), accent_color=self.accent_color.get(),
            glow_color=self.glow_color.get(), custom_colors=[v.get() for v in self.color_vars],
            patterns=pool,
            **controls.as_dict(),
        )
        # only pass a trim range if the user actually narrowed it from the
        # full track -- avoids an unnecessary re-copy of the whole WAV
        if self.wav_duration > 0:
            trimmed = self.snippet_start > 0.05 or self.snippet_end < self.wav_duration - 0.05
            if trimmed:
                cfg["start"] = round(self.snippet_start, 2)
                cfg["end"] = round(self.snippet_end, 2)

        fd, cfg_path = tempfile.mkstemp(suffix=".json", prefix="y2k_gui_config_")
        with os.fdopen(fd, "w") as fh:
            json.dump(cfg, fh, indent=2)

        self._stop_playback()
        self.render_button.configure(state="disabled")
        self.cancel_button.configure(state="normal")
        self.progress.configure(value=0)
        self.status_var.set(f"Rendering -> {out_path} ...")

        creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        proc = subprocess.Popen(
            _render_subprocess_cmd(cfg_path),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
            creationflags=creationflags,
        )
        self._render_proc = proc
        threading.Thread(target=self._watch_render, args=(proc, cfg_path, out_path), daemon=True).start()

    def _watch_render(self, proc: subprocess.Popen, cfg_path: str, out_path: str):
        pct_re = re.compile(r"(\d+(?:\.\d+)?)%")
        last_line = ""
        try:
            for line in proc.stdout:
                last_line = line.strip()
                m = pct_re.search(line)
                if m:
                    pct = float(m.group(1))
                    self.root.after(0, lambda p=pct: self.progress.configure(value=p))
        except Exception:
            pass
        proc.wait()
        try:
            os.remove(cfg_path)
        except OSError:
            pass
        ok = proc.returncode == 0
        self.root.after(0, lambda: self._render_done(ok, out_path, last_line))

    def _render_done(self, ok: bool, out_path: str, last_line: str):
        self.render_button.configure(state="normal")
        self.cancel_button.configure(state="disabled")
        self._render_proc = None
        if ok:
            self.progress.configure(value=100)
            self.status_var.set(f"Done! Saved to {out_path}")
            messagebox.showinfo("Render complete", f"Your video is ready:\n{out_path}")
        else:
            self.status_var.set("Render failed or was canceled — see terminal for details.")
            messagebox.showerror("Render failed", f"Something went wrong.\n\nLast output line:\n{last_line}")

    def _cancel_render(self):
        if self._render_proc and self._render_proc.poll() is None:
            self._render_proc.terminate()
            self.status_var.set("Canceling render...")


def main():
    root = tk.Tk()
    VisualizerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
