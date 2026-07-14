# Kami — Y2K Chaotic Music Visualizer

Turns a `.wav` file into a chaotic, Y2K-styled music visualizer video (MP4).
It analyzes your audio's loudness, bass/mid/treble energy, beats, and
"drops," then drives 30 different generative visual patterns — organized
into 4 scene packs — that auto-cycle in sync with the music: chrome
tunnels and glitchy VHS plasma, cars burning out, drag-racing, and driving
head-on down a neon highway, purple sunsets with comets and shooting stars,
and chunky retro-Y2K desktop-toy scenes.

No paid libraries or GPU needed — just Python, numpy/scipy/Pillow, and
ffmpeg. Audio analysis is done directly from the WAV file (no librosa).

## Just want to use it? (Windows)

Grab **Kami-Setup.exe** from the project's GitHub Releases page, run it,
and follow the installer. It doesn't ask for admin rights, adds a Start
Menu entry and (optionally) a desktop shortcut, and bundles its own copy
of ffmpeg — nothing else to install. This is what you'd hand to someone
who just wants to use the app, not edit its code.

If you're setting this repo up for the first time and don't have a
Releases page yet, see **Building the Windows installer** near the bottom
of this file — it's built automatically, you just need to push the code to
GitHub once.

## Setup (running from source / making changes)

```bash
pip install -r requirements.txt --break-system-packages   # or drop the flag in a venv
```

You also need `ffmpeg` installed and on your PATH (it's used to encode the
final MP4 and mux in your audio). On Mac: `brew install ffmpeg`. On
Ubuntu/Debian: `sudo apt install ffmpeg`.

## Quick start

```bash
python3 render.py your_song.wav
```

This writes `your_song_y2k.mp4` next to your input file, using sensible
defaults (1280x720, 30fps, auto-cycling patterns, medium chaos).

Want a fast low-res test first before committing to a full render?

```bash
python3 render.py your_song.wav --preview
```

## GUI

For picking colors and dialing in reactivity visually, there's a desktop
control panel:

```bash
python3 gui.py
```

This opens a native window (Tkinter — ships with Python on Windows/Mac; on
Debian/Ubuntu Linux you may need `sudo apt install python3-tk` first, since
it's a separate system package there). It has:

- **A WAV file picker.** Once you pick a file, it's analyzed in the
  background and the live preview starts looping through your *actual*
  song's energy/beats — before that, the preview reacts to a simulated
  120bpm pulse so you can still dial in colors immediately.
- **Snippet picker.** Right under the file picker, a waveform strip lets
  you drag green/red handles to pick just the part of the song you want
  rendered (handy for a 20-second clip instead of committing to a full
  3-minute render). Click anywhere else on the waveform to hear a few
  seconds from that point — on Windows this plays instantly via the
  built-in `winsound` module (no extra install); the trim itself still
  works without audio playback on other platforms.
- **Colors tab** — start from a preset palette, then override the
  background, accent, glow, and 5 gradient colors individually with your
  own color pickers.
- **Reactivity tab** — global sliders for chaos, bass/mid/treble/beat
  sensitivity, glow strength, particle density, and pattern switch speed.
  These scale the audio-feature inputs and universal rendering knobs every
  pattern reads, rather than exposing each pattern's internal parameters
  one by one.
- **Patterns tab** — a scrollable, pack-grouped checklist of all 30 scenes
  (4 packs), with "select all / none" buttons per pack, controlling
  which ones are allowed in the auto-cycle pool.
- **Output tab** — resolution/fps/seed/output path.
- **Live preview canvas** — shows whichever pattern you select, animating
  in real time as you move sliders or pick colors.
- **Render Full Video** — kicks off the same pipeline as the CLI (as a
  background process, with a progress bar) using everything you've set.

The window (and its taskbar icon, on Windows) uses Kami's mascot icon — an
original smiling-flower design in a colorful Y2K/kawaii pop-art spirit
(`generate_icon.py` builds it from scratch; `kami.ico` / `kami_icon_1024.png`
are the pre-built results already included in this project).

### Desktop icon (Windows)

To get a proper double-click "Kami" icon on your Desktop instead of typing
`python3 gui.py` in a terminal every time:

1. Make sure you've done the `pip install` step above at least once.
2. Double-click **`create_desktop_shortcut.vbs`** (in this folder). It adds
   a "Kami" shortcut to your Desktop with the flower icon, launching
   `Kami.pyw` silently through `pythonw.exe` (no console window).
3. Double-click the new **Kami** icon on your Desktop any time you want to
   open it.

If the shortcut doesn't open anything, it's almost always because
`pythonw.exe` isn't on your PATH — right-click the Kami shortcut → Properties
→ change "Target" to the full path of `pythonw.exe` on your machine (e.g.
`C:\Users\<you>\AppData\Local\Programs\Python\Python312\pythonw.exe`), or
reinstall Python from python.org with "Add python.exe to PATH" checked.

Everything the GUI does is also available from the command line (see
below) — the GUI just writes a JSON config and runs
`render.py --config <that file>`.

## Options

| Flag | What it does |
|---|---|
| `--out PATH` | Output MP4 path (default: `<input>_y2k.mp4`) |
| `--resolution WxH` | Output resolution, e.g. `1920x1080` (default `1280x720`) |
| `--fps N` | Frames per second (default `30`) |
| `--preview` | Fast 640x360 @ 20fps test render |
| `--start SEC` / `--end SEC` | Only render a snippet of the WAV, from `--start` seconds to `--end` seconds (default: the whole file). The WAV is losslessly trimmed first, so analysis and the final audio track both reflect just that range. |
| `--chaos 0..1` | Overall intensity/chaos — shorter pattern segments, more particles, more glitch (default `0.65`) |
| `--palette NAME` | `chrome`, `millennium`, `candy`, `matrix`, or `vapor` (default `chrome`) |
| `--seed N` | Fix the random seed for a reproducible render |
| `--patterns a,b,c` | Restrict the auto-cycle pool to just these patterns |
| `--exclude a,b` | Drop specific patterns from the auto-cycle pool |
| `--weights "p=1.5,q=0.5"` | Bias how often specific patterns get picked |
| `--sequence "0:chrome_tunnel,20:glitch_vhs,45:kaleidoscope"` | Force specific patterns at specific timestamps (seconds); auto-fills any gaps you don't specify |
| `--list-patterns` | Print all pattern/palette names and exit |
| `--dry-run` | Analyze the audio and print the pattern schedule without rendering (fast — good for testing `--sequence`/`--seed`/`--chaos` combos) |
| `--config PATH` | Load any/all options from a JSON file (this is what `gui.py` uses) — explicit CLI flags still override individual config values |

### Color overrides

| Flag | What it does |
|---|---|
| `--bg-color "#0a0514"` | Override the palette's background color |
| `--accent-color "#ff2fb0"` | Override the palette's accent color (used for sun/glow highlights) |
| `--glow-color "#00ffff"` | Override the palette's glow tint |
| `--custom-colors "#ff2fb0,#2fe6ff,#fff23f,#b967ff,#ffffff"` | Override the full gradient color list used by tunnels/plasma/mandalas/bars |

### Reactivity knobs

| Flag | What it does | Default |
|---|---|---|
| `--bass-gain` | Multiplier on the bass-band feature | `1.0` |
| `--mid-gain` | Multiplier on the mid-band feature | `1.0` |
| `--treble-gain` | Multiplier on the treble-band feature | `1.0` |
| `--onset-gain` | Multiplier on beat/transient strength | `1.0` |
| `--glow-strength` | Multiplier on neon glow/bloom intensity | `1.0` |
| `--particle-density` | Multiplier on particle/spark/glitch-band counts | `1.0` |
| `--switch-speed` | Multiplier on how fast the director cuts between patterns (>1 = faster) | `1.0` |

## The 30 patterns (4 packs)

### Pack 1 — Waveforms

- **chrome_tunnel** — warping chrome/neon ring tunnel, pulses with bass
- **equalizer_bars** — classic mirrored spectrum bar visualizer, chrome-bevel style
- **particle_burst** — exploding Y2K confetti/sparkle particles on every beat
- **kaleidoscope** — 12-fold mirrored mandala of chaotic blobs
- **checker_tunnel** — Tron-style perspective checkerboard floor + pulsing horizon sun
- **glitch_vhs** — plasma field torn apart by scanlines, RGB-split, and datamosh glitches
- **starburst_pop** — halftone dot background with comic-style starburst shapes, CD-cover chaos
- **oscilloscope_wave** — scrolling neon oscilloscope line over a glowing grid

### Pack 2 — Pixel Cars

- **pixel_cars** — multi-lane night traffic, front/rear-view cars driving toward and away down the highway
- **cars_headlights** — a car approaching head-on out of the dark, headlight beams growing as it closes in
- **cars_burnout** — a rear-view car doing a burnout, drift lean, tire smoke and skid marks
- **cars_taillight_trails** — a rear-view car driving away, its taillights streaking off into the distance
- **cars_drag_race** — 5-light countdown tree, two rear-view cars waiting at the launch line, spark burst
- **cars_showroom_spin** — a car rotating on a pedestal under spotlight beams, alternating side profile and front/rear as it turns

### Pack 3 — Sunsets & Space

- **sunset_pixel** — purple/pink synthwave sunset with a chunky striped retro sun, pixel-art style
- **shooting_stars** — pixelated night sky with streaking shooting stars and a crescent moon
- **comet_flyby** — a large comet with a fading tail crossing the screen
- **purple_asteroid** — a tumbling rocky asteroid with craters and a dust trail
- **constellations** — connected star-point constellations twinkling in the dark
- **aurora_borealis** — wavy bands of shifting aurora color
- **meteor_shower** — multiple simultaneous streaking meteors
- **galaxy_swirl** — a bright-cored, 3-armed spiral galaxy

### Pack 4 — Retro Y2K

- **pixel_globe** — spinning blocky globe/orb, straight out of a late-90s "under construction" page
- **pixel_bounce** — chunky pixel heart/star/diamond bouncing DVD-logo style, changing color on each wall hit
- **pixel_rain** — falling 8-bit hearts/stars/diamonds shower
- **cd_burn_spin** — a spinning rainbow CD with a "burning" progress bar
- **virtual_pet** — a bouncing blob creature on a generic egg-shaped handheld device
- **holo_sticker** — concentric rainbow rings and a rotating holographic star
- **chrome_bubble_text** — chunky chrome bubble-letter Y2K words with drop shadows
- **crt_boot** — scanline noise and a fake "BOOTING KAMI OS..." startup screen

---

Most of packs 2-4 (plus `sunset_pixel`/`shooting_stars` in Pack 3 and
`pixel_globe`/`pixel_bounce`/`pixel_rain` in Pack 4) render onto a small
low-resolution internal canvas and scale it back up with nearest-neighbor
upscaling — that's what gives them the chunky, old-game pixel-art look,
independent of your actual output resolution. The cars pack renders at full
resolution with smooth anti-aliased shapes instead, so the cars read as
actual detailed cars rather than tiny pixel blobs.

The cars in Pack 2 are always shown either from the front/rear (on scenes
where the road runs straight into the distance, toward or away from the
camera) or from the side (only on the showroom turntable, where the car is
genuinely being viewed from the side as it spins) — never a side profile on
a straight road, which is what used to make them look like they were
driving sideways.

The **director** (`director.py`) auto-cycles through whichever patterns you
allow, weighting choices by the music's current energy (louder/bassier ->
more likely to cut to high-energy scenes like `particle_burst`, `glitch_vhs`,
`cars_burnout`, `cars_drag_race`, `meteor_shower`;
calmer -> more likely low-energy scenes like `kaleidoscope`, `chrome_tunnel`,
`sunset_pixel`, `pixel_globe`, `galaxy_swirl`), avoids
repeating the last few patterns back-to-back, snaps cuts to nearby beats,
and throws in brief glitch flash-cuts on big volume drops for extra chaos.

Run `python3 render.py --list-patterns` to print the full pack/pattern
listing straight from the code at any time.

## Examples

```bash
# High-chaos vaporwave-colored render
python3 render.py song.wav --chaos 0.9 --palette vapor

# Only use 3 patterns, biased toward particle bursts
python3 render.py song.wav --patterns particle_burst,chrome_tunnel,glitch_vhs --weights "particle_burst=2.0"

# Force the intro and drop, let the rest auto-cycle
python3 render.py song.wav --sequence "0:chrome_tunnel,30:particle_burst"

# Full 1080p render with a fixed seed (reproducible)
python3 render.py song.wav --resolution 1920x1080 --seed 42

# Custom neon-on-black palette with heavy bass reactivity and dense particles
python3 render.py song.wav --bg-color "#0a0014" --custom-colors "#39ff14,#ff073a,#ffee00" \
                            --bass-gain 1.8 --particle-density 2.0 --glow-strength 1.5
```

## Files

- `render.py` — CLI entry point / rendering pipeline (pipes frames into ffmpeg, muxes your audio)
- `gui.py` — Kami's desktop control panel (colors, reactivity sliders, live preview, render button)
- `Kami.pyw` — Windows entry point: double-click to launch the GUI (no console window); this is also what PyInstaller builds `Kami.exe` from, and what re-launches itself as a background render worker (see `--render-worker` in `gui.py`)
- `create_desktop_shortcut.vbs` — one-time script that adds a "Kami" icon to your Windows Desktop when running from source (the installer does this automatically)
- `generate_icon.py` — builds the flower mascot icon from scratch (already run — outputs below are included)
- `kami.ico` / `kami_icon_1024.png` — the pre-built app icon
- `audio_analysis.py` — WAV -> per-frame loudness/bass/mid/treble/beat/drop features, plus the waveform-preview/trim helpers behind the GUI's snippet picker and `--start`/`--end`
- `director.py` — decides which pattern plays when (auto-cycle + manual overrides)
- `patterns.py` — the 30 visual pattern generators, organized into 4 `SCENE_PACKS`
- `controls.py` — the global reactivity knobs (gains/glow/density/switch-speed) shared by the CLI and GUI
- `render_utils.py` — shared drawing/color helpers
- `palettes.py` — the 5 preset palettes + custom-color/hex helpers
- `make_test_wav.py` / `make_demo_wav.py` — synthetic test-tone generators (handy for testing without a real song)
- `packaging/kami.spec` — PyInstaller build spec (source -> `dist/Kami/Kami.exe`)
- `packaging/installer.iss` — Inno Setup script (`dist/Kami/` -> `Kami-Setup.exe`)
- `.github/workflows/build-windows.yml` — CI job that runs both of the above on a Windows runner

## Performance notes

Rendering is single-threaded and CPU-bound (no GPU used). On a modest
machine, expect roughly 8-15 frames/sec render speed at 1280x720 — so a
3-minute song takes somewhere around 5-10 minutes to render. Use
`--preview` while you're dialing in `--chaos`, `--palette`, or `--sequence`,
then do your final render at full resolution once you're happy with the
schedule (you can preview the schedule instantly with `--dry-run`).

## Building the Windows installer

There's a one-time setup step, then it's automatic from there on.

**One-time setup:**

1. Create a new repo on GitHub and push this project to it:
   ```bash
   git init                          # if this folder isn't already a repo
   git add -A
   git commit -m "Kami"
   git branch -M main
   git remote add origin https://github.com/<you>/<repo>.git
   git push -u origin main
   ```
2. That's it — pushing to `main` automatically triggers
   `.github/workflows/build-windows.yml`, which builds `Kami-Setup.exe` on
   a free GitHub-hosted Windows runner (no Windows machine of your own
   needed) and attaches it to the workflow run under the repo's **Actions**
   tab as a downloadable artifact.

**Every time after that:**

- Push a normal commit to `main` -> a fresh installer builds automatically;
  grab it from the **Actions** tab on that run.
- Push a version tag (e.g. `git tag v1.0.0 && git push origin v1.0.0`) ->
  same build, PLUS it's published to the repo's **Releases** page with the
  installer attached. That Releases page is what you'd link people to —
  it's a stable URL that always points at the latest `Kami-Setup.exe`,
  unlike an Actions artifact link (which requires a GitHub login to
  download and expires after 90 days).

To bump the version number that shows up in the installer (Programs and
Features, the installer's title bar), edit `MyAppVersion` in
`packaging/installer.iss` before tagging a release.

**Building locally instead** (only if you specifically want to build on
your own Windows machine rather than via GitHub Actions): install Python,
`pip install -r requirements.txt pyinstaller`, download a static
`ffmpeg.exe` and drop it next to the repo root, run
`pyinstaller packaging/kami.spec --noconfirm`, copy `ffmpeg.exe` into the
resulting `dist/Kami/` folder, then compile `packaging/installer.iss` with
[Inno Setup](https://jrsoftware.org/isinfo.php) (free). This mirrors
exactly what the CI workflow does.
