#!/usr/bin/env python3
"""
render.py — Y2K Chaotic Music Visualizer

Turn a song into a chaotic, Y2K-styled visualizer MP4. Accepts a WAV,
an MP3, or a video file (MP4 and anything else ffmpeg can decode audio
from -- its audio track is extracted automatically). Analyzes the audio
(loudness, bass/mid/treble energy, beats, drops) and drives dozens of
different generative visual patterns (multiple scene packs) that
auto-cycle with the music (or follow a sequence you specify).

USAGE
    python3 render.py song.wav
    python3 render.py song.mp3
    python3 render.py music_video.mp4                         # audio track is extracted
    python3 render.py song.wav --out song_viz.mp4 --chaos 0.8 --palette vapor
    python3 render.py song.wav --preview                     # fast low-res test
    python3 render.py song.wav --patterns chrome_tunnel,glitch_vhs,particle_burst
    python3 render.py song.wav --sequence "0:chrome_tunnel,20:glitch_vhs,45:kaleidoscope"
    python3 render.py song.wav --bg-color "#0a0514" --accent-color "#ff2fb0" \\
                                --custom-colors "#ff2fb0,#2fe6ff,#fff23f" \\
                                --bass-gain 1.4 --glow-strength 1.6 --particle-density 1.8
    python3 gui.py                                            # full desktop GUI

Or drive everything from a single JSON config (this is what gui.py writes):
    python3 render.py --config settings.json

Run `python3 render.py --list-patterns` to see all available patterns.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time

import numpy as np
from PIL import Image

from audio_analysis import analyze, trim_wav
from director import Director, parse_sequence_arg, parse_weights_arg
from patterns import PATTERN_REGISTRY, PATTERN_NAMES, SCENE_PACKS
from palettes import PALETTES, DEFAULT_PALETTE, build_custom_palette
from controls import Controls


def parse_resolution(s: str) -> tuple[int, int]:
    w, h = s.lower().split("x")
    return int(w), int(h)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Y2K chaotic music visualizer renderer")
    p.add_argument("wav", nargs="?",
                    help="Path to input audio (.wav, .mp3, ...) or video (.mp4, ...) file -- "
                         "anything ffmpeg can decode audio from. Non-WAV input has its audio "
                         "track extracted automatically.")
    p.add_argument("--config", default=None,
                    help="Path to a JSON config (as written by gui.py) providing any/all of the "
                         "options below; explicit CLI flags take precedence over config values")
    p.add_argument("--out", default=None, help="Output MP4 path (default: <wav name>_y2k.mp4)")
    p.add_argument("--fps", type=float, default=None, help="Frames per second (default 30)")
    p.add_argument("--resolution", default=None, help="WxH, e.g. 1920x1080 (default 1280x720)")
    p.add_argument("--preview", action="store_true", help="Fast low-res (640x360 @ 20fps) test render")
    p.add_argument("--start", type=float, default=None,
                    help="Only render from this many seconds into the WAV (default: 0 / start of file)")
    p.add_argument("--end", type=float, default=None,
                    help="Only render up to this many seconds into the WAV (default: end of file)")
    p.add_argument("--chaos", type=float, default=None, help="0.0 (calm) .. 1.0 (max chaos), default 0.65")
    p.add_argument("--seed", type=int, default=None, help="Random seed for reproducible output")
    p.add_argument("--palette", default=None, choices=list(PALETTES.keys()),
                    help=f"Color palette preset (default {DEFAULT_PALETTE})")
    p.add_argument("--bg-color", default=None, help="Hex color overriding the palette background, e.g. #0a0514")
    p.add_argument("--accent-color", default=None, help="Hex color overriding the palette accent")
    p.add_argument("--glow-color", default=None, help="Hex color overriding the palette glow tint")
    p.add_argument("--custom-colors", default=None,
                    help="Comma list of hex colors overriding the palette's main gradient colors, "
                         "e.g. '#ff2fb0,#2fe6ff,#fff23f,#b967ff'")
    p.add_argument("--bass-gain", type=float, default=None, help="Bass reactivity multiplier (default 1.0)")
    p.add_argument("--mid-gain", type=float, default=None, help="Mid reactivity multiplier (default 1.0)")
    p.add_argument("--treble-gain", type=float, default=None, help="Treble reactivity multiplier (default 1.0)")
    p.add_argument("--onset-gain", type=float, default=None, help="Beat/transient reactivity multiplier (default 1.0)")
    p.add_argument("--glow-strength", type=float, default=None, help="Neon glow intensity multiplier (default 1.0)")
    p.add_argument("--particle-density", type=float, default=None, help="Particle/spark/dot count multiplier (default 1.0)")
    p.add_argument("--switch-speed", type=float, default=None, help="Pattern cut-frequency multiplier (default 1.0)")
    p.add_argument("--patterns", default=None,
                    help="Comma list restricting the auto-cycle pool, e.g. chrome_tunnel,glitch_vhs")
    p.add_argument("--exclude", default=None, help="Comma list of patterns to exclude from the pool")
    p.add_argument("--weights", default=None,
                    help="Bias pattern selection, e.g. 'particle_burst=2.0,kaleidoscope=0.5'")
    p.add_argument("--sequence", default=None,
                    help="Force specific patterns at specific times (overrides auto for those ranges): "
                         "'0:chrome_tunnel,15.5:particle_burst,40:glitch_vhs'")
    p.add_argument("--list-patterns", action="store_true", help="List available patterns and exit")
    p.add_argument("--dry-run", action="store_true", help="Analyze + build the schedule, print a summary, skip rendering")
    return p


def merge_config(args: argparse.Namespace) -> argparse.Namespace:
    """Fill in any option left at its argparse default (None) from a JSON
    config file. Explicit CLI flags always win over the config."""
    if not args.config:
        return args
    with open(args.config) as fh:
        cfg = json.load(fh)
    for key, value in cfg.items():
        attr = key.replace("-", "_")
        if hasattr(args, attr) and getattr(args, attr) is None:
            setattr(args, attr, value)
    return args


def _ffmpeg_path() -> str:
    """Resolve the ffmpeg binary to run. The Windows installer bundles a
    static ffmpeg.exe right next to the app so people don't need to
    install anything separately -- prefer that if present, and fall back
    to whatever "ffmpeg" resolves to on PATH otherwise (dev machines,
    Linux/Mac, Docker, etc.)."""
    candidates = []
    if getattr(sys, "frozen", False):
        # PyInstaller sets sys.executable to the running app's own exe path
        base = os.path.dirname(sys.executable)
        candidates.append(os.path.join(base, "ffmpeg.exe"))
    here = os.path.dirname(os.path.abspath(__file__))
    candidates.append(os.path.join(here, "ffmpeg.exe"))
    for c in candidates:
        if os.path.isfile(c):
            return c
    return "ffmpeg"


# Extensions accepted straight through with a plain byte copy (no re-encode).
# Anything else -- mp3, mp4, m4a, mov, and whatever other container/codec
# ffmpeg can decode -- gets its audio extracted via extract_audio_to_wav
# instead, so the app isn't limited to literal .wav input.
_RAW_WAV_EXTS = {".wav"}


def extract_audio_to_wav(src_path: str, dst_path: str) -> None:
    """Decode the audio track of any ffmpeg-readable file (mp3, mp4, m4a,
    mov, ...) into a plain PCM WAV at dst_path. Used so the app can accept
    an MP3 or a video file directly instead of requiring a pre-extracted
    WAV -- everything downstream (analysis, trimming, the final mux) only
    ever has to deal with WAV either way.

    -vn drops any video stream (irrelevant for an MP4 input, a no-op for
    audio-only input) so ffmpeg doesn't spend time decoding/copying video
    it's just going to throw away.
    """
    cmd = [
        _ffmpeg_path(), "-y",
        "-i", src_path,
        "-vn", "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2",
        dst_path,
    ]
    creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, creationflags=creationflags)
    if result.returncode != 0 or not os.path.isfile(dst_path) or os.path.getsize(dst_path) < 64:
        raise RuntimeError(
            f"ffmpeg couldn't extract audio from {src_path!r} (exit code {result.returncode}). "
            f"This usually means the file has no audio track, or is corrupted.\n\n"
            f"ffmpeg output:\n{result.stdout}"
        )


def make_ffmpeg_process(out_path: str, wav_path: str, w: int, h: int, fps: float):
    cmd = [
        _ffmpeg_path(), "-y",
        "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", f"{w}x{h}",
        "-framerate", str(fps), "-i", "-",
        "-i", wav_path,
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "medium", "-crf", "18",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        out_path,
    ]
    # ffmpeg.exe is a console-subsystem program. This render worker itself
    # has no console (it was launched with CREATE_NO_WINDOW), so without
    # explicitly suppressing it here too, Windows auto-allocates a brand
    # new, VISIBLE console window for ffmpeg to attach to -- popping open
    # what looks like a random black terminal every single export. Worse,
    # that window defaults to "QuickEdit mode", where a single click
    # inside it freezes the process until Enter is pressed -- a very
    # plausible reason the pipe to it was breaking mid-render. Same fix as
    # the worker launch in gui.py: CREATE_NO_WINDOW keeps it fully hidden.
    creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                             creationflags=creationflags)


class _StderrDrain:
    """Continuously drains a subprocess's stderr pipe in a background thread.

    ffmpeg writes a steady stream of progress/stats text to stderr for as
    long as it runs. If nothing reads that pipe, the OS pipe buffer
    eventually fills up -- a render long enough to matter produces easily
    enough text to hit this, especially on Windows where the default
    anonymous-pipe buffer is much smaller than on Linux -- and ffmpeg
    blocks trying to write more of it. That in turn blocks *us* forever in
    proc.wait(), since ffmpeg can never get to exiting. This is a real,
    reproducible Python subprocess deadlock (see the "Popen.wait()" warning
    in the subprocess docs), and it looks exactly like a render silently
    hanging forever right at "0s remaining" -- all frames get written and
    the last progress line prints fine, but the process never actually
    finishes. Draining continuously here (keeping only the last N lines,
    which is all we need for diagnostics if something does go wrong)
    avoids the deadlock entirely, regardless of how much stderr ffmpeg
    produces over the life of a render.
    """

    def __init__(self, pipe, max_lines: int = 2000):
        self._pipe = pipe
        self._lines = collections.deque(maxlen=max_lines)
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        try:
            for line in iter(self._pipe.readline, b""):
                with self._lock:
                    self._lines.append(line)
        except Exception:
            pass
        finally:
            try:
                self._pipe.close()
            except Exception:
                pass

    def text(self) -> str:
        with self._lock:
            lines = list(self._lines)
        return b"".join(lines).decode(errors="ignore")

    def join(self, timeout: float = 2.0):
        self._thread.join(timeout=timeout)


def main(argv=None):
    # Force unbuffered/line-buffered stdout+stderr. In dev mode gui.py launches
    # this with `python -u`, which guarantees prompt flushing; a packaged/
    # frozen exe has no such flag to pass (PyInstaller bootloaders don't
    # accept Python's own CLI switches), so without this fix the C stdio
    # layer can sit on every print() in a block buffer for minutes at a
    # time -- the render is genuinely progressing, but the GUI watching
    # this process's stdout sees nothing arrive and looks completely
    # frozen. Also guard against sys.stdout/stderr being None, which
    # PyInstaller's windowed (console=False) builds can produce when no
    # handle was inherited -- printing to None would crash the worker
    # silently from the GUI's point of view.
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w")
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w")
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except Exception:
        pass

    args = build_arg_parser().parse_args(argv)
    args = merge_config(args)

    if args.list_patterns:
        print(f"Available patterns ({len(PATTERN_NAMES)} total, {len(SCENE_PACKS)} packs):")
        for pack, names in SCENE_PACKS.items():
            print(f"\n  [{pack}]")
            for n in names:
                print(f"    - {n}")
        print("\nAvailable palettes:")
        for n in PALETTES:
            print(f"  - {n}")
        return 0

    if not args.wav:
        print("error: a WAV file path is required (or use --list-patterns)", file=sys.stderr)
        return 2

    if args.preview:
        w, h = 640, 360
        fps = 20.0
    else:
        w, h = parse_resolution(args.resolution or "1280x720")
        fps = args.fps if args.fps is not None else 30.0

    out_path = args.out or (args.wav.rsplit(".", 1)[0] + "_y2k.mp4")
    seed = args.seed if args.seed is not None else np.random.randint(0, 1_000_000)
    chaos = args.chaos if args.chaos is not None else 0.65

    # Copy the input WAV to a local temp file up front, and work off that
    # copy for everything from here on (analysis AND the final ffmpeg
    # mux). Reading directly, repeatedly, over many seconds from a path
    # inside a cloud-synced folder is risky the same way writing to one
    # is -- OneDrive in particular silently syncs "Desktop" and
    # "Documents" on Windows by default, can mark files "cloud-only" and
    # fetch them on demand, and its sync client can interfere with a file
    # while something else is mid-read. A single upfront copy means the
    # rest of the run never touches the original path again, regardless
    # of where the user's WAV actually lives.
    local_wav_fd, local_wav_path = tempfile.mkstemp(suffix=".wav", prefix="y2k_input_")
    os.close(local_wav_fd)
    src_ext = os.path.splitext(args.wav)[1].lower()
    try:
        if src_ext in _RAW_WAV_EXTS:
            print(f"      copying input WAV to a local temp file (source may be cloud-synced): {args.wav}")
            shutil.copyfile(args.wav, local_wav_path)
        else:
            # Not a WAV -- an MP3, an MP4 (or any other video with an audio
            # track), or anything else ffmpeg can decode. Extract its audio
            # into the same local temp WAV rather than requiring the caller
            # to have pre-extracted it themselves.
            print(f"      extracting audio from {args.wav} (not a WAV -- decoding via ffmpeg)")
            extract_audio_to_wav(args.wav, local_wav_path)
    except Exception as e:
        # Consistent with every other failure path in this function: a
        # clear one-line reason, temp file cleaned up, exit code 1 --
        # rather than an uncaught exception's raw traceback (which,
        # unlike everywhere else here, wouldn't even clean up the empty
        # temp WAV this just created).
        print(f"\nCouldn't read the input file {args.wav!r}: {e}", file=sys.stderr)
        _cleanup_tmp(local_wav_path)
        return 1

    # if a snippet range was requested, losslessly trim the (now-local) WAV
    # first -- everything downstream (analysis + the audio ffmpeg muxes in)
    # then just works off this shorter file without any further changes
    wav_path = local_wav_path
    tmp_trim_path = None
    if args.start is not None or args.end is not None:
        start = max(0.0, args.start or 0.0)
        end = args.end
        fd, tmp_trim_path = tempfile.mkstemp(suffix=".wav", prefix="y2k_snippet_")
        os.close(fd)
        trim_wav(local_wav_path, start, end, tmp_trim_path)
        wav_path = tmp_trim_path
        end_label = f"{end:.1f}s" if end is not None else "end"
        print(f"      using snippet {start:.1f}s -> {end_label} of {args.wav}")

    controls = Controls(
        chaos=chaos,
        bass_gain=args.bass_gain if args.bass_gain is not None else 1.0,
        mid_gain=args.mid_gain if args.mid_gain is not None else 1.0,
        treble_gain=args.treble_gain if args.treble_gain is not None else 1.0,
        onset_gain=args.onset_gain if args.onset_gain is not None else 1.0,
        glow_strength=args.glow_strength if args.glow_strength is not None else 1.0,
        particle_density=args.particle_density if args.particle_density is not None else 1.0,
        switch_speed=args.switch_speed if args.switch_speed is not None else 1.0,
    ).clamp()

    print(f"[1/3] Analyzing audio: {args.wav}")
    t0 = time.time()
    feat = analyze(wav_path, fps=fps)
    print(f"      duration={feat.duration:.1f}s  frames={feat.n_frames}  "
          f"beats={int(feat.is_beat.sum())}  drops={int(feat.is_drop.sum())}  "
          f"({time.time()-t0:.1f}s)")

    if isinstance(args.patterns, list):
        pool = list(args.patterns)
    elif args.patterns:
        pool = args.patterns.split(",")
    else:
        pool = list(PATTERN_NAMES)
    if args.exclude:
        excl = set(args.exclude if isinstance(args.exclude, list) else args.exclude.split(","))
        pool = [p for p in pool if p not in excl]
    if not pool:
        print("error: pattern pool is empty after --patterns/--exclude filtering", file=sys.stderr)
        return 2

    weights = args.weights if isinstance(args.weights, dict) else parse_weights_arg(args.weights)
    forced = args.sequence if isinstance(args.sequence, list) else parse_sequence_arg(args.sequence)

    director = Director(feat, pool=pool, weights=weights, forced_segments=forced,
                         chaos=controls.chaos, seed=seed, switch_speed=controls.switch_speed)
    schedule = director.build_schedule()
    print(f"[2/3] Built pattern schedule (seed={seed}, chaos={controls.chaos}):")
    for line in director.summary().split("\n"):
        print(f"      {line}")

    if args.dry_run:
        _cleanup_tmp(local_wav_path)
        if tmp_trim_path:
            _cleanup_tmp(tmp_trim_path)
        print("Dry run complete (no video rendered).")
        return 0

    palette = build_custom_palette(
        base=args.palette or DEFAULT_PALETTE,
        bg=args.bg_color, accent=args.accent_color, glow=args.glow_color,
        colors=args.custom_colors.split(",") if isinstance(args.custom_colors, str) else args.custom_colors,
    )

    # Per-pattern persistent state + local clocks + dedicated RNGs (one per
    # pattern name, seeded deterministically so a given seed always
    # reproduces the same visuals).
    pattern_state = {name: {} for name in PATTERN_NAMES}
    pattern_local_t = {name: 0.0 for name in PATTERN_NAMES}
    pattern_rng = {name: np.random.default_rng(seed + i * 7919) for i, name in enumerate(PATTERN_NAMES)}
    ctrl_dict = controls.as_dict()

    # Render to an isolated local temp file first, then move the FINISHED
    # file into place at out_path as the very last step. Writing directly
    # into out_path is risky if it's inside a cloud-synced folder --
    # OneDrive in particular very often silently syncs "Desktop" and
    # "Documents" on Windows by default -- because the sync client can
    # grab/lock/upload the file while ffmpeg is still actively writing to
    # it over many seconds, which can truncate or corrupt it (missing moov
    # atom = "file was never properly finalized" is the classic symptom).
    # A single move of an already-complete file avoids that whole class of
    # bug regardless of where out_path actually points.
    tmp_out_fd, tmp_out_path = tempfile.mkstemp(suffix=".mp4", prefix="y2k_render_out_")
    os.close(tmp_out_fd)
    os.remove(tmp_out_path)  # ffmpeg needs to create this file itself

    print(f"[3/3] Rendering {feat.n_frames} frames at {w}x{h}@{fps}fps -> {out_path}")
    proc = make_ffmpeg_process(tmp_out_path, wav_path, w, h, fps)
    stderr_drain = _StderrDrain(proc.stderr)

    t0 = time.time()
    dt = 1.0 / fps
    last_print = t0
    try:
        for i in range(feat.n_frames):
            pattern = schedule[i]
            fn = PATTERN_REGISTRY[pattern]
            f = controls.apply_to_feature(feat[i])

            pattern_local_t[pattern] += dt
            img = fn(w, h, f, pattern_local_t[pattern], pattern_rng[pattern], palette,
                      ctrl_dict, pattern_state[pattern])
            if img.size != (w, h):
                img = img.resize((w, h))
            if img.mode != "RGB":
                img = img.convert("RGB")

            proc.stdin.write(img.tobytes())

            # Wall-clock-paced progress updates instead of every-Nth-frame --
            # a frame-count checkpoint (e.g. "every 60 frames") goes quiet
            # for longer and longer stretches of real time as rendering
            # slows down, which is exactly backwards: slow renders are
            # when you most need reassurance it's still moving. This way
            # the GUI gets a fresh line at least once a second no matter
            # how fast or slow any given pattern is, plus an ETA so a
            # genuinely slow-but-healthy render doesn't read as "stuck".
            now = time.time()
            if now - last_print >= 1.0 or i == feat.n_frames - 1:
                last_print = now
                elapsed = now - t0
                pct = (i + 1) / feat.n_frames * 100
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                remaining = feat.n_frames - (i + 1)
                eta = remaining / rate if rate > 0 else 0
                print(f"      {pct:5.1f}%  frame {i+1}/{feat.n_frames}  "
                      f"({rate:.1f} fps render speed, ~{eta:.0f}s remaining, pattern={pattern})",
                      flush=True)
    except (BrokenPipeError, OSError) as e:
        # Windows doesn't reliably raise BrokenPipeError for a dead pipe --
        # it can surface as a plain OSError (commonly errno 22, "Invalid
        # argument") instead, which the original `except BrokenPipeError`
        # here didn't catch. That let the real reason ffmpeg died (visible
        # in its own stderr) get lost behind an unhelpful, unexplained
        # Python-side traceback. Catch both, and always show ffmpeg's own
        # stderr -- that's the actual diagnostic information.
        stderr_drain.join()
        stderr = stderr_drain.text()
        print(f"\nffmpeg pipe broke ({e!r}). ffmpeg stderr:\n" + stderr, file=sys.stderr)
        _cleanup_tmp(tmp_out_path)
        _cleanup_tmp(local_wav_path)
        if tmp_trim_path:
            _cleanup_tmp(tmp_trim_path)
        return 1
    except Exception as e:
        # A bug triggered by a specific pattern/frame (or anything else
        # unexpected during rendering) previously propagated straight out
        # of this function uncaught -- the worker process would just die
        # with a raw Python traceback and no framing, which on a long
        # render buries the one useful fact (which pattern, which frame)
        # in whatever else happened to be in the traceback. Catching it
        # here makes the failure loud and specific, and lets the worker
        # exit cleanly (temp files removed, ffmpeg killed) rather than
        # however an uncaught exception happens to unwind through the
        # `finally` below and the interpreter shutdown path.
        import traceback as _tb
        print(f"\nRender crashed on frame {i + 1}/{feat.n_frames} (pattern={schedule[i]}): {e!r}\n"
              + _tb.format_exc(), file=sys.stderr)
        stderr_drain.join()
        try:
            proc.kill()
        except Exception:
            pass
        _cleanup_tmp(tmp_out_path)
        _cleanup_tmp(local_wav_path)
        if tmp_trim_path:
            _cleanup_tmp(tmp_trim_path)
        return 1
    finally:
        if proc.stdin:
            proc.stdin.close()

    ret = proc.wait()
    stderr_drain.join()
    _cleanup_tmp(local_wav_path)
    if tmp_trim_path:
        _cleanup_tmp(tmp_trim_path)
    if ret != 0:
        stderr = stderr_drain.text()
        print(f"\nffmpeg exited with code {ret}:\n{stderr}", file=sys.stderr)
        _cleanup_tmp(tmp_out_path)
        return 1

    # Sanity-check the rendered file before trusting it -- a file that's
    # missing or implausibly tiny for the frame count means something went
    # wrong even though ffmpeg reported success (e.g. it got killed by
    # something outside our control right at the very end).
    try:
        size = os.path.getsize(tmp_out_path)
    except OSError:
        size = 0
    if size < 4096:
        print(f"\nRendered file is suspiciously small ({size} bytes) -- treating this as a failure "
              f"rather than handing over a broken video.", file=sys.stderr)
        _cleanup_tmp(tmp_out_path)
        return 1

    try:
        out_dir = os.path.dirname(os.path.abspath(out_path))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        shutil.move(tmp_out_path, out_path)
    except Exception as e:
        print(f"\nRender finished ok, but couldn't move it to {out_path}: {e}\n"
              f"The finished video is sitting at: {tmp_out_path}", file=sys.stderr)
        return 1

    print(f"\nDone in {time.time()-t0:.1f}s -> {out_path}")
    return 0


def _cleanup_tmp(path: str):
    if not path:
        return
    try:
        os.remove(path)
    except OSError:
        pass


if __name__ == "__main__":
    raise SystemExit(main())
