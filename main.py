"""
main.py -- reel-ripper + reelkit, compressed into one file.

Turns Instagram reels into keyframes + timestamped transcript + caption
metadata, then (optionally) bundles a batch of them into one markdown file
for an LLM. This is a single-file merge of what used to be reel_ripper.py,
bundle.py, register_batch.py, server.py and app.html. Nothing here scores,
classifies, or judges a reel -- that job is left to the model you hand the
bundle to. See README.md in the original project for the full story on why
(TL;DR: pattern matching can't read).

Requires Python 3.10+, ffmpeg on PATH, and an NVIDIA GPU for anything bigger
than the `medium` Whisper model.

Commands
--------
    python main.py                                rip urls.txt (default)
    python main.py URL [URL ...]                   rip specific reels
    python main.py --language ar                   force a language
    python main.py --gpu-check                     full CUDA diagnostic
    python main.py gpu-check                       same, standalone
    python main.py serve                           start the local web app
    python main.py bundle --by date                stitch ripped reels to md
    python main.py register --name jan-hustle      record a batch by hand

Run `python main.py rip --help`, `python main.py bundle --help`, or
`python main.py serve --help` for the full flag list on each.

Files it reads/writes (all under this script's folder):
    urls.txt              one reel URL per line, '#' comments allowed
    out/<shortcode>/       per-reel meta.json, frames/, transcript.txt
    out/batches.json       named batch registry
    out/bundles/*.md       the markdown files handed to an LLM
"""

from __future__ import annotations

import argparse
import ctypes
import importlib.util
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
import uuid
import webbrowser
from collections import defaultdict, deque
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

# Hugging Face's Windows cache works without symlinks; the warning only
# explains that the cache may use a little more disk space. Keep real download
# and model-loading errors visible.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

HERE = Path(__file__).resolve().parent
DEFAULT_URLS = HERE / "urls.txt"
DEFAULT_OUT = HERE / "out"
OUT = DEFAULT_OUT  # used by the `serve` command
INSTAGRAM_SHORTCODE_RX = re.compile(
    r"(?:https?://)?(?:www\.)?instagram\.com/(?:reels?|p|tv)/([A-Za-z0-9_-]+)",
    re.IGNORECASE,
)


# ===========================================================================
# shared utilities
# ===========================================================================

def die(msg: str, code: int = 1):
    print(f"[fatal] {msg}", file=sys.stderr)
    sys.exit(code)


def need(binary: str, hint: str):
    if shutil.which(binary) is None:
        die(f"'{binary}' not on PATH. {hint}")


def run(cmd: list, **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def shortcode(url: str) -> str:
    m = INSTAGRAM_SHORTCODE_RX.search(url)
    return m.group(1) if m else re.sub(r"\W+", "_", url)[-24:]


def parse_urls(blob: str) -> tuple[list[str], list[str]]:
    """Extract and canonicalize reel/post URLs from pasted or file text."""
    urls, bad, seen = [], [], set()
    for raw in re.split(r"[\s,]+", blob.strip()):
        if not raw:
            continue
        match = INSTAGRAM_SHORTCODE_RX.search(raw)
        if match:
            code = match.group(1)
        elif (re.fullmatch(r"[A-Za-z0-9_-]{9,}", raw)
              and any(c.isdigit() for c in raw)
              and any(c.isupper() for c in raw)):
            code = raw
        else:
            if "http" in raw.lower() or "instagram" in raw.lower() or "/" in raw:
                bad.append(raw[:80])
            continue
        if code not in seen:
            seen.add(code)
            urls.append(f"https://www.instagram.com/reel/{code}/")
    return urls, bad


# ===========================================================================
# ripping: download
# ===========================================================================

def download(url: str, dest: Path, cookies: str | None) -> tuple[Path, dict]:
    """Fetch mp4 + metadata. Returns (video_path, info_dict)."""
    dest.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "yt_dlp",
        "--no-playlist",
        "--newline",
        "--retries", "5",
        "--fragment-retries", "5",
        "--socket-timeout", "30",
        "--sleep-requests", "1",
        "--sleep-interval", "1",
        "--max-sleep-interval", "3",
        "--write-info-json",
        "--merge-output-format", "mp4",
        "-o", str(dest / "video.%(ext)s"),
    ]
    if cookies:
        cmd += ["--cookies-from-browser", cookies]
    cmd.append(url)

    print("  [1/4] downloading...")
    tail: deque[str] = deque(maxlen=12)
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        encoding="utf-8", errors="replace", bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip()
        if line:
            tail.append(line)
            print(f"        {line}", flush=True)
    returncode = proc.wait()
    if returncode != 0:
        useful = [line for line in tail if not line.startswith("[download]")][-6:]
        if not useful:
            useful = list(tail)[-6:]
        raise RuntimeError("yt-dlp failed:\n    " + "\n    ".join(useful))

    vids = sorted(dest.glob("video.*"))
    video = next((v for v in vids if v.suffix in {".mp4", ".mkv", ".webm"}), None)
    if video is None:
        raise RuntimeError("no video file produced")

    info = {}
    ij = dest / "video.info.json"
    if ij.exists():
        info = json.loads(ij.read_text(encoding="utf-8", errors="replace"))
        ij.unlink()
    return video, info


def write_meta(info: dict, url: str, dest: Path) -> dict:
    desc = info.get("description") or ""
    meta = {
        "url": url,
        "shortcode": shortcode(url),
        "author": info.get("uploader") or info.get("channel") or info.get("uploader_id"),
        "title": info.get("title"),
        "caption": desc,
        "hashtags": sorted(set(re.findall(r"#(\w+)", desc))),
        "duration_sec": info.get("duration"),
        "like_count": info.get("like_count"),
        "comment_count": info.get("comment_count"),
        "upload_date": info.get("upload_date"),
        "width": info.get("width"),
        "height": info.get("height"),
    }
    (dest / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return meta


# ===========================================================================
# ripping: frames
# ===========================================================================

def extract_frames(video: Path, dest: Path, threshold: float, max_frames: int,
                   fallback_fps: float) -> list[Path]:
    """Scene-change keyframes; falls back to fixed interval if too few."""
    fdir = dest / "frames"
    if fdir.exists():
        shutil.rmtree(fdir)
    fdir.mkdir(parents=True)

    print("  [2/4] extracting frames...")
    result = run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(video),
        "-vf", f"select='gt(scene,{threshold})',scale=768:-2",
        "-vsync", "vfr", "-q:v", "4",
        str(fdir / "f%04d.jpg"),
    ])
    if result.returncode != 0:
        raise RuntimeError("ffmpeg scene extraction failed: "
                           + (result.stderr or "unknown error").strip()[-500:])
    frames = sorted(fdir.glob("*.jpg"))

    if len(frames) < 3:
        for f in frames:
            f.unlink()
        result = run([
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(video),
            "-vf", f"fps={fallback_fps},scale=768:-2",
            "-q:v", "4",
            str(fdir / "f%04d.jpg"),
        ])
        if result.returncode != 0:
            raise RuntimeError("ffmpeg fallback extraction failed: "
                               + (result.stderr or "unknown error").strip()[-500:])
        frames = sorted(fdir.glob("*.jpg"))

    if not frames:
        raise RuntimeError("ffmpeg produced no frames")

    if len(frames) > max_frames:
        step = len(frames) / max_frames
        keep = {frames[int(i * step)] for i in range(max_frames)}
        for f in frames:
            if f not in keep:
                f.unlink()
        frames = sorted(fdir.glob("*.jpg"))

    return frames


# ===========================================================================
# CUDA runtime (Blackwell / sm_120 aware)
# ===========================================================================

CUBLAS_DLLS = ("cublasLt64_12.dll", "cublas64_12.dll")
CUDNN_DLLS = (
    "cudnn_graph64_9.dll",
    "cudnn_ops64_9.dll",
    "cudnn_cnn64_9.dll",
    "cudnn_adv64_9.dll",
    "cudnn64_9.dll",
)
OPTIONAL_DLLS = ("nvrtc64_120_0.dll",)

_CUDA_PREPARED = False
_CUDA_REPORT: dict = {"dirs": [], "loaded": [], "missing": [], "optional": []}
_AV_STUBBED = False
_MODEL_CACHE: dict = {}
_MODEL_PATH_CACHE: dict[tuple[str, bool], str] = {}
_CUDA_DEAD = False
GPU_TIMEOUT_SEC = 90


def _package_dir(name: str) -> Path | None:
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ValueError):
        return None
    if spec is None:
        return None
    if spec.submodule_search_locations:
        return Path(list(spec.submodule_search_locations)[0])
    if spec.origin:
        return Path(spec.origin).parent
    return None


def _cuda_search_dirs() -> list[Path]:
    dirs: list[Path] = []
    nvidia_root = _package_dir("nvidia")
    if nvidia_root:
        dirs += sorted(p for p in nvidia_root.glob("*/bin") if p.is_dir())
        dirs += sorted(p for p in nvidia_root.glob("*/lib") if p.is_dir())
    ct2 = _package_dir("ctranslate2")
    if ct2:
        dirs.append(ct2)
    for var in ("CUDA_PATH", "CUDA_HOME"):
        p = os.environ.get(var)
        if p and (Path(p) / "bin").is_dir():
            dirs.append(Path(p) / "bin")
    seen, out = set(), []
    for d in dirs:
        s = str(d)
        if s not in seen:
            seen.add(s)
            out.append(d)
    return out


def prepare_cuda_runtime() -> dict:
    """Register CUDA dirs, then force-load each DLL by absolute path before
    ctranslate2 is imported, since directory registration alone is unreliable
    for a dependency pulled in by CPython's extension loader."""
    global _CUDA_PREPARED
    if _CUDA_PREPARED:
        return _CUDA_REPORT
    _CUDA_PREPARED = True

    dirs = _cuda_search_dirs()
    _CUDA_REPORT["dirs"] = [str(d) for d in dirs]

    if os.name != "nt":
        return _CUDA_REPORT

    for d in dirs:
        try:
            os.add_dll_directory(str(d))
        except OSError:
            pass
    os.environ["PATH"] = (
        os.pathsep.join(str(d) for d in dirs) + os.pathsep + os.environ.get("PATH", "")
    )

    def preload(names, bucket):
        for name in names:
            hit = next((d / name for d in dirs if (d / name).is_file()), None)
            if hit is None:
                _CUDA_REPORT[bucket].append(f"{name}: not found")
                continue
            try:
                ctypes.WinDLL(str(hit))
                _CUDA_REPORT["loaded"].append(f"{name}  <-  {hit.parent}")
            except OSError as e:
                _CUDA_REPORT[bucket].append(f"{name}: found but load failed ({e})")

    preload(CUBLAS_DLLS + CUDNN_DLLS, "missing")
    preload(OPTIONAL_DLLS, "optional")
    return _CUDA_REPORT


def cuda_hint() -> str:
    miss = _CUDA_REPORT["missing"]
    if not miss:
        return ""
    pkgs = set()
    for m in miss:
        if m.startswith("cublas"):
            pkgs.add("nvidia-cublas-cu12")
        elif m.startswith("cudnn"):
            pkgs.add("nvidia-cudnn-cu12")
    lines = ["", "missing CUDA libraries:"]
    lines += [f"  - {m}" for m in miss]
    if pkgs:
        lines += ["", "install them into this venv:",
                  f"  {sys.executable} -m pip install --upgrade " + " ".join(sorted(pkgs))]
    return "\n".join(lines)


def gpu_info() -> dict:
    out = {}
    exe = shutil.which("nvidia-smi")
    if not exe:
        return out
    r = run([exe, "--query-gpu=name,compute_cap,driver_version,memory.total",
             "--format=csv,noheader"])
    line = (r.stdout or "").strip().splitlines()
    if not line:
        return out
    parts = [p.strip() for p in line[0].split(",")]
    keys = ["name", "compute_cap", "driver", "vram"]
    return dict(zip(keys, parts))


def resolve_compute_type(requested: str, device: str) -> str:
    """int8 on CUDA fails on Blackwell (sm_120) with CUBLAS_STATUS_NOT_
    SUPPORTED, so refuse it up front instead of letting it explode mid-run."""
    if requested != "auto":
        if device == "cuda" and requested.startswith("int8"):
            cap = gpu_info().get("compute_cap", "")
            try:
                blackwell = float(cap) >= 12.0
            except ValueError:
                blackwell = False
            if blackwell:
                die(f"compute_type '{requested}' is not usable on this GPU "
                    f"(compute capability {cap}). ctranslate2 int8 GEMMs fail on "
                    f"Blackwell. Use --compute-type float16.")
        return requested
    return "float16" if device == "cuda" else "int8"


# ===========================================================================
# PyAV bypass (Windows Application Control)
# ===========================================================================

def stub_av_if_blocked() -> str | None:
    """faster_whisper.audio does `import av` at module scope just to decode
    audio. ffmpeg already does that job here, so if PyAV's unsigned native
    extensions get blocked by Application Control, stub the module instead
    of letting the import kill faster-whisper entirely."""
    global _AV_STUBBED
    if _AV_STUBBED or "av" in sys.modules:
        return None

    try:
        import av  # noqa: F401, PLC0415
        return None
    except Exception:  # noqa: BLE001 - policy block raises ImportError, others too
        pass

    import types

    av_mod = types.ModuleType("av")

    def _unavailable(*_a, **_kw):
        raise RuntimeError(
            "PyAV is unavailable on this machine (blocked or broken). "
            "main.py decodes audio with ffmpeg instead, so this should "
            "never be reached - it means something passed a file path to "
            "faster-whisper rather than a decoded array."
        )

    audio_mod = types.ModuleType("av.audio")
    resampler_mod = types.ModuleType("av.audio.resampler")
    fifo_mod = types.ModuleType("av.audio.fifo")
    error_mod = types.ModuleType("av.error")

    resampler_mod.AudioResampler = _unavailable
    fifo_mod.AudioFifo = _unavailable

    class InvalidDataError(Exception):
        pass

    error_mod.InvalidDataError = InvalidDataError

    audio_mod.resampler = resampler_mod
    audio_mod.fifo = fifo_mod
    av_mod.audio = audio_mod
    av_mod.error = error_mod
    av_mod.open = _unavailable
    av_mod.__version__ = "0.0.0-stub"

    sys.modules.update({
        "av": av_mod,
        "av.audio": audio_mod,
        "av.audio.resampler": resampler_mod,
        "av.audio.fifo": fifo_mod,
        "av.error": error_mod,
    })
    _AV_STUBBED = True
    return ("PyAV could not load (Application Control policy?) - stubbed it "
            "out; audio is decoded with ffmpeg instead.")


def decode_wav(path: Path) -> "object":
    """Decode any audio file to the float32 mono 16 kHz array Whisper wants,
    using ffmpeg. Replaces faster_whisper.audio.decode_audio."""
    import numpy as np  # noqa: PLC0415

    r = subprocess.run(
        ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
         "-i", str(path), "-vn", "-f", "s16le", "-ac", "1", "-ar", "16000", "-"],
        capture_output=True,
    )
    if r.returncode != 0:
        raise RuntimeError(
            "ffmpeg failed to decode audio: "
            + (r.stderr.decode("utf-8", "replace").strip().splitlines() or [""])[-1]
        )
    pcm = np.frombuffer(r.stdout, dtype=np.int16)
    return pcm.astype(np.float32) / 32768.0


# ===========================================================================
# transcription
# ===========================================================================

def has_audio(video: Path) -> bool:
    r = run([
        "ffprobe", "-v", "error", "-select_streams", "a",
        "-show_entries", "stream=codec_type", "-of", "csv=p=0", str(video),
    ])
    if r.returncode != 0:
        raise RuntimeError("ffprobe could not inspect the downloaded video: "
                           + (r.stderr or "unknown error").strip()[-500:])
    return "audio" in (r.stdout or "")


def _run_with_timeout(fn, timeout):
    """Run fn() on a daemon thread so a genuine CUDA hang can't freeze the
    whole batch. Returns (value, error)."""
    box = {}

    def target():
        try:
            box["value"] = fn()
        except BaseException as e:  # noqa: BLE001 - CUDA can raise anything
            box["error"] = e

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        return None, TimeoutError(f"GPU call did not return within {timeout}s")
    if "error" in box:
        return None, box["error"]
    return box.get("value"), None


class _HFUnauthenticatedNoticeFilter(logging.Filter):
    """Drop only HF's optional-token rate-limit notice, not real errors."""

    def filter(self, record: logging.LogRecord) -> bool:
        return "You are sending unauthenticated requests to the HF Hub" not in record.getMessage()


def ensure_model_downloaded(size: str, local_files_only: bool = False) -> str:
    """Resolve/download a Faster-Whisper model with Hugging Face progress.

    Faster-Whisper deliberately supplies a tqdm subclass that disables Hub
    download bars. Temporarily replace only that hook while its downloader
    runs, then give WhisperModel the resulting local path. Doing this before
    the CUDA timeout also prevents a slow first download being called a GPU
    hang.
    """
    local_model = Path(size).expanduser()
    if local_model.is_dir():
        required = (local_model / "config.json", local_model / "model.bin")
        if not all(path.is_file() for path in required):
            raise ValueError(
                f"local model directory is missing config.json or model.bin: {local_model}"
            )
        return str(local_model.resolve())

    cache_key = (size, local_files_only)
    if cache_key in _MODEL_PATH_CACHE:
        return _MODEL_PATH_CACHE[cache_key]

    note = stub_av_if_blocked()
    if note:
        print(f"        ({note})")

    try:
        import faster_whisper.utils as fw_utils
        from tqdm.auto import tqdm
    except ImportError as e:
        die(f"faster-whisper not importable: {e}\n"
            f"Run: {sys.executable} -m pip install -r requirements.txt")

    hf_http_log = logging.getLogger("huggingface_hub.utils._http")
    notice_filter = _HFUnauthenticatedNoticeFilter()
    hf_http_log.addFilter(notice_filter)
    hidden_tqdm = fw_utils.disabled_tqdm
    fw_utils.disabled_tqdm = tqdm
    mode = "local cache only" if local_files_only else "first use downloads model weights"
    print(f"        (checking Whisper {size}; {mode})")
    try:
        model_path = fw_utils.download_model(size, local_files_only=local_files_only)
    finally:
        fw_utils.disabled_tqdm = hidden_tqdm
        hf_http_log.removeFilter(notice_filter)

    _MODEL_PATH_CACHE[cache_key] = model_path
    return model_path


def load_model(size: str, model_path: str, device: str, compute_type: str,
               batched: bool):
    key = (size, device, compute_type, batched)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    if device == "cuda":
        prepare_cuda_runtime()

    note = stub_av_if_blocked()
    if note:
        print(f"        ({note})")

    try:
        from faster_whisper import BatchedInferencePipeline, WhisperModel
    except ImportError as e:
        die(f"faster-whisper not importable: {e}\n"
            f"Run: {sys.executable} -m pip install -r requirements.txt")

    m = WhisperModel(model_path, device=device, compute_type=compute_type)
    if batched:
        m = BatchedInferencePipeline(model=m)
    print(f"        (whisper {size} on {device}/{compute_type}"
          f"{', batched' if batched else ''})")
    _MODEL_CACHE[key] = m
    return m


# Seeding the decoder with punctuated text in the target language nudges
# Whisper into emitting punctuation and diacritic-free modern orthography.
LANG_PRIMERS = {
    "ar": "مرحباً بكم. هذا تسجيل صوتي واضح، وفيه علامات ترقيم كاملة.",
}


def build_options(args, language: str | None) -> dict:
    prompt = args.initial_prompt
    if prompt is None and language in LANG_PRIMERS:
        prompt = LANG_PRIMERS[language]

    opts = dict(
        language=language,
        beam_size=args.beam_size,
        temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        condition_on_previous_text=False,
        initial_prompt=prompt,
        vad_filter=not args.no_vad,
        vad_parameters=dict(
            threshold=args.vad_threshold,
            min_silence_duration_ms=args.vad_min_silence_ms,
            speech_pad_ms=args.vad_pad_ms,
            min_speech_duration_ms=100,
        ),
        language_detection_segments=args.lang_detect_segments,
        word_timestamps=args.word_timestamps,
        multilingual=args.multilingual,
    )
    if args.batch_size:
        opts["batch_size"] = args.batch_size
        opts.pop("condition_on_previous_text", None)
        opts.pop("multilingual", None)
    return opts


def transcribe(video: Path, dest: Path, args) -> str:
    out = dest / "transcript.txt"
    if not has_audio(video):
        out.write_text("(no audio track)\n", encoding="utf-8")
        return "(no audio track)"

    print("  [3/4] transcribing...")
    wav = dest / "audio.wav"
    audio_result = run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(video), "-vn", "-ac", "1", "-ar", "16000", str(wav),
    ])
    if audio_result.returncode != 0 or not wav.is_file():
        raise RuntimeError("ffmpeg audio extraction failed: "
                           + (audio_result.stderr or "unknown error").strip()[-500:])

    batched = bool(args.batch_size)

    def model_size(device: str) -> str:
        if device == "cpu" and args.cpu_model:
            return args.cpu_model
        return args.model

    def attempt(device: str, model_path: str):
        compute = resolve_compute_type(args.compute_type, device)
        size = model_size(device)
        model = load_model(size, model_path, device, compute, batched)
        segments, info = model.transcribe(
            decode_wav(wav), log_progress=True,
            **build_options(args, args.language),
        )
        return [f"[{s.start:6.1f}-{s.end:6.1f}] {s.text.strip()}"
                for s in segments], info, device

    global _CUDA_DEAD
    if args.device == "cpu" or _CUDA_DEAD:
        first_device = "cpu"
    elif args.device == "cuda":
        first_device = "cuda"
    else:
        first_device = "cuda" if gpu_info() else "cpu"
        print(f"        (device auto-selected {first_device})")

    first_path = ensure_model_downloaded(
        model_size(first_device), local_files_only=args.offline)
    t0 = time.perf_counter()

    if first_device == "cpu":
        lines, info, used = attempt("cpu", first_path)
    else:
        audio_seconds = max(0.0, (wav.stat().st_size - 44) / 32000)
        gpu_timeout = max(GPU_TIMEOUT_SEC, audio_seconds * 4 + 30)
        result, err = _run_with_timeout(
            lambda: attempt("cuda", first_path), gpu_timeout)
        if err is None:
            lines, info, used = result
        else:
            _MODEL_CACHE.pop((args.model, "cuda",
                              resolve_compute_type(args.compute_type, "cuda"),
                              batched), None)
            detail = str(err).strip().splitlines()
            detail = detail[0][:200] if detail else repr(err)
            kind = "hung" if isinstance(err, TimeoutError) else "failed"
            if isinstance(err, TimeoutError):
                wav.unlink(missing_ok=True)
                die(f"GPU hung: {detail}\nA timed-out CUDA worker cannot be "
                    "safely reused in this process. Restart with --force-cpu "
                    "or run 'main.py gpu-check'.")
            fallback_allowed = args.device == "auto" or args.allow_cpu_fallback
            if not fallback_allowed:
                wav.unlink(missing_ok=True)
                die(f"GPU {kind}: {detail}\n{cuda_hint()}\n\n"
                    f"Run 'main.py gpu-check' for a full diagnostic, or pass "
                    f"--allow-cpu-fallback to grind it out on the CPU anyway.")
            _CUDA_DEAD = True
            print(f"        (GPU {kind}, CPU for the rest of this run: {detail})")
            cpu_path = ensure_model_downloaded(
                model_size("cpu"), local_files_only=args.offline)
            lines, info, used = attempt("cpu", cpu_path)

    wav.unlink(missing_ok=True)
    dt = time.perf_counter() - t0

    body = "\n".join(lines) if lines else "(no speech detected)"
    lang = getattr(info, "language", "?")
    prob = getattr(info, "language_probability", 0) or 0
    used_model = model_size(used)
    if Path(used_model).is_dir():
        used_model = Path(used_model).name
    header = (f"# language={lang} prob={prob:.2f} "
              f"model={used_model} device={used} {dt:.1f}s\n\n")
    out.write_text(header + body + "\n", encoding="utf-8")
    print(f"        (lang={lang} p={prob:.2f}, {dt:.1f}s on {used})")
    return body


# ===========================================================================
# per-reel bundle.md + driver
# ===========================================================================

def write_reel_bundle(dest: Path, meta: dict, frames: list, transcript: str):
    print("  [4/4] writing bundle.md")
    tags = " ".join("#" + t for t in meta["hashtags"]) or "-"
    dur = meta.get("duration_sec")
    lines = [
        f"# Reel {meta['shortcode']}",
        "",
        f"- **Author:** @{meta.get('author') or 'unknown'}",
        f"- **URL:** {meta['url']}",
        f"- **Duration:** {dur}s" if dur else "- **Duration:** unknown",
        f"- **Likes:** {meta.get('like_count')}  |  **Comments:** {meta.get('comment_count')}",
        f"- **Posted:** {meta.get('upload_date') or 'unknown'}",
        f"- **Hashtags:** {tags}",
        "",
        "## Caption",
        "",
        "```",
        (meta.get("caption") or "(empty)").strip(),
        "```",
        "",
        "## Transcript",
        "",
        "```",
        transcript.strip(),
        "```",
        "",
        f"## Frames ({len(frames)})",
        "",
    ]
    for f in frames:
        lines.append(f"![{f.name}](frames/{f.name})")
    lines.append("")
    (dest / "bundle.md").write_text("\n".join(lines), encoding="utf-8")


def process_reel(url: str, outroot: Path, args) -> bool:
    sc = shortcode(url)
    dest = outroot / sc
    if (dest / "bundle.md").exists() and not args.force:
        print(f"[skip] {sc} (already done, --force to redo)")
        return True
    if args.force:
        (dest / "bundle.md").unlink(missing_ok=True)
        (dest / "transcript.txt").unlink(missing_ok=True)

    print(f"[reel] {sc}  <- {url}")
    try:
        video, info = download(url, dest, args.cookies)
        meta = write_meta(info, url, dest)
        frames = extract_frames(video, dest, args.scene_threshold,
                                args.max_frames, args.fallback_fps)
        text = transcribe(video, dest, args)
        write_reel_bundle(dest, meta, frames, text)
        if not args.keep_video:
            video.unlink(missing_ok=True)
        print(f"[done] {dest}  ({len(frames)} frames)\n")
        return True
    except SystemExit:
        raise
    except Exception as e:  # noqa: BLE001 - one bad reel shouldn't kill a batch
        print(f"[fail] {sc}: {e}\n", file=sys.stderr)
        return False


def gpu_check(args):
    """Report everything relevant, then actually run inference - constructing
    the model proves nothing, because the math libraries load lazily."""
    print("== GPU diagnostic ==\n")

    info = gpu_info()
    if info:
        print(f"GPU          : {info.get('name')}")
        print(f"compute cap  : {info.get('compute_cap')}")
        print(f"driver       : {info.get('driver')}")
        print(f"VRAM         : {info.get('vram')}")
    else:
        print("GPU          : nvidia-smi not found - no NVIDIA driver on PATH")
    print(f"python       : {sys.executable}")

    rep = prepare_cuda_runtime()
    print("\nsearch dirs:")
    for d in rep["dirs"]:
        print(f"  {d}")
    print("\nloaded:")
    for s in rep["loaded"] or ["  (none)"]:
        print(f"  {s}" if not s.startswith("  ") else s)
    if rep["missing"]:
        print("\nMISSING (CUDA cannot work without these):")
        for s in rep["missing"]:
            print(f"  {s}")
    if rep["optional"]:
        print("\noptional, usually harmless:")
        for s in rep["optional"]:
            print(f"  {s}")

    size = args.model
    compute = resolve_compute_type(args.compute_type, "cuda")
    model_path = ensure_model_downloaded(
        size, local_files_only=getattr(args, "offline", False))
    print(f"\nrunning real inference: {size} / float16 ... ", end="", flush=True)

    def go():
        import numpy as np
        stub_av_if_blocked()
        from faster_whisper import WhisperModel
        model = WhisperModel(model_path, device="cuda", compute_type=compute)
        silence = np.zeros(16000, dtype=np.float32)
        list(model.transcribe(silence, vad_filter=False)[0])

    t0 = time.perf_counter()
    _, err = _run_with_timeout(go, 600)
    if err is None:
        print(f"OK ({time.perf_counter() - t0:.1f}s)")
        print("\nCUDA is working. Nothing to fix.")
        sys.exit(0)

    print("FAILED")
    msg = str(err).strip()
    print(f"\nreason: {msg.splitlines()[0] if msg else repr(err)}")

    low = msg.lower()
    if "application control" in low or "smart app control" in low:
        print("\nThis is Windows Application Control blocking an unsigned DLL,")
        print("not a CUDA problem. The CUDA libraries above all loaded fine.")
        print("\nmain.py stubs PyAV out automatically, so if you are still")
        print("seeing this, the blocked DLL belongs to something else. Find out")
        print("exactly what was denied:")
        print("  Get-WinEvent -LogName Microsoft-Windows-CodeIntegrity/Operational "
              "-MaxEvents 20 |")
        print("    Where-Object Id -in 3077,3033 | Format-List TimeCreated, Message")
        print("\nIf it is ctranslate2.dll or a cuDNN/cuBLAS DLL, the policy is")
        print("blocking the inference stack itself and has to be relaxed:")
        print("  Settings > Privacy & security > Windows Security >")
        print("    App & browser control > Smart App Control")
        print("  (note: turning Smart App Control off is irreversible without")
        print("   reinstalling Windows, and if this machine is managed by your")
        print("   university or employer, the policy is deliberate - ask first)")
        sys.exit(1)

    print(cuda_hint())
    if not rep["missing"] and os.name == "nt":
        print("\nAll expected DLLs loaded, so this is not a missing-library problem.")
        print("Next things to try, in order:")
        print(f"  1. {sys.executable} -m pip install --upgrade ctranslate2")
        print("  2. update the NVIDIA driver (Blackwell needs a recent one)")
        print("  3. --compute-type float32  (rules out a float16 kernel issue)")
    sys.exit(1)


# ===========================================================================
# bundle.py -- stitching ripped reels into one markdown file
# ===========================================================================

def read_text(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def parse_transcript(raw: str):
    """Return (segments, header). Segments are {start, text}."""
    segs, header = [], {}
    for line in raw.splitlines():
        line = line.rstrip()
        if not line:
            continue
        if line.startswith("#"):
            for k, v in re.findall(r"(\w+)=([^\s]+)", line):
                header[k] = v
            continue
        m = re.match(r"\[\s*([\d.]+)-\s*([\d.]+)\]\s*(.*)", line)
        if m:
            segs.append({"start": float(m.group(1)), "end": float(m.group(2)),
                         "text": m.group(3)})
        elif segs:
            segs[-1]["text"] += " " + line.strip()
    return segs, header


def load_reel(d: Path) -> dict | None:
    segs, header = parse_transcript(read_text(d / "transcript.txt"))
    meta = {}
    if (d / "meta.json").exists():
        try:
            meta = json.loads(read_text(d / "meta.json"))
        except json.JSONDecodeError:
            pass
    text = " ".join(s["text"] for s in segs).strip()
    date = meta.get("upload_date") or ""
    return {
        "shortcode": d.name,
        "author": (meta.get("author") or "unknown").strip(),
        "url": meta.get("url") or f"https://www.instagram.com/p/{d.name}/",
        "caption": (meta.get("caption") or "").strip(),
        "hashtags": meta.get("hashtags") or [],
        "likes": meta.get("like_count"),
        "comments": meta.get("comment_count"),
        "date": f"{date[:4]}-{date[4:6]}-{date[6:8]}" if len(date) == 8 and date.isdigit() else date,
        "language": header.get("language") or "?",
        "duration": round(segs[-1]["end"], 0) if segs else None,
        "segments": segs,
        "text": text,
        "words": len(text.split()),
    }


def load_registry(outdir: Path) -> dict:
    p = outdir / "batches.json"
    if p.exists():
        try:
            return json.loads(read_text(p))
        except json.JSONDecodeError:
            pass
    return {"batches": {}}


def save_registry(outdir: Path, reg: dict) -> None:
    (outdir / "batches.json").write_text(json.dumps(reg, ensure_ascii=False, indent=1), encoding="utf-8")


def scope_for(outdir: Path, names: list[str]) -> tuple[list[str], str]:
    reg = load_registry(outdir)
    if not names:
        skip = {"bundles", "exports"}
        return sorted(
            p.name for p in outdir.iterdir()
            if p.is_dir() and p.name not in skip
            and (p / "bundle.md").is_file()
            and (p / "transcript.txt").is_file()
        ), "all reels"
    codes, missing = [], []
    for n in names:
        b = reg["batches"].get(n)
        (codes.extend(b["shortcodes"]) if b else missing.append(n))
    if missing:
        raise SystemExit(f"unknown batch(es): {', '.join(missing)}\n"
                         f"known: {', '.join(reg['batches']) or '(none yet)'}")
    seen, ordered = set(), []
    for c in codes:
        if c not in seen:
            seen.add(c)
            ordered.append(c)
    return ordered, " + ".join(names)


BUNDLE_HEADER = """# {name}

{n} Instagram reels, transcribed. Generated {when}.

Each entry below is one reel: its poster, link, engagement numbers as reported
by Instagram, and the full spoken transcript. Transcripts come from Whisper and
contain recognition errors; some reels start mid-sentence. Captions are included
where the post had one.

Nothing here has been categorised, scored, or filtered — this file is the raw
material. Instagram reports likes and comments but not views, so like counts
reflect a creator's audience size as much as a given reel's reach.

"""


def slugify(s: str) -> str:
    keep = "".join(c if c.isalnum() or c in "-_ " else "" for c in s)
    return "-".join(keep.lower().split())[:60] or "bundle"


def group_reels(reels: list[dict], by: str) -> list[tuple[str, list[dict]]]:
    """Group by something Instagram actually told us. No inference."""
    if by == "none":
        return [("", reels)]
    if by == "date":
        g = defaultdict(list)
        for r in reels:
            g[r["date"] or "no date"].append(r)
        return sorted(g.items(), reverse=True)
    g = defaultdict(list)
    for r in reels:
        g[r["author"]].append(r)
    many = sorted(((a, rs) for a, rs in g.items() if len(rs) > 1), key=lambda kv: -len(kv[1]))
    ones = [r for a, rs in g.items() if len(rs) == 1 for r in rs]
    if ones:
        many.append((f"Single reels ({len(ones)} creators)", sorted(ones, key=lambda r: r["author"].lower())))
    return many


def render_bundle(reels: list[dict], name: str, by: str, timestamps: bool) -> str:
    out = [BUNDLE_HEADER.format(name=name, n=len(reels), when=datetime.now().strftime("%Y-%m-%d %H:%M"))]
    groups = group_reels(reels, by)

    if by != "none" and len(groups) > 1:
        out.append("## Contents\n")
        for title, rs in groups:
            out.append(f"- {title} — {len(rs)} reel{'s' if len(rs) != 1 else ''}")
        out.append("\n---\n")

    for title, rs in groups:
        if title:
            out.append(f"\n# {title}\n")
        for r in rs:
            bits = [r["author"]]
            if r["date"]:
                bits.append(r["date"])
            if r["duration"]:
                bits.append(f"{r['duration']:.0f}s")
            if r["likes"] is not None:
                bits.append(f"{r['likes']:,} likes")
            if r["comments"] is not None:
                bits.append(f"{r['comments']:,} comments")
            if r["language"] not in ("?", "en"):
                bits.append(f"lang: {r['language']}")

            out.append(f"\n## {r['shortcode']}\n")
            out.append(" · ".join(bits))
            out.append(f"\n{r['url']}\n")
            if r["caption"]:
                cap = r["caption"].replace("\n", " ").strip()
                out.append(f"\n*Caption:* {cap[:600]}{'…' if len(cap) > 600 else ''}\n")
            if r["hashtags"]:
                out.append(f"\n*Hashtags:* {' '.join('#' + h for h in r['hashtags'][:20])}\n")

            out.append("\n**Transcript**\n")
            if not r["segments"]:
                out.append("\n_No speech transcribed._\n")
            elif timestamps:
                out.append("\n```")
                out.extend(f"[{s['start']:6.1f}] {s['text']}" for s in r["segments"])
                out.append("```\n")
            else:
                out.append("\n" + r["text"] + "\n")
    return "\n".join(out) + "\n"


def build(outdir: Path, batches: list[str], name: str | None, by: str,
          timestamps: bool, min_words: int) -> tuple[Path, dict]:
    codes, label = scope_for(outdir, batches)
    reels, skipped = [], 0
    for c in codes:
        d = outdir / c
        if not d.is_dir():
            continue
        r = load_reel(d)
        if r["words"] < min_words:
            skipped += 1
            continue
        reels.append(r)
    if not reels:
        raise SystemExit("nothing to bundle")

    title = name or label
    body = render_bundle(reels, title, by, timestamps)
    dest = outdir / "bundles"
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / f"{slugify(title)}.md"
    path.write_text(body, encoding="utf-8")

    stats = {"name": title, "file": path.name, "reels": len(reels), "skipped": skipped,
             "bytes": len(body.encode("utf-8")),
             "approx_tokens": int(len(body.split()) * 1.35),
             "generated": datetime.now().isoformat(timespec="seconds"),
             "grouped_by": by}
    (dest / (slugify(title) + ".json")).write_text(json.dumps(stats, ensure_ascii=False, indent=1),
                                                   encoding="utf-8")
    return path, stats


# ===========================================================================
# register_batch.py -- record a named batch from a URL list
# ===========================================================================

def cmd_register(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        prog="main.py register",
        description="Record a named batch from a URL list. The app does this "
                    "itself; you only need this on the command-line path.")
    ap.add_argument("--name", required=True)
    ap.add_argument("--urls-file", default=str(DEFAULT_URLS))
    ap.add_argument("-o", "--out", default=str(DEFAULT_OUT))
    args = ap.parse_args(argv)

    out = Path(args.out).resolve()
    text = Path(args.urls_file).read_text(encoding="utf-8", errors="replace")
    urls, _ = parse_urls(text)
    wanted = [shortcode(url) for url in urls]

    landed = [
        c for c in wanted
        if (out / c / "transcript.txt").is_file()
        and (out / c / "bundle.md").is_file()
    ]
    failed = [c for c in wanted if c not in landed]

    reg = load_registry(out)
    reg["batches"][args.name] = {
        "name": args.name,
        "created": datetime.now().isoformat(timespec="seconds"),
        "shortcodes": landed,
        "requested": len(wanted),
        "failed": failed,
    }
    save_registry(out, reg)
    print(f"batch '{args.name}': {len(landed)} reels"
          + (f", {len(failed)} failed ({', '.join(failed[:5])})" if failed else ""), file=sys.stderr)
    return 0


# ===========================================================================
# bundle.py CLI
# ===========================================================================

def cmd_bundle(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        prog="main.py bundle",
        description="Stitch ripped reels into one markdown file for an LLM. "
                    "Deliberately dumb: no scoring, no classification, no "
                    "clustering. Grouped only by facts Instagram supplied.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--out", default=str(DEFAULT_OUT))
    ap.add_argument("--batch", action="append", default=[], help="batch name (repeatable; omit for all)")
    ap.add_argument("--name", default=None, help="title and filename for the bundle")
    ap.add_argument("--by", choices=["author", "date", "none"], default="author")
    ap.add_argument("--timestamps", action="store_true", help="keep [12.4s] markers (bigger file)")
    ap.add_argument("--min-words", type=int, default=8, help="skip reels with fewer words")
    args = ap.parse_args(argv)

    outdir = Path(args.out).resolve()
    if not outdir.is_dir():
        print(f"no such directory: {outdir}", file=sys.stderr)
        return 1

    try:
        path, st = build(outdir, args.batch, args.name, args.by, args.timestamps, args.min_words)
    except SystemExit as e:
        print(str(e), file=sys.stderr)
        return 1
    print(f"{st['reels']} reels -> {path}", file=sys.stderr)
    if st["skipped"]:
        print(f"  skipped {st['skipped']} with under {args.min_words} words", file=sys.stderr)
    print(f"  {st['bytes'] // 1024} KB, roughly {st['approx_tokens'] // 1000}k tokens", file=sys.stderr)
    return 0


# ===========================================================================
# server.py -- the local app behind the paste box
# ===========================================================================

JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()


def _job_log(job: str, msg: str) -> None:
    with JOBS_LOCK:
        JOBS[job]["log"].append(msg.rstrip())
        JOBS[job]["log"] = JOBS[job]["log"][-400:]


def existing_codes() -> set[str]:
    if not OUT.is_dir():
        return set()
    return {
        p.name for p in OUT.iterdir()
        if p.is_dir() and (p / "transcript.txt").is_file()
        and (p / "bundle.md").is_file()
    }


def list_bundles() -> list[dict]:
    d = OUT / "bundles"
    if not d.is_dir():
        return []
    out = []
    for j in d.glob("*.json"):
        try:
            s = json.loads(j.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if isinstance(s, dict) and s.get("file") and (d / s["file"]).is_file():
            out.append(s)
    return sorted(out, key=lambda b: b.get("generated", ""), reverse=True)


def run_job(job_id: str, urls: list[str], name: str, opts: dict) -> None:
    try:
        OUT.mkdir(exist_ok=True)
        have = existing_codes()
        codes = [INSTAGRAM_SHORTCODE_RX.search(u).group(1) for u in urls]
        todo = [u for u, c in zip(urls, codes) if c not in have]
        skipped = len(urls) - len(todo)

        with JOBS_LOCK:
            JOBS[job_id].update(phase="ripping", total=len(todo), done=0)
        if skipped:
            _job_log(job_id, f"{skipped} reel(s) already ripped, reusing them.")

        if todo:
            listfile = OUT / f".urls-{job_id}.txt"
            listfile.write_text("\n".join(todo), encoding="utf-8")
            cmd = [sys.executable, "-u", str(HERE / "main.py"), "rip",
                   "--urls-file", str(listfile), "-o", str(OUT)]
            if opts.get("model"):
                cmd += ["--model", opts["model"]]
            device = opts.get("device", "auto")
            if device in {"auto", "cpu", "cuda"}:
                cmd += ["--device", device]
            if opts.get("language"):
                cmd += ["--language", opts["language"]]
            if opts.get("multilingual"):
                cmd += ["--multilingual"]
            if opts.get("cookies"):
                cmd += ["--cookies", opts["cookies"]]
            if opts.get("offline"):
                cmd += ["--offline"]
            _job_log(job_id, "$ " + " ".join(cmd[1:]))

            proc = subprocess.Popen(cmd, cwd=str(HERE), stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True,
                                    encoding="utf-8", errors="replace", bufsize=1)
            for line in proc.stdout:
                line = line.rstrip()
                if line:
                    _job_log(job_id, line)
                    if line.startswith("[done]"):
                        with JOBS_LOCK:
                            JOBS[job_id]["done"] = min(JOBS[job_id]["done"] + 1, len(todo))
            proc.wait()
            try:
                listfile.unlink()
            except OSError:
                pass
            if proc.returncode != 0:
                _job_log(job_id, f"ripper exited {proc.returncode}; continuing with whatever landed.")

        got = existing_codes()
        landed = [c for c in codes if c in got]
        missing = [c for c in codes if c not in got]
        if missing:
            _job_log(job_id, f"{len(missing)} reel(s) could not be ripped: "
                        f"{', '.join(missing[:8])}{'...' if len(missing) > 8 else ''}")
        if not landed:
            raise RuntimeError("nothing was ripped successfully")

        reg = load_registry(OUT)
        reg["batches"][name] = {"name": name, "created": datetime.now().isoformat(timespec="seconds"),
                                "shortcodes": landed, "requested": len(urls), "failed": missing}
        save_registry(OUT, reg)
        _job_log(job_id, f"batch '{name}': {len(landed)} reels")

        with JOBS_LOCK:
            JOBS[job_id]["phase"] = "bundling"
        path, stats = build(OUT, [name], name, opts.get("by", "author"),
                            bool(opts.get("timestamps")), int(opts.get("min_words", 8)))
        _job_log(job_id, f"wrote bundles/{path.name} — {stats['reels']} reels, "
                    f"~{stats['approx_tokens'] // 1000}k tokens")

        with JOBS_LOCK:
            JOBS[job_id].update(phase="done", batch=name, bundle=stats)
    except Exception as e:
        _job_log(job_id, "ERROR: " + str(e))
        _job_log(job_id, traceback.format_exc()[-1500:])
        with JOBS_LOCK:
            JOBS[job_id].update(phase="error", error=str(e))


class Handler(BaseHTTPRequestHandler):
    server_version = "reelkit"

    def log_message(self, fmt, *args):
        pass  # the UI is the log

    def send_json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_html_string(self, html: str):
        data = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def send_file(self, path: Path, ctype: str, download=False):
        if not path.is_file():
            return self.send_json({"error": "not found"}, 404)
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        if download:
            self.send_header("Content-Disposition", f'attachment; filename="{path.name}"')
        self.end_headers()
        self.wfile.write(data)

    def safe_under(self, base: Path, *parts: str) -> Path | None:
        p = base.joinpath(*parts).resolve()
        return p if base.resolve() in p.parents or p == base.resolve() else None

    # ----------------------------------------------------------------- GET
    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        path = unquote(u.path)

        if path in ("/", "/index.html"):
            return self.send_html_string(APP_HTML)

        if path == "/api/state":
            reg = load_registry(OUT)
            return self.send_json({
                "batches": sorted(reg["batches"].values(), key=lambda b: b.get("created", ""), reverse=True),
                "bundles": list_bundles(),
                "ripped": len(existing_codes()),
            })

        if path.startswith("/api/job/"):
            with JOBS_LOCK:
                j = JOBS.get(path.rsplit("/", 1)[-1])
            return self.send_json(j or {"error": "no such job"}, 200 if j else 404)

        if path.startswith("/bundles/"):
            p = self.safe_under(OUT / "bundles", *path[len("/bundles/"):].split("/"))
            if p is None:
                return self.send_json({"error": "bad path"}, 400)
            ctype = "application/json" if p.suffix == ".json" else "text/markdown; charset=utf-8"
            return self.send_file(p, ctype, download="dl" in q)

        return self.send_json({"error": "not found"}, 404)

    # ---------------------------------------------------------------- POST
    def do_POST(self):
        path = urlparse(self.path).path
        try:
            n = int(self.headers.get("Content-Length") or 0)
            if n > 1_000_000:
                return self.send_json({"error": "request body is too large"}, 413)
            body = json.loads(self.rfile.read(n) or "{}")
            if not isinstance(body, dict):
                return self.send_json({"error": "JSON body must be an object"}, 400)
        except (ValueError, json.JSONDecodeError):
            return self.send_json({"error": "bad json"}, 400)

        if path == "/api/rip":
            with JOBS_LOCK:
                active = next((job for job in JOBS.values()
                               if job.get("phase") in {"queued", "ripping", "bundling"}), None)
            if active:
                return self.send_json(
                    {"error": f"job {active['id']} is still running; wait for it to finish"},
                    409,
                )
            url_blob = body.get("urls", "")
            options = body.get("options") or {}
            if not isinstance(url_blob, str) or not isinstance(options, dict):
                return self.send_json({"error": "urls must be text and options an object"}, 400)
            urls, bad = parse_urls(url_blob)
            if not urls:
                return self.send_json({"error": "No Instagram reel links found in that paste.",
                                       "rejected": bad[:10]}, 400)
            raw_name = body.get("name") or ""
            if not isinstance(raw_name, str):
                return self.send_json({"error": "batch name must be text"}, 400)
            name = re.sub(r"[^\w .-]", "", raw_name.strip())[:60] \
                or datetime.now().strftime("batch-%Y%m%d-%H%M")
            reg = load_registry(OUT)
            if name in reg["batches"] and not body.get("overwrite"):
                return self.send_json({"error": f"A batch named '{name}' already exists.",
                                       "exists": True}, 409)
            job_id = uuid.uuid4().hex[:12]
            with JOBS_LOCK:
                JOBS[job_id] = {"id": job_id, "phase": "queued", "log": [], "done": 0,
                                "total": len(urls), "name": name, "rejected": bad,
                                "started": datetime.now().isoformat(timespec="seconds")}
            threading.Thread(target=run_job, args=(job_id, urls, name, options),
                             daemon=True).start()
            return self.send_json({"job": job_id, "urls": len(urls), "rejected": bad, "name": name})

        if path == "/api/bundle":
            try:
                by = body.get("by", "author")
                if by not in {"author", "date", "none"}:
                    raise ValueError("invalid grouping mode")
                name = body.get("name")
                if name is not None and not isinstance(name, str):
                    raise ValueError("bundle name must be text")
                p, stats = build(OUT, body.get("batches") or [], body.get("name"),
                                 by, bool(body.get("timestamps")),
                                 int(body.get("min_words", 8)))
            except (SystemExit, TypeError, ValueError) as e:
                return self.send_json({"error": str(e)}, 400)
            return self.send_json(stats)

        return self.send_json({"error": "not found"}, 404)


def cmd_serve(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        prog="main.py serve",
        description="Start the local reelkit app. Paste links, get one "
                    "markdown file, without watching any of the reels. "
                    "Binds to 127.0.0.1 only; nothing is uploaded anywhere.")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-open", action="store_true")
    args = ap.parse_args(argv)

    OUT.mkdir(exist_ok=True)
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"reelkit running at {url}  (ctrl-c to stop)", file=sys.stderr)
    if not args.no_open:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped", file=sys.stderr)
    return 0


# ===========================================================================
# app.html -- the paste-a-box UI, embedded
# ===========================================================================

APP_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>reelkit</title>
<style>
:root{
  --bg:#f7f7f5; --panel:#fff; --panel-2:#f0f0ed; --line:#e2e2dc;
  --ink:#1b1b19; --ink-2:#5e5e57; --ink-3:#8d8d84;
  --accent:#2f6f4f; --accent-ink:#fff; --warn:#a86a1f;
  --mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  --sans:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){
  --bg:#15151a; --panel:#1d1d23; --panel-2:#25252c; --line:#32323b;
  --ink:#e9e9e5; --ink-2:#a6a69e; --ink-3:#75756d;
  --accent:#4f9d73; --accent-ink:#0e0e11; --warn:#d29a4a;
}}
:root[data-theme=dark]{
  --bg:#15151a; --panel:#1d1d23; --panel-2:#25252c; --line:#32323b;
  --ink:#e9e9e5; --ink-2:#a6a69e; --ink-3:#75756d;
  --accent:#4f9d73; --accent-ink:#0e0e11; --warn:#d29a4a;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 var(--sans);-webkit-font-smoothing:antialiased}
button,input,select,textarea{font:inherit;color:inherit}
a{color:var(--accent)}
header{position:sticky;top:0;z-index:20;background:var(--panel);border-bottom:1px solid var(--line);padding:10px 18px;display:flex;gap:14px;align-items:center;flex-wrap:wrap}
h1{font-size:15px;margin:0;letter-spacing:.02em;font-weight:700}
h1 span{color:var(--ink-3);font-weight:400}
.spacer{flex:1}
.meta{color:var(--ink-3);font-size:12.5px;font-family:var(--mono)}
.tabs{display:flex;gap:2px;background:var(--panel-2);padding:3px;border-radius:8px}
.tabs button{border:0;background:transparent;padding:5px 14px;border-radius:6px;cursor:pointer;color:var(--ink-2);font-size:13.5px}
.tabs button[aria-selected=true]{background:var(--panel);color:var(--ink);font-weight:600;box-shadow:0 1px 2px rgba(0,0,0,.07)}
main{padding:20px 18px 60px;max-width:940px;margin:0 auto}
.card{background:var(--panel);border:1px solid var(--line);border-radius:11px;padding:16px 18px;margin-bottom:14px}
.card h2{margin:0 0 4px;font-size:16px}
.card h2 + p{margin:0 0 14px;color:var(--ink-2);font-size:13.5px}
textarea{width:100%;min-height:160px;padding:11px 13px;border:1px solid var(--line);border-radius:9px;background:var(--bg);font-family:var(--mono);font-size:12.5px;line-height:1.6;resize:vertical}
.row{display:flex;gap:9px;flex-wrap:wrap;align-items:center;margin-top:11px}
input[type=text],select{padding:7px 10px;border:1px solid var(--line);border-radius:8px;background:var(--bg)}
input[type=text]{min-width:190px}
label.chk{display:flex;align-items:center;gap:6px;font-size:13px;color:var(--ink-2);cursor:pointer}
button.go{background:var(--accent);color:var(--accent-ink);border:0;border-radius:8px;padding:9px 20px;font-weight:600;cursor:pointer}
button.go:disabled{opacity:.5;cursor:default}
button.ghost,a.ghost{border:1px solid var(--line);background:var(--panel);border-radius:8px;padding:7px 13px;cursor:pointer;font-size:13px;color:var(--ink-2);text-decoration:none;display:inline-block}
button.ghost:hover,a.ghost:hover{border-color:var(--ink-3);color:var(--ink)}
.log{background:var(--bg);border:1px solid var(--line);border-radius:9px;padding:11px 13px;font-family:var(--mono);font-size:11.5px;line-height:1.65;max-height:280px;overflow:auto;white-space:pre-wrap;margin-top:12px;color:var(--ink-2)}
.pbar{height:5px;background:var(--panel-2);border-radius:3px;overflow:hidden;margin-top:11px}
.pbar i{display:block;height:100%;background:var(--accent);transition:width .3s}
.warn{background:color-mix(in srgb,var(--warn) 15%,transparent);border:1px solid color-mix(in srgb,var(--warn) 42%,transparent);border-radius:9px;padding:11px 14px;font-size:13px;color:var(--ink-2);line-height:1.55;margin-bottom:14px}
.warn b{color:var(--ink)}
table{width:100%;border-collapse:collapse;font-size:13.5px}
th,td{text-align:left;padding:8px 10px;border-bottom:1px solid var(--line)}
th{color:var(--ink-2);font-size:11.5px;text-transform:uppercase;letter-spacing:.04em;font-weight:600}
td.num{font-family:var(--mono);text-align:right}
.q{color:var(--ink-3)}
.empty{text-align:center;color:var(--ink-3);padding:50px 20px}
code{font-family:var(--mono);font-size:12.5px;background:var(--panel-2);padding:1px 5px;border-radius:4px}
.big{display:flex;align-items:baseline;gap:9px;margin:4px 0 12px}
.big b{font-size:26px;font-family:var(--mono);font-weight:600}
</style>
</head>
<body>
<header>
  <h1>reelkit <span>· local</span></h1>
  <span class="meta" id="stat"></span>
  <span class="spacer"></span>
  <div class="tabs" role="tablist">
    <button role="tab" data-v="ingest" aria-selected="true">Ingest</button>
    <button role="tab" data-v="bundles" aria-selected="false">Bundles</button>
  </div>
  <button class="ghost" id="theme">◐</button>
</header>
<main>
  <div id="v-ingest"></div>
  <div id="v-bundles" hidden></div>
</main>

<script>
const $ = s => document.querySelector(s);
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const kb = b => b > 1e6 ? (b/1e6).toFixed(1)+" MB" : Math.round(b/1024)+" KB";
let STATE = {batches:[], bundles:[], ripped:0}, POLL = null;

async function api(path, opts){
  const r = await fetch(path, opts);
  const j = await r.json().catch(()=>({error:"bad response"}));
  if(!r.ok) throw new Error(j.error || ("HTTP "+r.status));
  return j;
}
async function loadState(){
  STATE = await api("/api/state");
  $("#stat").textContent = `${STATE.ripped} reels ripped · ${STATE.bundles.length} bundles`;
}

const groupOpts = id => `
  <select id="${id}-by" title="How to order the file. All three are facts from Instagram, not guesses.">
    <option value="author">group by creator</option>
    <option value="date">group by date</option>
    <option value="none">no grouping</option>
  </select>
  <label class="chk"><input type="checkbox" id="${id}-ts"> keep timestamps</label>
  <label class="chk">skip under
    <input type="text" id="${id}-min" value="8" style="width:52px;min-width:0"> words</label>`;

const readOpts = id => ({
  by: $("#"+id+"-by").value,
  timestamps: $("#"+id+"-ts").checked,
  min_words: parseInt($("#"+id+"-min").value) || 0,
});

function renderIngest(){
  const b = STATE.batches;
  $("#v-ingest").innerHTML = `
  <div class="card">
    <h2>Paste reel links</h2>
    <p>Full URLs, bare shortcodes, or messy text with links in it. Reels already ripped are
       reused, so overlapping batches cost nothing. You get one markdown file out.</p>
    <textarea id="urls" placeholder="https://www.instagram.com/reel/ABC123.../&#10;https://www.instagram.com/p/XYZ789.../"></textarea>
    <div class="row">
      <input type="text" id="bname" placeholder="batch name (optional)">
      <select id="device" title="Auto uses NVIDIA CUDA when available and otherwise CPU.">
        <option value="auto">auto device</option>
        <option value="cuda">GPU only (CUDA)</option>
        <option value="cpu">CPU only</option>
      </select>
      <select id="model" title="Larger models are more accurate but slower and use more memory.">
        <option value="tiny">tiny</option>
        <option value="base">base</option>
        <option value="small">small</option>
        <option value="medium" selected>medium</option>
        <option value="large-v3-turbo">large-v3-turbo</option>
        <option value="large-v3">large-v3</option>
      </select>
      <select id="lang">
        <option value="">auto-detect language</option>
        <option value="en">English</option>
        <option value="ar">Arabic</option>
        <option value="__multi">Arabic + English mixed</option>
      </select>
      <select id="cookies">
        <option value="">no cookies</option>
        <option value="firefox">Firefox cookies</option>
        <option value="chrome">Chrome cookies</option>
        <option value="edge">Edge cookies</option>
      </select>
    </div>
    <div class="row">${groupOpts("i")}<span class="spacer"></span>
      <button class="go" id="run">Rip &amp; bundle</button>
    </div>
    <div id="prog"></div>
  </div>

  ${STATE.ripped ? `<div class="card">
    <h2>Bundle what's already ripped</h2>
    <p>${STATE.ripped} reels are on disk from earlier runs${b.length?"":" but aren't in any named batch yet"}.
       Build a file from them without downloading anything again.</p>
    <div class="row">
      <input type="text" id="allname" placeholder="file name (optional)">
      ${groupOpts("a")}
      <button class="ghost" id="all">Bundle all ${STATE.ripped}</button>
    </div>
    <div id="allmsg" class="meta" style="margin-top:9px"></div>
  </div>` : ""}

  ${b.length ? `<div class="card">
    <h2>Batches</h2>
    <p>Tick any number and combine them into one file.</p>
    <table><thead><tr><th></th><th>batch</th><th>reels</th><th>created</th></tr></thead>
    <tbody>${b.map(x=>`<tr>
      <td><input type="checkbox" class="bsel" value="${esc(x.name)}"></td>
      <td><b>${esc(x.name)}</b>${x.failed && x.failed.length?` <span class="q">(${x.failed.length} failed)</span>`:""}</td>
      <td class="num">${x.shortcodes.length}</td>
      <td class="q">${esc((x.created||"").replace("T"," ").slice(0,16))}</td>
    </tr>`).join("")}</tbody></table>
    <div class="row">
      <input type="text" id="rollname" placeholder="file name">
      ${groupOpts("r")}
      <button class="ghost" id="roll">Bundle selected</button>
    </div>
    <div id="rollmsg" class="meta" style="margin-top:9px"></div>
  </div>` : ""}`;

  $("#run").onclick = startRip;
  const all = $("#all"), roll = $("#roll");
  if(all) all.onclick = () => makeBundle([], $("#allname").value.trim() || "all reels", readOpts("a"), "#allmsg");
  if(roll) roll.onclick = () => {
    const sel = [...document.querySelectorAll(".bsel:checked")].map(c=>c.value);
    if(!sel.length) return $("#rollmsg").textContent = "Pick at least one batch.";
    makeBundle(sel, $("#rollname").value.trim() || sel.join(" + "), readOpts("r"), "#rollmsg");
  };
}

async function makeBundle(batches, name, opts, msgSel){
  const msg = $(msgSel);
  msg.textContent = "building…";
  try{
    const s = await api("/api/bundle", {method:"POST", headers:{"Content-Type":"application/json"},
      body: JSON.stringify({batches, name, ...opts})});
    await loadState();
    msg.innerHTML = `<b>${esc(s.file)}</b> — ${s.reels} reels, ${kb(s.bytes)}, ~${(s.approx_tokens/1000).toFixed(0)}k tokens
      ${s.skipped?`<span class="q">(${s.skipped} skipped as near-silent)</span>`:""}
      &nbsp;<a class="ghost" href="/bundles/${encodeURIComponent(s.file)}?dl=1">Download</a>`;
  }catch(e){ msg.textContent = "Failed: " + e.message; }
}

async function startRip(){
  const urls = $("#urls").value.trim();
  if(!urls) return;
  const lang = $("#lang").value;
  const opts = {
    cookies: $("#cookies").value || null,
    device: $("#device").value,
    model: $("#model").value,
    ...readOpts("i")
  };
  if(lang === "__multi") opts.multilingual = true; else if(lang) opts.language = lang;
  $("#run").disabled = true;
  $("#prog").innerHTML = `<div class="log">starting…</div>`;
  try{
    const j = await api("/api/rip", {method:"POST", headers:{"Content-Type":"application/json"},
      body: JSON.stringify({urls, name: $("#bname").value, options: opts})});
    if(j.rejected && j.rejected.length)
      $("#prog").insertAdjacentHTML("afterbegin",
        `<div class="warn">Ignored ${j.rejected.length} line(s) with no reel link: <code>${j.rejected.map(esc).join("</code> <code>")}</code></div>`);
    pollJob(j.job);
  }catch(e){
    $("#prog").innerHTML = `<div class="warn"><b>Could not start.</b> ${esc(e.message)}</div>`;
    $("#run").disabled = false;
  }
}

function pollJob(id){
  clearInterval(POLL);
  POLL = setInterval(async ()=>{
    let j; try{ j = await api("/api/job/"+id); }catch(e){ return; }
    const pct = j.total ? Math.round(100*j.done/j.total) : 0;
    const head = j.phase === "ripping" ? `ripping ${j.done}/${j.total}`
               : j.phase === "bundling" ? "writing the markdown file"
               : j.phase;
    $("#prog").innerHTML =
      `<div class="pbar"><i style="width:${j.phase==="done"?100:pct}%"></i></div>
       <div class="meta" style="margin-top:7px">${esc(head)}</div>
       <div class="log" id="jlog">${esc((j.log||[]).join("\n"))}</div>`;
    const el = $("#jlog"); if(el) el.scrollTop = el.scrollHeight;

    if(j.phase === "done" || j.phase === "error"){
      clearInterval(POLL);
      $("#run").disabled = false;
      await loadState();
      if(j.phase === "done" && j.bundle){
        const s = j.bundle;
        $("#prog").insertAdjacentHTML("beforeend",
          `<div class="row"><a class="go" style="text-decoration:none"
             href="/bundles/${encodeURIComponent(s.file)}?dl=1">Download ${esc(s.file)}</a>
           <span class="meta">${s.reels} reels · ${kb(s.bytes)} · ~${(s.approx_tokens/1000).toFixed(0)}k tokens</span></div>`);
      }
    }
  }, 900);
}

function renderBundles(){
  const el = $("#v-bundles");
  if(!STATE.bundles.length)
    return el.innerHTML = `<div class="empty">No bundles yet — make one on the Ingest tab.</div>`;
  el.innerHTML = `<div class="warn">Drop a bundle into a Claude Project, OpenWebUI, or any model with
    room for it. The file explains itself at the top — nothing in it has been categorised or scored,
    so ask the model to do that.</div>` + STATE.bundles.map(s=>`
    <div class="card">
      <h2>${esc(s.name)}</h2>
      <div class="big"><b>${(s.approx_tokens/1000).toFixed(0)}k</b><span class="q">approx tokens</span>
        <span class="spacer"></span>
        <span class="meta">${s.reels} reels · ${kb(s.bytes)} · grouped by ${esc(s.grouped_by)}
        ${s.skipped?` · ${s.skipped} skipped`:""}</span></div>
      <div class="row">
        <a class="ghost" href="/bundles/${encodeURIComponent(s.file)}?dl=1">Download</a>
        <a class="ghost" href="/bundles/${encodeURIComponent(s.file)}" target="_blank">View</a>
        <span class="meta">${esc((s.generated||"").replace("T"," ").slice(0,16))} · <code>${esc(s.file)}</code></span>
      </div>
    </div>`).join("");
}

function setView(v){
  document.querySelectorAll("[role=tab]").forEach(b=>b.setAttribute("aria-selected", b.dataset.v===v));
  $("#v-ingest").hidden = v!=="ingest";
  $("#v-bundles").hidden = v!=="bundles";
  const fns = {ingest:renderIngest, bundles:renderBundles};
  try{ fns[v](); }
  catch(e){
    console.error(e);
    $("#v-"+v).innerHTML = `<div class="warn"><b>Something broke while rendering the ${v} tab.</b>
      <code>${esc(e.message)}</code><br>This is a bug in the page, not a problem with your data —
      the server is running. Reload; the browser console has the stack trace.</div>`;
  }
}
document.querySelectorAll("[role=tab]").forEach(b=> b.onclick = ()=>setView(b.dataset.v));
$("#theme").onclick = ()=>{
  const c = document.documentElement.getAttribute("data-theme");
  const n = c==="dark" ? "light" : c==="light" ? "" : "dark";
  n ? document.documentElement.setAttribute("data-theme",n) : document.documentElement.removeAttribute("data-theme");
};

loadState().then(()=>setView("ingest")).catch(e=>{
  document.querySelector("main").innerHTML =
    `<div class="warn"><b>Cannot reach the local server.</b> <code>${esc(e.message)}</code><br>
     This page has to be served by main.py — opening the file directly will not work.
     Start it with <code>python main.py serve</code>, then go to
     <a href="http://127.0.0.1:8765/">127.0.0.1:8765</a>.</div>`;
});
</script>
</body>
</html>
"""


# ===========================================================================
# rip.py CLI (reel_ripper.py's original argparse surface, as `main.py rip`)
# ===========================================================================

def cmd_rip(argv: list[str]) -> int:
    p = argparse.ArgumentParser(
        prog="main.py rip",
        description="Instagram reels -> frames + transcript",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("urls", nargs="*", help="reel URLs (default: read urls.txt)")
    p.add_argument("--urls-file", default=str(DEFAULT_URLS))
    p.add_argument("-o", "--out", default=str(DEFAULT_OUT))
    p.add_argument("--cookies", default=None, metavar="BROWSER",
                   help="chrome|firefox|edge - for private/login-walled reels")

    g = p.add_argument_group("transcription")
    g.add_argument("--model", default="medium",
                   help="tiny|base|small|medium|large-v3|large-v3-turbo")
    g.add_argument("--cpu-model", default=None,
                   help="optional model override used after a GPU-to-CPU fallback")
    g.add_argument("--offline", action="store_true",
                   help="use only a local model directory or files already in the HF cache")
    g.add_argument("--language", default=None,
                   help="force a language code, e.g. en or ar (default: autodetect)")
    g.add_argument("--multilingual", action="store_true",
                   help="allow the language to change mid-clip (Arabic/English mixing)")
    g.add_argument("--initial-prompt", default=None,
                   help="seed text; auto-seeded for Arabic to force punctuation")
    g.add_argument("--beam-size", type=int, default=5)
    g.add_argument("--batch-size", type=int, default=0,
                   help="0 = sequential (exact timestamps); 8-16 = batched, faster")
    g.add_argument("--lang-detect-segments", type=int, default=4,
                   help="windows to vote over when autodetecting language")
    g.add_argument("--word-timestamps", action="store_true")
    g.add_argument("--no-vad", action="store_true",
                   help="disable voice-activity filtering entirely")
    g.add_argument("--vad-threshold", type=float, default=0.4)
    g.add_argument("--vad-min-silence-ms", type=int, default=400)
    g.add_argument("--vad-pad-ms", type=int, default=250)

    d = p.add_argument_group("device")
    d.add_argument("--compute-type", default="auto",
                   choices=["auto", "float16", "float32", "int8", "int8_float16", "bfloat16"],
                   help="auto = float16 on GPU, int8 on CPU")
    d.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto",
                   help="auto chooses CUDA when available and otherwise uses CPU")
    forced = d.add_mutually_exclusive_group()
    forced.add_argument("--force-cpu", "--forcecpu", "--cpu", dest="force_cpu",
                        action="store_true", help="always use CPU (--cpu kept as an alias)")
    forced.add_argument("--force-gpu", "--forcegpu", dest="force_gpu",
                        action="store_true", help="require CUDA; fail instead of silently using CPU")
    d.add_argument("--allow-cpu-fallback", action="store_true",
                   help="allow an explicit CUDA run to fall back if CUDA fails")
    d.add_argument("--gpu-check", action="store_true",
                   help="diagnose the CUDA stack and exit")

    f = p.add_argument_group("frames")
    f.add_argument("--max-frames", type=int, default=12)
    f.add_argument("--scene-threshold", type=float, default=0.30)
    f.add_argument("--fallback-fps", type=float, default=0.5,
                   help="frames/sec when scene detection finds nothing")

    p.add_argument("--keep-video", action="store_true")
    p.add_argument("--force", action="store_true", help="reprocess already-done reels")
    args = p.parse_args(argv)

    if (args.force_cpu or args.force_gpu) and args.device != "auto":
        die("use either --device or a --force-cpu/--force-gpu switch, not both")
    if args.force_gpu and args.allow_cpu_fallback:
        die("--force-gpu cannot be combined with --allow-cpu-fallback")
    if args.force_cpu:
        args.device = "cpu"
    elif args.force_gpu:
        args.device = "cuda"

    if args.gpu_check:
        gpu_check(args)

    if args.batch_size and args.no_vad:
        die("--batch-size and --no-vad are incompatible: batched inference "
            "chunks the audio using VAD. Drop one of them.")

    need("ffmpeg", "Install ffmpeg and reopen your shell.")
    need("ffprobe", "Comes with ffmpeg - reopen your shell after installing it.")

    raw_urls = "\n".join(args.urls)
    if not raw_urls:
        uf = Path(args.urls_file)
        if not uf.exists():
            die(f"no URLs given and {uf} does not exist")
        raw_urls = uf.read_text(encoding="utf-8", errors="replace")
    urls, rejected = parse_urls(raw_urls)
    if rejected:
        print(f"[warning] ignored {len(rejected)} item(s) that were not valid "
              "Instagram reel/post links", file=sys.stderr)
    if not urls:
        die("URL list is empty")

    outroot = Path(args.out)
    outroot.mkdir(parents=True, exist_ok=True)

    print(f"{len(urls)} reel(s) -> {outroot}\n")
    ok = 0
    try:
        for url in urls:
            ok += process_reel(url, outroot, args)
    except KeyboardInterrupt:
        print("\n[cancelled] stopped by user; partial downloads can resume next time.",
              file=sys.stderr)
        return 130
    print(f"== {ok}/{len(urls)} succeeded ==")
    print(f"Point Claude at: {outroot}")
    return 0 if ok == len(urls) else 1


def cmd_gpu_check(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="main.py gpu-check",
                                 description="Full CUDA diagnostic, standalone.")
    ap.add_argument("--model", default="tiny")
    ap.add_argument("--compute-type", default="auto",
                    choices=["auto", "float16", "float32", "int8", "int8_float16", "bfloat16"])
    args = ap.parse_args(argv)
    gpu_check(args)  # exits internally
    return 0


# ===========================================================================
# dispatch
# ===========================================================================

def main() -> int:
    argv = sys.argv[1:]
    known = {"rip", "gpu-check", "serve", "bundle", "register"}

    if not argv:
        argv = ["rip"]
    elif argv[0] not in known and argv[0] not in ("-h", "--help"):
        # bare flags/URLs default to ripping, so `main.py --gpu-check` and
        # `main.py URL1 URL2` behave like the old reel_ripper.py did.
        argv = ["rip", *argv]

    cmd, rest = argv[0], argv[1:]

    if cmd in ("-h", "--help"):
        print(__doc__)
        return 0
    if cmd == "rip":
        return cmd_rip(rest)
    if cmd == "gpu-check":
        return cmd_gpu_check(rest)
    if cmd == "serve":
        return cmd_serve(rest)
    if cmd == "bundle":
        return cmd_bundle(rest)
    if cmd == "register":
        return cmd_register(rest)

    print(f"unknown command: {cmd}", file=sys.stderr)
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
