# reel-ripper

Turn a pile of Instagram reel links into one markdown file an LLM can actually
read — keyframes, timestamped transcript, caption, and engagement numbers, per
reel. No AI is used to rip; the only model involved is the one *you* hand the
output to.

Everything runs locally. One Python file, no server-side component, nothing
uploaded anywhere.

## Why this exists

You've got 80 tabs of reels you meant to watch. You don't want to watch them —
you want to know what's in them. This rips each reel into a transcript and a
handful of scene keyframes, then optionally stitches a batch of them into a
single markdown file you can paste into Claude, ChatGPT, or any local model.

**It does not categorize, score, or judge anything.** An earlier version of
this tool tried to detect "sales pitches" and cluster topics with regexes. It
was bad at it — it invented topics out of noise, missed half the real hooks,
and mistook plain numbers for something meaningful. Pattern matching can't
read. The model you paste the bundle into can, so the tool gets out of its
way and gives it the raw material instead:

> Group these reels by what they're actually teaching. Separate the ones with
> concrete technique from the ones selling a course. For each group, list
> what's worth knowing and which reels to ignore.

## Features

- **Rips** Instagram reels (and posts/IGTV) into frames + a Whisper transcript
  + caption/engagement metadata, via `yt-dlp` + `ffmpeg` + `faster-whisper`.
- **Bundles** any batch of ripped reels into one markdown file, grouped by
  creator, upload date, or paste order — grouping options are limited to
  facts Instagram actually reported, on purpose.
- **Local web UI** (`main.py serve`) — paste links in a browser, watch it rip,
  download the file. No terminal needed after the first setup.
- **Command-line pipeline** (`main.py rip` / `bundle` / `register`) for
  scripted or unattended runs.
- **Real GPU support.** `faster-whisper` doesn't declare its own CUDA
  dependencies, so most setups silently fall back to the CPU and never notice.
  This script finds, preloads, and verifies the CUDA/cuDNN libraries itself,
  and refuses to start a run that would silently be slow — see
  [GPU notes](#gpu-notes) below.
- **Skips reels you've already ripped.** Overlapping batches cost nothing.
- **One file.** `main.py` is the entire tool — easy to read, easy to vendor,
  easy to audit before you run it.

## Requirements

- **Python 3.10+**
- **ffmpeg** and **ffprobe** on `PATH`
- An **NVIDIA GPU** if you want anything above the `medium` Whisper model to
  run at a usable speed (CPU works fine for `tiny`/`base`/`small`/`medium`,
  just slower)

On Windows, use the standard 64-bit CPython installation available through
the `py` launcher (for example, `py -3.14`). MSYS2/MinGW Python is not
supported because packages such as NumPy and CTranslate2 may not provide
compatible binary wheels for it.

## Install

Download or clone this repository, open a terminal in its folder, then run:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

pip install -r requirements.txt          # CPU only
# or, with an NVIDIA GPU:
pip install -r requirements-gpu.txt
```

**ffmpeg:**

| OS | Command |
|---|---|
| Windows | `winget install --id Gyan.FFmpeg -e` (then reopen your shell) |
| macOS | `brew install ffmpeg` |
| Linux | `sudo apt install ffmpeg` (or your distro's equivalent) |

Verify everything's in place:

```bash
python main.py gpu-check
```

This is a full diagnostic — driver, compute capability, every CUDA/cuDNN
library it needs, and a real (not simulated) transcription run. If you're
CPU-only, skip it; ripping will just use the CPU automatically.

> The tool was built and tested on Windows. It's plain Python with no
> PowerShell dependency, so CPU mode should work unmodified on macOS/Linux.
> The automatic CUDA DLL preloading in `gpu-check` is Windows-specific; on
> Linux, a normal system CUDA + cuDNN install (or a conda environment with
> them) should be picked up by `ctranslate2` without extra steps.

## Quick start

**1. The app (easiest — no flags to remember):**

```bash
python main.py serve
```

Opens `http://127.0.0.1:8765`. Paste reel links, hit **Rip & bundle**,
download the markdown file. Reels you've already ripped are reused
automatically.

**2. The command line:**

```bash
echo "https://www.instagram.com/reel/ABC123/" > urls.txt
python main.py rip
python main.py register --name my-batch
python main.py bundle --batch my-batch
```

Either way you end up with `out/bundles/<name>.md` — one file, ready to paste
into an LLM.

## Commands

```
python main.py                        rip urls.txt (the default)
python main.py URL [URL ...]          rip specific reels, ad hoc
python main.py rip [options]          same, with the full flag set below
python main.py gpu-check              diagnose the CUDA stack and exit
python main.py serve                  start the local web app
python main.py bundle [options]       stitch ripped reels into one .md file
python main.py register --name NAME   record a batch from urls.txt by hand
```

Run `python main.py <command> --help` for the full flag list on any of them.

### `rip` — turn URLs into transcripts + frames

```bash
python main.py rip                              # reads urls.txt
python main.py rip https://instagram.com/reel/x # ad hoc
python main.py rip --language ar                # skip autodetect for Arabic
python main.py rip --multilingual               # Arabic/English code-switching
python main.py rip --cookies firefox            # login-walled or private reels
python main.py rip --force-cpu --model tiny     # require CPU
python main.py rip --force-gpu --model medium   # require NVIDIA CUDA
python main.py rip --force                      # reprocess already-done reels
```

### Choosing CPU or GPU

The default is `--device auto`: use NVIDIA CUDA when it is available and fall
back to CPU when it is not. You can make the choice strict:

| Command | Behavior |
|---|---|
| `--force-cpu` | Always use CPU. `--forcecpu` and the older `--cpu` spelling are aliases. |
| `--force-gpu` | Require CUDA. Exit with a diagnostic if it cannot start; never silently use CPU. |
| `--device cuda --allow-cpu-fallback` | Prefer CUDA, but continue on CPU if CUDA fails. |

`--model` now means the same thing on both devices. `--cpu-model` is only an
optional override for a GPU-to-CPU fallback; it no longer silently replaces
your requested model.

The first use of each `--model` size downloads its weights from Hugging Face
once, then reuses the local cache. A progress bar is shown while this happens;
an interrupted download resumes on the next run. Rough download sizes are:

| Model | Download |
|---|---:|
| `tiny` | ~75 MB |
| `base` | ~145 MB |
| `small` | ~465 MB |
| `medium` | ~1.5 GB |
| `large-v3` | ~3.1 GB |
| `large-v3-turbo` | ~1.6 GB |

No Hugging Face account or API token is required for these public models.
Anonymous access merely has lower rate limits. If you already have a token,
set it in the standard `HF_TOKEN` environment variable; do not put tokens in
this repository.

For a network-free run, either use a previously cached model with `--offline`,
or pass a local CTranslate2 model directory:

```bash
python main.py rip --force-cpu --offline --model tiny
python main.py rip --force-cpu --model path/to/faster-whisper-tiny
```

Produces, per reel:

```
out/<shortcode>/
    meta.json        caption, author, likes, comments, hashtags, dates
    frames/f0001.jpg  scene-change keyframes
    transcript.txt    timestamped speech
    bundle.md         a self-contained per-reel bundle (frames not included)
```

### `serve` — the paste-a-box app

```bash
python main.py serve                 # http://127.0.0.1:8765
python main.py serve --port 9000
python main.py serve --no-open       # don't launch a browser automatically
```

Binds to `127.0.0.1` only. Batches are named and remembered across sessions;
the **Bundles** tab lists every file you've built with a one-click download.
The ingest screen includes explicit device and model selectors. Its live log
shows download and transcription progress plus the reason for each failed reel.

### `bundle` — stitch ripped reels into one markdown file

```bash
python main.py bundle                                   # everything, by author
python main.py bundle --batch my-batch
python main.py bundle --batch jan --batch feb --name combined
python main.py bundle --by date                          # group by upload date
python main.py bundle --by none                           # paste order, no grouping
python main.py bundle --timestamps                        # keep [12.4s] markers
python main.py bundle --min-words 20                       # skip near-silent reels
```

The only grouping options are facts Instagram actually reported:

| `--by` | What it does |
|---|---|
| `author` (default) | Grouped by creator, most prolific first, single-reel creators gathered at the end. |
| `date` | Grouped by upload date, newest first. |
| `none` | No grouping — the order you pasted them in. |

Roughly 175 reels of transcript lands around 50k tokens; `bundle` prints the
approximate token count for whatever you build.

### `register` — record a batch without the app

```bash
python main.py register --name jan-hustle --urls-file urls.txt
```

Only shortcodes that actually produced a transcript get added to the batch,
so a batch never claims reels that failed to rip. The app does this itself;
you only need it on the pure command-line path.

## Flags worth knowing (`rip`)

| Flag | Why |
|---|---|
| `--language ar` | Biggest single accuracy win on Arabic. Always use it when you know the language. |
| `--multilingual` | Lets the language switch mid-clip — for Arabic/English code-switching. |
| `--force-cpu` | Require CPU, even when CUDA is installed. |
| `--force-gpu` | Require CUDA and fail clearly if it is unavailable. |
| `--offline` | Use a local model or the existing Hugging Face cache without network access. |
| `--model large-v3-turbo` | Faster decode, small accuracy cost, when you're ripping a big batch. |
| `--batch-size 16` | Batched inference. Meaningful on long clips or big batches; timestamps get slightly coarser. |
| `--compute-type float32` | Diagnostic — rules out a float16 kernel problem. |
| `--word-timestamps` | Per-word timing, for cutting clips later. |
| `--no-vad` | If voice-activity detection is eating quiet speech. |
| `--max-frames 20` | More visual coverage on fast-cut reels. |
| `--fallback-fps 1` | Denser frame sampling for static/talking-head reels. |
| `--allow-cpu-fallback` | Permit an explicitly requested CUDA run to continue on CPU. Auto mode already falls back. |
| `--force` | Reprocess reels already in `out/` (skipped by default — safe to rerun on a growing `urls.txt`). |
| `--keep-video` | Keep the downloaded mp4 (off by default so `out/` doesn't balloon). |

## GPU notes

`faster-whisper` declares **no CUDA dependencies**. It only depends on
`ctranslate2`, whose wheel dynamically links `cublas64_12.dll` and ships a
`cudnn64_9.dll` that in turn needs four more cuDNN DLLs. None of that is
installed by default, so a fresh environment's CUDA backend can never
initialize — and naive scripts catch that failure and silently transcribe on
the CPU instead, which just looks like a slow GPU.

`main.py` instead:

1. Locates the CUDA libraries inside `site-packages/nvidia/*/bin` (installed
   via `requirements-gpu.txt`).
2. Preloads each one by absolute path, in dependency order, before
   `ctranslate2` is imported — registering the directory alone isn't
   reliable, since `ctranslate2.dll` is pulled in by Python's own extension
   loader.
3. Honors the device policy you chose: `--force-gpu` stops with a diagnostic,
   `--force-cpu` never touches CUDA, and the default `auto` mode falls back to
   CPU with an explicit message.

### Blackwell (RTX 50-series, sm_120)

- Needs **cuBLAS ≥ 12.8** and **cuDNN ≥ 9.7** — older builds have no `sm_120`
  kernels, and cuBLAS doesn't JIT around that.
- **int8 on CUDA doesn't work** on these cards (`CUBLAS_STATUS_NOT_SUPPORTED`
  in ctranslate2's int8 path). `--compute-type` defaults to `float16` on GPU,
  and requesting `int8` on a Blackwell card is refused up front.
- `large-v3` at float16 is ~3 GB of VRAM — trivial on a 12 GB card; raise
  `--batch-size` freely.

### Windows Application Control / Smart App Control

`faster_whisper.audio` imports PyAV purely to decode audio files. If
Application Control blocks PyAV's unsigned native extensions, the import
fails and takes faster-whisper down with it — before CUDA is ever reached.
`main.py` doesn't need PyAV (ffmpeg already does that job), so it stubs the
module out automatically and decodes audio itself. If `gpu-check` still
reports an Application Control block afterward, it prints the exact
`Get-WinEvent` command to find out which DLL was actually denied.

## Troubleshooting

**`gpu-check` says a DLL is missing.** Run the pip command it prints — that's
the whole fix.

**`gpu-check` loads every DLL but inference still fails.** In order: `pip
install --upgrade ctranslate2`, then update your NVIDIA driver, then try
`--compute-type float32`.

**`--cookies chrome` fails with a decryption error.** Chrome periodically
tightens cookie encryption. Use `--cookies firefox`, or log in once on
Firefox and point at that.

**yt-dlp errors on a URL that works in the browser.** Instagram changes its
frontend constantly. `pip install --upgrade yt-dlp` fixes this more often
than not — try that before debugging anything else.

**Only some reels in a server batch work.** Read the per-reel log in the app.
Public reels can still be unavailable because they were deleted, made private,
restricted by region/age, or blocked by Instagram's anonymous rate limit. Pick
the browser you are logged into from the cookie selector and retry the failed
links. The tool retries transient downloads, but it cannot bypass access rules.

**Transcription looks frozen after the model loads.** Recent Faster-Whisper
versions expose transcription progress and this tool enables it. If no
percentage appears, upgrade the environment with `python -m pip install -U -r
requirements.txt`.

**Rate limiting on big batches.** Instagram throttles aggressive downloading.
Keep batches modest and spread them out; a logged-in cookie session
(`--cookies firefox`) tolerates more than an anonymous one.

## How it's organized

Everything lives in `main.py` — one file, no imports from the rest of this
repo. Internally it's five sections doing what used to be five separate
scripts:

| Section | Does what `...` used to do |
|---|---|
| ripping (`cmd_rip`) | download, extract frames, transcribe |
| CUDA/PyAV handling | make the GPU actually work, or fail loudly |
| bundling (`cmd_bundle`, `build`) | stitch ripped reels into one markdown file |
| batch registry (`cmd_register`) | name and remember a set of ripped reels |
| local server (`cmd_serve`, `APP_HTML`) | the paste-a-box web UI |

Reading the file top to bottom follows that same order.

## Tests

The fast test suite does not download a reel or model:

```bash
python -m unittest discover -s tests -v
```

Before a release, also test one real public reel on every device mode you claim
to support:

```bash
python main.py rip --force-cpu --model tiny URL
python main.py gpu-check
python main.py rip --force-gpu --model tiny URL
python main.py serve --no-open
```

## Publish this repository on GitHub

This repository intentionally excludes `out/`, `.venv/`, `urls.txt`, Python
caches, downloaded models, browser cookies, and tokens. Do not override those
entries in `.gitignore`.

1. Create a new empty repository on GitHub. Do not add another README, license,
   or `.gitignore`; they are already included here.
2. Open a terminal in this folder and run the following, replacing the example
   remote with your GitHub username and repository name:

```bash
git init
git add .
git commit -m "Initial public release"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/reel-ripper.git
git push -u origin main
```

3. On GitHub, check that `out/`, `.venv/`, and `urls.txt` are absent. Then use
   **Releases → Draft a new release**, create tag `v1.0.0`, and summarize what
   was tested. Do not promise that every Instagram URL will work—Instagram
   controls availability and changes its extractor-facing behavior.

## Privacy

Nothing is uploaded anywhere. The web UI binds to `127.0.0.1` only. The only
network calls this tool makes are to Instagram (to download the reel you gave
it a link to) and, on first run, to Hugging Face to download the Whisper
model weights.

Use the tool only for content you are authorized to download and process.
Respect copyright, privacy, Instagram's terms, and the laws that apply to you.

## License

MIT — see [LICENSE](LICENSE).
