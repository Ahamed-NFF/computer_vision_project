# Touchless Writing System — Full Documentation

An air‑writing application for the **Image Processing & Computer Vision** course. You write and draw in the air with hand gestures tracked through your webcam; the strokes are rendered onto a digital canvas. You can then convert the handwriting to text with OCR (Google Gemini), preview the text live on screen, dictate text with your voice, and save pages as images or `.txt` files.

---

## Table of Contents
1. [Features](#features)
2. [How It Works (Architecture)](#how-it-works-architecture)
3. [Project Structure](#project-structure)
4. [Installation](#installation)
5. [Configuration (.env / API key)](#configuration-env--api-key)
6. [Running the App](#running-the-app)
7. [Hand Gestures](#hand-gestures)
8. [Keyboard Shortcuts](#keyboard-shortcuts)
9. [Feature Guides](#feature-guides)
10. [OCR Preprocessing Pipeline](#ocr-preprocessing-pipeline)
11. [Evaluation Harness](#evaluation-harness)
12. [Output Files & Folders](#output-files--folders)
13. [Troubleshooting](#troubleshooting)
14. [Tech Stack](#tech-stack)

---

## Features

| Feature | Trigger | Needs Internet | Needs API Key |
|---|---|---|---|
| ✍️ Gesture drawing (air‑writing) | Hand gestures | No | No |
| 🎨 8 colors + eraser | `Select` gesture | No | No |
| 📄 Multi‑page canvas | `N` / `P` / `→` | No | No |
| 🖼️ Save page(s) as image | `S` / `A` | No | No |
| 🔤 OCR to text file | `T` | Yes | Yes (Gemini) |
| 🔴 Live text overlay (on‑screen OCR) | `L` | Yes | Yes (Gemini) |
| 🎤 Voice‑to‑text dictation | `V` | Yes | No (Google Speech) |
| 🧼 OCR preprocessing toggle | `X` | No | No |

---

## How It Works (Architecture)

```
 Webcam frame
      │
      ▼
 MediaPipe Hands ──► 21 hand landmarks
      │
      ▼
 detect_gesture()  ──► move / draw / erase / select / clear / idle
      │                       (stabilized over 3 frames to stop flicker)
      ▼
 Canvas (NumPy 1280×720 image)  ◄── voice text / drawing strokes
      │
      ├──► Display (OpenCV window: webcam + canvas overlay + HUD)
      │
      └──► Save PNG ──► [preprocess.py] ──► [image_to_text.py → Gemini] ──► text
```

- **Hand tracking:** `mediapipe.solutions.hands` (single hand, legacy Solutions API).
- **Gesture logic:** each finger's up/down state is computed from landmark positions; the combination of raised fingers maps to a gesture ([main.py:119](main.py#L119)).
- **Rendering:** the canvas is a plain NumPy array; strokes are drawn with OpenCV line segments between smoothed cursor positions.
- **OCR path:** the current page is written to `temp/`, optionally cleaned by the classical pipeline in [preprocess.py](preprocess.py), then sent to Gemini in [image_to_text.py](image_to_text.py).

---

## Project Structure

```
computer_vision_project/
├── main.py               # The application (entry point) — TouchlessWritingSystem
├── image_to_text.py      # Gemini OCR wrapper (image → text)
├── preprocess.py         # Classical CV preprocessing pipeline for OCR
├── evaluate.py           # OCR accuracy/latency evaluation harness (CER/WER)
├── requirements.txt      # Pinned dependencies (Python 3.11 recommended)
├── .env                  # (you create this) GEMINI_API_KEY=...   [gitignored]
├── extracted_text/       # OCR output .txt files (auto-created)
├── saved_notes/          # Saved canvas images (auto-created)
├── temp/                 # Temp images during OCR (auto-created)
├── DOCUMENTATION.md      # ← this file
├── OCR_FEATURE_README.md # OCR (T key) details
├── LIVE_TEXT_MODE.md     # Live overlay (L key) details
└── VOICE_TO_TEXT.md      # Voice dictation (V key) details
```

Entry point: [main.py:1032](main.py#L1032) — `TouchlessWritingSystem().run()`.

---

## Installation

> **Important:** Use **Python 3.11** (Homebrew: `/opt/homebrew/bin/python3.11`). The pinned `mediapipe==0.10.21` has **no wheels for Python 3.13/3.14**, and it pins `numpy<2`, so OpenCV must match. Don't use the bare `python3` if it resolves to 3.13+.

```bash
cd "computer_vision_project"

# 1. Create the virtual environment on Python 3.11
/opt/homebrew/bin/python3.11 -m venv .venv

# 2. Activate it
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate           # Windows (PowerShell: .venv\Scripts\Activate.ps1)

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Dependency notes (from `requirements.txt`)
- `mediapipe==0.10.21` — still ships the legacy `mp.solutions.hands` API the app uses; pins `numpy<2`.
- `opencv-contrib-python==4.11.0.86` — a superset of `opencv-python` that provides `cv2`. **Do not also install `opencv-python`** — having both corrupts the shared `cv2/` folder.
- `numpy==1.26.4` — kept `<2` for mediapipe compatibility.
- `google-genai`, `python-dotenv`, `Pillow` — OCR (Gemini) stack.
- `SpeechRecognition`, `PyAudio` — voice‑to‑text. On macOS, `PyAudio` builds against PortAudio; if the build fails run `brew install portaudio` first.

---

## Configuration (.env / API key)

OCR features (`T` and `L`) call Google Gemini and need an API key. Drawing, voice, and image saving work **without** it.

1. Get a key from https://aistudio.google.com/apikey
2. Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_key_here
```

The Gemini client is created **lazily** ([image_to_text.py:11](image_to_text.py#L11)), so the app launches fine without a key — it only errors if you actually press `T` or `L`. `.env` is gitignored so the key is never committed.

> The `google-genai` SDK reads `GEMINI_API_KEY` (it also accepts `GOOGLE_API_KEY`). Use `GEMINI_API_KEY`.

---

## Running the App

```bash
source .venv/bin/activate
python main.py
```

A window opens showing your webcam feed with the drawing canvas and a heads‑up display. Raise one hand and use the [gestures](#hand-gestures) below.

- **macOS camera permission:** the first run prompts for camera access — allow it for your terminal / VS Code, otherwise the feed stays black.
- **macOS microphone permission:** required for voice mode (`V`).

---

## Hand Gestures

Detected from which fingers are raised ([main.py:119](main.py#L119)). Gestures are stabilized over 3 frames to avoid flicker.

| Gesture | Fingers raised | Action |
|---|---|---|
| **Move** | Index only | Move the cursor without drawing |
| **Draw** | Index + Thumb | Draw on the canvas |
| **Erase** | Index + Middle | Erase strokes |
| **Select** | Index + Pinky | Pick a color from the palette / UI |
| **Clear** | All five fingers | Clear the current page |
| **Idle** | None / other | No action |

---

## Keyboard Shortcuts

From the key handler at [main.py:986](main.py#L986):

| Key | Action |
|---|---|
| `N` | New page |
| `P` | Previous page |
| `→` (Right arrow) | Next page |
| `↑` (Up arrow) | Scroll live text up |
| `↓` (Down arrow) | Scroll live text down |
| `S` | Save current page as image |
| `A` | Save all pages as images |
| `T` | OCR current page → save `.txt` file |
| `L` | Toggle live text overlay (on‑screen OCR) |
| `V` | Voice mode: 1st press enables, 2nd press records |
| `ESC` | Exit voice mode |
| `C` | Clear current page |
| `X` | Toggle OCR preprocessing on/off |
| `Q` | Quit |

---

## Feature Guides

### 🔤 OCR to File (`T`)
Saves the current canvas to `temp/`, sends it to Gemini, and writes the result to `extracted_text/page_<n>_<timestamp>.txt` (with a date/header). The text is also printed to the console. See [OCR_FEATURE_README.md](OCR_FEATURE_README.md).

### 🔴 Live Text Mode (`L`)
Same OCR, but the extracted text is drawn as a **semi‑transparent overlay** on the page instead of saved to a file — useful for quick verification. Toggle with `L`; scroll long text with `↑`/`↓`. No files are created. See [LIVE_TEXT_MODE.md](LIVE_TEXT_MODE.md).

### 🎤 Voice to Text (`V`)
Dictate text onto the canvas using Google Speech Recognition. Press `V` to enable voice mode, `V` again to start listening, speak, and the recognized text is rendered (centered, word‑wrapped) on the canvas. Runs on a background thread so the UI never freezes. `ESC` exits voice mode. Records up to 15 s with a 10 s no‑speech timeout. See [VOICE_TO_TEXT.md](VOICE_TO_TEXT.md).

### 🧼 Preprocessing Toggle (`X`)
Turns the classical CV preprocessing pipeline on/off before OCR, so you can compare OCR accuracy with and without it (see next section).

---

## OCR Preprocessing Pipeline

[preprocess.py](preprocess.py) implements a configurable classical pipeline applied before OCR:

```
grayscale → CLAHE contrast normalization → adaptive thresholding
          → morphological clean (close + open) → crop to ink region
          → invert back to black-on-white
```

- **Grayscale** — reduce color artifacts.
- **CLAHE** — local contrast equalization, robust to uneven lighting.
- **Adaptive threshold** — clean binary image across varying illumination (`blockSize=31`, `C=10`).
- **Morphology** — close reconnects broken strokes, open removes speckle.
- **Crop** — bounding box around the ink so OCR focuses on content.

Every stage is individually toggleable via `preprocess_image(...)` arguments, and the whole pipeline can be bypassed (`enable=False`) for the "without preprocessing" evaluation arm. Self‑test: `python preprocess.py`.

---

## Evaluation Harness

[evaluate.py](evaluate.py) measures **Character Error Rate (CER)**, **Word Error Rate (WER)**, and OCR **latency**, with and without preprocessing — no external metric libraries (Levenshtein distance is implemented inline).

```bash
# 1. Create eval_data/ next to the script
# 2. For each test page add a matching image + ground-truth text:
#      eval_data/sample01.png   (saved canvas)
#      eval_data/sample01.txt   (exact text you wrote)
# 3. Run:
python evaluate.py
# 4. Results:
#      eval_results/results.csv   (per-image numbers)
#      eval_results/summary.txt   (averages, ready for the report)
```

---

## Output Files & Folders

| Folder | Contents | Created by |
|---|---|---|
| `extracted_text/` | OCR `.txt` files (timestamped, with header) | `T` key |
| `saved_notes/` | Saved canvas PNG images | `S` / `A` keys |
| `temp/` | Temporary images during OCR processing | OCR pipeline |
| `eval_results/` | CER/WER/latency reports | `evaluate.py` |

`.env`, `.venv/`, `__pycache__/`, and `*.pyc` are gitignored.

---

## Troubleshooting

**`Could not find a version that satisfies the requirement mediapipe==0.10.21`**
Your venv is on Python 3.13/3.14, which has no wheels for that version. Rebuild on Python 3.11:
```bash
rm -rf .venv
/opt/homebrew/bin/python3.11 -m venv .venv
source .venv/bin/activate && pip install -r requirements.txt
```

**Black/blank webcam window** — Grant camera permission (macOS: System Settings → Privacy & Security → Camera) to your terminal/VS Code. The app uses camera index `0` ([main.py:33](main.py#L33)); change it if you have multiple cameras.

**`Gemini API key not configured`** when pressing `T`/`L` — Add `GEMINI_API_KEY` to `.env` (see [Configuration](#configuration-env--api-key)).

**`cv2` import errors / segfaults** — You likely have both `opencv-python` and `opencv-contrib-python` installed. Keep only `opencv-contrib-python`:
```bash
pip uninstall -y opencv-python opencv-contrib-python
pip install opencv-contrib-python==4.11.0.86
```

**`PyAudio` install fails (macOS)** — Install PortAudio first: `brew install portaudio`, then reinstall PyAudio.

**Voice mode: "No microphone found" / timeout** — Grant microphone permission, move closer to the mic, reduce background noise. Voice recognition requires an internet connection.

**OCR text is wrong** — Write larger and clearer, use dark colors on the white canvas, ensure good lighting, and try the preprocessing toggle (`X`).

---

## Tech Stack

- **Python 3.11**
- **OpenCV** (`opencv-contrib-python` 4.11) — capture, canvas rendering, UI
- **MediaPipe** 0.10.21 — hand landmark tracking
- **NumPy** 1.26 — canvas array math
- **Google Gemini** (`google-genai`, model `gemini-2.5-flash`) — OCR
- **SpeechRecognition + PyAudio** — voice‑to‑text (Google Speech API)
- **python-dotenv** — API key loading · **Pillow** — image handling

---

*Touchless Writing System — write in the air, get digital text.*
