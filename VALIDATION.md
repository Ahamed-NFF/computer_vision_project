# Validation & Testing — Touchless Writing System

This document describes how to **validate that the system works correctly**: environment setup, smoke tests, per‑feature manual validation, and the quantitative OCR accuracy evaluation (CER / WER / latency).

Use the checkboxes as a sign‑off sheet before a demo or submission.

---

## 1. Validation Scope

| Layer | What we validate | Method |
|---|---|---|
| Environment | Correct Python + dependencies installed | Smoke test (§3) |
| Core CV | Webcam capture + hand tracking + gestures | Manual (§4) |
| Drawing | Draw / erase / color / multi‑page / save | Manual (§4) |
| OCR | Handwriting → text accuracy | Quantitative (§6) + manual (§5) |
| Preprocessing | Pipeline improves OCR vs. raw | Quantitative A/B (§6) |
| Voice | Speech → text on canvas | Manual (§5) |

---

## 2. Pre‑conditions

- [ ] Python **3.11** virtual environment created and activated.
- [ ] `pip install -r requirements.txt` completed without errors.
- [ ] Webcam connected; OS camera permission granted to the terminal / IDE.
- [ ] Microphone available + permission granted (for voice validation only).
- [ ] `.env` present with a valid `GEMINI_API_KEY` (for OCR validation only).
- [ ] Internet connection available (for OCR and voice validation).

---

## 3. Environment Smoke Test

Confirms the critical libraries import and the pinned versions are correct **before** launching the GUI.

```bash
source .venv/bin/activate
python -c "import cv2, numpy, mediapipe as mp; \
mp.solutions.hands; \
print('cv2', cv2.__version__, '| numpy', numpy.__version__, '| mediapipe', mp.__version__)"
```

**Expected output:**
```
cv2 4.11.0 | numpy 1.26.4 | mediapipe 0.10.21
```

Pass criteria:
- [ ] No `ImportError` / `ModuleNotFoundError`.
- [ ] `numpy` is **1.26.x** (must be `<2` for mediapipe).
- [ ] `mediapipe.solutions.hands` resolves (legacy Solutions API present).
- [ ] Only `opencv-contrib-python` is installed (run `pip list | grep -i opencv` — it must **not** also list `opencv-python`).

### Preprocessing module self‑test
```bash
python preprocess.py
```
- [ ] Prints `Preprocessing self-test OK -> temp/_selftest_out.png`.
- [ ] `temp/_selftest_out.png` exists and shows clean black "Hello OCR" text on white.

---

## 4. Core & Drawing Validation (manual)

Launch the app: `python main.py`

| # | Test | Steps | Expected | Pass |
|---|---|---|---|---|
| 4.1 | Webcam feed | Launch app | Live mirrored webcam window opens with HUD | ☐ |
| 4.2 | Hand detection | Raise one hand | Hand landmarks/skeleton drawn on the hand | ☐ |
| 4.3 | Move | Index finger only | Cursor follows fingertip, **no** ink drawn | ☐ |
| 4.4 | Draw | Index + Thumb, move hand | Continuous stroke appears on canvas | ☐ |
| 4.5 | Erase | Index + Middle over a stroke | Stroke is removed under the eraser | ☐ |
| 4.6 | Select / color | Index + Pinky over a palette color | Active color changes; new strokes use it | ☐ |
| 4.7 | Clear gesture | All five fingers | Current page clears | ☐ |
| 4.8 | Stability | Hold a gesture steady | No rapid flicker between gestures (3‑frame stabilization) | ☐ |
| 4.9 | New page (`N`) | Press `N` | Blank new page; page counter increments | ☐ |
| 4.10 | Navigate (`P`/`→`) | Press `P` then `→` | Moves to previous / next page, content intact | ☐ |
| 4.11 | Save image (`S`) | Draw, press `S` | PNG written to `saved_notes/page_<n>_<timestamp>.png` | ☐ |
| 4.12 | Save all (`A`) | Multiple pages, press `A` | One PNG per page in `saved_notes/` | ☐ |
| 4.13 | Clear key (`C`) | Press `C` | Current page clears | ☐ |
| 4.14 | Quit (`Q`) | Press `Q` | Window closes, process exits cleanly | ☐ |

---

## 5. OCR & Voice Validation (manual, needs key/mic + internet)

| # | Test | Steps | Expected | Pass |
|---|---|---|---|---|
| 5.1 | OCR to file (`T`) | Write a clear word, press `T` | `.txt` saved in `extracted_text/`; text printed to console | ☐ |
| 5.2 | OCR accuracy | Compare file text to what you wrote | Text matches the handwriting | ☐ |
| 5.3 | Live text (`L`) | Write, press `L` | Semi‑transparent overlay shows recognized text on the page | ☐ |
| 5.4 | Live scroll | Long text, press `↑`/`↓` | Overlay scrolls; line counter updates | ☐ |
| 5.5 | Live toggle off | Press `L` again | Overlay disappears, drawing intact | ☐ |
| 5.6 | Preprocess toggle (`X`) | Press `X`, then `T` | OCR runs on the cleaned image (compare result) | ☐ |
| 5.7 | Missing key handling | Remove `.env`, launch, press `T` | App still launches; `T` shows a clear "API key not configured" error (no crash) | ☐ |
| 5.8 | Voice enable (`V`) | Press `V` | Green "Voice Mode" indicator appears | ☐ |
| 5.9 | Voice record (`V`) | Press `V` again, speak | Red "LISTENING" indicator; recognized text rendered, centered + wrapped, on canvas | ☐ |
| 5.10 | Voice exit (`ESC`) | Press `ESC` | Voice mode indicator disappears | ☐ |
| 5.11 | UI responsiveness | During voice processing | Window does not freeze (recognition runs on background thread) | ☐ |

---

## 6. Quantitative OCR Validation (CER / WER / latency)

[evaluate.py](evaluate.py) measures **Character Error Rate**, **Word Error Rate**, and **OCR latency**, **with and without** the preprocessing pipeline. This is the objective accuracy validation for the report.

### 6.1 Prepare the dataset
Create `eval_data/` and add matched pairs (same base name) for 5–10 pages:

```
eval_data/sample01.png   ← saved canvas / handwriting image
eval_data/sample01.txt   ← exact ground-truth text you wrote
eval_data/sample02.png
eval_data/sample02.txt
...
```

- [ ] At least 5 image/ground‑truth pairs created.
- [ ] Ground‑truth `.txt` matches the handwriting exactly (this is the reference).

### 6.2 Run the evaluation
```bash
python evaluate.py
```

Outputs:
- `eval_results/results.csv` — per‑image CER, WER, latency.
- `eval_results/summary.txt` — averages (paste into the report).

### 6.3 Metric definitions
- **CER** = edit distance (characters) ÷ reference character count. Lower is better; `0.0` = perfect.
- **WER** = edit distance (words) ÷ reference word count. Lower is better.
- **Latency** = seconds per OCR call (includes network round‑trip to Gemini).

### 6.4 Acceptance criteria (suggested targets)
| Metric | Target | Result | Pass |
|---|---|---|---|
| Mean CER (with preprocessing) | ≤ 0.15 | ____ | ☐ |
| Mean WER (with preprocessing) | ≤ 0.25 | ____ | ☐ |
| Preprocessing improves CER vs. raw | CER_pre < CER_raw | ____ vs ____ | ☐ |
| Mean latency | ≤ 5 s / page | ____ | ☐ |

> Targets are indicative — adjust to your dataset and report's stated goals. The key validation is the **A/B comparison**: preprocessing should reduce (or at least not worsen) CER/WER.

---

## 7. Negative / Robustness Checks

| # | Condition | Expected behavior | Pass |
|---|---|---|---|
| 7.1 | No hand in frame | Gesture = idle; no stray strokes | ☐ |
| 7.2 | No `.env` / no API key | App runs; OCR keys show graceful error, no crash | ☐ |
| 7.3 | No internet, press `T` | Clear error in status/console, app keeps running | ☐ |
| 7.4 | Voice: no speech | Times out (~10 s) with a "no speech" message, no hang | ☐ |
| 7.5 | OCR on blank page | Empty/short result, no crash; temp files cleaned up | ☐ |
| 7.6 | Camera index busy/missing | App reports the issue rather than silently hanging | ☐ |

---

## 8. Validation Sign‑off

| Section | Status | Notes |
|---|---|---|
| 3. Environment smoke test | ☐ Pass / ☐ Fail | |
| 4. Core & drawing | ☐ Pass / ☐ Fail | |
| 5. OCR & voice | ☐ Pass / ☐ Fail | |
| 6. Quantitative OCR | ☐ Pass / ☐ Fail | |
| 7. Robustness | ☐ Pass / ☐ Fail | |

**Validated by:** ______________   **Date:** ____________   **Build / commit:** ____________

---

*See [DOCUMENTATION.md](DOCUMENTATION.md) for full feature and setup details.*
