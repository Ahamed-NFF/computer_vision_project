# OCR Evaluation Guide — Touchless Writing System

This document explains how to run the **quantitative OCR evaluation** with
[evaluate.py](evaluate.py). It measures **Character Error Rate (CER)**,
**Word Error Rate (WER)** and **OCR latency**, both **with** and **without** the
classical preprocessing pipeline, so you can report objective accuracy numbers
and an A/B comparison.

This is the script behind **§6 of [VALIDATION.md](VALIDATION.md)**.

---

## 1. What it does

For every image in `eval_data/`, `evaluate.py`:

1. Loads the matching ground-truth `.txt`.
2. Runs OCR **twice** — once with preprocessing **off** (`no_pre`) and once with
   it **on** (`with_pre`) — via [preprocess.py](preprocess.py) →
   [image_to_text.py](image_to_text.py).
3. Computes CER and WER against the ground truth, and times each OCR call.
4. Writes per-image numbers and per-mode averages to `eval_results/`.

The A/B design is the key result: **preprocessing should reduce (or at least not
worsen) CER/WER** compared with the raw image.

---

## 2. Pre-conditions

- [ ] Python **3.11** virtual environment activated.
- [ ] `pip install -r requirements.txt` completed (no extra metric libraries are
      needed — CER/WER use a built-in Levenshtein implementation).
- [ ] `.env` present with a valid `OPENAI_API_KEY` (OCR calls the OpenAI API).
- [ ] Internet connection available (each OCR call is a network round-trip).

> Run `python validate.py` first to confirm the environment is sound before
> spending API calls on the evaluation.

---

## 3. Prepare the dataset

Create an `eval_data/` folder next to the scripts and add **matched pairs** that
share the same base name — the image and its exact ground-truth text:

```
eval_data/sample01.png   ← saved canvas / handwriting image
eval_data/sample01.txt   ← exact ground-truth text you wrote
eval_data/sample02.png
eval_data/sample02.txt
...
```

Tips:
- `.png` and `.jpg` images are both picked up.
- Generate images straight from the app: draw a page, press **`S`**, and copy the
  PNG from `saved_notes/` into `eval_data/` (rename to `sampleNN.png`).
- The `.txt` must match what you wrote **exactly** — it is the reference the
  metrics are scored against.
- An image with **no** matching `.txt` is skipped (a warning is printed).

Checklist:
- [ ] At least **5** image / ground-truth pairs created (5–10 is enough for a report).
- [ ] Each ground-truth `.txt` matches its handwriting exactly.

---

## 4. Run the evaluation

```bash
source .venv/bin/activate
python evaluate.py
```

You'll see per-sample progress in the console, e.g.:

```
Evaluating sample01 ...
    no_pre    CER=0.18 WER=0.30 latency=2.41s
    with_pre  CER=0.07 WER=0.12 latency=2.55s
```

If `eval_data/` is empty or missing, the script prints a notice and exits without
error — add pairs and re-run.

---

## 5. Outputs

| File | Contents |
|---|---|
| `eval_results/results.csv` | One row per image **per mode** — `sample, mode, cer, wer, latency_s, error` |
| `eval_results/summary.txt` | Mean CER / WER / latency for `no_pre` vs `with_pre` — paste into the report |

If a sample errors during OCR (e.g. a transient API failure), its CER/WER/latency
cells are left blank and the reason is recorded in the `error` column; that row is
excluded from the averages.

---

## 6. Metric definitions

- **CER** = character-level Levenshtein edit distance ÷ reference character count.
  Lower is better; `0.0` = perfect.
- **WER** = word-level Levenshtein edit distance ÷ reference word count. Lower is better.
- **Latency** = wall-clock seconds for one OCR call (includes the network
  round-trip to OpenAI; preprocessing time is **not** counted in latency).

Before scoring, both reference and hypothesis are lightly normalized: trimmed,
whitespace collapsed, and lowercased. So casing and spacing differences are not
penalized — only the actual characters/words.

---

## 7. Acceptance criteria (suggested targets)

| Metric | Target | Result | Pass |
|---|---|---|---|
| Mean CER (with preprocessing) | ≤ 0.15 | ____ | ☐ |
| Mean WER (with preprocessing) | ≤ 0.25 | ____ | ☐ |
| Preprocessing improves CER vs. raw | CER_pre < CER_raw | ____ vs ____ | ☐ |
| Mean latency | ≤ 5 s / page | ____ | ☐ |

> Targets are indicative — adjust to your dataset and the report's stated goals.
> The headline result is the **A/B comparison**: `with_pre` CER/WER should be no
> worse than `no_pre`.

---

## 8. Interpreting the result

- **`with_pre` clearly lower than `no_pre`** → preprocessing helps; report the
  delta as evidence the classical pipeline improves recognition.
- **Roughly equal** → the OCR model already handles clean canvas images well;
  preprocessing is still justified for noisy/real-world capture.
- **`with_pre` worse** → the pipeline is over-cleaning (e.g. morphology erasing
  thin strokes, crop clipping content). Tune parameters in
  [preprocess.py](preprocess.py) — `blockSize`/`C` in `adaptive_binarize`, the
  kernel/iterations in `morphological_clean`, or `pad` in `crop_ink_region` — and
  re-run.

---

## 9. Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `No images found in eval_data/` | Folder empty or wrong location — add `sampleNN.png` + `.txt` pairs next to `evaluate.py`. |
| `! skipping <img> (no matching .txt)` | The image has no same-base-name `.txt`. Add it. |
| All rows show an `error` about the API key | `.env` missing or `OPENAI_API_KEY` invalid — see [image_to_text.py](image_to_text.py). |
| Latencies very high / timeouts | Slow or no internet; OCR is a remote call. |
| CER/WER unexpectedly high on clean text | Ground-truth `.txt` doesn't match what was actually written. |

---

*See [VALIDATION.md](VALIDATION.md) for the full validation plan and
[DOCUMENTATION.md](DOCUMENTATION.md) for feature and setup details.*
