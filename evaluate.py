"""
evaluate.py
Evaluation harness for the Touchless Writing System OCR stage.

Measures Character Error Rate (CER), Word Error Rate (WER) and OCR latency,
WITH and WITHOUT the classical preprocessing pipeline, as promised in the
proposal (Methodology section 6, Dataset section 5, Expected Outcomes section 6).

------------------------------------------------------------------------------
HOW TO USE
------------------------------------------------------------------------------
1. Create a folder  eval_data/  next to this script.
2. For each test page, put two files with the SAME base name:
        eval_data/sample01.png      <- the saved canvas / handwriting image
        eval_data/sample01.txt      <- the exact ground-truth text you wrote
   Repeat for sample02, sample03, ... (5-10 pages is enough for the report).
3. Run:
        python evaluate.py
4. Results are written to:
        eval_results/results.csv        (per-image numbers)
        eval_results/summary.txt        (averages, ready to paste into report)

No external metric libraries are required - CER/WER use a standard
Levenshtein edit distance implemented below.
"""

import os
import csv
import time
import glob

from preprocess import preprocess_image
from image_to_text import image_to_text


# ---------- metric helpers ----------

def _levenshtein(a, b):
    """Edit distance between two sequences (lists or strings)."""
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j] + 1,      # deletion
                         cur[j - 1] + 1,   # insertion
                         prev[j - 1] + cost)  # substitution
        prev = cur
    return prev[m]


def _norm(text):
    """Light normalization: strip, collapse whitespace, lowercase."""
    return " ".join(text.strip().lower().split())


def cer(reference, hypothesis):
    ref = _norm(reference)
    hyp = _norm(hypothesis)
    if len(ref) == 0:
        return 0.0 if len(hyp) == 0 else 1.0
    return _levenshtein(ref, hyp) / len(ref)


def wer(reference, hypothesis):
    ref = _norm(reference).split()
    hyp = _norm(hypothesis).split()
    if len(ref) == 0:
        return 0.0 if len(hyp) == 0 else 1.0
    return _levenshtein(ref, hyp) / len(ref)


# ---------- evaluation run ----------

def run_one(image_path, gt_text, enable_pre):
    """Run OCR on one image, return (text, latency_seconds)."""
    pre_path = "eval_results/_pre_tmp.png"
    ocr_input = preprocess_image(image_path, out_path=pre_path, enable=enable_pre)
    t0 = time.perf_counter()
    text = image_to_text(ocr_input)
    latency = time.perf_counter() - t0
    return (text or ""), latency


def main():
    os.makedirs("eval_results", exist_ok=True)
    images = sorted(glob.glob("eval_data/*.png") + glob.glob("eval_data/*.jpg"))
    if not images:
        print("No images found in eval_data/. Add sampleNN.png + sampleNN.txt pairs.")
        return

    rows = []
    for img in images:
        base = os.path.splitext(img)[0]
        gt_file = base + ".txt"
        if not os.path.exists(gt_file):
            print(f"  ! skipping {img} (no matching .txt ground truth)")
            continue
        with open(gt_file, "r", encoding="utf-8") as f:
            gt = f.read()

        name = os.path.basename(base)
        print(f"Evaluating {name} ...")

        for enable in (False, True):
            mode = "with_pre" if enable else "no_pre"
            try:
                text, lat = run_one(img, gt, enable)
                row = {
                    "sample": name,
                    "mode": mode,
                    "cer": round(cer(gt, text), 4),
                    "wer": round(wer(gt, text), 4),
                    "latency_s": round(lat, 3),
                }
            except Exception as e:
                row = {"sample": name, "mode": mode,
                       "cer": "", "wer": "", "latency_s": "",
                       "error": str(e)}
            rows.append(row)
            print(f"    {mode:9s} CER={row['cer']} WER={row['wer']} "
                  f"latency={row['latency_s']}s")

    # write per-image csv
    with open("eval_results/results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["sample", "mode", "cer", "wer",
                                          "latency_s", "error"])
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in
                        ["sample", "mode", "cer", "wer", "latency_s", "error"]})

    # summary averages per mode
    def avg(mode, key):
        vals = [r[key] for r in rows
                if r["mode"] == mode and isinstance(r.get(key), (int, float))]
        return round(sum(vals) / len(vals), 4) if vals else None

    lines = ["OCR Evaluation Summary", "=" * 40, ""]
    for mode in ("no_pre", "with_pre"):
        label = "WITHOUT preprocessing" if mode == "no_pre" else "WITH preprocessing"
        lines += [
            label,
            f"  mean CER     : {avg(mode, 'cer')}",
            f"  mean WER     : {avg(mode, 'wer')}",
            f"  mean latency : {avg(mode, 'latency_s')} s",
            "",
        ]
    summary = "\n".join(lines)
    with open("eval_results/summary.txt", "w", encoding="utf-8") as f:
        f.write(summary)

    print("\n" + summary)
    print("Saved: eval_results/results.csv  and  eval_results/summary.txt")


if __name__ == "__main__":
    main()
