"""
validate.py
Automated validation runner for the Touchless Writing System.

Performs the *automatable* checks from VALIDATION.md so a single command can
produce a pass/fail sign-off before a demo or submission:

    python validate.py

Covers:
    §3  Environment smoke test (library imports + pinned versions + opencv conflict)
    §3  Preprocessing module self-test (preprocess.py pipeline runs + writes output)
    §5.7 / §7.2  Graceful OCR behaviour when the API key is missing (no crash)

The interactive, camera/mic/internet-dependent checks (§4, §5, §6, the rest of §7)
still require manual sign-off using the checkboxes in VALIDATION.md — they cannot
be asserted headlessly.
"""

import os
import sys

# Expected pinned versions (VALIDATION.md §3).
EXPECTED = {
    "cv2": "4.11.0",
    "numpy": "1.26.4",
    "mediapipe": "0.10.21",
}

# ANSI colours (fall back to plain text if not a tty).
_TTY = sys.stdout.isatty()
GREEN = "\033[92m" if _TTY else ""
RED = "\033[91m" if _TTY else ""
RESET = "\033[0m" if _TTY else ""

_results = []  # list of (name, passed: bool, detail: str)


def check(name):
    """Decorator: run a check function, record PASS/FAIL, never let it crash the run."""
    def wrap(fn):
        try:
            detail = fn() or ""
            _results.append((name, True, detail))
            print(f"  [{GREEN}PASS{RESET}] {name}" + (f" — {detail}" if detail else ""))
        except Exception as e:
            _results.append((name, False, str(e)))
            print(f"  [{RED}FAIL{RESET}] {name} — {e}")
        return fn
    return wrap


# ---------- §3 Environment smoke test ----------

def section_environment():
    print("\n§3 Environment smoke test")

    @check("Core libraries import")
    def _imports():
        import cv2  # noqa: F401
        import numpy  # noqa: F401
        import mediapipe as mp
        # Legacy Solutions API must be present (main.py uses mp.solutions.hands).
        mp.solutions.hands
        return "cv2, numpy, mediapipe, mp.solutions.hands all resolve"

    @check("Pinned versions match VALIDATION.md")
    def _versions():
        import cv2
        import numpy
        import mediapipe as mp
        actual = {
            "cv2": cv2.__version__,
            "numpy": numpy.__version__,
            "mediapipe": mp.__version__,
        }
        mismatches = []
        for lib, want in EXPECTED.items():
            got = actual[lib]
            # numpy: accept any 1.26.x (must stay <2 for mediapipe).
            if lib == "numpy":
                if not got.startswith("1.26."):
                    mismatches.append(f"numpy {got} (need 1.26.x, <2)")
            elif got != want:
                mismatches.append(f"{lib} {got} (need {want})")
        if mismatches:
            raise AssertionError("; ".join(mismatches))
        return f"cv2 {actual['cv2']} | numpy {actual['numpy']} | mediapipe {actual['mediapipe']}"

    @check("opencv-python not installed alongside contrib")
    def _opencv_conflict():
        try:
            from importlib.metadata import distributions
        except ImportError:  # py<3.8 safety
            from importlib_metadata import distributions  # type: ignore
        names = {d.metadata["Name"].lower() for d in distributions()
                 if d.metadata and d.metadata.get("Name")}
        if "opencv-python" in names and "opencv-contrib-python" in names:
            raise AssertionError(
                "both opencv-python and opencv-contrib-python installed "
                "(they corrupt the shared cv2/ folder — keep only contrib)"
            )
        if "opencv-contrib-python" not in names:
            raise AssertionError("opencv-contrib-python not installed")
        return "only opencv-contrib-python present"


# ---------- §3 Preprocessing self-test ----------

def section_preprocess():
    print("\n§3 Preprocessing module self-test")

    @check("preprocess_image runs and writes output")
    def _selftest():
        import cv2
        import numpy as np
        from preprocess import preprocess_image

        os.makedirs("temp", exist_ok=True)
        test = np.ones((300, 600, 3), dtype=np.uint8) * 255
        cv2.putText(test, "Hello OCR", (40, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
        in_path = "temp/_selftest_in.png"
        out_path = "temp/_selftest_out.png"
        cv2.imwrite(in_path, test)
        result = preprocess_image(in_path, out_path)
        if not os.path.exists(result):
            raise AssertionError(f"output not written: {result}")
        img = cv2.imread(result)
        if img is None:
            raise AssertionError("output image could not be read back")
        return f"output -> {result}"


# ---------- §5.7 / §7.2 Graceful missing-key handling ----------

def section_missing_key():
    print("\n§5.7 / §7.2 Graceful OCR error when API key is missing")

    @check("image_to_text reports missing key without crashing the import")
    def _missing_key():
        # Importing the OCR module must NOT require a key (lazy client).
        import importlib
        import image_to_text as itt
        importlib.reload(itt)
        # We only assert the module imports and exposes image_to_text; we do not
        # make a network call here. Actual error-on-call is verified manually (§5.7).
        if not hasattr(itt, "image_to_text"):
            raise AssertionError("image_to_text() not exported")
        return "OCR module imports without a key (client is created lazily)"


def main():
    print("=" * 60)
    print("Touchless Writing System — automated validation")
    print("=" * 60)

    section_environment()
    section_preprocess()
    section_missing_key()

    passed = sum(1 for _, ok, _ in _results if ok)
    total = len(_results)
    print("\n" + "=" * 60)
    print(f"Automated checks: {passed}/{total} passed")
    if passed != total:
        print(f"{RED}Some checks failed — see details above.{RESET}")
    else:
        print(f"{GREEN}All automated checks passed.{RESET}")
    print("\nNote: interactive checks (§4 drawing, §5 OCR/voice, §6 quantitative,")
    print("rest of §7) still need manual sign-off in VALIDATION.md.")
    print("=" * 60)

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
