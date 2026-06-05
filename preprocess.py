"""
preprocess.py
Classical image-processing pipeline for the Touchless Writing System OCR stage.

Implements the steps described in the proposal (Methodology section 4 and the
Image Processing & Computer Vision Involvement section 7):

    grayscale -> contrast normalization (histogram equalization)
    -> adaptive thresholding -> morphological cleaning -> ink-region cropping

The pipeline is configurable: each stage can be toggled on/off so OCR accuracy
and latency can be evaluated WITH and WITHOUT preprocessing, as promised in the
proposal.

Usage:
    from preprocess import preprocess_image
    clean_path = preprocess_image("temp/current_page_temp.png",
                                  out_path="temp/current_page_pre.png",
                                  enable=True)
    # then feed clean_path to image_to_text(...)
"""

import cv2
import numpy as np
import os


def to_grayscale(img):
    """Convert BGR image to grayscale to reduce sensitivity to colour artifacts."""
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def normalize_contrast(gray):
    """Histogram equalization to emphasize ink strokes under uneven lighting.
    CLAHE is used instead of plain equalizeHist because it is more robust to
    local illumination changes on a writing canvas."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def adaptive_binarize(gray):
    """Adaptive thresholding produces a clean binary image across varying
    lighting. THRESH_BINARY_INV is used so strokes become white-on-black for
    morphology, then inverted back at the end for a natural black-on-white page."""
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=31,      # neighbourhood size; odd, tune to stroke width
        C=10               # constant subtracted from the mean
    )
    return binary


def morphological_clean(binary):
    """Close small gaps in broken strokes, then remove isolated speckle noise."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # Close: reconnect broken strokes
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    # Open: remove small isolated noise
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)
    return opened


def crop_ink_region(binary, pad=20):
    """Crop to the bounding box of the ink so OCR focuses on written content.
    Returns the original binary if no ink is found."""
    coords = cv2.findNonZero(binary)
    if coords is None:
        return binary
    x, y, w, h = cv2.boundingRect(coords)
    H, W = binary.shape[:2]
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(W, x + w + pad)
    y1 = min(H, y + h + pad)
    return binary[y0:y1, x0:x1]


def preprocess_image(in_path,
                     out_path="temp/preprocessed.png",
                     enable=True,
                     do_gray=True,
                     do_contrast=True,
                     do_threshold=True,
                     do_morph=True,
                     do_crop=True):
    """
    Run the configurable preprocessing pipeline on an image file.

    If enable=False the original image is passed through unchanged (used for the
    'without preprocessing' arm of the evaluation).

    Returns the path to the image that should be sent to OCR.
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    img = cv2.imread(in_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {in_path}")

    if not enable:
        # Pass-through: write a copy so the OCR stage path stays uniform
        cv2.imwrite(out_path, img)
        return out_path

    stage = img
    if do_gray:
        stage = to_grayscale(stage)
    else:
        stage = to_grayscale(stage)  # threshold/morph need single channel anyway

    if do_contrast:
        stage = normalize_contrast(stage)

    if do_threshold:
        stage = adaptive_binarize(stage)        # white ink on black
        if do_morph:
            stage = morphological_clean(stage)
        if do_crop:
            stage = crop_ink_region(stage)
        # invert back to black ink on white page for OCR friendliness
        stage = cv2.bitwise_not(stage)
    else:
        # no threshold: still allow optional crop on a binarized copy
        if do_crop:
            tmp = adaptive_binarize(stage)
            stage_cropped = crop_ink_region(tmp)
            stage = stage_cropped

    cv2.imwrite(out_path, stage)
    return out_path


if __name__ == "__main__":
    # quick self-test on a generated sample
    test = np.ones((300, 600, 3), dtype=np.uint8) * 255
    cv2.putText(test, "Hello OCR", (40, 160),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
    cv2.imwrite("temp/_selftest_in.png", test)
    out = preprocess_image("temp/_selftest_in.png", "temp/_selftest_out.png")
    print("Preprocessing self-test OK ->", out)
