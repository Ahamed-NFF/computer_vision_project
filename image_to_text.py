import base64
import mimetypes
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # Load environment variables from a .env file if present

_client = None


def _get_client():
    """Create the OpenAI client lazily so the app can launch without an API key.

    The key is only needed for OCR (image-to-text), so we defer client creation
    until image_to_text() is actually called.
    """
    global _client
    if _client is None:
        try:
            _client = OpenAI()
        except Exception as e:
            raise RuntimeError(
                "OpenAI API key not configured. Add OPENAI_API_KEY to your .env file "
                "to use the image-to-text (OCR) feature."
            ) from e
    return _client


def _encode_image(image_path):
    """Read an image from disk and return a base64 data URL for the OpenAI API."""
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type is None:
        mime_type = "image/png"
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


_OCR_PROMPT = (
    "Transcribe the handwritten or printed text in this image exactly as written. "
    "Output ONLY the transcribed text itself — no quotes, no labels, no explanations, "
    "no leading phrases like 'The text is'. Preserve line breaks. "
    "If the image contains no readable text, output nothing (an empty response)."
)


def _clean_ocr_output(text):
    """Strip filler/quoting the model may add so the result is the bare transcription.

    The metrics in evaluate.py and the on-canvas rendering both expect raw text,
    not a sentence like: The text extracted from the image is: "Right".
    """
    if not text:
        return ""
    cleaned = text.strip()
    # Drop a single pair of surrounding quotes if the whole output is quoted.
    if len(cleaned) >= 2 and cleaned[0] in "\"'" and cleaned[-1] == cleaned[0]:
        cleaned = cleaned[1:-1].strip()
    return cleaned


def image_to_text(image_path):
    """OCR the image at `image_path` and return its transcribed text.

    Sends the image (base64-encoded) plus a strict transcription prompt to a
    vision-capable model, then strips any filler the model adds so the caller
    gets the bare text. This is the OCR stage used by main.py and evaluate.py.
    """
    # Encode the file as a data URL the chat API can accept inline.
    data_url = _encode_image(image_path)
    # One multimodal message: the transcription instruction + the image itself.
    response = _get_client().chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _OCR_PROMPT},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
    )
    # Pull the text out of the response and tidy it before returning.
    return _clean_ocr_output(response.choices[0].message.content)

if __name__ == "__main__":
    image_path = "path_to_your_image.jpg"  # Replace with your image path
    description = image_to_text(image_path)
    print("Image Description:", description)
