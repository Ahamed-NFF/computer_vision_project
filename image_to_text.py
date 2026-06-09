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


def image_to_text(image_path):
    data_url = _encode_image(image_path)
    response = _get_client().chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Extract the text from the image."},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    image_path = "path_to_your_image.jpg"  # Replace with your image path
    description = image_to_text(image_path)
    print("Image Description:", description)
