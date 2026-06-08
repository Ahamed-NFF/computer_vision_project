from google import genai
from PIL import Image
from dotenv import load_dotenv  
import os

load_dotenv()  # Load environment variables from a .env file if present

_client = None


def _get_client():
    """Create the Gemini client lazily so the app can launch without an API key.

    The key is only needed for OCR (image-to-text), so we defer client creation
    until image_to_text() is actually called.
    """
    global _client
    if _client is None:
        try:
            _client = genai.Client()
        except Exception as e:
            raise RuntimeError(
                "Gemini API key not configured. Add GEMINI_API_KEY to your .env file "
                "to use the image-to-text (OCR) feature."
            ) from e
    return _client


def image_to_text(image_path):
    image = Image.open(image_path)
    response = _get_client().models.generate_content(
        model="gemini-2.5-flash",
        contents=[image, "Extract the text from the image."],
    )
    return response.text

if __name__ == "__main__":
    image_path = "path_to_your_image.jpg"  # Replace with your image path
    description = image_to_text(image_path)
    print("Image Description:", description)