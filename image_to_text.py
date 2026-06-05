from google import genai
from PIL import Image
from dotenv import load_dotenv  
import os

load_dotenv()  # Load environment variables from a .env file if present

client = genai.Client()


def image_to_text(image_path):
    image = Image.open(image_path)
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[image, "Extract the text from the image."],
    )
    return response.text

if __name__ == "__main__":
    image_path = "path_to_your_image.jpg"  # Replace with your image path
    description = image_to_text(image_path)
    print("Image Description:", description)