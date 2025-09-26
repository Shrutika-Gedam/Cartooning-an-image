import openai
from PIL import Image
import requests
import cv2
import numpy as np
import base64
from io import BytesIO

def cartoonize_with_dalle(image_path, api_key):
    # Convert image to base64
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
    
    # Set up OpenAI client
    client = openai.OpenAI(api_key=api_key)
    
    # Generate cartoon version
    response = client.images.edit(
        model="dall-e-2",
        image=open(image_path, "rb"),
        prompt="Transform this image into a cartoon style with bold outlines and vibrant colors",
        n=1,
        size="1024x1024"
    )
    
    # Get the generated image URL
    image_url = response.data[0].url
    
    # Download the image
    generated_image = Image.open(requests.get(image_url, stream=True).raw)
    
    return generated_image

# Example usage
if __name__ == "__main__":
    # You would need to set your OpenAI API key
    # result = cartoonize_with_dalle("cat.jpg", "your-api-key-here")
    # result.save("dalle_cartoon.jpg")
    print("This requires an OpenAI API key to run")