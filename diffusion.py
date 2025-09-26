import torch
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from PIL import Image
import requests
from io import BytesIO
import cv2
import numpy as np

# Method 1: Text-to-Image with cartoon prompt
def generate_cartoon_with_diffusion(prompt, original_image=None):
    # Load pretrained model
    if original_image:
        # Image-to-image pipeline (modify existing image)
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16
        )
    else:
        # Text-to-image pipeline (generate from scratch)
        pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16
        )
    
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    
    # Generate cartoon image
    if original_image:
        # Convert OpenCV image to PIL
        if isinstance(original_image, np.ndarray):
            original_image = Image.fromarray(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        
        result = pipe(
            prompt=prompt,
            image=original_image,
            strength=0.7,  # How much to transform the image
            guidance_scale=7.5
        ).images[0]
    else:
        result = pipe(prompt).images[0]
    
    return result

# Method 2: Using specific cartoon-style models
def generate_cartoon_with_specialized_model(image_path):
    # Using a model specifically trained for cartoonization
    from transformers import pipeline
    
    # Load a cartoonization model (example - would need to find appropriate model)
    cartoonizer = pipeline("image-to-image", model="ogkalu/Comic-Diffusion")
    
    # Load and process image
    image = Image.open(image_path)
    cartoon_image = cartoonizer(image)
    
    return cartoon_image

# Example usage
if __name__ == "__main__":
    # Load your image
    img = cv2.imread("cat.jpg")
    
    # Generate cartoon using diffusion model
    prompt = "cartoon style, vibrant colors, clean lines, digital art, cute cat"
    cartoon_result = generate_cartoon_with_diffusion(prompt, img)
    
    # Save result
    cartoon_result.save("diffusion_cartoon.jpg")
    print("Cartoon image generated using diffusion model!")