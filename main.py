import cv2
import numpy as np

# Global variables for trackbar values
color_levels = 8
blur_value = 5
edge_thickness = 1
style_option = 0
texture_intensity = 0
background_option = 0

# Callback function for trackbars
def nothing(x):
    pass

# Function to apply color quantization
def color_quantization(img, k):
    data = np.float32(img).reshape((-1, 3))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)
    ret, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    result = center[label.flatten()]
    result = result.reshape(img.shape)
    return result

# Function to apply comic style effect
def apply_comic_effect(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.Laplacian(gray, cv2.CV_8U, ksize=5)
    ret, mask = cv2.threshold(edges, 100, 255, cv2.THRESH_BINARY_INV)
    return cv2.bitwise_and(img, img, mask=mask)

# Function to apply pencil sketch effect
def apply_pencil_sketch(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (21, 21), 0, 0)
    sketch = cv2.divide(gray, gray_blur, scale=256)
    return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)

# Function to generate texture pattern
def generate_texture_pattern(size, pattern_type=0):
    height, width = size
    texture = np.zeros((height, width, 3), dtype=np.uint8)
    
    if pattern_type == 0:  # Canvas
        for i in range(0, height, 10):
            cv2.line(texture, (0, i), (width, i), (200, 200, 200), 1)
        for j in range(0, width, 10):
            cv2.line(texture, (j, 0), (j, height), (200, 200, 200), 1)
    
    elif pattern_type == 1:  # Dots
        for i in range(0, height, 8):
            for j in range(0, width, 8):
                cv2.circle(texture, (j, i), 1, (200, 200, 200), -1)
    
    elif pattern_type == 2:  # Crosshatch
        for i in range(0, height, 6):
            cv2.line(texture, (0, i), (width, i), (200, 200, 200), 1)
        for j in range(0, width, 6):
            cv2.line(texture, (j, 0), (j, height), (200, 200, 200), 1)
        for i in range(0, height, 6):
            cv2.line(texture, (0, i), (width, i), (200, 200, 200), 1)
    
    return texture

# Function to apply background effect
def apply_background_effect(img, option):
    if option == 0:  # Original background
        return img
    
    # Create a mask of the foreground
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    
    if option == 1:  # White background
        background = np.ones_like(img) * 255
    elif option == 2:  # Black background
        background = np.zeros_like(img)
    elif option == 3:  # Gradient background
        background = np.zeros_like(img)
        for i in range(3):  # Apply gradient to each channel
            background[:, :, i] = np.tile(np.linspace(100, 255, img.shape[1]), (img.shape[0], 1))
    
    # Apply the mask
    foreground = cv2.bitwise_and(img, img, mask=mask)
    background = cv2.bitwise_and(background, background, mask=cv2.bitwise_not(mask))
    return cv2.add(foreground, background)

# Main function to apply cartoon effect
def apply_cartoon_effect(img):
    global color_levels, blur_value, edge_thickness, style_option, texture_intensity, background_option
    
    # Apply selected style
    if style_option == 0:  # Original cartoon style
        # Color quantization
        quantized = color_quantization(img, color_levels)
        
        # Prep grayscale & blur
        g = cv2.cvtColor(quantized, cv2.COLOR_BGR2GRAY)
        g = cv2.medianBlur(g, blur_value * 2 + 1)  # Ensure odd number
        
        # Edge detection
        e = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
        
        # Morphological operations to enhance edges
        kernel = np.ones((edge_thickness, edge_thickness), np.uint8)
        e = cv2.morphologyEx(e, cv2.MORPH_CLOSE, kernel)
        
        # Smooth color
        c = cv2.bilateralFilter(quantized, 9, 250, 250)
        
        # Combine
        cartoon = cv2.bitwise_and(c, c, mask=e)
        
    elif style_option == 1:  # Comic style
        cartoon = apply_comic_effect(img)
        
    elif style_option == 2:  # Pencil sketch
        cartoon = apply_pencil_sketch(img)
    
    # Apply background effect
    cartoon = apply_background_effect(cartoon, background_option)
    
    # Apply texture if intensity > 0
    if texture_intensity > 0:
        texture = generate_texture_pattern(cartoon.shape[:2], pattern_type=texture_intensity % 3)
        cartoon = cv2.addWeighted(cartoon, 0.9, texture, 0.1, 0)
    
    return cartoon

# Load image
img = cv2.imread("cat.jpg")
if img is None:
    print("Image not found")
    exit()

# Create window and trackbars
cv2.namedWindow('Cartoon Effect')
cv2.createTrackbar('Color Levels', 'Cartoon Effect', color_levels, 50, nothing)
cv2.createTrackbar('Blur', 'Cartoon Effect', blur_value, 15, nothing)
cv2.createTrackbar('Edge Thickness', 'Cartoon Effect', edge_thickness, 5, nothing)
cv2.createTrackbar('Style', 'Cartoon Effect', style_option, 2, nothing)
cv2.createTrackbar('Texture', 'Cartoon Effect', texture_intensity, 3, nothing)
cv2.createTrackbar('Background', 'Cartoon Effect', background_option, 3, nothing)

print("Controls:")
print("1. Color Levels: Adjust color quantization (2-50)")
print("2. Blur: Adjust blur intensity (1-15)")
print("3. Edge Thickness: Adjust edge thickness (1-5)")
print("4. Style: 0=Cartoon, 1=Comic, 2=Pencil Sketch")
print("5. Texture: 0=None, 1=Canvas, 2=Dots, 3=Crosshatch")
print("6. Background: 0=Original, 1=White, 2=Black, 3=Gradient")
print("Press 's' to save image, 'q' to quit")

while True:
    # Get trackbar positions
    color_levels = cv2.getTrackbarPos('Color Levels', 'Cartoon Effect')
    blur_value = cv2.getTrackbarPos('Blur', 'Cartoon Effect')
    edge_thickness = cv2.getTrackbarPos('Edge Thickness', 'Cartoon Effect')
    style_option = cv2.getTrackbarPos('Style', 'Cartoon Effect')
    texture_intensity = cv2.getTrackbarPos('Texture', 'Cartoon Effect')
    background_option = cv2.getTrackbarPos('Background', 'Cartoon Effect')
    
    # Apply cartoon effect with current parameters
    cartoon = apply_cartoon_effect(img)
    
    # Display images
    #cv2.imshow('Original', img)
    cv2.imshow('Cartoon Effect', cartoon)
    
    # Handle keyboard input
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Quit
        break
    elif key == ord('s'):  # Save
        cv2.imwrite("cartoon_output.jpg", cartoon)
        print("Image saved as cartoon_output.jpg")

cv2.destroyAllWindows()