import cv2
import numpy as np
import os

# Define input and output directories
input_folder = "solution1/bounded_images"
output_folder = "solution1/enhanced_images"
os.makedirs(output_folder, exist_ok=True)

def enhance_image(image_path, output_path):
    # Load the cropped image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load {image_path}")
        return

    # Step 1: Denoise the image
    denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

    # Step 2: Apply sharpening filter
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(denoised, -1, kernel)

    # Step 3: Upscale the image
    upscaled = cv2.resize(sharpened, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # Save the processed image
    cv2.imwrite(output_path, upscaled)
    print(f"Saved enhanced image to {output_path}")

# Process each cropped image
for filename in os.listdir(input_folder):
    input_path = os.path.join(input_folder, filename)
    output_path = os.path.join(output_folder, f"enhanced_{filename}")
    enhance_image(input_path, output_path)

print("Image enhancement completed.")
