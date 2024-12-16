import cv2
import os
import random
import numpy as np

# Path to the dataset
dataset_path = '/Users/martinapanini/Library/Mobile Documents/com~apple~CloudDocs/Università/Signal_Image_Video/MonsterProject/DatasetInference/train'

# List all image files in the dataset directory and its subdirectories
image_files = []
for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.endswith(('.png')):
            image_files.append(os.path.join(root, file))
#print(f"Found {len(image_files)} images in the dataset")
# Randomly select images to process
num_images_to_process = 100  # Change this number to process more or fewer images
selected_images = random.sample(image_files, num_images_to_process)

def apply_random_transformations(image):
    # Apply random rotation
    angle = random.uniform(-30, 30)
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))

    # Apply random zoom
    scale = random.uniform(0.8, 1.2)
    zoomed = cv2.resize(rotated, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

    # Apply random filter (Gaussian Blur)
    ksize = random.choice([(3, 3), (5, 5), (7, 7)])
    filtered = cv2.GaussianBlur(zoomed, ksize, 0)

    return filtered

for image_file in selected_images:
    image_path = os.path.join(dataset_path, image_file)
    image = cv2.imread(image_path)

    if image is not None:
        processed_image = apply_random_transformations(image)
        directory, filename = os.path.split(image_file)
        new_filename = 'M_' + filename
        new_image_path = os.path.join(dataset_path, directory, new_filename)
        cv2.imwrite(new_image_path, processed_image)
        print(f"Saved processed image: {new_image_path}")
    else:
        print(f"Failed to load image: {image_file}")
print("\nDone!\n")