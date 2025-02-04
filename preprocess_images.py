import cv2
import os
import random
import numpy as np

def preprocess_images(dataset_path):
    # List all image files in the dataset directory and its subdirectories
    image_files = []
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):  # Check for image file extensions
                if 'M_' not in file:  # Skip already processed images
                    image_files.append(os.path.join(root, file))

    # Randomly select images to process
    num_images_to_process = 3000  # Change this number to process more or fewer images
    selected_images = random.sample(image_files, num_images_to_process)

    def apply_random_transformations(image):
        # Apply random horizontal flip
        if random.random() > 0.5:
            image = cv2.flip(image, 1)  # 1 = Horizontal flip

        # Apply random rotation
        angle = random.uniform(-30, 30)
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h))

        # Apply random zoom (scaling)
        scale = random.uniform(0.8, 1.2)
        zoomed = cv2.resize(rotated, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

        # Apply random translation (shift)
        tx = random.uniform(-0.1, 0.1) * zoomed.shape[1]  
        ty = random.uniform(-0.1, 0.1) * zoomed.shape[0]  
        M_translation = np.float32([[1, 0, tx], [0, 1, ty]])
        translated = cv2.warpAffine(zoomed, M_translation, (zoomed.shape[1], zoomed.shape[0]))

        # Apply random brightness/contrast adjustment
        alpha = random.uniform(0.7, 1.3)  # Contrast control
        beta = random.uniform(-50, 50)   # Brightness control
        adjusted = cv2.convertScaleAbs(translated, alpha=alpha, beta=beta)

        # Apply random filter (Gaussian Blur)
        ksize = random.choice([(3, 3), (5, 5), (7, 7)])
        filtered = cv2.GaussianBlur(adjusted, ksize, 0)

        # Denoising 
        denoised = cv2.fastNlMeansDenoisingColored(filtered, None, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21)

        # Normalizing luminance and contrast
        img_yuv = cv2.cvtColor(denoised, cv2.COLOR_BGR2YUV)
        img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])  # Equalizzazione dell'istogramma sul canale Y (luminosità)
        normalized = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

        return normalized

    # Process selected images
    for image_file in selected_images:
        image = cv2.imread(image_file)

        if image is not None:
            processed_image = apply_random_transformations(image)
            # Save the processed image with a new name (prefixed with 'M_')
            directory, filename = os.path.split(image_file)
            new_filename = 'M_' + filename
            new_image_path = os.path.join(directory, new_filename)
            # Save the processed image
            print(f"Saving processed image: {new_image_path}")
            cv2.imwrite(new_image_path, processed_image)
        