import cv2
import os
import random
import numpy as np

def preprocess_images(dataset_path):
    """
    Preprocess images in the given dataset path by applying random transformations and saving the processed images.
    Args:
        dataset_path (str): The path to the dataset containing images to be processed.
    Returns:
        None
    """
    image_files = []
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):
                if 'M_' not in file:  # Skip already processed images
                    image_files.append(os.path.join(root, file))
    # Randomly select images to process
    num_images_to_process = 3000  # Change this number to process more or fewer images
    selected_images = random.sample(image_files, num_images_to_process)

    def apply_random_transformations(image):
        """
        Apply a series of random transformations to an input image.
        This function performs the following transformations:
        1. Random horizontal flip with a 50% chance.
        2. Random rotation between -30 and 30 degrees.
        3. Random scaling between 0.8 and 1.2 times the original size.
        4. Random translation up to 10% of the image dimensions.
        5. Random adjustment of contrast (0.7 to 1.3) and brightness (-50 to 50).
        6. Random Gaussian blur with kernel sizes of (3, 3), (5, 5), or (7, 7).
        7. Denoising using Non-Local Means Denoising algorithm.
        8. Histogram equalization on the Y channel of the YUV color space.
        Args:
            image (numpy.ndarray): Input image to be transformed.
        Returns:
            numpy.ndarray: Transformed image.
        """
        if random.random() > 0.5:
            image = cv2.flip(image, 1)  # 1 = Horizontal flip

        angle = random.uniform(-30, 30)
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h))

        scale = random.uniform(0.8, 1.2)
        zoomed = cv2.resize(rotated, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

        tx = random.uniform(-0.1, 0.1) * zoomed.shape[1]  
        ty = random.uniform(-0.1, 0.1) * zoomed.shape[0]  
        M_translation = np.float32([[1, 0, tx], [0, 1, ty]])
        translated = cv2.warpAffine(zoomed, M_translation, (zoomed.shape[1], zoomed.shape[0]))

        alpha = random.uniform(0.7, 1.3)  # Contrast control
        beta = random.uniform(-50, 50)   # Brightness control
        adjusted = cv2.convertScaleAbs(translated, alpha=alpha, beta=beta)

        ksize = random.choice([(3, 3), (5, 5), (7, 7)])
        filtered = cv2.GaussianBlur(adjusted, ksize, 0)

        denoised = cv2.fastNlMeansDenoisingColored(filtered, None, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21)

        img_yuv = cv2.cvtColor(denoised, cv2.COLOR_BGR2YUV)
        img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])  
        normalized = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

        return normalized

    for image_file in selected_images:
        image = cv2.imread(image_file)

        if image is not None:
            processed_image = apply_random_transformations(image)
            # Save the processed image with a new name (prefixed with 'M_')
            directory, filename = os.path.split(image_file)
            new_filename = 'M_' + filename
            new_image_path = os.path.join(directory, new_filename)
            print(f"Saving processed image: {new_image_path}")
            cv2.imwrite(new_image_path, processed_image)
        