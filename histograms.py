import os
import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops

def extract_histogram_rgb(image, bins=32):
    """
    Extracts the RGB histograms from an image.
    This function calculates the histogram for each of the three color channels (Red, Green, and Blue) of the input image.
    The histograms are then flattened and concatenated into a single list.
    Parameters:
    image (numpy.ndarray): The input image from which to extract the histograms.
    bins (int): The number of bins to use for the histograms. Default is 32.
    Returns:
    list: A list containing the concatenated histograms for the R, G, and B channels.
    """
    histograms = []
    for channel in range(3):  # Channels R, G, B
        hist = cv2.calcHist([image], [channel], None, [bins], [0, 256])
        histograms.extend(hist.flatten())
    return histograms

def extract_histogram_hsv(image, bins=32):
    """
    Extracts the HSV histogram from an image.
    This function converts the input image from BGR to HSV color space and 
    computes the histogram for each of the three channels (Hue, Saturation, 
    and Value). The histograms are then flattened and concatenated into a 
    single list.
    Parameters:
    image (numpy.ndarray): The input image in BGR color space.
    bins (int): The number of bins for the histogram (default is 32).
    Returns:
    list: A list containing the concatenated histograms for the H, S, and V 
    channels.
    """
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    histograms = []
    for channel in range(3):  # Channels H, S, V
        hist = cv2.calcHist([hsv_image], [channel], None, [bins], [0, 256])
        histograms.extend(hist.flatten())
    return histograms

def extract_glcm_features(image):
    """
    Extracts GLCM (Gray Level Co-occurrence Matrix) features from an image.
    Parameters:
    image (numpy.ndarray): Input image in BGR format.
    Returns:
    list: A list containing the following GLCM features:
        - Contrast
        - Dissimilarity
        - Homogeneity
        - Energy
        - Correlation
        If an error occurs during feature extraction, a list of five zeros is returned.
    Raises:
    Exception: If there is an error during the GLCM feature extraction process, it prints an error message and returns a list of five zeros.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    try:
        glcm = graycomatrix(gray, distances=[1], angles=[0], symmetric=True, normed=True)
        features = [
            graycoprops(glcm, 'contrast')[0, 0],
            graycoprops(glcm, 'dissimilarity')[0, 0],
            graycoprops(glcm, 'homogeneity')[0, 0],
            graycoprops(glcm, 'energy')[0, 0],
            graycoprops(glcm, 'correlation')[0, 0]
        ]
        return features
    except Exception as e:
        print(f"Errore GLCM su {image}: {e}")
        return [0] * 5  # Pre-fill with zeros in case of error

def extract_features(image_path, bins=32):
    """
    Extracts features from an image located at the given path.
    This function extracts three types of features: RGB histogram, HSV histogram, and GLCM features. 
    Args:
        image_path (str): The file path to the image.
        bins (int, optional): The number of bins to use for the histograms. Default is 32.
    Returns:
        list: A list containing the concatenated features from the RGB histogram, HSV histogram, 
              and GLCM features. If an error occurs, an empty list is returned.
    """
    image = cv2.imread(image_path)
    if image is None:
        return []

    try:
        rgb_hist = extract_histogram_rgb(image, bins)
        if len(rgb_hist) != bins * 3:
            print(f"Errore RGB: {image_path}")

        hsv_hist = extract_histogram_hsv(image, bins)
        if len(hsv_hist) != bins * 3:
            print(f"Errore HSV: {image_path}")

        glcm_features = extract_glcm_features(image)
        if len(glcm_features) != 5:
            print(f"Errore GLCM: {image_path}")

        total_features = rgb_hist + hsv_hist + glcm_features
        return total_features
    except Exception as e:
        print(f"Errore durante l'estrazione delle feature da {image_path}: {e}")
        return []

def create_histograms(dataset_path, bins=32):
    """
    Create histograms and extract features from images in a dataset.
    Args:
        dataset_path (str): The path to the dataset directory. The directory should contain 
                            subdirectories, each representing a different class label.
        bins (int, optional): The number of bins to use for the histograms. Default is 32.
    Returns:
        tuple: A tuple containing:
            - np.array: A numpy array of the extracted features for each image.
            - np.array: A numpy array of the corresponding labels for each image.
            - list: A list of column names for the features.
    """
    data = []
    labels = []
    # Define column names
    column_names = [f'Bin_R{i+1}' for i in range(bins)] + \
                   [f'Bin_G{i+1}' for i in range(bins)] + \
                   [f'Bin_B{i+1}' for i in range(bins)] + \
                   [f'Bin_H{i+1}' for i in range(bins)] + \
                   [f'Bin_S{i+1}' for i in range(bins)] + \
                   [f'Bin_V{i+1}' for i in range(bins)] + \
                   ['GLCM_Contrast', 'GLCM_Dissimilarity', 'GLCM_Homogeneity', 'GLCM_Energy', 'GLCM_Correlation'] + \
                    ['Label']
    expected_feature_length = len(column_names) - 1  # Exclude 'Label'
    # Extract features from images in the dataset
    subdirectories = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    for label in subdirectories:
        folder_path = os.path.join(dataset_path, label)
        for image_name in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_name)
            if image_name.lower().endswith(('.jpg', '.png', '.jpeg')):
                features = extract_features(image_path, bins=bins)
                if features and len(features) == expected_feature_length:
                    features = [0 if np.isnan(f) else f for f in features]
                    data.append(features + [label])
                    labels.append(label)
                else:
                    print(f"Feature mancanti o incomplete per {image_path} (estratte {len(features)} su {expected_feature_length})")        
    if not data:
        print(f"No valid data found in {dataset_path}")
        return [], [], column_names

    return np.array(data), np.array(labels), column_names
