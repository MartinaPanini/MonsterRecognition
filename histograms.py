import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops

# Extract histograms RGB from an image
def extract_histogram_rgb(image, bins=32):
    histograms = []
    for channel in range(3):  # Canali R, G, B
        hist = cv2.calcHist([image], [channel], None, [bins], [0, 256])
        histograms.extend(hist.flatten())
    return histograms

# Extract HSV histograms from an image
def extract_histogram_hsv(image, bins=32):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    histograms = []
    for channel in range(3):  # Canali H, S, V
        hist = cv2.calcHist([hsv_image], [channel], None, [bins], [0, 256])
        histograms.extend(hist.flatten())
    return histograms

# Function to extract GLCM features from an image
def extract_glcm_features(image):
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
        return [0] * 5  # Valori predefiniti se fallisce

# Principal function to extract features
def extract_features(image_path, bins=32):
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

# Function to create dataset with new features
def create_histograms(dataset_path, bins=32):
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

    # Estrazione delle feature per ogni immagine
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
