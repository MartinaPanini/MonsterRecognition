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
    glcm = graycomatrix(gray, distances=[1], angles=[0], symmetric=True, normed=True)
    features = [
        graycoprops(glcm, 'contrast')[0, 0],
        graycoprops(glcm, 'dissimilarity')[0, 0],
        graycoprops(glcm, 'homogeneity')[0, 0],
        graycoprops(glcm, 'energy')[0, 0],
        graycoprops(glcm, 'correlation')[0, 0]
    ]
    return features

# Function to extract ORB features from an image
def extract_orb_features(image, max_features=50):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=max_features)
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    if descriptors is not None:
        return descriptors.flatten()[:max_features * 32]  # Limits number of features
    else:
        return np.zeros(max_features * 32)  # Fill with zeros if no features found

# Principal function to extract features
def extract_features(image_path, bins=32):
    image = cv2.imread(image_path)
    if image is None:
        return []

    rgb_hist = extract_histogram_rgb(image, bins)
    hsv_hist = extract_histogram_hsv(image, bins)
    glcm_features = extract_glcm_features(image)
    orb_features = extract_orb_features(image)

    return rgb_hist + hsv_hist + glcm_features + list(orb_features)

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
                   [f'ORB_{i+1}' for i in range(50 * 32)] + \
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
                if features:
                    # Pad or trim feature vector to match expected length
                    features = [0 if np.isnan(f) else f for f in features]
                    # if len(features) < expected_feature_length:
                    #     features += [0] * (expected_feature_length - len(features))
                    # elif len(features) > expected_feature_length:
                    #     features = features[:expected_feature_length]

                    data.append(features + [label])
                    labels.append(label)

    if not data:
        print(f"No valid data found in {dataset_path}")
        return [], [], column_names

    return np.array(data), np.array(labels), column_names
