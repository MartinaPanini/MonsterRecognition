import os
import cv2
import numpy as np
import pandas as pd

def extract_histogram(image_path, bins=32):
    #print(f"Extracting histogram for: {image_path}")
    image = cv2.imread(image_path)
    # if image is None:
    #     print(f"Unable to read image: {image_path}")
    #     return []
    histograms = []
    for channel in range(3):  # Canali R, G, B
        hist = cv2.calcHist([image], [channel], None, [bins], [0, 256])
        histograms.extend(hist.flatten())
    return histograms


def create_histograms(dataset_path, bins=32):
    """Processa un dataset per estrarre istogrammi e gestisce sia directory con sottocartelle sia senza."""
    data = []
    labels = []
    column_names = []

    # Crea i nomi delle colonne per i bin degli istogrammi
    for i in range(1, bins + 1):
        column_names.append(f'Bin_R{i}')
    for i in range(1, bins + 1):
        column_names.append(f'Bin_G{i}')
    for i in range(1, bins + 1):
        column_names.append(f'Bin_B{i}')
    column_names.append('Label')  # Aggiungi la colonna per la label

    # Controlla se la directory è suddivisa in sottocartelle o contiene solo immagini
    subdirectories = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    for label in subdirectories:
        folder_path = os.path.join(dataset_path, label)
        for image_name in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_name)
            if image_name.lower().endswith(('.jpg', '.png', '.jpeg')):
                histogram = extract_histogram(image_path, bins=bins)
                if histogram:  # Verifica che l'istogramma non sia vuoto
                    data.append(histogram + [label])  # Aggiungi la classe (label)
                    labels.append(label)
    if not data:
        print(f"No valid data found in {dataset_path}")
        return [], [], column_names

    return np.array(data), np.array(labels), column_names
