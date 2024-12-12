import os
import cv2
import numpy as np
import pandas as pd

def extract_histogram(image_path, bins=32):
    """Estrae l'istogramma RGB da un'immagine e lo appiattisce."""
    image = cv2.imread(image_path)
    histograms = []
    for channel in range(3):  # Canali R, G, B
        hist = cv2.calcHist([image], [channel], None, [bins], [0, 256])
        histograms.extend(hist.flatten())
    return histograms

def process_dataset(dataset_path, bins=32):
    """Processa il dataset per estrarre istogrammi e associazioni di classi."""
    data = []
    labels = []
    classes = os.listdir(dataset_path)

    column_names = []
    # Aggiungi i nomi per i bin degli istogrammi
    for i in range(1, bins + 1):
        column_names.append(f'Bin_R{i}')
    for i in range(1, bins + 1):
        column_names.append(f'Bin_G{i}')
    for i in range(1, bins + 1):
        column_names.append(f'Bin_B{i}')
    
    column_names.append('Label')  # Aggiungi la colonna 'Label's
    
    for label in classes:
        folder_path = os.path.join(dataset_path, label)
        if os.path.isdir(folder_path):
            for image_name in os.listdir(folder_path):
                image_path = os.path.join(folder_path, image_name)
                if image_name.endswith(('.jpg', '.png', '.jpeg')):  # Filtra per immagini
                    histogram = extract_histogram(image_path, bins=bins)
                    data.append(histogram + [label])  # Aggiungi la label agli istogrammi
                    labels.append(label)
    
    return np.array(data), np.array(labels), column_names

# Percorso al dataset
dataset_path = '/Users/martinapanini/Library/Mobile Documents/com~apple~CloudDocs/Università/Signal_Image_Video/MonsterProject/DatasetInference/train'

# Estrazione istogrammi e classi
data, labels, column_names = process_dataset(dataset_path)

# Salvataggio dei dati in un file CSV
output_df = pd.DataFrame(data, columns=column_names)
output_df.to_csv('/Users/martinapanini/Library/Mobile Documents/com~apple~CloudDocs/Università/Signal_Image_Video/MonsterProject/MonsterRecognition/solution1/histograms_features.csv', index=False)

print("Istogrammi estratti e salvati con successo!")
