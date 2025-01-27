import os
import shutil
import pandas as pd
import torch
import numpy as np
from joblib import dump, load
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from run_inference import process_dataset, run_inference_and_draw
from preprocess_images import preprocess_images
from histograms import extract_histogram, create_histograms
from color_class import train_model, classify_data, evaluate_model

# PATHS 
input_dir = "/Users/martinapanini/Library/Mobile Documents/com~apple~CloudDocs/Università/I Semestre/Signal_Image_Video/MonsterProject/Monster_energy_drink/Monster_energy_drink/train"  
output_dir = "/Users/martinapanini/Library/Mobile Documents/com~apple~CloudDocs/Università/I Semestre/Signal_Image_Video/MonsterProject/DatasetInference/train" 

output_results = "/Users/martinapanini/Library/Mobile Documents/com~apple~CloudDocs/Università/I Semestre/Signal_Image_Video/MonsterProject/MonsterRecognition/Results"
model_statistics_path = "/Users/martinapanini/Library/Mobile Documents/com~apple~CloudDocs/Università/I Semestre/Signal_Image_Video/MonsterProject/MonsterRecognition/ModelResults/model_statistics.txt"
model_path = "/Users/martinapanini/Library/Mobile Documents/com~apple~CloudDocs/Università/I Semestre/Signal_Image_Video/MonsterProject/MonsterRecognition/ModelResults/trained_model.joblib"
encoder_path = "/Users/martinapanini/Library/Mobile Documents/com~apple~CloudDocs/Università/I Semestre/Signal_Image_Video/MonsterProject/MonsterRecognition/ModelResults/label_encoder.pkl"

images_folder = "/Users/martinapanini/Library/Mobile Documents/com~apple~CloudDocs/Università/I Semestre/Signal_Image_Video/MonsterProject/MonsterRecognition/images"
output_folder = "/Users/martinapanini/Library/Mobile Documents/com~apple~CloudDocs/Università/I Semestre/Signal_Image_Video/MonsterProject/MonsterRecognition"

# Create Dataset with bounded boxes images from the main Dataset 
# process_dataset(input_dir, output_dir)
# print("Train Dataset created\n")
# preprocess_images(output_dir)
# print("Train Dataset preprocessed\n")


# Estrazione istogrammi e classi dal Train Dataset
data, labels, column_names = create_histograms(input_dir)
# Salvataggio dei dati in un file CSV
output_train_df = pd.DataFrame(data, columns=column_names)
output_train_df['Label'] = labels
output_train_df.to_csv("/Users/martinapanini/Library/Mobile Documents/com~apple~CloudDocs/Università/I Semestre/Signal_Image_Video/MonsterProject/MonsterRecognition/Results/dataset_histograms_features.csv", index=False)
print("Istogrammi di train estratti e salvati con successo!\n")

# Carica i dati del train dataset e divisione in train set e test set
train_data = pd.read_csv("/Users/martinapanini/Library/Mobile Documents/com~apple~CloudDocs/Università/I Semestre/Signal_Image_Video/MonsterProject/MonsterRecognition/Results/dataset_histograms_features.csv")
X_train, X_test, y_train, y_test = train_test_split(
    train_data.drop(columns=['Label']),
    train_data['Label'],
    test_size=0.2,
    random_state=42
)

# Allenare il modello
model, label_encoder = train_model(train_data)
print("Modello allenato con successo!\n")

# Valutazione del modello
metrics = evaluate_model(
    model,
    X_test,
    label_encoder.transform(y_test),
    label_encoder,
    model_statistics_path
    )

# Salvataggio del modello e dell'encoder per riutilizzo futuro
dump(model, model_path)
print("Modello salvato con successo!\n")
with open(encoder_path, 'wb') as f:
    pickle.dump(label_encoder, f)
print("Label encoder salvato con successo!\n")


# Load image to recognize and run inference on the image (create bounding boxes of cans)
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
cropped_folder = os.path.join(output_folder, "ImageCropped")
os.makedirs(cropped_folder, exist_ok=True)
out_image = os.path.join(output_results, "ImageBounded.png")
image_name = "monster_wall7.jpeg"
image_path = os.path.join(images_folder, image_name)
if not os.path.exists(image_path):
    raise FileNotFoundError(f"The image at path {image_path} does not exist.")

run_inference_and_draw(image_path, out_image)
print("\nImage bounded\n")
# Remove any existing files in the image_path directory
for file_name in os.listdir(cropped_folder):
    file_path = os.path.join(cropped_folder, file_name)
    if os.path.isfile(file_path) or os.path.islink(file_path):
        os.unlink(file_path)
    elif os.path.isdir(file_path):
        shutil.rmtree(file_path)
process_dataset(image_path, cropped_folder)
print("Image cropped\n")

# Estrai gli istogrammi da tutte le immagini in CroppedImages
test_data = []
image_names = []
for image_name in os.listdir(cropped_folder):
    image_path = os.path.join(cropped_folder, image_name)
    if image_name.lower().endswith(('.jpg', '.png', '.jpeg')):
        histogram = extract_histogram(image_path, bins=32)
        if histogram:  # Verifica che l'istogramma non sia vuoto
            test_data.append(histogram)
            image_names.append(image_name)

# Definisci i nomi delle colonne per i bin degli istogrammi
column_names = [f'Bin_R{i}' for i in range(1, 33)] + [f'Bin_G{i}' for i in range(1, 33)] + [f'Bin_B{i}' for i in range(1, 33)]

# Crea un DataFrame per i dati di test
test_data_df = pd.DataFrame(test_data, columns=column_names)
print(f"Istogrammi estratti per {len(test_data)} immagini e convertiti in DataFrame.")

# Classifica le immagini usando il DataFrame di test
predicted_labels = classify_data(model, label_encoder, test_data_df)

# Crea un DataFrame per salvare i risultati
import pandas as pd
results_df = pd.DataFrame({
    'Image': image_names,
    'PredictedLabel': predicted_labels
})

# Salva i risultati in un file CSV
results_path = os.path.join(output_results, "classification_results.csv")
results_df.to_csv(results_path, index=False)
print(f"Classificazione completata e salvata in: {results_path}")

# Confronto lattine riconosciute con quelle nel dataset
true_labels = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
# Stampa il numero di lattine nel dataset e il numero di predicted labels
num_true_labels = len(true_labels)
num_predicted_labels = len(set(predicted_labels))
missing_labels = set(true_labels) - set(predicted_labels)
print(f"Lattine mancanti: {num_true_labels - num_predicted_labels}")
print(f"Lattine mancanti nel dataset: {missing_labels}\n")