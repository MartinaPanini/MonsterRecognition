import os
import shutil
import pandas as pd
import numpy as np
from joblib import dump, load
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.utils.class_weight import compute_class_weight
from run_inference import process_dataset, run_inference_and_draw
from preprocess_images import preprocess_images
from histograms import extract_histogram, create_histograms
from color_class import train_model, classify_data, evaluate_model

# Set this to True to create the dataset and preprocess it
dataset = False
histograms = True
process_images = False

# PATHS 
train_dir = "/Users/martinapanini/Library/Mobile Documents/com~apple~CloudDocs/Università/I Semestre/Signal_Image_Video/MonsterProject/Monster_energy_drink/train"  
test_dir = "/Users/martinapanini/Library/Mobile Documents/com~apple~CloudDocs/Università/I Semestre/Signal_Image_Video/MonsterProject/Monster_energy_drink/test"
output_train_dir = "/Users/martinapanini/Library/Mobile Documents/com~apple~CloudDocs/Università/I Semestre/Signal_Image_Video/MonsterProject/DatasetInference/train" 
output_test_dir = "/Users/martinapanini/Library/Mobile Documents/com~apple~CloudDocs/Università/I Semestre/Signal_Image_Video/MonsterProject/DatasetInference/test"

output_results = "/Users/martinapanini/Library/Mobile Documents/com~apple~CloudDocs/Università/I Semestre/Signal_Image_Video/MonsterProject/MonsterRecognition/Results"
model_statistics_path = "/Users/martinapanini/Library/Mobile Documents/com~apple~CloudDocs/Università/I Semestre/Signal_Image_Video/MonsterProject/MonsterRecognition/ModelResults/model_statistics.txt"
model_path = "/Users/martinapanini/Library/Mobile Documents/com~apple~CloudDocs/Università/I Semestre/Signal_Image_Video/MonsterProject/MonsterRecognition/ModelResults/trained_model.joblib"
encoder_path = "/Users/martinapanini/Library/Mobile Documents/com~apple~CloudDocs/Università/I Semestre/Signal_Image_Video/MonsterProject/MonsterRecognition/ModelResults/label_encoder.pkl"

images_folder = "/Users/martinapanini/Library/Mobile Documents/com~apple~CloudDocs/Università/I Semestre/Signal_Image_Video/MonsterProject/MonsterRecognition/images"
output_folder = "/Users/martinapanini/Library/Mobile Documents/com~apple~CloudDocs/Università/I Semestre/Signal_Image_Video/MonsterProject/MonsterRecognition"

image_name = "monster_wall.jpeg"   # CHANGE THIS TO THE IMAGE YOU WANT TO TEST

# Create Dataset with bounded boxes images from the main Dataset and preprocess them 
if dataset:
    process_dataset(train_dir, output_train_dir)
    print("Train Dataset created\n")
    process_dataset(test_dir, output_test_dir)
    print("Test Dataset created\n")
if process_images:
    preprocess_images(output_train_dir)
    print("Train Dataset preprocessed\n")

# Extract and save histograms and classes from the Train Dataset
if histograms:
    train_data, train_labels, train_column_names = create_histograms(train_dir)
    output_train_df = pd.DataFrame(train_data, columns=train_column_names)
    output_train_df['Label'] = train_labels
    output_train_df.to_csv("/Users/martinapanini/Library/Mobile Documents/com~apple~CloudDocs/Università/I Semestre/Signal_Image_Video/MonsterProject/MonsterRecognition/Results/train_histograms_features.csv", index=False)
train_data_csv = pd.read_csv("/Users/martinapanini/Library/Mobile Documents/com~apple~CloudDocs/Università/I Semestre/Signal_Image_Video/MonsterProject/MonsterRecognition/Results/train_histograms_features.csv")
X_train, y_train = train_data_csv.drop(columns=['Label']), train_data_csv['Label']

if histograms:
    test_data, test_labels, test_column_names = create_histograms(test_dir)
    output_test_df = pd.DataFrame(test_data, columns=test_column_names)
    output_test_df['Label'] = test_labels
    output_test_df.to_csv("/Users/martinapanini/Library/Mobile Documents/com~apple~CloudDocs/Università/I Semestre/Signal_Image_Video/MonsterProject/MonsterRecognition/Results/test_histograms_features.csv", index=False)
test_data_csv = pd.read_csv("/Users/martinapanini/Library/Mobile Documents/com~apple~CloudDocs/Università/I Semestre/Signal_Image_Video/MonsterProject/MonsterRecognition/Results/test_histograms_features.csv")
X_test, y_test = test_data_csv.drop(columns=['Label']), test_data_csv['Label']

# Encode the classes and compte the class weights
y_train = y_train.astype(str).str.strip().values  
y_test = y_test.astype(str).str.strip().values
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)  
y_test_encoded = label_encoder.transform(y_test)  
classes = np.unique(y_train_encoded)
class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train_encoded)
class_weight_dict = {cls: weight for cls, weight in zip(classes, class_weights)}

# Stampa per debug
# print("Mappatura Classi:", dict(zip(label_encoder.classes_, np.unique(y_train_encoded))))
# print("Pesi delle Classi:", class_weight_dict)

# Train and evaluate model the model
model = train_model(X_train, y_train_encoded, class_weight_dict) 
cv_scores = cross_val_score(model, X_train, y_train_encoded, cv=5)
metrics = evaluate_model(
    model,
    X_test,
    y_test_encoded,  
    label_encoder,
    model_statistics_path
)

# Save model and encoder 
dump(model, model_path)
with open(encoder_path, 'wb') as f:
    pickle.dump(label_encoder, f)

# Load image to recognize and run inference on the image (create bounding boxes of cans)
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
cropped_folder = os.path.join(output_folder, "ImageCropped")
os.makedirs(cropped_folder, exist_ok=True)
out_image = os.path.join(output_results, "ImageBounded.png")
image_path = os.path.join(images_folder, image_name)
if not os.path.exists(image_path):
    raise FileNotFoundError(f"The image at path {image_path} does not exist.")

run_inference_and_draw(image_path, out_image)
for file_name in os.listdir(cropped_folder):
    file_path = os.path.join(cropped_folder, file_name)
    if os.path.isfile(file_path) or os.path.islink(file_path):
        os.unlink(file_path)
    elif os.path.isdir(file_path):
        shutil.rmtree(file_path)
# Save cropped images from the image loaded
process_dataset(image_path, cropped_folder)

# Extrat histrograms from the cropped images and classify them
test_data = []
image_names = []
for image_name in os.listdir(cropped_folder):
    image_path = os.path.join(cropped_folder, image_name)
    if image_name.lower().endswith(('.jpg', '.png', '.jpeg')):
        histogram = extract_histogram(image_path, bins=32)
        if histogram:  # Verifica che l'istogramma non sia vuoto
            test_data.append(histogram)
            image_names.append(image_name)
column_names = [f'Bin_R{i}' for i in range(1, 33)] + [f'Bin_G{i}' for i in range(1, 33)] + [f'Bin_B{i}' for i in range(1, 33)]
image_data_df = pd.DataFrame(test_data, columns=column_names)

# Classify the cropped images and save the results
model = load(model_path)
with open(encoder_path, 'rb') as f:
    label_encoder = pickle.load(f)

# Classifica le immagini croppate
predicted_labels = classify_data(model, label_encoder, image_data_df)

# Salva i risultati
results_df = pd.DataFrame({
    'Image': image_names,
    'PredictedLabel': predicted_labels
})
results_path = os.path.join(output_results, "classification_results.csv")
results_df.to_csv(results_path, index=False)

# Compare cans in the dataset with the predicted labels
true_labels = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
num_true_labels = len(true_labels)
num_predicted_labels = len(set(predicted_labels))
missing_labels = set(true_labels) - set(predicted_labels)
print(f"\nMissing Monster's cans: {num_true_labels - num_predicted_labels}")
print(f"\nYou have: {predicted_labels}")
print(f"\nMissing Monster are: {missing_labels}\n")