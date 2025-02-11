import os
import shutil
import pandas as pd
import numpy as np
from joblib import dump, load
import pickle
import gdown
import zipfile
from PIL import Image
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.utils.class_weight import compute_class_weight

from run_inference import process_dataset, run_inference_and_draw
from preprocess_images import preprocess_images
from histograms import extract_features, create_histograms
from color_class import train_model, classify_data, evaluate_model
from text_detection import text_classification

# Change the following variables to run the corresponding part of the code
dataset = False
download_dataset = False
histograms = False
process_images = False

# Dataset Google Drive link
FILE_ID = "1Zl8z7pFG6xbdUcioMZrC4dQ70KX5qKot" #https://drive.google.com/file/d/1Zl8z7pFG6xbdUcioMZrC4dQ70KX5qKot/view?usp=sharing
OUTPUT_ZIP = "dataset.zip"
EXTRACT_FOLDER = "MonsterDataset"

# PATHS 
# Change root_path and repo_path to the correct ones
root_path = "/Users/martinapanini/Library/Mobile Documents/com~apple~CloudDocs/Università/I Semestre/Signal_Image_Video/MonsterProject"
repo_path = "/Users/martinapanini/Library/Mobile Documents/com~apple~CloudDocs/Università/I Semestre/Signal_Image_Video/MonsterProject/MonsterRecognition"

train_dir = os.path.join(root_path, "Monster_energy_drink/train")
test_dir = os.path.join(root_path, "Monster_energy_drink/test")
output_train_dir = os.path.join(root_path, "DatasetInference/train")
output_test_dir = os.path.join(root_path, "DatasetInference/test")

output_results = os.path.join(repo_path, "Results")
model_statistics_path = os.path.join(repo_path, "ModelResults/model_statistics.txt")
model_path = os.path.join(repo_path, "ModelResults/trained_model.joblib")
encoder_path = os.path.join(repo_path, "ModelResults/label_encoder.pkl")

images_folder = os.path.join(repo_path, "Images")
output_folder = repo_path
cropped_folder = os.path.join(output_folder, "ImageCropped")    

image_name = "tris2.JPG"   # CHANGE THIS TO THE IMAGE YOU WANT TO TEST

# Create Dataset with bounded boxes images from the main Dataset and preprocess them 
if dataset:
    process_dataset(train_dir, output_train_dir)
    print("Train Dataset created")
    process_dataset(test_dir, output_test_dir)
    print("Test Dataset created\n")
if download_dataset:
    if not os.path.exists(OUTPUT_ZIP):
        print("Downloading dataset...")
        gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", OUTPUT_ZIP, quiet=False)
    if not os.path.exists(EXTRACT_FOLDER):
        print("Extracting dataset...")
        with zipfile.ZipFile(OUTPUT_ZIP, 'r') as zip_ref:
            zip_ref.extractall(EXTRACT_FOLDER)
    output_train_dir = os.path.join(root_path, "MonsterDataset/DatasetInference/train")
    output_test_dir = os.path.join(root_path, "MonsterDataset/DatasetInference/test")
    print("Dataset downloaded and extracted\n")
if process_images:
    preprocess_images(output_train_dir)
    print("Train Dataset preprocessed\n")

# Extract and save histograms and classes from the Train Dataset
if histograms:
    train_data, train_labels, train_column_names = create_histograms(output_train_dir)
    train_data = [item[0] if isinstance(item[0], list) else item for item in train_data]
    output_train_df = pd.DataFrame(train_data, columns=train_column_names)
    output_train_df['Label'] = train_labels
    output_train_df.to_csv("/Users/martinapanini/Library/Mobile Documents/com~apple~CloudDocs/Università/I Semestre/Signal_Image_Video/MonsterProject/MonsterRecognition/Results/train_histograms_features.csv", index=False)
    print("Train Dataset histograms extracted")
train_data_csv = pd.read_csv("/Users/martinapanini/Library/Mobile Documents/com~apple~CloudDocs/Università/I Semestre/Signal_Image_Video/MonsterProject/MonsterRecognition/Results/train_histograms_features.csv")
X_train, y_train = train_data_csv.drop(columns=['Label']), train_data_csv['Label']

if histograms:
    test_data, test_labels, test_column_names = create_histograms(output_test_dir)
    output_test_df = pd.DataFrame(test_data, columns=test_column_names)
    output_test_df['Label'] = test_labels
    output_test_df.to_csv("/Users/martinapanini/Library/Mobile Documents/com~apple~CloudDocs/Università/I Semestre/Signal_Image_Video/MonsterProject/MonsterRecognition/Results/test_histograms_features.csv", index=False)
    print("Test Dataset histograms extracted\n")
test_data_csv = pd.read_csv("/Users/martinapanini/Library/Mobile Documents/com~apple~CloudDocs/Università/I Semestre/Signal_Image_Video/MonsterProject/MonsterRecognition/Results/test_histograms_features.csv")
X_test, y_test = test_data_csv.drop(columns=['Label']), test_data_csv['Label']

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

# Encode the classes and compte the class weights
y_train = y_train.astype(str).str.strip().values  
y_test = y_test.astype(str).str.strip().values
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(pd.Series(y_train).str.strip().values)
y_test_encoded = label_encoder.fit_transform(pd.Series(y_test).str.strip().values)
classes = np.unique(y_train_encoded)
class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train_encoded)
class_weight_dict = {cls: weight for cls, weight in zip(classes, class_weights)}

# Train and evaluate model the model
ros = RandomOverSampler(random_state=42)
X_train_balanced, y_train_balanced = ros.fit_resample(X_train_scaled, y_train_encoded)

model = train_model(X_train_scaled, y_train_encoded) 
cv_scores = cross_val_score(model, X_train, y_train_encoded, cv=5)
metrics = evaluate_model(
    model,
    X_test_scaled,
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
# Remove images that are wider than they are tall
for image_name in os.listdir(cropped_folder):
    image_path = os.path.join(cropped_folder, image_name)
    if image_name.lower().endswith(('.jpg', '.png', '.jpeg')):
        with open(image_path, 'rb') as img_file:
            img = Image.open(img_file)
            width, height = img.size
            if width > height:
                os.remove(image_path)

# Extrat histrograms from the cropped images and classify them
test_data = []
image_names = []
for image_name in os.listdir(cropped_folder):
    image_path = os.path.join(cropped_folder, image_name)
    if image_name.lower().endswith(('.jpg', '.png', '.jpeg')):
        histogram = extract_features(image_path, bins=32)
        if histogram:  # Verifica che l'istogramma non sia vuoto
            test_data.append(histogram)
            image_names.append(image_name)
image_data_df = pd.DataFrame(test_data, columns=X_train.columns)
image_data_df_scaled = scaler.transform(image_data_df)
image_data_df_scaled = pd.DataFrame(image_data_df_scaled, columns=X_train.columns)

# Classify the cropped images and save the results
model = load(model_path)
with open(encoder_path, 'rb') as f:
    label_encoder = pickle.load(f)

# Classifica le immagini croppate
predicted_labels, pred_accuracy, prediction_probs = classify_data(model, label_encoder, image_data_df_scaled)

text_results = pd.DataFrame(columns=['Filter','Text','Accuracy'])
i = 0
for img_name in image_names:
    img_path = os.path.join(cropped_folder, img_name)
    try:
        # text_classification legge l'immagine internamente e restituisce informazioni
        text_info = text_classification(img_path)
        text_results.loc[i] = text_info.iloc[0]
    except Exception as e:
        print(f"Errore durante il riconoscimento del testo per {img_name}: {e}")
        text_results.loc[i] = ({'Filter': None, 'Text': None, 'Accuracy': 0})
    i += 1


results_df = pd.DataFrame({
    'Image': image_names,
    'TextFilter':text_results['Filter'],
    'Text': text_results['Text'],
    'TextAccuracy': text_results['Accuracy'],
    'Color Prediction': predicted_labels,
    'Color Accuracy': pred_accuracy
})
print("\nCOLOR AND TEXT CLASSIFICATION RESULTS:")
print(results_df)

results_path = os.path.join(output_results, "classification_results.csv")
results_df.to_csv(results_path, index=False)

# Compare text recognition accuracy with color prediction accuracy and print the labels with the highest accuracy
highest_accuracy_labels = []

for i in range(len(results_df)):
    if results_df.loc[i, 'TextAccuracy'] > results_df.loc[i, 'Color Accuracy'] and results_df.loc[i, 'TextAccuracy'] > 60:
        highest_accuracy_labels.append(results_df.loc[i, 'Text'])
    elif results_df.loc[i, 'Color Accuracy'] > 60:
        highest_accuracy_labels.append(results_df.loc[i, 'Color Prediction'])

print("\nMONSTER THAT YOU HAVE:")
print(highest_accuracy_labels)

# Compare cans in the dataset with the predicted labels
true_labels = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
missing_labels = set(true_labels) - set(predicted_labels)
print(f"\nMISSING MONSTER ARE: {missing_labels}\n")

# Save results to a text file
results_text_path = os.path.join(output_results, "classification_results.txt")
with open(results_text_path, 'w') as f:
    f.write("\nCOLOR AND TEXT CLASSIFICATION RESULTS:\n")
    f.write(results_df.to_string() + "\n")
    f.write("\nMONSTER THAT YOU HAVE:\n")
    f.write("\n".join(highest_accuracy_labels) + "\n")
    f.write("\nMISSING MONSTER ARE:\n")
    f.write(", ".join(missing_labels) + "\n")
