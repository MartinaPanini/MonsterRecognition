# Monster Energy Can Recognition

This project uses machine learning and image processing techniques to recognize different types of Monster Energy cans based on their color and text. It leverages YOLO for object detection, extracts color histograms for classification, and combines text recognition for improved accuracy.

## Features
- Download and preprocess dataset
- Detect cans in images using YOLO
- Extract color histograms and train a classification model
- Recognize text on cans for additional verification
- Save classification results with bounding boxes

## Prerequisites
Make sure you have the following installed:

- Python 3.10.12 (Recommended to use [pyenv](https://github.com/pyenv/pyenv))
- Required Python packages:
  ```bash
  pip install -r requirements.txt
  ```
  
## Required Libraries
- pandas
- numpy
- joblib
- gdown
- scikit-learn
- imbalanced-learn
- pillow

## Dataset
The dataset is downloaded from Google Drive. The script will automatically download and extract the dataset:

**Google Drive Link:** [Monster Dataset](https://drive.google.com/file/d/1Zl8z7pFG6xbdUcioMZrC4dQ70KX5qKot/view?usp=sharing)

## Project Structure
```
MonsterRecognition/
│
├── ModelResults/
│   ├── trained_model.joblib        # Trained ML model
│   └── label_encoder.pkl           # Label encoder for classes
│
├── Monster_energy_drink/
│   ├── train/                      # Training images
│   └── test/                       # Test images
│
├── DatasetInference/
│   ├── train/                      # YOLO processed training images
│   └── test/                       # YOLO processed test images
│
├── Results/
│   ├── train_histograms_features.csv
│   ├── test_histograms_features.csv
│   └── classification_results.csv  # Final classification results
│
├── images/                         # Images to classify
├── ImageCropped/                   # Cropped images after YOLO detection
└── main.py                         # Main script to run the pipeline
```

## How to Use

### 1. Run the Pipeline
The main script `main.py` handles the entire process from downloading the dataset to classifying cans in images.

```bash
python main.py
```

This will:
- Download and preprocess the dataset
- Train the model
- Run inference on an image specified by the variable `image_name`
- Save cropped images with bounding boxes
- Classify cans based on color and text
- Save results in `Results/classification_results.csv`

### 2. Customize Inference
To classify a different image, modify the following line in `main.py`:
```python
image_name = "monster_wall8.JPG"  # Change this to the image you want to classify
```
Ensure the image is placed in the `images/` directory.

### 3. Boolean Variables Configuration
In `main.py`, you can control different parts of the pipeline by setting the following boolean variables:

```python
dataset = True             # Set to True to create and preprocess the dataset
download_dataset = True    # Set to True to download and extract the dataset from Google Drive
histograms = True          # Set to True to extract and save histograms from images
process_images = True      # Set to True to preprocess images before feature extraction
```

- **dataset**: Enables dataset creation and preprocessing, including bounding box creation for images.
- **download_dataset**: Downloads the dataset from Google Drive if it doesn't exist locally.
- **histograms**: Extracts color histograms from images, which are used as features for training the model.
- **process_images**: Applies preprocessing steps (e.g., resizing, normalization) to the dataset images.

Adjust these variables depending on which steps you want to execute.

### 4. View Results
Classification results will be printed in the console and saved in `Results/classification_results.csv`. The file includes:
- Image name
- Text recognized on the can
- Accuracy of text recognition
- Color-based prediction
- Accuracy of color prediction

### 5. Model Files
After training, the model and label encoder are saved in the `ModelResults/` folder:
- `trained_model.joblib`: The trained machine learning model
- `label_encoder.pkl`: Label encoder for class mapping

### 6. Troubleshooting
- **FileNotFoundError**: Ensure the image specified exists in the `images/` folder.

## References



