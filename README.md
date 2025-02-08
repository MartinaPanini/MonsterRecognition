# Monster Energy Can Recognition

This project uses machine learning and image processing techniques to recognize different types of Monster Energy cans based on their color and text. It detects a single can, extracts color histograms for classification, and combines text recognition for improved accuracy.

## Features
- Download and preprocess dataset
- Detect cans in images using ```tin-can-r0yev/1``` model
- Extract color histograms and train a classification model
- Recognize text on cans for additional verification
- Save classification results with bounding boxes

## Bounding boxes
In the `BoundingBox.py` file there is an attempt to create bounding boxes with the OpenCV library.

## Prerequisites
Make sure you have the following installed:

- Python 3.10.12 (Recommended to use [pyenv](https://github.com/pyenv/pyenv))
- Required Python packages:
  ```bash
  pip install -r requirements.txt
  ```

## Dataset
The dataset is downloaded from Google Drive. The script will automatically download and extract the dataset:

**Google Drive Link:** [Monster Dataset](https://drive.google.com/file/d/1Zl8z7pFG6xbdUcioMZrC4dQ70KX5qKot/view?usp=sharing)

## Project Structure
```
MonsterRecognition
├── ImageCropped
│   
├── Images
│   
├── ModelResults
│   ├── label_encoder.pkl
│   ├── model_statistics.txt
│   └── trained_model.joblib
├── README.md
├── Results
│   ├── ImageBounded.png
│   ├── classification_results.csv
│   ├── classification_results.txt
│   ├── test_histograms_features.csv
│   └── train_histograms_features.csv
├── __pycache__
│
├── color_class.py
├── download_dataset.ipynb
├── histograms.py
├── main.py
├── preprocess_images.py
├── requirements.txt
├── run_inference.py
├── BoundingBox.py
├── monster_dataset.py
└── text_detection.py
```

## How to Use

### Download dataset
Download the dataset from the notebook `download_dataset.ipynb` or from the Drive folder (dataset already preprocessed and with inference images) in `main.py`.

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

- **dataset**: After downloaded the dataset, it provides to create bounding boxes and save the new dataset with single cans. It will be saved in ```DatasetInference``` folder.
- **download_dataset**: Downloads the dataset from Google Drive if it doesn't exist locally. It allow to download the dataset with images already extracted with bounding boxes. In this dataset are also added classes of Monster cans. 
- **histograms**: Extracts color histograms from images, which are used as features for training the model. This operation must be done only the first time, then results are stored in csv files. 
- **process_images**: Applies preprocessing steps (e.g., resizing, normalization) to the dataset images. This step can be skipped if you have downloaded the dataset from the Drive folder. 

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
[![Dataset su Kaggle](https://img.shields.io/badge/Kaggle-Monster%20Energy%20Drink-blue?logo=kaggle)](https://www.kaggle.com/datasets/tmmarquess/monster-energy-drink)

[EasyOCR](https://github.com/JaidedAI/EasyOCR)

[Roboflow Inference SDK](https://github.com/roboflow/inference)

[OpenCV](https://opencv.org/)
