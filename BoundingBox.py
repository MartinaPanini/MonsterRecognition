
import os
import cv2
import shutil
import numpy as np

import nbimporter
from monster_dataset import CanDataset, transform
from torch.utils.data import DataLoader, Dataset

# Tentativo di prova per la creazione di bounding box sulle lattine, usando matchtemplate di opencv, fornendo come template un immagine di lattina dal dataset 
# e andando a provarla su tutte le altre immagini della cartella train (modificabile in test) del dataset. 
# Il codice prova a usare i tre metodi prefissati della funzione di opencv normalizzati (TM_CCOEFF_NORMED, TM_CCOEFF, TM_SQDIFF_NORMED), ottenendo risultati 
# decenti, specialmente sulle immagini più semplici. 
# C'è sia la versione normale sia una in cui provo a ciclare variando gli angoli di rotazione e la scalatura del template.
# Ho provato a testarla anche su una delle foto della cartella Images (tris9) ma il risultato non è così buono.


import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Paths to images
image_path = "/home/elia_avanzolini/.cache/kagglehub/datasets/tmmarquess/monster-energy-drink/versions/2/Monster_energy_drink_png/Monster_energy_drink/train/Aussie Lemonade/Imagem_29.png"
template_path = "/home/elia_avanzolini/.cache/kagglehub/datasets/tmmarquess/monster-energy-drink/versions/2/Monster_energy_drink_png/Monster_energy_drink/train/Aussie Lemonade/Imagem_25.png"

# Load images in both grayscale and color
img_gray = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
img_color = cv.imread(image_path, cv.IMREAD_COLOR)  # Load in color
assert img_gray is not None, "Image file could not be read, check with os.path.exists()"
assert img_color is not None, "Image file could not be read, check with os.path.exists()"
template = cv.imread(template_path, cv.IMREAD_GRAYSCALE)
assert template is not None, "Template file could not be read, check with os.path.exists()"

# Template dimensions
w, h = template.shape[::-1]

# Methods for template matching
methods = ['TM_CCOEFF_NORMED', 'TM_CCOEFF', 'TM_SQDIFF_NORMED']  # Only normalized and non-normalized coefficient methods

for meth in methods:
    img_display = img_color.copy()  # Work with the original color image
    method = getattr(cv, meth)

    # Apply template matching
    res = cv.matchTemplate(img_gray, template, method)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

    # Find the best match (min for TM_SQDIFF, max for others)
    if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc

    # Compute bottom-right corner
    bottom_right = (top_left[0] + w, top_left[1] + h)

    # Draw a rectangle around the detected area (in red for visibility)
    cv.rectangle(img_display, top_left, bottom_right, (0, 0, 255), 2)

    # Plot the matching result and detected image in color
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(res, cmap='gray')
    plt.title('Matching Result')
    plt.xticks([]), plt.yticks([])

    plt.subplot(1, 2, 2)
    plt.imshow(cv.cvtColor(img_display, cv.COLOR_BGR2RGB))  # Convert BGR to RGB for proper display
    plt.title('Detected Template (Color)')
    plt.xticks([]), plt.yticks([])

    plt.suptitle(f"Method: {meth}")
    plt.show()


#####################################################################################################################################################
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

# Specify the type of can
can_type = "Aussie Lemonade"  # Change this to select another type
base_path = "/home/elia_avanzolini/.cache/kagglehub/datasets/tmmarquess/monster-energy-drink/versions/2/Monster_energy_drink_png/Monster_energy_drink/train/"

# Path to the directory containing can images
can_dir = os.path.join(base_path, can_type)
assert os.path.exists(can_dir), f"Directory {can_dir} does not exist"

# Select the first file as the template
template_path = os.path.join(can_dir, sorted(os.listdir(can_dir))[0])
template = cv.imread(template_path, cv.IMREAD_GRAYSCALE)
assert template is not None, "Template file could not be read"

# Template dimensions
w, h = template.shape[::-1]

# Template matching methods (normalized only)
methods = ['TM_CCOEFF_NORMED', 'TM_CCORR_NORMED', 'TM_SQDIFF_NORMED']

# Apply template matching to all images in the directory
for image_file in sorted(os.listdir(can_dir)):
    image_path = os.path.join(can_dir, image_file)
    img_gray = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    img_color = cv.imread(image_path, cv.IMREAD_COLOR)  # Color version
    if img_gray is None or img_color is None:
        print(f"File {image_file} could not be read, skipping...")
        continue

    # Check that the image is larger than the template
    if img_gray.shape[0] < h or img_gray.shape[1] < w:
        print(f"Image {image_file} is smaller than the template, skipping...")
        continue

    fig, axes = plt.subplots(1, len(methods), figsize=(15, 5))

    for i, meth in enumerate(methods):
        img_display = img_color.copy()
        method = getattr(cv, meth)

        # Apply template matching
        res = cv.matchTemplate(img_gray, template, method)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

        # Determine the best match location
        if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc

        # Calculate the bottom-right corner
        bottom_right = (top_left[0] + w, top_left[1] + h)

        # Draw a rectangle around the detected template
        cv.rectangle(img_display, top_left, bottom_right, (0, 0, 255), 2)

        # Display the detected template in the image
        axes[i].imshow(cv.cvtColor(img_display, cv.COLOR_BGR2RGB))
        axes[i].set_title(f"Method: {meth}")
        axes[i].axis("off")

    plt.suptitle(f"Template Matching Results (Image: {image_file})", fontsize=14)
    plt.tight_layout()
    plt.show()
##########################################################################################################################################################
import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

# Paths to the dataset and categories
dataset_path = "/home/elia_avanzolini/.cache/kagglehub/datasets/tmmarquess/monster-energy-drink/versions/2/Monster_energy_drink_png/Monster_energy_drink/train"
can_type = "Original green"  # Change this to the desired can type
template_filename = "Imagem_6.png"  # Template file for the given type
scene_folder = os.path.join(dataset_path, can_type)

# Dictionary to map method names to OpenCV constants
matching_methods = {
    "TM_CCORR_NORMED": cv2.TM_CCORR_NORMED,
    "TM_CCOEFF_NORMED": cv2.TM_CCOEFF_NORMED,
    "TM_SQDIFF_NORMED": cv2.TM_SQDIFF_NORMED
}

# Variable to select the desired matching method (change this to any key from the dictionary)
selected_method = "TM_SQDIFF_NORMED"  # You can change this to any of the available methods

# Get the selected matching method
method = matching_methods.get(selected_method, cv2.TM_CCORR_NORMED)  # Default to TM_CCOEFF_NORMED if not found


# Load the template
template_path = os.path.join(scene_folder, template_filename)
template = cv2.imread(template_path, cv2.IMREAD_COLOR)
assert template is not None, f"Template file not found at {template_path}"

# Convert the template to grayscale
template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

# Get template dimensions
h, w = template_gray.shape

# Initialize parameters for scaling and rotation
scales = np.linspace(0.5, 1.5, 10)  # From 50% to 150% of the original size
angles = np.arange(0, 360, 10)     # Every 10 degrees

# Iterate over all images in the scene folder
for scene_filename in os.listdir(scene_folder):
    if scene_filename == template_filename:
        continue  # Skip the template itself
    
    scene_path = os.path.join(scene_folder, scene_filename)
    scene = cv2.imread(scene_path, cv2.IMREAD_COLOR)
    assert scene is not None, f"Scene file not found at {scene_path}"

    # Convert the scene to grayscale
    scene_gray = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)

    # Initialize variables to store the best match
    best_match = None
    best_val = -np.inf
    best_bbox = None

    # Perform template matching with scaling and rotation
    for scale in scales:
        for angle in angles:
            # Resize template based on scale
            scaled_template = cv2.resize(template_gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
            sh, sw = scaled_template.shape

            # Check if template size exceeds scene size
            if sw > scene_gray.shape[1] or sh > scene_gray.shape[0]:
                continue  # Skip this iteration if the template is too large for the scene

            # Rotate the scaled template
            center = (sw // 2, sh // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated_template = cv2.warpAffine(scaled_template, rotation_matrix, (sw, sh), flags=cv2.INTER_LINEAR)

            # Match the template to the scene
            result = cv2.matchTemplate(scene_gray, rotated_template, method)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

            # Update the best match if necessary
            if max_val > best_val:
                best_val = max_val
                best_match = rotated_template
                best_bbox = (max_loc[0], max_loc[1], max_loc[0] + sw, max_loc[1] + sh)

    # Draw the best match on the scene image
    if best_bbox:
        x1, y1, x2, y2 = best_bbox
        cv2.rectangle(scene, (x1, y1), (x2, y2), (0, 255, 0), 3)

    # Display the results
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(template, cv2.COLOR_BGR2RGB))
    plt.title("Template")

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(scene, cv2.COLOR_BGR2RGB))
    plt.title(f"Detected Match in {scene_filename}")

    plt.show()
################################################################################################################################################################
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the template and scene images
template_path = "/home/elia_avanzolini/.cache/kagglehub/datasets/tmmarquess/monster-energy-drink/versions/2/Monster_energy_drink_png/Monster_energy_drink/train/Aussie Lemonade/Imagem_1.png"
scene_path = "/home/elia_avanzolini/Scaricati/tris9.jpeg"

template = cv2.imread(template_path, cv2.IMREAD_COLOR)
scene = cv2.imread(scene_path, cv2.IMREAD_COLOR)

# Dictionary to map method names to OpenCV constants
matching_methods = {
    "TM_CCORR_NORMED": cv2.TM_CCORR_NORMED,
    "TM_CCOEFF_NORMED": cv2.TM_CCOEFF_NORMED,
    "TM_SQDIFF_NORMED": cv2.TM_SQDIFF_NORMED
}

# Variable to select the desired matching method (change this to any key from the dictionary)
selected_method = "TM_CCOEFF_NORMED"  # You can change this to any of the available methods

# Get the selected matching method
method = matching_methods.get(selected_method, cv2.TM_CCORR_NORMED)  # Default to TM_CCOEFF_NORMED if not found

# Convert images to grayscale
template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
scene_gray = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)

# Get template dimensions
h, w = template_gray.shape

# Initialize variables to store the best match
best_match = None
best_val = -np.inf
best_bbox = None

# Define scales and rotations to search
scales = np.linspace(0.5, 1.5, 20)  # From 50% to 150% of the original size
angles = np.arange(0, 360, 10)      # Every 30 degrees

# Perform template matching with scaling and rotation
for scale in scales:
    for angle in angles:
        # Resize template based on scale
        scaled_template = cv2.resize(template_gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        sh, sw = scaled_template.shape

        # Rotate the scaled template
        center = (sw // 2, sh // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_template = cv2.warpAffine(scaled_template, rotation_matrix, (sw, sh), flags=cv2.INTER_LINEAR)

        # Match the template to the scene
        result = cv2.matchTemplate(scene_gray, rotated_template, method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        # Update the best match if necessary
        if max_val > best_val:
            best_val = max_val
            best_match = rotated_template
            best_bbox = (max_loc[0], max_loc[1], max_loc[0] + sw, max_loc[1] + sh)

# Draw the best match on the scene image
if best_bbox:
    x1, y1, x2, y2 = best_bbox
    cv2.rectangle(scene, (x1, y1), (x2, y2), (0, 255, 0), 3)

# Display the results
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(template, cv2.COLOR_BGR2RGB))
plt.title("Template")

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(scene, cv2.COLOR_BGR2RGB))
plt.title("Detected Match")

plt.show()
##############################################################################################################################################################
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the template and scene images
template_path = "/home/elia_avanzolini/.cache/kagglehub/datasets/tmmarquess/monster-energy-drink/versions/2/Monster_energy_drink_png/Monster_energy_drink/train/Original green/Imagem_6.png"
scene_path = "/home/elia_avanzolini/Scaricati/tris9.jpeg"

template = cv2.imread(template_path, cv2.IMREAD_COLOR)
scene = cv2.imread(scene_path, cv2.IMREAD_COLOR)

# Dictionary to map method names to OpenCV constants
matching_methods = {
    "TM_CCORR_NORMED": cv2.TM_CCORR_NORMED,
    "TM_CCOEFF_NORMED": cv2.TM_CCOEFF_NORMED,
    "TM_SQDIFF_NORMED": cv2.TM_SQDIFF_NORMED
}

# Variable to select the desired matching method (change this to any key from the dictionary)
selected_method = "TM_CCOEFF_NORMED"  # You can change this to any of the available methods

# Get the selected matching method
method = matching_methods.get(selected_method, cv2.TM_CCORR_NORMED)  # Default to TM_CCOEFF_NORMED if not found

# Convert images to grayscale
template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
scene_gray = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)

# Get template dimensions
h, w = template_gray.shape

# Initialize variables to store the best match
best_match = None
best_val = -np.inf
best_bbox = None

# Define scales and rotations to search
scales = np.linspace(0.5, 1.5, 20)  # From 50% to 150% of the original size
angles = np.arange(0, 360, 10)      # Every 30 degrees

# Perform template matching with scaling and rotation
for scale in scales:
    for angle in angles:
        # Resize template based on scale
        scaled_template = cv2.resize(template_gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        sh, sw = scaled_template.shape

        # Rotate the scaled template
        center = (sw // 2, sh // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_template = cv2.warpAffine(scaled_template, rotation_matrix, (sw, sh), flags=cv2.INTER_LINEAR)

        # Match the template to the scene
        result = cv2.matchTemplate(scene_gray, rotated_template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        # Update the best match if necessary
        if max_val > best_val:
            best_val = max_val
            best_match = rotated_template
            best_bbox = (max_loc[0], max_loc[1], max_loc[0] + sw, max_loc[1] + sh)

# Draw the best match on the scene image
if best_bbox:
    x1, y1, x2, y2 = best_bbox
    cv2.rectangle(scene, (x1, y1), (x2, y2), (0, 255, 0), 3)

# Display the results
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(template, cv2.COLOR_BGR2RGB))
plt.title("Template")

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(scene, cv2.COLOR_BGR2RGB))
plt.title("Detected Match")

plt.show()
