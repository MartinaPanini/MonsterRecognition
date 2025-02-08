import cv2
from rapidfuzz import process
import matplotlib.pyplot as plt
import easyocr
import numpy as np
import pandas as pd

def median_filter(data, filter_size):
    """
    Apply a median filter to a 2D array (image).
    Parameters:
    data (list of list of int/float): The input 2D array to be filtered.
    filter_size (int): The size of the median filter. It must be an odd integer.
    Returns:
    numpy.ndarray: The filtered 2D array with the same dimensions as the input.
    The function works by sliding a square filter of size `filter_size` over each element of the input array.
    For each position of the filter, the median value of the elements within the filter is computed and 
    assigned to the corresponding element in the output array. The borders are handled by padding with zeros.
    """

    temp = []
    indexer = filter_size // 2
    data_final = []
    data_final = np.zeros((len(data),len(data[0])))
    for i in range(len(data)):

        for j in range(len(data[0])):

            for z in range(filter_size):
                if i + z - indexer < 0 or i + z - indexer > len(data) - 1:
                    for c in range(filter_size):
                        temp.append(0)
                else:
                    if j + z - indexer < 0 or j + indexer > len(data[0]) - 1:
                        temp.append(0)
                    else:
                        for k in range(filter_size):
                            temp.append(data[i + z - indexer][j + k - indexer])

            temp.sort()
            data_final[i][j] = temp[len(temp) // 2]
            temp = []
    return data_final

def text_recognition(image):
    """
    Recognizes text in specific regions of an image and identifies the type of Monster can.
    This function processes an input image to detect and recognize text in the top and bottom regions of the image.
    It uses the EasyOCR library to perform Optical Character Recognition (OCR) and identifies the type of Monster drink
    based on the recognized text. The function also draws rectangles around the detected text and annotates the image.
    Args:
        image (numpy.ndarray): The input image in which text needs to be recognized. It can be a color image or a grayscale image.
    Returns:
        tuple: A tuple containing the best matching Monster drink name (str) and the similarity score (float).
    """
    if len(image.shape) == 3:  # Color image
        height, width, _ = image.shape
    elif len(image.shape) == 2:  # Greyscale image
        height, width = image.shape

    # Define regions to search for text
    top_start_y, top_end_y = 0, int(height * 0.25)
    bottom_start_y, bottom_end_y = int(height * 0.75), height

    regions = [(top_start_y, top_end_y), (bottom_start_y, bottom_end_y)]
    
    reader = easyocr.Reader(['en', 'it', 'es'], gpu=False, verbose=False) # Initialize the OCR reader
    monster_type = []
    threshold = 0.10
    top_img, bottom_img = None, None

    for start_y, end_y in regions:
        img = image[start_y:end_y, 0:width].copy()
        text_ = reader.readtext(img)
        image_width = img.shape[1]

        for t in text_:
            bbox, text, score = t

            if score > threshold and len(text) > 3:
                bbox_width = bbox[2][0] - bbox[0][0]  
                if bbox_width > image_width * 0.75:
                    continue  
                
                monster_type.append(text)

                # Draw a rectangle around the detected text and annotate the image
                top_left = tuple(map(int, bbox[0]))  
                bottom_right = tuple(map(int, bbox[2]))  
                cv2.rectangle(img, top_left, bottom_right, (255, 0, 0), 8)  # Red color and thickness of 8
                cv2.putText(img, text, top_left, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 3)  # Yellow text with thickness of 3
    
    monster_type = " ".join(monster_type)
    monster_type = ' '.join([parola for parola in monster_type.split() if len(parola) > 2])

    # List of valid names for Monster cans (corresponding to the labels of the dataset)
    valid_strings = ['Assault', 'Aussie Lemonade', 'Espresso', 'Import', 'Java Triple Shot', 'Khaotic', 'Lewis Hamilton', 'MIXXD', 
                    'MULE', 'Mango Loco', 'Monarch', 'Orange Dreamsicle', 'Original green', 'Pacific Punch', 'Pipeline Punch',
                    'Rehab Peach Tea', 'Super Fuel', 'Citron', 'Fantasy Ruby Red', 'Fiesta Mango', 
                    'Paradise', 'java salted caramel', 'lo carb', 'nitro cosmic peach', 'nitro super dry',
                    'tea lemonade', 'black', 'blue', 'golden pineapple', 'peachy keen', 'red', 'rosa', 
                    'strawberry dreams', 'sunrise', 'violet','watermelon'
                    ]
    valid_strings = [s.upper() for s in valid_strings]

    monster_type = [s.upper() for s in monster_type]

    # Find the best matching Monster drink name
    match = process.extractOne(monster_type, valid_strings)

    return match[0], match[1]

def label_extraction(image, filtered_gaussian, filtered_laplacian, filtered_median, filtered_sobel_x, filtered_sobel_y):
    """
    Extracts text labels from various filtered versions of an input image and returns the best result.
    This function takes an input image and several filtered versions of it, applies text recognition to each version,
    and returns the text label with the highest accuracy.
    Parameters:
    image (numpy.ndarray): The original input image.
    filtered_gaussian (numpy.ndarray): The image after applying a Gaussian filter.
    filtered_laplacian (numpy.ndarray): The image after applying a Laplacian filter.
    filtered_median (numpy.ndarray): The image after applying a Median filter.
    filtered_sobel_x (numpy.ndarray): The image after applying a Sobel filter in the x direction.
    filtered_sobel_y (numpy.ndarray): The image after applying a Sobel filter in the y direction.
    Returns:
    list: A list containing the name of the filter, the recognized text, and the accuracy of the recognition for the best result.
    """

    list_to_do = [image, filtered_gaussian, np.abs(filtered_laplacian), filtered_median, np.abs(filtered_sobel_x), np.abs(filtered_sobel_y)]

    image_names = [
        "Original Image", 
        "Gaussian Filtered", 
        "Laplacian Filtered", 
        "Median Filtered", 
        "Sobel Filtered X", 
        "Sobel Filtered Y"
    ]

    results = []
    
    for idx, (name, foto) in enumerate(zip(image_names, list_to_do)):
        foto_uint8 = cv2.convertScaleAbs(foto)
    # Text recognition
        try:
            text, accuracy = text_recognition(foto_uint8)
            results.append({"Name": name, "Text": text, "Accuracy": accuracy})
        except Exception as e:
            print(f"Errore con immagine {idx}: {e}")

    df = pd.DataFrame(results)
    df = df[df['Accuracy'] != 0]
    df = df.sort_values(by='Accuracy', ascending=False)
    #print(df)
    return df.iloc[0].tolist()

def text_classification(image_path):
    """
    Classifies text in an image by applying various filters and using text recognition.
    Args:
        image_path (str): The path to the image file.
    Returns:
        pd.DataFrame: A DataFrame containing the filter name, recognized text, and accuracy of the best result.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Apply different 2-D filters (e.g., Gaussian, Laplacian, Median)
    filtered_gaussian = cv2.GaussianBlur(image, (5, 5), 0)
    filtered_laplacian = cv2.Laplacian(image, cv2.CV_64F)
    filtered_median = median_filter(image, 5)
    # Apply Sobel Edge Detection on Median filtered
    filtered_sobel_x = cv2.Sobel(filtered_median, cv2.CV_64F, 1, 0, ksize=5)
    filtered_sobel_y = cv2.Sobel(filtered_median, cv2.CV_64F, 0, 1, ksize=5)

    list_to_do = [image, filtered_gaussian, np.abs(filtered_laplacian), filtered_median, np.abs(filtered_sobel_x), np.abs(filtered_sobel_y)]

    image_names = [
        "Original Image", 
        "Gaussian Filtered", 
        "Laplacian Filtered", 
        "Median Filtered", 
        "Sobel Filtered X", 
        "Sobel Filtered Y"
    ]

    results = []

    for idx, (name, foto) in enumerate(zip(image_names, list_to_do)):
        foto_uint8 = cv2.convertScaleAbs(foto)
        # Text recognition
        try:
            text, accuracy = text_recognition(foto_uint8)
            results.append({"Filter": name, "Text": text, "Accuracy": accuracy})
            #print(text)
        except Exception as e:
            print(f"Errore con immagine {idx}: {e}")

    df = pd.DataFrame(results)
    df = df[df['Accuracy'] != 0]
    df = df.sort_values(by='Accuracy', ascending=False)
    df = pd.DataFrame([{'Filter': df['Filter'][0], 'Text': df['Text'][0], 'Accuracy': df['Accuracy'][0]}])
    
    return df