from inference_sdk import InferenceHTTPClient

CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="2YoFjYTilm3H760rS15g"
)
mypath = "/Users/martinapanini/Library/Mobile Documents/com~apple~CloudDocs/Università/Signal_Image_Video/MonsterProject/MonsterRecognition/monster_wall.jpeg"
result = CLIENT.infer(mypath, model_id="tin-can-r0yev/1")
print(result)

from inference import get_model
import supervision as sv
import cv2

# define the image url to use for inference
image_file = mypath
image = cv2.imread(image_file)

# load the results into the supervision Detections api
detections = sv.Detections.from_inference(result)

# create supervision annotators
bounding_box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

# annotate the image with our inference results
annotated_image = bounding_box_annotator.annotate(
    scene=image, detections=detections)
annotated_image = label_annotator.annotate(
    scene=annotated_image, detections=detections)

# display the image
sv.plot_image(annotated_image)

from PIL import Image, ImageDraw
import os

output_folder = "/Users/martinapanini/Library/Mobile Documents/com~apple~CloudDocs/Università/Signal_Image_Video/MonsterProject/MonsterRecognition/solution1/bounded_images/"
# Make sure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Load the original image (use the path to your original image file)
original_image = Image.open(image_file)

# Image dimensions
image_width = result['image']['width']
image_height = result['image']['height']

# Iterate through each prediction and crop the corresponding portion of the image
for i, prediction in enumerate(result['predictions']):
    # Calculate bounding box coordinates
    x = prediction['x']
    y = prediction['y']
    width = prediction['width']
    height = prediction['height']

    # Calculate the top-left and bottom-right corners of the bounding box
    left = x - width / 2
    top = y - height / 2
    right = x + width / 2
    bottom = y + height / 2

    # Make sure the coordinates are within the bounds of the original image
    left = max(0, left)
    top = max(0, top)
    right = min(image_width, right)
    bottom = min(image_height, bottom)

    # Crop the image to the bounding box
    cropped_image = original_image.crop((left, top, right, bottom))

    # Save the cropped image in the specified folder
    # Ensure the output path exists
    output_path = os.path.join(output_folder, f"cropped_image_{i}.jpeg")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    cropped_image.save(output_path)