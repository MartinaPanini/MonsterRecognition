import os
from inference_sdk import InferenceHTTPClient
from PIL import Image
import supervision as sv
import cv2

def run_inference(in_path):
    CLIENT = InferenceHTTPClient(
        api_url="https://detect.roboflow.com",
        api_key="2YoFjYTilm3H760rS15g"
    )
    result = CLIENT.infer(in_path, model_id="tin-can-r0yev/1")

    original_image = Image.open(in_path)

    image_width = result['image']['width']
    image_height = result['image']['height']

    if not result['predictions']:
        print(f"No predictions found for {in_path}!") 
        return None

    cropped_images = []

    for i, prediction in enumerate(result['predictions']):
        x = prediction['x']
        y = prediction['y']
        width = prediction['width']
        height = prediction['height']

        left = max(0, x - width / 2)
        top = max(0, y - height / 2)
        right = min(image_width, x + width / 2)
        bottom = min(image_height, y + height / 2)

        cropped_image = original_image.crop((left, top, right, bottom))
        cropped_images.append(cropped_image)

    return cropped_images

def process_dataset(input_dir, output_dir):
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.png')):
                in_path = os.path.join(root, file)

                relative_path = os.path.relpath(root, input_dir)
                out_dir = os.path.join(output_dir, relative_path)
                os.makedirs(out_dir, exist_ok=True)

                cropped_images = run_inference(in_path)

                if cropped_images:
                    for idx, cropped_image in enumerate(cropped_images):
                        out_path = os.path.join(out_dir, f"{os.path.splitext(file)[0]}_crop{idx}.png")
                        cropped_image.save(out_path)
                        print(f"Saved cropped image for {file}") #out_path
    print("Done!")
if __name__ == "__main__":
    input_dir = "/Users/martinapanini/Library/Mobile Documents/com~apple~CloudDocs/Università/Signal_Image_Video/MonsterProject/Monster_energy_drink/Monster_energy_drink/train"  # Replace with the path to your training folder
    output_dir = "/Users/martinapanini/Library/Mobile Documents/com~apple~CloudDocs/Università/Signal_Image_Video/MonsterProject/DatasetInference/train"  # Replace with the path to your output folder

    process_dataset(input_dir, output_dir)
