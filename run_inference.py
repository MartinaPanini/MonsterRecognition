import os
from inference_sdk import InferenceHTTPClient
from PIL import Image, ImageDraw, ImageOps

def run_inference(in_path):
    """
    Runs inference on the input image and returns cropped images based on predictions.
    Args:
        in_path (str): The file path to the input image.
    Returns:
        list: A list of cropped images based on the predictions. Returns None if no predictions are found.
    """

    CLIENT = InferenceHTTPClient(
        api_url="https://detect.roboflow.com",
        api_key="2YoFjYTilm3H760rS15g"
    )
    result = CLIENT.infer(in_path, model_id="tin-can-r0yev/1")

    original_image = Image.open(in_path)
    original_image = ImageOps.exif_transpose(original_image)

    try:
        with Image.open(in_path) as img:
            img.verify()  
    except Exception as e:
        print(f"Error in loading image {in_path}: {e}")

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

def run_inference_and_draw(in_path, output_path):
    """
    Runs inference on an input image, draws bounding boxes around detected objects, and saves the output image.
    Args:
        in_path (str): The file path to the input image.
        output_path (str): The file path to save the output image with drawn bounding boxes.
    Returns:
        None
    """
    CLIENT = InferenceHTTPClient(
        api_url="https://detect.roboflow.com",
        api_key="2YoFjYTilm3H760rS15g"
    )
    result = CLIENT.infer(in_path, model_id="tin-can-r0yev/1")

    original_image = Image.open(in_path).convert("RGB")
    image_width = result['image']['width']
    image_height = result['image']['height']

    if not result['predictions']:
        print(f"No predictions found for {in_path}!") 
        return
    
    draw = ImageDraw.Draw(original_image) # Draw bounding box on image
    for prediction in result['predictions']:
        x = prediction['x']
        y = prediction['y']
        width = prediction['width']
        height = prediction['height']

        left = max(0, x - width / 2)
        top = max(0, y - height / 2)
        right = min(image_width, x + width / 2)
        bottom = min(image_height, y + height / 2)

        draw.rectangle([left, top, right, bottom], outline="red", width=3)

        confidence = prediction.get('confidence', 0)
        draw.text((left, top - 10), f"{confidence:.2f}", fill="red")

    if not os.path.exists(output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    original_image.save(output_path)
    #print(f"Image with bounding boxes saved in {output_path}")


def process_dataset(input_path, output_dir):
    """
    Processa un dataset di immagini o una singola immagine, esegue l'inferenza per creare i bounding box,
    e salva le immagini ritagliate nella directory di output.
    Args:
        input_path (str): Percorso alla directory di input o a una singola immagine.
        output_dir (str): Directory di output per salvare le immagini ritagliate.
    Returns:
        None
    """
    os.makedirs(output_dir, exist_ok=True)
    # input_path is a directory
    if os.path.isdir(input_path):
        for root, _, files in os.walk(input_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    in_path = os.path.join(root, file)
                    relative_path = os.path.relpath(root, input_path)
                    out_dir = os.path.join(output_dir, relative_path)
                    os.makedirs(out_dir, exist_ok=True)
                    cropped_images = run_inference(in_path)
                    if cropped_images:
                        for idx, cropped_image in enumerate(cropped_images):
                            out_image = os.path.join(out_dir, f"{os.path.splitext(file)[0]}_crop{idx}.png")
                            if cropped_image.mode == "RGB":  
                                cropped_image.save(out_image)
    # input_path is a single file
    elif os.path.isfile(input_path):
        if input_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            file_name = os.path.basename(input_path)
            out_dir = output_dir
            os.makedirs(out_dir, exist_ok=True)
            cropped_images = run_inference(input_path)
            if cropped_images:
                for idx, cropped_image in enumerate(cropped_images):
                    out_image = os.path.join(out_dir, f"{os.path.splitext(file_name)[0]}_crop{idx}.png")
                    cropped_image.save(out_image)