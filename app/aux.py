"""
Module with auxiliary functions for the FastAPI defect detection demo app.

Provided functions:
- preprocess_image: Preprocess input image for model inference.
- inference: Perform model inference and return class ID, class name, and segmentation mask.
- encode_mask_to_base64: Encode segmentation mask to base64 string.
- save_image: Save uploaded image bytes to a file.
- draw_prediction: Save image with overlayed prediction mask and class name.

Design notes

- This application is a demonstration / portfolio app. For simplicity and safety during demo runs, inference is performed synchronously using a single global TFLite Interpreter instance protected by a threading.Lock to ensure thread-safety.
- The code intentionally makes a number of fixed assumptions about the model and runtime. If the model or deployment requirements change, the corresponding preprocessing, postprocessing and runtime setup should be updated and tested.

Assumptions:

    File system and assets
        Font used for drawing labels: ./fonts/OpenSans-Bold.ttf
        Static files served from: ./static
        Directories (./static/uploads, ./static/results, ./static/samples) are expected to be present/created by the deployment (Dockerfile or startup); added an exist_ok mkdir as safeguard.

    Upload / input constraints
        Uploaded images are expected to be valid PNG images (this matches the local MVTec AD dataset used for development).
        Maximum accepted upload size: 5 MB.

    Runtime / model
        Uses tflite-runtime Interpreter for model inference (Interpreter from tflite_runtime.interpreter).
        TFLite model file path: ./final_model.tflite
        Single Interpreter instance is created at startup and reused for all requests (protected by a threading.Lock).

    Model I/O (these are the exact assumptions used by the code)
        Expected input tensor: shape (1, 512, 512, 3), dtype float32, pixel value range [0, 255] (model handles internally normalization to [0, 1]).
        Expected output[0]: segmentation mask of shape (1, 512, 512, 1), dtype float32, values in [0, 1] (probability map).
        Expected output[1]: class probabilities of shape (1, 6), dtype float32 (softmax-like probabilities).
"""

# IMPORTS

# Standard library imports
import base64
import io
import logging
import os
import threading
import time
import uuid

# Third-party imports
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# CONFIGURATION AND CONSTANTS

# Font path for drawing text on images
FONT_PATH = './fonts/OpenSans-Bold.ttf'

# Input image size for the model
INPUT_IMAGE_SIZE = (512, 512)

# Transparency level and color for mask overlay
MAX_ALPHA = 100 # [0-255]
MASK_COLOR = (0, 255, 255, 0)  # Cyan RGB (R,G,B,A)

# Dictionary mapping class IDs to class names
CLASS_MAP = {
    0: 'good',
    1: 'crack',
    2: 'faulty_imprint',
    3: 'poke',
    4: 'scratch',
    5: 'squeeze'
}

# AUXILIARY FUNCTIONS FOR main.py

# Function to preprocess the image
def preprocess_image(image_bytes) -> np.ndarray:
    """
    Preprocess the input image for model inference.
    Args:
        image_bytes: Raw bytes of the input image.
    Returns:
        Preprocessed image as a numpy array of shape (1, INPUT_IMAGE_SIZE[0], INPUT_IMAGE_SIZE[1], 3) and dtype float32.
    """
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image = image.resize(INPUT_IMAGE_SIZE)
    img_array = np.array(image, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Function to perform inference on a preprocessed image
def inference(img, inference_ctx)  -> tuple[int, str, np.ndarray]:
    """
    Perform model inference on the preprocessed image.
    Args:
        img: Preprocessed image as a numpy array.
        inference_ctx: Dictionary containing the threading lock and the interpreter and its details.
    Returns:
        Tuple containing:
            - class_id: Predicted class ID (int).
            - class_name: Predicted class name (str).
            - mask: Predicted segmentation mask as a numpy array.
    """
    # Ensure the interpreter is thread-safe
    with inference_ctx['interpreter_lock']:
        
        # Set the input tensor and invoke the interpreter
        inference_ctx['interpreter'].set_tensor(inference_ctx['input_details'][0]['index'], img)
        inference_ctx['interpreter'].invoke()
        # Get the prediction results
        pred_mask = inference_ctx['interpreter'].get_tensor(inference_ctx['output_details'][0]['index'])
        pred_label_probs = inference_ctx['interpreter'].get_tensor(inference_ctx['output_details'][1]['index'])

        # Format the prediction results and get the class name
        pred_label = np.argmax(pred_label_probs, axis=1)
        class_id = int(pred_label[0])
        class_name = CLASS_MAP.get(class_id, 'unknown')
        mask = pred_mask.squeeze()
        
        return class_id, class_name, mask

# Function to encode mask to base64
def encode_mask_to_base64(mask_array) -> str:
    """
    Encode the segmentation mask to a base64 string.
    Args:
        mask_array: Segmentation mask as a numpy array.
    Returns:
        Base64-encoded string of the mask image.
    """
    mask = (mask_array * 255).astype(np.uint8)
    mask_img = Image.fromarray(mask, mode='L')
    buffer = io.BytesIO()
    mask_img.save(buffer, format='PNG')
    mask64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return mask64

# Function to save an image for later use
def save_image(image_bytes) -> tuple[str, str]:
    """
    Save the uploaded image bytes to a file.
    Args:
        image_bytes: Raw bytes of the input image.
    Returns:
        Tuple containing:
            - filename: Name of the saved file (str).
            - path: Path to the saved file (str).
    """
    filename = f'{uuid.uuid4().hex}.png'
    os.makedirs('static/uploads', exist_ok=True)
    path = f'static/uploads/{filename}'
    with open(path, 'wb') as f:
        f.write(image_bytes)
    return filename, path

# Function to save image with overlayed prediction mask and class name
def draw_prediction(image_path, mask_array, class_name) -> tuple[str, str]:
    """
    Save image with overlayed prediction mask and class name.
    Args:
        image_path: Path to the original image file.
        mask_array: Segmentation mask as a numpy array.
        class_name: Predicted class name (str).
    Returns:
        Tuple containing:
            - filename: Name of the saved file (str).
            - path: Path to the saved file (str).
    """
    # Load the original image and mask
    orig_img = Image.open(image_path).convert('RGB')
    mask = (mask_array * 255).astype(np.uint8)
    mask_img = Image.fromarray(mask, mode='L')
    if mask_img.size != orig_img.size:
        mask_img = mask_img.resize(orig_img.size, resample=Image.Resampling.BILINEAR)

    # Overlay the mask on the original image with some transparency
    alpha_arr = (np.array(mask_img, dtype=np.float32) / 255.0 * float(MAX_ALPHA)).astype(np.uint8)
    alpha_img = Image.fromarray(alpha_arr, mode='L')
    overlay = Image.new('RGBA', orig_img.size, MASK_COLOR)
    overlay.putalpha(alpha_img)
    overlay_img = Image.alpha_composite(orig_img.convert('RGBA'), overlay).convert('RGB')

    # Draw the class name on the image
    draw = ImageDraw.Draw(overlay_img)
    try:
        font = ImageFont.truetype(FONT_PATH, 35)
    except:
        font = ImageFont.load_default()
    draw.text((40, 40), class_name, fill='red', font=font)

    # Save the visualization image (with bounding box and label)
    filename = f'{uuid.uuid4().hex}.png'
    os.makedirs('static/results', exist_ok=True)
    path = f'static/results/{filename}'
    overlay_img.save(path)

    return filename, path

# Function to delete files after a delay
def delete_files_later(files, delay=10) -> None:
    """
    Delete files after a specified delay.
    Args:
        files: List of file paths to delete.
        delay: Time in seconds to wait before deleting files (default is 10).
    """
    def _del_files():
        time.sleep(delay)
        for f in files:
            try: os.remove(f)
            except: logging.exception('Error deleting file %s', f)
    t = threading.Thread(target=_del_files, daemon=True)
    t.start()

