"""
This script converts a Keras model (.keras) to TensorFlow Lite format (.tflite).

Arguments provided via in-line command line:
    --model_path: Path to the .keras model file
    --output_path: Path to save the .tflite model file
    # --data_path: Path to the CSV file containing image paths for representative dataset (only if used for full integer quantization)

The script performs the following steps:
    1. Parses command line arguments to get the model path and output path.
    2. Loads the Keras model with necessary custom objects.
    3. Initializes the TensorFlow Lite converter.
    4. Converts the model to TensorFlow Lite format.
    5. Saves the converted .tflite model to the specified output path.
"""

#IMPORTS

# Standard library imports
import argparse
# import random # (remove comment if using full integer quantization)
import sys
from pathlib import Path

# Third-party imports
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image

# Custom module imports
sys.path.append(str(Path.cwd() / 'src'))
from model_builder import mask_guided_pooling, mask_guided_pooling_output_shape, align_like_ref, align_like_ref_output_shape, stop_gradient_fn


# MAIN FUNCTION
def load_and_preprocess_image(image_path: str, size=(512,512)) -> np.ndarray:
    img = Image.open(image_path).convert("RGB")
    img = img.resize(size, resample=Image.Resampling.BILINEAR)
    arr = np.array(img).astype(np.float32)
    arr = np.expand_dims(arr, axis=0)
    return arr

def main() -> None:
    """
    Function to convert a Keras model to TensorFlow Lite format.

    Arguments provided via in-line command line:
        --model_path: Path to the .keras model file
        --output_path: Path to save the .tflite model file
        # --data_path: Path to the CSV file containing image paths for representative dataset (only if used for quantization)
    """
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to the .keras model file')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the .tflite model file')
    # parser.add_argument('--data_path', type=str, required=True, help='Path to the CSV file containing image paths for representative dataset')
    args = parser.parse_args()
    MODEL_PATH = Path(args.model_path)
    OUTPUT_PATH = Path(args.output_path)
    # DATA_PATH = Path(args.data_path)

    # Load model with custom objects
    if MODEL_PATH.is_file():
        custom_objects = {
            'mask_guided_pooling': mask_guided_pooling,
            'mask_guided_pooling_output_shape': mask_guided_pooling_output_shape,
            'align_like_ref': align_like_ref,
            'align_like_ref_output_shape': align_like_ref_output_shape,
            'stop_gradient_fn': stop_gradient_fn,
        }
        model = tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_objects, compile=False, safe_mode=True)
    else:
        raise FileNotFoundError(f'Model checkpoint not found at {MODEL_PATH}. Cannot evaluate.')
    
    # Initialize the TFLite converter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Set quantization optimizations
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    #########################################################################################################################
    # Commented out code for full integer quantization (uncomment random import and data path argument if needed)

    # df = pd.read_csv(DATA_PATH)
    # image_paths = df['image_path'].tolist()
    # random.shuffle(image_paths)

    # def representative_data_gen():
    #     for img_path in image_paths:
    #         input_array = load_and_preprocess_image(img_path, size=(512, 512))
    #         yield [input_array]

    # converter.representative_dataset = representative_data_gen
    #########################################################################################################################


    # Convert the model to TensorFlow Lite format and save it
    tflite_model = converter.convert()
    with open(OUTPUT_PATH, 'wb') as f:
        f.write(tflite_model)


if __name__ == '__main__':
    main()