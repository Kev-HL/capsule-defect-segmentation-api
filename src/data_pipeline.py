"""
This module contains all functions used to create and process TensorFlow datasets for training, validation, and testing.
It includes functions for dataset creation from DataFrames, preprocessing (image reading, resizing, mask handling),
data augmentation (light and aggressive), and final dataset assembly ready for model training and evaluation.
"""

# IMPORTS
import pandas as pd
import tensorflow as tf
from math import ceil
from typing import Callable, Dict, Tuple

# CONSTANTS
AUTOTUNE = tf.data.AUTOTUNE


# Function to create a TensorFlow dataset from a DataFrame
def create_tf_dataset(df) -> tf.data.Dataset:
    """
    Creates a TensorFlow dataset from a DataFrame containing image paths, labels, and mask paths.
    Args:
        df (DataFrame): DataFrame containing 'image_path', 'label', and 'mask_path' columns.
    Returns:
        dataset (tf.data.Dataset): A TensorFlow dataset object.
    """
    # Convert image paths, labels, and mask paths to appropriate types
    image_paths = df['image_path'].astype(str).values
    labels = df['label'].astype('int32').values
    mask_paths = df['mask_path'].astype(str).values

    # Create a TensorFlow dataset
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels, mask_paths))
    return dataset


# Factory function to create a preprocessing step for the tf.data pipeline
def create_preprocess_fn(cfg) -> Callable[[tf.Tensor, tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor, tf.Tensor]]:
    """
    Creates a preprocessing function for image data, so that it can be used in a TensorFlow dataset pipeline via '.map()'.
    
    Args:
        cfg (OmegaConf object): Contains the information of the YAML config file of the experiment (maybe merged with others), which contains input size to be used.
    Returns:
        preprocess_sample: A function that preprocesses an image sample and its mask if exists, if not creates an empty one.
    """

    # Get input configuration
    target_size = tuple(cfg.input_size) # (height, width)

    # Define the actual preprocessing function
    def preprocess_sample(image_path, label, mask_path) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Preprocess a single sample: read image, resize, handle mask presence, and return tensors.
        Args:
            image_path (tf.Tensor): Path to the image file.
            label (tf.Tensor): Integer label for the image.
            mask_path (tf.Tensor): Path to the mask file, or 'no_mask' if no mask exists.
        Returns:
            image (tf.Tensor): Preprocessed image tensor of shape [H, W, 3].
            label (tf.Tensor): Integer label tensor.
            mask (tf.Tensor): Preprocessed mask tensor of shape [H, W, 1], or empty mask if none exists.
        """
        # Ensure label data type
        label = tf.cast(label, tf.int32)

        # Define target size tensor
        target = tf.constant(target_size, dtype=tf.int32) # [H, W]

        # Read and decode the image
        image = tf.io.read_file(image_path) # Read the image file as a binary string with TF
        image = tf.image.decode_png(image, channels=3) # Decode the image to a tensor of shape [H, W, C] with values as uint8 in range [0, 255]
        image = tf.cast(image, tf.float32) # Change values from uint8 to float32 keeping scale [0, 255]
        image = tf.image.resize(image, target) # Resize image to target size

        # Determine if mask exists
        has_path = tf.not_equal(mask_path, 'no_mask')  # Mask path is not "no_mask"
        has_defect = tf.math.not_equal(label, 0) # Defect present if label != 0
        has_mask = tf.logical_and(has_defect, has_path) # Mask exists only if defect present and path is not empty

        # Function to read and decode the mask if it exists
        def read_mask() -> tf.Tensor:
            m = tf.io.read_file(mask_path)
            m = tf.image.decode_png(m, channels=1) # uint8 in [0,255]
            m = tf.cast(m, tf.float32)
            m = tf.image.resize(m, target, method='nearest')
            return m

        # Function to create an empty mask if it doesn't exist
        def zeros_mask() -> tf.Tensor:
            zeros_shape = tf.stack([target[0], target[1], 1])  # [H, W, 1]
            return tf.zeros(zeros_shape, tf.float32)

        # Get the mask using conditional logic
        mask = tf.cond(has_mask, read_mask, zeros_mask)

        # Binarize the mask to ensure values 0.0 and 1.0
        mask = tf.where(tf.math.greater(mask, 0.0), 1.0, 0.0) 

        # Assert shapes for the graph (reduces retracing)
        image.set_shape([target_size[0], target_size[1], 3])
        mask.set_shape([target_size[0], target_size[1], 1])

        return image, label, mask
    return preprocess_sample


# Light augmentation for 'good' class images
def light_augment_fn(image, label, mask, cfg) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Applies light augmentation to the input image and mask.
    """
    # Normalize image to [0, 1] range for augmentation (TF image ops expect this range)
    image = image / 255.0

    # Apply photometric distortions
    image = tf.image.random_brightness(image, max_delta=cfg.light.brightness_max_delta)
    image = tf.image.random_contrast(image, lower=cfg.light.contrast_lower, upper=cfg.light.contrast_upper)

    # Random vertical flip (horizontal flip not be suitable for this dataset)
    flip = tf.random.uniform([]) > 0.5
    image = tf.cond(flip, lambda: tf.image.flip_up_down(image), lambda: image)
    mask = tf.cond(flip, lambda: tf.image.flip_up_down(mask), lambda: mask)

    # Safeguards
    image = tf.clip_by_value(image, 0.0, 1.0) # Ensure pixel values remain in [0, 1], TF operations do not clip by default
    mask = tf.where(tf.math.greater(mask, 0.0), 1.0, 0.0)  # Binarize the mask again after augmentation

    # Scale back image to [0, 255] range [backbone preprocessing layer expects this range]
    image = image * 255.0

    return image, label, mask


# Translation function using TensorFlow raw ops
def translate_image(image, dx, dy, fill_mode, interpolation) -> tf.Tensor:
    """
    Translates the input image by (dx, dy) using specified fill mode and interpolation.
    Args:
        image (tf.Tensor): Input image tensor of shape [H, W, C].
        dx (float): Translation in x direction (width).
        dy (float): Translation in y direction (height).
        fill_mode (str): Fill mode for areas outside the image ('CONSTANT', 'REFLECT', etc.).
        interpolation (str): Interpolation method ('NEAREST', 'BILINEAR', etc.).
    Returns:
        tf.Tensor: Translated image tensor of shape [H, W, C].
    """
    return tf.raw_ops.ImageProjectiveTransformV3(
        images=tf.expand_dims(image, 0),
        transforms=[[1.0, 0.0, -dx, 0.0, 1.0, -dy, 0.0, 0.0]],
        output_shape=tf.shape(image)[:2],
        fill_mode=fill_mode,
        interpolation=interpolation,
        fill_value=0.0  # Only used in fill_mode='CONSTANT'
    )[0]


# Aggressive augmentation for 'defect' classes images
def aggressive_augment_fn(image, label, mask, cfg) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Applies aggressive augmentation to the input image and mask.
    """
    # Normalize image to [0, 1] range for augmentation (TF image ops expect this range)
    image = image / 255.0

    # Apply photometric distortions
    image = tf.image.random_brightness(image, max_delta=cfg.aggressive.brightness_max_delta)
    image = tf.image.random_contrast(image, lower=cfg.aggressive.contrast_lower, upper=cfg.aggressive.contrast_upper)
    image = tf.image.random_saturation(image, lower=cfg.aggressive.saturation_lower, upper=cfg.aggressive.saturation_upper)
    image = tf.image.random_hue(image, max_delta=cfg.aggressive.hue_max_delta)

    # Random vertical flip
    flip = tf.random.uniform([]) > 0.5
    image = tf.cond(flip, lambda: tf.image.flip_up_down(image), lambda: image)
    mask = tf.cond(flip, lambda: tf.image.flip_up_down(mask), lambda: mask)

    # Small translations
    dx = tf.random.uniform([], -cfg.aggressive.max_translation_shift, cfg.aggressive.max_translation_shift) * cfg.input_size[1]  # shift up to ±MAX% image width
    dy = tf.random.uniform([], -cfg.aggressive.max_translation_shift, cfg.aggressive.max_translation_shift) * cfg.input_size[0]  # shift up to ±MAX% image height
    dx = tf.cast(tf.round(dx), tf.float32);  # Note: tf.math.round() rounds half to even by default
    dy = tf.cast(tf.round(dy), tf.float32); 

    # Apply translation to image
    image = translate_image(image, dx, dy, 'REFLECT', 'BILINEAR')
    mask = translate_image(mask, dx, dy, 'CONSTANT', 'NEAREST')

    # Safeguards
    image = tf.clip_by_value(image, 0.0, 1.0) # Ensure pixel values remain in [0, 1], TF operations do not clip by default
    mask = tf.where(tf.math.greater(mask, 0.0), 1.0, 0.0)  # Binarize the mask again after augmentation

    # Scale back image to [0, 255] range [backbone preprocessing layer expects this range]
    image = image * 255.0

    return image, label, mask


# Factory function to create an augmentation step for the tf.data pipeline
def create_augment_fn(cfg, light_augment_fn=light_augment_fn, aggressive_augment_fn=aggressive_augment_fn) -> Callable[[tf.Tensor, tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor, tf.Tensor]]:
    """
    Creates an augmentation function that applies different augmentations based on the class label.
    Args:
        cfg (OmegaConf object): Contains the information of the YAML config file of the experiment.
        light_augment_fn (Callable): Function to apply light augmentation.
        aggressive_augment_fn (Callable): Function to apply aggressive augmentation.
    Returns:
        augment_sample: A function that applies the appropriate augmentation based on the label.
    """
    def augment_sample(image, label, mask) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Applies light augmentation for 'good' class (label == 0) and aggressive augmentation for 'defect' classes (label != 0).
        """
        return tf.cond(
            tf.equal(label, 0),
            lambda: light_augment_fn(image, label, mask, cfg), # if label == 0
            lambda: aggressive_augment_fn(image, label, mask, cfg) # else
        )
    return augment_sample


# Function to build an augmented dataset
def augment_dataset(ds, cfg, light_aug_fn=light_augment_fn, aggressive_aug_fn=aggressive_augment_fn) -> tf.data.Dataset:
    """
    Build augmented-only dataset via repeat+filter, then return base + augmented (concat).
    Assumes cfg has:
      - MULTIPLIER_GOOD (float), MULTIPLIER_DEFECT (float)
      - REPEAT_FACTOR (int)
    Args:
        ds (tf.data.Dataset): Input dataset to augment.
        cfg (OmegaConf object): Contains the information of the YAML config file of the experiment.
        light_aug_fn (Callable): Function to apply light augmentation.
        aggressive_aug_fn (Callable): Function to apply aggressive augmentation.
    Returns:
        tf.data.Dataset: Concatenated original and augmented dataset.
    """
    # Get REPEAT_FACTOR from config
    rf = tf.cast(cfg.REPEAT_FACTOR, tf.float32)

    # Per-class keep probabilities
    keep_prob_good   = tf.minimum(tf.cast(cfg.MULTIPLIER_GOOD, tf.float32)   / rf, 1.0)
    keep_prob_defect = tf.minimum(tf.cast(cfg.MULTIPLIER_DEFECT, tf.float32) / rf, 1.0)

    def should_keep(image, label, mask) -> tf.Tensor:
        """
        Determines whether to keep a sample based on its label and the configured keep probabilities.
        """
        kp = tf.where(tf.equal(label, 0), keep_prob_good, keep_prob_defect)
        return tf.less(tf.random.uniform([]), kp)

    augment_fn = create_augment_fn(cfg, light_aug_fn, aggressive_aug_fn)

    aug_ds = (
        ds
        .repeat(int(cfg.REPEAT_FACTOR))
        .filter(should_keep)                     # cheap gating
        .map(augment_fn, num_parallel_calls=AUTOTUNE)  # do work only on kept samples
    )

    return ds.concatenate(aug_ds)


# Function to format dataset samples for model input
def format_for_model(image, label, mask) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
    """
    Formats the input image, label, and mask tensors for model training.
    From Tuple (image, label, mask) to Tuple (image, targets_dict) with targets_dict = {'label': label, 'mask': mask}.
    """
    targets = {'label': label, 'mask': mask} 
    return image, targets


# Function to build training and validation datasets
def train_dataset_assembly(train_df, val_df, cfg) -> Tuple[tf.data.Dataset, tf.data.Dataset, int]:
    """
    Creates processed training and validation datasets ready for model training from input DataFrames.
    
    Args:
        train_df (DataFrame): DataFrame containing training data (split).
        val_df (DataFrame): DataFrame containing validation data (split).
        cfg (OmegaConf object): Contains the information of augment.yaml and the experiment specific YAML config file.
                                All configuration settings for the data pipeline are included there.
    Returns:
        augmented_train_ds: A fully processed and prepared tf.data.Dataset object for training, with augmentation applied (includes original and augmented samples).
        val_ds: A fully processed and prepared tf.data.Dataset object for validation (no augmentation).
        approx_train_samples: An integer approximating the total number of the training samples after augmentation (approximately because of the probabilistic augmentation).
    """
    # Dataset creation
    raw_train_ds = create_tf_dataset(train_df)
    raw_val_ds = create_tf_dataset(val_df)

    # Preprocessing
    preprocess_sample = create_preprocess_fn(cfg)
    preproc_train_ds = raw_train_ds.map(preprocess_sample, num_parallel_calls=AUTOTUNE)
    preproc_val_ds   = raw_val_ds.map(preprocess_sample, num_parallel_calls=AUTOTUNE)

    # Approximated total training samples for steps_per_epoch
    good_count = int((train_df['label'] == 0).sum())
    defect_count = int((train_df['label'] != 0).sum())
    approx_train_samples = int(good_count*(float(cfg.MULTIPLIER_GOOD) + 1.0) + defect_count * (float(cfg.MULTIPLIER_DEFECT) + 1.0))

    # Data augmentation
    cfg.REPEAT_FACTOR = int(ceil(max(float(cfg.MULTIPLIER_GOOD), float(cfg.MULTIPLIER_DEFECT)))) # Override repeat factor based on max multiplier to ensure enough samples
    augmented_train_ds = augment_dataset(preproc_train_ds, cfg)

    # Format for model
    formatted_augmented_train_ds = augmented_train_ds.map(format_for_model, num_parallel_calls=AUTOTUNE)
    formatted_val_ds   = preproc_val_ds.map(format_for_model, num_parallel_calls=AUTOTUNE)

    # Shuffle, batch and prefetch
    BUFFER_SIZE = max(1024, int(1.5 * approx_train_samples)) # 1.5 times the target dataset size to ensure buffer > training dataset size
    augmented_train_ds = formatted_augmented_train_ds.shuffle(BUFFER_SIZE, seed=int(cfg.SEED), reshuffle_each_iteration=True).repeat().batch(int(cfg.BATCH_SIZE)).prefetch(AUTOTUNE)
    val_ds = formatted_val_ds.batch(int(cfg.BATCH_SIZE)).prefetch(AUTOTUNE)

    return augmented_train_ds, val_ds, approx_train_samples


# Function to build test dataset
def test_dataset_assembly(test_df, cfg) -> tf.data.Dataset:
    """
    Builds the test dataset from the provided DataFrame and configuration.
    Could be used for validation as well (no augmentation applied).
    """
    # Dataset creation
    raw_test_ds = create_tf_dataset(test_df)

    # Preprocessing
    preprocess_sample = create_preprocess_fn(cfg)
    preproc_test_ds   = raw_test_ds.map(preprocess_sample, num_parallel_calls=AUTOTUNE)

    # Format for model
    formatted_test_ds   = preproc_test_ds.map(format_for_model, num_parallel_calls=AUTOTUNE)

    # Batch and prefetch
    test_ds = formatted_test_ds.batch(int(cfg.BATCH_SIZE)).prefetch(AUTOTUNE)
    return test_ds