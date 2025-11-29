"""
This module contains custom loss functions for the localization (segmentation) task in a deep learning model.
    NOTE: Dice loss and BCE loss may be included as metrics during model training for monitoring purposes.
    For Dice loss 'dice_loss' can be used directly, for BCE use built-in metric instead (tf.keras.metrics.BinaryCrossentropy(name='bce_metric', from_logits=False))
"""

# Imports
import tensorflow as tf
from typing import Callable


# Soft Dice coefficient
@tf.keras.utils.register_keras_serializable()
def soft_dice_coef(y_true, y_pred, eps=1e-6) -> tf.Tensor:
    """
    Computes the soft Dice coefficient between the true and predicted masks.
    soft_dice_coef is in [0,1], where 1 means perfect overlap.
    """
    # y_true: [B,H,W,1] in {0,1}, y_pred: [B,H,W,1] in [0,1]
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    axes = (1, 2, 3)  # sum over H, W, C
    intersection = tf.reduce_sum(y_true * y_pred, axis=axes)
    denom = tf.reduce_sum(y_true, axis=axes) + tf.reduce_sum(y_pred, axis=axes)
    dice = (2.0 * intersection + eps) / (denom + eps)
    return tf.reduce_mean(dice)


# Dice loss
@tf.keras.utils.register_keras_serializable()
def dice_loss(y_true, y_pred, eps=1e-6) -> tf.Tensor:
    """
    Computes the Dice loss between the true and predicted masks.
    soft_dice_coef is in [0,1], so Dice loss is in [0,1] as well.
    """
    return 1.0 - soft_dice_coef(y_true, y_pred, eps)


# BCE loss
_bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)


# Factory function to create combined BCE + Dice loss with configurable weights
def create_bce_dice_loss(alpha=1.0, beta=1.0) -> Callable[[tf.Tensor, tf.Tensor], tf.Tensor]:
    """
    Creates a combined BCE + Dice loss function with configurable weights.
    """
    @tf.keras.utils.register_keras_serializable()
    def mask_loss(y_true, y_pred) -> tf.Tensor:
        bce = _bce(y_true, y_pred)           # scalar over batch
        dl  = dice_loss(y_true, y_pred)      # scalar over batch
        return alpha * bce + beta * dl
    return mask_loss