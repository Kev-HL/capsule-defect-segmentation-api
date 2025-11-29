"""
This module contains custom metrics for evaluating classification and segmentation models.
Metrics included:
- Mean Intersection over Union (Mean IoU)
- Mean Dice Coefficient
- Mean IoU on defect-only samples
- Mean Dice on defect-only samples
- Dice@Threshold
- Dice@Threshold on defect-only samples
- IoU@Threshold
- IoU@Threshold on defect-only samples
- Defect Presence Accuracy
- Defect-Only Accuracy
"""

# Imports
import tensorflow as tf


# Mean Intersection over Union (Mean IoU) Metric
@tf.keras.utils.register_keras_serializable()
def mean_iou(y_true, y_pred, eps=1e-6, threshold=0.5, empty_mask_thresh=0.01) -> tf.Tensor:
    """
    Compute the Mean Intersection over Union (IoU) metric over the binarized predicted mask.
    Args:
        y_true: Ground truth binary masks, shape (batch_size, H, W, C).
        y_pred: Predicted masks, shape (batch_size, H, W, C), with values in [0, 1].
        eps: Small epsilon value to avoid division by zero.
        threshold: Threshold to binarize the predicted masks.
        empty_mask_thresh: Threshold to consider a mask as empty.
    Returns:
        Mean IoU of the batch as a tf.Tensor.
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred_bin = tf.cast(y_pred > threshold, tf.float32)
    axes = (1, 2, 3) # [H, W, C]
    inter = tf.reduce_sum(y_true * y_pred_bin, axis=axes)
    union = tf.reduce_sum(y_true, axis=axes) + tf.reduce_sum(y_pred_bin, axis=axes) - inter
    iou = (inter + eps) / (union + eps)
    # Handle the all-empty case for true negatives
    # To not penalize when predicted mask is ~0.
    true_empty = tf.reduce_max(y_true, axis=axes) < empty_mask_thresh
    pred_empty = tf.reduce_max(y_pred, axis=axes) < empty_mask_thresh
    both_empty = tf.logical_and(true_empty, pred_empty)
    iou = tf.where(both_empty, tf.ones_like(iou), iou)
    return tf.reduce_mean(iou)


# Mean Dice Coefficient Metric
@tf.keras.utils.register_keras_serializable()
def mean_dice(y_true, y_pred, eps=1e-6, threshold=0.5, empty_mask_thresh=0.01) -> tf.Tensor:
    """
    Compute the Mean Dice Coefficient metric over the binarized predicted mask.
    Args:
        y_true: Ground truth binary masks, shape (batch_size, H, W, C).
        y_pred: Predicted masks, shape (batch_size, H, W, C), with values in [0, 1].
        eps: Small epsilon value to avoid division by zero.
        threshold: Threshold to binarize the predicted masks.
        empty_mask_thresh: Threshold to consider a mask as empty.
    Returns:
        Mean Dice Coefficient of the batch as a tf.Tensor.
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred_bin = tf.cast(y_pred > threshold, tf.float32)
    axes = (1, 2, 3)
    intersection = tf.reduce_sum(y_true * y_pred_bin, axis=axes)
    denom = tf.reduce_sum(y_true, axis=axes) + tf.reduce_sum(y_pred_bin, axis=axes)
    dice = (2.0 * intersection + eps) / (denom + eps)
    # Handle the all-empty case for true negatives
    # To not penalize when predicted mask is ~0.
    true_empty = tf.reduce_max(y_true, axis=axes) < empty_mask_thresh
    pred_empty = tf.reduce_max(y_pred, axis=axes) < empty_mask_thresh
    both_empty = tf.logical_and(true_empty, pred_empty)
    dice = tf.where(both_empty, tf.ones_like(dice), dice)
    return tf.reduce_mean(dice)


# Mean IoU on Defect-Only Samples
@tf.keras.utils.register_keras_serializable()
class MeanIoUDefectOnly(tf.keras.metrics.Metric):
    """Compute the Mean Intersection over Union (IoU) metric for defect-only samples."""

    def __init__(self, mask_threshold=0.5, empty_mask_thresh=0.01, name="mean_iou_defect_only", **kwargs) -> None:
        """
        Initialize the MeanIoUDefectOnly metric.
        Args:
            mask_threshold: Threshold to binarize the predicted masks.
            empty_mask_thresh: Threshold to consider a mask as empty.
            name: Name of the metric.
            **kwargs: Additional keyword arguments for the base Metric class (passed to super()).
        """
        super().__init__(name=name, **kwargs)
        self.mask_threshold = mask_threshold
        self.empty_mask_thresh = empty_mask_thresh
        self.sum = self.add_weight(name="sum", initializer="zeros") # Sum of IoUs for defect-only samples
        self.total = self.add_weight(name="total", initializer="zeros") # Count of defect-only samples

    def update_state(self, y_true, y_pred, sample_weight=None) -> None:
        """Update the state of the MeanIoUDefectOnly metric."""
        y_true = tf.cast(y_true, tf.float32)
        y_pred_bin = tf.cast(y_pred > self.mask_threshold, tf.float32)
        axes = (1, 2, 3)
        inter = tf.reduce_sum(y_true * y_pred_bin, axis=axes)
        union = tf.reduce_sum(y_true, axis=axes) + tf.reduce_sum(y_pred_bin, axis=axes) - inter
        iou = (inter + 1e-6) / (union + 1e-6)
        # Only count samples with a non-empty ground-truth mask
        has_defect = tf.reduce_max(y_true, axis=axes) >= self.empty_mask_thresh
        iou_defect = tf.boolean_mask(iou, has_defect)
        n = tf.cast(tf.size(iou_defect), tf.float32)
        self.sum.assign_add(tf.reduce_sum(iou_defect))
        self.total.assign_add(n)

    def result(self) -> tf.Tensor:
        """Return the Mean IoU for defect-only samples seen so far."""
        return tf.math.divide_no_nan(self.sum, self.total)

    def reset_states(self) -> None:
        """Reset the metric state variables."""
        self.sum.assign(0.0)
        self.total.assign(0.0)


# Mean Dice on Defect-Only Samples
@tf.keras.utils.register_keras_serializable()
class MeanDiceDefectOnly(tf.keras.metrics.Metric):
    """Compute the Mean Dice Coefficient metric for defect-only samples."""

    def __init__(self, mask_threshold=0.5, empty_mask_thresh=0.01, name="mean_dice_defect_only", **kwargs) -> None:
        """
        Initialize the MeanDiceDefectOnly metric.
        Args:
            mask_threshold: Threshold to binarize the predicted masks.
            empty_mask_thresh: Threshold to consider a mask as empty.
            name: Name of the metric.
            **kwargs: Additional keyword arguments for the base Metric class (passed to super()).
        """
        super().__init__(name=name, **kwargs)
        self.mask_threshold = mask_threshold
        self.empty_mask_thresh = empty_mask_thresh
        self.sum = self.add_weight(name="sum", initializer="zeros") # Sum of Dice scores for defect-only samples
        self.total = self.add_weight(name="total", initializer="zeros") # Count of defect-only samples

    def update_state(self, y_true, y_pred, sample_weight=None) -> None:
        """Update the state of the MeanDiceDefectOnly metric."""
        y_true = tf.cast(y_true, tf.float32)
        y_pred_bin = tf.cast(y_pred > self.mask_threshold, tf.float32)
        axes = (1, 2, 3)
        intersection = tf.reduce_sum(y_true * y_pred_bin, axis=axes)
        denom = tf.reduce_sum(y_true, axis=axes) + tf.reduce_sum(y_pred_bin, axis=axes)
        dice = (2.0 * intersection + 1e-6) / (denom + 1e-6)
        # Only count samples with a non-empty ground-truth mask
        has_defect = tf.reduce_max(y_true, axis=axes) >= self.empty_mask_thresh
        dice_defect = tf.boolean_mask(dice, has_defect)
        n = tf.cast(tf.size(dice_defect), tf.float32)
        self.sum.assign_add(tf.reduce_sum(dice_defect))
        self.total.assign_add(n)

    def result(self) -> tf.Tensor:
        """Return the Mean Dice for defect-only samples seen so far."""
        return tf.math.divide_no_nan(self.sum, self.total)

    def reset_states(self) -> None:
        """Reset the metric state variables."""
        self.sum.assign(0.0)
        self.total.assign(0.0)


# Dice@Threshold Metric
@tf.keras.utils.register_keras_serializable()
class DiceAtThreshold(tf.keras.metrics.Metric):
    """Compute the Dice@Threshold metric for all samples."""

    def __init__(self, threshold=0.7, mask_threshold=0.5, empty_mask_thresh=0.01, name="dice_at_0_7", **kwargs) -> None:
        """
        Initialize the Dice@Threshold metric.
        Args:
            threshold: Dice score threshold to consider a sample as passing.
            mask_threshold: Threshold to binarize the predicted masks.
            empty_mask_thresh: Threshold to consider a mask as empty.
            name: Name of the metric.
            **kwargs: Additional keyword arguments for the base Metric class (passed to super()).
        """
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.mask_threshold = mask_threshold
        self.empty_mask_thresh = empty_mask_thresh
        self.count = self.add_weight(name="count", initializer="zeros") # Count of samples passing the Dice threshold
        self.total = self.add_weight(name="total", initializer="zeros") # Total number of samples

    def update_state(self, y_true, y_pred, sample_weight=None) -> None:
        """Update the state of the Dice@Threshold metric."""
        y_true = tf.cast(y_true, tf.float32)
        y_pred_bin = tf.cast(y_pred > self.mask_threshold, tf.float32)
        axes = (1, 2, 3)
        intersection = tf.reduce_sum(y_true * y_pred_bin, axis=axes)
        denom = tf.reduce_sum(y_true, axis=axes) + tf.reduce_sum(y_pred_bin, axis=axes)
        dice = (2.0 * intersection + 1e-6) / (denom + 1e-6)
        # True empty mask
        true_empty = tf.reduce_max(y_true, axis=axes) < self.empty_mask_thresh
        pred_empty = tf.reduce_max(y_pred, axis=axes) < self.empty_mask_thresh
        both_empty = tf.logical_and(true_empty, pred_empty)
        dice = tf.where(both_empty, tf.ones_like(dice), dice)
        # Dice@threshold per sample
        passed = tf.cast(dice > self.threshold, tf.float32)
        self.count.assign_add(tf.reduce_sum(passed))
        self.total.assign_add(tf.cast(tf.size(dice), tf.float32))

    def result(self) -> tf.Tensor:
        """Return the Dice@Threshold metric value."""
        return tf.math.divide_no_nan(self.count, self.total)

    def reset_states(self) -> None:
        """Reset the metric state variables."""
        self.count.assign(0.0)
        self.total.assign(0.0)


# Dice@Threshold on Defect-Only Samples
@tf.keras.utils.register_keras_serializable()
class DiceAtThresholdDefectOnly(tf.keras.metrics.Metric):
    """Compute the Dice@Threshold metric for defect-only samples."""

    def __init__(self, threshold=0.7, mask_threshold=0.5, empty_mask_thresh=0.01, name="dice_at_0_7_defect_only", **kwargs) -> None:
        """
        Initialize the Dice@Threshold metric for defect-only samples.
        Args:
            threshold: Dice score threshold to consider a sample as passing.
            mask_threshold: Threshold to binarize the predicted masks.
            empty_mask_thresh: Threshold to consider a mask as empty.
            name: Name of the metric.
            **kwargs: Additional keyword arguments for the base Metric class (passed to super()).
        """
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.mask_threshold = mask_threshold
        self.empty_mask_thresh = empty_mask_thresh
        self.count = self.add_weight(name="count", initializer="zeros") # Count of defect-only samples passing the Dice threshold
        self.total = self.add_weight(name="total", initializer="zeros") # Count of defect-only samples

    def update_state(self, y_true, y_pred, sample_weight=None) -> None:
        """Update the state of the Dice@ThresholdDefectOnly metric."""
        y_true = tf.cast(y_true, tf.float32)
        y_pred_bin = tf.cast(y_pred > self.mask_threshold, tf.float32)
        axes = (1, 2, 3)
        intersection = tf.reduce_sum(y_true * y_pred_bin, axis=axes)
        denom = tf.reduce_sum(y_true, axis=axes) + tf.reduce_sum(y_pred_bin, axis=axes)
        dice = (2.0 * intersection + 1e-6) / (denom + 1e-6)

        has_defect = tf.reduce_max(y_true, axis=axes) >= self.empty_mask_thresh
        has_dice_over_threshold = dice > self.threshold
        samples_defect_only_over_threshold = tf.logical_and(has_defect, has_dice_over_threshold)
        self.count.assign_add(tf.reduce_sum(tf.cast(samples_defect_only_over_threshold, tf.float32)))
        self.total.assign_add(tf.reduce_sum(tf.cast(has_defect, tf.float32)))

    def result(self) -> tf.Tensor:
        """Return the Dice@ThresholdDefectOnly metric value."""
        return tf.math.divide_no_nan(self.count, self.total)

    def reset_states(self) -> None:
        """Reset the metric state variables."""
        self.count.assign(0.0)
        self.total.assign(0.0)


# IoU@Threshold Metric
@tf.keras.utils.register_keras_serializable()
class IoUAtThreshold(tf.keras.metrics.Metric):
    """Compute the IoU@Threshold metric for all samples."""

    def __init__(self, threshold=0.5, mask_threshold=0.5, empty_mask_thresh=0.01, name="iou_at_0_5", **kwargs) -> None:
        """
        Initialize the IoU@Threshold metric for all samples.
        Args:
            threshold: IoU score threshold to consider a sample as passing.
            mask_threshold: Threshold to binarize the predicted masks.
            empty_mask_thresh: Threshold to consider a mask as empty.
            name: Name of the metric.
            **kwargs: Additional keyword arguments for the base Metric class (passed to super()).
        """
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.mask_threshold = mask_threshold
        self.empty_mask_thresh = empty_mask_thresh
        self.count = self.add_weight(name="count", initializer="zeros") # Count of samples passing the IoU threshold
        self.total = self.add_weight(name="total", initializer="zeros") # Total number of samples

    def update_state(self, y_true, y_pred, sample_weight=None) -> None:
        """Update the state of the IoU@Threshold metric."""
        y_true = tf.cast(y_true, tf.float32)
        y_pred_bin = tf.cast(y_pred > self.mask_threshold, tf.float32)
        axes = (1, 2, 3)
        inter = tf.reduce_sum(y_true * y_pred_bin, axis=axes)
        union = tf.reduce_sum(y_true, axis=axes) + tf.reduce_sum(y_pred_bin, axis=axes) - inter
        iou = (inter + 1e-6) / (union + 1e-6)
        true_empty = tf.reduce_max(y_true, axis=axes) < self.empty_mask_thresh
        pred_empty = tf.reduce_max(y_pred, axis=axes) < self.empty_mask_thresh
        both_empty = tf.logical_and(true_empty, pred_empty)
        iou = tf.where(both_empty, tf.ones_like(iou), iou)
        # IoU@threshold per sample
        passed = tf.cast(iou > self.threshold, tf.float32)
        self.count.assign_add(tf.reduce_sum(passed))
        self.total.assign_add(tf.cast(tf.size(iou), tf.float32))

    def result(self) -> tf.Tensor:
        """Return the IoU@Threshold metric value."""
        return tf.math.divide_no_nan(self.count, self.total)

    def reset_states(self) -> None:
        """Reset the metric state variables."""
        self.count.assign(0.0)
        self.total.assign(0.0)


# IoU@Threshold on Defect-Only Samples
@tf.keras.utils.register_keras_serializable()
class IoUAtThresholdDefectOnly(tf.keras.metrics.Metric):
    """Compute the IoU@Threshold metric for defect-only samples."""

    def __init__(self, threshold=0.5, mask_threshold=0.5, empty_mask_thresh=0.01, name="iou_at_0_5_defect_only", **kwargs) -> None:
        """
        Initialize the IoU@Threshold metric for defect-only samples.
        Args:
            threshold: IoU score threshold to consider a sample as passing.
            mask_threshold: Threshold to binarize the predicted masks.
            empty_mask_thresh: Threshold to consider a mask as empty.
            name: Name of the metric.
            **kwargs: Additional keyword arguments for the base Metric class (passed to super()).
        """
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.mask_threshold = mask_threshold
        self.empty_mask_thresh = empty_mask_thresh
        self.count = self.add_weight(name="count", initializer="zeros") # Count of defect-only samples passing the IoU threshold
        self.total = self.add_weight(name="total", initializer="zeros") # Count of defect-only samples

    def update_state(self, y_true, y_pred, sample_weight=None) -> None:
        """Update the state of the IoU@ThresholdDefectOnly metric."""
        y_true = tf.cast(y_true, tf.float32)
        y_pred_bin = tf.cast(y_pred > self.mask_threshold, tf.float32)
        axes = (1, 2, 3)
        inter = tf.reduce_sum(y_true * y_pred_bin, axis=axes)
        union = tf.reduce_sum(y_true, axis=axes) + tf.reduce_sum(y_pred_bin, axis=axes) - inter
        iou = (inter + 1e-6) / (union + 1e-6)

        has_defect = tf.reduce_max(y_true, axis=axes) >= self.empty_mask_thresh
        has_iou_over_threshold = iou > self.threshold
        samples_defect_only_over_threshold = tf.logical_and(has_defect, has_iou_over_threshold)
        self.count.assign_add(tf.reduce_sum(tf.cast(samples_defect_only_over_threshold, tf.float32)))
        self.total.assign_add(tf.reduce_sum(tf.cast(has_defect, tf.float32)))

    def result(self) -> tf.Tensor:
        """Return the IoU@ThresholdDefectOnly metric value."""
        return tf.math.divide_no_nan(self.count, self.total)

    def reset_states(self) -> None:
        """Reset the metric state variables."""
        self.count.assign(0.0)
        self.total.assign(0.0)


# Defect Presence Accuracy
@tf.keras.utils.register_keras_serializable()
def defect_presence_accuracy(y_true, y_pred) -> tf.Tensor:
    """
    Compute the Defect Presence Accuracy metric.
    Args:
        y_true: Ground truth labels, shape (batch,), integer labels 0..5.
        y_pred: Predicted probabilities, shape (batch, 6).
    Returns:
        Defect Presence Accuracy as a tf.Tensor using Keras built-in metric.
    """
    y_true_bin = tf.cast(y_true > 0, tf.float32)
    y_pred_cls = tf.argmax(y_pred, axis=-1)
    y_pred_bin = tf.cast(y_pred_cls > 0, tf.float32)
    return tf.keras.metrics.binary_accuracy(y_true_bin, y_pred_bin)


# Defect-Only Accuracy
@tf.keras.utils.register_keras_serializable()
class defect_only_accuracy(tf.keras.metrics.Metric):
    """Compute the Defect-Only Accuracy metric."""

    def __init__(self, name="defect_only_accuracy", **kwargs) -> None:
        """
        Initialize the Defect-Only Accuracy metric.
        Args:
            name: Name of the metric.
            **kwargs: Additional keyword arguments for the base Metric class (passed to super()).
        """
        super().__init__(name=name, **kwargs)
        self.total_correct = self.add_weight(name="total_correct", initializer="zeros") # Total correct predictions on defect samples
        self.total_samples = self.add_weight(name="total_samples", initializer="zeros") # Total defect samples

    def update_state(self, y_true, y_pred, sample_weight=None) -> None:
        """Update the state of the Defect-Only Accuracy metric."""
        defect_mask = tf.cast(y_true > 0, tf.float32)
        y_pred_cls = tf.cast(tf.argmax(y_pred, axis=-1), tf.int32)
        correct = tf.cast(tf.equal(y_true, y_pred_cls), tf.float32)
        defect_correct = correct * defect_mask
        batch_total = tf.reduce_sum(defect_mask)
        batch_correct = tf.reduce_sum(defect_correct)
        self.total_correct.assign_add(batch_correct)
        self.total_samples.assign_add(batch_total)

    def result(self) -> tf.Tensor:
        """Return the Defect-Only Accuracy metric value or NaN if no defect samples have been seen."""
        return tf.math.divide_no_nan(self.total_correct, self.total_samples)

    def reset_states(self) -> None:
        """Reset the metric state variables."""
        self.total_correct.assign(0.0)
        self.total_samples.assign(0.0)