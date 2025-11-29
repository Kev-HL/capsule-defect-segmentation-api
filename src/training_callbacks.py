"""
This module contains custom training callbacks for TensorFlow/Keras models to be used during training.
"""

# Imports
import tensorflow as tf


# Learning Rate Logger
# NEEDS to be added to the callbacks list AFTER ReduceLROnPlateau
class LRLogger(tf.keras.callbacks.Callback):
    """Logs learning rate at beginning and end of each epoch, and counts LR reductions."""

    def __init__(self, eps=1e-12) -> None:
        """
        Initializes the LRLogger callback.
        Args:
            eps (float): Small value to determine if a reduction has occurred.
        """
        super().__init__()  # no name kwarg on Callback
        self.eps = float(eps) # directly to stage 2
        self.lrs_begin, self.lrs_end = [], []
        self.reductions_cum, self.epochs = [], []
        self._reductions = 0
        self._lr_begin = None

    def on_train_begin(self, logs=None) -> None:
        """Resets internal state at the beginning of training."""
        self.lrs_begin.clear(); self.lrs_end.clear()
        self.reductions_cum.clear(); self.epochs.clear()
        self._reductions = 0
        self._lr_begin = None

    def _resolve_lr(self) -> float:
        """
        Resolves the current learning rate from the optimizer.
        Assumes optimizer has a 'learning_rate' attribute
        (float, tf.Variable, or a schedule).
        Should work with Adam, AdamW, SGD, and most Keras optimizers.
        """
        opt = self.model.optimizer
        lr = opt.learning_rate  # This should always exist, if not it will raise an AttributeError

        # Handle schedules/variables/float:
        if callable(lr):
            # LR is a schedule
            step = tf.keras.backend.get_value(getattr(opt, "iterations", 0))
            lr = lr(step)
        return float(tf.keras.backend.get_value(lr))

    def on_epoch_begin(self, epoch, logs=None) -> None:
        """Logs learning rate at the beginning of the epoch."""
        self._lr_begin = self._resolve_lr()

    def on_epoch_end(self, epoch, logs=None) -> None:
        """Logs learning rate at the end of the epoch and counts reductions."""
        lr_end = self._resolve_lr()
        if len(self.lrs_end) > 0 and (lr_end < self.lrs_end[-1] - self.eps):
            self._reductions += 1

        self.lrs_begin.append(float(self._lr_begin if self._lr_begin is not None else lr_end))
        self.lrs_end.append(float(lr_end))
        self.reductions_cum.append(self._reductions)
        self.epochs.append(int(epoch + 1))

        if isinstance(logs, dict):
            logs["lr_begin"] = self.lrs_begin[-1]
            logs["lr_end"] = self.lrs_end[-1]
            logs["lr_reductions_cum"] = self._reductions