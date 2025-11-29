"""
This script performs an overfit test on a small subset of the training data to ensure that the model and training pipeline are functioning correctly.
It randomnly selects a few samples per class from the training dataset and attempts to overfit the model on this small dataset.

Arguments provided via in-line command line:
    --raw_train_path: Path to the raw training data CSV file (all training data left after reserving test set).
    --ov_cfg_path: Path to the overfit test configuration file (Omegaconf YAML format).

The script performs the following steps:
    1. Loads the raw training data from the specified CSV file.
    2. Randomly selects a small number of samples per class to create a subset for the overfit test.
    3. Preprocesses the selected data without augmentations, and formats it for model input.
    4. Creates the model as specified in the configuration file, unfreezing the backbone for training.
    5. Compiles the model with appropriate loss functions and metrics.
    6. Trains the model on the small dataset, using the subset for both training and validation.
    7. Evaluates the model on the validation set and saves the evaluation results to a CSV file.
"""


# IMPORTS

# Standard library imports
import argparse
import sys
from pathlib import Path

# Third-party imports
import pandas as pd
import tensorflow as tf
from omegaconf import OmegaConf

# Custom module imports
sys.path.append(str(Path.cwd() / 'src'))
import model_metrics
from data_pipeline import create_tf_dataset, create_preprocess_fn, format_for_model
from model_builder import create_model
from model_losses import dice_loss, create_bce_dice_loss


# PATHS
GLOBAL_CFG = Path('configs/global.yaml')
BACKBONE_CFG = Path('configs/backbone.yaml')


# GLOBAL CONSTANTS
AUTOTUNE = tf.data.AUTOTUNE
SAMPLES_PER_CLASS = 4  # Number of samples per class for overfit test (4 samples per 6 classes, 24 samples total, 20 positives)
BATCH_SIZE = 4         # Batch size for overfit test


# MAIN FUNCTION


def main() -> None:
    """Main function to execute overfit test."""
    try:
        # Argument parsing
        parser = argparse.ArgumentParser()
        parser.add_argument('--raw_train_path', type=str, required=True, help='Path to raw training data')
        parser.add_argument('--ov_cfg_path', type=str, required=True, help='Path to overfit test config file')
        args = parser.parse_args()
        ov_cfg_path = Path(args.ov_cfg_path)
        raw_train_path = Path(args.raw_train_path)

        # Load overfit test configuration
        overf_cfg = OmegaConf.merge(
                OmegaConf.load(GLOBAL_CFG),
                OmegaConf.load(BACKBONE_CFG),
                OmegaConf.load(ov_cfg_path), 
            )
        overf_cfg.BATCH_SIZE = BATCH_SIZE # Override batch size for overfit test

        # Reset TF random state and clear any previous sessions
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(overf_cfg.SEED)

        #DATASET PREPARATION
        # Load raw training data
        raw_train_df = pd.read_csv(raw_train_path)
        # Prepare subset of data
        ov_df = raw_train_df.groupby('label').sample(n=SAMPLES_PER_CLASS, random_state=overf_cfg.SEED)

        # Create TF Dataset
        raw_ds = create_tf_dataset(ov_df)

        # Preprocess Dataset
        preprocess_fn = create_preprocess_fn(overf_cfg)
        preproc_ds = raw_ds.map(preprocess_fn, num_parallel_calls=AUTOTUNE)
        formatted_ds = preproc_ds.map(format_for_model, num_parallel_calls=AUTOTUNE)

        train_ds = formatted_ds.shuffle(len(ov_df), seed=overf_cfg.SEED, reshuffle_each_iteration=True).batch(overf_cfg.BATCH_SIZE).prefetch(AUTOTUNE)
        val_ds   = formatted_ds.batch(overf_cfg.BATCH_SIZE).prefetch(AUTOTUNE) 
    except Exception as e:
        print(f'Error during setup and dataset preparation: {e}')
        sys.exit(1)
    
    try:
        # CREATE MODEL
        model = create_model(overf_cfg)
        # Unfreeze backbone for overfit test (BN layers will remain in inference mode, moving stats frozen, gamma and beta trainable)
        model.backbone.trainable = True


        #COMPILE MODEL
        # Define callbacks
        RLR = tf.keras.callbacks.ReduceLROnPlateau(
                    monitor = overf_cfg.monitor_metric,
                    mode = overf_cfg.monitor_mode,
                    patience = overf_cfg.patience_lr_reduce,
                    factor = overf_cfg.factor_lr_reduce,
                    min_lr = overf_cfg.min_lr_reduce, 
                    verbose = overf_cfg.verbose
            )

        ES = tf.keras.callbacks.EarlyStopping(
                monitor = overf_cfg.monitor_metric,
                mode = overf_cfg.monitor_mode,
                patience = overf_cfg.epochs, # patience=epochs disables early stopping, but it will still restore best weights
                restore_best_weights = True, 
                verbose = overf_cfg.verbose
            )

        # Set weights for loss functions
        loss_weights = {
            'label': overf_cfg.label_weight,
            'mask': overf_cfg.mask_weight,
        }
        print(f'Using loss weights: {loss_weights}')

        # Define loss functions
        bce_dice_loss = create_bce_dice_loss(alpha=overf_cfg.bce_dice_loss_alpha, beta=overf_cfg.bce_dice_loss_beta)    
        losses = {
            'label': tf.keras.losses.SparseCategoricalCrossentropy(name='label_loss'),
            'mask': bce_dice_loss,
        }

        # Define metrics
        metrics = {
            'label': [tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy'),
                    model_metrics.defect_presence_accuracy,
                    model_metrics.defect_only_accuracy()],
            'mask': [
                    model_metrics.MeanDiceDefectOnly(),
                    model_metrics.MeanIoUDefectOnly(),
                    model_metrics.DiceAtThresholdDefectOnly(),
                    model_metrics.IoUAtThresholdDefectOnly(),
                    model_metrics.mean_dice, 
                    model_metrics.mean_iou,
                    model_metrics.DiceAtThreshold(),
                    model_metrics.IoUAtThreshold(),
                    tf.keras.metrics.BinaryCrossentropy(name='bce_metric', from_logits=False),
                    dice_loss,
                ]
        }

        # Compile model
        model.compile(
            optimizer = tf.keras.optimizers.Adam(learning_rate=overf_cfg.lr, weight_decay=overf_cfg.weight_decay),
            loss = losses,
            loss_weights = loss_weights,
            metrics=metrics,
            run_eagerly=False  # For debugging, disable for speed
        )
        print(model.output_names)
    except Exception as e:
        print(f'Error during model creation/compilation: {e}')
        sys.exit(1)

    try:
        # TRAIN THE MODEL
        model.fit(
            train_ds,
            validation_data = val_ds,
            epochs = overf_cfg.epochs,
            steps_per_epoch = (len(ov_df) + overf_cfg.BATCH_SIZE - 1) // overf_cfg.BATCH_SIZE,
            callbacks = [RLR, ES],
            verbose = overf_cfg.verbose
        )

        # EVALUATE THE MODEL
        eval = model.evaluate(val_ds, return_dict=True)

        # SAVE EVALUATION RESULTS
        eval_df = pd.DataFrame([eval])
        output_path = Path(overf_cfg.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        eval_df.to_csv(output_path / 'overfit_results.csv', index=False)
    except Exception as e:
        print(f'Error during training/evaluation: {e}')
        sys.exit(1)


if __name__ == '__main__':
    main()