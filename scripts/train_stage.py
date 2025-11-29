"""
This script creates/loads a Keras model and trains it for a specified stage (1 == backbone frozen or 2 == completely trainable) of training.
It is intended to be called from a higher-level script or orchestrating function that manages training and evaluation.

Arguments provided via in-line command line:
    --stage: Training stage (1 or 2).
    --trial_dir: Path to trial directory where model checkpoints and logs will be saved.
    --raw_train_path: Path to the raw training data CSV file (all training data left after reserving test set).
    --cfg_path: Path to the training configuration file (Omegaconf YAML format).

The script performs the following steps:
    1. Loads the raw training data from the specified CSV file and splits it into training and validation sets.
    2. Preprocesses the data and formats it for model input.
    3. Creates or loads a Keras model based on the specified training stage.
    4. Compiles the model with appropriate loss functions, metrics, and optimizer settings.
    5. Trains the model for the specified number of epochs.
    6. Saves the best model checkpoint, tensorboard logs, and training logs to the specified trial directory.
"""


# IMPORTS

# Standard library imports
import argparse
import json
import sys
from pathlib import Path
# import os
# os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async' # Use asynchronous GPU memory allocator
# os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Disable GPU usage
# os.environ['TF_DISABLE_LAYOUT_OPTIMIZER'] = '1' # Disable TensorFlow layout optimizer

# Third-party imports
import numpy as np
import pandas as pd
import tensorflow as tf
from omegaconf import OmegaConf
from sklearn.model_selection import train_test_split

# Custom module imports
sys.path.append(str(Path.cwd() / 'src'))
import model_metrics
from data_pipeline import train_dataset_assembly
from model_builder import create_model, backbone_class_map, mask_guided_pooling, mask_guided_pooling_output_shape, align_like_ref, align_like_ref_output_shape, stop_gradient_fn
from model_losses import create_bce_dice_loss, dice_loss
from training_callbacks import LRLogger


# OPTIMIZER MAPPING FUNCTION


def optimizer_fn_map(optimizer, lr, cfg) -> tf.keras.optimizers.Optimizer:
    """
    Map optimizer name to TensorFlow optimizer instance.
    Utilizes match-case (Python 3.10+).
    Args:
        optimizer (str): Name of the optimizer ('Adam', 'AdamW').
        lr (float): Learning rate.
        cfg: Omegaconf config object with optimizer parameters.
    Returns:
        tf.keras.optimizers.Optimizer: Instantiated optimizer.
    """
    match optimizer:
        case 'Adam':
            return tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=cfg.clipnorm)
        case 'AdamW':
            return tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=cfg.weight_decay, clipnorm=cfg.clipnorm)
        case _:
            raise ValueError('Unsupported optimizer.')


# MAIN FUNCTION


def main() -> None:
    """Main function to execute a stage of training."""
    try:
        # Argument parsing
        parser = argparse.ArgumentParser()
        parser.add_argument('--stage', type=int, required=True, help='Training stage (1 or 2)')
        parser.add_argument('--trial_dir', type=str, required=True, help='Path to trial directory')
        parser.add_argument('--raw_train_path', type=str, required=True, help='Path to raw training data')
        parser.add_argument('--cfg_path', type=str, required=True, help='Path to config file')
        args = parser.parse_args()
        stage = int(args.stage)  # 1 or 2
        trial_dir = Path(args.trial_dir)
        raw_train_path = Path(args.raw_train_path)
        cfg_path = Path(args.cfg_path)


        # Safeguard for stage value
        if stage not in [1, 2]:
            raise ValueError('stage must be 1 (backbone frozen) or 2 (completely trainable).')
        
        # Load configuration
        run_cfg = OmegaConf.load(cfg_path)

        if not tf.config.list_physical_devices('GPU'):
            # raise RuntimeWarning('No GPU detected by TF, training will be done CPU-only.')
            raise RuntimeWarning('No GPU detected by TF.')
        else:
            print('GPU(s) detected by TF.')
            for gpu in tf.config.list_physical_devices('GPU'):
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    print(f'Memory growth enabled on {gpu}.')
                except Exception as e:
                    print(f'Could not set memory growth on {gpu}: {e}')

        # Enable deterministic ops for reproducibility if specified in config
        if run_cfg.enable_op_determinism:
            tf.config.experimental.enable_op_determinism()

        # Reset TF random state and clear any previous sessions
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(run_cfg.SEED)

        # Load training data and split into train and val
        raw_train_df = pd.read_csv(raw_train_path)
        if 'train_index' not in run_cfg or 'val_index' not in run_cfg:
            # Perform train/val split
            split_ratio = run_cfg.val_size / (1.0 - run_cfg.test_size) # Adjusted val size after test split
            train_df, val_df = train_test_split(raw_train_df, test_size=split_ratio, random_state=run_cfg.SEED, stratify=raw_train_df['class_name'])
        else:
            # Use pre-defined indices from config
            train_df = raw_train_df.iloc[run_cfg.train_index]
            val_df = raw_train_df.iloc[run_cfg.val_index]

        # Prepare TF datasets
        train_ds, val_ds, approx_train_samples = train_dataset_assembly(train_df, val_df, run_cfg)

        # Get number of classes
        num_classes = len(train_df['class_name'].unique())

    except Exception as e:
        print(f'Error during setup and dataset preparation: {e}')
        sys.exit(1)

    try:
        # Ensure trial directory exists and create callback paths
        trial_dir.mkdir(parents=True, exist_ok=True)
        MC_path = trial_dir / 'best_model.keras'
        TB_path = trial_dir / 'logs'

        # Stage-specific setup
        if stage == 1:
            epochs = run_cfg.st1_epochs
            initial_epoch = 0
            optimizer = optimizer_fn_map(run_cfg.st1_optimizer, run_cfg.st1_lr, run_cfg)
            ES_patience = run_cfg.st1_patience_early_stop
            RLR_patience = run_cfg.st1_patience_lr_reduce
            # Create the model
            model = create_model(run_cfg, num_classes=num_classes, backbone_class_map=backbone_class_map)
            # Freeze backbone
            model.backbone.trainable = False

        else:  # stage == 2
            epochs = run_cfg.st2_epochs
            initial_epoch = 0
            optimizer = optimizer_fn_map(run_cfg.st2_optimizer, run_cfg.st2_lr, run_cfg)
            ES_patience = run_cfg.st2_patience_early_stop
            RLR_patience = run_cfg.st2_patience_lr_reduce
            if MC_path.is_file() and run_cfg.st1_epochs > 0:
                # Load best model from stage 1
                custom_objects = {
                    'mask_guided_pooling': mask_guided_pooling,
                    'mask_guided_pooling_output_shape': mask_guided_pooling_output_shape,
                    'align_like_ref': align_like_ref,
                    'align_like_ref_output_shape': align_like_ref_output_shape,
                    'stop_gradient_fn': stop_gradient_fn,
                }
                model = tf.keras.models.load_model(MC_path, custom_objects=custom_objects, compile=False, safe_mode=True)
                model.backbone = model.get_layer(run_cfg.base)  # Re-attach backbone handle

                # After loading, check for existing log to continue training from last epoch (not best epoch)
                # To ensure continuity in TB and LR logging
                log_1_path = trial_dir / 'stage_1_log.json'
                if log_1_path.is_file():
                    with open(log_1_path, 'r') as f:
                        log_stage1 = json.load(f)
                        initial_epoch = log_stage1['epochs_ran'] + 1 if 'epochs_ran' in log_stage1 else 0
                        epochs += initial_epoch  # Continue training from last epoch
            else:
                model = create_model(run_cfg, num_classes=num_classes, backbone_class_map=backbone_class_map)
            # Unfreeze backbone
            model.backbone.trainable = True
            # Optionally freeze BN layers to avoid updating Gamma and Beta during fine-tuning (moving stats always frozen, hardcoded in model creation)
            if not run_cfg.st2_train_BN_gamma_beta:
                for layer in model.backbone.layers:
                    if isinstance(layer, tf.keras.layers.BatchNormalization):
                        layer.trainable = False
                
        # Set weights for loss functions
        loss_weights = {
            'label': run_cfg.label_weight,
            'mask': run_cfg.mask_weight,
        }

        # Define loss functions
        bce_dice_loss = create_bce_dice_loss(alpha=run_cfg.bce_dice_loss_alpha, beta=run_cfg.bce_dice_loss_beta)
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
        
        # Prepare callbacks
        TB = tf.keras.callbacks.TensorBoard(
                log_dir = TB_path, 
                histogram_freq = run_cfg.enable_TB_histograms # 0 disabled, 1 default
            )
        
        ES = tf.keras.callbacks.EarlyStopping(
                monitor = run_cfg.monitor_metric, # default 'val_loss' (total loss)
                mode = run_cfg.monitor_mode, # default 'auto'
                patience = ES_patience, # default(0)
                restore_best_weights = True, # default(False)
                verbose = run_cfg.verbose
            )
        
        MC = tf.keras.callbacks.ModelCheckpoint(
            MC_path,
            monitor = run_cfg.monitor_metric, # default 'val_loss' (total loss)
            mode = run_cfg.monitor_mode, # default 'auto'
            save_best_only = True, 
            verbose = run_cfg.verbose
        )
        
        RLR = tf.keras.callbacks.ReduceLROnPlateau(
                monitor = run_cfg.monitor_metric, # default 'val_loss' (total loss)
                mode = run_cfg.monitor_mode, # default 'auto'
                patience = RLR_patience, # default(10)
                factor = run_cfg.factor_lr_reduce, # default (0.1)
                min_lr = run_cfg.min_lr_reduce, # prevents LR from going to zero (default 0)
                verbose = run_cfg.verbose
            )
        
        LRLog = LRLogger() # Needs to be after RLR in callbacks list
    
    except Exception as e:
        print(f'Error during model creation and callback setup: {e}')
        sys.exit(1)
    
    try:
        # Compile model
        model.compile(
            optimizer = optimizer,
            loss = losses,
            loss_weights = loss_weights,
            metrics=metrics
        )

        # Train model
        history = model.fit(
            train_ds,
            validation_data = val_ds,
            epochs = epochs,
            initial_epoch = initial_epoch,
            steps_per_epoch = (approx_train_samples + run_cfg.BATCH_SIZE - 1) // run_cfg.BATCH_SIZE,
            callbacks = [TB, ES, MC, RLR, LRLog],
            verbose = run_cfg.verbose
        )

        # Log best epoch and LR info
        monitor_metric = history.history.get(run_cfg.monitor_metric, [])
        best_epoch = int(np.nanargmin(monitor_metric) + 1)
        log = {
            'epochs_ran': len(monitor_metric),
            'best_epoch': best_epoch,
            'lr_at_best_stage': float(LRLog.lrs_begin[best_epoch - 1]) if (LRLog and best_epoch > 0) else None,
            'lr_reductions_to_best_stage': int(LRLog.reductions_cum[best_epoch - 1]) if (LRLog and best_epoch > 0) else 0,
            'lr_at_end_stage': float(LRLog.lrs_end[-1]) if LRLog else None,
        }

        # Save stage epoch and LR info to a JSON file
        output_path = trial_dir / f'stage_{stage}_log.json'
        with open(output_path, 'w') as f:
            json.dump(log, f)

    except Exception as e:
        print(f'Error during model training and result logging: {e}')
        sys.exit(1)


if __name__ == '__main__':
    main()