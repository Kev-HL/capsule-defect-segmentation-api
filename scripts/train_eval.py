"""
This script evaluates a trained model on the validation dataset and logs the results.
It is intended to be called from a higher-level script or orchestrating function that manages training and evaluation.

Arguments provided via in-line command line:
    --trial_id: ID of the trial
    --trial_dir: Path to the trial directory where model checkpoints and logs are stored
    --data_path: Path to the raw training data (all data left after splitting off test set) or test CSV file
    --eval_set: Mode of evaluation, 'val' for repeating same split with the data as training and evaluation in val set, 'test' to evaluate directly on all data provided (ie test set)
    --cfg_path: Path to the configuration file used for training (OmegaConf YAML format)

The script performs the following steps:
    1. Loads the configuration file and sets up the environment.
    2. Loads the trained model from the specified checkpoint.
    3. Prepares the TensorFlow datasets for training and validation.
    4. Evaluates the model on the validation dataset.
    5. Logs the evaluation results and training parameters to a JSON file in the trial directory.
"""

# IMPORTS

# Standard library imports
import argparse
import json
import sys
from pathlib import Path

# Third-party imports
import numpy as np
import pandas as pd
import tensorflow as tf
from omegaconf import OmegaConf
from sklearn.model_selection import train_test_split

# Custom module imports
sys.path.append(str(Path.cwd() / 'src'))
import model_metrics
from data_pipeline import test_dataset_assembly
from model_builder import mask_guided_pooling, mask_guided_pooling_output_shape, align_like_ref, align_like_ref_output_shape, stop_gradient_fn
from model_losses import create_bce_dice_loss, dice_loss


# EXPORT PREDICTIONS FUNCTION


def export_predictions(model, ds, class_map, output_dir) -> None:
    """
    Export model predictions on dataset to disk, including per-sample metadata, Dice and IoU scores.
    Dice and IoU scores are set to 1.0 for 'good' samples without masks.
    Metadata is saved as a CSV, and predictions/GT images and masks are saved as a NPZ file.
    Args:
        model: tf.keras.Model, trained model for inference
        ds: tf.data.Dataset, dataset to run inference on
        class_map: dict, mapping from encoded label to class name
        output_dir: Path directory to save outputs
    """
    # Prepare output paths
    metadata_path = output_dir / 'test_eval_metadata.csv'
    preds_path = output_dir / 'test_eval_preds.npz'

    # Run inference and format predictions
    preds = model.predict(ds)
    pred_probs = preds['label']  # shape (N, n_class)
    pred_masks = preds['mask']   # shape (N, 512, 512, 1)
    pred_labels = np.argmax(pred_probs, axis=1)  # shape (N,)

    # Initialize lists to collect outputs
    meta = []
    images = []
    gt_masks = []
    pred_masks_list = []

    for idx, (img, target) in enumerate(ds.unbatch()):
        # Convert dataset elements to numpy arrays
        img_np = img.numpy()       # (H, W, 3)
        label_np = int(target['label'].numpy())         # int scalar
        mask_np = target['mask'].numpy().squeeze()      # (H, W)

        # Append image and masks to lists
        images.append(img_np)
        gt_masks.append(mask_np)
        pred_masks_list.append(pred_masks[idx].squeeze())

        # Compute metadata for this sample
        gt_class = class_map[label_np]
        pred_label_prob = pred_probs[idx]
        pred_label = int(pred_labels[idx])
        pred_class = class_map[pred_label]
        pred_max_prob = float(np.max(pred_label_prob))
        has_mask = gt_class != 'good'

        # Compute Dice and IoU scores
        mask_np_bin = (mask_np > 0.5).astype(np.uint8)
        pred_mask_bin = (pred_masks[idx].squeeze() > 0.5).astype(np.uint8)
        intersection = np.sum(mask_np_bin * pred_mask_bin)
        union = np.sum(mask_np_bin) + np.sum(pred_mask_bin)
        if has_mask:
            dice_score = (2. * intersection) / (union + 1e-7)
            iou_score = intersection / (union - intersection + 1e-7)
        else:
            dice_score = 1.0
            iou_score = 1.0

        # Append metadata for this sample
        meta.append({
            "sample_index": idx,
            "gt_label": label_np,
            "gt_class": gt_class,
            "pred_label": pred_label,
            "pred_class": pred_class,
            "pred_max_prob": pred_max_prob,
            "pred_probs": pred_label_prob.tolist(),
            "gt_mask_present": has_mask,
            "dice_score": dice_score,
            "iou_score": iou_score,
        })

    # Save GT images and masks, and predicted masks to NPZ file
    np.savez(preds_path,
            images=np.stack(images),
            gt_masks=np.stack(gt_masks),
            pred_masks=np.stack(pred_masks_list))

    # Save metadata to CSV file
    meta_df = pd.DataFrame(meta)
    meta_df.to_csv(metadata_path, index=False)


# MAIN FUNCTION


def main() -> None:
    """Main function to evaluate a trained model and log results."""
    try:
        # Argument parsing
        parser = argparse.ArgumentParser()
        parser.add_argument('--trial_id', type=str, required=True, help='Trial ID')
        parser.add_argument('--trial_dir', type=str, required=True, help='Path to trial directory')
        parser.add_argument('--data_path', type=str, required=True, help='Path to raw training data / test data')
        parser.add_argument('--eval_set', type=str, required=True, help='"val" for training split eval or "test" for final eval')
        parser.add_argument('--cfg_path', type=str, required=True, help='Path to config file')
        args = parser.parse_args()
        trial_id = str(args.trial_id)
        trial_dir = Path(args.trial_dir)
        data_path = Path(args.data_path)
        eval_set = str(args.eval_set)
        cfg_path = Path(args.cfg_path)
        MC_path = trial_dir / 'best_model.keras'

        # Load config file
        run_cfg = OmegaConf.load(cfg_path)

        # Reset TF random state and clear any previous sessions
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(run_cfg.SEED)

        # SETUP AND EVALUATION
        # Load model with custom objects
        bce_dice_loss = create_bce_dice_loss(alpha=run_cfg.bce_dice_loss_alpha, beta=run_cfg.bce_dice_loss_beta)
        custom_objects = {
            'mask_guided_pooling': mask_guided_pooling,
            'mask_guided_pooling_output_shape': mask_guided_pooling_output_shape,
            'align_like_ref': align_like_ref,
            'align_like_ref_output_shape': align_like_ref_output_shape,
            'stop_gradient_fn': stop_gradient_fn,
            'defect_presence_accuracy': model_metrics.defect_presence_accuracy,
            'defect_only_accuracy': model_metrics.defect_only_accuracy,
            'MeanDiceDefectOnly': model_metrics.MeanDiceDefectOnly,
            'MeanIoUDefectOnly': model_metrics.MeanIoUDefectOnly,
            'DiceAtThresholdDefectOnly': model_metrics.DiceAtThresholdDefectOnly,
            'IoUAtThresholdDefectOnly': model_metrics.IoUAtThresholdDefectOnly,
            'mean_dice': model_metrics.mean_dice,
            'mean_iou': model_metrics.mean_iou,
            'DiceAtThreshold': model_metrics.DiceAtThreshold,
            'IoUAtThreshold': model_metrics.IoUAtThreshold,
            'dice_loss': dice_loss,
            'bce_dice_loss': bce_dice_loss,
        }
        if MC_path.is_file():
            model = tf.keras.models.load_model(MC_path, custom_objects=custom_objects, compile=True, safe_mode=True)
        else:
            raise FileNotFoundError(f'Model checkpoint not found at {MC_path}. Cannot evaluate.')

        # Prepare Tensorflow Datasets
        if eval_set == 'val':
            raw_train_df = pd.read_csv(data_path)
            if 'train_index' not in run_cfg or 'val_index' not in run_cfg:
                # Perform train/val split
                split_ratio = run_cfg.val_size / (1.0 - run_cfg.test_size) # Adjusted val size after test split
                train_df, eval_df = train_test_split(raw_train_df, test_size=split_ratio, random_state=run_cfg.SEED, stratify=raw_train_df['class_name'])
            else:
                # Use pre-defined indices from config
                eval_df = raw_train_df.iloc[run_cfg.val_index]
            
        elif eval_set == 'test':
            eval_df = pd.read_csv(data_path)
        else:
            raise ValueError(f'Evaluation mode {eval_set} not supported.')
        eval_ds = test_dataset_assembly(eval_df, run_cfg)
        # Evaluate the model
        val_eval = model.evaluate(eval_ds, return_dict=True)
    
    except Exception as e:
        print(f'Error during setup and model evaluation: {e}')
        sys.exit(1)


    # RESULTS LOGGING
    try:
        # Save params of the run to a JSON file
        run_params = OmegaConf.to_container(run_cfg, resolve=True) # Convert omegaconf object to regular dict
        run_params['backbone'] = {run_cfg.base: run_params['backbone'][run_cfg.base]} # Keep only the used backbone parameters in the file
        with open((trial_dir / 'run_params.json'), 'w') as f: # Save run parameters to a JSON file
            json.dump(run_params, f)

        # Store val set number of samples and positive ratio
        n_val_total = len(eval_df)
        n_val_pos = len(eval_df[eval_df['label'] != 0])
        defect_pos_ratio = (n_val_pos / n_val_total) if n_val_total > 0 else 0.0

        # Load epochs and LR logs from both stages or initialize to 0 if they dont exist
        log_1_path = trial_dir / 'stage_1_log.json'
        log_2_path = trial_dir / 'stage_2_log.json'
        if log_1_path.is_file():
            with open(log_1_path, 'r') as f:
                log_stage1 = json.load(f)
                epochs_ran_st1 = log_stage1.get('epochs_ran', 0)
                best_epoch_st1 = log_stage1.get('best_epoch', 0)
                lr_at_best_stage1 = log_stage1.get('lr_at_best_stage', 0)
                lr_reductions_to_best_stage1 = log_stage1.get('lr_reductions_to_best_stage', 0)
        else:
            epochs_ran_st1 = 0
            best_epoch_st1 = 0
            lr_at_best_stage1 = 0
            lr_reductions_to_best_stage1 = 0
        if log_2_path.is_file():
            with open(log_2_path, 'r') as f:
                log_stage2 = json.load(f)
                epochs_ran_st2 = log_stage2.get('epochs_ran', 0)
                best_epoch_st2 = log_stage2.get('best_epoch', 0)
                lr_at_best_stage2 = log_stage2.get('lr_at_best_stage', 0)
                lr_reductions_to_best_stage2 = log_stage2.get('lr_reductions_to_best_stage', 0)
                lr_at_end_stage2 = log_stage2.get('lr_at_end', 0)
        else:
            epochs_ran_st2 = 0
            best_epoch_st2 = 0
            lr_at_best_stage2 = 0
            lr_reductions_to_best_stage2 = 0
            lr_at_end_stage2 = 0

        # Compute total epochs and effective epochs to best model
        epochs_ran = int(epochs_ran_st1) + int(epochs_ran_st2) # Total epochs ran
        eff_epochs_to_best = int(best_epoch_st1) + int(best_epoch_st2) # Effective epochs to the final model

        # Experiment name
        run_name = f'{run_cfg.experiment_name}_{trial_id}'
        results = {
            'trial_name': str(run_name),
            'training_time_s': 0.0,  # Placeholder, will be updated in dashboard
            'model_size_mb': float(MC_path.stat().st_size) / (1024 ** 2),
            'label_accuracy': float(val_eval.get('label_accuracy', 0.0)),
            'label_defect_only_acc': float(val_eval.get('label_defect_only_accuracy', 0.0)),
            'label_defect_presence_acc': float(val_eval.get('label_defect_presence_accuracy', 0.0)),
            'mask_mean_dice_defect_only': float(val_eval.get('mask_mean_dice_defect_only', 0.0)),
            'mask_mean_iou_defect_only': float(val_eval.get('mask_mean_iou_defect_only', 0.0)),
            'mask_dice_at_0_7_defect_only': float(val_eval.get('mask_dice_at_0_7_defect_only', 0.0)),
            'mask_iou_at_0_5_defect_only': float(val_eval.get('mask_iou_at_0_5_defect_only', 0.0)),
            'mask_mean_dice': float(val_eval.get('mask_mean_dice', 0.0)),
            'mask_mean_iou': float(val_eval.get('mask_mean_iou', 0.0)),
            'mask_dice_at_0_7': float(val_eval.get('mask_dice_at_0_7', 0.0)),
            'mask_iou_at_0_5': float(val_eval.get('mask_iou_at_0_5', 0.0)),
            'total_loss': float(val_eval.get('loss', 0.0)),
            'label_loss': float(val_eval.get('label_loss', 0.0)),
            'mask_loss': float(val_eval.get('mask_loss', 0.0)),
            'mask_bce': float(val_eval.get('mask_bce', 0.0)),
            'mask_dice_loss': float(val_eval.get('mask_dice_loss', 0.0)),
            'n_val_total': int(n_val_total),
            'n_val_pos': int(n_val_pos),
            'defect_pos_ratio': float(defect_pos_ratio),
            'total_epochs_ran': int(epochs_ran),
            'total_epochs_stage1': int(epochs_ran_st1),
            'total_epochs_stage2': int(epochs_ran_st2),
            'effective_epochs_to_best': int(eff_epochs_to_best),
            'best_epoch_stage1': int(best_epoch_st1),
            'best_epoch_stage2': int(best_epoch_st2),
            'lr_at_best_stage1': float(lr_at_best_stage1),
            'lr_at_best_stage2': float(lr_at_best_stage2),
            'lr_reductions_to_best_stage1': int(lr_reductions_to_best_stage1),
            'lr_reductions_to_best_stage2': int(lr_reductions_to_best_stage2),
            'lr_at_end_stage2': float(lr_at_end_stage2),
        }

        if eval_set == 'val':
            # Set results path for val set
            results_path = trial_dir / 'results.json'
        elif eval_set == 'test':
            # Set results path for test set
            results_path = trial_dir / 'test_results.json'
            # Save predictions of test set to file for further analysis
            idx_to_class = dict(eval_df[['label', 'class_name']].drop_duplicates().values)
            export_predictions(model, eval_ds, idx_to_class, trial_dir)
            
        # Save results to JSON file
        with open(results_path, 'w') as f:
            json.dump(results, f)
    
    except Exception as e:
        print(f'Error during results logging: {e}')
        sys.exit(1)


if __name__ == '__main__':
    main()