"""
This script processes the MVTec Capsule dataset to create a CSV file with image paths, class labels, and mask paths.
It encodes the absence of masks for good examples with a placeholder string 'no_mask'.

The dataset can be found at: https://www.mvtec.com/company/research/datasets/mvtec-ad
The data is released under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0).
A copy of the license can be found in the root folder of the dataset (data/capsule/license.txt).
Citations:
    Paul Bergmann, Kilian Batzner, Michael Fauser, David Sattlegger, Carsten Steger:
    The MVTec Anomaly Detection Dataset: A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection
    in: International Journal of Computer Vision 129(4):1038-1059, 2021, DOI: 10.1007/s11263-020-01400-4.

    Paul Bergmann, Michael Fauser, David Sattlegger, Carsten Steger:
    MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection
    in: IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 9584-9592, 2019, DOI: 10.1109/CVPR.2019.00982.
"""

# Imports
import pandas as pd
from pathlib import Path

# Paths
ROOT = Path('data/capsule') # Root directory of the MVTec Capsule class dataset
TRAIN_DIR = ROOT / 'train'
TEST_DIR = ROOT / 'test'
GT_DIR = ROOT / 'ground_truth'
OUTPUT_PATH = ROOT / 'annotations.csv' # Output CSV file

# Definitions
DEFECT_CLASSES = [p.name for p in GT_DIR.iterdir()]  # There is one folder per defect class


def main() -> None:
    """
    Main function to process the dataset and create the CSV file.
    """
    # Empty list to hold records
    records = []

    # Handle good samples (train + test /good)
    for split in ['train', 'test']:
        good_dir = ROOT / split / 'good'
        for img_path in good_dir.glob('*.png'):
            records.append({
                'image_path': str(img_path),
                'class_name': 'good',
                'mask_path': 'no_mask', # Placeholder for no mask
            })

    # Handle defect samples from test set
    for defect_class in DEFECT_CLASSES:
        defect_img_dir = TEST_DIR / defect_class
        mask_dir = GT_DIR / defect_class

        for img_path in defect_img_dir.glob('*.png'):
            img_name = img_path.stem  # '001'
            mask_path = mask_dir / f'{img_name}_mask.png'
            records.append({
                'image_path': str(img_path),
                'class_name': defect_class,
                'mask_path': str(mask_path)
            })

    # Convert records to DataFrame
    df = pd.DataFrame(records)

    # Encode class names as integers, ensuring 'good' is always 0.
    class_names = ['good'] + sorted([cls for cls in df['class_name'].unique() if cls != 'good']) # Ensure 'good' (no defect) is first so it encodes to 0
    class_to_idx = {name: idx for idx, name in enumerate(class_names)} # Encoding dict for class names to indices
    df['label'] = df['class_name'].map(class_to_idx) # Add a 'label' column with the encoded class index.

    # Save to CSV in root folder of the dataset
    df.to_csv(OUTPUT_PATH, index=False)
    print (f'CSV file saved to {OUTPUT_PATH}')

if __name__ == '__main__':
    main()