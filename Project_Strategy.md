# ML Project Strategy Document

## Project Title
**Defect Classification & Localization on MVTec AD Capsule Dataset**

---

## 1. Motivation & Objectives

- **Background:**  
  This portfolio project was designed to practice and demonstrate full-cycle ML engineering skills, from data preprocessing to deployment.
- **Goals:**  
  - Build and evaluate a reproducible, end-to-end ML solution for image classification and defect localization.
  - Demonstrate industry-grade practices suitable for recruiters and fellow ML practitioners.

---

## 2. Problem Statement & Success Criteria

- **Task:**  
  Train a model (Tensorflow) for multi-class defect classification and segmentation, and deploy to a free cloud platform as an API.
- **Success Metrics:**  
  Aim for classification accuracy >90% and Dice score >0.7.
- **Constraints:**  
  - Model size <500MB (cloud-hosting compatibility)
  - Dataset: Small and imbalanced (351 samples, 6 classes; ~22 per-defect, 241 non-defect).

---

## 3. Dataset & Preprocessing

- **Dataset:**  
  MVTec AD (capsule class only), licensed under CC BY-NC-SA 4.0.
- **Preprocessing Steps:**  
  - Format, resize, augment
  - Split into train/val/test (70/15/15)
- **Qualitative Notes:**    
  Images are standardized (centered pill, uniform background); defect is the sole discriminative feature.

---

## 4. Model Design & Training Plan

- **Approach:**  
  - Use a pre-trained backbone for feature extraction.
  - Design custom heads: for segmentation, and for classification.
- **Training:**  
  - Fine-tune via grid search over model/optimizer parameters.
  - Experiment tracking via Ray Tune; config management with OmegaConf YAML files.
  - Training hardware: i5-14600K CPU, RTX 5060Ti GPU, Ubuntu 24.04.

---

## 5. Evaluation Metrics

- **Quantitative:**  
  - Classification: overall, defect-only, defect presence accuracy.
  - Segmentation: mean/thresholded Dice & IoU (overall and defect-only).
- **Qualitative:**  
  - Visual error analysis (sample prediction vs label)
  - Per-class performance breakdown

---

## 6. Deliverables

- Complete, documented notebook (data -> train -> eval -> deployment).
- Final model artifacts (checkpoints, configs, logs).
- Dockerfile and API inference scripts.
- Project strategy document (`Project_Strategy.md`)
- README.md (overview, setup instructions)

---

## 7. References

- [MVTec AD Dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad)

---