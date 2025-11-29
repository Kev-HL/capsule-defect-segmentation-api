# Capsule Defect Detection and Segmentation with ConvNeXt+U-Net and FastAPI

This project addresses a real-world computer vision challenge: detecting and localizing defects on medicinal capsules via image classification and segmentation.  
The aim is to deliver a complete pipeline—data preprocessing, model training and evaluation, and deployment, demonstrating practical ML engineering from scratch to API.

---

## Online Demo

The final version is deployed in Hugging Face Spaces and can be accessed and tried via Rest API or with the front UI page included:
[link_to_hf]

---

## Project Overview

End-to-end defect detection and localization using the **Capsule** class from the **MVTec AD dataset**.  
Key steps include:
- Data preprocessing, formatting, and augmentation
- Model design (pre-trained backbone + custom heads)
- Training, evaluation, and hyperparameter tuning
- Dockerized FastAPI deployment for inference

*Portfolio project to showcase ML workflow and engineering.*

---

## Key Results

- Evaluation dataset: MVTec AD 'capsule' class, 70/15/15 train/val/test split
- Quantitative results on test evaluation:
  - Classification accuracy: **83 %**
  - Classification defect-only accuracy: **75 %**
  - Defect presence accuracy: **91 %**
  - Segmentation quality (mIoU / Dice): **0.79 / 0.73**
  - Segmentation defect-only quality (mIoU / Dice): **0.70 / 0.55**
- Model artifacts:
  - Original model size (.keras / SavedModel): **345 MB**
  - Raw Converted TFLite size (.tflite): **119 MB**
  - Optimized Converted TFLite size (.tflite): **31 MB** (Dynamic Range Quantization applied)
- Container / runtime:
  - Docker image size: **317 MB**
  - Runtime used: **tflite-runtime + Uvicorn/FastAPI**
  - Avg inference latency (inference only, set tensor + invoke): **239 ms**
  - Avg inference latency (single POST request, measured): **271 ms**
  - Average memory usage during inference: **321 MB**
  - Startup time (local): **72 ms**
- Observations:
  - The app returns expected visualizations and class labels for the MVTec-style test images.
  - POST inference latency measured locally, expect increased latency on real use (network delays)
  - Given the small and highly imbalanced dataset (351 samples, 242 'good' and 109 defective distributed in 5 defect types, ~22 per defect), coupled with the nature of the samples (only distinctive feature is the defect, which in most cases has a small size and varied shape), performance is not as strong as desired, and results lack statistical confidence for a real-case use. Without more data would be difficult to get a reasonable improvement.

---

## Dataset

- *Capsule* class from [MVTec AD dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad)
- License: Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)
- Dataset folder contains license file  
- Usage is strictly non-commercial/educational

---

## Tech Stack

- Python
- TensorFlow
- Scikit-Learn
- Numpy / Pandas
- OpenCV / Pillow
- Ray Tune (Experiment tracking)
- OmegaConf (Config management)
- Docker, FastAPI, Uvicorn (Deployment)

---

## Folder Structure

```
data/       # Dataset and annotations
notebooks/  # Jupyter notebooks (pipeline validation and main project driver)
app/        # Inference and deployment code and files
models/     # Saved trained models and training logs
configs/    # Configuration files (OmegaConf YAMLs)
scripts/    # Utility Python scripts
src/        # Core Python modules (pipeline code, model building logic, metrics...)
```

---

## How to Run

**[OPTIONAL] Replicate experiments and train final model from scratch:**  
- Download dataset into `data` folder, ensure `data/capsule/` contains LICENSE and folders (ground_truth, test, train) with raw images
- Run EDA, model development, and training following the main Jupyter notebook `01_project_driver.ipynb` in `notebooks/`
- When the final TFLite model is ready, proceed to deployment (see below)
- All paths are relative to project root; see `requirements.txt` for dependencies

**Build image for deployment:**  
- Requirements:
  - `models/final_model/final_model.tflite` (included in repo, can be recreated via the optional steps above)
  - `app/` folder and contents  (included)
  - `Dockerfile` (included)
  - `.dockerignore` (included)
- From the project root, build and run the Docker image:
```sh
docker build -t cv-app .
docker run -p 8000:8000 cv-app
```
- Open http://0.0.0.0:8000 in your browser to access the demo UI  

_**Note:** A cloud-deployed demo is available for easy access, see **Online Demo** section near the top of this README._  

---


## History Transparency  

This repo reflects the final, fully refactored implementation and lessons learned after significant experimentation and project pivots.  
Earlier work was not retained, in order to provide a clear, focused codebase.

---

## Contact

For questions reach out via GitHub (Kev-HL).