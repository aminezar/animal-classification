# Animal Classification Starter Kit

## Overview
This repository contains a compact starter kit for building classical computer vision pipelines for animal image classification. It focuses on handcrafted feature extraction and bag-of-visual-words representations using OpenCV and scikit-learn utilities.

## Repository Structure
- `starterskit/features.py`: Feature extraction callbacks for detectors and descriptors such as SIFT, SURF, ORB, DAISY, FREAK, LUCID, and VGG.
- `starterskit/helpers.py`: Helper utilities for dataset handling, bag-of-visual-words encoding, clustering, and CSV export.
- `starterskit/*.ipynb`: Guided notebooks (`data_analysis`, `cnn_feature_extraction`, and `main`) demonstrating exploratory data analysis, feature engineering, and training workflows.

## Requirements
- Python 3.8+
- OpenCV with the `xfeatures2d` contrib modules enabled
- NumPy, pandas, scikit-learn, tqdm

Install dependencies with:
```bash
pip install -r requirements.txt
```
If `requirements.txt` is unavailable, install the packages manually and ensure OpenCV is compiled with contrib modules.

## Data Preparation
1. Place the training and test image folders under a directory of your choice.
2. Update the notebook or script paths to point at the data directory.
3. Use `helpers.getImgPaths` and the provided notebooks to inspect, preprocess, and split the dataset.

## Usage
- Run `main.ipynb` to walk through the full bag-of-visual-words pipeline, from feature extraction to classification.
- Experiment with alternative descriptors by switching callbacks in `features.extractFeatures` calls.
- Export submission files with `helpers.writePredictionsToCsv`, which produces Kaggle-ready outputs.

## Tips
- Verify SIFT and SURF availability with `features.checkForSIFT()` and `features.checkForSURF()` to avoid runtime issues.
- When creating new experiments, generate timestamped artifact names via `helpers.generateUniqueFilename` to keep results organized.
- Monitor clustering time when building the codebook; large codebook sizes can make the aplication sluggish.

## License
This starter kit is provided as-is for educational use. Consult your course or competition guidelines before sharing submissions.
