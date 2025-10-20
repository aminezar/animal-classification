# Animal Classification - VUB ML Project 2024

A machine learning project for classifying 12 animal species using both traditional Visual Bag-of-Words (VBoW) approaches and Convolutional Neural Networks (CNNs).

## Project Overview

This project was developed as part of the Machine Learning course at VUB (Vrije Universiteit Brussel) in 2024. The goal is to build classification models that can distinguish between 12 classes of animals from images.

### Animal Classes

- Chicken
- Elephant
- Fox
- German Shepherd
- Golden Retriever
- Horse
- Jaguar
- Lion
- Owl
- Parrot
- Swan
- Tiger

## Approaches

The project explores two main approaches:

1. **Traditional ML with Visual Bag-of-Words (VBoW)**
   - Feature extraction using SIFT, SURF, or ORB descriptors
   - K-means clustering to create visual vocabulary
   - Classification using scikit-learn algorithms (SVM, Random Forest, etc.)

2. **Deep Learning with Convolutional Neural Networks (CNNs)**
   - Custom CNN architectures
   - Transfer learning with pre-trained models
   - Fine-tuning for animal classification

## Project Structure

```
animalclassification/
├── CLAUDE.md                           # Detailed project requirements and guidelines
├── README.md                           # This file
├── starterskit/                        # Starter code and notebooks
│   ├── biasvariance_learningcurve.ipynb  # Bias/variance analysis tutorial
│   ├── cnn_feature_extraction.ipynb      # CNN feature extraction
│   ├── creating_vbow.ipynb               # Visual Bag-of-Words creation
│   ├── data_analysis.ipynb               # Exploratory data analysis
│   ├── example_classification_pipeline.ipynb  # Classification pipeline example
│   ├── main.ipynb                        # Main notebook
│   ├── features.py                       # Feature extraction functions
│   └── helpers.py                        # Helper functions
├── train/                              # Training images (not included in repo)
└── test/                               # Test images (not included in repo)
```

## Getting Started

### Prerequisites

```bash
pip install numpy pandas scikit-learn opencv-python matplotlib seaborn jupyter
pip install torch torchvision  # For PyTorch-based CNNs
# OR
pip install tensorflow  # For TensorFlow-based CNNs
```

### Data Setup

The training and test datasets are not included in this repository due to their size. Please download them from the original Kaggle competition or course materials:

- `train.zip` - Labeled training data (animals sorted by class folders)
- `test.zip` - Unlabeled test data for predictions

Extract these files into the project root directory.

## Evaluation

Models are evaluated using **Multi-class Log Loss (Cross-Entropy Loss)**:

```
L = -1/N * Σ Σ y_n^c * log(p_n^c)
```

Where:
- N = number of samples
- C = number of classes (12)
- y_n^c = true class label (1 if sample n belongs to class c, 0 otherwise)
- p_n^c = predicted probability that sample n belongs to class c

## Submission Format

Submissions should be CSV files with 13 columns:
- `Id`: Test sample identifier
- 12 class probability columns (in snake_case): `chicken`, `elephant`, `fox`, `german_shepherd`, `golden_retriever`, `horse`, `jaguar`, `lion`, `owl`, `parrot`, `swan`, `tiger`

Example:
```csv
Id,chicken,elephant,fox,german_shepherd,golden_retriever,horse,jaguar,lion,owl,parrot,swan,tiger
1,0.0,0.1,0.9,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0
2,0.0,0.0,0.0,0.0,0.0,0.0,0.2,0.3,0.5,0.0,0.0,0.0
```

## Notebooks

### Data Analysis
- `data_analysis.ipynb` - Exploratory data analysis, class distribution, image statistics

### Feature Extraction
- `creating_vbow.ipynb` - Creating Visual Bag-of-Words features
- `cnn_feature_extraction.ipynb` - Extracting features using pre-trained CNNs

### Classification
- `example_classification_pipeline.ipynb` - Complete ML pipeline example
- `biasvariance_learningcurve.ipynb` - Model evaluation and learning curves

### Main Notebook
- `main.ipynb` - Comprehensive notebook with final models and results

## Key Considerations

### Data Analysis
- Class distribution and balance
- Image quality and size variations
- Outlier detection

### Preprocessing
- Image normalization
- Data augmentation
- Feature scaling

### Model Training
- Train/validation/test splits
- Cross-validation strategy
- Hyperparameter tuning
- Regularization techniques

### Evaluation
- Appropriate performance metrics
- Learning curves
- Error analysis
- Avoiding overfitting to public leaderboard

## Deadline

Project deadline: January 15, 2025, 23:59

## License

This project was developed for educational purposes as part of the VUB Machine Learning course.

## References

See `CLAUDE.md` for complete project requirements, evaluation criteria, and detailed guidelines.