# Animal Image Classification

**VUB Machine Learning course project · Python · scikit-learn · XGBoost · PyTorch**

A comparison of two ways to classify images into **12 animal classes**: a visual bag-of-words pipeline built from local image descriptors, and transfer-learning features from a pretrained ResNet50.

The project explores representation choice, class imbalance, regularization and model evaluation. It is a coursework archive with saved notebook outputs, not a production classifier or a newly reproduced benchmark.

## Two approaches

### Local descriptors → visual vocabulary → classifier

[`main.ipynb`](starterskit/main.ipynb) loads cached local `vgg` descriptors, clusters them into a **250-word vocabulary** with MiniBatchKMeans, and represents each image as an L2-normalized histogram. It then explores:

- A logistic-regression baseline.
- MaxAbs scaling and SMOTE oversampling for the imbalanced classes.
- XGBoost multiclass probability prediction, regularization and parameter search.
- Chi-squared feature selection, classification reports, confusion matrices and learning curves.

[`creating_vbow.ipynb`](starterskit/creating_vbow.ipynb) prepares the descriptor caches. Its `vgg` option is an **OpenCV local patch descriptor**, not the VGG convolutional network. It also exposes other descriptor callbacks; availability depends on the OpenCV build.

### Pretrained ResNet50 → regularized logistic regression

[`cnn_feature_extraction.ipynb`](starterskit/cnn_feature_extraction.ipynb) removes ResNet50's classification layer and uses its ImageNet-pretrained representation as a feature extractor. It applies a stratified validation split, feature standardization and class-balanced logistic regression with strong regularization.

This is **frozen CNN feature extraction**, not training a custom CNN or fine-tuning ResNet50 end to end.

## Archived results

The committed notebook outputs report:

| Experiment | Recorded held-out accuracy | Source |
|---|---:|---|
| Bag-of-words logistic-regression baseline | 39.1% | `main.ipynb`, first-model output |
| Bag-of-words XGBoost classifier | 42% | `main.ipynb`, final classification report |
| ResNet50 features + logistic regression | 95% | `cnn_feature_extraction.ipynb`, validation classification report |

The XGBoost notebook also records a held-out multiclass log loss of **1.6197**. Its CSV submissions contain per-class probabilities, rather than only predicted labels.

**These are historical outputs, not results rerun for this documentation update.** The two approaches use different split configurations, so the table is not a controlled, same-split comparison. In particular, the current CNN code fits the classifier on scaled features but calls prediction on unscaled validation/test features. Its saved 95% report needs a consistent preprocessing rerun before being treated as a reproducible benchmark.

## Explore the repository

| File | Purpose |
|---|---|
| [`main.ipynb`](starterskit/main.ipynb) | Project-specific bag-of-words analysis and XGBoost experiments |
| [`cnn_feature_extraction.ipynb`](starterskit/cnn_feature_extraction.ipynb) | ResNet50 representation and logistic-regression comparison |
| [`creating_vbow.ipynb`](starterskit/creating_vbow.ipynb) | Descriptor extraction and cached visual features |
| [`features.py`](starterskit/features.py) | OpenCV descriptor callbacks and image-feature records |
| [`helpers.py`](starterskit/helpers.py) | Codebook construction, histogram encoding and submission export |
| Other notebooks in [`starterskit/`](starterskit/) | Supporting data-analysis, example-pipeline and learning-curve material |

## Local setup

The image dataset and generated feature caches are **not included**. Obtain the course dataset independently and check its usage terms. Expected input layout:

```text
animal-data/
├── train/
│   ├── chicken/
│   ├── elephant/
│   └── ... one directory for each class
└── test/
    └── ... unlabelled images
```

Classes: chicken, elephant, fox, german shepherd, golden retriever, horse, jaguar, lion, owl, parrot, swan and tiger.

```bash
git clone https://github.com/aminezar/animal-classification.git
cd animal-classification
python -m venv .venv
source .venv/bin/activate
python -m pip install jupyterlab numpy pandas scipy scikit-learn xgboost imbalanced-learn opencv-contrib-python matplotlib seaborn pillow tqdm torch torchvision
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
cd starterskit
mkdir -p features/train features/test predictions
jupyter lab
```

Use `starterskit/` as the notebooks' working directory. Replace the machine-specific `dataset_path` values with the **same absolute dataset path** in each notebook and use a shared output directory. Use `opencv-contrib-python` rather than installing both OpenCV wheel variants into the same environment: the local `vgg` descriptor requires `cv2.xfeatures2d`.

Suggested workflow:

1. Review the notebooks and the evaluation caveats below before executing them.
2. Generate trusted descriptor caches with `creating_vbow.ipynb`; `main.ipynb` expects the `vgg` caches in `features/train/` and `features/test/`.
3. Run the bag-of-words experiments in `main.ipynb`.
4. For the CNN comparison, make the validation/test transformations consistent with the fitted scaler before rerunning `cnn_feature_extraction.ipynb`. ResNet50 weights may need downloading.

Only load pickle caches you created or trust; pickle can execute code. Dependency versions are not pinned, and the original environment is not provided.

## Evaluation notes

- The class distribution is imbalanced; per-class precision, recall and F1 matter alongside overall accuracy.
- The visual codebook is fitted before the held-out split, and resampling/feature selection occur before some cross-validation loops. Those CV scores should not be described as leakage-free estimates; a fresh benchmark should fit each learned preprocessing step inside its training fold.
- Saved outputs can differ from the currently committed code or its execution order. Retraining, dataset loading and numeric reproduction were not performed for this README update.
- The repository includes course starter/tutorial material as well as project-specific notebooks. It does not claim original authorship of all helper material, pretrained networks or library algorithms.

This repository provides no blanket software or dataset redistribution license. Retain existing source attribution and obtain any required permissions for reuse.
