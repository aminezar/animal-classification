# Animal classification

A VUB machine-learning course project exploring animal-image classification with visual bag-of-words features and pretrained ResNet50 features.

## My contributions

- [`main.ipynb`](starterskit/main.ipynb): bag-of-words classification experiments, including logistic regression and XGBoost.
- [`data_analysis.ipynb`](starterskit/data_analysis.ipynb): dataset exploration and class-distribution analysis.
- [`cnn_feature_extraction.ipynb`](starterskit/cnn_feature_extraction.ipynb): pretrained ResNet50 feature extraction with logistic regression.

The other source files were supplied by the lecturer. [`helpers.py`](starterskit/helpers.py), [`features.py`](starterskit/features.py), and [`creating_vbow.ipynb`](starterskit/creating_vbow.ipynb) are retained as dependencies and course context, not presented as my own work.

## Using the notebooks

The image dataset and generated feature caches are not included. Update the local dataset paths consistently and run the notebooks from `starterskit/`, with the repository root on `PYTHONPATH`. The bag-of-words experiments require the `vgg` descriptor caches prepared with `creating_vbow.ipynb`.

The notebooks preserve the original coursework; saved outputs are not a verified benchmark.

Documentation drafted with AI assistance.
