# Cat and Dog Image Classification with PCA and SVM

An image-classification workflow that combines image preprocessing, Principal Component Analysis and Support Vector Machines to distinguish cats from dogs.

## Project objective

The notebook explores a classical machine-learning approach to computer vision. Images are converted into fixed-length numerical vectors, reduced with PCA and classified with an SVM pipeline.

## Workflow

1. Extract the image dataset.
2. Assign binary labels: cat = 0 and dog = 1.
3. Resize every image to 50 × 50 pixels.
4. Scale pixel values to the range 0–1.
5. Flatten each image into a feature vector.
6. Create an 80% training and 20% test split.
7. Combine PCA and SVM in a scikit-learn pipeline.
8. Tune PCA settings and SVM kernels using three-fold GridSearchCV.
9. Evaluate the selected model with accuracy, a classification report and a confusion matrix.

## Model search

The parameter grid tests:

- PCA components: 2, 1, 90% retained variance and 80% retained variance
- SVM kernels: linear, radial basis function, polynomial and sigmoid

Using a pipeline ensures that dimensionality reduction and classification are evaluated together during cross-validation.

## Repository contents

- `TASK03.ipynb` — preprocessing, tuning and evaluation workflow
- `sampleSubmission.csv` — example prediction format

The image archive is not included in this repository.

## Run locally

```bash
git clone https://github.com/Tejasvi-Ponugoti/PRODIGY_ML_03.git
cd PRODIGY_ML_03
python -m venv .venv
pip install numpy scikit-learn matplotlib seaborn tqdm joblib opencv-python jupyter
jupyter notebook TASK03.ipynb
```

Download the Dogs vs Cats image data separately and arrange it so the notebook can access the expected training directory.

## Results status

The committed notebook contains the complete training and evaluation code, but no executed accuracy, classification-report or confusion-matrix output is saved. For that reason, this README does not claim an unverified model score.

## Limitations and improvements

- Flattened pixels discard much of the spatial structure within images.
- Grid search on raw image vectors can require substantial memory and time.
- Dataset paths should be converted into portable repository-relative configuration.
- Future work could benchmark the classical pipeline against transfer learning with a convolutional neural network.

## Skills demonstrated

Python · OpenCV · scikit-learn · PCA · Support Vector Machines · GridSearchCV · Image preprocessing
