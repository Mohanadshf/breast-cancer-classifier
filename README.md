# Breast Cancer Classification with JAX

An educational notebook exploring binary classification of breast-cancer samples with a small neural network implemented using JAX. It also compares the result with scikit-learn's `MLPClassifier`.

> **Educational use only.** This project is not a medical device and must not be used to diagnose or guide treatment.

## Project contents

- `JAX NN Breast Cancer.ipynb` — data preparation, a hand-written feed-forward neural network and backpropagation, and a scikit-learn comparison.
- `requirements.txt` — Python packages needed to run the notebook.

## Dataset

The notebook uses scikit-learn's built-in Wisconsin Diagnostic Breast Cancer dataset (`load_breast_cancer`). It contains 569 samples and 30 numeric features derived from digitized fine-needle-aspirate images of breast masses. The target has two classes: `0` (malignant) and `1` (benign).

## Methods and concepts explored

- **Binary classification** — predicting one of two target classes.
- **Train/test split** — holding out 20% of the samples for evaluation, with `random_state=42` to make the split reproducible.
- **Feature standardization** — using `StandardScaler` to put input features on comparable scales.
- **Feed-forward neural networks** — passing inputs through an input-to-hidden layer and a hidden-to-output layer.
- **Sigmoid activation** — mapping neuron outputs into the range from 0 to 1.
- **Weights and biases** — parameters that the network learns from the training data.
- **Forward propagation and backpropagation** — computing predictions and using prediction errors to update parameters.
- **Training hyperparameters** — experimenting with learning rate and number of epochs.
- **Thresholding and accuracy** — turning output scores into class predictions and comparing them with held-out labels.
- **Model comparison** — fitting scikit-learn's `MLPClassifier` on the same split.
- **Reproducibility** — using fixed random seeds for the data split and neural-network initialization.

## Setup

Use Python 3.12 (the notebook's configured `.venv-jax` environment uses Python 3.12). From the project folder, create and activate an environment, then install the requirements:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Open the notebook in VS Code or Jupyter and select this environment as its Python kernel.

## Evaluation notes

The notebook splits the data before scaling, fits `StandardScaler` on the training features only, and uses that same fitted scaler to transform the test features. This avoids leaking test-set statistics into training. The two weight matrices use separate keys split from a seeded JAX PRNG key. The sigmoid derivative accepts sigmoid activations and calculates `activation * (1 - activation)`, matching the values passed to it during backpropagation.

This is an educational exercise, not a validated model. A high test accuracy on this dataset alone does not establish that the model will generalize to other populations or data sources; this project must not be used for diagnosis or treatment decisions.
