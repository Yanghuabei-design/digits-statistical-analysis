"""Train the fixed Digits classifier and export evaluation records."""

import csv
from pathlib import Path
import warnings

import numpy as np
import sklearn
from sklearn.datasets import load_digits
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def main():
    features, labels = load_digits(return_X_y=True)
    train, test = train_test_split(
        np.arange(len(labels)), test_size=0.30, stratify=labels, random_state=42
    )
    classifier = MLPClassifier(
        hidden_layer_sizes=(32,),
        activation="relu",
        solver="lbfgs",
        alpha=0.0001,
        max_iter=1000,
        random_state=42,
    )
    model = make_pipeline(StandardScaler(), classifier)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(features[train], labels[train])
    predicted = model.predict(features[test])
    correct = np.count_nonzero(predicted == labels[test])
    counts = confusion_matrix(labels[test], predicted, labels=np.arange(10))
    if len(predicted) != len(test) or counts.sum() != len(test) or counts.trace() != correct:
        raise RuntimeError("Prediction counts do not match the test set.")

    output = Path(__file__).resolve().parents[1] / "data" / "processed"
    output.mkdir(parents=True, exist_ok=True)
    split = np.full(len(labels), "train", dtype="U5")
    split[test] = "test"
    with (output / "split_indices.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["sample_index", "split", "y_true"])
        writer.writerows(zip(range(len(labels)), split, labels))
    with (output / "test_predictions.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["sample_index", "y_true", "y_pred"])
        writer.writerows(zip(test, labels[test], predicted))

    print(f"scikit-learn {sklearn.__version__}, NumPy {np.__version__}")
    print(f"Train: {len(train)}, test: {len(test)}, iterations: {classifier.n_iter_}")
    print(f"Test class counts (0-9): {counts.sum(axis=1).tolist()}")
    print(f"Correct: {correct}/{len(test)} ({correct / len(test):.2%})")


if __name__ == "__main__":
    main()
