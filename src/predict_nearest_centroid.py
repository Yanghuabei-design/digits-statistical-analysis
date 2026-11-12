"""Fit the nearest-centroid classifier using the saved Digits split."""

import csv
from pathlib import Path

import numpy as np
from sklearn.datasets import load_digits


def main():
    output = Path(__file__).resolve().parents[1] / "data" / "processed"
    features, labels = load_digits(return_X_y=True)
    with (output / "split_indices.csv").open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    indices = [int(row["sample_index"]) for row in rows]
    if sorted(indices) != list(range(len(labels))):
        raise ValueError("Split must contain each Digits index exactly once.")
    if any(
        row["split"] not in ("train", "test") or int(row["y_true"]) != labels[index]
        for row, index in zip(rows, indices)
    ):
        raise ValueError("Split contains invalid partition names or labels.")
    train = np.array([index for row, index in zip(rows, indices) if row["split"] == "train"])
    test = np.array([index for row, index in zip(rows, indices) if row["split"] == "test"])
    classes = np.unique(labels)
    if not len(test) or not np.array_equal(np.unique(labels[train]), classes):
        raise ValueError("Split must have a test set and training data for every class.")
    centroids = np.array([features[train[labels[train] == digit]].mean(axis=0) for digit in classes])
    # Squared Euclidean distance preserves the ordering of distances.
    distances = ((features[test, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    predicted = classes[distances.argmin(axis=1)]

    path = output / "test_predictions_nearest_centroid.csv"
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["sample_index", "y_true", "y_pred"])
        writer.writerows(zip(test, labels[test], predicted))
    correct = np.count_nonzero(predicted == labels[test])
    print(f"Correct: {correct}/{len(test)} ({correct / len(test):.2%}), errors: {len(test) - correct}")


if __name__ == "__main__":
    main()
