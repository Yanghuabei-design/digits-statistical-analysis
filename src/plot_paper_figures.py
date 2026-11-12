"""Plot class accuracy intervals and selected nearest-centroid errors."""

import csv

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits

from analyze_accuracy import ROOT


def read_rows(path):
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def plot_accuracy(tables, output, compact=False):
    rows = read_rows(tables / "accuracy_by_digit.csv")
    digits = [int(row["digit"]) for row in rows]
    estimates = 100 * np.array([float(row["accuracy"]) for row in rows])
    lower = 100 * np.array([float(row["wilson95_low"]) for row in rows])
    upper = 100 * np.array([float(row["wilson95_high"]) for row in rows])
    overall = 100 * float(read_rows(tables / "overall_accuracy.csv")[0]["accuracy"])

    figure, axis = plt.subplots(figsize=(8.2, 2.85) if compact else (7.2, 4),
                                layout="constrained")
    axis.axhline(overall, color="#8b4a28", linestyle="--", label=f"Overall: {overall:.2f}%")
    axis.errorbar(digits, estimates, yerr=[estimates - lower, upper - estimates],
                  fmt="o", capsize=4, color="#226783", label="Accuracy and 95% Wilson interval")
    axis.set(xlabel="True digit", ylabel="Accuracy (%)", xticks=digits,
             ylim=(5 * np.floor(lower.min() / 5) - 2, 103))
    axis.grid(axis="y", alpha=0.2)
    axis.legend(loc="lower center" if compact else "lower right", ncols=2 if compact else 1)
    figure.savefig(output / "accuracy_wilson_by_digit.png", dpi=240)
    plt.close(figure)


def plot_examples(data, output, compact=False):
    digits = load_digits()
    split = read_rows(data / "split_indices.csv")
    predictions = read_rows(data / "test_predictions_nearest_centroid.csv")
    train = np.array([int(row["sample_index"]) for row in split if row["split"] == "train"])
    test = {int(row["sample_index"]) for row in split if row["split"] == "test"}
    centroids = [
        digits.images[train[digits.target[train] == digit]].mean(axis=0)
        for digit in range(10)
    ]

    figure, axes = plt.subplots(2, 6, figsize=(9.3, 2.65) if compact else (9.3, 3.4),
                                layout="constrained")
    for row_axes, (actual, predicted) in zip(axes, [(5, 9), (8, 1)]):
        indices = sorted(
            int(row["sample_index"]) for row in predictions
            if int(row["y_true"]) == actual and int(row["y_pred"]) == predicted
        )[:4]
        if len(indices) != 4 or any(i not in test or digits.target[i] != actual for i in indices):
            raise ValueError("Expected four matching test errors for each confusion direction.")
        images = [centroids[actual], centroids[predicted], *digits.images[indices]]
        titles = [f"True centroid: {actual}", f"Pred. centroid: {predicted}",
                  *(f"Test #{index}" for index in indices)]
        for axis, pixels, title in zip(row_axes, images, titles):
            axis.imshow(pixels, cmap="gray", vmin=0, vmax=16, interpolation="nearest")
            axis.set_title(title, fontsize=9)
            axis.set_axis_off()
        print(f"{actual} -> {predicted}: test indices {indices}")
    figure.savefig(output / "misclassified_examples_and_centroids.png", dpi=240)
    plt.close(figure)


def main(paper_output=None):
    output = paper_output or ROOT / "results" / "figures"
    output.mkdir(parents=True, exist_ok=True)
    plot_accuracy(ROOT / "results" / "tables", output, compact=paper_output is not None)
    plot_examples(ROOT / "data" / "processed", output, compact=paper_output is not None)


if __name__ == "__main__":
    main()
