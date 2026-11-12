"""Summarize nearest-centroid errors and the digit-5 prior scenario."""

import csv

import matplotlib.pyplot as plt
import numpy as np

from analyze_accuracy import ROOT, write_table


def main(paper_output=None):
    path = ROOT / "data" / "processed" / "test_predictions_nearest_centroid.csv"
    counts = np.zeros((10, 10), dtype=int)
    with path.open(encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            counts[int(row["y_true"]), int(row["y_pred"])] += 1
    if paper_output is not None:
        plot_confusion(counts, paper_output)
        return
    sizes = counts.sum(axis=1)
    errors = sizes - counts.diagonal()
    conditional = counts / sizes[:, None]
    write_table(
        "confusion_matrix.csv", ["true_digit", *range(10)],
        ([digit, *row] for digit, row in enumerate(counts)),
    )
    observed = [(i, j) for i, j in zip(*counts.nonzero()) if i != j]
    observed.sort(key=lambda pair: (-counts[pair], pair))
    write_table(
        "observed_misclassifications.csv",
        ["true_digit", "predicted_digit", "count", "true_class_n", "true_class_errors",
         "rate_given_true_digit", "share_given_true_digit_and_error"],
        ((i, j, counts[i, j], sizes[i], errors[i], conditional[i, j],
          counts[i, j] / errors[i]) for i, j in observed),
    )

    baseline = sizes / sizes.sum()
    scenario = baseline.copy()
    scenario[5] *= 3
    scenario /= scenario.sum()
    predicted_five = conditional[:, 5]
    if not counts[:, 5].sum():
        raise ValueError("The posterior is undefined because no test image was predicted as 5.")
    joint_baseline = baseline * predicted_five
    joint_scenario = scenario * predicted_five
    posterior_baseline = joint_baseline / joint_baseline.sum()
    posterior_scenario = joint_scenario / joint_scenario.sum()
    np.testing.assert_allclose(posterior_baseline, counts[:, 5] / counts[:, 5].sum())
    write_table(
        "posterior_given_prediction_5.csv",
        ["true_digit", "count_predicted_5", "baseline_prior", "scenario_prior",
         "p_predict_5_given_true", "baseline_posterior", "scenario_posterior"],
        zip(range(10), counts[:, 5], baseline, scenario, predicted_five,
            posterior_baseline, posterior_scenario),
    )

    plot_confusion(counts)
    print(f"Test images: {counts.sum()}, errors: {errors.sum()}")
    print(f"P(true 5 | predicted 5): {posterior_baseline[5]:.2%} -> {posterior_scenario[5]:.2%}")


def plot_confusion(counts, paper_output=None):
    size = (8.2, 3.65) if paper_output else (7.2, 6)
    figure, axis = plt.subplots(figsize=size, layout="constrained")
    matrix = axis.imshow(counts, cmap="Blues", aspect="auto" if paper_output else "equal")
    for (i, j), count in np.ndenumerate(counts):
        if count:
            color = "white" if count > counts.max() / 2 else "#152333"
            axis.text(j, i, str(count), ha="center", va="center", color=color)
    axis.set(
        xlabel="Predicted digit", ylabel="True digit", xticks=range(10), yticks=range(10),
        title="" if paper_output else "Nearest centroid confusion matrix",
    )
    figure.colorbar(matrix, ax=axis, label="Test images")
    output = paper_output or ROOT / "results" / "figures"
    output.mkdir(parents=True, exist_ok=True)
    figure.savefig(output / "confusion_matrix.png", dpi=240)
    plt.close(figure)


if __name__ == "__main__":
    main()
