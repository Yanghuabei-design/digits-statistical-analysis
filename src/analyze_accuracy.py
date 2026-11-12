"""Estimate accuracy and test equality of class accuracies."""

import csv
from pathlib import Path

import numpy as np
from scipy.stats import norm


ROOT = Path(__file__).resolve().parents[1]
REPLICATES = 19_999


def write_table(filename, header, rows):
    path = ROOT / "results" / "tables" / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)


def main():
    path = ROOT / "data" / "processed" / "test_predictions_nearest_centroid.csv"
    with path.open(encoding="utf-8", newline="") as file:
        pairs = [(int(row["y_true"]), int(row["y_pred"])) for row in csv.DictReader(file)]
    actual, predicted = np.array(pairs).T
    correct = actual == predicted
    groups = np.arange(10)
    sizes = np.bincount(actual, minlength=10)
    if np.any(sizes == 0):
        raise ValueError("Accuracy analysis requires test observations in each class.")
    hits = np.bincount(actual[correct], minlength=10)
    errors = sizes - hits
    rates = hits / sizes
    weights = sizes / sizes.sum()
    overall = weights @ rates
    standard_error = np.sqrt(np.sum(weights**2 * rates * (1 - rates) / sizes))
    z = norm.ppf(0.975)
    lower, upper = overall - z * standard_error, overall + z * standard_error
    write_table(
        "overall_accuracy.csv",
        ["n", "correct", "accuracy", "standard_error", "ci95_low", "ci95_high"],
        [(sizes.sum(), hits.sum(), overall, standard_error, lower, upper)],
    )

    denominator = 1 + z**2 / sizes
    center = (rates + z**2 / (2 * sizes)) / denominator
    margin = z * np.sqrt(rates * (1 - rates) / sizes + z**2 / (4 * sizes**2)) / denominator
    write_table(
        "accuracy_by_digit.csv",
        ["digit", "n", "correct", "errors", "accuracy", "wilson95_low", "wilson95_high"],
        zip(groups, sizes, hits, errors, rates, np.clip(center - margin, 0, 1),
            np.clip(center + margin, 0, 1)),
    )

    expected = np.column_stack([sizes * overall, sizes * (1 - overall)])
    if np.any(expected == 0):
        statistic = p_value = None
        simulations = 0
    else:
        def pearson(error_counts):
            tables = np.stack([sizes - error_counts, error_counts], axis=-1)
            return ((tables - expected) ** 2 / expected).sum(axis=(-2, -1))

        statistic = pearson(errors)
        rng = np.random.default_rng(42)
        # Sampling error counts without replacement is equivalent to permuting error labels.
        simulated_errors = rng.multivariate_hypergeometric(sizes, int(errors.sum()), size=REPLICATES)
        p_value = (1 + np.count_nonzero(pearson(simulated_errors) >= statistic)) / (REPLICATES + 1)
        simulations = REPLICATES
    write_table(
        "class_difference_test.csv",
        ["pearson_chi2", "min_expected_count", "monte_carlo_p", "replicates", "seed"],
        [(statistic, expected.min(), p_value, simulations, 42)],
    )
    print(f"Accuracy: {overall:.2%}, 95% CI: [{lower:.2%}, {upper:.2%}]")
    print(f"Pearson statistic: {statistic}, Monte Carlo p: {p_value}")


if __name__ == "__main__":
    main()
