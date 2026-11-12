"""Simulate accuracy estimates at several stratified test sample sizes."""

import csv

import matplotlib.pyplot as plt
import numpy as np

from analyze_accuracy import ROOT, write_table


SAMPLE_SIZES = (50, 100, 200, 400)
REPLICATES = 1000


def main(paper_output=None):
    path = ROOT / "data" / "processed" / "test_predictions_nearest_centroid.csv"
    with path.open(encoding="utf-8", newline="") as file:
        pairs = [(int(row["y_true"]), int(row["y_pred"])) for row in csv.DictReader(file)]
    actual, predicted = np.array(pairs).T
    outcomes = [(actual == predicted)[actual == digit] for digit in range(10)]
    sizes = np.array([len(group) for group in outcomes])
    if np.any(sizes == 0):
        raise ValueError("Stratified bootstrap requires observations from each digit.")
    weights = sizes / sizes.sum()
    rates = np.array([group.mean() for group in outcomes])
    baseline = weights @ rates
    rng = np.random.default_rng(42)
    summaries = []
    distributions = []
    for size in SAMPLE_SIZES:
        # Integer division keeps exact ties in the fractional quotas.
        allocation, remainders = np.divmod(size * sizes, sizes.sum())
        order = np.argsort(-remainders, kind="stable")
        allocation[order[:size - allocation.sum()]] += 1
        if np.any(allocation == 0):
            raise ValueError("Every digit must receive a bootstrap allocation.")
        means = np.column_stack([
            rng.choice(group, size=(REPLICATES, count), replace=True).mean(axis=1)
            for group, count in zip(outcomes, allocation)
        ])
        estimates = means @ weights
        distributions.append(estimates)
        low, high = np.quantile(estimates, [0.025, 0.975])
        formula_se = np.sqrt(np.sum(weights**2 * rates * (1 - rates) / allocation))
        summaries.append((size, estimates.mean(), estimates.std(ddof=1), formula_se,
                          low, high, high - low, *allocation))

    if paper_output is None:
        write_bootstrap_tables(summaries, distributions)
    plot_bootstrap(summaries, distributions, baseline, paper_output)


def write_bootstrap_tables(summaries, distributions):
    write_table(
        "bootstrap_accuracy.csv",
        ["sample_size", "mean_accuracy", "empirical_sd", "formula_se",
         "central95_low", "central95_high", "central95_width",
         *(f"n_digit_{digit}" for digit in range(10))],
        summaries,
    )
    write_table(
        "bootstrap_accuracy_draws.csv",
        ["sample_size", "replicate", "accuracy"],
        ((size, trial, estimate) for size, estimates in zip(SAMPLE_SIZES, distributions)
         for trial, estimate in enumerate(estimates, start=1)),
    )


def plot_bootstrap(summaries, distributions, baseline, paper_output):
    output = paper_output or ROOT / "results" / "figures"
    output.mkdir(parents=True, exist_ok=True)
    rows, columns = (1, 4) if paper_output else (2, 2)
    size_inches = (10, 2.45) if paper_output else (8, 5.6)
    figure, axes = plt.subplots(rows, columns, figsize=size_inches, sharex=True, sharey=True,
                                layout="constrained")
    bins = np.arange(0, 102, 1)
    for axis, size, estimates in zip(axes.flat, SAMPLE_SIZES, distributions):
        axis.hist(100 * estimates, bins=bins, color="#287594", rwidth=0.94)
        axis.axvline(100 * baseline, color="#9c4424", linestyle="--")
        axis.set_title(f"Sample size {size}")
        axis.set_xlim(min(group.min() for group in distributions) * 100 - 1, 101)
    figure.supxlabel("Estimated accuracy (%)")
    figure.supylabel("Replicates per bin" if paper_output else
                     "Bootstrap replicates per 1 percentage point")
    figure.savefig(output / "bootstrap_accuracy.png", dpi=240)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.5, 2.8) if paper_output else (6.5, 4),
                                layout="constrained")
    summary = np.array(summaries)
    axis.plot(SAMPLE_SIZES, 100 * summary[:, 2], "o-", label="Empirical SD")
    axis.plot(SAMPLE_SIZES, 100 * summary[:, 3], "s--", label="Formula SE")
    axis.set(xlabel="Sample size", ylabel="Standard error (pp)" if paper_output else
             "Standard error (percentage points)",
             xticks=SAMPLE_SIZES)
    axis.legend()
    figure.savefig(output / "bootstrap_standard_error.png", dpi=240)
    plt.close(figure)


if __name__ == "__main__":
    main()
