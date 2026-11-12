# Digits Statistical Analysis

A reproducible analysis of handwritten-digit classification using scikit-learn's Digits dataset and a nearest-centroid classifier. The pipeline estimates test accuracy, tests for differences across digits, analyzes confusion patterns, and measures how test sample size affects the variability of accuracy estimates.

## Quick start

Use Python 3.13 to match the original run. From the repository root, preferably in a virtual environment:

```sh
python -m pip install numpy==2.3.5 scikit-learn==1.7.2 scipy matplotlib
python src/train_predict.py
python src/predict_nearest_centroid.py
python src/analyze_accuracy.py
python src/analyze_confusion.py
python src/analyze_bootstrap.py
python src/plot_paper_figures.py
```

The first command under `src/` creates the seeded 70/30 stratified split and runs an exploratory MLP baseline. The nearest-centroid script reuses that split, fits one mean image per digit using training pixels, and predicts the closest centroid by Euclidean distance. All analysis scripts read the nearest-centroid predictions.

## Outputs

| Path | Contents |
| --- | --- |
| `data/processed/` | Generated split and per-image predictions, keyed by the original dataset index |
| `results/tables/` | Accuracy estimates, Wilson intervals, permutation test, confusion counts, prior-shift analysis, and bootstrap summaries |
| `results/figures/` | Confusion matrix, per-digit intervals, example errors, and bootstrap plots |

The nearest-centroid classifier correctly classified **471 of 540** test images (**87.22%**). The approximate stratified 95% confidence interval was **84.55%–89.90%**. The fixed-margin Monte Carlo Pearson test used 19,999 replicates and returned a plus-one corrected p-value of 0.00005. These results are exploratory and describe this dataset and split.

## Data and license

Data are loaded through scikit-learn's `load_digits`. The original dataset is *Optical Recognition of Handwritten Digits* by E. Alpaydin and C. Kaynak (1998), distributed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Example figures show selected dataset images and class centroids computed from training images. [UCI dataset and citation](https://doi.org/10.24432/C50P49)

Project code is released under the [MIT License](LICENSE).
