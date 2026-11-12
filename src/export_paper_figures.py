"""Export compact versions of the experiment figures for the paper."""

from analyze_accuracy import ROOT
from analyze_bootstrap import main as plot_bootstrap
from analyze_confusion import main as plot_confusion
from plot_paper_figures import main as plot_accuracy_and_examples


if __name__ == "__main__":
    output = ROOT / "paper" / "figures"
    plot_confusion(output)
    plot_bootstrap(output)
    plot_accuracy_and_examples(output)
