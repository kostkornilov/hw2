"""Plot saved RegBench validation histories for the reproduction report.

Example:
    python scripts/plot_icl_results.py artifacts/kaggle/icl-full-lite/results/icl/full_20261010_143712_771702/in_context_learning

Requires matplotlib; no model, checkpoint or GPU is needed.
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


METHODS = {
    "softmax": ("Softmax", "#0072B2", "-"),
    "epanechnikov": ("Sparsemax", "#D55E00", "--"),
    "relumax": ("1-ReLUmax", "#009E73", "-."),
}


def plot_results(source: Path, output: Path):
    histories = {}
    for method in METHODS:
        path = source / method / "timing.json"
        history = json.loads(path.read_text())
        epochs = history["epochs"]
        if not epochs or any(not math.isfinite(e["val/loss"]) for e in epochs):
            raise ValueError(f"Missing or non-finite validation history: {path}")
        if [e["epoch"] for e in epochs] != list(range(len(epochs))):
            raise ValueError(f"Expected consecutive epochs starting at zero: {path}")
        histories[method] = history

    sizes = {h["examples"] for h in histories.values()}
    lengths = {len(h["epochs"]) for h in histories.values()}
    if len(sizes) != 1 or len(lengths) != 1:
        raise ValueError("Methods must have the same dataset size and epoch count")
    examples, epochs = sizes.pop(), lengths.pop()
    size_label = f"{examples // 1000}k" if examples % 1000 == 0 else str(examples)

    with plt.rc_context({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42}):
        fig, ax = plt.subplots(figsize=(7.2, 4.2), layout="constrained")
        for method, (label, color, style) in METHODS.items():
            history = histories[method]["epochs"]
            ax.plot([e["epoch"] + 1 for e in history],
                    [e["val/loss"] for e in history],
                    label=label, color=color, linestyle=style, linewidth=2)
        ax.set(title=f"RegBench — {size_label} training examples, {epochs} epochs",
               xlabel="Epoch", ylabel="Validation loss", xlim=(1, max(2, epochs)))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
        ax.grid(alpha=0.2)
        ax.legend(frameon=False)
        output.parent.mkdir(parents=True, exist_ok=True)
        for extension in ("png", "pdf"):
            path = output.with_suffix(f".{extension}")
            fig.savefig(path, dpi=300, facecolor="white")
            print(path)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path,
                        help="in_context_learning directory containing per-method timing.json")
    parser.add_argument("--output", type=Path, default=Path("figures/regbench_validation_loss"),
                        help="Output filename stem (exports both PNG and PDF)")
    args = parser.parse_args()
    plot_results(args.source, args.output)


if __name__ == "__main__":
    main()
