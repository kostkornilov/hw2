"""Validate saved LM results and export tables and loss curves (no training)."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

LABELS = {"softmax": "Gaussian / softmax", "epanechnikov": "Epanechnikov / sparsemax",
          "relumax": "1-ReLUmax"}


def export(root, methods, kind):
    summaries, histories, manifest = [], {}, {}
    for method in methods:
        run = root / "runs" / method
        paths = [run / name for name in
                 ("ckpt.pt", "loss_history.json", "metrics.csv", "train.log", "run_config.json")]
        for path in paths:
            if not path.is_file() or not path.stat().st_size:
                raise ValueError(f"Missing or empty artifact: {path}")
        metadata = json.loads((run / "run_config.json").read_text())
        parameters = metadata.get("parameters", metadata.get("benchmark_parameters"))
        history = json.loads((run / "loss_history.json").read_text())
        steps, train, val = (history[key] for key in ("iter", "train", "val"))
        if not steps or len(steps) != len(train) or len(steps) != len(val):
            raise ValueError(f"{method}: inconsistent history lengths")
        if any(not isinstance(s, int) or s <= 0 for s in steps) or steps != sorted(set(steps)):
            raise ValueError(f"{method}: invalid evaluation steps")
        if any(not math.isfinite(v) for v in train + val):
            raise ValueError(f"{method}: non-finite loss")
        if parameters["method"] != method or steps[-1] != parameters["max_iters"]:
            raise ValueError(f"{method}: method or final step differs from run configuration")
        with (run / "metrics.csv").open(newline="") as stream:
            csv_rows = list(csv.DictReader(stream))
        expected = list(zip(steps, train, val))
        actual = [(int(r["step"]), float(r["train_loss"]), float(r["val_loss"])) for r in csv_rows]
        if actual != expected:
            raise ValueError(f"{method}: CSV differs from JSON")
        # Load only locally produced, trusted checkpoints. mmap avoids copying the entire file.
        checkpoint = torch.load(run / "ckpt.pt", map_location="cpu", weights_only=False, mmap=True)
        for key in ("model", "optimizer", "scaler", "model_args", "iter_num", "loss_history"):
            if key not in checkpoint:
                raise ValueError(f"{method}: checkpoint missing {key}")
        if checkpoint["iter_num"] != steps[-1] or checkpoint["loss_history"] != history:
            raise ValueError(f"{method}: checkpoint differs from history")
        for key, value in parameters.items():
            if checkpoint["model_args"].get(key) != value:
                raise ValueError(f"{method}: checkpoint configuration differs for {key}")
        if not checkpoint["model"] or not checkpoint["optimizer"].get("state") or not checkpoint["scaler"]:
            raise ValueError(f"{method}: empty model, optimizer or FP16 scaler state")
        if any(not torch.isfinite(t).all().item() for t in checkpoint["model"].values()):
            raise ValueError(f"{method}: non-finite checkpoint weights")
        del checkpoint
        best = min(range(len(val)), key=val.__getitem__)
        global_batch = parameters["batch_size"] * parameters["gradient_accumulation_steps"] * len(metadata["gpu_ids"])
        summaries.append({"method": method, "run_kind": kind, "steps": steps[-1],
                          "train_tokens_processed": steps[-1] * global_batch * parameters["block_size"],
                          "final_train_loss": train[-1], "final_val_loss": val[-1],
                          "best_observed_val_loss": val[best], "best_observed_step": steps[best],
                          "repository_commit": metadata["repository_commit"]})
        histories[method] = history
        for path in paths:
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
                    digest.update(chunk)
            manifest[str(path.relative_to(root))] = {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}

    # Export only after every run has passed validation.
    with (root / "results_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    rows = [f"Run kind: {kind}", "", "| Method | Steps | Final train loss | Final val loss | Best observed val loss | Step |",
            "|---|---:|---:|---:|---:|---:|"]
    for row in summaries:
        rows.append(f"| {LABELS[row['method']]} | {row['steps']} | {row['final_train_loss']:.6f} | "
                    f"{row['final_val_loss']:.6f} | {row['best_observed_val_loss']:.6f} | {row['best_observed_step']} |")
    rows += ["", "ckpt.pt contains the final model. Best observed loss refers to recorded evaluations;",
             "a separate checkpoint for the best step is not retained."]
    (root / "results_summary.md").write_text("\n".join(rows) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for method, history in histories.items():
        for axis, split in zip(axes, ("train", "val")):
            axis.plot(history["iter"], history[split], marker="o", label=LABELS[method])
            axis.set(xlabel="Optimizer step", ylabel=f"{split.capitalize()} loss")
            axis.grid(alpha=0.3)
    axes[1].legend()
    fig.suptitle(f"BabiStories, 1 layer — {kind}")
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(root / f"loss_curves.{extension}", dpi=180)
    plt.close(fig)
    (root / "artifact_validation.json").write_text(json.dumps({
        "status": "passed", "run_kind": kind, "methods": methods, "artifacts": manifest,
    }, indent=2) + "\n")
    print(f"Validated {len(methods)} runs. Tables and curves: {root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--methods", nargs="+", choices=LABELS, default=list(LABELS))
    parser.add_argument("--kind", choices=("timing-smoke", "budget-training"), required=True)
    args = parser.parse_args()
    export(args.root, args.methods, args.kind)
