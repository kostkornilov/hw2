"""Recover recorded LM losses from notebook output; no checkpoints required."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re


LABELS = {"softmax": "Softmax", "epanechnikov": "Sparsemax", "relumax": "1-ReLUmax"}
STAGE = re.compile(r"=== lm: lm_(softmax|epanechnikov|relumax) ===")
EVAL = re.compile(r"\[eval\] step (\d+): train loss ([\d.eE+-]+), val loss ([\d.eE+-]+)")
TIME = re.compile(r"INFO - (\d{2}/\d{2}/\d{2} \d{2}:\d{2}:\d{2}) -")


def recover(notebook, output, comparison_step):
    raw = notebook.read_bytes()
    document = json.loads(raw)
    rows, evidence = {}, []
    method = None
    for cell_index, cell in enumerate(document["cells"]):
        text = "".join("".join(item.get("text", [])) for item in cell.get("outputs", []))
        for line_index, line in enumerate(text.splitlines()):
            stage = STAGE.search(line)
            if stage:
                method = stage.group(1)
            match = EVAL.search(line)
            if not match:
                continue
            if method is None:
                raise ValueError("Evaluation without a method heading")
            step, train, val = int(match[1]), float(match[2]), float(match[3])
            assert step > 0 and all(math.isfinite(v) and v >= 0 for v in (train, val))
            time = TIME.search(line)
            row = {"method": method, "step": step, "train_loss": train, "val_loss": val,
                   "timestamp_utc": time[1] if time else ""}
            key = method, step
            if key in rows:
                assert rows[key] == row, f"Conflicting duplicate: {key}"
                continue
            rows[key] = row
            evidence.append({"method": method, "step": step, "cell_index": cell_index,
                             "line_index": line_index, "text": line})
    expected = list(range(100, comparison_step + 1, 100))
    assert expected and expected[-1] == comparison_step
    for method in LABELS:
        steps = sorted(s for m, s in rows if m == method and s <= comparison_step)
        assert steps == expected, (method, steps, expected)
    ordered = sorted(rows.values(), key=lambda row: (list(LABELS).index(row["method"]), row["step"]))
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "loss_history.csv", ordered)
    summary = [{"method": method, "step": comparison_step,
                "train_loss": rows[method, comparison_step]["train_loss"],
                "val_loss": rows[method, comparison_step]["val_loss"],
                "last_recorded_step": max(s for m, s in rows if m == method)} for method in LABELS]
    write_csv(output / "comparison_800.csv", summary)
    table = [f"| Method | Step | Train loss | Validation loss |",
             "|---|---:|---:|---:|"]
    for row in summary:
        table.append(f"| {LABELS[row['method']]} | {comparison_step} | {row['train_loss']:.4f} | {row['val_loss']:.4f} |")
    (output / "results_summary.md").write_text("\n".join(table) + "\n")
    (output / "provenance.json").write_text(json.dumps({
        "source_notebook": str(notebook), "source_sha256": hashlib.sha256(raw).hexdigest(),
        "source_type": "saved_notebook_stdout", "comparison_step": comparison_step,
        "recorded_precision_decimal_places": 4, "checkpoints_available": False,
        "metric_checks": "Finite values; all steps 100..800 for every method; no conflicting duplicates",
        "limitations": "Metrics extracted from stdout; model weights and optimizer states cannot be verified",
        "rows": len(ordered), "evidence": evidence,
    }, ensure_ascii=False, indent=2) + "\n")
    (output / "report_fragment.md").write_text(
        "Однослойная модель обучалась на BabiStories с авторским размером архитектуры "
        "(embedding 768, 12 голов, persistent memory 2688, context 512), общим batch 512 и FP16 на двух T4. "
        "Планировалось 1000 шагов вместо авторских 80000; warmup 25, cosine decay 1000, один seed. "
        "Softmax и sparsemax завершили 1000 шагов; последняя записанная оценка 1-ReLUmax — шаг 800. "
        "Основное сравнение проводится на одинаковом шаге 800 для всех методов, "
        "при этом расписание learning rate оставалось рассчитанным на 1000 шагов.\n\n"
        + "\n".join(table) + "\n\n"
        "На шаге 800 sparsemax имеет наименьший validation loss; softmax и 1-ReLUmax дают близкие значения. "
        "Одного seed и короткого обучения недостаточно для вывода о статистической значимости "
        "различий или воспроизведении итоговых значений после 80000 шагов.\n\n"
        "Метрики восстановлены из сохранённого вывода обучения. Checkpoints утрачены после "
        "остановки интерактивной сессии Kaggle; повторная проверка весов недоступна. "
        "Числа в stdout записаны с четырьмя знаками после запятой. CSV сохраняет все доступные "
        "оценки, график использует только общие шаги 100–800.\n")
    plot(output, ordered, comparison_step)
    print(f"Recovered {len(ordered)} evaluations; common comparison: {comparison_step} steps")
    for row in summary:
        print(f"{row['method']}: validation={row['val_loss']:.4f}; last recorded step={row['last_recorded_step']}")


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(output, rows, comparison_step):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with plt.rc_context({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False}):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
        colors = {"softmax": "#3572A5", "epanechnikov": "#D97732", "relumax": "#36865B"}
        for method, label in LABELS.items():
            history = [row for row in rows if row["method"] == method and row["step"] <= comparison_step]
            for axis, split in zip(axes, ("train_loss", "val_loss")):
                axis.plot([row["step"] for row in history], [row[split] for row in history],
                          label=label, color=colors[method], marker="o", markersize=3.5, linewidth=1.8)
        for axis, title in zip(axes, ("Training", "Validation")):
            axis.set(title=title, xlabel="Optimizer step", xticks=list(range(100, comparison_step + 1, 100)))
            axis.grid(alpha=0.18)
        axes[0].set_ylabel("Cross-entropy loss")
        axes[1].legend(frameon=False)
        fig.suptitle("BabiStories · 1 layer · comparison through step 800")
        fig.text(0.5, 0.015, "Recovered from saved training output · 1 seed · checkpoints unavailable",
                 ha="center", fontsize=9, color="#666666")
        fig.tight_layout(rect=(0, 0.05, 1, 0.94))
        for extension in ("png", "pdf"):
            fig.savefig(output / f"loss_curves.{extension}", dpi=200)
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebook", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/lm_recovered"))
    args = parser.parse_args()
    recover(args.notebook, args.output, 800)
