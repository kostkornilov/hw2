"""Prepare RegBench, probe author training, and validate its local artifacts.

Uses the pinned seq_icl checkout with the paper's ICLL files overlaid.
Training uses two GPUs; final prediction export uses one GPU to avoid concurrent
writes to the author's generations files. No long training is launched here.
"""
import argparse
import csv
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import statistics
import sys
import time


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def config_for(args, author):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    with initialize_config_dir(config_dir=str(args.source / "configs"), version_base="1.2"):
        config = compose(config_name="config", overrides=[
            "experiment=dfa/mm", f"dataset.num_examples={args.examples}",
            "dataset.num_test_examples=1000", "dataset.batch_size=16",
            "model.n_layer=2", "model.n_embd=128", "model.n_head=8",
            f"+model.method={args.method}", "optimizer.lr=1e-4",
            "optimizer.weight_decay=0.01", "train.seed=0", "trainer.max_epochs=200",
            "wandb=null",
        ])
    # Resolve the 200-epoch LR schedule before imposing the short stopping point.
    config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    return author.utils.train.process_config(config)


def load_cached(args, config):
    import torch
    from src.dataloaders.dfa import ICLDFADataModule

    meta = json.loads(args.cache.with_suffix(".json").read_text())
    assert meta["examples"] == args.examples
    assert meta["sha256"] == digest(args.cache), "Dataset cache changed"
    original_setup = ICLDFADataModule.setup

    def setup(self, stage=None):
        if not hasattr(self, "dataset"):
            self.dataset = torch.load(args.cache, map_location="cpu", weights_only=False)

    ICLDFADataModule.setup = setup
    return meta, original_setup


def prepare(args, config):
    import torch
    from src.dataloaders.dfa import ICLDFADataModule
    from omegaconf import OmegaConf

    assert not args.cache.exists(), "Use the existing cache or choose a new path"
    assert os.environ.get("PYTHONHASHSEED") == "0", "Run generation with PYTHONHASHSEED=0"
    dataset = ICLDFADataModule(**OmegaConf.to_container(config.dataset, resolve=True))
    start = time.perf_counter()
    dataset.setup()
    args.cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dataset.dataset, args.cache)
    meta = {"examples": args.examples, "seed": 0,
            "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
            "dataset_config": OmegaConf.to_container(config.dataset, resolve=True),
            "generation_seconds": time.perf_counter() - start,
            "sha256": digest(args.cache), "splits": {}}
    for split, data in dataset.dataset.items():
        lengths = (data.inputs != dataset.vocab.get_id(".")).sum(dim=1)
        meta["splits"][split] = {"examples": len(data), "padded_tokens": data.inputs.shape[1],
            "min_tokens": int(lengths.min()), "median_tokens": float(lengths.float().median()),
            "max_tokens": int(lengths.max()), "scored_tokens": int((data.targets != -100).sum())}
        assert len(data) == (args.examples if split == "train" else 500)
        assert data.inputs.shape == data.targets.shape
        assert data.inputs.shape[1] <= config.model.block_size
    save_json(args.cache.with_suffix(".json"), meta)
    print(json.dumps(meta, indent=2))


def train(args, config, author):
    import torch
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
    from pytorch_lightning.loggers import CSVLogger
    from omegaconf import OmegaConf

    assert torch.cuda.device_count() == 2, "This timing probe requires T4 x2"
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    meta, _ = load_cached(args, config)
    args.run.mkdir(parents=True, exist_ok=True)
    os.chdir(args.run)
    pl.seed_everything(0, workers=True)

    class Probe(pl.Callback):
        def __init__(self):
            self.epochs, self.steps, self.checkpoints = [], [], []

        def now(self):
            torch.cuda.synchronize()
            return time.perf_counter()

        def on_train_epoch_start(self, trainer, model):
            self.epoch_start = self.now()
            self.val_seconds = 0.0

        def on_train_batch_start(self, trainer, model, batch, batch_idx):
            self.batch_start = self.now()

        def on_train_batch_end(self, trainer, model, outputs, batch, batch_idx):
            assert torch.isfinite(outputs["loss"]).all(), "Non-finite training loss"
            self.steps.append({"step": trainer.global_step, "epoch": trainer.current_epoch,
                               "seconds": self.now() - self.batch_start})

        def on_validation_start(self, trainer, model):
            self.val_start = self.now()

        def on_validation_end(self, trainer, model):
            if not trainer.sanity_checking:
                self.val_seconds += self.now() - self.val_start

        def on_train_epoch_end(self, trainer, model):
            metrics = {k: float(v.detach().cpu()) for k, v in trainer.callback_metrics.items()
                       if k in ("train/loss", "val/loss", "test/loss", "val/dfa_accuracy", "test/dfa_accuracy")}
            assert all(math.isfinite(v) for v in metrics.values())
            self.epochs.append({"epoch": trainer.current_epoch, "step": trainer.global_step,
                "train_seconds": self.now() - self.epoch_start - self.val_seconds,
                "validation_seconds": self.val_seconds, **metrics})

    probe = Probe()

    class Checkpoint(ModelCheckpoint):
        def _save_checkpoint(self, trainer, filepath):
            start = probe.now()
            super()._save_checkpoint(trainer, filepath)
            probe.checkpoints.append(probe.now() - start)

    checkpoint = Checkpoint(monitor="val/loss", mode="min", save_top_k=1, save_last=True,
                            dirpath="checkpoints", filename="best", auto_insert_metric_name=False)
    config.trainer.max_steps = args.epochs * math.ceil(args.examples / 32)
    config.trainer.devices = 2
    config.trainer.strategy = "ddp_find_unused_parameters_false"
    config.trainer.precision = 32  # Author default; no BF16 requirement on T4.
    config.trainer.num_sanity_val_steps = 0
    config.trainer.enable_progress_bar = False
    config.trainer.log_every_n_steps = 1
    params = OmegaConf.to_container(config.trainer, resolve=True)
    params.pop("_target_")
    if int(os.environ.get("RANK", 0)) == 0:
        assert not Path("config.json").exists(), "Do not overwrite an earlier run"
        save_json(Path("config.json"), OmegaConf.to_container(config, resolve=True))
        save_json(Path("dataset_manifest.json"), meta)
    trainer = pl.Trainer(**params, callbacks=[probe, checkpoint, LearningRateMonitor("step")],
                         logger=CSVLogger("metrics", name="", version="probe"))
    start = time.perf_counter()
    model = author.SequenceLightningModule(config)
    assert all(block.attn.method == args.method for block in model.model.blocks)
    trainer.fit(model)
    if trainer.is_global_zero:
        assert trainer.global_step == config.trainer.max_steps
        assert len(probe.epochs) == args.epochs
        save_json(Path("timing.json"), {"method": args.method, "examples": args.examples,
            "full_epochs": 200, "probe_epochs": args.epochs, "global_step": trainer.global_step,
            "elapsed_seconds": time.perf_counter() - start, "epochs": probe.epochs,
            "steps": probe.steps, "checkpoint_seconds": probe.checkpoints,
            "best_checkpoint": checkpoint.best_model_path,
            "gpu_names": [torch.cuda.get_device_name(i) for i in range(2)],
            "peak_allocated_rank0_bytes": torch.cuda.max_memory_allocated()})
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


def evaluate(args, config, author):
    import torch
    import pytorch_lightning as pl
    from pytorch_lightning.loggers import CSVLogger

    load_cached(args, config)
    config.dataset.batch_size = 32
    os.chdir(args.run)
    assert not Path("generations").exists(), "Evaluation must use a fresh output directory"
    pl.seed_everything(0, workers=True)
    model = author.SequenceLightningModule(config)
    trainer = pl.Trainer(accelerator="gpu", devices=1, precision=32,
                         logger=CSVLogger("metrics", name="", version="evaluation"),
                         enable_checkpointing=False, enable_progress_bar=False)
    timing = json.loads(Path("timing.json").read_text())
    start = time.perf_counter()
    result = trainer.test(model, ckpt_path=timing["best_checkpoint"])
    save_json(Path("evaluation.json"), {"elapsed_seconds": time.perf_counter() - start,
                                      "metrics": result, "checkpoint": timing["best_checkpoint"]})


def validate_run(run):
    """Compute token-weighted DFA accuracy/TVD from author prediction dumps."""
    import numpy as np
    import torch
    from analyze import get_dfa_probs
    from src.dataloaders.dfa import Vocab

    timing = json.loads((run / "timing.json").read_text())
    evaluation = json.loads((run / "evaluation.json").read_text())
    ckpt = torch.load(timing["best_checkpoint"], map_location="cpu", weights_only=False)
    assert ckpt["state_dict"] and ckpt["optimizer_states"] and ckpt["lr_schedulers"]
    assert all(torch.isfinite(v).all() for v in ckpt["state_dict"].values() if v.is_floating_point())
    cfg = json.loads((run / "config.json").read_text())
    assert ckpt["hyper_parameters"]["model"]["method"] == timing["method"] == cfg["model"]["method"]
    assert 0 < ckpt["global_step"] <= timing["global_step"] == cfg["trainer"]["max_steps"]
    last = torch.load(run / "checkpoints/last.ckpt", map_location="cpu", weights_only=False)
    assert last["global_step"] == timing["global_step"]
    assert (run / "metrics/probe/metrics.csv").stat().st_size > 0
    metrics = {}
    for split in ("val", "test"):
        folders = list((run / "generations").glob(f"*_{split}_batch"))
        assert len(folders) == 1
        folder = folders[0]
        rows = list(csv.reader(folder.with_name(folder.name.removesuffix("_batch") + ".txt").open(), delimiter="\t"))
        offset, total, correct, l1 = 0, 0, 0, 0.0
        last_correct, last_l1 = 0, 0.0
        for path in sorted(folder.glob("*.pkl.gz"), key=lambda p: int(p.name.split(".")[0])):
            with gzip.open(path, "rb") as stream:
                raw = pickle.load(stream)
            logits = np.asarray(raw["probs"])
            assert logits.ndim == 3 and np.isfinite(logits).all()
            vocab = Vocab(len(raw["vocab"]) - 2, special_vocabs={"seperator": "|", "noop": "."})
            assert vocab.vocab == raw["vocab"]
            assert logits.shape[0] == len(raw["dfas"])
            for i, dfa in enumerate(raw["dfas"]):
                target = get_dfa_probs(rows[offset][0], dfa, vocab)
                probs = torch.softmax(torch.from_numpy(logits[i, :len(target)]), -1).numpy()
                assert probs.shape == target.shape
                assert np.allclose(probs.sum(-1), 1, atol=1e-5)
                correct += (target[np.arange(len(target)), probs.argmax(-1)] > 0).sum()
                l1 += np.abs(probs - target).sum()
                last_correct += target[-1, probs[-1].argmax()] > 0
                last_l1 += np.abs(probs[-1] - target[-1]).sum()
                total += len(target)
                offset += 1
        assert offset == len(rows) == 500 and total > 0
        metrics[f"{split}_accuracy"] = float(correct / total)
        metrics[f"{split}_tvd"] = float(l1 / (2 * total))
        metrics[f"{split}_last_token_accuracy"] = float(last_correct / offset)
        metrics[f"{split}_last_token_tvd"] = float(last_l1 / (2 * offset))
        assert 0 <= metrics[f"{split}_tvd"] <= 1
    stable = timing["epochs"][1:]
    assert len(stable) >= 2, "At least three full epochs needed for the estimate"
    train_s = statistics.median(e["train_seconds"] for e in stable)
    val_s = statistics.median(e["validation_seconds"] for e in stable)
    ckpt_s = statistics.median(timing["checkpoint_seconds"])
    final_s = evaluation["elapsed_seconds"]
    startup_s = max(0, timing["elapsed_seconds"] - sum(e["train_seconds"] + e["validation_seconds"] for e in timing["epochs"]) - sum(timing["checkpoint_seconds"]))
    # Upper estimate: best + last saved at every epoch, two files per epoch.
    projected_s = startup_s + 200 * (train_s + val_s + 2 * ckpt_s) + final_s
    manifest = [{"path": str(p.relative_to(run)), "bytes": p.stat().st_size, "sha256": digest(p)}
                for p in sorted(run.rglob("*")) if p.is_file() and p.name != "artifact_validation.json"]
    assert all(p["bytes"] > 0 for p in manifest)
    save_json(run / "artifact_validation.json", {"status": "passed", "metrics": metrics, "files": manifest})
    return {"method": timing["method"], "examples": timing["examples"],
            "train_seconds_per_epoch": train_s, "validation_seconds_per_epoch": val_s,
            "final_evaluation_seconds": final_s, "estimated_200_epochs_hours": projected_s / 3600,
            "within_80_minutes": projected_s <= 80 * 60, **metrics}


def summarize(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [validate_run(args.run / method) for method in ("softmax", "epanechnikov", "relumax")]
    with (args.run / "summary.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    lines = ["Короткая проверка RegBench: это не результаты полного обучения.", "",
             "| Method | Train s/epoch | Validation s/epoch | 200 epochs, h | Test accuracy | Test TVD | Last-token accuracy | Last-token TVD |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['method']} | {row['train_seconds_per_epoch']:.2f} | {row['validation_seconds_per_epoch']:.2f} | {row['estimated_200_epochs_hours']:.2f} | {100*row['test_accuracy']:.2f}% | {100*row['test_tvd']:.2f}% | {100*row['test_last_token_accuracy']:.2f}% | {100*row['test_last_token_tvd']:.2f}% |")
    lines += ["", "Прогноз включает validation на каждой эпохе, сохранение best/last и финальную оценку.",
              "Первую эпоху исключаем из медианы; scheduler остаётся рассчитанным на 200 эпох.",
              f"Суммарный прогноз трёх методов: {sum(row['estimated_200_epochs_hours'] for row in rows):.2f} ч вычислений.",
              "Accuracy и TVD пересчитаны по всем позициям сохранённых тестовых предсказаний",
              "по формулам авторского plot_ICL.py; это не отдельная last-token метрика."]
    (args.run / "summary.md").write_text("\n".join(lines) + "\n")
    fig, ax = plt.subplots(figsize=(7, 4))
    for method in ("softmax", "epanechnikov", "relumax"):
        timing = json.loads((args.run / method / "timing.json").read_text())
        ax.plot([e["epoch"] + 1 for e in timing["epochs"]], [e["val/loss"] for e in timing["epochs"]], marker="o", label=method)
    ax.set(xlabel="Epoch", ylabel="Validation loss", title="RegBench 1k — timing probe")
    ax.set_xticks([e["epoch"] + 1 for e in timing["epochs"]])
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(args.run / f"validation_loss.{extension}", dpi=160)
    plt.close(fig)
    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "train", "evaluate", "summarize"))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--output-dir", "--run", dest="run", type=Path)
    parser.add_argument("--examples", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--method", choices=("softmax", "epanechnikov", "relumax"), default="softmax")
    args = parser.parse_args()
    args.source = args.source.resolve()
    for name in ("cache", "run"):
        if getattr(args, name) is not None:
            setattr(args, name, getattr(args, name).resolve())
    sys.path.insert(0, str(args.source))
    if args.action == "summarize":
        summarize(args)
        return
    import train as author
    config = config_for(args, author)
    if args.action == "prepare":
        prepare(args, config)
    elif args.action == "train":
        train(args, config, author)
    else:
        evaluate(args, config, author)


if __name__ == "__main__":
    main()
