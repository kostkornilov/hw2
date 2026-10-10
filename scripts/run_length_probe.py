"""Sort probe using the authors' synthetic-task model, with explicit provenance.

The released epoch trainer ignores max_iters as a stopping condition and does
not save a final checkpoint. test_synthetic.py leaves sort metric variables
undefined. This runner bounds the probe, saves its last state, and computes
the same teacher-forced output exact-match metric over each example once.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import struct
import subprocess
import sys
import time
from types import SimpleNamespace
from urllib.request import urlopen

METHODS = ("softmax", "epanechnikov", "relumax")
FULL_STEPS = 156250  # 40M examples / global batch 256, Table 6
GENERATOR_COMMIT = "be54bd3f9cfb9f17c80a8d956ea827f5cf93d07d"
GENERATOR_URL = f"https://raw.githubusercontent.com/deep-spin/asentmax/{GENERATOR_COMMIT}/scripts/generate_data.py"
GENERATOR_SHA256 = "a603d5fbf08a9aad2f636a2f6a8397dae519a3ad17fa2d20db31a5054b0009e8"


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def prepare(args):
    import numpy as np
    root = args.data
    assert not root.exists(), "Choose a new data directory; existing data are never overwritten."
    root.mkdir(parents=True)
    generator = root / "generate_data.py"
    if args.generator is not None:
        shutil.copyfile(args.generator, generator)
    else:
        with urlopen(GENERATOR_URL, timeout=60) as response:
            generator.write_bytes(response.read())
    assert sha256(generator) == GENERATOR_SHA256, "Author generator checksum mismatch."
    # Only borrow the generator. Lengths and vocabulary follow Sparse Attention,
    # Table 6 / Appendix B.3, rather than the asentmax README training example.
    command = [sys.executable, str(generator), "--task_type", "sort", "--out_dir", str(root),
               "--vocab_size", "32", "--seq_len", "64", "--vary_len", "0",
               "--mdps_seq_len", "64 128 256", "--mdps_vary_len", "0 0 0",
               "--mdps_file_format", "{split}_{val}", "--seed", str(args.seed),
               "--train_size", str(args.examples), "--dev_size", str(args.test_examples),
               "--test_size", str(args.test_examples)]
    subprocess.run(command, check=True)
    shards = {}
    specifications = [("train", 64, args.examples, args.seed)]
    specifications += [(f"{split}_{n}", n, args.test_examples, args.seed)
                       for split in ("validation", "test") for n in (64, 128, 256)]
    for name, n, count, seed in specifications:
        for suffix in ("src", "trg"):
            path = root / f"{name}.{suffix}"
            with path.open("rb") as stream, Path(f"{path}.idx").open("wb") as index:
                lines = 0
                while True:
                    offset = stream.tell()
                    if not stream.readline():
                        break
                    index.write(struct.pack("Q", offset))
                    lines += 1
            assert lines == count, f"Unexpected number of lines in {path}"
        shards[name] = {"length": n, "examples": count, "seed": seed}
    files = {p.name: {"bytes": p.stat().st_size, "sha256": sha256(p)} for p in root.iterdir()}
    write_json(root / "manifest.json", {
        "task": "sort", "provenance": "Unmodified author asentmax Sort generator; fixed length 64 from Sparse Attention Table 6",
        "generator": {"url": GENERATOR_URL, "commit": GENERATOR_COMMIT,
                      "sha256": GENERATOR_SHA256, "command": command},
        "generation": {"vocab_size": 32, "seq_len": 64, "vary_len": 0,
                       "evaluation_lengths": [64, 128, 256], "seed": args.seed},
        "numpy_version": np.__version__, "shards": shards, "files": files,
        "format": "Author IndexedPairDataset: text pairs and native uint64 byte offsets",
    })
    check_data(root, args.library)
    print(f"Data validated: {root}", flush=True)


def check_data(root, library):
    sys.path.insert(0, str(library))
    from memory_mosaics.data.dataset_synthetic import IndexedPairDataset
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["generator"]["sha256"] == GENERATOR_SHA256, "Recreate data using the pinned author generator."
    assert manifest["generation"]["seq_len"] == 64 and manifest["generation"]["vary_len"] == 0
    for name, spec in manifest["files"].items():
        p = root / name
        assert p.stat().st_size == spec["bytes"] and sha256(p) == spec["sha256"], name
    for name, spec in manifest["shards"].items():
        dataset = IndexedPairDataset(str(root / f"{name}.src"), str(root / f"{name}.trg"), "sort")
        assert dataset.src_lines == dataset.trg_lines == spec["examples"]
        # Small probes/tests are checked exhaustively. A 1.536M-example training
        # run must not reopen every line in each DDP worker before each method.
        indices = range(len(dataset)) if len(dataset) <= 4096 else sorted({
            i * (len(dataset) - 1) // 127 for i in range(128)})
        for i in indices:
            x, y, ctx = dataset[i]
            n = spec["length"]
            assert ctx == 2 * n + 3 and len(x) == len(y) == 2 * n + 2
            assert x[0].item() == 32 and x[n + 1].item() == 33 and y[-1].item() == 34
            src, trg = x[1:n + 1].tolist(), y[n + 1:2 * n + 1].tolist()
            assert all(0 <= v < 32 for v in src) and trg == sorted(src)
    return manifest


def config(args):
    return dict(method=args.method, task="sort", vocab_size=36, n_layer=2, n_head=8,
                n_embd=512, pmem_size=2688, block_size=512, skip_tokens=0,
                attention_backend="adasplash", dtype="float16", compile=False,
                seed=args.seed, learning_rate=1e-3, min_lr=5e-5, weight_decay=0.1,
                beta1=0.9, beta2=0.95, grad_clip=1.0, warmup_iters=getattr(args, "lr_warmup", 40000),
                lr_decay_iters=getattr(args, "lr_decay", FULL_STEPS), global_batch=256,
                micro_batch=args.micro_batch, probe_steps=args.steps,
                measurement_warmup=args.warmup)


def make_model(conf, library, device):
    import torch
    sys.path.insert(0, str(library))
    from memory_mosaics.models.memory_mosaics_eft import StackAssoMem
    # Match the synthetic trainer's SDPA backend for a comparable speed estimate.
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    model = StackAssoMem(**{k: conf[k] for k in
        ("method", "task", "vocab_size", "n_layer", "n_head", "n_embd", "pmem_size", "block_size")},
        config=SimpleNamespace(**conf))
    model.to(device)  # Author .to() returns None.
    return model


def train(args):
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    from torch.utils.data import DataLoader, DistributedSampler
    sys.path.insert(0, str(args.library))
    from memory_mosaics.data.dataset_synthetic import IndexedPairDataset, make_indexedpair_collate
    from utils import get_cosine_lr
    rank, world, local = (int(os.environ[k]) for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK"))
    assert world == 2, "This probe requires two GPUs."
    torch.cuda.set_device(local)
    dist.init_process_group("nccl")
    device = torch.device("cuda", local)
    try:
        conf = config(args)
        assert 256 % (world * args.micro_batch) == 0
        accumulation = 256 // (world * args.micro_batch)
        manifest = check_data(args.data, args.library)
        assert manifest["shards"]["train"]["examples"] == args.steps * 256
        if rank == 0:
            args.output_dir.mkdir(parents=True, exist_ok=False)
            write_json(args.output_dir / "run_config.json", {
                "config": conf, "dataset": manifest, "world_size": world, "accumulation": accumulation,
                "deviations": ["Data regenerated by author asentmax generator; original Sparse Attention data unavailable",
                               "Uniform sampling follows the generator; paper does not specify the sampling distribution",
                               "Reduced training set and one seed",
                               f"LR schedule: warmup {conf['warmup_iters']}, decay {conf['lr_decay_iters']} (author defaults: 40000 / 156250)",
                               "FP16 on T4 instead of README BF16", "torch.compile disabled for probe and its projections",
                               "Author debug anomaly detection disabled; loss and gradients explicitly checked",
                               "Gradient accumulation preserves global batch 256; each microbatch is distinct"],
                "torch_version": torch.__version__, "gpu": torch.cuda.get_device_name(local),
            })
        dist.barrier()
        started = time.perf_counter()
        torch.manual_seed(args.seed)  # Same initialization on both ranks.
        model = make_model(conf, args.library, device)
        optimizer = model.configure_optimizers(0.1, 1e-3, (0.9, 0.95), "cuda")
        wrapped = DistributedDataParallel(model, device_ids=[local], find_unused_parameters=True)
        scaler = torch.amp.GradScaler("cuda")
        dataset = IndexedPairDataset(str(args.data / "train.src"), str(args.data / "train.trg"), "sort")
        loader = DataLoader(dataset, batch_size=args.micro_batch,
                            sampler=DistributedSampler(dataset, shuffle=True, seed=args.seed),
                            num_workers=0, collate_fn=make_indexedpair_collate(35), pin_memory=True)
        batches = iter(loader)
        torch.manual_seed(args.seed + rank)  # Independent dropout after DDP initialization.
        torch.cuda.synchronize()
        setup_seconds = time.perf_counter() - started
        rows = []
        for step in range(args.steps):
            torch.cuda.synchronize()
            tic = time.perf_counter()
            lr = get_cosine_lr(step, conf["learning_rate"], conf["min_lr"], conf["lr_decay_iters"], conf["warmup_iters"])
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for micro in range(accumulation):
                wrapped.require_backward_grad_sync = micro == accumulation - 1
                x, y, ctx = next(batches)
                x, y, ctx = (v.to(device, non_blocking=True) for v in (x, y, ctx))
                with torch.autocast("cuda", dtype=torch.float16):
                    _, loss = wrapped(x, y, ctx=ctx)
                assert torch.isfinite(loss).item(), f"Nonfinite loss at step {step}"
                losses.append(loss.detach())
                scaler.scale(loss / accumulation).backward()
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            finite_gradients = torch.isfinite(norm).item()
            assert finite_gradients or step < args.warmup, f"Nonfinite gradients at measured step {step}"
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            skipped = scaler.get_scale() < old_scale
            assert not skipped or step < args.warmup, "FP16 overflow during measured steps"
            torch.cuda.synchronize()
            seconds = torch.tensor(time.perf_counter() - tic, device=device)
            dist.all_reduce(seconds, op=dist.ReduceOp.MAX)
            mean_loss = torch.stack(losses).mean()
            dist.all_reduce(mean_loss)
            row = dict(step=step + 1, loss=mean_loss.item() / world, lr=lr,
                       seconds=seconds.item(), optimizer_skipped=skipped)
            assert finite_gradients or skipped, "Nonfinite gradients must be rejected by GradScaler."
            rows.append(row)
            if rank == 0:
                if step == 0 or (step + 1) % args.log_interval == 0 or step + 1 == args.steps:
                    print(f"{args.method}: step {step + 1}/{args.steps}, loss {row['loss']:.4f}, lr {lr:.6g}, {row['seconds']:.3f} s", flush=True)
        dist.barrier()
        if rank == 0:
            tic = time.perf_counter()
            torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
                            model_args=conf, iter_num=args.steps), args.output_dir / "ckpt.pt")
            checkpoint_seconds = time.perf_counter() - tic
            with (args.output_dir / "steps.csv").open("w") as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            write_json(args.output_dir / "timing.json", {
                "steps": rows, "setup_seconds": setup_seconds, "checkpoint_seconds": checkpoint_seconds,
                "step_median_seconds": statistics.median(r["seconds"] for r in rows[args.warmup:]),
                "peak_gpu_gib_rank0": torch.cuda.max_memory_allocated() / 2**30,
            })
        dist.barrier()
    finally:
        dist.destroy_process_group()


def evaluate(args):
    import torch
    from torch.utils.data import DataLoader
    sys.path.insert(0, str(args.library))
    from memory_mosaics.data.dataset_synthetic import IndexedPairDataset, make_indexedpair_collate
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    started = time.perf_counter()
    ckpt = torch.load(args.output_dir / "ckpt.pt", map_location="cpu", weights_only=False)
    model = make_model(ckpt["model_args"], args.library, device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    manifest = check_data(args.data, args.library)
    results = []
    for n in (64, 128, 256):
        dataset = IndexedPairDataset(str(args.data / f"test_{n}.src"), str(args.data / f"test_{n}.trg"), "sort")
        loader = DataLoader(dataset, batch_size=args.eval_batch, shuffle=False,
                            collate_fn=make_indexedpair_collate(35), num_workers=0)
        predictions, targets, losses = [], [], []
        tic = time.perf_counter()
        with torch.inference_mode():
            for x, y, ctx in loader:
                x, y, ctx = (v.to(device) for v in (x, y, ctx))
                with torch.autocast("cuda", dtype=torch.float16):
                    logits, loss = model(x, y, ctx=ctx)
                assert logits.shape == (*y.shape, 36) and torch.isfinite(logits).all().item()
                assert torch.isfinite(loss).item()
                # Teacher-forced exact match, as in the authors' test_synthetic.py;
                # includes all n output tokens, excludes EOS and input tokens.
                predictions.append(logits.argmax(-1)[:, n + 1:2 * n + 1].cpu())
                targets.append(y[:, n + 1:2 * n + 1].cpu())
                losses.append((loss.item(), x.size(0)))
        pred, target = torch.cat(predictions), torch.cat(targets)
        assert pred.shape == target.shape == (manifest["shards"][f"test_{n}"]["examples"], n)
        torch.save({"predictions": pred, "targets": target, "length": n,
                    "evaluation": "teacher_forced_output_tokens_excluding_EOS"}, args.output_dir / f"predictions_{n}.pt")
        row = {"method": ckpt["model_args"]["method"], "length": n, "multiplier": n // 64,
               "examples": len(dataset), "loss": sum(v * k for v, k in losses) / len(dataset),
               "exact_accuracy": (pred == target).all(1).float().mean().item(),
               "token_accuracy": (pred == target).float().mean().item(),
               "seconds": time.perf_counter() - tic}
        results.append(row)
        print(f"length {n}: exact={row['exact_accuracy']:.4f}, {row['seconds']:.1f} s", flush=True)
    write_json(args.output_dir / "evaluation.json", {"results": results, "wall_seconds": time.perf_counter() - started})


def validate_run(run, data):
    import torch
    run = Path(run)
    settings = json.loads((run / "run_config.json").read_text())
    conf = settings["config"]
    assert (conf["task"], conf["n_layer"], conf["n_embd"], conf["n_head"], conf["global_batch"]) == ("sort", 2, 512, 8, 256)
    assert conf["method"] in METHODS
    checkpoint = torch.load(run / "ckpt.pt", map_location="cpu", weights_only=False)
    assert checkpoint["iter_num"] == conf["probe_steps"] and checkpoint["model_args"] == conf
    assert checkpoint["model"] and checkpoint["optimizer"]["state"] and checkpoint["scaler"]
    for tensor in checkpoint["model"].values():
        assert torch.isfinite(tensor).all().item()
    for state in checkpoint["optimizer"]["state"].values():
        for value in state.values():
            if torch.is_tensor(value):
                assert torch.isfinite(value).all().item()
    timing = json.loads((run / "timing.json").read_text())
    with (run / "steps.csv").open() as f:
        steps = list(csv.DictReader(f))
    assert len(steps) == len(timing["steps"]) == conf["probe_steps"]
    assert [int(row["step"]) for row in steps] == list(range(1, conf["probe_steps"] + 1))
    assert all(math.isfinite(float(row["loss"])) and float(row["seconds"]) > 0 for row in steps)
    assert not any(row["optimizer_skipped"] for row in timing["steps"][conf["measurement_warmup"]:])
    assert abs(timing["step_median_seconds"] - statistics.median(
        row["seconds"] for row in timing["steps"][conf["measurement_warmup"]:])) < 1e-7
    evaluation = json.loads((run / "evaluation.json").read_text())
    assert [r["length"] for r in evaluation["results"]] == [64, 128, 256]
    manifest = json.loads((data / "manifest.json").read_text())
    assert manifest == settings["dataset"]
    for row in evaluation["results"]:
        n = row["length"]
        saved = torch.load(run / f"predictions_{n}.pt", map_location="cpu", weights_only=False)
        pred, target = saved["predictions"], saved["targets"]
        assert pred.shape == target.shape == (row["examples"], n)
        assert row["examples"] == manifest["shards"][f"test_{n}"]["examples"]
        assert pred.dtype == target.dtype == torch.int64
        assert ((pred >= 0) & (pred < 36)).all().item()
        expected = torch.tensor([list(map(int, s.split())) for s in (data / f"test_{n}.trg").read_text().splitlines()])
        assert torch.equal(target, expected)
        assert row["method"] == conf["method"] and math.isfinite(row["loss"])
        assert abs(row["exact_accuracy"] - (pred == target).all(1).float().mean().item()) < 1e-7
        assert abs(row["token_accuracy"] - (pred == target).float().mean().item()) < 1e-7
    validation = {"valid": True, "files": {
        p.name: {"bytes": p.stat().st_size, "sha256": sha256(p)}
        for p in run.iterdir() if p.is_file() and p.name != "artifact_validation.json"}}
    write_json(run / "artifact_validation.json", validation)
    return conf, timing, evaluation


def summarize(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    check_data(args.data, args.library)
    rows, estimates = [], []
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for method in METHODS:
        conf, timing, evaluation = validate_run(args.output_dir / method, args.data)
        assert conf["method"] == method
        rows.extend(evaluation["results"])
        steady = timing["step_median_seconds"]
        wall_path = args.output_dir / method / "wall_timing.json"
        if wall_path.exists():
            wall = json.loads(wall_path.read_text())
            train_fixed = max(0, wall["train_seconds"] - conf["probe_steps"] * steady)
            eval_wall = wall["evaluation_seconds"]
        else:
            train_fixed = timing["setup_seconds"] + timing["checkpoint_seconds"] + max(
                0, sum(r["seconds"] for r in timing["steps"]) - conf["probe_steps"] * steady)
            eval_wall = evaluation["wall_seconds"]
        # Scale measured inference to the planned final test size; keep process
        # startup/model loading costs fixed. This remains a short-probe estimate.
        eval_fixed = max(0, eval_wall - sum(r["seconds"] for r in evaluation["results"]))
        projected_eval = eval_fixed + sum(
            r["seconds"] * args.evaluation_examples / r["examples"] for r in evaluation["results"])
        fixed = train_fixed + projected_eval
        per_method_seconds = args.budget_minutes * 60 / len(METHODS)
        estimates.append({"method": method, "step_seconds": steady,
                          "full_40M_hours": (FULL_STEPS * steady + fixed) / 3600,
                          "steps_in_method_budget": min(FULL_STEPS, max(0, int((per_method_seconds - fixed) / steady))),
                          "fixed_seconds": fixed, "projected_evaluation_seconds": projected_eval,
                          "evaluation_seconds": eval_wall})
        ax.plot([r["length"] for r in evaluation["results"]],
                [r["exact_accuracy"] for r in evaluation["results"]], "o-", label=method)
    for name, table in (("results_summary.csv", rows), ("timing_summary.csv", estimates)):
        with (args.output_dir / name).open("w") as f:
            writer = csv.DictWriter(f, fieldnames=table[0].keys())
            writer.writeheader()
            writer.writerows(table)
    ax.set(xlabel="Input length", ylabel="Exact match (teacher forced)", ylim=(-0.02, 1.02), xticks=[64, 128, 256])
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(args.output_dir / f"length_accuracy.{suffix}", dpi=160)
    plt.close(fig)
    common_steps = min(r["steps_in_method_budget"] for r in estimates)
    write_json(args.output_dir / "budget_projection.json", {
        "total_budget_minutes": args.budget_minutes, "evaluation_examples_per_length": args.evaluation_examples,
        "common_steps": common_steps, "training_examples_per_method": common_steps * 256,
        "projected_total_hours": sum(common_steps * r["step_seconds"] + r["fixed_seconds"] for r in estimates) / 3600,
        "lr_schedule": {"warmup_iters": conf["warmup_iters"], "lr_decay_iters": conf["lr_decay_iters"]},
        "estimates": estimates,
    })
    run_kind = getattr(args, "kind", "timing-smoke")
    lines = ["Проверка артефактов sort завершена. " + ("Короткий запуск не оценивает качество воспроизведения." if run_kind == "timing-smoke" else "Результаты сокращённого обучения; один seed."),
             "", f"Общий бюджет: {args.budget_minutes:g} мин; тест: {args.evaluation_examples} примеров на длину.",
             "", "| Метод | с/шаг после прогрева | 40M примеров, ч | шагов в бюджете метода |", "|---|---:|---:|---:|"]
    for r in estimates:
        lines.append(f"| {r['method']} | {r['step_seconds']:.3f} | {r['full_40M_hours']:.2f} | {r['steps_in_method_budget']} |")
    lines += ["", f"Общее число шагов по самому медленному методу: {common_steps}; примеров на метод: {common_steps * 256}.",
              "Прогноз включает запуск, прогрев, сохранение и экстраполяцию финальной оценки,",
              "но не дополнительные validation/checkpoints и не генерацию полного датасета.",
              "Оценка по короткому замеру после прогрева приблизительная; оставьте запас времени.",
              "Архитектура соответствует статье; авторский генератор asentmax, train length 64, один seed, FP16, compile=False.",
              f"LR schedule: warmup {conf['warmup_iters']}, decay {conf['lr_decay_iters']}; run kind: {run_kind}."]
    (args.output_dir / "results_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "train", "evaluate", "summarize"))
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--method", choices=METHODS, default="softmax")
    parser.add_argument("--seed", type=int, default=1338)
    parser.add_argument("--examples", type=int, default=6144)
    parser.add_argument("--test-examples", type=int, default=128)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--micro-batch", type=int, default=32)
    parser.add_argument("--eval-batch", type=int, default=8)
    parser.add_argument("--generator", type=Path, help="Offline copy of the pinned author generate_data.py; checksum is verified")
    parser.add_argument("--budget-minutes", type=float, default=150, help="Total budget across the three methods")
    parser.add_argument("--evaluation-examples", type=int, default=1000, help="Planned final test size per length for projection")
    parser.add_argument("--lr-warmup", type=int, default=40000)
    parser.add_argument("--lr-decay", type=int, default=FULL_STEPS)
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--kind", choices=("timing-smoke", "budget-training"), default="timing-smoke")
    args = parser.parse_args()
    assert 0 <= args.warmup < args.steps and args.examples > 0 and args.test_examples > 0
    assert args.budget_minutes > 0 and args.evaluation_examples > 0
    assert 0 <= args.lr_warmup < args.lr_decay and args.log_interval > 0
    if args.action != "prepare":
        assert args.output_dir is not None
    globals()[args.action](args)


if __name__ == "__main__":
    main()
