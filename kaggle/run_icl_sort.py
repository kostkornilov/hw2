"""Kaggle script: independent RegBench or sort, or both. Defaults to smoke.

Uses shared setup functions from the pinned repository's run_lm.py.
No new trainer or model implementation is introduced.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

RUN_MODE = "smoke"
TASK = "icl_sort"
REPO_REVISION = "main"  # --package replaces this with the local committed SHA.
REPO_URL = "https://github.com/kostkornilov/hw2.git"


def icl(job):
    source = job.base / "seq_icl"
    if not source.exists():
        job.run("seq_icl_clone", ["git", "clone", "https://github.com/berlino/seq_icl.git", source])
    commit = job.config["in_context_learning"]["source_commit"]
    job.run("seq_icl_checkout", ["git", "checkout", "--detach", commit], cwd=source)
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip() == commit
    author = job.repo / "sparse_kernel_regression/ICLL"
    for relative in ("train.py", "src/models/sequence/mm.py", "configs/experiment/dfa/mm.yaml"):
        target = source / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(author / relative, target)
    registry = source / "src/utils/registry.py"
    registration = '\nmodel["mm"] = "src.models.sequence.mm.StackAssoMem"\n'
    if registration not in registry.read_text():
        registry.write_text(registry.read_text() + registration)
    (source / "src/dataloaders/__init__.py").write_text("from . import dfa\nfrom .base import SequenceDataset\n")
    examples = job.limits.get("icl_train_examples", job.config["in_context_learning"]["examples"])
    eval_examples = job.limits["icl_eval_examples"]
    seq_len = job.limits.get("icl_sequence_length")
    cache = job.base / f"regbench/dfa_{examples}_eval{eval_examples}_len{seq_len or 'author'}_seed0.pt"
    helper = job.repo / "scripts/run_icl_probe.py"
    common = ["--source", source, "--cache", cache, "--examples", examples, "--eval-examples", eval_examples]
    if seq_len is not None:
        common += ["--sequence-length", seq_len]
    if not cache.exists():
        job.run("icl_data", [job.python, helper, "prepare", *common])
    root = job.root / "in_context_learning"
    root.mkdir()
    shutil.copy2(cache.with_suffix(".json"), root / "dataset_manifest.json")
    # Preserve the small cache as well, including DFA objects, for later analysis.
    shutil.copy2(cache, root / "dataset.pt")
    for method in job.config["methods"]:
        run = root / method
        args = [*common, "--output-dir", run, "--method", method]
        job.run(f"icl_{method}_train", [job.python, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2",
                helper, "train", *args, "--epochs", job.limits["icl_epochs"]])
        job.run(f"icl_{method}_evaluate", [job.python, helper, "evaluate", *args], single_gpu=True)
    job.run("icl_export", [job.python, helper, "summarize", "--source", source, "--output-dir", root,
                           "--kind", "timing-smoke" if job.mode == "smoke" else "budget-training"])


def sort(job):
    settings = job.config["length_generalization"]
    root = job.root / "length_generalization"
    root.mkdir()
    helper = job.repo / "scripts/run_length_probe.py"
    data = job.base / f"sort_{job.mode}"
    common = ["--library", job.library, "--data", data]
    if not data.exists():
        job.run("sort_data", [job.python, helper, "prepare", *common, "--examples", job.limits["sort_steps"] * 256,
                              "--test-examples", job.limits["sort_test_examples"], "--seed", settings["seed"]])
    manifest = json.loads((data / "manifest.json").read_text())
    assert manifest["shards"]["train"]["examples"] == job.limits["sort_steps"] * 256
    shutil.copy2(data / "manifest.json", root / "dataset_manifest.json")
    shutil.copy2(data / "generate_data.py", root / "author_generate_data.py")
    # Keep test sequences for independent checks; large training data can be
    # regenerated from the exact source, seed and parameters in the manifest.
    for path in data.glob("test_*"):
        target = root / "test_data" / path.name
        target.parent.mkdir(exist_ok=True)
        shutil.copy2(path, target)
    for method in job.config["methods"]:
        run = root / method
        train_seconds = job.run(f"sort_{method}_train", [job.python, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2",
            helper, "train", *common, "--output-dir", run, "--method", method, "--steps", job.limits["sort_steps"],
            "--warmup", job.limits["sort_measurement_warmup"], "--micro-batch", settings["micro_batch"],
            "--seed", settings["seed"], "--lr-warmup", settings["lr_warmup"], "--lr-decay", settings["lr_decay"],
            "--log-interval", 1 if job.mode == "smoke" else 100])
        evaluation_seconds = job.run(f"sort_{method}_evaluate", [job.python, helper, "evaluate", *common,
                                    "--output-dir", run, "--eval-batch", settings["eval_batch"]], single_gpu=True)
        (run / "wall_timing.json").write_text(json.dumps({"train_seconds": train_seconds, "evaluation_seconds": evaluation_seconds}, indent=2))
    job.run("sort_export", [job.python, helper, "summarize", *common, "--output-dir", root,
            "--budget-minutes", settings["budget_minutes"], "--evaluation-examples", 1000,
            "--kind", "timing-smoke" if job.mode == "smoke" else "budget-training"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), default=RUN_MODE)
    parser.add_argument("--task", choices=("icl", "sort", "icl_sort"), default=TASK)
    parser.add_argument("--package", type=Path)
    parser.add_argument("--revision")
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--check-output", type=Path)
    args = parser.parse_args()
    if args.package or args.plan or args.check_output:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
    else:
        base = Path("/tmp/hw2-runtime") / args.task
        repo = base / "repo"
        base.mkdir(parents=True, exist_ok=True)
        if not repo.exists():
            subprocess.run(["git", "clone", REPO_URL, str(repo)], check=True)
        subprocess.run(["git", "checkout", "--detach", REPO_REVISION], cwd=repo, check=True)
        sys.path.insert(0, str(repo / "kaggle"))
    from run_lm import Job, package, configuration, check_output
    if args.check_output:
        check_output(args.check_output)
        return
    if args.package:
        package(Path(__file__), args.package, args.mode, args.revision, group=args.task)
        return
    if args.plan:
        config, limits = configuration(Path(__file__).resolve().parent.parent, args.mode)
        print(json.dumps({"job": args.task, "mode": args.mode, "limits": limits, "config": config}, indent=2))
        return
    job = Job(args.task, args.mode, repo, base)
    try:
        job.environment(icl=args.task != "sort")
        if args.task != "sort":
            icl(job)
        if args.task != "icl":
            sort(job)
    except BaseException as error:
        job.finish(error)
        raise
    job.finish()


if __name__ == "__main__":
    main()
