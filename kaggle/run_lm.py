"""Kaggle entry point for LM; shared setup utilities for the second job.

Local: --plan prints settings; --package DIR creates a pinned API upload.
Kaggle: packaged script defaults to smoke, never starts full training implicitly.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from urllib.request import urlretrieve
from zipfile import ZipFile, ZIP_STORED

RUN_MODE = "smoke"
REPO_REVISION = "main"  # --package replaces this with the local committed SHA.
REPO_URL = "https://github.com/kostkornilov/hw2.git"


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def configuration(repo, mode):
    config = json.loads((repo / "kaggle/config.json").read_text())
    return config, config["modes"][mode]


def package(entry, directory, mode, revision=None):
    """Package only the entry script and metadata; config is read from pinned git."""
    repo = entry.resolve().parent.parent
    sha = revision or subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    if len(sha) != 40 or any(c not in "0123456789abcdef" for c in sha):
        raise ValueError("Package revision must be a full git commit SHA")
    # Refuse to package uncommitted pipeline changes; notebook/presentation edits
    # are unrelated and deliberately excluded from this check.
    changed = subprocess.check_output(["git", "status", "--porcelain", "--", "kaggle", "scripts"], cwd=repo, text=True)
    if changed.strip():
        raise ValueError("Commit kaggle/ and scripts/ changes before packaging")
    text = entry.read_text().replace('RUN_MODE = "smoke"', f'RUN_MODE = "{mode}"', 1)
    text = text.replace('REPO_REVISION = "main"', f'REPO_REVISION = "{sha}"', 1)
    directory.mkdir(parents=True, exist_ok=False)
    (directory / entry.name).write_text(text)
    group = "lm" if entry.name == "run_lm.py" else "icl_sort"
    metadata = json.loads((repo / "kaggle" / group / "kernel-metadata.json").read_text())
    metadata["code_file"] = entry.name
    save_json(directory / "kernel-metadata.json", metadata)
    print(f"Packaged {group} / {mode} / {sha}: {directory}")


def checkout(group, revision):
    base = Path("/tmp/hw2-runtime") / group
    repo = base / "repo"
    base.mkdir(parents=True, exist_ok=True)
    if not repo.exists():
        subprocess.run(["git", "clone", REPO_URL, str(repo)], check=True)
    subprocess.run(["git", "checkout", "--detach", revision], cwd=repo, check=True)
    return repo, base


class Job:
    def __init__(self, group, mode, repo, base):
        self.repo, self.base, self.mode, self.group = repo, base, mode, group
        self.config, self.limits = configuration(repo, mode)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        self.root = Path("/kaggle/working/results") / group / f"{mode}_{stamp}"
        self.root.mkdir(parents=True, exist_ok=False)
        self.library = repo / "sparse_kernel_regression/Library"
        self.python = base / "env/bin/python"
        self.env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONHASHSEED": "0",
                    "CUDA_VISIBLE_DEVICES": "0,1", "OMP_NUM_THREADS": "1", "MAX_JOBS": "2",
                    "UV_CACHE_DIR": str(base / "uv-cache"), "MPLCONFIGDIR": str(base / "mpl"),
                    "TOKENIZERS_PARALLELISM": "false"}
        self.commands = []
        self.started = time.monotonic()
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
        save_json(self.root / "configuration.json", {"mode": mode, "repository_commit": sha,
                  "config": self.config, "effective_limits": self.limits,
                  "deviations": {"lm": "1000 instead of 80000 steps; warmup 25 / decay 1000; one seed; FP16",
                                 "icl": "Only train 1k; one seed; two T4 instead of author A6000",
                                 "sort": "6000 instead of 156250 steps; warmup 1536 / decay 6000; regenerated author data; one seed; FP16; compile=False"}})
        self.status("running")

    def status(self, status, **extra):
        save_json(self.root / "job_status.json", {"status": status, "mode": self.mode,
                  "elapsed_seconds": time.monotonic() - self.started, **extra})

    def run(self, name, command, cwd=None, single_gpu=False):
        command = [str(item) for item in command]
        log = self.root / "logs" / f"{name}.log"
        log.parent.mkdir(exist_ok=True)
        record = {"stage": name, "command": command, "cwd": str(cwd or self.repo)}
        self.commands.append(record)
        save_json(self.root / "commands.json", self.commands)
        print(f"\n=== {self.group}: {name} ===", flush=True)
        started = time.monotonic()
        env = {**self.env, "CUDA_VISIBLE_DEVICES": "0"} if single_gpu else self.env
        with log.open("w") as stream:
            process = subprocess.Popen(command, cwd=cwd or self.repo, env=env, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, start_new_session=True)
            try:
                for line in process.stdout:
                    stream.write(line)
                    stream.flush()
                    print(line, end="", flush=True)
                code = process.wait()
                record.update(return_code=code, seconds=time.monotonic() - started)
                save_json(self.root / "commands.json", self.commands)
                if code:
                    raise subprocess.CalledProcessError(code, command)
            except BaseException:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                raise
        return record["seconds"]

    def environment(self, icl=False):
        self.run("install_uv", [sys.executable, "-m", "pip", "install", "uv"])
        if not self.python.exists():
            self.run("python311", [sys.executable, "-m", "uv", "venv", "--python", "3.11", "--seed", self.base / "env"])
        self.run("dependencies", [self.python, "-m", "pip", "install", "-r", self.library / "requirements.txt"])
        self.run("ninja", [self.python, "-m", "pip", "install", "ninja"])
        self.run("memory_mosaics", [self.python, "-m", "pip", "install", "-e", self.library, "--no-deps"])
        if icl:
            self.run("icl_dependencies", [self.python, "-m", "pip", "install", "-r", self.repo / "scripts/requirements-icl.txt"])
        self.run("gpu_check", [self.python, "-c",
            "import torch; from importlib.metadata import version; from adasplash import adasplash_no_block_mask; "
            "assert torch.cuda.device_count()==2; assert version('torch').split('+')[0]=='2.4.1'; "
            "assert version('adasplash')=='0.1.1b0'; assert version('triton')=='3.0.0'; "
            "print([torch.cuda.get_device_name(i) for i in range(2)])"])
        versions = subprocess.check_output([str(self.python), "-m", "pip", "freeze"], env=self.env, text=True)
        (self.root / "environment.txt").write_text(versions)

    def finish(self, error=None):
        self.status("failed" if error else "passed", error=str(error) if error else None)
        files = {}
        for path in sorted(self.root.rglob("*")):
            if path.is_file():
                h = hashlib.sha256()
                with path.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
                        h.update(chunk)
                files[str(path.relative_to(self.root))] = {"bytes": path.stat().st_size, "sha256": h.hexdigest()}
        save_json(self.root / "files_manifest.json", files)
        archive = self.root.with_suffix(".zip")
        with ZipFile(archive, "w", compression=ZIP_STORED) as zipfile:
            for path in sorted(self.root.rglob("*")):
                if path.is_file():
                    zipfile.write(path, path.relative_to(self.root))
        print(f"\nOutputs: {self.root}\nArchive: {archive}", flush=True)


def lm_data(job):
    data = job.base / "babistories"
    data.mkdir(exist_ok=True)
    base_url = "https://raw.githubusercontent.com/facebookresearch/MemoryMosaics/main/BabiStories/data"
    for i in range(1, 6):
        name = f"babistories-dataset.7z.{i:03d}"
        path = data / name
        if not path.exists():
            temporary = path.with_suffix(path.suffix + ".part")
            print(f"Downloading {name}", flush=True)
            urlretrieve(f"{base_url}/{name}", temporary)
            temporary.replace(path)
    if shutil.which("7z") is None:
        job.run("apt_update", ["sudo", "apt-get", "update"])
        job.run("archive_tool", ["sudo", "apt-get", "install", "-y", "p7zip-full"])
    job.run("extract_babistories", ["7z", "x", data / "babistories-dataset.7z.001", f"-o{data}", "-aos"])
    token_dir = job.library / "memory_mosaics/data/BabiStories"
    if not (token_dir / "dataset_stats.json").exists():
        job.run("tokenize", [job.python, token_dir / "prepare.py", "--train-file", data / "traindataset.txt",
                              "--val-file", data / "valdataset.txt", "--output-dir", token_dir])
    stats = json.loads((token_dir / "dataset_stats.json").read_text())
    for split, tokens in (("train", 474704907), ("val", 4749107)):
        assert stats[split]["tokens"] == tokens
        assert (token_dir / f"{split}.bin").stat().st_size == tokens * 2
    save_json(job.root / "dataset_stats.json", stats)


def lm(job):
    lm_data(job)
    root = job.root / "language_modeling"
    root.mkdir()
    for method in job.config["methods"]:
        run = root / "runs" / method
        run.mkdir(parents=True)
        params = {**job.config["language_modeling"], "method": method, "out_dir": str(root / "runs"),
                  "max_iters": job.limits["lm_steps"], "eval_interval": job.limits["lm_eval_interval"],
                  "save_checkpoint_interval": job.limits["lm_steps"]}
        save_json(run / "run_config.json", {"parameters": params, "gpu_ids": [0, 1],
                  "repository_commit": json.loads((job.root / "configuration.json").read_text())["repository_commit"]})
        command = [job.python, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2", "train_memory_mosaics.py"]
        for key, value in params.items():
            command += [f"--{key}", str(value)]
        job.run(f"lm_{method}", command, cwd=job.library)
    job.run("lm_export", [job.python, job.repo / "scripts/export_lm_artifacts.py", root,
                           "--kind", "timing-smoke" if job.mode == "smoke" else "budget-training"])


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), default=RUN_MODE)
    parser.add_argument("--package", type=Path, help="Prepare a local API upload; does not run training")
    parser.add_argument("--revision", help="Full git SHA for package; defaults to local HEAD")
    parser.add_argument("--plan", action="store_true", help="Print effective configuration without running")
    parser.add_argument("--check-output", type=Path, help="Validate a downloaded job ZIP locally, without training")
    return parser.parse_args()


def check_output(path):
    with ZipFile(path) as archive:
        manifest = json.loads(archive.read("files_manifest.json"))
        assert set(archive.namelist()) == set(manifest) | {"files_manifest.json"}
        for name, spec in manifest.items():
            h = hashlib.sha256()
            size = 0
            with archive.open(name) as stream:
                for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
                    size += len(chunk)
                    h.update(chunk)
            assert size == spec["bytes"] and h.hexdigest() == spec["sha256"], name
        status = json.loads(archive.read("job_status.json"))
        assert status["status"] == "passed", status
        config = json.loads(archive.read("configuration.json"))
        commands = json.loads(archive.read("commands.json"))
        assert commands and all(c.get("return_code") == 0 for c in commands)
        if "language_modeling/artifact_validation.json" in manifest:
            validation = json.loads(archive.read("language_modeling/artifact_validation.json"))
            assert validation["status"] == "passed"
            for method in config["config"]["methods"]:
                history = json.loads(archive.read(f"language_modeling/runs/{method}/loss_history.json"))
                assert history["iter"][-1] == config["effective_limits"]["lm_steps"]
        else:
            for method in config["config"]["methods"]:
                icl = json.loads(archive.read(f"in_context_learning/{method}/artifact_validation.json"))
                sort = json.loads(archive.read(f"length_generalization/{method}/artifact_validation.json"))
                assert icl["status"] == "passed" and sort["valid"]
                timing = json.loads(archive.read(f"in_context_learning/{method}/timing.json"))
                assert timing["probe_epochs"] == config["effective_limits"]["icl_epochs"]
                timing = json.loads(archive.read(f"length_generalization/{method}/timing.json"))
                assert len(timing["steps"]) == config["effective_limits"]["sort_steps"]
    print(f"Downloaded archive passed: {path}; mode={status['mode']}; {len(manifest)} files verified")


def main():
    args = arguments()
    if args.check_output:
        check_output(args.check_output)
        return
    if args.package:
        package(Path(__file__), args.package, args.mode, args.revision)
        return
    if args.plan:
        config, limits = configuration(Path(__file__).resolve().parent.parent, args.mode)
        print(json.dumps({"job": "lm", "mode": args.mode, "limits": limits, "config": config}, indent=2))
        return
    repo, base = checkout("lm", REPO_REVISION)
    job = Job("lm", args.mode, repo, base)
    try:
        job.environment()
        lm(job)
    except BaseException as error:
        job.finish(error)
        raise
    job.finish()


if __name__ == "__main__":
    main()
