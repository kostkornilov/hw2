# Copyright (c) Meta Platforms, Inc. and affiliates.
# See file LICENSE.txt in the main directory.
# Adapted for streaming tokenization without changing GPT-2/EOT token semantics.

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import tiktoken
from tqdm import tqdm


def tokenize_file(source, destination, encoding):
    stories = tokens = 0
    digest = hashlib.sha256()
    # Use a temporary output so interrupted tokenization is not mistaken for completion.
    temporary = destination.with_suffix(".bin.part")
    with source.open("rb") as inputs, temporary.open("wb") as outputs:
        for line in tqdm(inputs, desc=source.name, unit=" stories"):
            digest.update(line)
            text = json.loads(line)
            if not isinstance(text, str):
                raise ValueError(f"Expected a JSON string at story {stories + 1}")
            ids = encoding.encode_ordinary(text) + [encoding.eot_token]
            np.asarray(ids, dtype=np.uint16).tofile(outputs)
            stories += 1
            tokens += len(ids)
    temporary.replace(destination)
    return {"stories": stories, "tokens": tokens, "source": str(source),
            "source_sha256": digest.hexdigest()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", type=Path, default=Path("traindataset.txt"))
    parser.add_argument("--val-file", type=Path, default=Path("valdataset.txt"))
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    encoding = tiktoken.get_encoding("gpt2")
    stats = {"tokenizer": "gpt2", "dtype": "uint16", "eot_per_story": True}
    for split, source in [("train", args.train_file), ("val", args.val_file)]:
        stats[split] = tokenize_file(source, args.output_dir / f"{split}.bin", encoding)
    (args.output_dir / "dataset_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
