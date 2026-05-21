from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from glob import glob
from pathlib import Path

def extract_letter(text: str) -> str:
    raw = str(text).strip().upper()
    match = re.search(r"\b([A-H])\b", raw)
    return match.group(1) if match else raw[:1]

def iter_records(paths: list[str]):
    for item in paths:
        path = Path(item)
        files = sorted(glob(str(path / "*.json"))) if path.is_dir() else [str(path)]
        for file_path in files:
            if file_path.endswith("_summary.json"):
                continue
            try:
                with open(file_path, "r", encoding="utf-8") as handle:
                    data = json.load(handle)
            except Exception as exc:
                print(f"skip unreadable {file_path}: {exc}")
                continue
            if isinstance(data, list):
                yield from data
            else:
                yield data

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PathMMU predictions")
    parser.add_argument("paths", nargs="+", help="Result JSON file(s) or directory/directories")
    args = parser.parse_args()

    total = correct = 0
    by_subset: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    by_split: dict[str, list[int]] = defaultdict(lambda: [0, 0])

    for record in iter_records(args.paths):
        gt = extract_letter(record.get("ground_truth", record.get("Answer", record.get("answer", ""))))
        pred = extract_letter(record.get("pred_answer", record.get("prediction", "")))
        if not gt:
            continue
        is_correct = pred == gt
        total += 1
        correct += int(is_correct)
        subset = str(record.get("subset", "unknown"))
        split = str(record.get("split", "unknown"))
        by_subset[subset][1] += 1
        by_subset[subset][0] += int(is_correct)
        by_split[split][1] += 1
        by_split[split][0] += int(is_correct)

    print(f"PathMMU letter-exact: {correct}/{total} = {correct / max(total, 1) * 100:.2f}%")
    print("\nBy split:")
    for split, (split_correct, split_total) in sorted(by_split.items()):
        print(f"  {split:<10s} {split_correct:>5d}/{split_total:<5d} {split_correct / max(split_total, 1) * 100:6.2f}%")
    print("\nBy subset:")
    for subset, (subset_correct, subset_total) in sorted(by_subset.items()):
        print(f"  {subset:<12s} {subset_correct:>5d}/{subset_total:<5d} {subset_correct / max(subset_total, 1) * 100:6.2f}%")

if __name__ == "__main__":
    main()
