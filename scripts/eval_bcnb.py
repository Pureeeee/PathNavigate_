import argparse
import json
import os
import sys
from collections import defaultdict
from glob import glob
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_processing.utils import load_all_vqa_pairs

TASKS_ORDER = [
    "Tumor",
    "ER",
    "PR",
    "HER2",
    "HER2 Expression",
    "Histological grading",
    "Molecular subtype",
]

def load_records(save_dir):
    for p in sorted(glob(os.path.join(save_dir, "*.json"))):
        if p.endswith("_summary.json"):
            continue
        try:
            with open(p, encoding="utf-8") as handle:
                record = json.load(handle)
            record["_result_path"] = p
            yield record
        except Exception as e:
            print(f"  skip unreadable {p}: {e}")

def load_reference(path, dataset_name):
    if not path:
        return {}
    refs = {}
    for item in load_all_vqa_pairs(path, dataset_name=dataset_name):
        key = (item["long_id"], item["question"])
        if key in refs:
            raise ValueError(f"Duplicate reference key for long_id={item['long_id']!r}, question={item['question']!r}")
        refs[key] = str(item["answer"]).strip()
    return refs

def reference_for(record, refs):
    if not refs:
        return ""
    question = str(record.get("question", ""))
    for long_id in (record.get("long_id", ""), record.get("wsi_full_id", "")):
        hit = refs.get((str(long_id), question))
        if hit:
            return hit
    return ""

def eval_records(records, refs):
    total = 0
    correct = 0
    by_task = defaultdict(lambda: [0, 0])
    unknown_task_examples = []

    for rec in records:
        gt = str(rec.get("ground_truth", "")).strip()
        if not gt:
            gt = reference_for(rec, refs)
        pred = str(rec.get("pred_answer", "")).strip()
        task = rec.get("inferred_bcnb_task") or "(unknown)"

        if not gt:
            raise ValueError(
                f"{rec.get('_result_path', '<record>')} has no ground_truth; pass --questions_file for evaluation join."
            )
        total += 1
        is_correct = (pred == gt)
        if is_correct:
            correct += 1
        by_task[task][1] += 1
        if is_correct:
            by_task[task][0] += 1
        if task == "(unknown)" and len(unknown_task_examples) < 3:
            unknown_task_examples.append({
                "q": (rec.get("question", "") or "")[:80],
                "gt": gt,
                "pred": pred,
            })

    if total == 0:
        raise RuntimeError("No evaluable records found.")
    return total, correct, dict(by_task), unknown_task_examples

def print_report(label, total, correct, by_task, unknown_examples):
    print(f"\n=== {label} ===")
    acc = correct / total if total else 0.0
    print(f"Total letter-exact: {correct}/{total} = {acc*100:.2f}%")
    print()
    print(f"  {'Task':<24s}  {'Acc':>6s}  {'n_correct':>10s}/{'n_total':<7s}")
    print(f"  {'-'*24}  {'-'*6}  {'-'*10} {'-'*7}")
    for t in TASKS_ORDER:
        c, n = by_task.get(t, [0, 0])
        a = c / n if n else 0.0
        print(f"  {t:<24s}  {a*100:>5.1f}%  {c:>10d}/{n:<7d}")
    other_tasks = set(by_task) - set(TASKS_ORDER)
    for t in sorted(other_tasks):
        c, n = by_task[t]
        a = c / n if n else 0.0
        print(f"  {t:<24s}  {a*100:>5.1f}%  {c:>10d}/{n:<7d}")
    if unknown_examples:
        print()
        print("  (3 sample (unknown)-task records for debugging):")
        for ex in unknown_examples:
            print(f"    q='{ex['q']}' gt={ex['gt']} pred={ex['pred']}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+",
                    help="Result directories. Use --merge to pool all into one eval.")
    ap.add_argument("--merge", action="store_true",
                    help="Pool all dirs into a single eval instead of per-dir rows.")
    ap.add_argument("--label", default=None,
                    help="Label for --merge mode (defaults to 'merged').")
    ap.add_argument("--questions_file", default="",
                    help="Annotation JSON used to join ground truth when result JSON files do not store it.")
    ap.add_argument("--dataset_name", default="bcnb",
                    help="Dataset parser name passed to load_all_vqa_pairs.")
    args = ap.parse_args()
    refs = load_reference(args.questions_file, args.dataset_name)

    if args.merge:
        all_records = []
        for d in args.dirs:
            all_records.extend(load_records(d))
        label = args.label or f"merged ({len(args.dirs)} dirs)"
        total, correct, by_task, unk = eval_records(all_records, refs)
        print_report(label, total, correct, by_task, unk)
    else:
        for d in args.dirs:
            records = list(load_records(d))
            total, correct, by_task, unk = eval_records(records, refs)
            print_report(d, total, correct, by_task, unk)

if __name__ == "__main__":
    main()
