import argparse
import difflib
import hashlib
import json
import os
import re
import sys
from pathlib import Path

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_processing.utils import load_all_vqa_pairs

CHOICE_MARKER_RE = re.compile(r'(?:^|\s)([A-H])\.\s*')

def make_unique_id(long_id, question):
    return f"{long_id}_{hashlib.md5(question.encode('utf-8')).hexdigest()[:8]}"

def parse_choices_str(value):
    if isinstance(value, list):
        return [str(item).strip() for item in value]
    if not isinstance(value, str) or not value.strip():
        return None
    matches = list(CHOICE_MARKER_RE.finditer(value))
    if not matches:
        return None
    parts = []
    for idx, match in enumerate(matches):
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(value)
        parts.append(value[match.end():end].strip())
    return parts

def expand_letter(text, choices):
    if not choices:
        return text
    value = str(text).strip().upper()
    if len(value) == 1 and value in "ABCDEFGH":
        idx = ord(value) - ord("A")
        if 0 <= idx < len(choices):
            return choices[idx]
    return str(text)

def seqmatch_correct(choices, gt, pred):
    if not choices:
        return False
    score = difflib.SequenceMatcher(None, pred, gt).quick_ratio()
    for item in choices:
        if difflib.SequenceMatcher(None, pred, item).quick_ratio() > score:
            return False
    return True

def load_reference(path, dataset_name):
    if not path:
        return {}
    refs = {}
    for item in load_all_vqa_pairs(path, dataset_name=dataset_name):
        key = (item["long_id"], item["question"])
        payload = {"answer": item["answer"], "choices": item["choices"], "is_mcq": item["is_mcq"]}
        if key in refs:
            raise ValueError(f"Duplicate reference key for long_id={item['long_id']!r}, question={item['question']!r}")
        refs[key] = payload
    return {"by_key": refs}

def reference_for(record, refs):
    if not refs:
        return None
    question = str(record.get("question", ""))
    long_ids = [str(record.get("long_id", "")), str(record.get("wsi_full_id", ""))]
    for long_id in long_ids:
        hit = refs["by_key"].get((long_id, question))
        if hit:
            return hit
    return None

def files_in(directory):
    return [
        os.path.join(directory, name)
        for name in sorted(os.listdir(directory))
        if name.endswith(".json") and "_summary" not in name
    ]

def collect(files_iter, refs):
    open_gts, open_preds = {}, {}
    mcq_n = mcq_c = 0
    open_n = open_sub_c = 0
    n_files = 0
    n_eval = 0
    for path in files_iter:
        n_files += 1
        with open(path, encoding="utf-8") as handle:
            record = json.load(handle)
        ref = reference_for(record, refs)
        gt = str(record.get("ground_truth", "")).strip()
        choices_value = record.get("choices")
        is_mcq = bool(record.get("is_mcq", False))
        if ref:
            gt = gt or str(ref["answer"]).strip()
            choices_value = choices_value or ref["choices"]
            is_mcq = bool(is_mcq or ref["is_mcq"])
        if not gt:
            raise ValueError(f"{path} has no ground_truth; pass --questions_file for evaluation join.")
        pred = str(record.get("pred_answer", "")).strip()
        if not pred:
            continue
        choices = parse_choices_str(choices_value)
        if choices:
            is_mcq = True
        n_eval += 1
        if is_mcq:
            mcq_n += 1
            pred_expanded = expand_letter(pred, choices)
            gt_expanded = expand_letter(gt, choices)
            if pred not in ("?", "") and seqmatch_correct(choices, gt_expanded, pred_expanded):
                mcq_c += 1
        else:
            open_n += 1
            cid = make_unique_id(str(record.get("long_id") or record.get("wsi_full_id") or n_files), record.get("question", ""))
            open_gts[cid] = [gt]
            open_preds[cid] = [pred]
            if pred.lower() in gt.lower() or gt.lower() in pred.lower():
                open_sub_c += 1
    return open_gts, open_preds, mcq_n, mcq_c, open_n, open_sub_c, n_files, n_eval

def text_scores(gts, preds):
    if not gts:
        return [0.0, 0.0, 0.0, 0.0], 0.0
    bleu, _ = Bleu(4).compute_score(gts, preds)
    rouge, _ = Rouge().compute_score(gts, preds)
    return bleu, rouge

def eval_files(files, label, refs):
    gts, preds, mcq_n, mcq_c, open_n, open_sub_c, n_files, n_eval = collect(files, refs)
    bleu, rouge = text_scores(gts, preds)
    total_c = mcq_c + open_sub_c
    total_n = mcq_n + open_n
    print(f"\n=== {label} ===")
    print(f"files: {n_files}, eval: {n_eval}")
    print(f"Total:   {total_c}/{total_n} = {total_c/max(total_n,1)*100:.2f}%")
    print(f"  MCQ (SeqMatcher):  {mcq_c}/{mcq_n} = {mcq_c/max(mcq_n,1)*100:.2f}%")
    print(f"  Open (substring):  {open_sub_c}/{open_n} = {open_sub_c/max(open_n,1)*100:.2f}%")
    print(f"Open BLEU-1: {bleu[0]:.4f}  BLEU-4: {bleu[3]:.4f}  ROUGE-L: {rouge:.4f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--questions_file", default="")
    parser.add_argument("--dataset_name", default="wsi_vqa")
    parser.add_argument("targets", nargs="+")
    args = parser.parse_args()
    refs = load_reference(args.questions_file, args.dataset_name)
    if args.merge:
        label = args.targets[0].split("=", 1)[1] if "=" in args.targets[0] else "merged"
        files = []
        for directory in args.targets[1:]:
            files.extend(files_in(directory))
        eval_files(files, label, refs)
    else:
        for target in args.targets:
            directory, label = target.split("=", 1)
            eval_files(files_in(directory), label, refs)

if __name__ == "__main__":
    main()
