import os
import json
import h5py
import torch
import numpy as np
import argparse
from collections import defaultdict

def build_index(
    train_json: str,
    feature_dir: str,
    output_path: str,
):

    train_data = json.load(open(train_json))
    qa_by_wsi = defaultdict(list)
    for item in train_data:
        wsi_id = item["Id"]
        qa_by_wsi[wsi_id].append({
            "question": item["Question"],
            "choice": item.get("Choice", []),
            "answer": item["Answer"],
        })

    print(f"Training data: {len(train_data)} QA pairs, {len(qa_by_wsi)} WSIs")

    feat_files = sorted(os.listdir(feature_dir))
    prefix_to_h5 = {}
    for f in feat_files:
        if not f.endswith(".h5"):
            continue
        base = f.split(".")[0]
        parts = base.split("-")
        if len(parts) >= 3:
            prefix = "-".join(parts[:3])
            if prefix not in prefix_to_h5:
                prefix_to_h5[prefix] = os.path.join(feature_dir, f)

    print(f"Feature files: {len(prefix_to_h5)} unique WSI prefixes")

    embeddings = []
    wsi_ids = []
    qa_pairs = {}
    skipped = 0

    for wsi_id in sorted(qa_by_wsi.keys()):

        h5_path = prefix_to_h5.get(wsi_id)
        if h5_path is None:
            skipped += 1
            continue

        with h5py.File(h5_path, "r") as h:
            feats = h["features"][:]

        global_vec = np.mean(feats, axis=0)

        norm = np.linalg.norm(global_vec)
        if norm > 0:
            global_vec = global_vec / norm

        embeddings.append(global_vec)
        wsi_ids.append(wsi_id)
        qa_pairs[wsi_id] = qa_by_wsi[wsi_id]

    embeddings = np.stack(embeddings)
    print(f"Built index: {len(wsi_ids)} WSIs, {embeddings.shape}, skipped {skipped}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save({
        "embeddings": torch.from_numpy(embeddings).float(),
        "wsi_ids": wsi_ids,
        "qa_pairs": qa_pairs,
    }, output_path)
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_json", required=True)
    parser.add_argument("--feature_dir", required=True)
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()
    build_index(
        train_json=args.train_json,
        feature_dir=args.feature_dir,
        output_path=args.output_path,
    )
