import os

import numpy as np
import torch

_TEXTBOOK_CACHE = {}
_KB_STOPWORDS = {"what", "does", "this", "that", "with", "from", "have", "been",
                 "the", "and", "for", "are", "was", "were", "can", "help",
                 "patient", "slide", "breast", "according", "predict", "please",
                 "result", "pathologist", "case", "tissue", "image"}

def _search_detailed_kb(kb_data, q_keywords, results, max_depth=3, _depth=0):

    if _depth >= max_depth or len(results) >= 10:
        return
    if isinstance(kb_data, dict):
        for key, value in kb_data.items():
            k_words = set(key.lower().replace("_", " ").split())
            if q_keywords & k_words:
                if isinstance(value, str) and len(value) > 20:
                    results.append(value[:250])
                elif isinstance(value, list):
                    for item in value[:3]:
                        if isinstance(item, str) and len(item) > 10:
                            results.append(item[:250])
                elif isinstance(value, dict):
                    _search_detailed_kb(value, q_keywords, results, max_depth, _depth + 1)
            elif isinstance(value, (dict, list)):
                _search_detailed_kb(value, q_keywords, results, max_depth, _depth + 1)
    elif isinstance(kb_data, list):
        for item in kb_data[:5]:
            if isinstance(item, dict):
                _search_detailed_kb(item, q_keywords, results, max_depth, _depth + 1)

def _retrieve_textbook_knowledge(question: str, q_type: str = "other") -> str:

    import json as _json

    q_lower = question.lower()
    q_keywords = set(q_lower.split()) - _KB_STOPWORDS
    q_keywords = {w for w in q_keywords if len(w) > 3}

    concise_rules = []
    kb_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "pathology_knowledge_base.json")
    if "concise_kb" not in _TEXTBOOK_CACHE:
        if os.path.exists(kb_path):
            try:
                with open(kb_path, "r") as f:
                    _TEXTBOOK_CACHE["concise_kb"] = _json.load(f)
            except Exception:
                _TEXTBOOK_CACHE["concise_kb"] = {}
        else:
            _TEXTBOOK_CACHE["concise_kb"] = {}
    for category, data in _TEXTBOOK_CACHE["concise_kb"].items():
        if any(kw in q_lower for kw in data.get("keywords", [])):
            concise_rules.extend(data.get("rules", []))

    detailed_rules = []
    detailed_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "breast_pathology_knowledge_base.json")
    if "detailed_kb" not in _TEXTBOOK_CACHE:
        if os.path.exists(detailed_path):
            try:
                with open(detailed_path, "r") as f:
                    _TEXTBOOK_CACHE["detailed_kb"] = _json.load(f)
            except Exception:
                _TEXTBOOK_CACHE["detailed_kb"] = {}
        else:
            _TEXTBOOK_CACHE["detailed_kb"] = {}
    if _TEXTBOOK_CACHE["detailed_kb"] and q_keywords:
        _search_detailed_kb(_TEXTBOOK_CACHE["detailed_kb"], q_keywords, detailed_rules)

    selected = concise_rules[:3] + detailed_rules[:3]
    if not selected:
        return ""
    rules_text = "\n".join(f"  - {r}" for r in selected)
    return (
        "[DIAGNOSTIC REFERENCE] Relevant pathology diagnostic criteria:\n"
        + rules_text
        + "\nApply these criteria to the morphological evidence when making your diagnosis.\n"
    )

def _load_rag_index(path: str):

    if not path or not os.path.exists(path):
        return None
    data = torch.load(path, map_location="cpu")
    print(f"  RAG index loaded: {len(data['wsi_ids'])} WSIs")
    return data

def _retrieve_rag_context(
    feature_cache_5x: dict,
    rag_index,
    top_k: int = 3,
    question: str = "",
) -> str:

    if rag_index is None:
        return ""

    feats = []
    for name in sorted(feature_cache_5x.keys()):
        v = feature_cache_5x[name]
        if isinstance(v, torch.Tensor):
            feats.append(v.cpu().numpy())
        else:
            feats.append(np.asarray(v))
    if not feats:
        return ""

    feats = np.stack(feats)
    query_vec = np.mean(feats, axis=0)
    norm = np.linalg.norm(query_vec)
    if norm > 0:
        query_vec = query_vec / norm
    query_vec = torch.from_numpy(query_vec).float().unsqueeze(0)

    index_embeddings = rag_index["embeddings"]

    if query_vec.shape[1] != index_embeddings.shape[1]:
        return ""
    sims = torch.mm(query_vec, index_embeddings.t()).squeeze(0)
    topk_vals, topk_idxs = torch.topk(sims, min(top_k * 2, len(sims)))

    q_lower = question.lower()
    _RAG_STOPWORDS = {"the", "a", "an", "is", "of", "in", "what", "does",
                      "was", "are", "do", "did", "can", "you", "to", "for",
                      "this", "that", "it", "be", "has", "have", "been",
                      "which", "about", "with", "there", "case", "show",
                      "image", "slide", "tissue", "patient", "present",
                      "from", "following", "seen", "based", "would"}
    q_keywords = set(q_lower.split()) - _RAG_STOPWORDS

    lines = []
    relevant_count = 0
    max_relevant = 5

    for val, idx in zip(topk_vals.tolist(), topk_idxs.tolist()):
        if relevant_count >= max_relevant:
            break
        wsi_id = rag_index["wsi_ids"][idx]
        qa_pairs = rag_index["qa_pairs"][wsi_id]

        for qa in qa_pairs:
            if relevant_count >= max_relevant:
                break
            train_q = qa["question"].lower()

            train_words = set(train_q.split()) - _RAG_STOPWORDS
            overlap = q_keywords & train_words
            if len(overlap) >= 2 or any(kw in train_q for kw in q_keywords if len(kw) > 4):
                lines.append(f"  Reference (sim={val:.3f}): Q: {qa['question']} -> A: {qa['answer']}")
                relevant_count += 1

    if not lines:
        return ""

    header = "[REFERENCE] Answers from similar tissue cases in training database:\n"
    footer = "\nUse these references as supporting evidence, but base your answer primarily on the morphological observations.\n"
    return header + "\n".join(lines) + footer
