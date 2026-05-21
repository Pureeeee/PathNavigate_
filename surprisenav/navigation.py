import re

import numpy as np
import torch
from data_processing.feature_utils import extract_patch_from_svs

try:
    from transformers import CLIPModel as _CLIPModel, CLIPProcessor as _CLIPProcessor
    _PLIP_AVAILABLE = True
except ImportError:
    _PLIP_AVAILABLE = False

def load_plip(plip_ckpt, device='cuda:0'):

    if not _PLIP_AVAILABLE:
        raise ImportError("transformers CLIPModel not available")
    model = _CLIPModel.from_pretrained(plip_ckpt).to(device).eval()
    processor = _CLIPProcessor.from_pretrained(plip_ckpt)
    return model, processor

@torch.no_grad()
def query_guided_selection(
    question,
    candidate_coords,
    surprise_scores,
    svs_path,
    plip_model,
    plip_processor,
    alpha=0.5,
    top_k=5,
    magnification=5,
):

    if not candidate_coords:
        return []

    device = plip_model.device

    text_inputs = plip_processor(text=[question], return_tensors="pt", padding=True, truncation=True)
    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
    text_emb = plip_model.get_text_features(**text_inputs)
    if not isinstance(text_emb, torch.Tensor):
        text_emb = text_emb[0] if hasattr(text_emb, '__getitem__') else text_emb.text_embeds
    text_emb = torch.nn.functional.normalize(text_emb, dim=-1)

    relevance_scores = {}
    for (x, y) in candidate_coords:
        try:
            img = extract_patch_from_svs(svs_path, x, y, magnification=magnification)
            img_inputs = plip_processor(images=img, return_tensors="pt")
            img_inputs = {k: v.to(device) for k, v in img_inputs.items()}
            img_emb = plip_model.get_image_features(**img_inputs)
            if not isinstance(img_emb, torch.Tensor):
                img_emb = img_emb[0] if hasattr(img_emb, '__getitem__') else img_emb.image_embeds
            img_emb = torch.nn.functional.normalize(img_emb, dim=-1)
            sim = (text_emb @ img_emb.T).item()
            relevance_scores[(x, y)] = sim
        except Exception:
            relevance_scores[(x, y)] = 0.0

    s_vals = [surprise_scores.get((x, y), 0.0) for (x, y) in candidate_coords]
    s_min, s_max = min(s_vals), max(s_vals)
    s_range = s_max - s_min if s_max > s_min else 1.0

    r_vals = list(relevance_scores.values())
    r_min, r_max = min(r_vals), max(r_vals)
    r_range = r_max - r_min if r_max > r_min else 1.0

    scored = []
    for (x, y) in candidate_coords:
        s_norm = (surprise_scores.get((x, y), 0.0) - s_min) / s_range
        r_norm = (relevance_scores.get((x, y), 0.0) - r_min) / r_range
        combined = alpha * r_norm + (1 - alpha) * s_norm
        scored.append((x, y, combined, relevance_scores.get((x, y), 0.0)))

    scored.sort(key=lambda item: item[2], reverse=True)
    return [(x, y) for x, y, _, _ in scored[:top_k]]

class SurpriseCache:
    def __init__(self):
        self._cache = {}

    def has(self, wsi_id: str) -> bool:
        return wsi_id in self._cache

    def get(self, wsi_id: str):
        return self._cache.get(wsi_id)

    def put(self, wsi_id: str, data):
        self._cache[wsi_id] = data

    def clear(self):
        self._cache.clear()

def _build_meanpool_report(
    feature_cache_5x,
    total_5x_patches: int,
    jump_targets: list,
) -> str:

    if not feature_cache_5x:
        return ""
    feats = []
    for name in feature_cache_5x:
        v = feature_cache_5x[name]
        try:
            arr = v.cpu().numpy() if hasattr(v, 'cpu') else np.array(v)
        except Exception:
            continue
        feats.append(arr.flatten())
    if not feats:
        return ""
    feats = np.stack(feats, axis=0)
    norms = np.linalg.norm(feats, axis=1)
    if len(norms) == 0:
        return ""
    high_norm_thresh = float(np.median(norms))
    high_norm_frac = float(np.mean(norms > high_norm_thresh))
    n_targets = len(jump_targets)
    if high_norm_frac > 0.55:
        hetero = "highly heterogeneous"
    elif high_norm_frac > 0.45:
        hetero = "moderately heterogeneous"
    else:
        hetero = "mostly homogeneous"
    return (
        f"[Slide Summary] Scanned {total_5x_patches} regions at 5x, "
        f"tissue is {hetero} ({high_norm_frac*100:.0f}% high-norm). "
        f"Selected {n_targets} regions for detailed examination.\n"
    )

def _build_global_memory_report(
    yaad_stats,
    total_5x_patches: int,
    coord_surprise_5x: list,
    jump_targets: list,
) -> str:

    if yaad_stats is None:
        return ""

    count = yaad_stats.get("count", 0)
    threshold = yaad_stats.get("threshold", 0)
    mean = yaad_stats.get("mean", 0)
    std = yaad_stats.get("std", 0)
    max_s = yaad_stats.get("max", 0)
    high_ratio = yaad_stats.get("high_surprise_ratio", 0)

    n_targets = len(jump_targets)

    if high_ratio > 0.3:
        hetero = "highly heterogeneous"
    elif high_ratio > 0.1:
        hetero = "moderately heterogeneous"
    else:
        hetero = "mostly homogeneous"

    report = (
        f"[Navigation Summary] Scanned {total_5x_patches} regions at 5x, "
        f"tissue is {hetero} ({high_ratio*100:.0f}% high-surprise). "
        f"Selected {n_targets} surprising regions for detailed examination.\n"
    )

    return report

_MACRO_INTENT_KEYWORDS = {
    "multifocal", "multifocality", "multi-focal",
    "overall extent", "tumor extent", "whole-lesion",
    "gross border", "margin layout", "distribution",
}
_MICRO_INTENT_KEYWORDS = {
    "er ", "pr ", "her2", "erbb2", "receptor", "estrogen", "progesterone",
    "grade", "nuclear grade", "nottingham", "diagnosis", "histological type",
    "histologic type", "subtype", "in-situ", "in situ", "dcis",
    "carcinoma type", "tumor type", "size",
}

def _should_start_with_macro(question: str, q_type: str) -> bool:

    if q_type == "morphology":
        return False
    q_lower = question.lower()

    if any(kw in q_lower for kw in _MICRO_INTENT_KEYWORDS):
        return False

    if any(kw in q_lower for kw in _MACRO_INTENT_KEYWORDS):
        return True

    return False

_LOWMAG_FORBIDDEN_PATTERNS = [
    r'\b\d+\.?\d*\s*cm\b',
    r'\b\d+\.?\d*\s*mm\b',
    r'\bt[1-4]\s*n[0-3]\s*m[0-1x]',
    r'\b(ER|PR|HER2|Ki-?67)\s*(positive|negative|[0-3]\+)',
    r'\bmargin\s*(positive|negative|involved|clear)\b',
    r'\b\d+\s*%\s*(ER|PR|HER2|Ki-?67|staining|positivity)\b',
    r'\b(ER|PR|HER2|Ki-?67)\s*\d+\s*%\b',
]

def _sanitize_lowmag_evidence(text: str, magnification: int) -> str:

    if magnification > 10:
        return text
    import re
    violations = []
    for pattern in _LOWMAG_FORBIDDEN_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            violations.append(pattern)
    if not violations:
        return text

    sanitized = text
    for pattern in _LOWMAG_FORBIDDEN_PATTERNS:
        sanitized = re.sub(pattern, "[REMOVED: not assessable from low-mag view]",
                          sanitized, flags=re.IGNORECASE)
    sanitized += (
        "\n[LIMITATION: Some claims were removed because exact measurements, "
        "staging, and IHC results cannot be determined from a low-magnification patch.]"
    )
    return sanitized
