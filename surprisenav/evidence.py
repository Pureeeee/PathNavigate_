from .json_utils import _extract_json

_ENUM_RELEVANT = {"yes", "no"}
_ENUM_CONFIDENCE = {"high", "medium", "low"}
_ENUM_BORDER = {"yes", "no", "partial"}
_ENUM_BIN = {"yes", "no"}
_ENUM_FOCI = {"yes", "no", "unclear"}

_PLACEHOLDER_VALUES = {
    "yes|no", "yes/no", "high|medium|low", "high/medium/low",
    "yes|no|partial", "yes|no|unclear", "yes/no/partial", "yes/no/unclear",
    "<= 18 words describing visible morphology",
    "<= 5 short items separated by ;",
    "<= 12 words or 'none'",
}

def _repair_structured_json(raw_text: str):

    import re
    if not raw_text:
        return None, "empty"

    cleaned = re.sub(r'```(?:json)?\s*', '', raw_text)
    cleaned = re.sub(r'```\s*', '', cleaned)
    cleaned = cleaned.strip()

    parsed = _extract_json(cleaned)
    if parsed:
        return parsed, None

    start = cleaned.find('{')
    if start == -1:
        return None, "no_json_braces"

    end = cleaned.rfind('}')
    if end == -1 or end <= start:

        truncated = cleaned[start:]

        soft_closed = truncated.rstrip(',').rstrip()

        nq = soft_closed.count('"') - soft_closed.count('\\"')
        if nq % 2 == 1:
            soft_closed += '"'
        if not soft_closed.endswith('}'):
            soft_closed += '}'
        parsed = _extract_json(soft_closed)
        if parsed:
            return parsed, "repaired_soft_close"
        return None, "no_json_braces"

    candidate = cleaned[start:end+1]

    candidate = re.sub(r'","?\s+"', '","', candidate)

    candidate = re.sub(r'"\s+"', '","', candidate)

    candidate = re.sub(r':\s*none\b', ': "none"', candidate, flags=re.IGNORECASE)

    repaired = re.sub(r'(["\d\]\}])\s*\n\s*"', r'\1,\n  "', candidate)

    def _quote_bare(m):
        key = m.group(1)
        val = m.group(2).strip().rstrip(',').rstrip()
        if (val.startswith('"') or val in ('true','false','null')
            or val.startswith('[') or val.startswith('{')
            or re.match(r'^-?\d', val)):
            return m.group(0)
        val_esc = val.replace('"', '\\"')
        return f'"{key}": "{val_esc}"'
    repaired = re.sub(r'"(\w+)":\s*([^",\[\{\n][^,\n}]*?)(?=[,\n}])', _quote_bare, repaired)

    parsed = _extract_json(repaired)
    if parsed:
        return parsed, "repaired"

    return None, "json_decode_failed"

def _repair_was_applied(parse_err) -> bool:
    return parse_err in ("repaired", "repaired_soft_close")

def _parse_structured_patch_output(raw_text: str):

    parsed, err = _repair_structured_json(raw_text)
    if parsed is None:
        return None, err
    if not isinstance(parsed, dict):
        return None, "not_a_dict"
    if "summary" not in parsed:
        return None, "missing_summary"
    return parsed, None

def _normalize_structured_patch(parsed: dict, magnification: int):

    import re

    normalize_meta = {
        "soft_defaults_used": [],
        "summary_imputed": False,
        "relevant_imputed": False,
        "confidence_imputed": False,
        "lowmag_defaults_used": [],
        "hard_valid": True,
    }

    def _is_placeholder(v):
        if not isinstance(v, str):
            return False
        return v.strip().lower() in _PLACEHOLDER_VALUES

    _KEY_ALIASES = {
        "conficiency": "confidence", "conficence": "confidence", "confidance": "confidence",
        "summery": "summary", "feature": "visible_features",
        "features": "visible_features", "border": "border_visibility",
        "extent": "extent_assessable", "foci": "separate_foci_visible",
    }

    _lowered = {}
    for k, v in list(parsed.items()):
        if isinstance(k, str):
            _lowered[k.lower()] = v
    parsed = _lowered

    for alias, canonical in _KEY_ALIASES.items():
        if alias in parsed and canonical not in parsed:
            parsed[canonical] = parsed[alias]

    rel_raw = parsed.get("relevant", "")
    if isinstance(rel_raw, bool):
        relevant = "yes" if rel_raw else "no"
    else:
        relevant = str(rel_raw).lower().strip()
        if relevant in ("true", "1"): relevant = "yes"
        elif relevant in ("false", "0"): relevant = "no"
    if _is_placeholder(rel_raw) or relevant not in _ENUM_RELEVANT:
        relevant = "yes"
        normalize_meta["relevant_imputed"] = True
        normalize_meta["soft_defaults_used"].append("relevant")

    summary = str(parsed.get("summary", "")).strip()
    if not summary or _is_placeholder(summary):

        vf_for_synth = parsed.get("visible_features", [])
        synth_items = []
        if isinstance(vf_for_synth, list):
            synth_items = [str(x).strip() for x in vf_for_synth if str(x).strip()][:2]
        elif isinstance(vf_for_synth, str) and vf_for_synth.strip():
            parts = re.split(r'[;,]', vf_for_synth)
            synth_items = [p.strip() for p in parts if p.strip()][:2]
        if synth_items:
            summary = "; ".join(synth_items)
            normalize_meta["summary_imputed"] = True
            normalize_meta["soft_defaults_used"].append("summary")
        else:
            return None, "empty_summary", normalize_meta

    if summary.count('.') >= 1:
        first_sent = summary.split('.', 1)[0].strip()
        if first_sent:
            summary = first_sent
    summary_words = summary.split()
    if len(summary_words) > 20:
        summary = " ".join(summary_words[:20])

    conf_raw = parsed.get("confidence", "")
    confidence = str(conf_raw).lower().strip()
    if _is_placeholder(conf_raw) or confidence not in _ENUM_CONFIDENCE:
        confidence = "medium"
        normalize_meta["confidence_imputed"] = True
        normalize_meta["soft_defaults_used"].append("confidence")

    vf_raw = parsed.get("visible_features", [])
    if isinstance(vf_raw, list):
        features = [str(x).strip() for x in vf_raw if str(x).strip()]
    elif isinstance(vf_raw, str):
        if _is_placeholder(vf_raw):
            features = []
        else:
            parts = re.split(r'[;,]|\d+\.\s*', vf_raw)
            features = [p.strip() for p in parts if p.strip()]
    else:
        features = []

    features = [
        " ".join(f.split()[:8]) if len(f.split()) > 8 else f
        for f in features
    ]
    features = features[:5]

    lim_raw = parsed.get("limitation", "none")
    if isinstance(lim_raw, bool) or lim_raw is None:
        limitation = "none"
    else:
        limitation = str(lim_raw).strip()
        if _is_placeholder(limitation) or not limitation or limitation.lower() in ("false", "null"):
            limitation = "none"
    if limitation.lower() != "none":
        lim_words = limitation.split()
        if len(lim_words) > 18:
            limitation = " ".join(lim_words[:18])

    result = {
        "relevant": relevant,
        "summary": summary,
        "visible_features": features,
        "limitation": limitation,
        "confidence": confidence,
    }

    if magnification <= 10:
        def _norm_enum(field_name, raw, allowed, default):
            v = str(raw).lower().strip()
            if v in allowed:
                return v

            for tok in allowed:
                if tok in v:
                    return tok
            normalize_meta["lowmag_defaults_used"].append(field_name)
            normalize_meta["soft_defaults_used"].append(field_name)
            return default
        result["border_visibility"] = _norm_enum(
            "border_visibility", parsed.get("border_visibility", ""), _ENUM_BORDER, "partial")
        result["extent_assessable"] = _norm_enum(
            "extent_assessable", parsed.get("extent_assessable", ""), _ENUM_BIN, "no")
        result["separate_foci_visible"] = _norm_enum(
            "separate_foci_visible", parsed.get("separate_foci_visible", ""), _ENUM_FOCI, "unclear")

    normalize_meta["hard_valid"] = not any([
        normalize_meta["summary_imputed"],
        normalize_meta["relevant_imputed"],
        normalize_meta["confidence_imputed"],
        bool(normalize_meta["lowmag_defaults_used"]),
    ])

    return result, None, normalize_meta

def _structured_patch_to_evidence_line(obj: dict, coords, magnification, score,
                                        region_num: int) -> str:

    header = f"- [Region {region_num}, Surprise: {score:.3f}, Coord: ({coords[0]},{coords[1]}), {magnification}x]"
    summary = obj.get("summary", "No description")
    features = obj.get("visible_features", [])
    limitation = obj.get("limitation", "")
    conf = obj.get("confidence", "medium")

    line = f"{header} {summary}"
    if features:
        feat_str = "; ".join(features) if isinstance(features, list) else str(features)
        line += f" Features: {feat_str}"

    if magnification <= 10:
        bv = obj.get("border_visibility")
        ea = obj.get("extent_assessable")
        sf = obj.get("separate_foci_visible")
        macro_parts = []
        if bv: macro_parts.append(f"border_visibility={bv}")
        if ea: macro_parts.append(f"extent_assessable={ea}")
        if sf: macro_parts.append(f"separate_foci={sf}")
        if macro_parts:
            line += f" [Macro: {', '.join(macro_parts)}]"

    if limitation and limitation.lower() != "none":
        line += f" [Limitation: {limitation}]"
    line += f" [Confidence: {conf}]"
    return line
