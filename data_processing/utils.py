import os
import json
import hashlib
import re
from typing import List, Dict, Optional
from PIL import Image

CHOICE_MARKER_RE = re.compile(r'(?:^|\s)([A-H])\.\s*')

def parse_choice_pairs(choices_text):

    if isinstance(choices_text, list):
        return [(chr(65 + idx), str(text).strip()) for idx, text in enumerate(choices_text)]
    if not choices_text:
        return []
    text = str(choices_text)
    matches = list(CHOICE_MARKER_RE.finditer(text))
    parsed = []
    for idx, match in enumerate(matches):
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        parsed.append((match.group(1), text[match.end():end].strip()))
    return parsed

def load_all_vqa_pairs(
    questions_file: str,
    dataset_name: str = "wsi_vqa",
    image_dir: Optional[str] = None,
) -> List[Dict]:

    with open(questions_file, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    if isinstance(raw_data, list):
        items = raw_data
    elif isinstance(raw_data, dict):
        for key in ['data', 'questions', 'items', 'test', 'samples']:
            if key in raw_data:
                items = raw_data[key]
                break
        else:
            items = list(raw_data.values())[0] if raw_data else []
    else:
        items = []

    pairs = []
    for item in items:

        long_id = (item.get('Id') or item.get('long_id') or item.get('slide_id')
                   or item.get('wsi_id') or item.get('image_id', ''))
        question = item.get('Question') or item.get('question', '')

        choices_raw = item.get('Choice') or item.get('choices') or item.get('options') or item.get('choice')
        if isinstance(choices_raw, list) and len(choices_raw) > 0:
            choices = ' '.join([f"{chr(65+i)}. {c}" for i, c in enumerate(choices_raw)])
            is_mcq = True
        elif isinstance(choices_raw, str) and choices_raw.strip():
            choices = choices_raw
            is_mcq = True
        else:
            choices = ''
            is_mcq = False

        answer = item.get('Answer') or item.get('answer') or item.get('ground_truth') or item.get('label', '')

        pairs.append({
            'long_id': str(long_id),
            'question': question,
            'choices': choices,
            'answer': str(answer),
            'is_mcq': is_mcq,
        })

    print(f"Loaded {len(pairs)} VQA pairs from {questions_file}")
    mcq_count = sum(1 for p in pairs if p['is_mcq'])
    print(f"  MCQ: {mcq_count}, Open-ended: {len(pairs) - mcq_count}")
    return pairs

def make_unique_id(long_id: str, question: str) -> str:

    raw = f"{long_id}_{question}"
    return hashlib.md5(raw.encode()).hexdigest()[:16]

def load_image(image_path: str) -> Image.Image:

    return Image.open(image_path).convert("RGB")

def get_patch_fullpath(patch_root: str, wsi_id: str, patch_name: str) -> str:

    stem = os.path.splitext(patch_name)[0]

    candidates = [
        os.path.join(patch_root, wsi_id, patch_name),
        os.path.join(patch_root, wsi_id, f"{stem}.png"),
        os.path.join(patch_root, wsi_id, f"{stem}.jpg"),
        os.path.join(patch_root, wsi_id, "patches", f"{stem}.png"),
        os.path.join(patch_root, wsi_id, "patches", f"{stem}.jpg"),
    ]

    for c in candidates:
        if os.path.exists(c):
            return c

    return os.path.join(patch_root, wsi_id, f"{stem}.png")

def extract_answer_letter(raw_text: str, valid_choices: str = "ABCDEFGH",
                          choices_text: str = "") -> str:

    text = raw_text.strip()

    json_matches = list(re.finditer(r'\{[^{}]*"answer"[^{}]*\}', text))
    if json_matches:
        last_json_str = json_matches[-1].group()
        try:
            import json
            parsed = json.loads(last_json_str)
            ans = str(parsed.get("answer", "")).strip().upper()
            if len(ans) == 1 and ans in valid_choices:
                return ans
        except Exception:
            pass

    tail = text[-1500:] if len(text) > 1500 else text
    commitment_patterns = [
        r'"answer"\s*:\s*"([A-H])"',
        r'[Ff]inal\s+[Aa]nswer[:\s]+([A-H])\b',
        r'[Aa]nswer\s+is\s+(?:\w+\s+){0,3}([A-H])\b',
        r'[Cc]orrect\s+[Aa]nswer[:\s]+(?:\w+\s+){0,2}([A-H])\b',
        r'[Tt]herefore[,:\s]+([A-H])\b',
        r'[Ss]elect(?:ed|ing)?\s+([A-H])\b',
        r'[Aa]nswer[:\s]+([A-H])\b',
        r'\*\*([A-H])\*\*',
    ]
    for pattern in commitment_patterns:
        matches = re.findall(pattern, tail)
        valid = [m for m in matches if m in valid_choices]
        if valid:
            return valid[-1]

    if choices_text and len(text) > 200:
        parsed_choices = parse_choice_pairs(choices_text)
        tail_lower = tail.lower()

        found = []
        for letter, choice_val in parsed_choices:
            if letter not in valid_choices:
                continue

            if choice_val.strip().lower() in tail_lower:
                found.append(letter)
        if len(found) == 1:
            return found[0]

    if len(text) < 200:
        head_match = re.match(r'^([A-H])\b', text)
        if head_match and head_match.group(1) in valid_choices:
            return head_match.group(1)
        dot_match = re.search(r'\b([A-H])\.\s', text)
        if dot_match and dot_match.group(1) in valid_choices:
            return dot_match.group(1)
        for char in text:
            if char in valid_choices:
                return char

    return "?"
