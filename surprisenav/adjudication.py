import re

import torch
from data_processing.utils import extract_answer_letter

from .evidence import _repair_structured_json, _repair_was_applied
from .json_utils import _extract_json

def adjudicator_answer(
    question: str,
    choices: str,
    evidence_texts: list,
    tokenizer,
    model,
    is_mcq: bool = True,
    q_type: str = "other",
    max_new_tokens: int = 3072,
    reflection_mode: bool = False,
    memory_context: str = "",
) -> dict:

    combined_evidence = "\n".join(evidence_texts)

    base_guidance = (
        "You are an expert pathology adjudicator. Multiple regions of a whole slide image "
        "have been examined by a pathology analyst, producing objective morphological "
        "descriptions for each region.\n"
        "Keep your reasoning VERY concise (under 300 words). Do not repeat the evidence. "
        "After reasoning, you MUST output your final answer as JSON.\n"
    )

    if q_type == "morphology":
        domain_guidance = base_guidance + (
            "You are performing DIFFERENTIAL DIAGNOSIS based on microscopic morphology.\n"
            "Your task:\n"
            "1) For EACH region, extract specific diagnostic criteria: growth pattern "
            "(tubular/solid/single-file/papillary), nuclear features (pleomorphism, grade), "
            "stromal reaction, invasion pattern, and special features\n"
            "2) Use these criteria to narrow the differential diagnosis\n"
            "3) Key discriminators: single-file invasion -> lobular; tubule formation -> ductal; "
            "medullary pattern -> syncytial growth + lymphocytes; mucinous -> extracellular mucin pools\n"
            "4) For grading: assess tubule formation, nuclear pleomorphism, and mitotic count\n"
            "5) Favor the PREDOMINANT pattern if regions show mixed findings\n\n"
            "If a smaller number of regions contain more diagnostically specific findings, "
            "weigh them by diagnostic specificity rather than by simple region count.\n"
            "IMPORTANT: Always commit to a specific diagnosis. Choose the option best "
            "supported by the morphological evidence described.\n"
        )
    else:
        domain_guidance = base_guidance + (
            "Your task:\n"
            "1) Synthesize ALL the morphological evidence across regions\n"
            "2) Identify the most consistent pattern across regions\n"
            "3) Resolve any contradictions by favoring the predominant finding\n"
            "4) Select the best answer based on your synthesis\n\n"
            "IMPORTANT: Always commit to a specific answer based on available evidence. "
            "When uncertain between options, choose the one best supported by the morphological findings. "
            "Do NOT answer 'not mentioned' or 'cannot be determined' unless that is truly "
            "the only reasonable choice.\n"
        )

    if is_mcq and choices:
        format_instruction = (
            "REASONING STRATEGY for multiple choice:\n"
            "1) For EACH option, briefly assess: does the morphological evidence SUPPORT, REFUTE, or provide NO INFO for this option?\n"
            "2) Eliminate options that are directly contradicted by the evidence\n"
            "3) Among remaining options, select the one with the STRONGEST morphological support\n"
            "4) If evidence is ambiguous between options, consider which option is more consistent with the overall tissue pattern\n"
            "5) For quantitative questions (size, grade, score) where evidence gives partial info, use the closest matching option\n\n"
            "Your answer MUST be ONLY the letter (A, B, C, or D). Do NOT include the option text.\n"
            "Respond in JSON: {\"answer\": \"A\", \"explanation\": \"brief reasoning with per-option assessment\"}"
        )
    else:

        q_lower_adj = question.lower()
        if any(kw in q_lower_adj for kw in ["her2", "erbb2", "her-2"]):
            format_instruction = (
                "For HER2 related questions, your final answer MUST be strictly formatted as "
                "one of the following exact strings: '0', '1+', '2+', or '3+'. "
                "Translate textual descriptions accordingly: 'negative' -> '0', "
                "'equivocal'/'borderline' -> '2+', 'positive' -> '3+'.\n"
                "Respond in JSON: {\"answer\": \"your answer\", \"explanation\": \"brief reasoning\"}"
            )
        else:
            format_instruction = (
                "Respond in JSON: {\"answer\": \"your answer\", \"explanation\": \"brief reasoning\"}"
            )

    if reflection_mode:
        format_instruction += (
            "\nAlso include a \"sufficient\" field: \"yes\" if you have enough evidence to be "
            "confident, \"no\" if additional tissue regions would help. "
            "If \"no\", include \"missing_info\": a short phrase describing what evidence would help."
        )

    system_prompt = domain_guidance + "\n" + format_instruction

    user_prompt = (
        f"Question: {question}\n"
    )
    if is_mcq and choices:
        user_prompt += f"Choices: {choices}\n\n"

    if memory_context:
        user_prompt += (
            f"=== Navigation & Reference Context ===\n"
            f"{memory_context}\n"
            f"=== End Context ===\n\n"
        )

    user_prompt += (
        f"Morphological evidence from {len(evidence_texts)} examined regions:\n{combined_evidence}\n\n"
        f"Synthesize all evidence above and respond in JSON."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generate_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": False}
    _has_thinking = False
    with torch.no_grad():
        outputs = model.generate(**inputs, **generate_kwargs)

    output_ids = outputs[0][len(inputs.input_ids[0]):].tolist()
    raw_response = tokenizer.decode(output_ids, skip_special_tokens=False).strip()

    raw_response = raw_response.replace('<|im_end|>', '').replace('<|endoftext|>', '').strip()

    del inputs, outputs
    torch.cuda.empty_cache()

    thinking_text = ""
    if '</think>' in raw_response:

        last_think_end = raw_response.rfind('</think>')
        thinking_text = raw_response[:last_think_end].strip()

        thinking_text = re.sub(r'^<think>\s*', '', thinking_text)
        raw_response = raw_response[last_think_end + len('</think>'):].strip()
    elif '<think>' in raw_response:

        raw_response = re.sub(r'<think>.*', '', raw_response, flags=re.DOTALL).strip()
    else:

        pass

    parsed = _extract_json(raw_response)
    if not parsed and thinking_text:

        parsed = _extract_json(thinking_text)
    if parsed:
        answer_text = str(parsed.get("answer", "")).strip()
        explanation = str(parsed.get("explanation", "")).strip()
    else:

        search_text = raw_response[-1000:] if len(raw_response) > 1000 else raw_response
        answer_text = search_text
        explanation = ""

    if is_mcq and choices:

        if not parsed:
            answer_letter = extract_answer_letter(raw_response, choices_text=choices)
        else:
            answer_letter = extract_answer_letter(answer_text, choices_text=choices)
    else:
        answer_letter = answer_text if answer_text else "?"

    return {
        "answer": answer_letter,
        "raw_response": raw_response,
        "explanation": explanation or raw_response[:500],
    }

def adjudicator_answer_vllm(
    question: str,
    choices: str,
    evidence_texts: list,
    vllm_url: str,
    vllm_model_name: str = "",
    is_mcq: bool = True,
    q_type: str = "other",
    max_new_tokens: int = 512,
    memory_context: str = "",
    is_final_round: bool = True,
) -> dict:

    import requests as _requests

    combined_evidence = "\n".join(evidence_texts)

    base_guidance = (
        "You are an expert pathology adjudicator. Multiple regions of a whole slide image "
        "have been examined by a pathology analyst, producing objective morphological "
        "descriptions for each region.\n"
    )

    if q_type == "morphology":
        domain_guidance = base_guidance + (
            "You are performing DIFFERENTIAL DIAGNOSIS based on microscopic morphology.\n"
            "Key discriminators: single-file invasion -> lobular; tubule formation -> ductal; "
            "medullary pattern -> syncytial growth + lymphocytes; mucinous -> extracellular mucin pools.\n"
            "For grading: assess tubule formation, nuclear pleomorphism, and mitotic count.\n"
            "Favor the PREDOMINANT pattern if regions show mixed findings.\n"
            "If a smaller number of regions contain more diagnostically specific findings, "
            "weigh them by diagnostic specificity rather than by simple region count.\n"
            "IMPORTANT: Always commit to a specific diagnosis.\n"
        )
    else:
        domain_guidance = base_guidance + (
            "Synthesize ALL the morphological evidence across regions.\n"
            "Identify the most consistent pattern. Resolve contradictions by favoring the predominant finding.\n"
            "IMPORTANT: Always commit to a specific answer. Do NOT answer 'not mentioned' or "
            "'cannot be determined' unless truly the only reasonable choice.\n"
        )

    guardrail = (
        "A local patch cannot establish exact tumor size, TNM stage, lymph node status, "
        "surgical margin status, or clinical history. "
        "Do not trust patch-level claims about receptor staining or HER2 score unless "
        "the image explicitly shows an immunostain. "
        "Prefer the best morphology-supported answer rather than inventing a gross measurement.\n"
    )

    if is_final_round:
        if is_mcq and choices:

            format_instruction = (
                "REASONING STRATEGY for multiple choice:\n"
                "1) For EACH option, briefly assess: does the evidence SUPPORT, REFUTE, or provide NO INFO?\n"
                "2) Eliminate options directly contradicted by the evidence\n"
                "3) Among remaining, select the one with STRONGEST morphological support\n"
                "4) For quantitative questions (size, grade, score), use the closest matching option\n\n"
                "Your answer MUST be ONLY the letter (A, B, C, or D).\n"
                "Respond in JSON: {\"answer\": \"A\", \"explanation\": \"brief per-option assessment\"}"
            )
        else:
            q_lower_adj = question.lower()
            if any(kw in q_lower_adj for kw in ["her2", "erbb2", "her-2"]):
                format_instruction = (
                    "For HER2: answer MUST be one of: '0', '1+', '2+', or '3+'. "
                    "Translate: 'negative' -> '0', 'equivocal' -> '2+', 'positive' -> '3+'.\n"
                    "Respond in JSON: {\"answer\": \"your answer\", \"explanation\": \"brief reasoning\"}"
                )
            else:
                format_instruction = (
                    "Respond in JSON: {\"answer\": \"your answer\", \"explanation\": \"brief reasoning\"}"
                )
        system_prompt = domain_guidance + guardrail + format_instruction
    else:

        system_prompt = domain_guidance + guardrail
        answer_field = '"A"' if (is_mcq and choices) else '"your best guess so far"'
        system_prompt += (
            "You are also a navigation controller. Assess the current evidence and decide what to do next.\n"
            "Return exactly one JSON object with these fields:\n"
            f"- \"answer\": {answer_field} (your current best answer even if uncertain)\n"
            "- \"explanation\": \"one sentence\"\n"
            "- \"sufficient\": \"yes\" or \"no\" (is evidence enough to be confident?)\n"
            "- \"missing_info\": \"what specific evidence is still needed\" (short phrase, e.g. "
            "\"tumor border context\", \"nuclear detail for grading\", \"invasion pattern\")\n"
            "- \"next_action\": one of [\"need_macro_context\", \"need_micro_detail\", "
            "\"need_more_regions\", \"done\"]\n"
            "- \"next_magnification\": 5, 10, or 20\n"
            "- \"next_query\": \"short search phrase for finding relevant regions\""
        )

    user_prompt = f"Question: {question}\n"
    if is_mcq and choices:
        user_prompt += f"Choices: {choices}\n\n"
    if memory_context:
        user_prompt += f"=== Reference Context ===\n{memory_context}\n=== End Context ===\n\n"
    user_prompt += (
        f"Morphological evidence from {len(evidence_texts)} examined regions:\n"
        f"{combined_evidence}\n\nRespond in JSON."
    )

    def _resolve_vllm_model_name(session, base_url: str, requested_name: str) -> str:
        requested_name = (requested_name or "").strip()
        try:
            models_resp = session.get(f"{base_url}/models", timeout=30)
            models_resp.raise_for_status()
            models_payload = models_resp.json()
            model_entries = models_payload.get("data", []) if isinstance(models_payload, dict) else []
            if not model_entries:
                return requested_name

            if not requested_name:
                return str(model_entries[0].get("id", "")).strip() or requested_name

            requested_tail = requested_name.rstrip("/").split("/")[-1]
            for entry in model_entries:
                model_id = str(entry.get("id", "")).strip()
                model_root = str(entry.get("root", "")).strip()
                model_root_tail = model_root.rstrip("/").split("/")[-1] if model_root else ""
                if requested_name in {model_id, model_root}:
                    return model_id or requested_name
                if requested_tail and requested_tail in {model_id, model_root_tail}:
                    return model_id or requested_tail
        except Exception:
            return requested_name
        return requested_name

    try:
        _session = _requests.Session()
        _session.trust_env = False
        request_model_name = _resolve_vllm_model_name(_session, vllm_url, vllm_model_name)
        api_resp = _session.post(
            f"{vllm_url}/chat/completions",
            json={
                "model": request_model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "max_tokens": max_new_tokens,

                "temperature": 0.0,
                "top_p": 1.0,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            timeout=300,
        )
        api_resp.raise_for_status()
        data = api_resp.json()
        if "choices" not in data:
            raise RuntimeError(f"unexpected response payload: {str(data)[:400]}")
        raw_response = data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        raise RuntimeError(
            f"vLLM adjudicator request failed at {vllm_url}: {e}"
        ) from e

    if '</think>' in raw_response:
        raw_response = raw_response[raw_response.rfind('</think>') + 8:].strip()

    parsed = _extract_json(raw_response)
    if parsed:
        answer_text = str(parsed.get("answer", "")).strip()
        explanation = str(parsed.get("explanation", "")).strip()
    else:
        answer_text = raw_response[-200:] if len(raw_response) > 500 else raw_response
        explanation = ""

    if is_mcq and choices:
        if not parsed:
            answer_letter = extract_answer_letter(raw_response, choices_text=choices)
        else:
            answer_letter = extract_answer_letter(answer_text, choices_text=choices)
    else:
        answer_letter = answer_text if answer_text else "?"

    result = {
        "answer": answer_letter,
        "raw_response": raw_response,
        "explanation": explanation or raw_response[:500],
    }

    if parsed and not is_final_round:
        for key in ("sufficient", "missing_info", "next_action", "next_magnification", "next_query"):
            if key in parsed:
                result[key] = parsed[key]

    return result

def adjudicator_vllm_with_debiasing(
    question: str,
    choices: str,
    evidence_texts: list,
    vllm_url: str,
    vllm_model_name: str = "",
    is_mcq: bool = True,
    q_type: str = "other",
    memory_context: str = "",
) -> dict:

    if not is_mcq or not choices:
        return adjudicator_answer_vllm(
            question=question, choices=choices, evidence_texts=evidence_texts,
            vllm_url=vllm_url, vllm_model_name=vllm_model_name,
            is_mcq=is_mcq, q_type=q_type, memory_context=memory_context,
            is_final_round=True,
        )

    parsed = _parse_choices(choices)
    if len(parsed) < 2:
        return adjudicator_answer_vllm(
            question=question, choices=choices, evidence_texts=evidence_texts,
            vllm_url=vllm_url, vllm_model_name=vllm_model_name,
            is_mcq=is_mcq, q_type=q_type, memory_context=memory_context,
            is_final_round=True,
        )

    result_1 = adjudicator_answer_vllm(
        question=question, choices=choices, evidence_texts=evidence_texts,
        vllm_url=vllm_url, vllm_model_name=vllm_model_name,
        is_mcq=True, q_type=q_type, memory_context=memory_context,
        is_final_round=True,
    )
    answer_1 = result_1["answer"].strip().upper()

    n = len(parsed)
    permutation = [(i + 1) % n for i in range(n)]
    shuffled_choices, new_to_orig = _build_shuffled_choices(parsed, permutation)

    result_2 = adjudicator_answer_vllm(
        question=question, choices=shuffled_choices, evidence_texts=evidence_texts,
        vllm_url=vllm_url, vllm_model_name=vllm_model_name,
        is_mcq=True, q_type=q_type, memory_context=memory_context,
        is_final_round=True,
    )
    answer_2_shuffled = result_2["answer"].strip().upper()
    answer_2_orig = new_to_orig.get(answer_2_shuffled, answer_2_shuffled)

    valid_letters = {pc[0] for pc in parsed}
    chosen, chosen_result, debias_note = _resolve_debias_vote(
        answer_1=answer_1,
        answer_2_orig=answer_2_orig,
        result_1=result_1,
        result_2=result_2,
        valid_letters=valid_letters,
    )

    print(f"    {debias_note}")

    return {
        "answer": chosen,
        "raw_response": chosen_result["raw_response"],
        "explanation": chosen_result["explanation"] + f"\n{debias_note}",
    }

def _verify_answer_vllm(
    question: str,
    choices: str,
    evidence_texts: list,
    candidate_answer: str,
    vllm_url: str,
    vllm_model_name: str,
    q_type: str = "other",
) -> dict:

    import requests as _requests

    combined_evidence = "\n".join(evidence_texts)
    n_regions = len(evidence_texts)

    system_prompt = (
        "You are a pathology answer auditor. Output ONLY one short JSON object. "
        "Do not write any text outside the JSON. Do not write any explanation paragraph. "
        "You only flag risks; you NEVER suggest alternative answers.\n\n"
        "Field constraints:\n"
        "- has_conflict: must be exactly true or false\n"
        "- single_source_risk: must be exactly true or false\n"
        "- keep_answer: must be exactly \"yes\" or \"no\"\n"
        "- reason: at most 20 words, must be a quoted JSON string\n\n"
        "Example output (use real values, not placeholders):\n"
        "{\"has_conflict\": false, \"single_source_risk\": false, "
        "\"keep_answer\": \"yes\", \"reason\": \"all regions consistently support the proposed answer\"}"
    )

    user_prompt = (
        f"Question: {question}\n"
        f"Choices: {choices}\n\n"
        f"Proposed answer: {candidate_answer}\n\n"
        f"Evidence from {n_regions} regions:\n{combined_evidence}\n\n"
        "Audit this answer. Respond with ONLY the JSON object."
    )

    raw = None
    try:
        _session = _requests.Session()
        _session.trust_env = False
        api_resp = _session.post(
            f"{vllm_url}/chat/completions",
            json={
                "model": vllm_model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "max_tokens": 128,
                "temperature": 0.0,
                "top_p": 1.0,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            timeout=60,
        )
        data = api_resp.json()
        raw = data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return {
            "called": True,
            "parse_failed": True,
            "raw_output": str(raw) if raw else "",
            "validation_error": f"api_error: {str(e)[:100]}",
            "parse_status": "api_error",
            "repair_applied": False,
            "answer_changed": False,
        }

    parsed, parse_err = _repair_structured_json(raw)
    if parsed is None or not isinstance(parsed, dict):
        return {
            "called": True,
            "parse_failed": True,
            "raw_output": raw,
            "validation_error": f"parse_error:{parse_err}",
            "parse_status": parse_err or "parse_failed",
            "repair_applied": _repair_was_applied(parse_err),
            "answer_changed": False,
        }

    keep_answer = str(parsed.get("keep_answer", "")).lower().strip()
    if keep_answer not in ("yes", "no"):
        return {
            "called": True,
            "parse_failed": True,
            "raw_output": raw,
            "validation_error": f"bad_keep_answer:{keep_answer!r}",
            "parse_status": parse_err or "validation_failed",
            "repair_applied": _repair_was_applied(parse_err),
            "answer_changed": False,
        }

    return {
        "called": True,
        "parse_failed": False,
        "raw_output": raw,
        "has_conflict": bool(parsed.get("has_conflict", False)),
        "single_source_risk": bool(parsed.get("single_source_risk", False)),
        "keep_answer": keep_answer,
        "reason": str(parsed.get("reason", ""))[:300],
        "parse_status": parse_err or "direct_parse",
        "repair_applied": _repair_was_applied(parse_err),
        "answer_changed": False,
    }

CHOICE_MARKER_RE = re.compile(r'(?:^|\s)([A-H])\.\s*')

def _parse_choices(choices: str) -> list:

    if isinstance(choices, list):
        return [(chr(65 + idx), str(text).strip()) for idx, text in enumerate(choices)]
    if not choices:
        return []
    matches = list(CHOICE_MARKER_RE.finditer(str(choices)))
    parsed = []
    for idx, match in enumerate(matches):
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(str(choices))
        parsed.append((match.group(1), str(choices)[match.end():end].strip()))
    return parsed

def _build_shuffled_choices(parsed_choices: list, permutation: list) -> tuple:

    letters = [pc[0] for pc in parsed_choices]
    new_choices_parts = []
    new_to_orig = {}
    for new_idx, orig_idx in enumerate(permutation):
        new_letter = letters[new_idx]
        orig_letter = parsed_choices[orig_idx][0]
        orig_text = parsed_choices[orig_idx][1]
        new_choices_parts.append(f"{new_letter}. {orig_text}")
        new_to_orig[new_letter] = orig_letter
    return " ".join(new_choices_parts), new_to_orig

def _resolve_debias_vote(
    answer_1: str,
    answer_2_orig: str,
    result_1: dict,
    result_2: dict,
    valid_letters: set,
) -> tuple:

    a1_valid = answer_1 in valid_letters
    a2_valid = answer_2_orig in valid_letters

    if answer_1 == answer_2_orig and a1_valid:
        chosen = answer_1
        chosen_result = result_1
        debias_note = f"[Debias: AGREE] Both runs -> {chosen}"
    elif a1_valid and a2_valid:
        chosen = answer_2_orig
        chosen_result = result_2
        debias_note = (
            f"[Debias: DISAGREE] Run1->{answer_1}, Run2(shuffled)->{answer_2_orig}, using Run2(shared)"
        )
    elif a1_valid:
        chosen = answer_1
        chosen_result = result_1
        debias_note = f"[Debias: Run2 invalid({answer_2_orig})] using Run1->{answer_1}"
    elif a2_valid:
        chosen = answer_2_orig
        chosen_result = result_2
        debias_note = f"[Debias: Run1 invalid({answer_1})] using Run2->{answer_2_orig}"
    else:
        chosen = answer_1
        chosen_result = result_1
        debias_note = f"[Debias: BOTH INVALID] Run1={answer_1}, Run2={answer_2_orig}"

    return chosen, chosen_result, debias_note

def adjudicator_with_debiasing(
    question: str,
    choices: str,
    evidence_texts: list,
    tokenizer,
    model,
    is_mcq: bool = True,
    q_type: str = "other",
    max_new_tokens: int = 4096,
    reflection_mode: bool = False,
    memory_context: str = "",
) -> dict:

    if not is_mcq or not choices:
        return adjudicator_answer(
            question=question, choices=choices, evidence_texts=evidence_texts,
            tokenizer=tokenizer, model=model, is_mcq=is_mcq, q_type=q_type,
            max_new_tokens=max_new_tokens, reflection_mode=reflection_mode,
            memory_context=memory_context,
        )

    parsed = _parse_choices(choices)
    if len(parsed) < 2:

        return adjudicator_answer(
            question=question, choices=choices, evidence_texts=evidence_texts,
            tokenizer=tokenizer, model=model, is_mcq=is_mcq, q_type=q_type,
            max_new_tokens=max_new_tokens, reflection_mode=reflection_mode,
            memory_context=memory_context,
        )

    result_1 = adjudicator_answer(
        question=question, choices=choices, evidence_texts=evidence_texts,
        tokenizer=tokenizer, model=model, is_mcq=is_mcq, q_type=q_type,
        max_new_tokens=max_new_tokens, reflection_mode=reflection_mode,
        memory_context=memory_context,
    )
    answer_1 = result_1["answer"].strip().upper()

    n = len(parsed)
    permutation = [(i + 1) % n for i in range(n)]
    shuffled_choices, new_to_orig = _build_shuffled_choices(parsed, permutation)

    result_2 = adjudicator_answer(
        question=question, choices=shuffled_choices, evidence_texts=evidence_texts,
        tokenizer=tokenizer, model=model, is_mcq=is_mcq, q_type=q_type,
        max_new_tokens=max_new_tokens, reflection_mode=False,
        memory_context=memory_context,
    )
    answer_2_shuffled = result_2["answer"].strip().upper()

    answer_2_orig = new_to_orig.get(answer_2_shuffled, answer_2_shuffled)

    valid_letters = {pc[0] for pc in parsed}
    chosen, chosen_result, debias_note = _resolve_debias_vote(
        answer_1=answer_1,
        answer_2_orig=answer_2_orig,
        result_1=result_1,
        result_2=result_2,
        valid_letters=valid_letters,
    )

    print(f"    {debias_note}")

    return {
        "answer": chosen,
        "raw_response": chosen_result["raw_response"],
        "explanation": chosen_result["explanation"] + f"\n{debias_note}",
    }
