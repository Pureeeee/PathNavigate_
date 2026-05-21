import re

import torch
from qwen_vl_utils import process_vision_info
from data_processing.utils import extract_answer_letter

from .json_utils import _extract_json

def vl_adjudicator_answer(
    question: str,
    choices: str,
    patch_images: list,
    patch_infos: list,
    vl_processor,
    vl_model,
    is_mcq: bool = True,
    q_type: str = "other",
    max_new_tokens: int = 2048,
    memory_context: str = "",
) -> dict:

    if q_type == "morphology":
        system_prompt = (
            "You are an expert pathologist examining tissue patches from a whole slide image. "
            "You are performing DIFFERENTIAL DIAGNOSIS based on microscopic morphology.\n"
            "Key discriminators: single-file invasion -> lobular; tubule formation -> ductal; "
            "medullary pattern -> syncytial growth + lymphocytes; mucinous -> extracellular mucin pools.\n"
            "For grading: assess tubule formation, nuclear pleomorphism, and mitotic count.\n"
            "IMPORTANT: Always commit to a specific diagnosis based on the morphological evidence you observe.\n"
        )
    else:
        system_prompt = (
            "You are an expert pathologist examining tissue patches from a whole slide image. "
            "Synthesize your observations across ALL patches to answer the clinical question.\n"
            "Identify the most consistent pattern, resolve contradictions by favoring the predominant finding.\n"
            "IMPORTANT: Always commit to a specific answer. Do NOT answer 'cannot be determined' "
            "unless truly no relevant evidence is visible.\n"
        )

    if is_mcq and choices:
        system_prompt += (
            "\nFor multiple choice: assess each option against what you observe. "
            "Eliminate contradicted options, select the one with strongest visual support.\n"
            "Your answer MUST be ONLY the letter (A, B, C, or D).\n"
            "Respond in JSON: {\"answer\": \"A\", \"explanation\": \"brief reasoning\"}"
        )
    else:
        q_lower = question.lower()
        if any(kw in q_lower for kw in ["her2", "erbb2", "her-2"]):
            system_prompt += (
                "\nFor HER2: answer MUST be one of: '0', '1+', '2+', or '3+'.\n"
                "Respond in JSON: {\"answer\": \"your answer\", \"explanation\": \"brief reasoning\"}"
            )
        else:
            system_prompt += (
                "\nRespond in JSON: {\"answer\": \"your answer\", \"explanation\": \"brief reasoning\"}"
            )

    content_parts = []
    for i, (img, info) in enumerate(zip(patch_images, patch_infos)):
        content_parts.append({"type": "image", "image": img})
        coord_str = f"({info['coords'][0]},{info['coords'][1]})"
        content_parts.append({
            "type": "text",
            "text": f"[Patch {i+1}: {info['magnification']}x, Coord: {coord_str}, "
                    f"Surprise: {info['surprise_score']:.3f}]"
        })

    user_text = f"\nQuestion: {question}\n"
    if is_mcq and choices:
        user_text += f"Choices: {choices}\n"
    if memory_context:
        user_text += f"\n=== Reference Context ===\n{memory_context}\n=== End Context ===\n"
    user_text += (
        f"\nYou have been shown {len(patch_images)} tissue patches from this slide. "
        "Examine all patches carefully, then answer the question based on your observations. "
        "Respond in JSON."
    )
    content_parts.append({"type": "text", "text": user_text})

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content_parts},
    ]

    try:
        text = vl_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = vl_processor(
            text=[text],
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        ).to(vl_model.device)

        with torch.no_grad():
            generated_ids = vl_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        raw_response = vl_processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

        del inputs, generated_ids, generated_ids_trimmed
        torch.cuda.empty_cache()

        thinking_text = ""
        if '</think>' in raw_response:
            last_think_end = raw_response.rfind('</think>')
            thinking_text = raw_response[:last_think_end].strip()
            raw_response = raw_response[last_think_end + len('</think>'):].strip()
        else:
            raw_response = re.sub(r'<think>.*?</think>\s*', '', raw_response, flags=re.DOTALL).strip()

        parsed = _extract_json(raw_response)
        if not parsed and thinking_text:
            parsed = _extract_json(thinking_text)
        if parsed:
            answer_text = str(parsed.get("answer", "")).strip()
            explanation = str(parsed.get("explanation", "")).strip()
        else:
            if len(raw_response) > 500:
                answer_text = raw_response[-200:]
            else:
                answer_text = raw_response
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

    except Exception as e:
        return {
            "answer": "",
            "raw_response": f"[VL Error: {str(e)[:300]}]",
            "explanation": f"VL adjudicator failed: {str(e)[:200]}",
        }
