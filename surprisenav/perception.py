import re

import torch
from PIL import Image
from qwen_vl_utils import process_vision_info

def patho_r1_describe(
    image: Image.Image,
    question: str,
    choices: str,
    patho_r1_processor,
    patho_r1_model,
    coords: tuple = (0, 0),
    magnification: int = 20,
    max_new_tokens: int = 512,
    structured_max_tokens: int = 288,
    episodic_memory: str = "",
    option_aware: bool = False,
    structured: bool = False,
    missing_info: str = "",
) -> str:

    meta_text = f"[IMAGE META] Patch coordinates: ({coords[0]},{coords[1]}) | Magnification: {magnification}x\n\n"

    q_lower = question.lower()
    _is_biomarker_q = any(
        kw in q_lower for kw in [
            "her2", "erbb2", "receptor", "estrogen", "progesterone",
            "ki-67", "ki67", "ihc", "immunohistochem",
        ]
    )
    if any(kw in q_lower for kw in ["her2", "erbb2"]):
        domain_hint = (
            "For HER2 questions: if you see brown membrane staining, describe intensity and "
            "completeness. Estimate IHC score if possible (0, 1+, 2+, 3+)."
        )
    elif any(kw in q_lower for kw in ["grade", "differentiation"]):
        domain_hint = (
            "For grading: assess tubule/gland formation, nuclear pleomorphism, "
            "and mitotic count (Nottingham criteria)."
        )
    elif any(kw in q_lower for kw in ["diagnosis", "type", "carcinoma", "histologic", "histological"]):
        domain_hint = (
            "For diagnosis: describe growth pattern (tubular/solid/single-file/papillary), "
            "invasion pattern, cell type, and stromal reaction."
        )
    elif any(kw in q_lower for kw in ["receptor", "estrogen", "progesterone"]):
        domain_hint = (
            "For receptor-status questions: describe any visible nuclear staining pattern "
            "and supportive morphology (grade, differentiation)."
        )
    else:
        domain_hint = ""

    if option_aware and choices:
        prompt_body = (
            f"You are examining this tissue patch to help answer: {question}\n\n"
            f"The clinical question asks: {question}\n"
            f"The possible answers are: {choices}\n"
            "Focus your description on morphological features that DISTINGUISH between these options.\n\n"
        )
        if domain_hint:
            prompt_body += domain_hint + "\n"
    else:
        prompt_body = (
            f"You are examining this tissue patch to help answer: {question}\n\n"
            "Focus your description on features DIRECTLY RELEVANT to this question.\n"
        )
        if domain_hint:
            prompt_body += domain_hint + "\n"

    if magnification <= 10:
        prompt_body += (
            "\nCRITICAL CONSTRAINTS for low-magnification view:\n"
            "- Do NOT infer exact tumor size, TNM stage, nodal status, margin status, "
            "multifocality count, or clinical history from this patch.\n"
            "- Do NOT mention immunohistochemistry staining percentage or receptor score "
            "unless the image explicitly shows an immunostain.\n"
            "- Describe ONLY visible global architecture, border visibility, tissue distribution, "
            "and whether full tumor extent is assessable.\n"
            "- If exact measurement is not directly visible, say so explicitly.\n"
        )

    if _is_biomarker_q:
        prompt_body += (
            "\nIMPORTANT: only mention biomarker staining or score if an immunostain is "
            "explicitly visible in this patch. If only H&E is visible, describe the "
            "morphology that is actually there rather than inventing IHC intensity.\n"
        )

    if missing_info:
        prompt_body += f"\nFocus area (from previous round): {missing_info}\n"

    if structured:

        prompt_body += (
            "\nOUTPUT: return ONE compact JSON object. No prose, no markdown fences. "
            "Values must be short phrases (not paragraphs).\n"
            "Fields: relevant, confidence, limitation, summary, visible_features"
        )
        if magnification <= 10:
            prompt_body += ", border_visibility, extent_assessable, separate_foci_visible"
        prompt_body += ".\n"
        if magnification <= 10:
            prompt_body += (
                "Example: {\"relevant\":\"yes\",\"confidence\":\"medium\",\"limitation\":\"cellular detail unclear\","
                "\"summary\":\"infiltrative tumor with irregular borders\","
                "\"visible_features\":[\"irregular nests\",\"desmoplastic stroma\"],"
                "\"border_visibility\":\"partial\",\"extent_assessable\":\"no\",\"separate_foci_visible\":\"unclear\"}\n"
            )
        else:
            prompt_body += (
                "Example: {\"relevant\":\"yes\",\"confidence\":\"high\",\"limitation\":\"none\","
                "\"summary\":\"ductal carcinoma with high nuclear grade\","
                "\"visible_features\":[\"pleomorphic nuclei\",\"frequent mitoses\"]}\n"
            )
        prompt_body += (
            "Hard guardrail: do NOT claim final slide-level conclusions (definitive ER/PR/HER2/Ki-67 "
            "status, exact size, TNM, margin status) from a routine patch. You MAY describe supporting "
            "morphology or explicitly visible immunostain only.\n"
            "Return the JSON now."
        )
    else:

        prompt_body += (
            "\nIf this region does not contain relevant evidence, state "
            "'No relevant features for the question found here' but still briefly "
            "describe what IS visible.\n"
            "Provide your morphological assessment with a clear conclusion. "
            "Do NOT select any answer choice letter (A/B/C/D)."
        )

    full_text = meta_text + prompt_body

    if structured:
        system_prompt = (
            "You output a SINGLE compact JSON object describing what you see in this pathology patch. "
            "No prose. No markdown fences. Short values only."
        )
    else:
        system_prompt = (
            "You are an expert pathology image analyst. Your task is to describe "
            "morphological features that help answer a specific clinical question. "
            "Be focused and diagnostic. Provide a clear morphological impression."
        )

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": full_text},
            ],
        },
    ]

    try:
        text = patho_r1_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = patho_r1_processor(
            text=[text],
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        ).to(patho_r1_model.device)

        _gen_max_tokens = structured_max_tokens if structured else max_new_tokens

        with torch.no_grad():
            generated_ids = patho_r1_model.generate(
                **inputs,
                max_new_tokens=_gen_max_tokens,
                do_sample=False,
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = patho_r1_processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0].strip()

        output_text = re.sub(r'<think>.*?</think>\s*', '', output_text, flags=re.DOTALL)
        output_text = re.sub(r'<answer>.*?</answer>\s*', '', output_text, flags=re.DOTALL)
        output_text = output_text.strip()

        del inputs, generated_ids, generated_ids_trimmed
        torch.cuda.empty_cache()

        return output_text if output_text else "[No description generated]"

    except Exception as e:
        return f"[VLM Error: {str(e)[:200]}]"
