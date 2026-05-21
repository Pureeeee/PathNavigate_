import gc
import json
import os
import random
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer, Qwen2_5_VLForConditionalGeneration

try:
    from transformers import Qwen3VLForConditionalGeneration
    _QWEN3_VL_AVAILABLE = True
except ImportError:
    _QWEN3_VL_AVAILABLE = False

try:
    from transformers import Qwen3_5ForConditionalGeneration
    _QWEN35_AVAILABLE = True
except ImportError:
    _QWEN35_AVAILABLE = False

try:
    from flash_attn import flash_attn_func
    _FLASH_ATTN_AVAILABLE = True
except ImportError:
    _FLASH_ATTN_AVAILABLE = False

from data_processing.feature_utils import close_all_slides, extract_coords_from_name, extract_patch_from_svs, find_svs_path, get_20x_patches_in_region, get_wsi_full_id_from_h5, load_feature_cache, load_plip_feature_cache
from data_processing.utils import load_all_vqa_pairs, make_unique_id
from models.surprise_navigation import generate_surprise_heatmap, nms_jump_targets
from models.yaad_memory import YaadMemory
from visualization.heatmap_viz import log_surprise_stats, visualize_surprise_heatmap

from .adjudication import adjudicator_answer, adjudicator_answer_vllm, adjudicator_vllm_with_debiasing, adjudicator_with_debiasing, _verify_answer_vllm
from .config import parse_args
from .evidence import _normalize_structured_patch, _parse_structured_patch_output, _structured_patch_to_evidence_line
from .navigation import SurpriseCache, _build_global_memory_report, _build_meanpool_report, _sanitize_lowmag_evidence, _should_start_with_macro, load_plip, query_guided_selection
from .perception import patho_r1_describe
from .retrieval import _load_rag_index, _retrieve_rag_context, _retrieve_textbook_knowledge
from .vl_adjudicator import vl_adjudicator_answer

def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    if args.save_heatmap:
        os.makedirs(os.path.join(args.save_dir, "heatmaps"), exist_ok=True)

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        print(f"  Random seed fixed: {args.seed}")

    device = args.device

    if args.use_plip_features:
        args.yaad_dim = 512
        print(f"  [PLIP mode] yaad_dim auto-set to 512")

    print("=" * 60)
    print("  SurpriseNav: Training-Free WSI VQA Agent")
    print("=" * 60)
    print(f"  Features: 5x={args.feature_dir_5x}")
    print(f"           20x={args.feature_dir_20x}")
    print(f"  SVS dir:  {args.svs_dir}")
    if args.use_vl_adjudicator:
        print(f"  Model:    Qwen3-VL={args.vl_ckpt} (unified VLM+Adjudicator)")
    else:
        print(f"  Model:    Patho-R1={args.patho_r1_ckpt}")
        print(f"            Qwen={args.qwen_ckpt}")
    if args.yaad_hidden_dim <= 0:
        args.yaad_hidden_dim = args.yaad_dim

    print(
        f"  Settings: top_k={args.top_k_jumps}, yaad_dim={args.yaad_dim}, "
        f"yaad_hidden_dim={args.yaad_hidden_dim}, yaad_num_layers={args.yaad_num_layers}, lr={args.yaad_lr}"
    )
    if args.query_navigation:
        print(f"  Query Nav: PLIP={args.plip_ckpt}, alpha={args.query_alpha}, candidates={args.surprise_candidates}")
    if args.ablation != "none":
        print(f"  Ablation: {args.ablation}")
    print("=" * 60)

    pairs = load_all_vqa_pairs(args.questions_file, dataset_name=args.dataset_name)

    if args.max_samples > 0:
        pairs = pairs[:args.max_samples]
        print(f"  [DEBUG] Limited to first {args.max_samples} samples")

    if args.num_shards > 1:
        assert 0 <= args.shard_id < args.num_shards, \
            f"shard_id={args.shard_id} must be in [0, {args.num_shards})"
        pairs = [p for i, p in enumerate(pairs) if i % args.num_shards == args.shard_id]
        print(f"  [Shard {args.shard_id}/{args.num_shards}] Processing {len(pairs)} samples")

    print("\nLoading models...")

    vl_model, vl_processor = None, None
    tokenizer, qwen_model = None, None
    patho_r1_model, patho_r1_processor = None, None

    if args.use_vl_adjudicator:
        print(f"Loading VL Adjudicator on {args.vl_device}...")

        if _QWEN35_AVAILABLE:
            try:
                vl_model = Qwen3_5ForConditionalGeneration.from_pretrained(
                    args.vl_ckpt, torch_dtype="auto", device_map={"": args.vl_device}
                )
                print(f"  Loaded as Qwen3.5: {args.vl_ckpt}")
            except Exception:
                vl_model = None
        if vl_model is None and _QWEN3_VL_AVAILABLE:
            vl_model = Qwen3VLForConditionalGeneration.from_pretrained(
                args.vl_ckpt, torch_dtype="auto", device_map={"": args.vl_device}
            )
            print(f"  Loaded as Qwen3-VL: {args.vl_ckpt}")
        assert vl_model is not None, "No VL model class available. Upgrade transformers."
        vl_processor = AutoProcessor.from_pretrained(args.vl_ckpt)
        print(f"  Qwen3-VL loaded: {args.vl_ckpt}")
    else:

        _attn_impl = "flash_attention_2" if _FLASH_ATTN_AVAILABLE else "sdpa"

        if not args.vllm_url:
            tokenizer = AutoTokenizer.from_pretrained(args.qwen_ckpt, trust_remote_code=True)
            qwen_model = AutoModelForCausalLM.from_pretrained(
                args.qwen_ckpt, torch_dtype="auto", device_map={"": args.qwen_device},
                trust_remote_code=True, attn_implementation=_attn_impl,
            )
        else:
            print(f"  [vLLM mode] Skipping local Qwen loading, using API at {args.vllm_url}")

        patho_r1_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.patho_r1_ckpt, torch_dtype="auto", device_map={"": args.vlm_device},
            attn_implementation=_attn_impl,
        )
        patho_r1_processor = AutoProcessor.from_pretrained(args.patho_r1_ckpt)

    plip_model, plip_processor = None, None
    if args.query_navigation and args.ablation == "none":
        plip_device = args.plip_device or args.vlm_device
        print(f"Loading PLIP for query navigation on {plip_device}...")
        plip_model, plip_processor = load_plip(args.plip_ckpt, device=plip_device)
        print("PLIP loaded.")

    rag_index = _load_rag_index(args.rag_index)

    print("Models loaded.\n")

    surprise_cache = SurpriseCache() if args.cache_surprise else None

    finished_cases = set(
        f.replace(".json", "") for f in os.listdir(args.save_dir) if f.endswith(".json")
    )

    total_count = 0
    skipped_count = 0

    _MORPHOLOGY_KW = ["diagnosis", "type", "grade", "differentiation", "in-situ",
                      "invasion", "carcinoma", "tumor type", "histologic", "histological"]
    _CLINICAL_KW = ["size", "stage", "margin", "lymph node", "metastasis",
                    "receptor", "her2", "ki-67", "prognosis"]
    type_stats = {"morphology": 0, "clinical": 0, "other": 0}

    for idx, pair in enumerate(pairs):
        wsi_id = pair["long_id"]
        question = pair["question"]
        choices = pair["choices"]
        gt_answer = pair["answer"]
        is_mcq = pair.get("is_mcq", bool(choices))
        unique_id = make_unique_id(wsi_id, question)

        if unique_id in finished_cases:
            skipped_count += 1
            continue

        t0 = time.time()
        print(f"\n{'-'*50}")
        print(f"[{idx+1}/{len(pairs)}] WSI: {wsi_id}")
        print(f"  Q: {question[:80]}...")
        if choices:
            print(f"  Choices: {choices[:80]}...")

        if args.use_plip_features:

            feature_cache_5x = load_plip_feature_cache(args.plip_feat_dir, wsi_id)
            feature_cache_20x = {}
        else:

            feature_cache_5x = load_feature_cache(args.feature_dir_5x, wsi_id)
            feature_cache_20x = load_feature_cache(args.feature_dir_20x, wsi_id)

        if not feature_cache_5x:
            print(f"  [WARN] No 5x features found for {wsi_id}. Skipping.")
            skipped_count += 1
            continue

        wsi_full_id = get_wsi_full_id_from_h5(args.feature_dir_5x, wsi_id)
        if wsi_full_id is None:
            wsi_full_id = wsi_id

        print(f"  Features: 5x={len(feature_cache_5x)} patches, 20x={len(feature_cache_20x)} patches")
        print(f"  Full WSI ID: {wsi_full_id}")

        q_lower = question.lower()
        if any(kw in q_lower for kw in _MORPHOLOGY_KW):
            q_type = "morphology"
        elif any(kw in q_lower for kw in _CLINICAL_KW):
            q_type = "clinical"
        else:
            q_type = "other"
        print(f"  Q-type: {q_type}")

        _is_survival_q = ("survival" in q_lower and
                          any(w in q_lower for w in ["time", "predict", "day", "long"]))
        _is_vital_q = ("vital" in q_lower and "status" in q_lower)

        if _is_survival_q or _is_vital_q:
            if _is_survival_q:
                default_answer = "cannot be determined from histology"
                unanswerable_type = "survival_time"
            else:
                default_answer = "Alive"
                unanswerable_type = "vital_status"

            pred_answer = default_answer
            total_count += 1
            type_stats[q_type] += 1

            elapsed = time.time() - t0
            print(f"  [Unanswerable:{unanswerable_type}] Default: {default_answer}")
            print(f"  -> Pred: {pred_answer[:50]} | {elapsed:.1f}s | [{q_type}]")

            case_result = {
                "long_id": wsi_id,
                "wsi_full_id": wsi_full_id if wsi_full_id else wsi_id,
                "question": question,
                "choices": choices,
                "pred_answer": pred_answer,
                "is_mcq": is_mcq,
                "q_type": q_type,
                "explanation": f"Unanswerable ({unanswerable_type}): default answer used",
                "raw_response": "",
                "evidence": [],
                "jump_targets": [],
                "vlm_calls": 0,
                "time_seconds": round(elapsed, 1),
                "ablation": args.ablation,
                "query_navigation": False,
                "query_alpha": None,
                "total_5x_patches": len(feature_cache_5x),
                "total_20x_patches": len(feature_cache_20x),
                "total_20x_scanned": 0,
                "high_surprise_ratio": 0.0,
                "num_jump_targets": 0,
                "unanswerable": unanswerable_type,
            }
            if args.save_ground_truth_for_debug:
                case_result["ground_truth"] = gt_answer

            with open(os.path.join(args.save_dir, f"{unique_id}.json"), "w", encoding="utf-8") as f:
                json.dump(case_result, f, indent=2, ensure_ascii=False)

            del feature_cache_5x, feature_cache_20x
            gc.collect()
            continue

        yaad_5x_stats = None
        if surprise_cache and surprise_cache.has(wsi_id):
            print("  [Cache Hit] Reusing 5x surprise scan")
            cached = surprise_cache.get(wsi_id)

            surprise_map_5x, coord_surprise_5x, heatmap_5x, grid_meta_5x, yaad_5x_stats = cached
        else:
            print("  [Step 1] 5x Yaad global scan...")

            yaad_5x = YaadMemory(
                input_dim=args.yaad_dim,
                hidden_dim=args.yaad_hidden_dim,
                num_layers=args.yaad_num_layers,
                lr=args.yaad_lr,
                huber_delta=args.yaad_huber_delta,
                forget_alpha=args.yaad_forget_alpha,
                warmup_steps=min(args.yaad_warmup, len(feature_cache_5x)),
                threshold_std_scale=args.yaad_threshold_scale,
                reverse_update=args.yaad_reverse_update,
            ).to(device)

            surprise_map_5x, coord_surprise_5x, heatmap_5x, grid_meta_5x = generate_surprise_heatmap(
                feature_cache_5x, list(feature_cache_5x.keys()), yaad_5x, device=device
            )

            log_surprise_stats(yaad_5x, wsi_id)
            yaad_5x_stats = yaad_5x.get_stats()

            if args.save_heatmap:
                viz_path = os.path.join(args.save_dir, "heatmaps", f"{wsi_id}.png")
                visualize_surprise_heatmap(
                    heatmap_5x, [], grid_meta_5x, save_path=viz_path, title=wsi_id
                )

            if surprise_cache:
                surprise_cache.put(wsi_id, (
                    surprise_map_5x, coord_surprise_5x, heatmap_5x, grid_meta_5x,
                    yaad_5x_stats,
                ))

            del yaad_5x
            torch.cuda.empty_cache()

        if q_type == "morphology":
            current_min_distance = max(args.min_jump_distance, 4096)
            planner_top_k = args.top_k_jumps + 2
        elif q_type == "clinical":
            current_min_distance = max(args.min_jump_distance, 20480)
            planner_top_k = args.top_k_jumps + 3
        else:
            if heatmap_5x is not None:
                h, w = heatmap_5x.shape
                step_px = grid_meta_5x[2]
                diagonal = np.sqrt((h * step_px)**2 + (w * step_px)**2)
                current_min_distance = max(args.min_jump_distance, int(diagonal * 0.08))
            else:
                current_min_distance = args.min_jump_distance
            planner_top_k = args.top_k_jumps

        pool_k = max(args.surprise_candidates, planner_top_k * args.max_reflection_rounds)

        jump_targets_pool = nms_jump_targets(
            heatmap_5x, grid_meta_5x,
            K=pool_k,
            nms_window=args.nms_window,
            min_distance=current_min_distance,
        )
        print(f"  [Step 2] NMS: {len(jump_targets_pool)} targets (q_type={q_type}, "
              f"min_dist={current_min_distance}, pool_k={pool_k})")

        if args.pool_source == "plip" and plip_model is not None:
            svs_path_for_plip = find_svs_path(args.svs_dir, wsi_full_id)
            if svs_path_for_plip:
                all_5x_coords = [extract_coords_from_name(n) for n in feature_cache_5x.keys()]

                if len(all_5x_coords) > 400:
                    all_5x_coords = random.sample(all_5x_coords, 400)
                jump_targets_pool = query_guided_selection(
                    question=question,
                    candidate_coords=all_5x_coords,
                    surprise_scores={c: 0.0 for c in all_5x_coords},
                    svs_path=svs_path_for_plip,
                    plip_model=plip_model,
                    plip_processor=plip_processor,
                    alpha=1.0,
                    top_k=pool_k,
                    magnification=5,
                )
                print(f"  [Step 2] PLIP-built pool: {len(jump_targets_pool)} targets "
                      f"(rebuilt from {len(all_5x_coords)} candidates by PLIP relevance)")

        if plip_model is not None and jump_targets_pool and args.ablation == "none":
            svs_path_for_plip = find_svs_path(args.svs_dir, wsi_full_id)
            if svs_path_for_plip:

                _rerank_k = len(jump_targets_pool) if args.max_reflection_rounds > 1 else planner_top_k
                jump_targets_pool = query_guided_selection(
                    question=question,
                    candidate_coords=jump_targets_pool,
                    surprise_scores=coord_surprise_5x,
                    svs_path=svs_path_for_plip,
                    plip_model=plip_model,
                    plip_processor=plip_processor,
                    alpha=args.query_alpha,
                    top_k=_rerank_k,
                    magnification=5,
                )
                print(f"  [Step 2] PLIP re-ranked: {len(jump_targets_pool)} targets "
                      f"(alpha={args.query_alpha}, pool kept for {args.max_reflection_rounds} rounds)")

        if args.ablation == "random_roi":
            all_5x_names = list(feature_cache_5x.keys())
            k = min(pool_k, len(all_5x_names))
            sampled = random.sample(all_5x_names, k)
            jump_targets_pool = [extract_coords_from_name(n) for n in sampled]
            print(f"  [Ablation: random_roi] {len(jump_targets_pool)} random targets")
        elif args.ablation == "grid_roi":
            all_coords = sorted(set(
                extract_coords_from_name(n) for n in feature_cache_5x.keys()
            ))
            step = max(1, len(all_coords) // pool_k)
            jump_targets_pool = all_coords[::step][:pool_k]
            print(f"  [Ablation: grid_roi] {len(jump_targets_pool)} grid targets")

        if not jump_targets_pool:
            print("  [WARN] No jump targets found. Using fallback.")
            if surprise_map_5x:
                best_patch = max(surprise_map_5x, key=surprise_map_5x.get)
                bx, by = extract_coords_from_name(best_patch)
                jump_targets_pool = [(bx, by)]
            else:
                skipped_count += 1
                continue

        roi_evidence_texts = []
        vlm_calls = 0
        total_20x_scanned = 0
        target_cursor = 0
        final_result = None

        if args.qtype_adaptive_mag and _should_start_with_macro(question, q_type):
            sample_primary_mag = 10
            sample_is_macro_first = True
        else:
            sample_primary_mag = args.vlm_magnification
            sample_is_macro_first = False

        vl_patch_images = []
        vl_patch_infos = []
        structured_traces = []
        macro_confirmation_triggered = False
        first_evidence_magnification = None

        if args.no_nav_summary:
            global_memory_report = ""
        elif args.readout_meanpool:
            global_memory_report = _build_meanpool_report(
                feature_cache_5x, len(feature_cache_5x),
                jump_targets_pool[:planner_top_k]
            )
        else:
            global_memory_report = _build_global_memory_report(
                yaad_5x_stats, len(feature_cache_5x), coord_surprise_5x,
                jump_targets_pool[:planner_top_k]
            )

        episodic_memory = ""

        rag_context = _retrieve_rag_context(
            feature_cache_5x, rag_index, top_k=args.rag_top_k, question=question
        )

        controller_requested_mag = None
        controller_trace_rounds = []
        pending_missing_info = ""

        for reflection_round in range(1, args.max_reflection_rounds + 1):

            if args.vllm_url and args.max_reflection_rounds > 1 and reflection_round > 1:
                round_vlm_budget = args.max_vlm_calls
                print(f"  [Round {reflection_round}] Fresh VLM budget: {round_vlm_budget}")
            else:
                round_vlm_budget = args.max_vlm_calls - vlm_calls

            _evidence_before = len(vl_patch_images) if args.use_vl_adjudicator else len(roi_evidence_texts)

            remaining_pool = len(jump_targets_pool) - target_cursor
            remaining_rounds = args.max_reflection_rounds - reflection_round + 1
            round_batch_k = max(1, remaining_pool // remaining_rounds) if remaining_rounds > 0 else remaining_pool

            round_targets = jump_targets_pool[target_cursor:target_cursor + round_batch_k]

            if not round_targets:
                print(f"  [Round {reflection_round}] No more targets available")
                break

            print(f"  [Round {reflection_round}] Processing {len(round_targets)} targets "
                  f"(pool remaining: {remaining_pool}, batch: {round_batch_k}, "
                  f"VLM calls so far: {vlm_calls}, round budget: {round_vlm_budget})")

            effective_primary_mag = (
                controller_requested_mag if controller_requested_mag is not None
                else sample_primary_mag
            )

            round_vlm_start = vlm_calls

            for target_idx, (target_x, target_y) in enumerate(round_targets):
                if (vlm_calls - round_vlm_start) >= round_vlm_budget:
                    break

                local_20x_patches = get_20x_patches_in_region(
                    target_x, target_y, feature_cache_20x,
                    radius=args.local_20x_radius,
                )

                skip_20x_yaad = (args.use_plip_features
                                 or effective_primary_mag != 20
                                 or not local_20x_patches)

                if not local_20x_patches and not skip_20x_yaad:
                    continue

                if skip_20x_yaad:

                    top_patches_info = [(target_x, target_y,
                                         coord_surprise_5x.get((target_x, target_y), 0.0))]
                elif args.ablation == "no_20x":
                    top_patches_info = [(target_x, target_y,
                                         coord_surprise_5x.get((target_x, target_y), 0.0))]
                else:

                    total_20x_scanned += len(local_20x_patches)
                    yaad_20x = YaadMemory(
                        input_dim=args.yaad_dim,
                        hidden_dim=args.yaad_hidden_dim,
                        num_layers=args.yaad_num_layers,
                        lr=args.yaad_lr,
                        huber_delta=args.yaad_huber_delta,
                        forget_alpha=args.yaad_forget_alpha,
                        warmup_steps=min(max(args.yaad_warmup // 5, 5), len(local_20x_patches)),
                        threshold_std_scale=args.yaad_threshold_scale,
                        reverse_update=args.yaad_reverse_update,
                    ).to(device)

                    local_surprise, _, _, _ = generate_surprise_heatmap(
                        feature_cache_20x, local_20x_patches, yaad_20x, device=device
                    )

                    del yaad_20x
                    torch.cuda.empty_cache()

                    if not local_surprise:
                        continue

                    sorted_patches = sorted(local_surprise.items(), key=lambda x: x[1], reverse=True)
                    top_n = min(args.top_n_per_roi, len(sorted_patches))
                    top_patches_info = []
                    for patch_name, score in sorted_patches[:top_n]:
                        px, py = extract_coords_from_name(patch_name)
                        top_patches_info.append((px, py, score))

                svs_path = find_svs_path(args.svs_dir, wsi_full_id)
                if svs_path is None:
                    for px, py, score in top_patches_info:
                        roi_evidence_texts.append(
                            f"- [Surprise: {score:.3f}, Coord: ({px},{py}), 20x] "
                            f"High-surprise region detected but SVS unavailable."
                        )
                    continue

                for px, py, score in top_patches_info:
                    if (vlm_calls - round_vlm_start) >= round_vlm_budget:
                        break

                    if args.multi_scale and controller_requested_mag is None and vlm_calls < 3:
                        current_mag = args.secondary_magnification
                    else:
                        current_mag = effective_primary_mag

                    try:
                        img = extract_patch_from_svs(svs_path, px, py, magnification=current_mag)
                    except Exception as e:
                        print(f"    ROI ({px},{py}): Failed to extract patch: {e}")
                        continue

                    vlm_calls += 1

                    if args.use_vl_adjudicator:

                        vl_patch_images.append(img)
                        vl_patch_infos.append({
                            "coords": (px, py),
                            "magnification": current_mag,
                            "surprise_score": score,
                        })
                        print(f"    R{reflection_round} ({px},{py}): surprise={score:.3f}, "
                              f"collected for VL adjudicator")
                    else:

                        desc = patho_r1_describe(
                            image=img,
                            question=question,
                            choices=choices,
                            patho_r1_processor=patho_r1_processor,
                            patho_r1_model=patho_r1_model,
                            coords=(px, py),
                            magnification=current_mag,
                            structured_max_tokens=args.structured_max_tokens,
                            episodic_memory=episodic_memory,
                            option_aware=(is_mcq and args.option_aware_vlm),
                            structured=args.structured_evidence,
                            missing_info=pending_missing_info,
                        )

                        _patch_trace = None
                        if args.structured_evidence:

                            _parsed_patch, _parse_err = _parse_structured_patch_output(desc)
                            _norm = None
                            _validation_err = None
                            _normalize_meta = {
                                "soft_defaults_used": [],
                                "summary_imputed": False,
                                "relevant_imputed": False,
                                "confidence_imputed": False,
                                "lowmag_defaults_used": [],
                                "hard_valid": False,
                            }
                            if _parsed_patch is not None:
                                _norm, _validation_err, _normalize_meta = _normalize_structured_patch(
                                    _parsed_patch, current_mag
                                )

                            if _norm is not None:

                                if _norm.get("summary"):
                                    _norm["summary"] = _sanitize_lowmag_evidence(_norm["summary"], current_mag)
                                if _norm.get("visible_features") and current_mag <= 10:
                                    _norm["visible_features"] = [
                                        _sanitize_lowmag_evidence(f, current_mag)
                                        for f in _norm["visible_features"]
                                    ]
                                evidence_line = _structured_patch_to_evidence_line(
                                    _norm, (px, py), current_mag, score, vlm_calls
                                )
                                _patch_trace = {
                                    "raw_output": desc,
                                    "parsed_output": _norm,
                                    "parse_failed": False,
                                    "parse_error": None,
                                    "parse_status": _parse_err or "direct_parse",
                                    "validation_error": None,
                                    "repair_applied": _repair_was_applied(_parse_err),
                                    "hard_valid": bool(_normalize_meta.get("hard_valid", False)),
                                    "soft_defaults_used": _normalize_meta.get("soft_defaults_used", []),
                                    "summary_imputed": _normalize_meta.get("summary_imputed", False),
                                    "relevant_imputed": _normalize_meta.get("relevant_imputed", False),
                                    "confidence_imputed": _normalize_meta.get("confidence_imputed", False),
                                    "lowmag_defaults_used": _normalize_meta.get("lowmag_defaults_used", []),
                                    "magnification": current_mag,
                                    "coords": (px, py),
                                    "region_num": vlm_calls,
                                }
                            else:

                                evidence_line = (
                                    f"- [Region {vlm_calls}, Surprise: {score:.3f}, "
                                    f"Coord: ({px},{py}), {current_mag}x] "
                                    f"[structured parse failed] patch produced unusable structured output"
                                )
                                _patch_trace = {
                                    "raw_output": desc,
                                    "parsed_output": None,
                                    "parse_failed": True,
                                    "parse_error": _parse_err,
                                    "parse_status": _parse_err or "validation_failed",
                                    "validation_error": _validation_err,
                                    "repair_applied": _repair_was_applied(_parse_err),
                                    "hard_valid": False,
                                    "soft_defaults_used": _normalize_meta.get("soft_defaults_used", []),
                                    "summary_imputed": _normalize_meta.get("summary_imputed", False),
                                    "relevant_imputed": _normalize_meta.get("relevant_imputed", False),
                                    "confidence_imputed": _normalize_meta.get("confidence_imputed", False),
                                    "lowmag_defaults_used": _normalize_meta.get("lowmag_defaults_used", []),
                                    "magnification": current_mag,
                                    "coords": (px, py),
                                    "region_num": vlm_calls,
                                }
                        else:

                            desc = _sanitize_lowmag_evidence(desc, current_mag)
                            evidence_line = (
                                f"- [Region {vlm_calls}, Surprise: {score:.3f}, "
                                f"Coord: ({px},{py}), {current_mag}x] {desc}"
                            )

                        roi_evidence_texts.append(evidence_line)
                        if _patch_trace:
                            structured_traces.append(_patch_trace)

                        if first_evidence_magnification is None:
                            first_evidence_magnification = current_mag

                        brief_desc = desc[:100].rsplit(" ", 1)[0] if len(desc) > 100 else desc
                        episodic_memory += (
                            f"R{vlm_calls}({px},{py}): {brief_desc}\n"
                        )

                        print(f"    R{reflection_round} ({px},{py}): surprise={score:.3f}, "
                              f"desc={desc[:60]}...")

            _evidence_after = len(vl_patch_images) if args.use_vl_adjudicator else len(roi_evidence_texts)
            _round_has_new_evidence = _evidence_after > _evidence_before

            if not _round_has_new_evidence:

                target_cursor += len(round_targets)
                print(f"  [Round {reflection_round}] No new evidence from {len(round_targets)} targets, skipping")
                continue

            target_cursor += len(round_targets)

            _mcq_can_reflect = (not is_mcq) or bool(args.vllm_url)

            if args.vllm_url and args.max_reflection_rounds > 1:

                _vlm_budget_ok = True
            else:
                _vlm_budget_ok = vlm_calls < args.max_vlm_calls
            can_reflect = (reflection_round < args.max_reflection_rounds
                           and target_cursor < len(jump_targets_pool)
                           and _vlm_budget_ok
                           and _mcq_can_reflect)

            memory_ctx_parts = []
            if global_memory_report:
                memory_ctx_parts.append(global_memory_report)
            if rag_context:
                memory_ctx_parts.append(rag_context)
            if args.textbook_rag:
                textbook_ctx = _retrieve_textbook_knowledge(question, q_type)
                if textbook_ctx:
                    memory_ctx_parts.append(textbook_ctx)
            memory_context = "\n".join(memory_ctx_parts)

            if args.use_vl_adjudicator:
                result = vl_adjudicator_answer(
                    question=question, choices=choices,
                    patch_images=vl_patch_images, patch_infos=vl_patch_infos,
                    vl_processor=vl_processor, vl_model=vl_model,
                    is_mcq=is_mcq, q_type=q_type, memory_context=memory_context,
                )
            elif args.vllm_url:
                if can_reflect:

                    result = adjudicator_answer_vllm(
                        question=question, choices=choices,
                        evidence_texts=roi_evidence_texts,
                        vllm_url=args.vllm_url,
                        vllm_model_name=args.vllm_model or args.qwen_ckpt,
                        is_mcq=is_mcq, q_type=q_type,
                        memory_context=memory_context,
                        is_final_round=False,
                    )
                else:

                    _final_fn = (adjudicator_vllm_with_debiasing
                                 if args.mcq_debiasing and is_mcq
                                 else adjudicator_answer_vllm)
                    _final_kwargs = dict(
                        question=question, choices=choices,
                        evidence_texts=roi_evidence_texts,
                        vllm_url=args.vllm_url,
                        vllm_model_name=args.vllm_model or args.qwen_ckpt,
                        is_mcq=is_mcq, q_type=q_type,
                        memory_context=memory_context,
                    )
                    if _final_fn == adjudicator_answer_vllm:
                        _final_kwargs["is_final_round"] = True
                    result = _final_fn(**_final_kwargs)
            else:
                _adjudicator_fn = (adjudicator_with_debiasing
                                   if args.mcq_debiasing and is_mcq
                                   else adjudicator_answer)
                result = _adjudicator_fn(
                    question=question, choices=choices,
                    evidence_texts=roi_evidence_texts,
                    tokenizer=tokenizer, model=qwen_model,
                    is_mcq=is_mcq, q_type=q_type,
                    reflection_mode=can_reflect,
                    memory_context=memory_context,
                )

            _is_controller_round = bool(args.vllm_url) and can_reflect
            _controller_parsed = "sufficient" in result

            if _is_controller_round and not _controller_parsed:

                print(f"  [Round {reflection_round}] Controller: parse failed, defaulting to insufficient")
                sufficient = "no"
                missing_info = question
                next_action = "need_more_regions"
                next_mag = sample_primary_mag
                next_query = question
            else:
                sufficient = str(result.get("sufficient", "yes")).lower().strip()
                missing_info = result.get("missing_info", "")
                next_action = result.get("next_action", "done")
                next_mag = result.get("next_magnification", sample_primary_mag)
                next_query = result.get("next_query", "")

            controller_trace_rounds.append({
                "round": reflection_round,
                "sufficient": sufficient,
                "missing_info": missing_info,
                "next_action": next_action,
                "next_magnification": next_mag,
                "next_query": next_query,
                "parse_failed": (_is_controller_round and not _controller_parsed),
            })

            _SAFE_ACTIONS = {"need_micro_detail"}
            _action_allowed = next_action in _SAFE_ACTIONS

            if can_reflect and sufficient == "no" and _action_allowed:

                print(f"  [Round {reflection_round}] Controller: INSUFFICIENT (action={next_action})")
                print(f"    missing_info: {missing_info}")
                print(f"    next_action: {next_action}, next_mag: {next_mag}x")
                print(f"    next_query: {next_query}")

                pending_missing_info = missing_info or next_query or ""

                _VALID_MAGS = {5, 10, 20, 40}
                try:
                    _parsed_mag = int(float(str(next_mag)))
                except (ValueError, TypeError):
                    _parsed_mag = None
                if _parsed_mag in _VALID_MAGS:
                    controller_requested_mag = _parsed_mag
                    print(f"    -> Next round magnification set to {controller_requested_mag}x")

                remaining_targets = jump_targets_pool[target_cursor:]
                if next_query and remaining_targets and plip_model is not None:
                    rerank_query = f"{question} {next_query}"
                    svs_for_rerank = find_svs_path(args.svs_dir, wsi_full_id)
                    if svs_for_rerank:
                        try:
                            reranked = query_guided_selection(
                                question=rerank_query,
                                candidate_coords=remaining_targets,
                                surprise_scores=coord_surprise_5x,
                                svs_path=svs_for_rerank,
                                plip_model=plip_model,
                                plip_processor=plip_processor,
                                alpha=0.7,
                                top_k=len(remaining_targets),
                                magnification=5,
                            )
                            jump_targets_pool = (
                                jump_targets_pool[:target_cursor] + reranked
                            )
                            print(f"    -> Re-ranked {len(reranked)} remaining targets "
                                  f"with query: '{next_query[:50]}'")
                        except Exception as e:
                            print(f"    -> Re-rank failed: {e}")

                continue

            elif can_reflect and sufficient == "no" and sample_is_macro_first:

                print(f"  [Round {reflection_round}] Controller: {next_action} -> 20x confirmation + rerank")
                controller_requested_mag = 20
                macro_confirmation_triggered = True
                pending_missing_info = missing_info or next_query or ""

                remaining_targets = jump_targets_pool[target_cursor:]
                if next_query and remaining_targets and plip_model is not None:
                    rerank_query = f"{question} {next_query}"
                    svs_for_rerank = find_svs_path(args.svs_dir, wsi_full_id)
                    if svs_for_rerank:
                        try:
                            reranked = query_guided_selection(
                                question=rerank_query,
                                candidate_coords=remaining_targets,
                                surprise_scores=coord_surprise_5x,
                                svs_path=svs_for_rerank,
                                plip_model=plip_model,
                                plip_processor=plip_processor,
                                alpha=0.7,
                                top_k=len(remaining_targets),
                                magnification=5,
                            )
                            jump_targets_pool = (
                                jump_targets_pool[:target_cursor] + reranked
                            )
                            print(f"    -> Re-ranked {len(reranked)} remaining targets "
                                  f"with query: '{next_query[:50]}'")
                        except Exception as e:
                            print(f"    -> Re-rank failed: {e}")
                continue

            elif can_reflect and sufficient == "no":

                print(f"  [Round {reflection_round}] Controller: {next_action} BLOCKED -> final answer")
                if args.vllm_url:
                    _final_fn = (adjudicator_vllm_with_debiasing
                                 if args.mcq_debiasing and is_mcq
                                 else adjudicator_answer_vllm)
                    _final_kwargs = dict(
                        question=question, choices=choices,
                        evidence_texts=roi_evidence_texts,
                        vllm_url=args.vllm_url,
                        vllm_model_name=args.vllm_model or args.qwen_ckpt,
                        is_mcq=is_mcq, q_type=q_type,
                        memory_context=memory_context,
                    )
                    if _final_fn == adjudicator_answer_vllm:
                        _final_kwargs["is_final_round"] = True
                    result = _final_fn(**_final_kwargs)

            else:
                if can_reflect and args.vllm_url:

                    print(f"  [Round {reflection_round}] Controller: sufficient -> final answer")
                    _final_fn = (adjudicator_vllm_with_debiasing
                                 if args.mcq_debiasing and is_mcq
                                 else adjudicator_answer_vllm)
                    _final_kwargs = dict(
                        question=question, choices=choices,
                        evidence_texts=roi_evidence_texts,
                        vllm_url=args.vllm_url,
                        vllm_model_name=args.vllm_model or args.qwen_ckpt,
                        is_mcq=is_mcq, q_type=q_type,
                        memory_context=memory_context,
                    )
                    if _final_fn == adjudicator_answer_vllm:
                        _final_kwargs["is_final_round"] = True
                    result = _final_fn(**_final_kwargs)

            final_result = result
            break

        if final_result is None:
            has_evidence = (vl_patch_images if args.use_vl_adjudicator else roi_evidence_texts)
            if has_evidence:
                if args.use_vl_adjudicator:
                    final_result = vl_adjudicator_answer(
                        question=question,
                        choices=choices,
                        patch_images=vl_patch_images,
                        patch_infos=vl_patch_infos,
                        vl_processor=vl_processor,
                        vl_model=vl_model,
                        is_mcq=is_mcq,
                        q_type=q_type,
                        memory_context=memory_context,
                    )
                elif args.vllm_url:
                    _fb_fn = (adjudicator_vllm_with_debiasing
                              if args.mcq_debiasing and is_mcq
                              else adjudicator_answer_vllm)
                    _fb_kwargs = dict(
                        question=question, choices=choices,
                        evidence_texts=roi_evidence_texts,
                        vllm_url=args.vllm_url,
                        vllm_model_name=args.vllm_model or args.qwen_ckpt,
                        is_mcq=is_mcq, q_type=q_type,
                        memory_context=memory_context,
                    )
                    if _fb_fn == adjudicator_answer_vllm:
                        _fb_kwargs["is_final_round"] = True
                    final_result = _fb_fn(**_fb_kwargs)
                else:
                    _fallback_fn = (adjudicator_with_debiasing
                                    if args.mcq_debiasing and is_mcq
                                    else adjudicator_answer)
                    final_result = _fallback_fn(
                        question=question,
                        choices=choices,
                        evidence_texts=roi_evidence_texts,
                        tokenizer=tokenizer,
                        model=qwen_model,
                        is_mcq=is_mcq,
                        q_type=q_type,
                        memory_context=memory_context,
                    )
            else:
                final_result = {
                    "answer": "?",
                    "raw_response": "No valid evidence collected.",
                    "explanation": "SurpriseNav found no analyzable regions.",
                }

        result = final_result

        for _ck in ("sufficient", "missing_info", "next_action", "next_magnification", "next_query"):
            result.pop(_ck, None)
        n_evidence = len(vl_patch_images) if args.use_vl_adjudicator else len(roi_evidence_texts)
        print(f"  [Done] VLM calls: {vlm_calls}, evidence regions: {n_evidence}")

        verification_trace = None

        _has_enough_evidence = len(roi_evidence_texts) >= 2
        _parsed_structured = 0
        _hard_valid_structured = 0
        if args.structured_evidence:
            _parsed_structured = sum(1 for t in structured_traces if not t.get("parse_failed"))
            _hard_valid_structured = sum(1 for t in structured_traces if t.get("hard_valid"))
            _has_enough_evidence = _has_enough_evidence and _hard_valid_structured >= 2
        _last_sufficient = (
            controller_trace_rounds[-1].get("sufficient") if controller_trace_rounds else "no_trace"
        )
        _should_verify = (
            args.verification and args.vllm_url and is_mcq
            and _has_enough_evidence
            and controller_trace_rounds
            and controller_trace_rounds[-1].get("sufficient") == "yes"
        )

        if args.verification:
            print(f"  [Verify gate] should_verify={_should_verify} "
                  f"(mcq={is_mcq}, evidence={len(roi_evidence_texts)}, "
                  f"parsed_structured={_parsed_structured}, "
                  f"hard_valid_structured={_hard_valid_structured}, "
                  f"last_sufficient={_last_sufficient})")
        if _should_verify:
            verified = _verify_answer_vllm(
                question=question,
                choices=choices,
                evidence_texts=roi_evidence_texts,
                candidate_answer=result["answer"],
                vllm_url=args.vllm_url,
                vllm_model_name=args.vllm_model or args.qwen_ckpt,
                q_type=q_type,
            )

            if verified is not None:
                verified["original_answer"] = result["answer"]

                verification_trace = verified
                if verified.get("parse_failed"):
                    print(f"  [Verification AUDIT FAILED] {verified.get('validation_error','?')}")
                else:
                    print(f"  [Verification AUDIT] answer={result['answer']} "
                          f"keep={verified.get('keep_answer')} "
                          f"conflict={verified.get('has_conflict')} "
                          f"single_source={verified.get('single_source_risk')}")

        pred_answer = result["answer"]
        total_count += 1
        type_stats[q_type] += 1

        elapsed = time.time() - t0
        print(f"  -> Pred: {pred_answer[:50]} | {elapsed:.1f}s | [{q_type}]")

        surprise_vals = list(surprise_map_5x.values()) if surprise_map_5x else []
        mean_surprise = float(np.mean(surprise_vals)) if surprise_vals else 0.0
        high_surprise_count = sum(1 for v in surprise_vals if v > mean_surprise)

        case_result = {
            "long_id": wsi_id,
            "wsi_full_id": wsi_full_id,
            "question": question,
            "choices": choices,
            "pred_answer": pred_answer,
            "is_mcq": is_mcq,
            "q_type": q_type,
            "explanation": result["explanation"],
            "raw_response": result.get("raw_response", ""),
            "evidence": roi_evidence_texts,
            "planned_jump_targets": jump_targets_pool[:planner_top_k],
            "num_jump_targets_pool": len(jump_targets_pool),
            "actual_rounds": len(controller_trace_rounds),
            "vlm_calls": vlm_calls,
            "controller_trace": controller_trace_rounds if controller_trace_rounds else None,
            "time_seconds": round(elapsed, 1),
            "ablation": args.ablation,
            "query_navigation": args.query_navigation and plip_model is not None,
            "query_alpha": args.query_alpha if (args.query_navigation and plip_model is not None) else None,
            "mcq_debiasing": args.mcq_debiasing,
            "option_aware_vlm": args.option_aware_vlm,
            "qtype_adaptive_mag": args.qtype_adaptive_mag,
            "structured_evidence": args.structured_evidence,
            "structured_parsed_count": _parsed_structured,
            "structured_hard_valid_count": _hard_valid_structured,
            "verification_enabled": args.verification,
            "verification_trace": verification_trace,
            "structured_trace": structured_traces if structured_traces else None,

            "sample_is_macro_first": sample_is_macro_first,
            "macro_confirmation_triggered": macro_confirmation_triggered,
            "first_evidence_magnification": first_evidence_magnification,

            "total_5x_patches": len(feature_cache_5x),
            "total_20x_patches": len(feature_cache_20x),
            "total_20x_scanned": total_20x_scanned,
            "high_surprise_ratio": round(high_surprise_count / max(len(surprise_vals), 1), 4),
            "num_jump_targets": len(jump_targets_pool),
        }
        if args.save_ground_truth_for_debug:
            case_result["ground_truth"] = gt_answer

        with open(os.path.join(args.save_dir, f"{unique_id}.json"), "w", encoding="utf-8") as f:
            json.dump(case_result, f, indent=2, ensure_ascii=False)

        del feature_cache_5x, feature_cache_20x
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print(f"  SurpriseNav Evaluation Complete")
    print(f"  Total processed: {total_count}")
    print(f"  Skipped: {skipped_count}")
    for qt, t in type_stats.items():
        if t > 0:
            print(f"  {qt:12s}: {t}")
    print(f"{'='*60}")

    close_all_slides()

    summary = {
        "total": total_count,
        "skipped": skipped_count,
        "by_type": {qt: {"total": t} for qt, t in type_stats.items() if t > 0},
        "config": {
            "yaad_dim": args.yaad_dim,
            "yaad_hidden_dim": args.yaad_hidden_dim,
            "yaad_num_layers": args.yaad_num_layers,
            "query_navigation": args.query_navigation,
            "query_alpha": args.query_alpha,
            "top_k_jumps": args.top_k_jumps,
            "max_reflection_rounds": args.max_reflection_rounds,
            "max_vlm_calls": args.max_vlm_calls,
            "top_n_per_roi": args.top_n_per_roi,
            "multi_scale": args.multi_scale,
            "secondary_magnification": args.secondary_magnification,
            "mcq_debiasing": args.mcq_debiasing,
            "rag_enabled": bool(args.rag_index),
        },
    }
    with open(os.path.join(args.save_dir, "_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
