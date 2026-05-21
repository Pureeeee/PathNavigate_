import argparse
import os

def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)

_FEAT_5X_DIR = _env("FEATURE_DIR_5X")
_FEAT_20X_DIR = _env("FEATURE_DIR_20X")
_PLIP_FEAT_DIR = _env("PLIP_FEAT_DIR")
_SVS_DIR = _env("SVS_DIR")
_QUESTIONS_FILE = _env("QUESTIONS_FILE")
_QWEN_CKPT = _env("QWEN_CKPT")
_PATHO_R1_CKPT = _env("PATHO_R1_CKPT")
_SAVE_DIR = _env("SAVE_DIR", "outputs/surprisenav-vqa")
_PLIP_CKPT = _env("PLIP_CKPT")

def parse_args():
    parser = argparse.ArgumentParser(description="SurpriseNav: Training-Free WSI VQA")

    parser.add_argument("--qwen_ckpt", default=_QWEN_CKPT, type=str)
    parser.add_argument("--patho_r1_ckpt", default=_PATHO_R1_CKPT, type=str)

    parser.add_argument("--questions_file", default=_QUESTIONS_FILE, type=str)
    parser.add_argument("--feature_dir_5x", default=_FEAT_5X_DIR, type=str)
    parser.add_argument("--feature_dir_20x", default=_FEAT_20X_DIR, type=str)
    parser.add_argument("--svs_dir", default=_SVS_DIR, type=str,
                        help="Directory containing .svs WSI files")
    parser.add_argument("--save_dir", default=_SAVE_DIR, type=str)

    parser.add_argument("--use_plip_features", action="store_true", default=False,
                        help="Use PLIP 512-dim features instead of CONCH 768-dim for navigation. "
                             "Aligns navigation space with PLIP re-ranking (same 512-dim space)")
    parser.add_argument("--plip_feat_dir", type=str, default=_PLIP_FEAT_DIR,
                        help="Directory of PLIP pre-extracted features (.npy format)")

    parser.add_argument("--yaad_dim", type=int, default=768, help="Feature dimension (768 for CONCH, 512 for PLIP)")
    parser.add_argument("--yaad_hidden_dim", type=int, default=0,
                        help="Hidden width of Yaad MLP (0 = same as yaad_dim)")
    parser.add_argument("--yaad_num_layers", type=int, default=2,
                        help="Number of linear layers in Yaad MLP (default: 2)")
    parser.add_argument("--yaad_lr", type=float, default=0.05, help="Yaad memory learning rate")
    parser.add_argument("--yaad_huber_delta", type=float, default=1.0)
    parser.add_argument("--yaad_forget_alpha", type=float, default=0.999)
    parser.add_argument("--yaad_warmup", type=int, default=100)
    parser.add_argument("--yaad_threshold_scale", type=float, default=1.0)
    parser.add_argument("--yaad_reverse_update", action="store_true",
                        help="Update memory only on low-surprise patches")

    parser.add_argument("--top_k_jumps", type=int, default=5)
    parser.add_argument("--nms_window", type=int, default=3)
    parser.add_argument("--max_vlm_calls", type=int, default=5,
                        help="Max VLM calls across all rounds")
    parser.add_argument("--vlm_magnification", type=int, default=20,
                        help="Primary magnification for VLM patch extraction (5/10/20/40)")
    parser.add_argument("--qtype_adaptive_mag", action="store_true", default=False,
                        help="Use question-type adaptive VLM magnification")
    parser.add_argument("--structured_evidence", action="store_true", default=False,
                        help="Use structured JSON evidence from Patho-R1")
    parser.add_argument("--structured_max_tokens", type=int, default=288,
                        help="Max new tokens for structured Patho-R1 JSON output")
    parser.add_argument("--verification", action="store_true", default=False,
                        help="Enable a lightweight verification pass before final answer")
    parser.add_argument("--multi_scale", action="store_true", default=False,
                        help="Enable multi-scale VLM: first N calls at primary mag, "
                             "remaining calls at secondary mag (10x for macro context)")
    parser.add_argument("--secondary_magnification", type=int, default=10,
                        help="Secondary magnification for multi-scale mode")
    parser.add_argument("--local_20x_radius", type=int, default=4096,
                        help="20x patch search radius in level-0 pixels (4096 = one 5x patch)")
    parser.add_argument("--min_jump_distance", type=int, default=0,
                        help="Min pixel distance between jump targets (0=disable, 20480=5 patches)")
    parser.add_argument("--top_n_per_roi", type=int, default=2,
                        help="Top-N 20x patches per jump target to describe with VLM")
    parser.add_argument("--max_reflection_rounds", type=int, default=1,
                        help="Max reflection rounds (1=one-shot, 2=one reflection)")

    parser.add_argument("--dataset_name", type=str, default="wsi_vqa")
    parser.add_argument("--save_heatmap", action="store_true")
    parser.add_argument("--cache_surprise", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--vlm_device", type=str, default="cuda:0",
                        help="GPU for Patho-R1 VLM model")
    parser.add_argument("--qwen_device", type=str, default="cuda:5",
                        help="GPU for Qwen3 adjudicator model")
    parser.add_argument("--max_samples", type=int, default=0,
                        help="Max samples to process (0=all, useful for debugging)")

    parser.add_argument("--query_navigation", action="store_true", default=False,
                        help="Enable PLIP-based query-conditioned navigation")
    parser.add_argument("--no_query_navigation", dest="query_navigation", action="store_false",
                        help="Disable query-conditioned navigation (surprise-only)")
    parser.add_argument("--plip_ckpt", type=str, default=_PLIP_CKPT,
                        help="Path to PLIP checkpoint")
    parser.add_argument("--plip_device", type=str, default=None,
                        help="GPU for PLIP model (default: same as vlm_device)")
    parser.add_argument("--query_alpha", type=float, default=0.5,
                        help="Weight: alpha*relevance + (1-alpha)*surprise (0=surprise-only, 1=relevance-only)")
    parser.add_argument("--surprise_candidates", type=int, default=30,
                        help="Number of surprise candidates for PLIP re-ranking")

    parser.add_argument("--ablation", type=str, default="none",
                        choices=["none", "random_roi", "grid_roi", "no_20x"],
                        help="Ablation mode: none=full SurpriseNav, random_roi=random jump targets, "
                             "grid_roi=uniform grid targets, no_20x=skip 20x deep scan")

    parser.add_argument("--rag_index", type=str, default=_env("RAG_INDEX"),
                        help="Path to RAG index .pt file (set empty string to disable)")
    parser.add_argument("--rag_top_k", type=int, default=3,
                        help="Number of similar training WSIs to retrieve")

    parser.add_argument("--textbook_rag", action="store_true", default=False,
                        help="Inject pathology diagnostic criteria from knowledge base")

    parser.add_argument("--no_nav_summary", action="store_true", default=False,
                        help="Disable the Yaad-derived Navigation Summary fed to "
                             "the adjudicator.")

    parser.add_argument("--readout_meanpool", action="store_true", default=False,
                        help="Replace the Yaad-derived readout with a mean-pool CONCH "
                             "descriptor computed from slide-level feature statistics.")

    parser.add_argument("--pool_source", type=str, default="surprise",
                        choices=["surprise", "plip"],
                        help="Source of the candidate pool: 'surprise' (default, Yaad "
                             "NMS) or 'plip' (rank all 5x patches by PLIP relevance, "
                             "take top pool_k). Random pool already supported via "
                             "--ablation random_roi.")

    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility (fixes Yaad init + navigation)")

    parser.add_argument("--num_shards", type=int, default=1,
                        help="Total number of parallel shards (1=no sharding)")
    parser.add_argument("--shard_id", type=int, default=0,
                        help="This shard's ID (0-indexed, must be < num_shards)")

    parser.add_argument("--mcq_debiasing", action="store_true", default=True,
                        help="MCQ option shuffle de-biasing")
    parser.add_argument("--no_mcq_debiasing", dest="mcq_debiasing", action="store_false",
                        help="Disable MCQ option shuffle de-bias")
    parser.add_argument("--option_aware_vlm", action="store_true", default=True,
                        help="Inject MCQ choices into VLM prompts for option-aware descriptions")
    parser.add_argument("--no_option_aware_vlm", dest="option_aware_vlm", action="store_false",
                        help="Disable option-aware VLM description")

    parser.add_argument("--use_vl_adjudicator", action="store_true", default=False,
                        help="Use Qwen3-VL as unified VLM+Adjudicator (replaces Patho-R1+Qwen)")
    parser.add_argument("--vl_ckpt", type=str, default=_env("VL_CKPT"),
                        help="Path to Qwen3-VL checkpoint")
    parser.add_argument("--vl_device", type=str, default="cuda:1",
                        help="GPU for Qwen3-VL model")

    parser.add_argument("--vllm_url", type=str, default="",
                        help="vLLM server URL (e.g. http://localhost:8100/v1). "
                             "If set, adjudicator calls vLLM API instead of local model.generate()")
    parser.add_argument("--vllm_model", type=str, default="",
                        help="Model name for vLLM API (default: same as --qwen_ckpt)")

    parser.add_argument("--save_ground_truth_for_debug", action="store_true", default=False,
                        help="Write ground-truth answers into per-case JSON files for local debugging.")

    return parser.parse_args()
