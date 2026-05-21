# SurpriseNav Paper Code

This repository contains the cleaned inference code for SurpriseNav. The entry point is `main.py`, with implementation modules under `surprisenav/`.

## Installation

Install the Python requirements in an environment with CUDA-compatible PyTorch:

```bash
pip install -r requirements.txt
```

OpenSlide also requires the system OpenSlide library, for example `apt-get install openslide-tools` on Debian/Ubuntu.

## WSI-VQA Inference

Start an OpenAI-compatible vLLM server for the Qwen adjudicator:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model "$QWEN35_MODEL" \
  --port 8100 \
  --trust-remote-code
```

Then set local paths and run:

```bash
export QWEN_CKPT=/path/to/qwen-checkpoint
export PATHO_R1_CKPT=/path/to/patho-r1-checkpoint
export QWEN35_MODEL=/path/or/name/served/by/vllm
export QUESTIONS_FILE=/path/to/WsiVQA_test.json
export FEATURE_DIR_5X=/path/to/5x/features
export FEATURE_DIR_20X=/path/to/20x/features
export SVS_DIR=/path/to/wsi/slides
export PLIP_CKPT=/path/to/plip-checkpoint
export RAG_INDEX=/path/to/train_rag_index.pt
bash scripts/run_wsivqa.sh
```

`RAG_INDEX` is optional. If it is unset, the run skips the case archive.

## Evaluation

The WSI-VQA evaluator performs the paper evaluation join. If prediction files do not contain ground-truth answers, pass the question file:

```bash
python scripts/eval_wsivqa.py \
  --questions_file "$QUESTIONS_FILE" \
  "outputs/surprisenav-wsivqa=SurpriseNav"
```

The evaluator reports Total accuracy, MCQ SeqMatcher accuracy, open-ended substring accuracy, and open-ended BLEU/ROUGE.

BCNB uses letter-exact evaluation:

```bash
python scripts/eval_bcnb.py \
  --questions_file "$BCNB_QUESTIONS_FILE" \
  --merge \
  --label "SurpriseNav BCNB" \
  RESULTS_DIRS...
```

## Case Archive

The optional archive is rebuilt from training annotations and low-magnification features:

```bash
python build_rag_index.py \
  --train_json /path/to/WsiVQA_train.json \
  --feature_dir /path/to/5x/features \
  --output_path /path/to/train_rag_index.pt
```

## Layout

`surprisenav/config.py` defines CLI arguments.
`surprisenav/perception.py` wraps Patho-R1 patch description.
`surprisenav/adjudication.py` contains Qwen/vLLM answer synthesis.
`surprisenav/navigation.py` contains PLIP reranking and memory readout.
`surprisenav/retrieval.py` contains optional archive retrieval.
`surprisenav/evidence.py` handles structured evidence parsing.
`surprisenav/pipeline.py` runs inference.
