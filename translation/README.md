# ASL-to-English Translation (Stage B)

Continuous sign language translation from MediaPipe landmarks to English sentences using MultiStreamSemanticEncoder + LoRA FLAN-T5-base, deployed on Modal.

## Results

| Metric | Score |
|--------|-------|
| BLEU-4 | 1.94 (full val) / 2.45 (best during training) |
| ROUGE-L | 13.91 |
| chrF | 18.71 |

See [EVALUATION.md](EVALUATION.md) for the full evaluation report.

## Architecture

```
How2Sign landmarks [T, 543, 3]
  → engineer_features_multistream() → 8 tensors (body/face/hand pos+vel)
  → MultiStreamSemanticEncoder (4 layers, 8 heads) → [B, T/4, 512]
  → AttentionPooling (32 queries) → [B, 32, 512]
  → LatentToT5Adapter → [B, 32, 768]
  → FLAN-T5-base (LoRA r=16 on q,v) → English sentence
```

- **Total params**: 260M (12.7M trainable — 4.9%)
- **Externally pre-trained**: FLAN-T5-base (Google) — frozen, adapted via LoRA
- **Trained from scratch**: Encoder, pooling, adapter, LoRA weights, CTC head

## Files

| File | Description |
|------|-------------|
| `modal_translation_app.py` | Self-contained Modal app (model, data, training, evaluation) |
| `llm_judge.py` | Local Ollama LLM-as-judge scorer for `eval_results.json` translation outputs |
| `EVALUATION.md` | Full evaluation report with metrics, analysis, recommendations |
| `training_history.jsonl` | Authoritative per-epoch training metrics copied from the Modal checkpoint volume |
| `translation_train.log` | Raw training log for epochs 2-33; the local Modal log stream was interrupted after epoch 33 |
| `translation_resume.log` | Training log (resumed epochs 34-39) |
| `translation_eval.log` | Evaluation log (1,739 val samples, all predictions) |
| `translation_smoke_test.log` | Smoke test log (10 samples, 2 epochs) |

## Usage

```bash
# Smoke test (10 samples, 2 epochs)
modal run translation/modal_translation_app.py --mode smoke_test

# Full training (40 epochs, ~9 hours on A10G)
modal run translation/modal_translation_app.py --mode train

# Evaluation on full validation set
modal run translation/modal_translation_app.py --mode evaluate
```

## LLM-as-Judge Evaluation

The LLM judge scores each `(reference, prediction)` pair from `translation/eval_results.json` on semantic overlap, tone/register, and fluency. It runs locally through Ollama and writes resumable results to `translation/llm_judge_results.json`.

```bash
# Start Ollama if it is not already running
ollama serve

# Pull the required judge model
ollama pull llama3.1:8b

# Check input/output/model readiness without scoring
python3 translation/llm_judge.py --preflight

# Smoke test 25 examples
python3 translation/llm_judge.py --limit 25 --output translation/llm_judge_smoke_results.json --overwrite

# Full 1,739-sample judge run; rerun this same command if interrupted
python3 translation/llm_judge.py --resume
```

The full run is accepted when `meta.n_scored` is `1739`, `meta.n_errors` is `0` or explicitly reviewed, and aggregate means are present in `translation/llm_judge_results.json`.

## Checkpoints (Modal Volume: `asl-translation-checkpoints`)

- `translation/best.pt` — Best checkpoint (epoch 24)
- `translation/latest.pt` — Final checkpoint (epoch 39)
- `translation/history.jsonl` — Per-epoch training metrics
- `translation/eval_results.json` — Full evaluation results with all predictions

## Data

- **Source**: `bdanko/how2sign-landmarks-front-raw-parquet` (HuggingFace)
- **Train**: 31,046 utterances | **Val**: 1,739 utterances
- **Input**: MediaPipe Holistic landmarks [T, 543, 3] per utterance
- **Target**: English sentence per utterance
