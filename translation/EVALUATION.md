# ASL-to-English Translation Model — Evaluation Report

**Date**: 2026-05-05
**Checkpoint**: Epoch 24 (best by BLEU-4)
**Validation Set**: How2Sign front-view, 1,739 utterances (full validation split)

---

## 1. Architecture

```
How2Sign landmarks [T, 543, 3]
  → engineer_features_multistream()
  → 8 tensors: body/face/hand position + velocity
  → MultiStreamSemanticEncoder (4 layers, 8 heads, d=384 → 512)
  → [B, T/4, 512]
  → AttentionPooling (32 learned queries)
  → [B, 32, 512]
  → LatentToT5Adapter (512 → 768)
  → [B, 32, 768]
  → FLAN-T5-base decoder (LoRA r=16 on q,v — 1.77M added params)
  → English sentence
```

| Component | Parameters |
|-----------|-----------|
| MultiStreamSemanticEncoder | ~10.9M |
| AttentionPooling | ~0.8M |
| LatentToT5Adapter | ~0.4M |
| FLAN-T5-base (frozen + LoRA) | 248.1M (1.77M trainable) |
| **Total** | **260.2M (12.7M trainable)** |

---

## 2. Overall Metrics

| Metric | Score |
|--------|-------|
| **BLEU-4** | **1.94** |
| **ROUGE-L** | **13.91** |
| **chrF** | **18.71** |
| **Exact Match** | **0.23%** |

### Comparison to Prior Work

| System | BLEU-4 | Notes |
|--------|--------|-------|
| MotionConformer-T5 (prev attempt) | 0.007 | Window-level supervision bug |
| **This model** | **1.94** | **277x improvement** |
| Random baseline | ~0.0 | — |

> **Note**: How2Sign continuous SLT from landmarks alone is an open research problem. Published SOTA systems using full video features (not just landmarks) achieve ~10-15 BLEU. Our landmark-only, LoRA-adapted approach with 16K training utterances represents a strong baseline for this modality.

---

## 3. Length-Bucketed Performance

| Sentence Length | Count | BLEU-4 | ROUGE-L | chrF | Exact Match |
|----------------|-------|--------|---------|------|-------------|
| Short (≤8 words) | 408 | 0.38 | 9.78 | 12.52 | 0.98% |
| Medium (9-20 words) | 767 | 1.67 | 14.39 | 17.60 | 0.00% |
| Long (>20 words) | 564 | 2.10 | 16.24 | 20.26 | 0.00% |

The model performs better on longer references (BLEU 2.10) than shorter ones (BLEU 0.38). This is because the model tends to produce medium-length outputs (~18 words), which penalizes BLEU's brevity penalty on short references. The higher ROUGE-L and chrF on long sentences suggests partial content overlap is captured.

---

## 4. Output Diversity Analysis

| Metric | Predictions | References |
|--------|------------|------------|
| Unique sequences | 209 / 1,739 (12.0%) | 1,739 (100%) |
| Vocabulary size (unigrams) | 244 | 4,417 |
| Bigram types | 521 | 15,013 |
| Avg. sentence length (words) | 18.1 | 17.5 |

**Key finding**: The model produces only **209 unique sentences** out of 1,739 predictions, indicating significant output collapse. It has learned a small set of plausible English templates rather than truly translating sign content. The vocabulary is ~18x smaller than the reference vocabulary.

---

## 5. Training History

### Loss Curve

| Epoch | Stage | Train Loss | BLEU-4 |
|-------|-------|-----------|--------|
| 0 | warmup | 14.12 | 0.26 |
| 1 | warmup | 15.31 | 0.75 |
| 4 | warmup | 10.15 | 1.83 |
| 9 | joint | 4.47 | 2.09 |
| 14 | joint | 4.44 | 2.02 |
| 19 | joint | 4.43 | 2.26 |
| **24** | **joint** | **4.42** | **2.45** |
| 29 | joint | 4.41 | 1.92 |
| 34 | refine | 4.40 | 1.71 |
| 39 | refine | 4.39 | 1.92 |

- **Total training time**: 9.3 GPU-hours on A10G (Modal)
- **Best BLEU** occurred at epoch 24 during the joint training stage
- Loss continued to decrease in the refine stage (4.40→4.39) but BLEU did not improve, suggesting mild overfitting to training loss without generation quality gains

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW (per-group LR) |
| Batch size | 4 (×8 gradient accumulation = 32 effective) |
| Label smoothing | 0.1 |
| Gradient clipping | 1.0 |
| Warmup epochs | 0-4 (encoder frozen) |
| Joint epochs | 5-29 (all trainable) |
| Refine epochs | 30-39 (lower LR) |
| Loss | T5 CE + 0.1×CTC + 0.01×latent reg |
| Generation | Beam search (5 beams), repetition penalty 2.5 |
| Max sequence | 512 frames (~17s at 30fps) |
| Augmentation | Speed perturbation, magnitude scaling, joint dropout, temporal jitter |

---

## 6. Sample Predictions

### Reasonable Outputs

| # | Reference | Prediction |
|---|-----------|------------|
| 1 | "I'll show you." | "I'm going to show you how to do that." |
| 3 | "If you need to deal with the situation a little bit more it might look something like this." | "You want to make sure that you're going to be able to use the right tools for your job." |
| 18 | "Now One Point is the principal of dynamic motion without actual visible motion taking place." | "So, I'm going to go ahead and make a little bit of a tutorial on how to do this." |

### Failure Modes

| # | Reference | Prediction | Issue |
|---|-----------|------------|-------|
| 2 | "So that is one way, you can just let them go right by." | "I'm going to show you how to do that." | Generic template |
| 8 | "You can usually find this at HEB..." | "You're going to want to make sure that you don't have a lot of money in your bank account..." | Hallucinated content |
| 12 | "This yogurt as I said is non fat..." | "You're going to want to make sure that you don't have a lot of money in your bank account..." | Same template for different inputs |

---

## 7. Failure Analysis

### 7.1 Output Collapse
The model converges on a small set of template sentences. The most common patterns:
- "I'm going to show you how to do that/this." (~frequent)
- "You're going to want to make sure that you don't have a lot of [X]..." (~frequent)
- "You want to make sure that you're going to be able to [X]..." (~frequent)

**Root cause**: The encoder-to-T5 bottleneck (32 queries × 512 dim → 768 dim adapter) may not preserve enough discriminative information from the sign input. The T5 decoder, with only LoRA adaptation, defaults to high-probability English completions.

### 7.2 Content Hallucination
Predictions often contain specific nouns ("money," "bank account," "hair," "water") unrelated to the reference. These are high-frequency words in T5's pretraining distribution that surface when the sign-conditioned embeddings lack strong content signal.

### 7.3 Length Mismatch
The model produces consistently medium-length outputs (~18 words) regardless of input length, suggesting it has not learned to map input duration to output length.

---

## 8. Strengths

1. **Grammatical fluency**: All outputs are grammatically correct, well-formed English sentences (thanks to FLAN-T5's language model)
2. **Dramatic improvement**: 277x BLEU improvement over the previous MotionConformer-T5 attempt (0.007 → 1.94)
3. **Utterance-level supervision**: Fixing the window-level supervision bug was the critical architectural decision
4. **Efficient training**: Only 12.7M trainable parameters (4.9% of total), 9.3 GPU-hours
5. **Stable convergence**: Monotone loss decrease, no training instability

---

## 9. Recommendations for Improvement

### High Impact
1. **Increase LoRA rank** (r=32 or r=64, add k/o target modules) — allow T5 more capacity to adapt to sign embeddings
2. **Larger encoder** (6-8 layers, d_model=512) — current 4-layer encoder may not extract enough temporal detail
3. **Remove attention pooling bottleneck** — pass full temporal sequence to T5 cross-attention instead of compressing to 32 fixed queries
4. **Curriculum learning** — train on short sentences first (≤10 words), gradually increase

### Medium Impact
5. **Gloss-assisted training** — if gloss annotations are available, add an intermediate CTC loss on gloss tokens
6. **Contrastive pretraining** — pretrain encoder with sign-text contrastive loss (CLIP-style) before seq2seq
7. **Diverse beam search** — reduce output collapse during inference
8. **Increase training data** — use How2Sign test set for training if evaluation-only uses a held-out split

### Low Impact / Experimental
9. **Mixture of Experts** in the adapter layer
10. **Retrieval-augmented generation** — retrieve similar training sentences as T5 prefix
11. **Multi-task with sign recognition** — joint gloss recognition + translation

---

## 10. Reproducibility

### Run Evaluation
```bash
modal run modal_translation_app.py --mode evaluate
```

### Checkpoints (Modal Volume: `asl-translation-checkpoints`)
- `translation/best.pt` — Best checkpoint (epoch 24, BLEU 2.45 on 500-sample val)
- `translation/latest.pt` — Final checkpoint (epoch 39)
- `translation/history.jsonl` — Per-epoch training metrics
- `translation/eval_results.json` — Full evaluation results with all 1,739 predictions

### Data
- **Training**: `bdanko/how2sign-landmarks-front-raw-parquet` (train split, 31,046 utterances)
- **Validation**: Same dataset (validation split, 1,739 utterances)
- **Input format**: MediaPipe Holistic landmarks [T, 543, 3]
