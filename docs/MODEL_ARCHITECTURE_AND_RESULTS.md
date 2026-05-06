# Motion-Conformer-T5 Modal Experiment: Architecture And Results

## Executive Summary

This document describes the Motion-Conformer-T5 model implemented and tested in `modal_motion_conformer_app.py`, including the distinction between Phase 1-style motion representation learning and Phase 2 text generation.

The main finding is split:

- The model is strong at Phase 1-style motion reconstruction on held-out validation windows.
- The model is poor at Phase 2 text generation when evaluated with BLEU, chrF, ROUGE-L, and exact match.

The most likely reason for this split is the training target design. The Modal experiment trains 60-frame windows while assigning the full utterance sentence to every window. This can support reconstruction learning, but it is weak supervision for translation because most windows do not contain enough information to produce the entire sentence.

## Data Source

The Modal app used the Hugging Face dataset:

```text
bdanko/how2sign-landmarks-front-raw-parquet
```

The dataset preflight validated the required schema:

```json
{
  "repo_id": "bdanko/how2sign-landmarks-front-raw-parquet",
  "split": "train",
  "keys": ["features", "sentence", "shape", "video_id"],
  "shape": [365, 543, 3],
  "sentence_preview": "And I call them decorative elements because basically all they're meant to do is to enrich and color the page.",
  "video_id": "--7E2sU6zP4_10-5-rgb_front"
}
```

Held-out split checks also succeeded:

```text
validation: features, sentence, shape, video_id
test: features, sentence, shape, video_id
```

The RGB dataset `bdanko/how2sign-rgb-front-clips` was documented as an optional visual/debug source, but it was not used for training or evaluation.

## Input Feature Contract

The model consumes raw MediaPipe Holistic landmarks:

```text
Raw input: [T, 543, 3]
```

The raw 543 landmarks are arranged as:

```text
33 pose landmarks
468 face landmarks
21 left-hand landmarks
21 right-hand landmarks
```

Feature engineering reduces these to 90 selected keypoints:

```text
33 pose
15 selected face landmarks
21 left hand
21 right hand
```

For each frame, the app computes:

```text
position: x, y, z
delta/velocity: dx, dy, dz
```

This gives:

```text
90 keypoints * 6 values = 540 features per frame
```

Training and evaluation use sliding temporal windows:

```text
window_size = 60 frames
stride = 30 frames
model input = [B, 60, 540]
```

## Model Architecture Tested

The tested architecture is a Motion-Conformer-T5 model:

```text
[B, 60, 540]
  -> modality-aware projection
  -> 4-layer Motion-Conformer encoder
  -> latent Z [B, 15, 512]
  -> reconstruction decoder
  -> CTC head
  -> T5 adapter
  -> frozen FLAN-T5-small
```

### Modality-Aware Projection

The 540 features are split into body regions rather than treated as one flat vector.

Per-frame feature splits:

```text
positions: first 270 dims
deltas:    final 270 dims
```

Part-specific streams:

```text
pose:       33 landmarks * 3 coords * 2 position/delta = 198 dims -> 64 dims
face:       15 landmarks * 3 coords * 2 position/delta = 90 dims  -> 32 dims
left hand:  21 landmarks * 3 coords * 2 position/delta = 126 dims -> 96 dims
right hand: 21 landmarks * 3 coords * 2 position/delta = 126 dims -> 96 dims
```

The projected streams are concatenated:

```text
64 + 32 + 96 + 96 = 288 dims
```

Then fused to:

```text
[B, 60, 256]
```

### Motion-Conformer Encoder

The encoder uses four Conformer-style blocks. Each block contains:

- Feed-forward network
- Multi-head self-attention
- Depthwise temporal convolution
- Second feed-forward network
- Layer normalization and residual paths

Configuration:

```text
d_model = 256
num_layers = 4
num_heads = 4
ff_dim = 1024
conv_kernel = 7
dropout = 0.1
```

Temporal compression uses two strided Conv1d layers:

```text
[B, 60, 256] -> [B, 30, 384] -> [B, 15, 512]
```

Final latent:

```text
Z = [B, 15, 512]
```

### Reconstruction Decoder

The reconstruction decoder maps the latent representation back to the original feature space:

```text
Z [B, 15, 512]
  -> ConvTranspose1d 512 -> 384, stride 2
  -> ConvTranspose1d 384 -> 256, stride 2
  -> Conv1d 256 -> 540
  -> reconstruction [B, 60, 540]
```

This path is the Phase 1-style validation path.

### CTC Head

The CTC head maps each latent timestep to FLAN-T5 token vocabulary logits plus a blank token:

```text
Z [B, 15, 512] -> Linear(512, vocab_size + 1)
```

The CTC target uses FLAN-T5 tokenizer subword IDs because the dataset provides English sentences, not gloss labels.

This is a rough alignment pressure, not a true gloss-level CTC objective.

### T5 Adapter And FLAN-T5

The T5 adapter maps the motion latent into T5-compatible embedding space:

```text
Linear(512 -> 512)
LayerNorm
GELU
Linear(512 -> 512)
learned positional embedding
```

The adapted latent is passed to FLAN-T5-small through `inputs_embeds`.

FLAN-T5-small was frozen in this experiment:

```text
t5_model = google/flan-t5-small
freeze_t5 = true
```

Only the motion encoder, reconstruction decoder, CTC head, and adapter were trained.

## Phase 1 Versus Phase 2

### Phase 1-Style Objective In This Experiment

In a clean two-stage design, Phase 1 means learning a motion representation through reconstruction or denoising before translation training.

In this Modal experiment, we did not run a separate Phase 1-only training job. Instead, Phase 1-style behavior was trained as one component of a combined loss:

```text
reconstruction_loss = MSE(reconstruction, input_features)
```

The reconstruction path is:

```text
features [B, 60, 540]
  -> Motion-Conformer encoder
  -> Z [B, 15, 512]
  -> reconstruction decoder
  -> reconstructed features [B, 60, 540]
```

Phase 1-style validation measures whether the latent `Z` preserves enough motion information to reconstruct the input.

### Phase 2 Objective In This Experiment

Phase 2 means mapping motion latent features to English text.

The translation path is:

```text
features [B, 60, 540]
  -> Motion-Conformer encoder
  -> Z [B, 15, 512]
  -> T5 adapter
  -> FLAN-T5-small
  -> generated English text
```

Training uses token-level cross-entropy from FLAN-T5:

```text
translation_loss = T5 sequence-to-sequence loss
```

Evaluation uses generation-based text metrics:

```text
BLEU
chrF
ROUGE-L
Exact match
Validation loss
```

Important limitation: Phase 2 was trained on 60-frame windows while assigning the full utterance sentence target to every window. This is not a true utterance-level translation setup.

## Combined Training Loss

The full Modal training job optimized:

```text
loss = reconstruction_weight * reconstruction_loss
     + translation_loss
     + ctc_weight * ctc_loss
```

Configuration:

```json
{
  "batch_size": 4,
  "lr": 0.0001,
  "ctc_weight": 0.1,
  "reconstruction_weight": 1.0,
  "window_size": 60,
  "stride": 30,
  "t5_model": "google/flan-t5-small",
  "freeze_t5": true,
  "seed": 15179996
}
```

## Modal Commands Used

Data source check:

```bash
modal run modal_motion_conformer_app.py::check_data_source
```

Minimal GPU smoke run:

```bash
modal run modal_motion_conformer_app.py::train_smoke --max-samples 1 --batch-size 1
```

Configured smoke run:

```bash
modal run modal_motion_conformer_app.py::train_smoke --max-samples 100 --batch-size 4
```

Full training run:

```bash
modal run --detach modal_motion_conformer_app.py::train_full --batch-size 4
```

Phase 2 validation/generation evaluation:

```bash
modal run modal_motion_conformer_app.py::evaluate_full_checkpoint --split validation --batch-size 4 --max-samples 0
```

Phase 1-style reconstruction validation:

```bash
modal run modal_motion_conformer_app.py::evaluate_phase1_checkpoint --split validation --batch-size 16 --max-samples 0
```

## Training Results

### Minimal Smoke Run

Command:

```bash
modal run modal_motion_conformer_app.py::train_smoke --max-samples 1 --batch-size 1
```

Because one source utterance can produce multiple 60-frame windows, `max_samples=1` produced 11 training windows.

Results:

```json
{
  "device": "cuda",
  "samples": 11,
  "steps": 11,
  "reconstruction_loss": 0.47682272270321846,
  "ctc_loss": 111.03334903717041,
  "translation_loss": 75.86083841323853,
  "avg_reconstruction_loss": 0.04334752024574713,
  "avg_ctc_loss": 10.093940821560947,
  "avg_translation_loss": 6.896439855748957,
  "checkpoint": "/checkpoints/motion_conformer_t5_smoke/motion_conformer_t5_smoke.pt"
}
```

### 100-Sample Smoke Run

Command:

```bash
modal run modal_motion_conformer_app.py::train_smoke --max-samples 100 --batch-size 4
```

Run URL:

```text
https://modal.com/apps/mrcyrilgoud/main/ap-Rfy6pCCL6QQipdLNPBu4Ug
```

Results:

```json
{
  "device": "cuda",
  "samples": 576,
  "steps": 144,
  "reconstruction_loss": 2.441033056238666,
  "ctc_loss": 1273.184977054596,
  "translation_loss": 717.1772623062134,
  "avg_reconstruction_loss": 0.016951618446101848,
  "avg_ctc_loss": 8.841562340656916,
  "avg_translation_loss": 4.98039765490426,
  "checkpoint": "/checkpoints/motion_conformer_t5_smoke/motion_conformer_t5_smoke.pt"
}
```

### Full Training Run

Command:

```bash
modal run --detach modal_motion_conformer_app.py::train_full --batch-size 4
```

Run URL:

```text
https://modal.com/apps/mrcyrilgoud/main/ap-a9L3QAeuKtbITR7Lg1kUzA
```

Results:

```json
{
  "device": "cuda",
  "samples": 130312,
  "steps": 32578,
  "reconstruction_loss": 375.73004201150616,
  "ctc_loss": 208297.23644065857,
  "translation_loss": 128043.14262712002,
  "avg_reconstruction_loss": 0.011533244582586597,
  "avg_ctc_loss": 6.393800615159266,
  "avg_translation_loss": 3.930356149153417,
  "checkpoint": "/checkpoints/motion_conformer_t5_full/motion_conformer_t5_smoke.pt"
}
```

## Phase 1-Style Validation Results

This evaluation loads the completed full-run checkpoint and evaluates only the encoder plus reconstruction decoder on the held-out validation split.

Command:

```bash
modal run modal_motion_conformer_app.py::evaluate_phase1_checkpoint --split validation --batch-size 16 --max-samples 0
```

Run URL:

```text
https://modal.com/apps/mrcyrilgoud/main/ap-je4fVc9yEBQZkwjWLASg53
```

Results:

```json
{
  "device": "cuda",
  "split": "validation",
  "checkpoint": "/checkpoints/motion_conformer_t5_full/motion_conformer_t5_smoke.pt",
  "windows": 7088,
  "steps": 443,
  "val/full_recon_loss": 0.012569112721599352,
  "val/masked_pos_loss": 0.019928489563646085,
  "val/masked_vel_loss": 0.0052097358447331385,
  "val/latent_smooth_loss": 0.0383371860125264,
  "val/latent_reg_loss": 0.714032563748801,
  "z_mean": -0.006582529180205972,
  "z_std": 0.8449788144128887
}
```

Saved output:

```text
/checkpoints/motion_conformer_t5_full/eval_phase1_validation/metrics.json
```

Interpretation:

- Full reconstruction validation loss is low.
- Velocity reconstruction is especially strong.
- Latent distribution is not collapsed: `z_std = 0.845`.
- Latent regularization is high compared with the baseline Phase 1 metrics supplied separately.

## Phase 2 Text Generation Validation Results

This evaluation loads the completed full-run checkpoint and evaluates generated text on the held-out validation split.

Command:

```bash
modal run modal_motion_conformer_app.py::evaluate_full_checkpoint --split validation --batch-size 4 --max-samples 0
```

Run URL:

```text
https://modal.com/apps/mrcyrilgoud/main/ap-eAhzxhJr5jiTG2vzAPAtog
```

Results:

```json
{
  "device": "cuda",
  "split": "validation",
  "checkpoint": "/checkpoints/motion_conformer_t5_full/motion_conformer_t5_smoke.pt",
  "windows": 7088,
  "steps": 1772,
  "validation_loss": 3.7762172906985403,
  "bleu": 0.007387763828023424,
  "chrf": 0.17312468600138753,
  "rouge_l": 0.0,
  "exact_match": 0.0
}
```

Saved outputs:

```text
/checkpoints/motion_conformer_t5_full/eval_validation/metrics.json
/checkpoints/motion_conformer_t5_full/eval_validation/predictions.jsonl
```

Interpretation:

- Text generation is effectively failing.
- BLEU, ROUGE-L, and exact match are near zero.
- chrF is also extremely low.
- Validation loss is not enough by itself to indicate useful generation.

## Small Validation Subset Results

### Phase 2 Generation Subset

Command:

```bash
modal run modal_motion_conformer_app.py::evaluate_full_checkpoint --split validation --batch-size 4 --max-samples 20
```

Results:

```json
{
  "windows": 92,
  "steps": 23,
  "validation_loss": 3.9998340710349707,
  "bleu": 0.03904868424508724,
  "chrf": 0.17592167661006577,
  "rouge_l": 0.0,
  "exact_match": 0.0
}
```

### Phase 1 Reconstruction Subset

Command:

```bash
modal run modal_motion_conformer_app.py::evaluate_phase1_checkpoint --split validation --batch-size 16 --max-samples 20
```

Results:

```json
{
  "windows": 80,
  "steps": 5,
  "val/full_recon_loss": 0.012480772286653518,
  "val/masked_pos_loss": 0.016659131459891795,
  "val/masked_vel_loss": 0.008302412927150726,
  "val/latent_smooth_loss": 0.0383375309407711,
  "val/latent_reg_loss": 0.7140325665473938,
  "z_mean": -0.006582518915335337,
  "z_std": 0.8449788160894206
}
```

## Comparison To Provided Phase 1 Baseline

The user provided the following separate Phase 1-style metrics:

```json
{
  "train/total_loss": 0.12670975339327517,
  "train/masked_pos_loss": 0.07628750392361942,
  "train/masked_vel_loss": 0.03951423877289559,
  "train/full_recon_loss": 0.09676739671694386,
  "train/latent_smooth_loss": 0.0027600420370600142,
  "train/latent_reg_loss": 0.001203670417535821,
  "val/total_loss": 0.33045957051217556,
  "val/masked_pos_loss": 0.036813922226428986,
  "val/masked_vel_loss": 0.02053097076714039,
  "val/full_recon_loss": 0.09900692664086819,
  "val/latent_smooth_loss": 0.010438057128340006,
  "val/latent_reg_loss": 0.26310960575938225,
  "train/lr": 3.369801942738303e-05,
  "z_mean": 0.0050475504249334335,
  "z_std": 0.664700448513031
}
```

High-level comparison:

| Metric | Provided Phase 1 Baseline | Motion-Conformer-T5 Phase 1-Style Validation |
|---|---:|---:|
| `val/full_recon_loss` | `0.09900692664086819` | `0.012569112721599352` |
| `val/masked_pos_loss` | `0.036813922226428986` | `0.019928489563646085` |
| `val/masked_vel_loss` | `0.02053097076714039` | `0.0052097358447331385` |
| `val/latent_smooth_loss` | `0.010438057128340006` | `0.0383371860125264` |
| `val/latent_reg_loss` | `0.26310960575938225` | `0.714032563748801` |
| `z_mean` | `0.0050475504249334335` | `-0.006582529180205972` |
| `z_std` | `0.664700448513031` | `0.8449788144128887` |

Interpretation:

- Our Motion-Conformer-T5 checkpoint reconstructs held-out validation windows much better.
- Our latent vectors have higher variance.
- Our latent regularization and smoothness are worse by the provided metrics.
- Strong reconstruction does not translate into good text generation in the current setup.

## Overall Assessment

Phase 1-style representation learning looks promising:

```text
val/full_recon_loss = 0.01257
val/masked_vel_loss = 0.00521
z_std = 0.84498
```

Phase 2 text generation is poor:

```text
BLEU = 0.00739
chrF = 0.17312
ROUGE-L = 0.0
exact_match = 0.0
```

This means the tested model can encode and reconstruct motion windows, but it does not yet learn a reliable mapping from motion windows to fluent English translations.

## Main Failure Mode

The main issue is probably not the encoder. It is the supervision strategy for translation.

The current Modal implementation trains each 60-frame window against the entire utterance sentence. A 60-frame segment often represents only part of the sentence. That creates an ambiguous mapping:

```text
one short motion window -> full sentence
```

This can make the model learn weak correlations or generic language patterns rather than grounded translation.

## Recommended Next Steps

1. Move Phase 2 to utterance-level training.

Use the full landmark sequence for each `video_id`, encode all windows or variable-length chunks, then aggregate before T5. The target should be the full sentence once per utterance, not once per window.

2. Add attention pooling or hierarchical temporal aggregation.

The README already points in this direction. A stronger Phase 2 path should look like:

```text
full utterance landmarks
  -> per-window encoder latents
  -> temporal attention pooling / transformer aggregator
  -> T5 adapter
  -> FLAN-T5-small
```

3. Train Phase 1 separately before Phase 2.

The current combined objective worked for reconstruction, but a cleaner staged setup would make the experiment easier to interpret:

```text
Stage 1: train encoder + reconstruction decoder
Stage 2: freeze or partially unfreeze encoder, train adapter + T5 path
Stage 3: optionally add CTC/alignment loss
```

4. Evaluate at utterance level, not only window level.

Generation metrics should be computed per `video_id` / utterance. Window-level generation is useful for debugging, but it is not a fair CSLT metric.

5. Consider unfreezing part of FLAN-T5 or LoRA tuning.

The current run freezes FLAN-T5. If the adapter alone cannot align motion embeddings to T5's language space, partial decoder/cross-attention tuning or LoRA may be necessary.

## Artifact Inventory

Primary implementation:

```text
modal_motion_conformer_app.py
```

Full-run checkpoint:

```text
/checkpoints/motion_conformer_t5_full/motion_conformer_t5_smoke.pt
```

Phase 1 validation metrics:

```text
/checkpoints/motion_conformer_t5_full/eval_phase1_validation/metrics.json
```

Phase 2 validation metrics:

```text
/checkpoints/motion_conformer_t5_full/eval_validation/metrics.json
```

Phase 2 predictions:

```text
/checkpoints/motion_conformer_t5_full/eval_validation/predictions.jsonl
```

