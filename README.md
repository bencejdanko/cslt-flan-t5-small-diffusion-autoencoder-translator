# Continuous Sign Language Translation (CSLT)

> **Utterance-level sign-to-text translation** using a denoising autoencoder for representation learning and FLAN-T5-small for translation.

## Overview

This project implements a two-phase pipeline for Continuous Sign Language Translation (CSLT), trained on the [How2Sign](https://how2sign.github.io/) dataset with MediaPipe Holistic landmarks.

| Phase | What it does | Key innovation |
|-------|-------------|----------------|
| **Phase 1** | Self-supervised pretraining via a DDPM-style denoising autoencoder | Learns a compact motion manifold $Z$ from masked/noisy landmark sequences |
| **Phase 2** | Supervised translation fine-tuning | Maps $Z$ → English text via attention pooling + adapter → FLAN-T5-small |

At inference time, the diffusion decoder is discarded — only the encoder, adapter, and T5 remain.

## Architecture

```
Raw Landmarks [B, T, 543, 3]
        │
        ▼
┌─────────────────────────┐
│  Feature Engineering     │  Face downsample (468→15), normalization,
│  (data.py)               │  velocity computation
│  Output: [B, T, 540]    │  (90 keypoints × 6 features)
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Multi-Stream Encoder    │  Per-part spatial MLPs (body/face/hands)
│  (models.py)             │  + learned part embeddings
│                          │  + positional encoding
│                          │  + Transformer temporal encoder
│                          │  + strided Conv1d downsampling (4×)
│  Output: Z [B, T/4, 512]│
└────────────┬────────────┘
             │
    ┌────────┴────────┐
    │                 │
    ▼                 ▼
┌─────────┐    ┌──────────────┐
│ Phase 1 │    │  Phase 2     │
│ DDPM    │    │  Attention   │
│ Decoder │    │  Pooling     │
│(discard │    │  → Adapter   │
│ after   │    │  → FLAN-T5   │
│ training│    │              │
└─────────┘    └──────────────┘
```

### Phase 1: Denoising Pretraining (DDPM)

The encoder is trained with a **proper DDPM objective** — the decoder predicts the noise (epsilon) added to the clean signal, not the clean signal itself.

- **Noise schedule**: linear or cosine beta schedule with precomputed $\bar{\alpha}_t$
- **Forward process**: $x_t = \sqrt{\bar{\alpha}_t} \cdot x_0 + \sqrt{1 - \bar{\alpha}_t} \cdot \epsilon$
- **Training objective**: $\mathcal{L} = \| \epsilon - \epsilon_\theta(x_t, Z, t) \|^2$
- **Structured masking**: configurable feature corruption, time-span masking, whole-part dropout

The decoder uses per-part prediction heads (body, face, left hand, right hand) for position and velocity independently.

### Phase 2: Translation Fine-tuning

- **Utterance-level training**: full sign sequences paired with complete sentences (no chunk-level supervision)
- **Attention pooling**: learned query vectors attend over the variable-length encoder output to produce a fixed-length representation
- **Staged unfreezing**: encoder frozen during warmup → joint training with per-group LRs
- **Optional CTC alignment head**: latent alignment auxiliary loss for temporal structure

## Data Sources

### [bdanko/how2sign-landmarks-front-raw-parquet](https://huggingface.co/datasets/bdanko/how2sign-landmarks-front-raw-parquet)

Parquet shards with raw MediaPipe Holistic landmarks (543 keypoints/frame).

```python
from datasets import load_dataset
import numpy as np

dataset = load_dataset("bdanko/how2sign-landmarks-front-raw-parquet", split="train", streaming=True)
sample = next(iter(dataset))
landmarks = np.frombuffer(sample['features'], dtype=np.float32).reshape(sample['shape'])
print(f"Shape: {landmarks.shape}")  # [Frames, 543, 3]
```

### [bdanko/how2sign-rgb-front-clips](https://huggingface.co/datasets/bdanko/how2sign-rgb-front-clips)

WebDataset-formatted frontal RGB clips for visualization/verification.

## Quick Start

### Install

```bash
git clone https://github.com/bencejdanko/continuous-sign-language-translation.git
cd continuous-sign-language-translation
pip install -r requirements.txt
```

### Phase 1: Pretraining

```bash
# Smoke test (quick sanity check)
python phase1_pretrain.py --smoke_test true

# Debug run (100 samples)
python phase1_pretrain.py --max_samples 100 --epochs 3

# Full training
python phase1_pretrain.py --max_samples none --epochs 20 --batch_size 16 --mixed_precision true
```

### Phase 2: Fine-tuning

```bash
# Smoke test
python phase2_finetune.py --smoke_test true

# Debug run
python phase2_finetune.py --max_samples 100 --epochs 3 --phase1_ckpt checkpoints/phase1

# Full training with attention pooling + CTC
python phase2_finetune.py \
    --max_samples none \
    --epochs 10 \
    --batch_size 4 \
    --use_attention_pooling true \
    --use_ctc_head true \
    --phase1_ckpt checkpoints/phase1
```

### Inference

```bash
python inference.py --ckpt_dir checkpoints/phase2 --num_samples 5
```

### Colab Notebooks

The repository includes thin Colab notebook wrappers:
- `colab_phase1_diffusion.ipynb` — clones repo, installs deps, runs Phase 1
- `colab_phase2_translation.ipynb` — clones repo, installs deps, runs Phase 2

## Project Structure

```
├── config.py                   # Dataclass configs + CLI parsing
├── models.py                   # All model architectures
├── data.py                     # Dataset, feature engineering, collator
├── utils.py                    # Training utilities, metrics, checkpointing
├── phase1_pretrain.py          # Phase 1 training script
├── phase2_finetune.py          # Phase 2 training script
├── inference.py                # Standalone inference
├── requirements.txt            # Dependencies
├── colab_phase1_diffusion.ipynb
├── colab_phase2_translation.ipynb
└── README.md
```

## Configuration

All hyperparameters are configurable via CLI arguments or by editing the dataclass defaults in `config.py`.

### Key Phase 1 options

| Flag | Default | Description |
|------|---------|-------------|
| `--epochs` | 3 | Training epochs |
| `--batch_size` | 32 | Batch size |
| `--lr` | 1e-4 | Learning rate |
| `--latent_dim` | 512 | Encoder latent dimension |
| `--use_part_embeddings` | true | Learned part embeddings before fusion |
| `--ddpm_schedule_type` | linear | Noise schedule (linear/cosine) |
| `--mask_feature_corruption` | true | Enable random feature corruption |
| `--mask_time_span_masking` | true | Enable time-span masking |
| `--mask_whole_part_masking` | true | Enable whole-part dropout |
| `--mask_contrastive_consistency` | false | Enable contrastive loss between augmentations |
| `--mixed_precision` | false | FP16 mixed precision training |

### Key Phase 2 options

| Flag | Default | Description |
|------|---------|-------------|
| `--epochs` | 3 | Training epochs |
| `--warmup_epochs` | 1 | Epochs with encoder frozen |
| `--lr_encoder` | 5e-6 | Encoder learning rate (joint stage) |
| `--lr_adapter` | 1e-4 | Adapter learning rate |
| `--lr_t5` | 5e-5 | T5 learning rate |
| `--use_attention_pooling` | true | Attention-based temporal aggregation |
| `--use_ctc_head` | false | Latent CTC alignment auxiliary loss |
| `--num_beams` | 4 | Beam search width for generation |

## Evaluation Metrics

Phase 2 computes the following metrics at each epoch:

| Metric | Library | Description |
|--------|---------|-------------|
| **BLEU** | SacreBLEU | Standard corpus-level BLEU |
| **ROUGE-L** | Built-in LCS | Longest common subsequence F1 |
| **chrF** | SacreBLEU | Character n-gram F-score |
| **Exact Match** | — | Percentage of exactly matching predictions |
| **Avg Length** | — | Mean prediction/reference token count |

## Checkpoints

Each checkpoint directory contains:

```
checkpoints/phase2/best/
├── model.pt          # Full model state dict
├── encoder.pt        # Encoder weights (for Phase 1→2 transfer)
├── adapter.pt        # Adapter weights
├── optimizer.pt      # Optimizer state
├── config.json       # Complete training configuration
├── metrics.json      # Metric summary at save time
├── metadata.json     # Epoch, step, git hash, seed, timestamp
└── tokenizer/        # T5 tokenizer files
```

## Logging

Supports multiple backends via `--log_backend`:

| Backend | Flag | Output |
|---------|------|--------|
| CSV | `csv` (default) | `logs/train_log.csv` |
| JSONL | `jsonl` | `logs/train_log.jsonl` |
| Weights & Biases | `wandb` | Dashboard |
| TensorBoard | `tensorboard` | Event files |

## Edge Deployment

For real-time inference on NVIDIA Jetson AGX Orin, the pipeline is a simple forward pass:

1. Webcam → MediaPipe → `[1, T, 540]` feature buffer
2. Frozen Encoder → $Z$ `[1, T/4, 512]`
3. Attention Pool → Adapter → FLAN-T5 → English text

The diffusion decoder is not used at inference time.

## Demonstration

The associated demo repository: [bencejdanko/continuous-sign-language-demonstration](https://github.com/bencejdanko/continuous-sign-language-demonstration)

## Model Availability

Models available at [`bdanko/continuous-sign-language-translation`](https://huggingface.co/bdanko/continuous-sign-language-translation)

## Citations

```bibtex
@inproceedings{Duarte_CVPR2021,
    title={{How2Sign: A Large-scale Multimodal Dataset for Continuous American Sign Language}},
    author={Duarte, Amanda and Palaskar, Shruti and Ventura, Lucas and Ghadiyaram, Deepti and DeHaan, Kenneth and
                   Metze, Florian and Torres, Jordi and Giro-i-Nieto, Xavier},
    booktitle={Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2021}
}
```

---

*Keywords: ASL, Sign Language, MediaPipe, Holistic, Pose Landmarks, Hand Landmarks, Face Landmarks, Keypoints, Motion Capture, Time Series, Gesture Recognition, Computer Vision, Deep Learning, Sequence Modeling, DDPM, Denoising Diffusion, FLAN-T5, Translation*
