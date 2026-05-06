"""ASL-to-English translation using MultiStreamSemanticEncoder + LoRA FLAN-T5-base on Modal.

Self-contained Modal app. Modes:
  1. smoke_test — 10 samples, 2 epochs, verify loss decreases
  2. train      — full 40-epoch training on How2Sign
  3. evaluate   — generate translations on validation set

Architecture:
  How2Sign landmarks [T, 543, 3]
    → engineer_features_multistream()
    → MultiStreamSemanticEncoder (4 layers, 8 heads) → [B, T/4, 512]
    → AttentionPooling (32 queries) → [B, 32, 512]
    → LatentToT5Adapter → [B, 32, 768]
    → FLAN-T5-base decoder (LoRA r=16 on q,v)
    → English sentence
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import modal

APP_NAME = "asl-translation"
CKPT_VOLUME = "asl-translation-checkpoints"
LANDMARKS_REPO = "bdanko/how2sign-landmarks-front-raw-parquet"

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.3.1",
        "transformers==4.40.0",
        "peft==0.10.0",
        "datasets==2.18.0",
        "huggingface_hub==0.21.0",
        "sacrebleu==2.4.0",
        "rouge-score==0.1.2",
        "sentencepiece",
        "protobuf",
        "numpy",
        "tqdm",
        "pyarrow==15.0.0",
    )
)

ckpt_vol = modal.Volume.from_name(CKPT_VOLUME, create_if_missing=True)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class TranslationConfig:
    # Data
    landmarks_repo: str = LANDMARKS_REPO
    max_samples: Optional[int] = None
    max_seq_frames: int = 512
    max_target_length: int = 128

    # Encoder
    d_model: int = 384
    latent_dim: int = 512
    encoder_layers: int = 4
    encoder_heads: int = 8
    encoder_dropout: float = 0.1

    # T5
    t5_name: str = "google/flan-t5-base"
    t5_dim: int = 768

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    # Attention pooling
    pool_num_queries: int = 32
    pool_num_heads: int = 4

    # CTC
    use_ctc: bool = True
    ctc_vocab_size: int = 256
    ctc_weight: float = 0.1

    # Training
    epochs: int = 40
    warmup_epochs: int = 5
    refine_epoch: int = 30
    batch_size: int = 4
    grad_accum: int = 8
    label_smoothing: float = 0.1
    grad_clip: float = 1.0
    latent_reg_weight: float = 0.01

    # Learning rates per stage
    lr_encoder_joint: float = 5e-6
    lr_adapter_warmup: float = 1e-4
    lr_adapter_joint: float = 5e-5
    lr_adapter_refine: float = 1e-5
    lr_t5_warmup: float = 5e-5
    lr_t5_joint: float = 2e-5
    lr_t5_refine: float = 5e-6

    # Generation
    num_beams: int = 5
    max_new_tokens: int = 64

    # Augmentation
    augment: bool = True
    speed_perturb_range: Tuple[float, float] = (0.85, 1.15)
    magnitude_scale_range: Tuple[float, float] = (0.93, 1.07)
    joint_dropout_prob: float = 0.1
    temporal_jitter: int = 3

    # Checkpointing
    ckpt_dir: str = "/ckpt/translation"
    eval_every: int = 5
    seed: int = 42


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FACE_LANDMARK_IDXS = [70, 105, 336, 300, 33, 133, 362, 263, 4, 61, 291, 13, 14, 17, 0]
POSE_SLICE = slice(0, 33)
FACE_SLICE = slice(33, 501)
LHAND_SLICE = slice(501, 522)
RHAND_SLICE = slice(522, 543)

PART_KEYS = [
    "body_pos", "body_vel", "face_pos", "face_vel",
    "lhand_pos", "lhand_vel", "rhand_pos", "rhand_vel",
]


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
def engineer_features_multistream(raw):
    """[T, 543, 3] → dict of 8 tensors [T, feat_dim] or None if T < 2."""
    import numpy as np
    import torch

    T = raw.shape[0]
    if T < 2:
        return None

    pose = raw[:, POSE_SLICE, :]
    face = raw[:, FACE_SLICE, :][:, FACE_LANDMARK_IDXS, :]
    lhand = raw[:, LHAND_SLICE, :]
    rhand = raw[:, RHAND_SLICE, :]

    shoulder_center = (pose[:, 11, :] + pose[:, 12, :]) / 2.0
    pose_norm = pose - shoulder_center[:, None, :]
    face_center = face.mean(axis=1, keepdims=True)
    face_norm = face - face_center
    lhand_norm = lhand - lhand[:, 0:1, :]
    rhand_norm = rhand - rhand[:, 0:1, :]

    def vel(x):
        v = np.zeros_like(x)
        v[1:] = x[1:] - x[:-1]
        return v

    return {
        "body_pos": torch.from_numpy(pose_norm.reshape(T, -1).astype(np.float32)),
        "face_pos": torch.from_numpy(face_norm.reshape(T, -1).astype(np.float32)),
        "lhand_pos": torch.from_numpy(lhand_norm.reshape(T, -1).astype(np.float32)),
        "rhand_pos": torch.from_numpy(rhand_norm.reshape(T, -1).astype(np.float32)),
        "body_vel": torch.from_numpy(vel(pose_norm).reshape(T, -1).astype(np.float32)),
        "face_vel": torch.from_numpy(vel(face_norm).reshape(T, -1).astype(np.float32)),
        "lhand_vel": torch.from_numpy(vel(lhand_norm).reshape(T, -1).astype(np.float32)),
        "rhand_vel": torch.from_numpy(vel(rhand_norm).reshape(T, -1).astype(np.float32)),
    }


# ---------------------------------------------------------------------------
# Data augmentation
# ---------------------------------------------------------------------------
def augment_features(feat_dict, cfg: TranslationConfig):
    """Apply data augmentation to feature dict (in-place modification)."""
    import torch
    import numpy as np

    T = feat_dict["body_pos"].shape[0]

    # Speed perturbation (resample temporal axis)
    speed = random.uniform(*cfg.speed_perturb_range)
    new_T = max(2, int(T / speed))
    if new_T != T:
        for key in PART_KEYS:
            x = feat_dict[key].unsqueeze(0).unsqueeze(0)  # [1, 1, T, D]
            x = torch.nn.functional.interpolate(x, size=(new_T, x.shape[-1]), mode="bilinear", align_corners=False)
            feat_dict[key] = x.squeeze(0).squeeze(0)

    # Magnitude scaling
    scale = random.uniform(*cfg.magnitude_scale_range)
    for key in PART_KEYS:
        feat_dict[key] = feat_dict[key] * scale

    # Joint dropout (zero random joints per frame)
    if random.random() < 0.5:
        T_new = feat_dict["body_pos"].shape[0]
        for key in PART_KEYS:
            if "pos" in key:
                mask = (torch.rand(T_new, 1) > cfg.joint_dropout_prob).float()
                feat_dict[key] = feat_dict[key] * mask
                vel_key = key.replace("pos", "vel")
                feat_dict[vel_key] = feat_dict[vel_key] * mask

    # Temporal jitter (shift start)
    jitter = random.randint(-cfg.temporal_jitter, cfg.temporal_jitter)
    if jitter != 0:
        T_new = feat_dict["body_pos"].shape[0]
        for key in PART_KEYS:
            if jitter > 0 and jitter < T_new:
                feat_dict[key] = feat_dict[key][jitter:]
            elif jitter < 0 and abs(jitter) < T_new:
                feat_dict[key] = feat_dict[key][:jitter]

    return feat_dict


# ---------------------------------------------------------------------------
# Models (inlined from cslt/models.py)
# ---------------------------------------------------------------------------
def _build_models():
    """Lazy import and return model classes."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class PositionalEncoding(nn.Module):
        def __init__(self, d_model: int, max_len: int = 5000):
            super().__init__()
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
            )
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)
            self.register_buffer("pe", pe)

        def forward(self, x):
            return x + self.pe[:, :x.size(1), :]

    class PartSpatialEncoder(nn.Module):
        def __init__(self, in_dim: int, out_dim: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, out_dim), nn.GELU(),
                nn.LayerNorm(out_dim), nn.Linear(out_dim, out_dim),
            )

        def forward(self, x):
            return self.net(x)

    class MultiStreamSemanticEncoder(nn.Module):
        def __init__(self, d_model=384, latent_dim=512, num_layers=3, nhead=8, dropout=0.1):
            super().__init__()
            self.d_model = d_model
            self.latent_dim = latent_dim
            part_dim = d_model // 4

            self.body_encoder = PartSpatialEncoder(33 * 3 * 2, part_dim)
            self.face_encoder = PartSpatialEncoder(15 * 3 * 2, part_dim)
            self.lhand_encoder = PartSpatialEncoder(21 * 3 * 2, part_dim)
            self.rhand_encoder = PartSpatialEncoder(21 * 3 * 2, part_dim)

            self.part_embeddings = nn.ParameterDict({
                "body": nn.Parameter(torch.randn(1, 1, part_dim) * 0.02),
                "face": nn.Parameter(torch.randn(1, 1, part_dim) * 0.02),
                "lhand": nn.Parameter(torch.randn(1, 1, part_dim) * 0.02),
                "rhand": nn.Parameter(torch.randn(1, 1, part_dim) * 0.02),
            })

            self.fusion = nn.Sequential(
                nn.Linear(part_dim * 4, d_model), nn.GELU(), nn.LayerNorm(d_model)
            )
            self.pos_encoder = PositionalEncoding(d_model)

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
                dropout=dropout, batch_first=True, activation="gelu",
            )
            self.temporal = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

            self.downsample = nn.Conv1d(d_model, latent_dim, kernel_size=3, stride=2, padding=1)
            self.downsample2 = nn.Conv1d(latent_dim, latent_dim, kernel_size=3, stride=2, padding=1)

        def forward(self, inputs, src_key_padding_mask=None):
            body = torch.cat([inputs["body_pos"], inputs["body_vel"]], dim=-1)
            face = torch.cat([inputs["face_pos"], inputs["face_vel"]], dim=-1)
            lhand = torch.cat([inputs["lhand_pos"], inputs["lhand_vel"]], dim=-1)
            rhand = torch.cat([inputs["rhand_pos"], inputs["rhand_vel"]], dim=-1)

            b_feat = self.body_encoder(body) + self.part_embeddings["body"]
            f_feat = self.face_encoder(face) + self.part_embeddings["face"]
            l_feat = self.lhand_encoder(lhand) + self.part_embeddings["lhand"]
            r_feat = self.rhand_encoder(rhand) + self.part_embeddings["rhand"]

            fused = self.fusion(torch.cat([b_feat, f_feat, l_feat, r_feat], dim=-1))
            fused = self.pos_encoder(fused)
            temp_out = self.temporal(fused, src_key_padding_mask=src_key_padding_mask)

            x = temp_out.transpose(1, 2)
            x = F.gelu(self.downsample(x))
            x = F.gelu(self.downsample2(x))
            return x.transpose(1, 2)  # [B, T', latent_dim]

    class AttentionPooling(nn.Module):
        def __init__(self, d_model=512, num_heads=4, num_queries=32, dropout=0.1):
            super().__init__()
            self.query = nn.Parameter(torch.randn(1, num_queries, d_model) * 0.02)
            self.attn = nn.MultiheadAttention(
                embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True
            )
            self.norm = nn.LayerNorm(d_model)

        def forward(self, z, key_padding_mask=None):
            B = z.size(0)
            queries = self.query.expand(B, -1, -1)
            attn_out, _ = self.attn(queries, z, z, key_padding_mask=key_padding_mask)
            return self.norm(attn_out)

    class LatentToT5Adapter(nn.Module):
        def __init__(self, latent_dim=512, t5_dim=512, dropout=0.1):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(latent_dim, t5_dim), nn.GELU(),
                nn.LayerNorm(t5_dim), nn.Dropout(dropout),
                nn.Linear(t5_dim, t5_dim),
            )

        def forward(self, z):
            return self.net(z)

    class LatentAlignmentHead(nn.Module):
        def __init__(self, latent_dim=512, vocab_size=256):
            super().__init__()
            self.vocab_size = vocab_size
            self.projection = nn.Linear(latent_dim, vocab_size + 1)

        def forward(self, z):
            logits = self.projection(z)
            log_probs = F.log_softmax(logits, dim=-1)
            return log_probs.permute(1, 0, 2)  # [T', B, V+1]

    class SignToTextModel(nn.Module):
        def __init__(self, encoder, latent_dim=512, t5_name="google/flan-t5-base",
                     t5_dim=768, adapter_dropout=0.1, pool_num_heads=4,
                     pool_num_queries=32, use_ctc=False, ctc_vocab_size=256):
            super().__init__()
            self.encoder = encoder
            self.attention_pool = AttentionPooling(
                d_model=latent_dim, num_heads=pool_num_heads, num_queries=pool_num_queries
            )
            self.adapter = LatentToT5Adapter(
                latent_dim=latent_dim, t5_dim=t5_dim, dropout=adapter_dropout
            )
            self.use_ctc = use_ctc
            if use_ctc:
                self.ctc_head = LatentAlignmentHead(latent_dim=latent_dim, vocab_size=ctc_vocab_size)

            from transformers import T5ForConditionalGeneration
            self.t5 = T5ForConditionalGeneration.from_pretrained(t5_name)

        def forward(self, batch_inputs, labels=None, padding_mask=None):
            z = self.encoder(batch_inputs, src_key_padding_mask=padding_mask)
            ctc_log_probs = None
            if self.use_ctc:
                ctc_log_probs = self.ctc_head(z)
            z_pooled = self.attention_pool(z)
            z_adapted = self.adapter(z_pooled)
            t5_out = self.t5(inputs_embeds=z_adapted, labels=labels)
            if ctc_log_probs is not None:
                return t5_out, ctc_log_probs
            return t5_out

        def generate(self, batch_inputs, padding_mask=None, **kwargs):
            z = self.encoder(batch_inputs, src_key_padding_mask=padding_mask)
            z_pooled = self.attention_pool(z)
            z_adapted = self.adapter(z_pooled)
            kwargs.setdefault("repetition_penalty", 2.5)
            kwargs.setdefault("no_repeat_ngram_size", 3)
            kwargs.setdefault("length_penalty", 1.0)
            return self.t5.generate(inputs_embeds=z_adapted, **kwargs)

    return {
        "MultiStreamSemanticEncoder": MultiStreamSemanticEncoder,
        "SignToTextModel": SignToTextModel,
    }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data(cfg: TranslationConfig, split: str = "train"):
    """Load How2Sign utterances as list of (feat_dict, sentence) tuples."""
    import numpy as np
    from datasets import load_dataset

    ds = load_dataset(cfg.landmarks_repo, split=split, streaming=True)
    if split == "train":
        ds = ds.shuffle(seed=cfg.seed, buffer_size=1024)

    samples = []
    count = 0
    for sample in ds:
        if cfg.max_samples and count >= cfg.max_samples:
            break
        count += 1

        try:
            raw = np.frombuffer(sample["features"], dtype=np.float32).reshape(sample["shape"])
        except Exception:
            continue

        feat_dict = engineer_features_multistream(raw)
        if feat_dict is None:
            continue

        sentence = sample.get("sentence", "")
        if not sentence or len(sentence.strip()) < 2:
            continue

        # Cap sequence length
        T = feat_dict["body_pos"].shape[0]
        if T > cfg.max_seq_frames:
            # Uniform subsample to preserve temporal extent
            indices = np.linspace(0, T - 1, cfg.max_seq_frames, dtype=int)
            for key in PART_KEYS:
                feat_dict[key] = feat_dict[key][indices]

        samples.append((feat_dict, sentence.strip()))

    return samples


def collate_batch(batch, tokenizer, cfg: TranslationConfig, device):
    """Collate list of (feat_dict, sentence) into batched tensors."""
    import torch

    feat_dicts, sentences = zip(*batch)
    B = len(feat_dicts)

    # Find max T in batch
    seq_lens = [fd["body_pos"].shape[0] for fd in feat_dicts]
    max_T = max(seq_lens)

    # Pad features
    batched_features = {}
    for key in PART_KEYS:
        tensors = []
        for fd in feat_dicts:
            t = fd[key]
            T_i = t.shape[0]
            if T_i < max_T:
                pad = torch.zeros(max_T - T_i, t.shape[1])
                t = torch.cat([t, pad], dim=0)
            tensors.append(t)
        batched_features[key] = torch.stack(tensors).to(device)  # [B, T, D]

    # Padding mask
    padding_mask = torch.zeros(B, max_T, dtype=torch.bool, device=device)
    for i, sl in enumerate(seq_lens):
        padding_mask[i, sl:] = True

    # Tokenize sentences
    tokenized = tokenizer(
        list(sentences), return_tensors="pt", padding=True,
        truncation=True, max_length=cfg.max_target_length,
    )
    labels = tokenized.input_ids.to(device)
    labels[labels == tokenizer.pad_token_id] = -100

    return batched_features, padding_mask, labels, list(sentences)


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------
def compute_metrics(predictions: List[str], references: List[str]) -> dict:
    """Compute BLEU-4, ROUGE-L, chrF."""
    import sacrebleu
    from rouge_score import rouge_scorer

    # SacreBLEU
    bleu = sacrebleu.corpus_bleu(predictions, [references])

    # chrF
    chrf = sacrebleu.corpus_chrf(predictions, [references])

    # ROUGE-L
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_scores = [scorer.score(ref, pred) for pred, ref in zip(predictions, references)]
    rouge_l = sum(s["rougeL"].fmeasure for s in rouge_scores) / max(len(rouge_scores), 1) * 100

    return {
        "bleu": bleu.score,
        "chrf": chrf.score,
        "rouge_l": rouge_l,
        "exact_match": sum(1 for p, r in zip(predictions, references) if p.strip() == r.strip()) / max(len(predictions), 1) * 100,
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def _run_evaluate(device, models):
    """Run comprehensive evaluation on the best checkpoint."""
    import torch
    from transformers import AutoTokenizer
    from peft import LoraConfig, get_peft_model
    from collections import Counter

    cfg = TranslationConfig()
    tokenizer = AutoTokenizer.from_pretrained(cfg.t5_name)

    # Build model
    encoder = models["MultiStreamSemanticEncoder"](
        d_model=cfg.d_model, latent_dim=cfg.latent_dim,
        num_layers=cfg.encoder_layers, nhead=cfg.encoder_heads,
        dropout=cfg.encoder_dropout,
    )
    model = models["SignToTextModel"](
        encoder, latent_dim=cfg.latent_dim, t5_name=cfg.t5_name,
        t5_dim=cfg.t5_dim, pool_num_heads=cfg.pool_num_heads,
        pool_num_queries=cfg.pool_num_queries,
        use_ctc=cfg.use_ctc, ctc_vocab_size=cfg.ctc_vocab_size,
    )
    lora_cfg = LoraConfig(
        r=cfg.lora_r, lora_alpha=cfg.lora_alpha,
        target_modules=["q", "v"],
        lora_dropout=cfg.lora_dropout,
        bias="none", task_type="SEQ_2_SEQ_LM",
    )
    model.t5 = get_peft_model(model.t5, lora_cfg)

    # Load best checkpoint
    best_path = os.path.join(cfg.ckpt_dir, "best.pt")
    if not os.path.exists(best_path):
        return {"error": "No best.pt checkpoint found"}

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.to(device)
    model.eval()
    print(f"[eval] loaded best checkpoint (epoch {ckpt.get('epoch', '?')}, bleu={ckpt.get('best_bleu', '?')})")

    # Load full validation set
    val_cfg = TranslationConfig(max_samples=None, landmarks_repo=cfg.landmarks_repo, max_seq_frames=cfg.max_seq_frames)
    val_data = load_data(val_cfg, split="validation")
    print(f"[eval] validation samples: {len(val_data)}")

    # Generate predictions
    all_predictions, all_references = [], []
    with torch.no_grad():
        for i in range(0, len(val_data), cfg.batch_size):
            batch_items = val_data[i:i + cfg.batch_size]
            if not batch_items:
                break
            features, padding_mask, labels, sentences = collate_batch(
                batch_items, tokenizer, cfg, device
            )
            gen_ids = model.generate(
                features, padding_mask=padding_mask,
                max_new_tokens=cfg.max_new_tokens,
                num_beams=cfg.num_beams,
            )
            preds = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
            all_predictions.extend(preds)
            all_references.extend(sentences)
            if (i // cfg.batch_size) % 25 == 0:
                print(f"[eval] processed {len(all_predictions)}/{len(val_data)} samples")

    # Compute metrics
    metrics = compute_metrics(all_predictions, all_references)
    print(f"\n[eval] === METRICS ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    # Diversity analysis
    unique_preds = len(set(all_predictions))
    pred_lengths = [len(p.split()) for p in all_predictions]
    ref_lengths = [len(r.split()) for r in all_references]
    avg_pred_len = sum(pred_lengths) / max(len(pred_lengths), 1)
    avg_ref_len = sum(ref_lengths) / max(len(ref_lengths), 1)

    # N-gram diversity
    pred_unigrams = Counter()
    pred_bigrams = Counter()
    for p in all_predictions:
        words = p.lower().split()
        pred_unigrams.update(words)
        pred_bigrams.update(zip(words, words[1:]))

    ref_unigrams = Counter()
    ref_bigrams = Counter()
    for r in all_references:
        words = r.lower().split()
        ref_unigrams.update(words)
        ref_bigrams.update(zip(words, words[1:]))

    # Length-bucketed evaluation
    import sacrebleu
    from rouge_score import rouge_scorer

    short_preds, short_refs = [], []
    mid_preds, mid_refs = [], []
    long_preds, long_refs = [], []
    for p, r in zip(all_predictions, all_references):
        rlen = len(r.split())
        if rlen <= 8:
            short_preds.append(p); short_refs.append(r)
        elif rlen <= 20:
            mid_preds.append(p); mid_refs.append(r)
        else:
            long_preds.append(p); long_refs.append(r)

    bucketed = {}
    for name, bp, br in [("short (≤8 words)", short_preds, short_refs),
                          ("medium (9-20 words)", mid_preds, mid_refs),
                          ("long (>20 words)", long_preds, long_refs)]:
        if bp:
            bm = compute_metrics(bp, br)
            bm["count"] = len(bp)
            bucketed[name] = bm

    # Print all sample predictions
    print(f"\n[eval] === ALL PREDICTIONS ({len(all_predictions)} samples) ===")
    for j in range(len(all_predictions)):
        print(f"\n  [{j+1}] REF: {all_references[j]}")
        print(f"  [{j+1}] PRD: {all_predictions[j]}")

    result = {
        "metrics": metrics,
        "num_val_samples": len(val_data),
        "checkpoint_epoch": ckpt.get("epoch", -1),
        "unique_predictions": unique_preds,
        "unique_ratio": unique_preds / max(len(all_predictions), 1),
        "avg_pred_length": avg_pred_len,
        "avg_ref_length": avg_ref_len,
        "pred_vocab_size": len(pred_unigrams),
        "ref_vocab_size": len(ref_unigrams),
        "pred_bigram_types": len(pred_bigrams),
        "ref_bigram_types": len(ref_bigrams),
        "bucketed_metrics": bucketed,
        "predictions": [{"ref": r, "pred": p} for r, p in zip(all_references, all_predictions)],
    }

    # Save results to volume
    eval_path = os.path.join(cfg.ckpt_dir, "eval_results.json")
    with open(eval_path, "w") as f:
        json.dump(result, f, indent=2)
    ckpt_vol.commit()
    print(f"\n[eval] results saved to {eval_path}")

    return result


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    volumes={"/ckpt": ckpt_vol},
    gpu="A10G",
    timeout=60 * 60 * 12,
)
def train(mode: str = "train"):
    """Main training function."""
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer
    from peft import LoraConfig, get_peft_model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = _build_models()

    # --- Config ---
    if mode == "smoke_test":
        cfg = TranslationConfig(max_samples=10, epochs=2, eval_every=1, augment=False)
    elif mode == "evaluate":
        return _run_evaluate(device, models)
    else:
        cfg = TranslationConfig()

    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)

    print(f"[train] mode={mode}")
    print(f"[train] loading data...")

    # --- Data ---
    train_data = load_data(cfg, split="train")
    val_data = load_data(TranslationConfig(
        max_samples=min(500, cfg.max_samples or 99999),
        landmarks_repo=cfg.landmarks_repo,
        max_seq_frames=cfg.max_seq_frames,
    ), split="validation")
    print(f"[train] train={len(train_data)} val={len(val_data)} samples")

    # --- Model ---
    encoder = models["MultiStreamSemanticEncoder"](
        d_model=cfg.d_model, latent_dim=cfg.latent_dim,
        num_layers=cfg.encoder_layers, nhead=cfg.encoder_heads,
        dropout=cfg.encoder_dropout,
    )
    model = models["SignToTextModel"](
        encoder, latent_dim=cfg.latent_dim, t5_name=cfg.t5_name,
        t5_dim=cfg.t5_dim, pool_num_heads=cfg.pool_num_heads,
        pool_num_queries=cfg.pool_num_queries,
        use_ctc=cfg.use_ctc, ctc_vocab_size=cfg.ctc_vocab_size,
    )

    # Apply LoRA to T5
    lora_cfg = LoraConfig(
        r=cfg.lora_r, lora_alpha=cfg.lora_alpha,
        target_modules=["q", "v"],
        lora_dropout=cfg.lora_dropout,
        bias="none", task_type="SEQ_2_SEQ_LM",
    )
    model.t5 = get_peft_model(model.t5, lora_cfg)

    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(cfg.t5_name)

    # Count params
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] total_params={total_params:,} trainable={trainable_params:,}")
    model.t5.print_trainable_parameters()

    # --- Resume checkpoint ---
    start_epoch = 0
    best_bleu = -1.0
    resume_path = os.path.join(cfg.ckpt_dir, "latest.pt")
    if os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model_state"], strict=False)
        start_epoch = ckpt.get("epoch", 0) + 1
        best_bleu = ckpt.get("best_bleu", -1.0)
        print(f"[train] resumed from epoch {start_epoch}, best_bleu={best_bleu:.3f}")

    # --- Training loop ---
    history_path = os.path.join(cfg.ckpt_dir, "history.jsonl")

    for epoch in range(start_epoch, cfg.epochs):
        t0 = time.time()

        # Determine training stage
        if epoch < cfg.warmup_epochs:
            stage = "warmup"
        elif epoch < cfg.refine_epoch:
            stage = "joint"
        else:
            stage = "refine"

        # Setup optimizer for current stage
        model.encoder.requires_grad_(stage != "warmup")
        if stage == "warmup":
            model.encoder.eval()
        else:
            model.encoder.train()

        param_groups = []
        if stage != "warmup":
            lr_enc = cfg.lr_encoder_joint if stage == "joint" else cfg.lr_encoder_joint / 5
            param_groups.append({"params": list(model.encoder.parameters()), "lr": lr_enc})

        lr_adapter = {"warmup": cfg.lr_adapter_warmup, "joint": cfg.lr_adapter_joint, "refine": cfg.lr_adapter_refine}[stage]
        lr_t5 = {"warmup": cfg.lr_t5_warmup, "joint": cfg.lr_t5_joint, "refine": cfg.lr_t5_refine}[stage]

        adapter_params = list(model.attention_pool.parameters()) + list(model.adapter.parameters())
        if cfg.use_ctc:
            adapter_params += list(model.ctc_head.parameters())
        param_groups.append({"params": adapter_params, "lr": lr_adapter})
        param_groups.append({"params": [p for p in model.t5.parameters() if p.requires_grad], "lr": lr_t5})

        optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-5)

        # Shuffle training data
        random.shuffle(train_data)

        # Training
        model.train()
        if stage == "warmup":
            model.encoder.eval()

        total_loss = 0.0
        n_batches = 0

        for i in range(0, len(train_data), cfg.batch_size):
            batch_items = train_data[i:i + cfg.batch_size]
            if not batch_items:
                break

            # Apply augmentation
            if cfg.augment and stage != "warmup":
                aug_items = []
                for fd, sent in batch_items:
                    fd_copy = {k: v.clone() for k, v in fd.items()}
                    fd_copy = augment_features(fd_copy, cfg)
                    aug_items.append((fd_copy, sent))
                batch_items = aug_items

            features, padding_mask, labels, _ = collate_batch(batch_items, tokenizer, cfg, device)

            # Forward — split into encoder + rest to reuse z for latent reg
            z = model.encoder(features, src_key_padding_mask=padding_mask)

            # CTC loss on encoder output
            ctc_loss = torch.tensor(0.0, device=device)
            if cfg.use_ctc:
                ctc_log_probs = model.ctc_head(z)
                B_ctc = ctc_log_probs.size(1)
                T_ctc = ctc_log_probs.size(0)
                with torch.no_grad():
                    pseudo_targets = ctc_log_probs.argmax(dim=-1).T
                    target_list, target_lens = [], []
                    for b in range(B_ctc):
                        seq = pseudo_targets[b]
                        filtered = []
                        prev = -1
                        for s in seq:
                            s_val = s.item()
                            if s_val != prev and s_val != cfg.ctc_vocab_size:
                                filtered.append(s_val)
                            prev = s_val
                        if not filtered:
                            filtered = [0]
                        target_list.append(torch.tensor(filtered, device=device))
                        target_lens.append(len(filtered))
                    ctc_targets = torch.cat(target_list)
                    ctc_target_lens = torch.tensor(target_lens, device=device)
                    ctc_input_lens = torch.full((B_ctc,), T_ctc, device=device)
                ctc_loss = F.ctc_loss(
                    ctc_log_probs, ctc_targets, ctc_input_lens, ctc_target_lens,
                    blank=cfg.ctc_vocab_size, zero_infinity=True,
                )

            # Attention pool + adapter + T5
            z_pooled = model.attention_pool(z)
            z_adapted = model.adapter(z_pooled)
            t5_out = model.t5(inputs_embeds=z_adapted, labels=labels)

            # Translation loss with label smoothing
            logits = t5_out.logits
            vocab_size = logits.size(-1)
            loss = F.cross_entropy(
                logits.view(-1, vocab_size), labels.view(-1),
                label_smoothing=cfg.label_smoothing, ignore_index=-100,
            )

            # Add CTC + latent regularization
            loss = loss + cfg.ctc_weight * ctc_loss
            z_mean = z.mean()
            z_var = z.var()
            lat_reg = z_mean ** 2 + (z_var - 1) ** 2
            loss = loss + cfg.latent_reg_weight * lat_reg

            # Backward with gradient accumulation
            (loss / cfg.grad_accum).backward()
            total_loss += loss.item()
            n_batches += 1

            if n_batches % cfg.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()
                optimizer.zero_grad()

        # Final optimizer step for remaining gradients
        if n_batches % cfg.grad_accum != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            optimizer.zero_grad()

        avg_loss = total_loss / max(n_batches, 1)
        elapsed = time.time() - t0

        # --- Evaluation ---
        metrics = {}
        if (epoch + 1) % cfg.eval_every == 0 or epoch == cfg.epochs - 1:
            model.eval()
            predictions, references = [], []

            with torch.no_grad():
                for i in range(0, len(val_data), cfg.batch_size):
                    batch_items = val_data[i:i + cfg.batch_size]
                    if not batch_items:
                        break
                    features, padding_mask, labels, sentences = collate_batch(
                        batch_items, tokenizer, cfg, device
                    )
                    gen_ids = model.generate(
                        features, padding_mask=padding_mask,
                        max_new_tokens=cfg.max_new_tokens,
                        num_beams=cfg.num_beams,
                    )
                    preds = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
                    predictions.extend(preds)
                    references.extend(sentences)

            metrics = compute_metrics(predictions, references)

            # Print samples
            print(f"\n  --- Sample predictions (epoch {epoch}) ---")
            for j in range(min(5, len(predictions))):
                print(f"  REF: {references[j]}")
                print(f"  PRD: {predictions[j]}")
                print()

        # Log
        log_entry = {
            "epoch": epoch, "stage": stage, "train_loss": avg_loss,
            "time_sec": elapsed, **metrics,
        }
        print(f"[train] ep{epoch} stage={stage} loss={avg_loss:.4f} "
              f"bleu={metrics.get('bleu', 'N/A')} "
              f"rouge_l={metrics.get('rouge_l', 'N/A')} "
              f"time={elapsed:.1f}s")

        with open(history_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        # Save checkpoint
        ckpt_data = {
            "model_state": model.state_dict(),
            "epoch": epoch,
            "best_bleu": best_bleu,
            "cfg": cfg.__dict__,
        }
        torch.save(ckpt_data, os.path.join(cfg.ckpt_dir, "latest.pt"))

        if metrics.get("bleu", -1) > best_bleu:
            best_bleu = metrics["bleu"]
            torch.save(ckpt_data, os.path.join(cfg.ckpt_dir, "best.pt"))
            print(f"[train] saved new best (bleu={best_bleu:.3f})")

        ckpt_vol.commit()

    print(f"\n[train] complete. best_bleu={best_bleu:.3f}")
    return {"best_bleu": best_bleu, "final_epoch": cfg.epochs - 1}


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(mode: str = "train"):
    """Run translation training on Modal.

    Args:
        mode: 'smoke_test' | 'train' | 'evaluate'
    """
    print(f"\n[main] launching translation training (mode={mode})...")
    result = train.remote(mode=mode)
    print(f"\n[main] result: {json.dumps(result, indent=2)}")
