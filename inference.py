#!/usr/bin/env python3
"""
inference.py — Standalone inference entrypoint for CSLT.

Loads a trained Phase 2 checkpoint and generates English translations
from sign language landmark sequences.

This script does NOT require editing the training scripts.

Usage:
    # Translate samples from the validation set
    python inference.py --ckpt_dir checkpoints/phase2 --num_samples 5

    # Custom device
    python inference.py --ckpt_dir checkpoints/phase2 --device cpu
"""

from __future__ import annotations

import json
import logging
import os
import sys

import numpy as np
import torch
from transformers import T5Tokenizer

from config import InferenceConfig, load_config_dict, parse_inference_args
from data import engineer_features_multistream
from models import MultiStreamSemanticEncoder, SignToTextModel
from utils import get_device, set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_model_for_inference(ckpt_dir: str, device: str = "cuda", tag: str = "best"):
    """
    Load a complete SignToTextModel from a checkpoint directory.

    Returns:
        (model, tokenizer, config_dict)
    """
    load_dir = os.path.join(ckpt_dir, tag)

    # Load config
    config_path = os.path.join(load_dir, "config.json")
    if os.path.exists(config_path):
        cfg_dict = load_config_dict(config_path)
        logger.info(f"Config loaded from {config_path}")
    else:
        logger.warning("No config.json found, using defaults")
        cfg_dict = {}

    # Build model
    encoder = MultiStreamSemanticEncoder(
        d_model=cfg_dict.get("d_model", 384),
        latent_dim=cfg_dict.get("latent_dim", 512),
        num_layers=cfg_dict.get("encoder_layers", 3),
        nhead=cfg_dict.get("encoder_heads", 8),
        dropout=0.0,  # No dropout at inference
        use_part_embeddings=cfg_dict.get("use_part_embeddings", True),
    )

    t5_name = cfg_dict.get("t5_name", "google/flan-t5-small")

    model = SignToTextModel(
        encoder=encoder,
        latent_dim=cfg_dict.get("latent_dim", 512),
        t5_name=t5_name,
        t5_dim=cfg_dict.get("t5_dim", 512),
        adapter_dropout=0.0,
        use_attention_pooling=cfg_dict.get("use_attention_pooling", True),
        pool_num_heads=cfg_dict.get("pool_num_heads", 4),
        use_ctc_head=False,  # Not needed for inference
    )

    # Load weights
    model_path = os.path.join(load_dir, "model.pt")
    if os.path.exists(model_path):
        state = torch.load(model_path, map_location=device, weights_only=True)
        # Handle strict=False in case CTC head was used during training but not here
        model.load_state_dict(state, strict=False)
        logger.info(f"Model weights loaded from {model_path}")
    else:
        raise FileNotFoundError(f"No model.pt found in {load_dir}")

    model = model.to(device)
    model.eval()

    # Load tokenizer
    tok_dir = os.path.join(load_dir, "tokenizer")
    if os.path.exists(tok_dir):
        tokenizer = T5Tokenizer.from_pretrained(tok_dir)
        logger.info(f"Tokenizer loaded from {tok_dir}")
    else:
        tokenizer = T5Tokenizer.from_pretrained(t5_name)
        logger.info(f"Tokenizer loaded from HuggingFace: {t5_name}")

    return model, tokenizer, cfg_dict


def preprocess_sample(raw_landmarks: np.ndarray, device: str = "cpu"):
    """
    Preprocess a single raw landmark array into model input format.

    Args:
        raw_landmarks: [T, 543, 3] raw MediaPipe landmarks

    Returns:
        (features_dict, padding_mask) ready for model input (batch dim added)
    """
    feat_dict = engineer_features_multistream(raw_landmarks)
    if feat_dict is None:
        raise ValueError("Sequence too short (< 2 frames)")

    # Add batch dimension
    features = {k: v.unsqueeze(0).to(device) for k, v in feat_dict.items()}
    padding_mask = torch.zeros(1, feat_dict["body_pos"].shape[0], dtype=torch.bool, device=device)

    return features, padding_mask


def translate(
    model: SignToTextModel,
    tokenizer: T5Tokenizer,
    features: dict,
    padding_mask: torch.Tensor,
    num_beams: int = 4,
    max_new_tokens: int = 50,
) -> str:
    """
    Generate English translation from preprocessed sign features.

    Returns:
        Translated text string.
    """
    with torch.no_grad():
        gen_ids = model.generate(
            features,
            padding_mask=padding_mask,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
        )
    return tokenizer.decode(gen_ids[0], skip_special_tokens=True)


def run_inference(cfg: InferenceConfig):
    """Run inference on samples from the dataset."""
    device = cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu"

    # Load model
    model, tokenizer, config_dict = load_model_for_inference(
        cfg.ckpt_dir, device=device
    )

    # Load samples from dataset
    from datasets import load_dataset

    ds = load_dataset(cfg.dataset_repo, split="validation", streaming=True)

    print("\n" + "=" * 70)
    print("CSLT Inference — Sign Language → English Translation")
    print("=" * 70)

    count = 0
    for sample in ds:
        if count >= cfg.num_samples:
            break

        raw = np.frombuffer(sample["features"], dtype=np.float32).reshape(
            sample["shape"]
        )
        ref_sentence = sample.get("sentence", "N/A")

        try:
            features, pad_mask = preprocess_sample(raw, device=device)
            prediction = translate(
                model, tokenizer, features, pad_mask,
                num_beams=cfg.num_beams,
                max_new_tokens=cfg.max_new_tokens,
            )

            print(f"\n[Sample {count+1}]")
            print(f"  Video ID:    {sample.get('video_id', 'unknown')}")
            print(f"  Frames:      {raw.shape[0]}")
            print(f"  Reference:   {ref_sentence}")
            print(f"  Prediction:  {prediction}")

        except Exception as e:
            logger.warning(f"Skipping sample: {e}")
            continue

        count += 1

    print("\n" + "=" * 70)
    print(f"Translated {count} samples.")


if __name__ == "__main__":
    cfg = parse_inference_args()
    run_inference(cfg)
