"""
config.py — Dataclass-driven configuration for CSLT pipeline.

All important hyperparameters are centralized here. Supports:
  - CLI argument parsing via argparse
  - YAML save/load
  - Serialization into checkpoint metadata
"""

from __future__ import annotations

import argparse
import json
import os
import uuid
from dataclasses import asdict, dataclass, field
from typing import List, Optional


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------
FACE_LANDMARK_IDXS: List[int] = [
    70, 105, 336, 300, 33, 133, 362, 263, 4, 61, 291, 13, 14, 17, 0
]
POSE_SLICE = slice(0, 33)
FACE_SLICE = slice(33, 501)
LHAND_SLICE = slice(501, 522)
RHAND_SLICE = slice(522, 543)

PART_KEYS = [
    "body_pos", "body_vel", "face_pos", "face_vel",
    "lhand_pos", "lhand_vel", "rhand_pos", "rhand_vel",
]
PART_DIMS = {
    "body_pos": 33 * 3, "body_vel": 33 * 3,
    "face_pos": 15 * 3, "face_vel": 15 * 3,
    "lhand_pos": 21 * 3, "lhand_vel": 21 * 3,
    "rhand_pos": 21 * 3, "rhand_vel": 21 * 3,
}
TOTAL_FEAT_DIM = sum(PART_DIMS.values())  # 540


# ---------------------------------------------------------------------------
# Phase 1 config
# ---------------------------------------------------------------------------
@dataclass
class MaskingConfig:
    """Which masking / augmentation strategies are enabled."""
    feature_corruption: bool = True
    time_span_masking: bool = True
    whole_part_masking: bool = True
    velocity_reconstruction: bool = True
    latent_smoothness: bool = True
    contrastive_consistency: bool = False
    # Probabilities / strengths
    feature_corruption_prob: float = 0.15
    time_span_ratio: float = 0.2
    contrastive_weight: float = 0.05


@dataclass
class DDPMConfig:
    """Diffusion noise schedule parameters."""
    num_timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02
    schedule_type: str = "linear"  # "linear" or "cosine"


@dataclass
class Phase1Config:
    # Data
    dataset_repo: str = "bdanko/how2sign-landmarks-front-raw-parquet"
    split: str = "train"
    max_samples: Optional[int] = 100
    val_max_samples: Optional[int] = None  # auto-computed if None
    batch_size: int = 32

    # Architecture
    d_model: int = 384
    latent_dim: int = 512
    encoder_layers: int = 3
    encoder_heads: int = 8
    encoder_dropout: float = 0.1
    use_part_embeddings: bool = True

    # DDPM
    ddpm: DDPMConfig = field(default_factory=DDPMConfig)

    # Masking / augmentation
    masking: MaskingConfig = field(default_factory=MaskingConfig)

    # Loss weights
    w_masked_pos: float = 1.0
    w_masked_vel: float = 1.0
    w_full_recon: float = 0.1
    w_latent_smooth: float = 0.01
    w_contrastive: float = 0.05
    w_latent_reg: float = 0.01

    # Training
    epochs: int = 3
    lr: float = 1e-4
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    scheduler: str = "cosine"  # "cosine" or "linear"
    warmup_steps: int = 100
    mixed_precision: bool = False
    gradient_accumulation_steps: int = 1

    # Checkpoint
    ckpt_dir: str = "checkpoints/phase1"
    hf_repo: str = "bdanko/continuous-sign-language-translation"
    upload_hf: bool = False

    # Reproducibility
    seed: int = 42

    # Logging
    log_backend: str = "csv"  # "csv", "jsonl", "wandb", "tensorboard"
    wandb_project: str = "cslt-phase1"
    log_every_n_steps: int = 10
    shuffle_buffer: int = 1000
    streaming: bool = True

    # Smoke test
    smoke_test: bool = False

    # Run metadata
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])

    def __post_init__(self):
        if self.val_max_samples is None:
            if self.max_samples is not None:
                self.val_max_samples = max(10, self.max_samples // 5)
            else:
                self.val_max_samples = 100


# ---------------------------------------------------------------------------
# Phase 2 config
# ---------------------------------------------------------------------------
@dataclass
class Phase2Config:
    # Data
    dataset_repo: str = "bdanko/how2sign-landmarks-front-raw-parquet"
    split: str = "train"
    max_samples: Optional[int] = 100
    val_max_samples: Optional[int] = None
    batch_size: int = 8
    max_target_length: int = 128

    # Architecture
    d_model: int = 384
    latent_dim: int = 512
    encoder_layers: int = 3
    encoder_heads: int = 8
    encoder_dropout: float = 0.1
    use_part_embeddings: bool = True
    t5_name: str = "google/flan-t5-small"
    t5_dim: int = 512
    adapter_dropout: float = 0.1

    # Attention pooling
    use_attention_pooling: bool = True
    pool_num_heads: int = 4

    # Training
    epochs: int = 3
    warmup_epochs: int = 1
    lr_encoder: float = 5e-6
    lr_adapter: float = 1e-4
    lr_t5: float = 5e-5
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    scheduler: str = "cosine"
    warmup_steps: int = 100
    mixed_precision: bool = False
    gradient_accumulation_steps: int = 1

    # Generation
    num_beams: int = 4
    max_new_tokens: int = 50

    # CTC alignment head
    use_ctc_head: bool = False
    ctc_weight: float = 0.1
    ctc_vocab_size: int = 256  # latent alignment vocab size

    # Checkpoint
    ckpt_dir: str = "checkpoints/phase2"
    phase1_ckpt: str = "checkpoints/phase1/best"
    hf_repo: str = "bdanko/continuous-sign-language-translation"
    upload_hf: bool = False

    # Reproducibility
    seed: int = 42

    # Logging
    log_backend: str = "csv"
    wandb_project: str = "cslt-phase2"
    log_every_n_steps: int = 10
    shuffle_buffer: int = 1000
    streaming: bool = True

    # Smoke test
    smoke_test: bool = False

    # Run metadata
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])

    def __post_init__(self):
        if self.val_max_samples is None:
            if self.max_samples is not None:
                self.val_max_samples = max(10, self.max_samples // 5)
            else:
                self.val_max_samples = 100


# ---------------------------------------------------------------------------
# Inference config
# ---------------------------------------------------------------------------
@dataclass
class InferenceConfig:
    ckpt_dir: str = "checkpoints/phase2/best"
    device: str = "cuda"
    num_beams: int = 4
    max_new_tokens: int = 50
    dataset_repo: str = "bdanko/how2sign-landmarks-front-raw-parquet"
    num_samples: int = 5


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------
def config_to_dict(cfg) -> dict:
    """Convert a dataclass config to a JSON-serializable dict."""
    return asdict(cfg)


def save_config(cfg, path: str):
    """Save config as JSON."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(config_to_dict(cfg), f, indent=2)


def load_config_dict(path: str) -> dict:
    """Load a config dict from JSON."""
    with open(path, "r") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------
def _add_dataclass_args(parser: argparse.ArgumentParser, dc_class, prefix: str = ""):
    """Add dataclass fields as CLI arguments."""
    for fld in dc_class.__dataclass_fields__.values():
        name = f"--{prefix}{fld.name}" if prefix else f"--{fld.name}"
        ftype = fld.type

        # Skip nested dataclasses — they get their own prefix
        if hasattr(ftype, "__dataclass_fields__"):
            continue

        if ftype == bool or ftype == "bool":
            parser.add_argument(name, type=lambda x: x.lower() in ("true", "1", "yes"),
                                default=argparse.SUPPRESS, help=f"(default: {fld.default})")
        elif ftype in ("Optional[int]", "int", int):
            parser.add_argument(name, type=lambda x: None if x.lower() == "none" else int(x),
                                default=argparse.SUPPRESS)
        elif ftype in ("Optional[float]", "float", float):
            parser.add_argument(name, type=lambda x: None if x.lower() == "none" else float(x),
                                default=argparse.SUPPRESS)
        elif ftype in ("str", str):
            parser.add_argument(name, type=str, default=argparse.SUPPRESS)


def parse_phase1_args(args=None) -> Phase1Config:
    """Parse CLI arguments into a Phase1Config."""
    parser = argparse.ArgumentParser(description="Phase 1: Denoising Pretraining")
    _add_dataclass_args(parser, Phase1Config)
    # Nested DDPM args
    _add_dataclass_args(parser, DDPMConfig, prefix="ddpm_")
    # Nested masking args
    _add_dataclass_args(parser, MaskingConfig, prefix="mask_")

    parsed = parser.parse_args(args)
    parsed_dict = vars(parsed)

    # Extract nested
    ddpm_kwargs = {}
    mask_kwargs = {}
    top_kwargs = {}
    for k, v in parsed_dict.items():
        if k.startswith("ddpm_"):
            ddpm_kwargs[k[5:]] = v
        elif k.startswith("mask_"):
            mask_kwargs[k[5:]] = v
        else:
            top_kwargs[k] = v

    cfg = Phase1Config(**top_kwargs)
    if ddpm_kwargs:
        cfg.ddpm = DDPMConfig(**{**asdict(cfg.ddpm), **ddpm_kwargs})
    if mask_kwargs:
        cfg.masking = MaskingConfig(**{**asdict(cfg.masking), **mask_kwargs})
    return cfg


def parse_phase2_args(args=None) -> Phase2Config:
    """Parse CLI arguments into a Phase2Config."""
    parser = argparse.ArgumentParser(description="Phase 2: Translation Fine-tuning")
    _add_dataclass_args(parser, Phase2Config)
    parsed = parser.parse_args(args)
    return Phase2Config(**vars(parsed))


def parse_inference_args(args=None) -> InferenceConfig:
    """Parse CLI arguments into an InferenceConfig."""
    parser = argparse.ArgumentParser(description="Inference")
    _add_dataclass_args(parser, InferenceConfig)
    parsed = parser.parse_args(args)
    return InferenceConfig(**vars(parsed))
