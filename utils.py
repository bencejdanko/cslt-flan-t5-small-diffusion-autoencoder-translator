"""
utils.py — Training utilities for the CSLT pipeline.

Contains:
  - Reproducibility / seed setting
  - Optimizer and scheduler factories
  - Mixed precision + gradient accumulation helpers
  - Checkpoint save / load with full metadata
  - Logging backends (CSV, JSONL, W&B, TensorBoard)
  - Evaluation metrics (SacreBLEU, ROUGE-L, chrF, exact match)
  - DDPM masking and noise application
  - Phase 1 loss computation
"""

from __future__ import annotations

import csv
import datetime
import json
import logging
import os
import platform
import random
import subprocess
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import PART_KEYS, Phase1Config, Phase2Config, config_to_dict, save_config

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Reproducibility
# ═══════════════════════════════════════════════════════════════════════════
def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info(f"Random seed set to {seed}")


def get_git_hash() -> str:
    """Get the current git commit hash, or 'unknown' if not in a git repo."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode("utf-8").strip()
    except Exception:
        return "unknown"


# ═══════════════════════════════════════════════════════════════════════════
# Optimizer and scheduler
# ═══════════════════════════════════════════════════════════════════════════
def create_optimizer(
    model: nn.Module,
    lr: float = 1e-4,
    weight_decay: float = 1e-5,
    param_groups: Optional[List[dict]] = None,
) -> torch.optim.Optimizer:
    """Create an AdamW optimizer, optionally with per-group LRs."""
    if param_groups is not None:
        return torch.optim.AdamW(param_groups, weight_decay=weight_decay)
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str = "cosine",
    total_steps: int = 1000,
    warmup_steps: int = 100,
):
    """Create a learning rate scheduler with optional warmup."""
    from torch.optim.lr_scheduler import (
        CosineAnnealingLR,
        LambdaLR,
        LinearLR,
        SequentialLR,
    )

    if warmup_steps > 0:
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=warmup_steps,
        )

    main_steps = max(total_steps - warmup_steps, 1)
    if scheduler_type == "cosine":
        main_scheduler = CosineAnnealingLR(optimizer, T_max=main_steps)
    elif scheduler_type == "linear":
        main_scheduler = LinearLR(
            optimizer, start_factor=1.0, end_factor=0.01, total_iters=main_steps
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")

    if warmup_steps > 0:
        return SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_steps],
        )
    return main_scheduler


# ═══════════════════════════════════════════════════════════════════════════
# Gradient scaler (mixed precision)
# ═══════════════════════════════════════════════════════════════════════════
def get_grad_scaler(enabled: bool = False):
    """Return a GradScaler for mixed precision, or a no-op."""
    if enabled and torch.cuda.is_available():
        return torch.amp.GradScaler("cuda")
    return None


def get_autocast_context(enabled: bool = False, device_type: str = "cuda"):
    """Return an autocast context manager."""
    if enabled:
        return torch.amp.autocast(device_type=device_type)
    import contextlib
    return contextlib.nullcontext()


# ═══════════════════════════════════════════════════════════════════════════
# Checkpoint management
# ═══════════════════════════════════════════════════════════════════════════
def save_checkpoint(
    ckpt_dir: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: int = 0,
    step: int = 0,
    metrics: Optional[dict] = None,
    config: Optional[Any] = None,
    tokenizer=None,
    extra: Optional[dict] = None,
    tag: str = "latest",
):
    """
    Save a complete checkpoint directory with all metadata.

    Structure:
      ckpt_dir/<tag>/
        model.pt          — full model state dict
        optimizer.pt       — optimizer state (if provided)
        config.json        — training configuration
        metrics.json       — metric summary
        metadata.json      — epoch, step, git hash, timestamp, seed
        tokenizer/         — tokenizer files (if provided)
    """
    save_dir = os.path.join(ckpt_dir, tag)
    os.makedirs(save_dir, exist_ok=True)

    # Model weights
    torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))

    # Save encoder and adapter separately if the model has them
    if hasattr(model, "encoder"):
        torch.save(
            model.encoder.state_dict(),
            os.path.join(save_dir, "encoder.pt"),
        )
    if hasattr(model, "adapter"):
        torch.save(
            model.adapter.state_dict(),
            os.path.join(save_dir, "adapter.pt"),
        )

    # Optimizer
    if optimizer is not None:
        torch.save(optimizer.state_dict(), os.path.join(save_dir, "optimizer.pt"))

    # Config
    if config is not None:
        save_config(config, os.path.join(save_dir, "config.json"))

    # Metrics
    if metrics is not None:
        with open(os.path.join(save_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

    # Metadata
    metadata = {
        "epoch": epoch,
        "step": step,
        "git_hash": get_git_hash(),
        "timestamp": datetime.datetime.now().isoformat(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
    }
    if config is not None and hasattr(config, "seed"):
        metadata["seed"] = config.seed
    if config is not None and hasattr(config, "run_id"):
        metadata["run_id"] = config.run_id
    if extra:
        metadata.update(extra)
    with open(os.path.join(save_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    # Tokenizer
    if tokenizer is not None:
        tok_dir = os.path.join(save_dir, "tokenizer")
        os.makedirs(tok_dir, exist_ok=True)
        tokenizer.save_pretrained(tok_dir)

    logger.info(f"Checkpoint saved → {save_dir}")
    return save_dir


def load_checkpoint(
    ckpt_dir: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = "cpu",
    tag: str = "best",
    strict: bool = True,
) -> dict:
    """
    Load a checkpoint and return metadata.

    Returns:
        metadata dict (epoch, step, etc.)
    """
    load_dir = os.path.join(ckpt_dir, tag)

    model_path = os.path.join(load_dir, "model.pt")
    if os.path.exists(model_path):
        state = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state, strict=strict)
        logger.info(f"Model loaded from {model_path}")

    if optimizer is not None:
        opt_path = os.path.join(load_dir, "optimizer.pt")
        if os.path.exists(opt_path):
            optimizer.load_state_dict(
                torch.load(opt_path, map_location=device, weights_only=True)
            )

    meta_path = os.path.join(load_dir, "metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            return json.load(f)

    return {}


def load_encoder_from_phase1(
    encoder: nn.Module,
    phase1_ckpt_dir: str,
    device: str = "cpu",
    tag: str = "best",
):
    """Load just the encoder weights from a Phase 1 checkpoint."""
    load_dir = os.path.join(phase1_ckpt_dir, tag)

    # Try encoder.pt first, then model.pt
    encoder_path = os.path.join(load_dir, "encoder.pt")
    if os.path.exists(encoder_path):
        state = torch.load(encoder_path, map_location=device, weights_only=True)
        encoder.load_state_dict(state)
        logger.info(f"Encoder loaded from {encoder_path}")
        return

    # Fallback: try to load from the full model checkpoint
    # (Phase 1 saves encoder as part of encoder + decoder)
    model_path = os.path.join(load_dir, "model.pt")
    if os.path.exists(model_path):
        state = torch.load(model_path, map_location=device, weights_only=True)
        # Filter to encoder keys
        enc_state = {
            k.replace("encoder.", ""): v
            for k, v in state.items()
            if k.startswith("encoder.")
        }
        if enc_state:
            encoder.load_state_dict(enc_state)
            logger.info(f"Encoder extracted from full model at {model_path}")
            return

        # Maybe it IS the encoder directly
        encoder.load_state_dict(state, strict=True)
        logger.info(f"Encoder loaded directly from {model_path}")
        return

    raise FileNotFoundError(
        f"No encoder checkpoint found in {load_dir}. "
        f"Expected encoder.pt or model.pt."
    )


# ═══════════════════════════════════════════════════════════════════════════
# Logging
# ═══════════════════════════════════════════════════════════════════════════
class TrainLogger:
    """
    Unified logging backend supporting CSV, JSONL, W&B, and TensorBoard.

    Usage:
        logger = TrainLogger("csv", log_dir="logs/")
        logger.log({"epoch": 1, "loss": 0.5}, step=100)
        logger.close()
    """

    def __init__(self, backend: str = "csv", log_dir: str = "logs", **kwargs):
        self.backend = backend
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self._writer = None
        self._csv_file = None
        self._csv_writer = None
        self._headers_written = False

        if backend == "wandb":
            try:
                import wandb
                wandb.init(
                    project=kwargs.get("project", "cslt"),
                    config=kwargs.get("config", {}),
                    name=kwargs.get("run_name", None),
                )
                self._writer = wandb
            except ImportError:
                logger.warning("wandb not installed, falling back to CSV")
                self.backend = "csv"

        elif backend == "tensorboard":
            try:
                from torch.utils.tensorboard import SummaryWriter
                self._writer = SummaryWriter(log_dir=log_dir)
            except ImportError:
                logger.warning("tensorboard not installed, falling back to CSV")
                self.backend = "csv"

    def log(self, metrics: dict, step: int = 0):
        """Log a dict of metrics."""
        if self.backend == "wandb" and self._writer is not None:
            self._writer.log(metrics, step=step)

        elif self.backend == "tensorboard" and self._writer is not None:
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    self._writer.add_scalar(k, v, step)

        elif self.backend == "jsonl":
            path = os.path.join(self.log_dir, "train_log.jsonl")
            with open(path, "a") as f:
                f.write(json.dumps({"step": step, **metrics}) + "\n")

        else:  # CSV (default)
            path = os.path.join(self.log_dir, "train_log.csv")
            row = {"step": step, **metrics}
            if not self._headers_written:
                self._csv_file = open(path, "w", newline="")
                self._csv_writer = csv.DictWriter(
                    self._csv_file, fieldnames=list(row.keys()), extrasaction="ignore"
                )
                self._csv_writer.writeheader()
                self._headers_written = True
            if self._csv_writer:
                self._csv_writer.writerow(row)
                self._csv_file.flush()

    def close(self):
        """Clean up resources."""
        if self.backend == "wandb" and self._writer is not None:
            self._writer.finish()
        elif self.backend == "tensorboard" and self._writer is not None:
            self._writer.close()
        elif self._csv_file is not None:
            self._csv_file.close()


# ═══════════════════════════════════════════════════════════════════════════
# Evaluation metrics
# ═══════════════════════════════════════════════════════════════════════════
def compute_metrics(
    predictions: List[str],
    references: List[str],
) -> dict:
    """
    Compute a suite of translation metrics.

    Returns dict with:
      - bleu: SacreBLEU score
      - rouge_l: ROUGE-L F1
      - chrf: chrF score
      - exact_match: fraction of exact matches
      - avg_pred_len: average prediction length (tokens)
      - avg_ref_len: average reference length (tokens)
    """
    import sacrebleu

    if not predictions or not references:
        return {
            "bleu": 0.0, "rouge_l": 0.0, "chrf": 0.0,
            "exact_match": 0.0, "avg_pred_len": 0.0, "avg_ref_len": 0.0,
        }

    # SacreBLEU
    bleu = sacrebleu.corpus_bleu(predictions, [references])

    # chrF
    chrf = sacrebleu.corpus_chrf(predictions, [references])

    # ROUGE-L (simple implementation without external dependency)
    rouge_l_scores = []
    for pred, ref in zip(predictions, references):
        rouge_l_scores.append(_rouge_l_f1(pred, ref))
    avg_rouge_l = sum(rouge_l_scores) / len(rouge_l_scores) if rouge_l_scores else 0.0

    # Exact match
    exact_matches = sum(
        1 for p, r in zip(predictions, references)
        if p.strip().lower() == r.strip().lower()
    )
    exact_match_rate = exact_matches / len(predictions)

    # Average lengths
    avg_pred_len = sum(len(p.split()) for p in predictions) / len(predictions)
    avg_ref_len = sum(len(r.split()) for r in references) / len(references)

    return {
        "bleu": bleu.score,
        "rouge_l": avg_rouge_l * 100,
        "chrf": chrf.score,
        "exact_match": exact_match_rate * 100,
        "avg_pred_len": avg_pred_len,
        "avg_ref_len": avg_ref_len,
    }


def _rouge_l_f1(prediction: str, reference: str) -> float:
    """Compute ROUGE-L F1 between two strings using LCS."""
    pred_tokens = prediction.strip().lower().split()
    ref_tokens = reference.strip().lower().split()

    if not pred_tokens or not ref_tokens:
        return 0.0

    # LCS length
    m, n = len(pred_tokens), len(ref_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_tokens[i - 1] == ref_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs_len = dp[m][n]
    if lcs_len == 0:
        return 0.0

    precision = lcs_len / m
    recall = lcs_len / n
    f1 = 2 * precision * recall / (precision + recall)
    return f1


def print_sample_predictions(
    predictions: List[str],
    references: List[str],
    n: int = 3,
):
    """Print a few sample predictions for qualitative evaluation."""
    n = min(n, len(predictions))
    print("\n" + "=" * 60)
    print("Sample Predictions")
    print("=" * 60)
    for i in range(n):
        print(f"  [{i+1}] Pred: {predictions[i]}")
        print(f"       Ref:  {references[i]}")
        print()


def save_epoch_metrics(
    ckpt_dir: str,
    epoch: int,
    metrics: dict,
):
    """Save per-epoch metrics as JSON."""
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, f"metrics_epoch_{epoch:03d}.json")
    with open(path, "w") as f:
        json.dump({"epoch": epoch, **metrics}, f, indent=2)


# ═══════════════════════════════════════════════════════════════════════════
# DDPM noise + masking utilities (Phase 1)
# ═══════════════════════════════════════════════════════════════════════════
def apply_masking(
    clean_dict: Dict[str, torch.Tensor],
    masking_cfg=None,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Apply configurable masking strategies to a batch of clean features.

    Masking types (independently toggleable):
      - feature_corruption: random element-wise dropout
      - time_span_masking: contiguous temporal span zeroing
      - whole_part_masking: drop entire body parts

    Args:
        clean_dict: dict of [B, T, dim] tensors
        masking_cfg: MaskingConfig instance (or None for defaults)

    Returns:
        (masked_dict, mask_indicators): masked features and binary masks
    """
    if masking_cfg is None:
        from config import MaskingConfig
        masking_cfg = MaskingConfig()

    B, T, _ = clean_dict["body_pos"].shape
    masked_dict = {}
    masks = {}

    # Decide which masking type to use for this batch
    p = random.random()

    for key, val in clean_dict.items():
        mask = torch.zeros_like(val)

        if p < 0.3 and masking_cfg.feature_corruption:
            # Random element-wise corruption
            drop = (torch.rand_like(val) < masking_cfg.feature_corruption_prob).float()
            mask = drop

        elif p < 0.6 and masking_cfg.time_span_masking:
            # Contiguous time span
            span_len = int(T * masking_cfg.time_span_ratio)
            start = random.randint(0, max(0, T - span_len))
            mask[:, start : start + span_len, :] = 1.0

        elif p < 0.9 and masking_cfg.whole_part_masking:
            # Whole body part dropout
            if "lhand" in key and random.random() < 0.5:
                mask[:] = 1.0
            elif "rhand" in key and random.random() < 0.5:
                mask[:] = 1.0
            elif "face" in key and random.random() < 0.2:
                mask[:] = 1.0

        # Apply mask: zero out masked positions
        masked_val = val.clone()
        masked_val[mask == 1.0] = 0.0

        masked_dict[key] = masked_val
        masks[key] = mask

    return masked_dict, masks


def compute_phase1_loss(
    pred_noise: Dict[str, torch.Tensor],
    actual_noise: Dict[str, torch.Tensor],
    masks_dict: Dict[str, torch.Tensor],
    z: torch.Tensor,
    cfg: Optional[Phase1Config] = None,
) -> Tuple[torch.Tensor, dict]:
    """
    Compute Phase 1 denoising loss with decomposed components.

    Loss components:
      - masked_pos_loss: MSE on masked position features
      - masked_vel_loss: MSE on masked velocity features
      - full_recon_loss: MSE on all positions (epsilon prediction)
      - latent_smooth_loss: temporal smoothness of encoder latent

    Returns:
        (total_loss, loss_dict) for logging
    """
    if cfg is None:
        cfg = Phase1Config()

    L_masked_pos = torch.tensor(0.0, device=z.device)
    L_masked_vel = torch.tensor(0.0, device=z.device)
    L_full = torch.tensor(0.0, device=z.device)

    count_pos, count_vel = 0, 0

    for key in PART_KEYS:
        if key not in pred_noise or key not in actual_noise:
            continue

        pred = pred_noise[key]
        target = actual_noise[key]
        mask = masks_dict.get(key, torch.zeros_like(pred))

        mse = F.mse_loss(pred, target, reduction="none")

        # Masked loss (only where masking was applied)
        mask_sum = mask.sum()
        if mask_sum > 0:
            masked_mse = (mse * mask).sum() / (mask_sum + 1e-8)
        else:
            masked_mse = torch.tensor(0.0, device=z.device)

        # Full reconstruction loss
        full_mse = mse.mean()

        if "pos" in key:
            L_masked_pos = L_masked_pos + masked_mse
            count_pos += 1
        elif "vel" in key and cfg.masking.velocity_reconstruction:
            L_masked_vel = L_masked_vel + masked_mse
            count_vel += 1

        L_full = L_full + full_mse

    # Normalize by number of parts
    if count_pos > 0:
        L_masked_pos = L_masked_pos / count_pos
    if count_vel > 0:
        L_masked_vel = L_masked_vel / count_vel
    L_full = L_full / len(PART_KEYS)

    # Latent smoothness
    L_smooth = torch.tensor(0.0, device=z.device)
    if cfg.masking.latent_smoothness:
        L_smooth = torch.mean((z[:, 1:, :] - z[:, :-1, :]) ** 2)

    # Latent regularization (Unit Variance / Mean zero)
    # This prevents the latent space from collapsing or drifting.
    L_lat_mean = torch.mean(z)
    L_lat_var = torch.var(z)
    L_lat_reg = L_lat_mean**2 + (L_lat_var - 1)**2

    # Weighted sum
    total = (
        cfg.w_masked_pos * L_masked_pos
        + cfg.w_masked_vel * L_masked_vel
        + cfg.w_full_recon * L_full
        + cfg.w_latent_smooth * L_smooth
        + cfg.w_latent_reg * L_lat_reg
    )

    loss_dict = {
        "total_loss": total.item(),
        "masked_pos_loss": L_masked_pos.item(),
        "masked_vel_loss": L_masked_vel.item(),
        "full_recon_loss": L_full.item(),
        "latent_smooth_loss": L_smooth.item(),
        "latent_reg_loss": L_lat_reg.item(),
    }

    return total, loss_dict


def contrastive_consistency_loss(
    z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.1
) -> torch.Tensor:
    """
    NT-Xent style contrastive loss between two augmentations of the same batch.

    z1, z2: [B, T', D] — mean-pooled to [B, D] before comparison.
    """
    # Mean pool over time
    z1_pool = z1.mean(dim=1)  # [B, D]
    z2_pool = z2.mean(dim=1)  # [B, D]

    # L2 normalize
    z1_norm = F.normalize(z1_pool, dim=-1)
    z2_norm = F.normalize(z2_pool, dim=-1)

    B = z1_norm.size(0)
    # Similarity matrix
    sim = torch.mm(z1_norm, z2_norm.t()) / temperature  # [B, B]

    # Positive pairs are on the diagonal
    labels = torch.arange(B, device=sim.device)
    loss = F.cross_entropy(sim, labels)
    return loss


# ═══════════════════════════════════════════════════════════════════════════
# Misc utilities
# ═══════════════════════════════════════════════════════════════════════════
def flatten_features(feat_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Concatenate all part features into a single flat tensor [B, T, 540]."""
    return torch.cat([feat_dict[k] for k in PART_KEYS], dim=-1)


def split_flat_to_dict(
    flat: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Split a [B, T, 540] flat tensor back into per-part dict."""
    from config import PART_DIMS

    result = {}
    offset = 0
    for key in PART_KEYS:
        dim = PART_DIMS[key]
        result[key] = flat[:, :, offset : offset + dim]
        offset += dim
    return result


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device() -> str:
    """Get the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
