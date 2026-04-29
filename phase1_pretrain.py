#!/usr/bin/env python3
"""
phase1_pretrain.py — Phase 1: Denoising Autoencoder Pretraining.

Trains a MultiStreamSemanticEncoder + StructuredDiffusionDecoder using
a proper DDPM noise schedule with epsilon prediction.

Key features:
  - Utterance-level training (full sequences, not fixed chunks)
  - Variable-length batching with padding masks
  - Configurable masking ablations
  - Train/validation split with best checkpoint saving
  - Detailed loss decomposition logging
  - Optional contrastive consistency loss
  - Smoke test mode for quick sanity checks

Usage:
    python phase1_pretrain.py
    python phase1_pretrain.py --epochs 10 --batch_size 16 --max_samples 500
    python phase1_pretrain.py --smoke_test true
"""

from __future__ import annotations

import logging
import os
import sys

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from config import Phase1Config, parse_phase1_args, save_config
from data import create_dataloader
from models import (
    DDPMNoiseSchedule,
    MultiStreamSemanticEncoder,
    StructuredDiffusionDecoder,
)
from utils import (
    TrainLogger,
    apply_masking,
    compute_phase1_loss,
    contrastive_consistency_loss,
    count_parameters,
    create_optimizer,
    create_scheduler,
    flatten_features,
    get_autocast_context,
    get_device,
    get_git_hash,
    get_grad_scaler,
    save_checkpoint,
    save_epoch_metrics,
    set_seed,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Phase 1 encoder+decoder wrapper (for unified checkpointing)
# ---------------------------------------------------------------------------
class Phase1Model(torch.nn.Module):
    """Thin wrapper around encoder + decoder for checkpointing."""

    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------
def train_phase1(cfg: Phase1Config):
    device = get_device()
    set_seed(cfg.seed)

    logger.info(f"Device: {device}")
    logger.info(f"Run ID: {cfg.run_id}")
    logger.info(f"Config: {cfg}")

    # --- Data ---
    if cfg.smoke_test:
        cfg.max_samples = 4
        cfg.val_max_samples = 2
        cfg.epochs = 1
        cfg.batch_size = 2
        logger.info("🔥 SMOKE TEST MODE — minimal data")

    train_loader = create_dataloader(
        split=cfg.split,
        batch_size=cfg.batch_size,
        max_samples=cfg.max_samples,
        repo_id=cfg.dataset_repo,
        phase=1,
        shuffle_buffer=cfg.shuffle_buffer,
        streaming=cfg.streaming,
    )
    val_loader = create_dataloader(
        split="validation",
        batch_size=cfg.batch_size,
        max_samples=cfg.val_max_samples,
        repo_id=cfg.dataset_repo,
        phase=1,
        shuffle_buffer=cfg.shuffle_buffer,
        streaming=cfg.streaming,
    )

    # --- Models ---
    encoder = MultiStreamSemanticEncoder(
        d_model=cfg.d_model,
        latent_dim=cfg.latent_dim,
        num_layers=cfg.encoder_layers,
        nhead=cfg.encoder_heads,
        dropout=cfg.encoder_dropout,
        use_part_embeddings=cfg.use_part_embeddings,
    ).to(device)

    decoder = StructuredDiffusionDecoder(latent_dim=cfg.latent_dim).to(device)
    noise_schedule = DDPMNoiseSchedule(
        num_timesteps=cfg.ddpm.num_timesteps,
        beta_start=cfg.ddpm.beta_start,
        beta_end=cfg.ddpm.beta_end,
        schedule_type=cfg.ddpm.schedule_type,
    ).to(device)

    model = Phase1Model(encoder, decoder).to(device)

    logger.info(f"Encoder params: {count_parameters(encoder):,}")
    logger.info(f"Decoder params: {count_parameters(decoder):,}")

    # --- Optimizer & Scheduler ---
    optimizer = create_optimizer(
        model, lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    # Estimate total steps
    estimated_steps = (cfg.max_samples or 1000) * cfg.epochs // cfg.batch_size
    scheduler = create_scheduler(
        optimizer,
        scheduler_type=cfg.scheduler,
        total_steps=estimated_steps,
        warmup_steps=cfg.warmup_steps,
    )

    # Mixed precision
    scaler = get_grad_scaler(cfg.mixed_precision)

    # --- Logging ---
    log_dir = os.path.join(cfg.ckpt_dir, "logs")
    train_logger = TrainLogger(
        backend=cfg.log_backend,
        log_dir=log_dir,
        project=cfg.wandb_project,
        config=cfg,
    )

    # --- Training loop ---
    best_val_loss = float("inf")
    global_step = 0

    for epoch in range(cfg.epochs):
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch+1}/{cfg.epochs}")
        logger.info(f"{'='*60}")

        # === TRAIN ===
        model.train()
        epoch_losses = {
            "total_loss": 0.0,
            "masked_pos_loss": 0.0,
            "masked_vel_loss": 0.0,
            "full_recon_loss": 0.0,
            "latent_smooth_loss": 0.0,
            "latent_reg_loss": 0.0,
        }
        num_batches = 0
        accum_loss = 0.0

        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Train E{epoch+1}")):
            features = {k: v.to(device) for k, v in batch["features"].items()}
            pad_mask = batch["padding_mask"].to(device)

            with torch.amp.autocast("cuda", enabled=cfg.mixed_precision):
                # Apply masking augmentation
                masked_features, masks = apply_masking(features, cfg.masking)

                # Encode
                z = encoder(masked_features, src_key_padding_mask=pad_mask)

                # DDPM forward: add noise to clean flat features
                clean_flat = flatten_features(features)  # [B, T, 540]
                t = torch.randint(
                    0, cfg.ddpm.num_timesteps, (clean_flat.size(0),), device=device
                )
                noisy_flat, epsilon = noise_schedule.q_sample(clean_flat, t)

                # Predict noise
                pred_noise_dict = decoder(noisy_flat, z, t)

                # Split actual noise into parts for per-part loss
                from utils import split_flat_to_dict
                actual_noise_dict = split_flat_to_dict(epsilon)

                # Loss
                loss, loss_dict = compute_phase1_loss(
                    pred_noise_dict, actual_noise_dict, masks, z, cfg
                )

                # Optional contrastive consistency
                if cfg.masking.contrastive_consistency:
                    masked2, _ = apply_masking(features, cfg.masking)
                    z2 = encoder(masked2, src_key_padding_mask=pad_mask)
                    c_loss = contrastive_consistency_loss(z, z2)
                    loss = loss + cfg.w_contrastive * c_loss
                    loss_dict["contrastive_loss"] = c_loss.item()

            # Gradient accumulation
            loss = loss / cfg.gradient_accumulation_steps
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            accum_loss += loss.item()

            if (batch_idx + 1) % cfg.gradient_accumulation_steps == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad()

                global_step += 1

                # Log
                if global_step % cfg.log_every_n_steps == 0:
                    lr = optimizer.param_groups[0]["lr"]
                    train_logger.log(
                        {
                            "train/total_loss": accum_loss * cfg.gradient_accumulation_steps,
                            "train/lr": lr,
                            **{f"train/{k}": v for k, v in loss_dict.items() if k != "total_loss"},
                        },
                        step=global_step,
                    )
                accum_loss = 0.0

            # Accumulate epoch stats
            for k, v in loss_dict.items():
                if k in epoch_losses:
                    epoch_losses[k] += v
            num_batches += 1

        # Average train losses
        for k in epoch_losses:
            epoch_losses[k] /= max(num_batches, 1)

        # === VALIDATION ===
        model.eval()
        val_losses = {
            "total_loss": 0.0,
            "masked_pos_loss": 0.0,
            "masked_vel_loss": 0.0,
            "full_recon_loss": 0.0,
            "latent_smooth_loss": 0.0,
        }
        val_batches = 0
        z_stats = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Val E{epoch+1}"):
                features = {k: v.to(device) for k, v in batch["features"].items()}
                pad_mask = batch["padding_mask"].to(device)

                masked_features, masks = apply_masking(features, cfg.masking)
                z = encoder(masked_features, src_key_padding_mask=pad_mask)

                clean_flat = flatten_features(features)
                t = torch.randint(
                    0, cfg.ddpm.num_timesteps, (clean_flat.size(0),), device=device
                )
                noisy_flat, epsilon = noise_schedule.q_sample(clean_flat, t)
                pred_noise_dict = decoder(noisy_flat, z, t)

                from utils import split_flat_to_dict
                actual_noise_dict = split_flat_to_dict(epsilon)

                loss, loss_dict = compute_phase1_loss(
                    pred_noise_dict, actual_noise_dict, masks, z, cfg
                )

                for k, v in loss_dict.items():
                    if k in val_losses:
                        val_losses[k] += v
                val_batches += 1
                z_stats.append(z.cpu())

        for k in val_losses:
            val_losses[k] /= max(val_batches, 1)

        # Latent statistics (flattened over Time dimension)
        z_mean, z_std = 0.0, 0.0
        if z_stats:
            # Each z is [B, T', D] -> flatten to [B*T', D]
            all_z = torch.cat([z.view(-1, z.size(-1)) for z in z_stats], dim=0)
            z_mean = all_z.mean().item()
            z_std = all_z.std().item()

        # Log epoch summary
        logger.info(
            f"Epoch {epoch+1}/{cfg.epochs} | "
            f"Train: {epoch_losses['total_loss']:.4f} | "
            f"Val: {val_losses['total_loss']:.4f} | "
            f"z_mean: {z_mean:.4f}, z_std: {z_std:.4f}"
        )
        logger.info(
            f"  Decomposed val — "
            f"pos: {val_losses['masked_pos_loss']:.4f}, "
            f"vel: {val_losses['masked_vel_loss']:.4f}, "
            f"recon: {val_losses['full_recon_loss']:.4f}, "
            f"smooth: {val_losses['latent_smooth_loss']:.4f}, "
            f"reg: {val_losses['latent_reg_loss']:.4f}"
        )

        # Epoch-level logging
        lr = optimizer.param_groups[0]["lr"]
        epoch_metrics = {
            **{f"train/{k}": v for k, v in epoch_losses.items()},
            **{f"val/{k}": v for k, v in val_losses.items()},
            "train/lr": lr,
            "z_mean": z_mean,
            "z_std": z_std,
        }
        train_logger.log(epoch_metrics, step=global_step)
        save_epoch_metrics(cfg.ckpt_dir, epoch, epoch_metrics)

        # Save latest checkpoint
        save_checkpoint(
            ckpt_dir=cfg.ckpt_dir,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            step=global_step,
            metrics=epoch_metrics,
            config=cfg,
            tag="latest",
        )

        # Save best checkpoint
        if val_losses["total_loss"] < best_val_loss:
            best_val_loss = val_losses["total_loss"]
            save_checkpoint(
                ckpt_dir=cfg.ckpt_dir,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                step=global_step,
                metrics=epoch_metrics,
                config=cfg,
                tag="best",
            )
            logger.info(f"  ✓ New best model (val_loss={best_val_loss:.4f})")

    train_logger.close()
    logger.info("Phase 1 training complete.")

    # Smoke test shape assertions
    if cfg.smoke_test:
        logger.info("Running shape assertions for smoke test...")
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                features = {k: v.to(device) for k, v in batch["features"].items()}
                pad_mask = batch["padding_mask"].to(device)
                z = encoder(features, src_key_padding_mask=pad_mask)
                B = pad_mask.size(0)
                assert z.dim() == 3, f"Expected 3D z, got {z.dim()}"
                assert z.size(0) == B, f"Batch size mismatch: {z.size(0)} vs {B}"
                assert z.size(2) == cfg.latent_dim, f"Latent dim: {z.size(2)} vs {cfg.latent_dim}"
                logger.info(f"  ✓ Encoder output shape: {z.shape}")

                # Test decoder
                flat = flatten_features(features)
                t = torch.zeros(B, dtype=torch.long, device=device)
                noisy, eps = noise_schedule.q_sample(flat, t)
                pred = decoder(noisy, z, t)
                for key in pred:
                    assert pred[key].size(0) == B
                    logger.info(f"  ✓ Decoder head '{key}' shape: {pred[key].shape}")
                break
        logger.info("All smoke test assertions passed ✓")


if __name__ == "__main__":
    cfg = parse_phase1_args()
    train_phase1(cfg)
