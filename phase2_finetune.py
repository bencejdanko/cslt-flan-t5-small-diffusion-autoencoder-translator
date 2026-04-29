#!/usr/bin/env python3
"""
phase2_finetune.py — Phase 2: Translation Fine-tuning.

Loads a frozen encoder from Phase 1 and fine-tunes a SignToTextModel
(encoder → attention pooling → adapter → FLAN-T5) on utterance-level
sign-to-text translation.

Key features:
  - Utterance-level training (full sequences paired with sentences)
  - Staged unfreezing (warmup: encoder frozen → joint: encoder trainable)
  - Full evaluation protocol: SacreBLEU, ROUGE-L, chrF, exact match
  - Per-epoch metrics JSON + sample predictions
  - Optional CTC alignment head
  - Complete checkpoint packaging (model, config, tokenizer, metrics)
  - Smoke test mode

Usage:
    python phase2_finetune.py
    python phase2_finetune.py --epochs 5 --batch_size 4 --max_samples 200
    python phase2_finetune.py --smoke_test true
"""

from __future__ import annotations

import logging
import os
import sys

import torch
from tqdm.auto import tqdm
from transformers import T5Tokenizer

from config import Phase2Config, parse_phase2_args, save_config
from data import create_dataloader
from models import MultiStreamSemanticEncoder, SignToTextModel
from utils import (
    TrainLogger,
    compute_metrics,
    count_parameters,
    create_optimizer,
    create_scheduler,
    get_autocast_context,
    get_device,
    get_grad_scaler,
    load_encoder_from_phase1,
    print_sample_predictions,
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
# Staged unfreezing
# ---------------------------------------------------------------------------
def setup_for_stage(
    model: SignToTextModel,
    stage: str,
    cfg: Phase2Config,
) -> torch.optim.Optimizer:
    """
    Configure which parameters are trainable and return an optimizer.

    Stages:
      - "warmup": encoder frozen, train adapter + T5
      - "joint": everything trainable with per-group learning rates
    """
    if stage == "warmup":
        for p in model.encoder.parameters():
            p.requires_grad = False
        model.encoder.eval()
        for p in model.adapter.parameters():
            p.requires_grad = True
        for p in model.t5.parameters():
            p.requires_grad = True
        if model.use_attention_pooling:
            for p in model.attention_pool.parameters():
                p.requires_grad = True

        trainable_params = (
            list(model.adapter.parameters())
            + list(model.t5.parameters())
        )
        if model.use_attention_pooling:
            trainable_params += list(model.attention_pool.parameters())
        if model.use_ctc_head:
            trainable_params += list(model.ctc_head.parameters())

        return create_optimizer(
            model,
            param_groups=[{"params": trainable_params, "lr": cfg.lr_adapter}],
            weight_decay=cfg.weight_decay,
        )

    elif stage == "joint":
        for p in model.encoder.parameters():
            p.requires_grad = True
        model.encoder.train()

        param_groups = [
            {"params": model.encoder.parameters(), "lr": cfg.lr_encoder},
            {"params": model.adapter.parameters(), "lr": cfg.lr_adapter},
            {"params": model.t5.parameters(), "lr": cfg.lr_t5},
        ]
        if model.use_attention_pooling:
            param_groups.append(
                {"params": model.attention_pool.parameters(), "lr": cfg.lr_adapter}
            )
        if model.use_ctc_head:
            param_groups.append(
                {"params": model.ctc_head.parameters(), "lr": cfg.lr_adapter}
            )

        return create_optimizer(
            model, param_groups=param_groups, weight_decay=cfg.weight_decay
        )

    else:
        raise ValueError(f"Unknown stage: {stage}")


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------
def train_phase2(cfg: Phase2Config):
    device = get_device()
    set_seed(cfg.seed)

    logger.info(f"Device: {device}")
    logger.info(f"Run ID: {cfg.run_id}")
    logger.info(f"Config: {cfg}")

    if cfg.smoke_test:
        cfg.max_samples = 4
        cfg.val_max_samples = 2
        cfg.epochs = 1
        cfg.batch_size = 2
        logger.info("🔥 SMOKE TEST MODE — minimal data")

    # --- Tokenizer ---
    tokenizer = T5Tokenizer.from_pretrained(cfg.t5_name)

    # --- Data ---
    train_loader = create_dataloader(
        split=cfg.split,
        batch_size=cfg.batch_size,
        max_samples=cfg.max_samples,
        repo_id=cfg.dataset_repo,
        tokenizer=tokenizer,
        max_target_length=cfg.max_target_length,
        phase=2,
        shuffle_buffer=cfg.shuffle_buffer,
        streaming=cfg.streaming,
    )
    val_loader = create_dataloader(
        split="validation",
        batch_size=cfg.batch_size,
        max_samples=cfg.val_max_samples,
        repo_id=cfg.dataset_repo,
        tokenizer=tokenizer,
        max_target_length=cfg.max_target_length,
        phase=2,
        shuffle_buffer=cfg.shuffle_buffer,
        streaming=cfg.streaming,
    )

    # --- Model ---
    encoder = MultiStreamSemanticEncoder(
        d_model=cfg.d_model,
        latent_dim=cfg.latent_dim,
        num_layers=cfg.encoder_layers,
        nhead=cfg.encoder_heads,
        dropout=cfg.encoder_dropout,
        use_part_embeddings=cfg.use_part_embeddings,
    ).to(device)

    # Load Phase 1 encoder weights
    if os.path.exists(cfg.phase1_ckpt):
        load_encoder_from_phase1(encoder, cfg.phase1_ckpt, device=device)
        logger.info("Phase 1 encoder weights loaded ✓")
    else:
        logger.warning(
            f"Phase 1 checkpoint not found at {cfg.phase1_ckpt}. "
            "Training encoder from scratch."
        )

    model = SignToTextModel(
        encoder=encoder,
        latent_dim=cfg.latent_dim,
        t5_name=cfg.t5_name,
        t5_dim=cfg.t5_dim,
        adapter_dropout=cfg.adapter_dropout,
        use_attention_pooling=cfg.use_attention_pooling,
        pool_num_heads=cfg.pool_num_heads,
        use_ctc_head=cfg.use_ctc_head,
        ctc_vocab_size=cfg.ctc_vocab_size,
    ).to(device)

    logger.info(f"Total params: {count_parameters(model):,}")
    logger.info(f"Encoder params: {count_parameters(model.encoder):,}")
    logger.info(f"Adapter params: {count_parameters(model.adapter):,}")

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
    best_bleu = -1.0
    global_step = 0

    for epoch in range(cfg.epochs):
        stage = "warmup" if epoch < cfg.warmup_epochs else "joint"
        optimizer = setup_for_stage(model, stage, cfg)

        estimated_steps = (cfg.max_samples or 1000) // cfg.batch_size
        scheduler = create_scheduler(
            optimizer,
            scheduler_type=cfg.scheduler,
            total_steps=estimated_steps,
            warmup_steps=cfg.warmup_steps,
        )

        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch+1}/{cfg.epochs} [{stage.upper()}]")
        logger.info(f"{'='*60}")

        # === TRAIN ===
        model.train()
        if stage == "warmup":
            model.encoder.eval()

        total_loss = 0.0
        num_batches = 0
        accum_loss = 0.0

        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Train E{epoch+1}")):
            features = {k: v.to(device) for k, v in batch["features"].items()}
            pad_mask = batch["padding_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast("cuda", enabled=cfg.mixed_precision):
                output = model(features, labels=labels, padding_mask=pad_mask)

                if model.use_ctc_head:
                    t5_output, ctc_log_probs = output
                    loss = t5_output.loss

                    # CTC loss with pseudo-targets (self-supervised alignment)
                    T_ctc = ctc_log_probs.size(0)
                    B = ctc_log_probs.size(1)
                    input_lengths = torch.full((B,), T_ctc, dtype=torch.long, device=device)
                    # Generate pseudo targets from quantized latent
                    with torch.no_grad():
                        z = model.encoder(features, src_key_padding_mask=pad_mask)
                        pseudo_targets = z.mean(dim=-1).clamp(-3, 3)
                        pseudo_targets = ((pseudo_targets + 3) / 6 * (cfg.ctc_vocab_size - 1)).long()
                        target_lengths = (batch["seq_lens"].float() / 4).clamp(min=1).long().to(device)
                        # Trim targets to match target_lengths
                        max_tgt = target_lengths.max().item()
                        pseudo_targets = pseudo_targets[:, :max_tgt]

                    ctc_loss = torch.nn.functional.ctc_loss(
                        ctc_log_probs, pseudo_targets,
                        input_lengths, target_lengths,
                        blank=cfg.ctc_vocab_size, zero_infinity=True,
                    )
                    loss = loss + cfg.ctc_weight * ctc_loss
                else:
                    loss = output.loss

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

                if global_step % cfg.log_every_n_steps == 0:
                    lr = optimizer.param_groups[0]["lr"]
                    train_logger.log(
                        {"train/loss": accum_loss * cfg.gradient_accumulation_steps, "train/lr": lr},
                        step=global_step,
                    )
                accum_loss = 0.0

            total_loss += loss.item() * cfg.gradient_accumulation_steps
            num_batches += 1

        avg_train_loss = total_loss / max(num_batches, 1)

        # === VALIDATION ===
        model.eval()
        val_loss_total = 0.0
        val_batches = 0
        all_preds = []
        all_refs = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Val E{epoch+1}"):
                features = {k: v.to(device) for k, v in batch["features"].items()}
                pad_mask = batch["padding_mask"].to(device)
                labels = batch["labels"].to(device)
                sentences = batch["sentences"]

                output = model(features, labels=labels, padding_mask=pad_mask)
                if model.use_ctc_head:
                    t5_output, _ = output
                    val_loss_total += t5_output.loss.item()
                else:
                    val_loss_total += output.loss.item()
                val_batches += 1

                # Generation
                gen_ids = model.generate(
                    features,
                    padding_mask=pad_mask,
                    max_new_tokens=cfg.max_new_tokens,
                    num_beams=cfg.num_beams,
                )
                preds = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
                all_preds.extend(preds)
                all_refs.extend(sentences)

        avg_val_loss = val_loss_total / max(val_batches, 1)

        # Metrics
        metrics = compute_metrics(all_preds, all_refs)
        metrics["val_loss"] = avg_val_loss
        metrics["train_loss"] = avg_train_loss

        logger.info(
            f"Epoch {epoch+1}/{cfg.epochs} | "
            f"train_loss={avg_train_loss:.4f} | "
            f"val_loss={avg_val_loss:.4f} | "
            f"BLEU={metrics['bleu']:.2f} | "
            f"ROUGE-L={metrics['rouge_l']:.2f} | "
            f"chrF={metrics['chrf']:.2f} | "
            f"EM={metrics['exact_match']:.1f}%"
        )

        print_sample_predictions(all_preds, all_refs, n=3)

        # Log metrics
        epoch_metrics = {
            f"val/{k}": v for k, v in metrics.items()
        }
        epoch_metrics["train/loss"] = avg_train_loss
        train_logger.log(epoch_metrics, step=global_step)
        save_epoch_metrics(cfg.ckpt_dir, epoch, {**metrics, "epoch": epoch + 1})

        # Save latest
        save_checkpoint(
            ckpt_dir=cfg.ckpt_dir,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            step=global_step,
            metrics=metrics,
            config=cfg,
            tokenizer=tokenizer,
            tag="latest",
        )

        # Save best by BLEU
        if metrics["bleu"] > best_bleu:
            best_bleu = metrics["bleu"]
            save_checkpoint(
                ckpt_dir=cfg.ckpt_dir,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                step=global_step,
                metrics=metrics,
                config=cfg,
                tokenizer=tokenizer,
                tag="best",
            )
            logger.info(f"  ✓ New best model (BLEU={best_bleu:.2f})")

    train_logger.close()
    logger.info("Phase 2 training complete.")

    # Smoke test
    if cfg.smoke_test:
        logger.info("Running smoke test assertions...")
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                features = {k: v.to(device) for k, v in batch["features"].items()}
                pad_mask = batch["padding_mask"].to(device)
                labels = batch["labels"].to(device)

                output = model(features, labels=labels, padding_mask=pad_mask)
                if model.use_ctc_head:
                    t5_out, ctc_probs = output
                    assert t5_out.loss is not None
                    assert ctc_probs.dim() == 3
                    logger.info(f"  ✓ CTC output shape: {ctc_probs.shape}")
                else:
                    assert output.loss is not None
                    logger.info(f"  ✓ T5 loss: {output.loss.item():.4f}")

                gen = model.generate(features, padding_mask=pad_mask, max_new_tokens=10)
                assert gen.dim() == 2
                logger.info(f"  ✓ Generation shape: {gen.shape}")
                break
        logger.info("All smoke test assertions passed ✓")


if __name__ == "__main__":
    cfg = parse_phase2_args()
    train_phase2(cfg)
