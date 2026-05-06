#!/usr/bin/env python3
"""Modal-based validation testing on actual checkpoints stored in volumes."""

import json
import os
import modal
from pathlib import Path

app = modal.App("asl-skeleton-mae-validation")

# Reuse volumes from main app
ckpt_vol = modal.Volume.from_name("asl-skeleton-mae-checkpoints", create_if_missing=True)

# Use the same image as training
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.3.1",
        "datasets==2.18.0",
        "huggingface_hub==0.21.0",
        "numpy",
        "tqdm",
        "pyarrow==15.0.0",
        "pandas",
    )
)


@app.function(
    image=image,
    volumes={"/ckpt": ckpt_vol},
    gpu="A10G",
)
def validate_checkpoints():
    """Comprehensive validation of pretrained and fine-tuned models."""
    import torch
    import numpy as np
    import sys
    from pathlib import Path

    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = {}

    # Constants (copied from modal_skeleton_mae_app.py)
    N_BODY, N_FACE, N_LHAND, N_RHAND = 33, 15, 21, 21
    OFFSET_BODY, OFFSET_FACE = 0, 33
    OFFSET_LHAND, OFFSET_RHAND = 48, 69
    N_JOINTS = N_BODY + N_FACE + N_LHAND + N_RHAND  # 90
    IN_CHANNELS = 6  # (x, y, z, dx, dy, dz)

    print("="*70)
    print("COMPREHENSIVE MODEL VALIDATION ON MODAL VOLUMES")
    print("="*70)

    # ===== TEST 1: Pretrain Checkpoint =====
    print("\n[TEST 1] Pretrain Checkpoint Integrity")
    print("-" * 70)

    pretrain_ckpt = "/ckpt/skeleton_mae/best.pt"
    if os.path.exists(pretrain_ckpt):
        try:
            ckpt = torch.load(pretrain_ckpt, map_location=device)
            print(f"✅ Pretrain checkpoint loaded")
            print(f"   Size: {os.path.getsize(pretrain_ckpt) / 1e6:.2f} MB")
            print(f"   Keys: {list(ckpt.keys())}")
            print(f"   Best MSE loss: {ckpt['loss']:.6f}")
            print(f"   Trained for {ckpt['epoch'] + 1} epochs")

            # Validate state dict sizes
            mae_state = ckpt["mae_state"]
            enc_state = ckpt["encoder_state"]
            print(f"   MAE state keys: {len(mae_state)}")
            print(f"   Encoder state keys: {len(enc_state)}")

            results["pretrain_checkpoint"] = {
                "status": "PASS",
                "size_mb": os.path.getsize(pretrain_ckpt) / 1e6,
                "loss": float(ckpt["loss"]),
                "epochs": ckpt["epoch"] + 1,
                "mae_state_keys": len(mae_state),
                "encoder_state_keys": len(enc_state),
            }
        except Exception as e:
            print(f"❌ Error: {e}")
            results["pretrain_checkpoint"] = {"status": "FAIL", "error": str(e)}
    else:
        print(f"⚠️  Checkpoint not found: {pretrain_ckpt}")
        results["pretrain_checkpoint"] = {"status": "SKIP", "reason": "Not found"}

    # ===== TEST 2: Fine-tune Checkpoint =====
    print("\n[TEST 2] Fine-tune Checkpoint Integrity")
    print("-" * 70)

    finetune_ckpt = "/ckpt/how2sign_finetune/best.pt"
    if os.path.exists(finetune_ckpt):
        try:
            ckpt = torch.load(finetune_ckpt, map_location=device)
            print(f"✅ Fine-tune checkpoint loaded")
            print(f"   Size: {os.path.getsize(finetune_ckpt) / 1e6:.2f} MB")
            print(f"   Keys: {list(ckpt.keys())}")
            print(f"   Best top-1 accuracy: {ckpt['top1']:.3f}")
            print(f"   Trained for {ckpt['epoch'] + 1} epochs")

            model_state = ckpt["model_state"]
            print(f"   Model state keys: {len(model_state)}")

            results["finetune_checkpoint"] = {
                "status": "PASS",
                "size_mb": os.path.getsize(finetune_ckpt) / 1e6,
                "top1_accuracy": float(ckpt["top1"]),
                "top5_accuracy": float(ckpt["top5"]),
                "epochs": ckpt["epoch"] + 1,
                "model_state_keys": len(model_state),
            }
        except Exception as e:
            print(f"❌ Error: {e}")
            results["finetune_checkpoint"] = {"status": "FAIL", "error": str(e)}
    else:
        print(f"⚠️  Checkpoint not found: {finetune_ckpt}")
        results["finetune_checkpoint"] = {"status": "SKIP", "reason": "Not found"}

    # ===== TEST 3: History Logging =====
    print("\n[TEST 3] Training History Integrity")
    print("-" * 70)

    history_files = [
        ("/ckpt/skeleton_mae/history.jsonl", "pretrain"),
        ("/ckpt/how2sign_finetune/history.jsonl", "finetune"),
    ]

    for hist_path, stage in history_files:
        if os.path.exists(hist_path):
            try:
                with open(hist_path, "r") as f:
                    lines = f.readlines()

                print(f"✅ {stage.upper()} history loaded")
                print(f"   Entries: {len(lines)}")

                # Parse first and last entries
                first = json.loads(lines[0])
                last = json.loads(lines[-1])

                print(f"   First epoch: {first.get('epoch', 'N/A')}")
                print(f"   Last epoch: {last.get('epoch', 'N/A')}")
                print(f"   Keys logged: {list(first.keys())}")

                results[f"{stage}_history"] = {
                    "status": "PASS",
                    "entries": len(lines),
                    "first_epoch": first.get("epoch"),
                    "last_epoch": last.get("epoch"),
                    "keys": list(first.keys()),
                }
            except Exception as e:
                print(f"❌ Error: {e}")
                results[f"{stage}_history"] = {"status": "FAIL", "error": str(e)}
        else:
            print(f"⚠️  History not found: {hist_path}")
            results[f"{stage}_history"] = {"status": "SKIP", "reason": "Not found"}

    # ===== SUMMARY =====
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)

    passed = sum(1 for r in results.values() if r.get("status") == "PASS")
    total = len(results)
    skipped = sum(1 for r in results.values() if r.get("status") == "SKIP")

    print(f"\n📊 Results: {passed}/{total-skipped} tests passed")
    print(f"   Passed:  {passed}")
    print(f"   Failed:  {total - passed - skipped}")
    print(f"   Skipped: {skipped}")
    if total - skipped > 0:
        print(f"   Success rate: {100*passed/(total-skipped):.1f}%\n")
    else:
        print()

    for test_name, result in results.items():
        symbol = "✅" if result.get("status") == "PASS" else "❌" if result.get("status") == "FAIL" else "⏭️"
        print(f"{symbol} {test_name.upper()}")

    # Overall verdict
    print("\n" + "="*70)
    if passed >= (total - skipped) * 0.8 and passed > 0:
        print("🎉 VALIDATION PASSED - Checkpoints are intact!")
    else:
        print("⚠️  VALIDATION INCOMPLETE - Some tests failed")
    print("="*70)

    return results


@app.local_entrypoint()
def main():
    """Run Modal validation."""
    print("\n🚀 Starting Modal validation job...\n")
    results = validate_checkpoints.remote()

    # Save results
    report_path = "modal_validation_report.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n📄 Report saved to: {report_path}")
    print("\nValidation complete!")
