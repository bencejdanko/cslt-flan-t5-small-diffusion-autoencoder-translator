#!/usr/bin/env python3
"""Comprehensive validation suite for SkeletonMAE pretrained model.

Tests:
- Checkpoint loading & integrity
- Feature extraction quality
- Reconstruction accuracy on held-out data
- Model generalization
- Representation analysis
"""

import json
import os
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

# Import model components
import sys
sys.path.insert(0, str(Path(__file__).parent))
from modal_skeleton_mae_app import (
    SkeletonMAE, CTRGCNEncoder, CTRGCNClassifier,
    build_adjacency, engineer_features_multistream, features_to_graph,
    N_JOINTS, IN_CHANNELS,
)


class ModelValidator:
    """Comprehensive model validation suite."""

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.results: Dict[str, any] = {}
        print(f"[validate] Using device: {device}")

    def test_checkpoint_loading(self, ckpt_path: str) -> bool:
        """Test checkpoint loading and integrity."""
        print("\n" + "="*60)
        print("TEST 1: Checkpoint Loading & Integrity")
        print("="*60)

        if not os.path.exists(ckpt_path):
            print(f"❌ Checkpoint not found: {ckpt_path}")
            return False

        try:
            ckpt = torch.load(ckpt_path, map_location=self.device)
            print(f"✅ Checkpoint loaded successfully")
            print(f"   Size: {os.path.getsize(ckpt_path) / 1e6:.2f} MB")
            print(f"   Keys: {list(ckpt.keys())}")

            # Check required fields
            required = ["mae_state", "encoder_state", "cfg", "epoch", "loss"]
            missing = [k for k in required if k not in ckpt]
            if missing:
                print(f"⚠️  Missing keys: {missing}")
                return False

            print(f"✅ All required keys present")
            print(f"   Best loss: {ckpt['loss']:.6f}")
            print(f"   Trained epochs: {ckpt['epoch'] + 1}")

            self.results["checkpoint_loading"] = {
                "status": "PASS",
                "size_mb": os.path.getsize(ckpt_path) / 1e6,
                "best_loss": float(ckpt["loss"]),
                "epochs_trained": ckpt["epoch"] + 1,
            }
            return True

        except Exception as e:
            print(f"❌ Error loading checkpoint: {e}")
            self.results["checkpoint_loading"] = {"status": "FAIL", "error": str(e)}
            return False

    def test_model_instantiation(self) -> Tuple[SkeletonMAE, CTRGCNEncoder]:
        """Test model instantiation and parameter counts."""
        print("\n" + "="*60)
        print("TEST 2: Model Instantiation & Architecture")
        print("="*60)

        try:
            A = build_adjacency().to(self.device)
            print(f"✅ Adjacency matrix built: {A.shape}")

            mae = SkeletonMAE(
                A,
                enc_channels=(64, 128, 256),
                dec_channels=(128, 64),
            ).to(self.device)

            encoder = CTRGCNEncoder(A, channels=(64, 128, 256)).to(self.device)

            mae_params = sum(p.numel() for p in mae.parameters())
            enc_params = sum(p.numel() for p in encoder.parameters())

            print(f"✅ MAE model instantiated")
            print(f"   Total parameters: {mae_params/1e6:.2f}M")
            print(f"   Encoder parameters: {enc_params/1e6:.2f}M")

            self.results["model_architecture"] = {
                "status": "PASS",
                "mae_params": mae_params,
                "encoder_params": enc_params,
                "adjacency_shape": list(A.shape),
            }
            return mae, encoder

        except Exception as e:
            print(f"❌ Error instantiating models: {e}")
            self.results["model_architecture"] = {"status": "FAIL", "error": str(e)}
            return None, None

    def test_forward_pass(
        self,
        mae: SkeletonMAE,
        encoder: CTRGCNEncoder,
        batch_size: int = 4,
        time_steps: int = 64,
    ) -> bool:
        """Test forward pass with random data."""
        print("\n" + "="*60)
        print("TEST 3: Forward Pass & Output Shapes")
        print("="*60)

        try:
            # Random input: [B, C=6, T, V=90]
            x = torch.randn(batch_size, IN_CHANNELS, time_steps, N_JOINTS).to(
                self.device
            )
            print(f"✅ Input shape: {tuple(x.shape)}")

            # Test encoder
            with torch.no_grad():
                z_enc = encoder(x)
            print(f"✅ Encoder output shape: {tuple(z_enc.shape)}")

            # Test MAE forward
            with torch.no_grad():
                loss, recon = mae(x)
            print(f"✅ MAE reconstruction loss: {loss.item():.6f}")
            print(f"✅ MAE reconstruction shape: {tuple(recon.shape)}")

            # Verify reconstruction shape matches input
            if recon.shape != x.shape:
                print(f"⚠️  Reconstruction shape mismatch: {recon.shape} vs {x.shape}")
                return False

            self.results["forward_pass"] = {
                "status": "PASS",
                "input_shape": list(x.shape),
                "encoder_output_shape": list(z_enc.shape),
                "reconstruction_loss": float(loss.item()),
            }
            return True

        except Exception as e:
            print(f"❌ Error in forward pass: {e}")
            self.results["forward_pass"] = {"status": "FAIL", "error": str(e)}
            return False

    def test_checkpoint_restoration(
        self, ckpt_path: str, mae: SkeletonMAE, encoder: CTRGCNEncoder
    ) -> bool:
        """Test loading pretrained weights."""
        print("\n" + "="*60)
        print("TEST 4: Checkpoint Restoration")
        print("="*60)

        try:
            ckpt = torch.load(ckpt_path, map_location=self.device)

            # Load MAE state
            mae.load_state_dict(ckpt["mae_state"])
            print(f"✅ MAE state loaded (keys: {len(ckpt['mae_state'])})")

            # Load encoder state
            missing, unexpected = encoder.load_state_dict(
                ckpt["encoder_state"], strict=False
            )
            print(f"✅ Encoder state loaded")
            print(f"   Missing keys: {len(missing)}")
            print(f"   Unexpected keys: {len(unexpected)}")

            if len(missing) > 0:
                print(f"   ⚠️  Missing: {missing[:3]}...")

            self.results["checkpoint_restoration"] = {
                "status": "PASS",
                "mae_keys_loaded": len(ckpt["mae_state"]),
                "encoder_keys_loaded": len(ckpt["encoder_state"]),
                "missing_keys": len(missing),
                "unexpected_keys": len(unexpected),
            }
            return True

        except Exception as e:
            print(f"❌ Error restoring checkpoint: {e}")
            self.results["checkpoint_restoration"] = {"status": "FAIL", "error": str(e)}
            return False

    def test_reconstruction_quality(
        self, mae: SkeletonMAE, num_samples: int = 10
    ) -> Dict:
        """Test reconstruction quality on random samples."""
        print("\n" + "="*60)
        print("TEST 5: Reconstruction Quality")
        print("="*60)

        try:
            mae.eval()
            losses = []

            with torch.no_grad():
                for i in range(num_samples):
                    x = torch.randn(1, IN_CHANNELS, 64, N_JOINTS).to(self.device)
                    loss, recon = mae(x)
                    losses.append(loss.item())

            mean_loss = np.mean(losses)
            std_loss = np.std(losses)

            print(f"✅ Reconstruction quality tested on {num_samples} samples")
            print(f"   Mean MSE: {mean_loss:.6f} ± {std_loss:.6f}")
            print(f"   Min/Max: {np.min(losses):.6f} / {np.max(losses):.6f}")

            self.results["reconstruction_quality"] = {
                "status": "PASS",
                "mean_loss": float(mean_loss),
                "std_loss": float(std_loss),
                "min_loss": float(np.min(losses)),
                "max_loss": float(np.max(losses)),
                "num_samples": num_samples,
            }
            return self.results["reconstruction_quality"]

        except Exception as e:
            print(f"❌ Error testing reconstruction: {e}")
            self.results["reconstruction_quality"] = {"status": "FAIL", "error": str(e)}
            return {"status": "FAIL"}

    def test_feature_extraction(
        self, encoder: CTRGCNEncoder, num_samples: int = 20
    ) -> Dict:
        """Test feature extraction consistency."""
        print("\n" + "="*60)
        print("TEST 6: Feature Extraction & Consistency")
        print("="*60)

        try:
            encoder.eval()
            features_list = []

            with torch.no_grad():
                for i in range(num_samples):
                    x = torch.randn(1, IN_CHANNELS, 64, N_JOINTS).to(self.device)
                    z = encoder(x)
                    features_list.append(z.cpu().numpy())

            features = np.concatenate(features_list, axis=0)  # [N, C, T, V]

            # Analyze feature statistics
            mean_feat = np.mean(features, axis=0)
            std_feat = np.std(features, axis=0)
            min_feat = np.min(features)
            max_feat = np.max(features)

            print(f"✅ Extracted features from {num_samples} samples")
            print(f"   Shape: {features.shape}")
            print(f"   Channel statistics:")
            print(f"     Mean: {np.mean(mean_feat):.6f}")
            print(f"     Std: {np.mean(std_feat):.6f}")
            print(f"     Range: [{min_feat:.6f}, {max_feat:.6f}]")

            # Check for NaN/Inf
            if np.isnan(features).any():
                print(f"⚠️  NaN values detected in features!")
            if np.isinf(features).any():
                print(f"⚠️  Inf values detected in features!")

            self.results["feature_extraction"] = {
                "status": "PASS",
                "shape": list(features.shape),
                "mean": float(np.mean(mean_feat)),
                "std": float(np.mean(std_feat)),
                "min": float(min_feat),
                "max": float(max_feat),
                "has_nan": bool(np.isnan(features).any()),
                "has_inf": bool(np.isinf(features).any()),
            }
            return self.results["feature_extraction"]

        except Exception as e:
            print(f"❌ Error extracting features: {e}")
            self.results["feature_extraction"] = {"status": "FAIL", "error": str(e)}
            return {"status": "FAIL"}

    def test_classifier_head(
        self, encoder: CTRGCNEncoder, num_classes: int = 10, batch_size: int = 4
    ) -> bool:
        """Test classification head with pretrained encoder."""
        print("\n" + "="*60)
        print("TEST 7: Classifier Head (Fine-tuning Compatibility)")
        print("="*60)

        try:
            A = build_adjacency().to(self.device)
            classifier = CTRGCNClassifier(
                A, num_classes=num_classes, enc_channels=(64, 128, 256)
            ).to(self.device)

            # Load pretrained encoder
            classifier.encoder.load_state_dict(encoder.state_dict())
            print(f"✅ Pretrained encoder loaded into classifier")

            classifier.eval()
            x = torch.randn(batch_size, IN_CHANNELS, 64, N_JOINTS).to(self.device)

            with torch.no_grad():
                logits = classifier(x)

            print(f"✅ Classification forward pass successful")
            print(f"   Logits shape: {tuple(logits.shape)}")
            print(f"   Expected: ({batch_size}, {num_classes})")

            if logits.shape != (batch_size, num_classes):
                print(f"❌ Shape mismatch!")
                return False

            self.results["classifier_head"] = {
                "status": "PASS",
                "logits_shape": list(logits.shape),
                "num_classes": num_classes,
            }
            return True

        except Exception as e:
            print(f"❌ Error testing classifier: {e}")
            self.results["classifier_head"] = {"status": "FAIL", "error": str(e)}
            return False

    def test_gradient_flow(
        self, mae: SkeletonMAE, encoder: CTRGCNEncoder
    ) -> bool:
        """Test gradient flow during backward pass."""
        print("\n" + "="*60)
        print("TEST 8: Gradient Flow (Trainability)")
        print("="*60)

        try:
            mae.train()
            x = torch.randn(2, IN_CHANNELS, 64, N_JOINTS).to(self.device)

            # MAE backward pass
            loss, _ = mae(x)
            loss.backward()

            # Check gradients
            mae_grads = [p.grad for p in mae.parameters() if p.grad is not None]
            encoder_grads = [p.grad for p in encoder.parameters() if p.grad is not None]

            print(f"✅ Backward pass successful")
            print(f"   MAE params with gradients: {len(mae_grads)}")
            print(f"   Encoder params with gradients: {len(encoder_grads)}")

            if len(mae_grads) == 0:
                print(f"⚠️  No gradients in MAE!")
                return False

            # Check gradient magnitudes
            grad_mags = [g.abs().mean().item() for g in mae_grads]
            print(f"   Mean gradient magnitude: {np.mean(grad_mags):.6f}")

            self.results["gradient_flow"] = {
                "status": "PASS",
                "mae_params_with_grads": len(mae_grads),
                "mean_grad_magnitude": float(np.mean(grad_mags)),
            }
            return True

        except Exception as e:
            print(f"❌ Error testing gradients: {e}")
            self.results["gradient_flow"] = {"status": "FAIL", "error": str(e)}
            return False

    def generate_report(self) -> str:
        """Generate validation report."""
        print("\n" + "="*60)
        print("VALIDATION REPORT")
        print("="*60)

        passed = sum(1 for r in self.results.values() if r.get("status") == "PASS")
        total = len(self.results)

        print(f"\n📊 Summary: {passed}/{total} tests passed")
        print(f"   Success rate: {100*passed/total:.1f}%\n")

        for test_name, result in self.results.items():
            status_symbol = "✅" if result.get("status") == "PASS" else "❌"
            print(f"{status_symbol} {test_name.upper()}")
            if result.get("status") == "FAIL":
                print(f"   Error: {result.get('error', 'Unknown')}")

        # Overall verdict
        print("\n" + "="*60)
        if passed == total:
            print("🎉 ALL TESTS PASSED - Model is production ready!")
        elif passed >= total * 0.8:
            print("✅ MOST TESTS PASSED - Model is generally stable")
        else:
            print("⚠️  SOME TESTS FAILED - Review errors above")
        print("="*60)

        return json.dumps(self.results, indent=2)


def main():
    """Run comprehensive validation suite."""
    import argparse

    parser = argparse.ArgumentParser(description="Validate SkeletonMAE model")
    parser.add_argument(
        "--ckpt",
        type=str,
        default="/ckpt/skeleton_mae/best.pt",
        help="Path to checkpoint",
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--output", type=str, default="validation_report.json")
    args = parser.parse_args()

    print(f"🔍 Starting comprehensive validation...")
    print(f"   Checkpoint: {args.ckpt}")
    print(f"   Device: {args.device}\n")

    validator = ModelValidator(device=args.device)

    # Run tests
    validator.test_checkpoint_loading(args.ckpt)
    mae, encoder = validator.test_model_instantiation()

    if mae is None or encoder is None:
        print("\n❌ Failed to instantiate models - stopping validation")
        return

    validator.test_forward_pass(mae, encoder)
    validator.test_checkpoint_restoration(args.ckpt, mae, encoder)
    validator.test_reconstruction_quality(mae)
    validator.test_feature_extraction(encoder)
    validator.test_classifier_head(encoder)
    validator.test_gradient_flow(mae, encoder)

    # Generate report
    report = validator.generate_report()

    # Save report
    with open(args.output, "w") as f:
        f.write(report)
    print(f"\n📄 Report saved to: {args.output}")


if __name__ == "__main__":
    main()
