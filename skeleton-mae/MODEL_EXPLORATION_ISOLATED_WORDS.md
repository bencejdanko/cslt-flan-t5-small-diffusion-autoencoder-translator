# Architecture Exploration: Landmark-Based Isolated ASL Word Recognition

## Context

**Why this change.** Two landmark-based architectures already exist in this repo:
- A multi-stream Transformer + DDPM denoiser + FLAN-T5 pipeline ([models.py](models.py))
- A Conformer + reconstruction + FLAN-T5 variant ([modal_motion_conformer_app.py](modal_motion_conformer_app.py))

Both target full sign→English translation on How2Sign and produce near-zero BLEU (≈0.007). The Phase-1 utterance-word classifier reaches micro-F1 0.116 / hit@5 0.308 — usable but weak. We are pivoting to **isolated ASL word recognition** (single clip → single word), **staying on MediaPipe landmarks** (the existing 540-dim feature pipeline), and **staying within a single-GPU budget**. Isolated SLR is a much better-scoped problem with a strong recent literature: SOTA on WLASL2000 is ~62–65 % top-1 ([LA-Sign 2025](https://arxiv.org/html/2603.29057)) and >93 % on WLASL-100 ([DSLNet](https://arxiv.org/abs/2509.08661)), so there is real headroom.

This document is an **exploration menu**, not an implementation spec. It lays out a wide set of architecture families so we can pick one or two to prototype.

---

## What we keep (existing reusable pieces)

- **Feature pipeline** in [data.py](data.py) and [config.py](config.py): 90 keypoints × 6 features (x,y,z,dx,dy,dz) = 540 / frame, with shoulder/face/wrist normalization. All options below consume this format unchanged.
- **Per-part split** already encoded in [config.py](config.py): pose (33 kpts), face (15), left hand (21), right hand (21). Reused by every option.
- **Modal training harness** ([modal_motion_conformer_app.py](modal_motion_conformer_app.py)) — Modal app, optimizer, eval loop, W&B logging. Reuse, swap the model class.
- **Metrics** in [utils.py](utils.py): word classification metrics (micro-P/R/F1, P@5, R@5, hit@5) — directly applicable to isolated-word top-K accuracy.

---

## Architecture options (grouped by family)

### Family A — Skeleton Graph Convolutional Networks
*Strongest published track record on landmark-only isolated SLR. Treats the 90 keypoints as a graph over time.*

**A1. CTR-GCN** ([Channel-wise Topology Refinement GCN](https://github.com/firework8/Awesome-Skeleton-based-Action-Recognition))
Learns a shared graph topology and refines it per channel. Strong baseline; ~92 % on NTU-60 X-Sub. Small (~1.5 M params), single-GPU friendly. **Best starter option.**

**A2. HD-GCN** (Hierarchically Decomposed GCN, AAAI 2023)
Decomposes the skeleton into hierarchical edge sets to capture both adjacent and semantically distant joints (e.g. fingertip ↔ mouth). Slightly heavier than CTR-GCN, often +1–2 % on hard classes.

**A3. DSLNet — dual-stream spatio-temporal dynamic GCN** ([arXiv 2509.08661](https://arxiv.org/abs/2509.08661))
Decouples gesture *morphology* (handshape, in a body-relative frame) from *trajectory* (in a world frame) into two streams. SOTA on WLASL-100/300 with fewer params than CTR-GCN. Naturally fits our existing per-part normalization.

**A4. Multi-stream ensemble (joint, bone, motion)**
Standard "free lunch" trick from skeleton-action-recognition: train the same backbone on (a) joint coordinates, (b) bone vectors (parent-relative), (c) joint motion (already in our `dx,dy,dz`). Late-fuse softmax. Typically +2–4 % over single stream.

### Family B — Pose-as-Image / Pseudo-Volumetric
*Render landmarks as image-like tensors and use efficient 2D/3D CNNs.*

**B1. PoseConv3D** (ECCV 2022)
Rasterize each frame's joints into a 2D heatmap stack, then run a 3D CNN (SlowOnly variant). Avoids graph-design hassle; surprisingly strong on isolated SLR. Reuses standard video-CNN tooling.

**B2. Tree-Structure Skeleton Image** ([CVPRW 2023](https://openaccess.thecvf.com/content/CVPR2023W/LatinX/papers/Laines_Isolated_Sign_Language_Recognition_Based_on_Tree_Structure_Skeleton_Images_CVPRW_2023_paper.pdf))
Lay joints on a 2D image grid following kinematic tree, channels = (x,y,z,dx,dy,dz). One forward pass through a small ResNet/EfficientNet. Cheapest option to train.

### Family C — Pure Transformer over Keypoints
*Closest to current Conformer; isolates whether the issue was task framing vs architecture.*

**C1. Spatial–Temporal Transformer (ST-Transformer)**
Two-stage attention: spatial attention over 90 keypoints per frame, then temporal attention over frames. Cleaner than current single-stream Transformer; standard in skeleton literature.

**C2. SignBart** ([arXiv 2506.21592](https://arxiv.org/html/2506.21592))
BART-style encoder–decoder over skeleton sequences with denoising-style pretraining; reports 96 % on LSA-64 and good WLASL transfer. Maps cleanly onto our existing Phase-1 reconstruction objective.

**C3. LA-Sign** ([arXiv 2603.29057](https://arxiv.org/html/2603.29057), 2025 SOTA on WLASL)
Looped Transformer with geometry-aware alignment — recurrent latent refinement on top of a transformer encoder. Heavier; only worth attempting after a simpler baseline runs.

### Family D — Self-Supervised Pretraining (paired with any backbone above)
*Reuses our denoising Phase-1 idea, but with stronger modern recipes.*

**D1. SignBERT+** ([TPAMI 2023, arXiv 2305.04868](https://arxiv.org/abs/2305.04868))
Multi-level masked modeling (joint / frame / clip) with hand-model-aware prior. The hand-prior adds MANO-style anatomical constraints. Best-published self-supervised approach for isolated SLR; +7.87 % over RGB on WLASL-2000.

**D2. SkeletonMAE / MaskedPose**
Generic masked-autoencoder pretraining on landmark tubes — random joint+time masking, MSE reconstruction. Simpler than SignBERT+; near-equivalent results in skeleton-action papers. Drop-in for any encoder in Family A/C.

**D3. Contrastive (3s-AimCLR style)**
Augmentation-based contrastive pretraining (joint/bone/motion views). Pairs well with multi-stream A4. Only worth it if labeled data is scarce relative to unlabeled.

### Family E — Hand-Centric Specialization
*Hands carry most of the lexical signal; explicit hand emphasis often dominates ablations.*

**E1. Two-tower hand encoder + body context**
Dedicated heavier encoder for the 42 hand keypoints (LH+RH, possibly with bone+motion streams), lighter encoder for pose+face, cross-attention fusion. Particularly suited because our 90-kpt setup already heavily downsamples face (468→15) — hands are already the dominant signal.

**E2. MANO-prior hand modeling**
Project the 21 MediaPipe hand keypoints onto the MANO hand model (45-DoF pose + 10-DoF shape) and feed those parameters as features. Anatomically constrained; reduces noise from MediaPipe jitter.

---

## Comparison at a glance

| # | Family | Approx params | Train cost (single GPU) | Risk | Expected gain vs current Conformer |
|---|--------|---------------|--------------------------|------|------------------------------------|
| A1 CTR-GCN | Skeleton GCN | ~1.5 M | Low (hours) | Low | Baseline lift, well-trodden |
| A3 DSLNet | Skeleton GCN | ~2 M | Low–medium | Low | Likely best at WLASL-100 scale |
| A4 multi-stream ensemble | + any A | ×2–3 | Medium (train 2–3 models) | Low | +2–4 % over single stream |
| B1 PoseConv3D | Pose-as-image | ~3 M | Medium | Medium | Competitive, easy to debug |
| B2 Tree-Image ResNet | Pose-as-image | <2 M | Lowest | Medium | Cheap baseline; ceiling unclear |
| C2 SignBart | Transformer + pretraining | ~10 M | Medium | Medium | Closest to existing code; reuses Phase 1 |
| D1 SignBERT+ pretrain → A1/A3 | SSL + GCN | depends | High (pretrain + finetune) | Medium-high | Highest published gains |
| E1 Hand-emphasis dual-tower | Specialization | ~2 M | Low | Low | Likely cheap +1–3 % on top of A1 |

---

## Recommended preliminary shortlist

Based on the constraints (single GPU, landmarks, isolated word target):

1. **Track 1 — Fast baseline**: **A1 CTR-GCN** (or A3 DSLNet) trained from scratch on a labeled isolated-word dataset. Establishes a strong reference number quickly.
2. **Track 2 — Pretraining win**: **D2 SkeletonMAE pretraining → A1/A3 fine-tune**, reusing the existing How2Sign landmark stream as unlabeled data. Natural successor to the current Phase-1 denoising idea, with a much stronger recipe.
3. **Track 3 — Ensemble for the leaderboard run**: once Tracks 1–2 are stable, add **A4 multi-stream (joint/bone/motion)** and optionally **E1 hand-emphasis tower** as the final accuracy push.

C2 SignBart is a reasonable Track-1 alternative if we'd rather stay in pure-Transformer land and reuse more of the current Conformer code.

---

## Dataset implications (decision needed)

The current dataset (`bdanko/how2sign-landmarks-front-raw-parquet`) is **continuous translation data**, not labeled isolated word clips. Options:

- **D-1. Switch to WLASL-2000** — the standard isolated-SLR benchmark; would re-extract MediaPipe landmarks once. Best comparability with literature.
- **D-2. Use ASL-Citizen or MS-ASL** — newer / larger; ASL-Citizen has cleaner labels.
- **D-3. Word-segment How2Sign** — derive isolated clips from utterance-level word labels via forced alignment. Reuses existing data but introduces alignment noise.

Recommendation: **D-1 (WLASL-2000)** for the labeled phase, with How2Sign reused as **unlabeled pretraining data** for Track 2.

---

## Critical files to (eventually) modify

- [models.py](models.py) — add new model class(es) (e.g. `CTRGCN`, `SkeletonMAE`).
- [data.py](data.py) — add a WLASL/ASL-Citizen dataset adapter that produces the same 540-dim feature tensor format. Reuse existing per-part normalization.
- [config.py](config.py) — config block per architecture; `BODY_PART_SPECS` already there.
- [modal_motion_conformer_app.py](modal_motion_conformer_app.py) — clone into `modal_<arch>_app.py` per track; swap model + dataset; reuse optimizer/eval loop.
- [utils.py](utils.py) — extend metrics with top-1 / top-5 / mean-class accuracy for isolated SLR (current word-level metrics are most of the way there).
- [MODEL_ARCHITECTURES.md](MODEL_ARCHITECTURES.md) — add a third section for the chosen new architecture.

---

## Verification plan

For each prototyped architecture:

1. **Sanity-overfit**: train on 32 clips, confirm >90 % training accuracy in <500 steps. Catches plumbing bugs before any real training.
2. **Held-out top-1 / top-5** on the chosen dataset's standard split (WLASL-2000 cross-instance / cross-class).
3. **Ablation**: compare to A1 CTR-GCN baseline on the same data + features. Any new architecture must beat that or be cheaper at equivalent accuracy.
4. **Compute report**: wall-clock train time and peak VRAM on a single GPU, logged to W&B alongside accuracy.
5. **Qualitative confusion matrix**: top-20 most-confused word pairs — surfaces handshape vs trajectory failure modes and informs whether to add E1/E2.

---

## Out of scope (deferred)

- Raw-video / hybrid backbones (decision: landmarks-only).
- Continuous SLR / full SLT (decision: isolated word target).
- Multi-GPU foundation-model fine-tunes (compute budget).
- Cross-lingual / cross-view robustness (literature addresses this but not a Track-1 concern).
