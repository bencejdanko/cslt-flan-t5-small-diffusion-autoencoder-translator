# Track 2 Implementation Plan — SkeletonMAE Pretrain → CTR-GCN Fine-tune on WLASL-100/300

> Architectural rationale lives in [MODEL_EXPLORATION_ISOLATED_WORDS.md](MODEL_EXPLORATION_ISOLATED_WORDS.md). This file is the build doc.

## Decisions locked

- **Backbone**: CTR-GCN (single-partition variant for readability; learnable per-channel topology refinement).
- **Pretraining**: SkeletonMAE — random (joint × time) tube masking, MSE reconstruction. Data: existing How2Sign landmarks (unlabeled).
- **Fine-tune**: WLASL-100 *and* WLASL-300 (same trained weights, two heads / two runs).
- **Layout**: one self-contained file `modal_skeleton_mae_app.py` (mirrors [modal_motion_conformer_app.py](modal_motion_conformer_app.py) style).
- **Input**: existing 540-dim features from [data.py](data.py) reshaped to graph format `[B, C=6, T, V=90]`. No re-engineering.
- **Per-part normalization preserved**, compensated by a learned `part_embedding[V]` added at input.

## Joint graph (V = 90)

```
indices  0..32   body  (33, MediaPipe pose)
indices 33..47   face  (15, downsampled per FACE_LANDMARK_IDXS in config.py)
indices 48..68   lhand (21, MediaPipe hand)
indices 69..89   rhand (21, MediaPipe hand)
```

Cross-part bridges (so the graph is connected): `pose_left_wrist (15) ↔ lhand_wrist (48)`, `pose_right_wrist (16) ↔ rhand_wrist (69)`, `pose_nose (0) ↔ face_nose_tip (41)`.

---

## Module skeleton (file: `modal_skeleton_mae_app.py`)

### 1. Modal app + configs

```python
from __future__ import annotations
import json, os, random
from dataclasses import dataclass, field
from typing import Optional, List
import modal

APP_NAME = "asl-skeleton-mae"
HF_SECRET = "huggingface-secret"
CKPT_VOLUME = "asl-skeleton-mae-checkpoints"
WLASL_VOLUME = "asl-wlasl-landmarks"

app = modal.App(APP_NAME)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libgl1")
    .pip_install(
        "torch", "datasets", "huggingface_hub", "transformers",
        "numpy", "tqdm", "pyarrow", "pandas",
        "mediapipe==0.10.14", "opencv-python-headless", "yt-dlp",
    )
)
ckpt_vol = modal.Volume.from_name(CKPT_VOLUME, create_if_missing=True)
wlasl_vol = modal.Volume.from_name(WLASL_VOLUME, create_if_missing=True)

@dataclass
class PretrainConfig:
    landmarks_repo: str = "bdanko/how2sign-landmarks-front-raw-parquet"
    split: str = "train"
    max_samples: Optional[int] = None
    window_size: int = 64        # frames per pretrain clip
    stride: int = 32
    batch_size: int = 32
    epochs: int = 30
    lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    mask_ratio_joints: float = 0.4   # fraction of joints masked per clip
    mask_ratio_time: float = 0.3     # fraction of time steps masked per clip
    encoder_channels: tuple = (64, 128, 256)
    decoder_channels: tuple = (128, 64)
    ckpt_dir: str = "/ckpt/skeleton_mae"
    seed: int = 15179996

@dataclass
class FinetuneConfig:
    wlasl_parquet: str = "/wlasl/wlasl100.parquet"   # produced by extract_wlasl
    num_classes: int = 100                # 100 or 300
    window_size: int = 64
    batch_size: int = 32
    epochs: int = 80
    lr_encoder: float = 1e-4              # smaller — pretrained
    lr_head: float = 1e-3
    weight_decay: float = 1e-4
    label_smoothing: float = 0.1
    pretrained_ckpt: str = "/ckpt/skeleton_mae/best.pt"
    ckpt_dir: str = "/ckpt/wlasl100"
    seed: int = 15179996
```

### 2. WLASL landmark extraction (Modal function)

Downloads videos listed in `WLASL_v0.3.json`, runs MediaPipe Holistic, applies the same feature engineering as `engineer_features_multistream`, writes a parquet matching the existing schema (`features` bytes, `shape`, `label`, `split`).

```python
@app.function(image=image, volumes={"/wlasl": wlasl_vol}, timeout=60*60*6, cpu=4)
def extract_wlasl(subset: int = 100, wlasl_json_url: str = "..."):
    """Run once. Subset = 100 or 300 → produces wlasl{subset}.parquet on volume."""
    import urllib.request, subprocess, cv2, numpy as np, pandas as pd, mediapipe as mp
    from data import engineer_features_multistream  # reused

    # 1. Load WLASL_v0.3.json (label list + per-instance YouTube URL + start/end frame)
    meta = json.loads(urllib.request.urlopen(wlasl_json_url).read())
    classes = sorted({entry["gloss"] for entry in meta})[:subset]
    label2id = {g: i for i, g in enumerate(classes)}

    holistic = mp.solutions.holistic.Holistic(static_image_mode=False, model_complexity=1)
    rows = []
    for entry in meta:
        if entry["gloss"] not in label2id: continue
        for inst in entry["instances"]:
            video_path = _download_clip(inst["url"], inst["frame_start"], inst["frame_end"])
            if video_path is None: continue
            raw = _extract_holistic(video_path, holistic)   # [T, 543, 3]
            if raw is None or raw.shape[0] < 8: continue
            rows.append({
                "features": raw.astype(np.float32).tobytes(),
                "shape": list(raw.shape),
                "label": label2id[entry["gloss"]],
                "gloss": entry["gloss"],
                "split": inst["split"],   # "train"|"val"|"test"
            })
    pd.DataFrame(rows).to_parquet(f"/wlasl/wlasl{subset}.parquet")
    wlasl_vol.commit()
```

Helpers (`_download_clip` uses `yt-dlp` + `ffmpeg` to grab the frame range; `_extract_holistic` runs MediaPipe and stacks `[pose33 + face468 + lhand21 + rhand21, 3]` per frame with NaN→0 for missing detections). Both kept short (~30 lines each); failures skip silently.

### 3. Feature → graph reshape

```python
def features_to_graph(feat_dict, T_max: int) -> torch.Tensor:
    """[T, 540] split dict → [C=6, T, V=90]. Pads/truncates to T_max."""
    parts = []
    for pos_key, vel_key, n in [
        ("body_pos",  "body_vel",  33),
        ("face_pos",  "face_vel",  15),
        ("lhand_pos", "lhand_vel", 21),
        ("rhand_pos", "rhand_vel", 21),
    ]:
        pos = feat_dict[pos_key].view(-1, n, 3)        # [T, n, 3]
        vel = feat_dict[vel_key].view(-1, n, 3)
        parts.append(torch.cat([pos, vel], dim=-1))    # [T, n, 6]
    x = torch.cat(parts, dim=1)                         # [T, 90, 6]
    T = x.shape[0]
    if T < T_max: x = F.pad(x, (0, 0, 0, 0, 0, T_max - T))
    else:         x = x[:T_max]
    return x.permute(2, 0, 1).contiguous()              # [6, T_max, 90]
```

### 4. Adjacency builder

```python
def build_adjacency() -> torch.Tensor:
    """Return symmetric, self-looped, normalized A of shape [90, 90]."""
    edges: List[tuple] = []

    # Body: MediaPipe pose connections (subset of 35 edges)
    POSE_EDGES = [(0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),
                  (9,10),(11,12),(11,13),(13,15),(12,14),(14,16),
                  (11,23),(12,24),(23,24),(23,25),(25,27),(27,29),(27,31),
                  (24,26),(26,28),(28,30),(28,32)]
    edges += POSE_EDGES

    # Face (15 nodes, indices 33..47): chain plus center connections
    FACE_LOCAL = [(0,1),(2,3),(4,5),(6,7),(8,9),(10,11),(11,12),(12,13),(13,14)]
    edges += [(33+a, 33+b) for a, b in FACE_LOCAL]

    # Hands: MediaPipe 21-joint connectivity (palm + 5 fingers)
    HAND_EDGES = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
                  (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),
                  (0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17)]
    edges += [(48+a, 48+b) for a, b in HAND_EDGES]
    edges += [(69+a, 69+b) for a, b in HAND_EDGES]

    # Cross-part bridges
    edges += [(15, 48), (16, 69), (0, 41)]

    A = torch.zeros(90, 90)
    for a, b in edges:
        A[a, b] = 1; A[b, a] = 1
    A += torch.eye(90)
    # Symmetric normalization D^-1/2 A D^-1/2
    deg = A.sum(-1).clamp(min=1).rsqrt()
    return deg[:, None] * A * deg[None, :]
```

### 5. CTR-GCN building blocks

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CTRGC(nn.Module):
    """Channel-wise Topology Refinement Graph Conv.
    Adds a per-channel learned offset to a shared adjacency A.
    """
    def __init__(self, in_c: int, out_c: int, A: torch.Tensor, mid_c: int = 8):
        super().__init__()
        self.A = nn.Parameter(A.clone(), requires_grad=True)        # [V,V]
        self.q = nn.Conv2d(in_c, mid_c, 1)
        self.k = nn.Conv2d(in_c, mid_c, 1)
        self.refine = nn.Conv2d(mid_c, out_c, 1)
        self.value = nn.Conv2d(in_c, out_c, 1)
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x):                          # x: [B, C, T, V]
        q = self.q(x).mean(2)                      # [B, mid, V]
        k = self.k(x).mean(2)                      # [B, mid, V]
        diff = q.unsqueeze(-1) - k.unsqueeze(-2)   # [B, mid, V, V]
        delta = torch.tanh(self.refine(diff))      # [B, out, V, V]
        A = self.A[None, None] + self.alpha * delta
        v = self.value(x)                          # [B, out, T, V]
        return torch.einsum("boTv,bovw->boTw", v, A)


class MSTCN(nn.Module):
    """Multi-scale temporal conv: parallel kernels k=3, k=5, plus 1x1."""
    def __init__(self, c: int, dilations=(1, 2)):
        super().__init__()
        branches = []
        for k in (3, 5):
            for d in dilations:
                branches.append(nn.Sequential(
                    nn.Conv2d(c, c // 4, 1), nn.BatchNorm2d(c // 4), nn.ReLU(inplace=True),
                    nn.Conv2d(c // 4, c // 4, (k, 1), padding=(d * (k - 1) // 2, 0),
                              dilation=(d, 1)),
                ))
        self.branches = nn.ModuleList(branches)
        self.skip = nn.Conv2d(c, c // 4, 1)
        self.out = nn.Sequential(nn.BatchNorm2d(c // 4 * (len(branches) + 1)),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(c // 4 * (len(branches) + 1), c, 1),
                                 nn.BatchNorm2d(c))

    def forward(self, x):
        ys = [b(x) for b in self.branches] + [self.skip(x)]
        return self.out(torch.cat(ys, dim=1))


class CTRGCBlock(nn.Module):
    """One layer: CTR-GC → BN → ReLU → MS-TCN, with residual."""
    def __init__(self, in_c, out_c, A, stride=1):
        super().__init__()
        self.gc = CTRGC(in_c, out_c, A)
        self.bn = nn.BatchNorm2d(out_c)
        self.tcn = MSTCN(out_c)
        self.stride = stride
        self.res = (nn.Identity() if (in_c == out_c and stride == 1)
                    else nn.Sequential(nn.Conv2d(in_c, out_c, 1),
                                       nn.BatchNorm2d(out_c)))

    def forward(self, x):
        y = F.relu(self.bn(self.gc(x)), inplace=True)
        y = self.tcn(y)
        if self.stride > 1:
            y = y[:, :, ::self.stride]                 # temporal downsample
            x = x[:, :, ::self.stride]
        return F.relu(y + self.res(x), inplace=True)
```

### 6. Encoder

```python
class CTRGCNEncoder(nn.Module):
    """Input: [B, 6, T, 90]   Output: [B, C_out, T', 90]."""
    def __init__(self, A: torch.Tensor, channels=(64, 128, 256), in_c=6):
        super().__init__()
        self.input_bn = nn.BatchNorm1d(in_c * 90)
        self.part_embed = nn.Parameter(torch.zeros(1, in_c, 1, 90))   # learnable joint bias
        layers, c_prev = [], in_c
        for i, c in enumerate(channels):
            stride = 2 if i > 0 else 1                   # halve T at each new stage
            for j in range(2):                           # 2 blocks per stage
                layers.append(CTRGCBlock(c_prev, c, A, stride=stride if j == 0 else 1))
                c_prev = c
        self.blocks = nn.ModuleList(layers)
        self.out_channels = c_prev

    def forward(self, x):                                # x: [B, 6, T, 90]
        B, C, T, V = x.shape
        x = self.input_bn(x.reshape(B, C * V, T)).reshape(B, C, T, V)
        x = x + self.part_embed
        for blk in self.blocks: x = blk(x)
        return x                                         # [B, C_out, T', 90]
```

### 7. SkeletonMAE wrapper (pretrain)

```python
class SkeletonMAE(nn.Module):
    def __init__(self, A, enc_channels=(64,128,256), dec_channels=(128,64), in_c=6,
                 mask_ratio_joints=0.4, mask_ratio_time=0.3):
        super().__init__()
        self.encoder = CTRGCNEncoder(A, enc_channels, in_c=in_c)
        self.A = A
        self.mask_ratio_joints = mask_ratio_joints
        self.mask_ratio_time = mask_ratio_time
        self.mask_token = nn.Parameter(torch.zeros(1, in_c, 1, 1))
        # Lightweight decoder: temporal upsample + 1x1 conv back to in_c
        c0 = enc_channels[-1]
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(c0, dec_channels[0], (4,1), stride=(2,1), padding=(1,0)),
            nn.BatchNorm2d(dec_channels[0]), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(dec_channels[0], dec_channels[1], (4,1), stride=(2,1), padding=(1,0)),
            nn.BatchNorm2d(dec_channels[1]), nn.ReLU(inplace=True),
            nn.Conv2d(dec_channels[1], in_c, 1),
        )

    def random_mask(self, x):
        """Return masked input + boolean mask of shape [B, T, V]."""
        B, C, T, V = x.shape
        joint_keep = (torch.rand(B, V, device=x.device) > self.mask_ratio_joints)
        time_keep  = (torch.rand(B, T, device=x.device) > self.mask_ratio_time)
        mask = ~(joint_keep[:, None, :] & time_keep[:, :, None])     # [B, T, V] True=masked
        x_m = torch.where(mask[:, None], self.mask_token, x)
        return x_m, mask

    def forward(self, x):
        x_m, mask = self.random_mask(x)
        z = self.encoder(x_m)
        recon = self.dec(z)
        if recon.shape[2] != x.shape[2]:
            recon = F.interpolate(recon, size=(x.shape[2], x.shape[3]), mode="bilinear",
                                  align_corners=False)
        # MSE only on masked positions, averaged over channels
        loss = ((recon - x) ** 2).mean(1)                            # [B, T, V]
        loss = (loss * mask.float()).sum() / mask.float().sum().clamp(min=1)
        return loss, recon
```

### 8. Classifier wrapper (fine-tune)

```python
class CTRGCNClassifier(nn.Module):
    def __init__(self, A, num_classes, enc_channels=(64,128,256), in_c=6, dropout=0.3):
        super().__init__()
        self.encoder = CTRGCNEncoder(A, enc_channels, in_c=in_c)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(self.encoder.out_channels, num_classes)

    def forward(self, x):                       # [B, 6, T, V]
        z = self.encoder(x)                     # [B, C_out, T', V]
        z = z.mean(dim=(2, 3))                  # global average pool
        return self.head(self.dropout(z))

    @classmethod
    def from_pretrained(cls, ckpt_path: str, A, num_classes, **kw):
        m = cls(A, num_classes, **kw)
        sd = torch.load(ckpt_path, map_location="cpu")["encoder_state"]
        missing, _ = m.encoder.load_state_dict(sd, strict=False)
        print(f"[load] missing keys (head expected): {len(missing)}")
        return m
```

### 9. Pretrain entrypoint

```python
@app.function(image=image, gpu="A10G", volumes={"/ckpt": ckpt_vol},
              secrets=[modal.Secret.from_name(HF_SECRET)], timeout=60*60*12)
def pretrain(cfg_json: Optional[str] = None):
    cfg = PretrainConfig(**(json.loads(cfg_json) if cfg_json else {}))
    torch.manual_seed(cfg.seed); random.seed(cfg.seed)
    device = "cuda"

    A = build_adjacency().to(device)
    model = SkeletonMAE(A, cfg.encoder_channels, cfg.decoder_channels,
                        mask_ratio_joints=cfg.mask_ratio_joints,
                        mask_ratio_time=cfg.mask_ratio_time).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)

    loader = _build_pretrain_loader(cfg)        # streams How2Sign → windowed graph tensors
    best = float("inf")
    for ep in range(cfg.epochs):
        model.train(); running = 0; n = 0
        for x in loader:                                  # x: [B, 6, T, 90]
            x = x.to(device)
            loss, _ = model(x)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            running += loss.item() * x.size(0); n += x.size(0)
        sched.step()
        avg = running / max(1, n)
        print(f"epoch {ep}  pretrain_mse={avg:.4f}")
        if avg < best:
            best = avg
            os.makedirs(cfg.ckpt_dir, exist_ok=True)
            torch.save({"encoder_state": model.encoder.state_dict(),
                        "cfg": cfg.__dict__, "epoch": ep, "loss": best},
                       os.path.join(cfg.ckpt_dir, "best.pt"))
            ckpt_vol.commit()
```

`_build_pretrain_loader` reuses `UtteranceLevelStreamingDataset` from [data.py](data.py), wraps each utterance into sliding windows of `cfg.window_size`, and applies `features_to_graph`. ~25 lines.

### 10. Fine-tune + eval entrypoints

```python
@app.function(image=image, gpu="A10G",
              volumes={"/ckpt": ckpt_vol, "/wlasl": wlasl_vol}, timeout=60*60*8)
def finetune(cfg_json: Optional[str] = None):
    cfg = FinetuneConfig(**(json.loads(cfg_json) if cfg_json else {}))
    torch.manual_seed(cfg.seed); random.seed(cfg.seed)
    device = "cuda"

    A = build_adjacency().to(device)
    model = CTRGCNClassifier.from_pretrained(cfg.pretrained_ckpt, A,
                                             num_classes=cfg.num_classes).to(device)

    # Two LR groups: pretrained encoder vs fresh head
    enc_params = list(model.encoder.parameters())
    head_params = list(model.head.parameters())
    opt = torch.optim.AdamW([
        {"params": enc_params, "lr": cfg.lr_encoder},
        {"params": head_params, "lr": cfg.lr_head},
    ], weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

    train_loader, val_loader = _build_wlasl_loaders(cfg)
    best_top1 = 0.0
    for ep in range(cfg.epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            loss = loss_fn(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        top1, top5 = _evaluate(model, val_loader, device)
        print(f"ep {ep}  val_top1={top1:.3f}  val_top5={top5:.3f}")
        if top1 > best_top1:
            best_top1 = top1
            os.makedirs(cfg.ckpt_dir, exist_ok=True)
            torch.save({"model_state": model.state_dict(),
                        "cfg": cfg.__dict__, "epoch": ep, "top1": top1, "top5": top5},
                       os.path.join(cfg.ckpt_dir, "best.pt"))
            ckpt_vol.commit()


def _evaluate(model, loader, device):
    model.eval(); n = 0; c1 = 0; c5 = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            top5 = logits.topk(5, dim=-1).indices
            c1 += (top5[:, 0] == y).sum().item()
            c5 += (top5 == y[:, None]).any(-1).sum().item()
            n  += y.size(0)
    return c1 / max(1, n), c5 / max(1, n)
```

`_build_wlasl_loaders` reads the parquet produced by `extract_wlasl`, splits into `train` / (`val`+`test`) per WLASL's official split column, decodes `features` bytes with `engineer_features_multistream`, and wraps via `features_to_graph`. ~40 lines.

### 11. Local CLI dispatch

```python
@app.local_entrypoint()
def main(stage: str = "pretrain", subset: int = 100, **kw):
    if stage == "extract":   extract_wlasl.remote(subset=subset)
    elif stage == "pretrain": pretrain.remote(json.dumps(kw))
    elif stage == "finetune":
        kw.setdefault("wlasl_parquet", f"/wlasl/wlasl{subset}.parquet")
        kw.setdefault("num_classes", subset)
        kw.setdefault("ckpt_dir", f"/ckpt/wlasl{subset}")
        finetune.remote(json.dumps(kw))
    else: raise SystemExit(f"unknown stage: {stage}")
```

---

## Run sequence

```bash
# 1. one-time landmark extraction for each subset
modal run modal_skeleton_mae_app.py --stage extract --subset 100
modal run modal_skeleton_mae_app.py --stage extract --subset 300

# 2. SkeletonMAE pretraining on How2Sign (~12 h on A10G; checkpoint to ckpt_vol)
modal run modal_skeleton_mae_app.py --stage pretrain

# 3. Fine-tune on WLASL-100 and WLASL-300
modal run modal_skeleton_mae_app.py --stage finetune --subset 100
modal run modal_skeleton_mae_app.py --stage finetune --subset 300
```

---

## Verification

1. **Sanity overfit (CPU/local)**: instantiate `CTRGCNClassifier`, feed 32 random `[6, 64, 90]` clips with random labels, confirm CE loss < 0.05 within 200 steps.
2. **Adjacency check**: `build_adjacency().sum(-1)` has no zeros (all joints reachable); `(A == A.T).all()`.
3. **Pretrain plumbing**: first 50 steps show monotone decrease in `pretrain_mse`; reconstruction on a held-out clip is visually plausible (plot a hand trajectory pre-mask vs reconstructed).
4. **Fine-tune ablation**: train classifier from scratch (no pretraining) for one short run on WLASL-100 → record top-1. Pretrained init must beat this by ≥3 %, otherwise the SkeletonMAE objective is not transferring.
5. **Target numbers** (literature anchors):
   - WLASL-100: top-1 ≥ 75 % (DSLNet hits 93 %; CTR-GCN with SSL pretraining is published at ~80 %).
   - WLASL-300: top-1 ≥ 50 %.
6. **Compute report**: log per-epoch wall-clock + peak VRAM via `torch.cuda.max_memory_allocated()` on every checkpoint.

---

## Open items

- **WLASL JSON URL**: the official `WLASL_v0.3.json` is not committed here — confirm the source (GitHub `dxli94/WLASL` typically) before first `extract` run.
- **Missing videos**: rotted YouTube links — script logs and skips. If recovery rate < 60 %, switch fine-tune dataset to ASL-Citizen (HF-hosted, no scraping).
- **`features_to_graph` placement**: currently inline in the Modal file for self-containment; if reused later for other architectures, lift into [data.py](data.py).
