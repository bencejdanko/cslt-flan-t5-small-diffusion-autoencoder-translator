"""SkeletonMAE pretrain → CTR-GCN fine-tune for landmark-based isolated ASL word recognition.

Self-contained Modal app. Three stages:
  1. extract  — download WLASL videos, run MediaPipe Holistic, write parquet on volume.
  2. pretrain — masked-autoencoder pretraining of a CTR-GCN encoder on How2Sign landmarks.
  3. finetune — load pretrained encoder, attach linear head, train on WLASL.

See PLAN_TRACK2_SKELETON_MAE.md for design rationale.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field
from typing import Any, Iterable, List, Optional, Tuple

import modal


APP_NAME = "asl-skeleton-mae"
HF_SECRET = "huggingface-secret"
CKPT_VOLUME = "asl-skeleton-mae-checkpoints"
WLASL_VOLUME = "asl-wlasl-landmarks"

LANDMARKS_REPO = "bdanko/how2sign-landmarks-front-raw-parquet"
WLASL_JSON_URL = "https://raw.githubusercontent.com/dxli94/WLASL/master/start_kit/WLASL_v0.3.json"

app = modal.App(APP_NAME)

# Two images: light (training) and heavy (extraction with mediapipe + ffmpeg + yt-dlp).
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.3.1",
        "datasets",
        "huggingface_hub",
        "numpy",
        "tqdm",
        "pyarrow",
        "pandas",
    )
)

extract_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libgl1", "libglib2.0-0")
    .pip_install(
        "numpy",
        "pandas",
        "pyarrow",
        "tqdm",
        "opencv-python-headless==4.10.0.84",
        "mediapipe==0.10.14",
        "yt-dlp",
    )
)

ckpt_vol = modal.Volume.from_name(CKPT_VOLUME, create_if_missing=True)
wlasl_vol = modal.Volume.from_name(WLASL_VOLUME, create_if_missing=True)


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------
@dataclass
class PretrainConfig:
    landmarks_repo: str = LANDMARKS_REPO
    split: str = "train"
    max_samples: Optional[int] = None
    window_size: int = 64
    stride: int = 32
    batch_size: int = 32
    epochs: int = 30
    lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    mask_ratio_joints: float = 0.4
    mask_ratio_time: float = 0.3
    encoder_channels: Tuple[int, ...] = (64, 128, 256)
    decoder_channels: Tuple[int, ...] = (128, 64)
    log_every: int = 50
    ckpt_dir: str = "/ckpt/skeleton_mae"
    seed: int = 15179996


@dataclass
class FinetuneConfig:
    wlasl_parquet: str = "/wlasl/wlasl100.parquet"
    num_classes: int = 100
    window_size: int = 64
    batch_size: int = 32
    epochs: int = 80
    lr_encoder: float = 1e-4
    lr_head: float = 1e-3
    weight_decay: float = 1e-4
    label_smoothing: float = 0.1
    encoder_channels: Tuple[int, ...] = (64, 128, 256)
    pretrained_ckpt: str = "/ckpt/skeleton_mae/best.pt"
    ckpt_dir: str = "/ckpt/wlasl100"
    seed: int = 15179996


# ---------------------------------------------------------------------------
# Constants (mirrors config.py to stay self-contained inside Modal containers)
# ---------------------------------------------------------------------------
FACE_LANDMARK_IDXS: List[int] = [
    70, 105, 336, 300, 33, 133, 362, 263, 4, 61, 291, 13, 14, 17, 0,
]
POSE_SLICE = slice(0, 33)
FACE_SLICE = slice(33, 501)
LHAND_SLICE = slice(501, 522)
RHAND_SLICE = slice(522, 543)

# Joint counts and offsets in the unified V=90 graph.
N_BODY, N_FACE, N_LHAND, N_RHAND = 33, 15, 21, 21
OFFSET_BODY, OFFSET_FACE = 0, 33
OFFSET_LHAND, OFFSET_RHAND = 48, 69
N_JOINTS = N_BODY + N_FACE + N_LHAND + N_RHAND  # 90
IN_CHANNELS = 6  # (x, y, z, dx, dy, dz)


# ---------------------------------------------------------------------------
# Lazy torch import (matches modal_motion_conformer_app.py pattern)
# ---------------------------------------------------------------------------
def _load_torch_modules():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    return torch, nn, F


torch, nn, F = _load_torch_modules()


# ---------------------------------------------------------------------------
# Feature engineering (multistream — mirrors data.engineer_features_multistream)
# ---------------------------------------------------------------------------
def engineer_features_multistream(raw):
    """[T, 543, 3] MediaPipe landmarks → dict of per-part [T, n, 3] tensors (pos + vel).

    Returns None if T < 2.
    """
    import numpy as np

    T = raw.shape[0]
    if T < 2:
        return None

    pose = raw[:, POSE_SLICE, :]
    face = raw[:, FACE_SLICE, :][:, FACE_LANDMARK_IDXS, :]
    lhand = raw[:, LHAND_SLICE, :]
    rhand = raw[:, RHAND_SLICE, :]

    shoulder_center = (pose[:, 11, :] + pose[:, 12, :]) / 2.0
    pose_n = pose - shoulder_center[:, None, :]

    face_center = face.mean(axis=1, keepdims=True)
    face_n = face - face_center

    lhand_n = lhand - lhand[:, 0:1, :]
    rhand_n = rhand - rhand[:, 0:1, :]

    def vel(x):
        v = np.zeros_like(x)
        v[1:] = x[1:] - x[:-1]
        return v

    return {
        "body_pos": torch.from_numpy(pose_n.astype(np.float32)),       # [T, 33, 3]
        "face_pos": torch.from_numpy(face_n.astype(np.float32)),       # [T, 15, 3]
        "lhand_pos": torch.from_numpy(lhand_n.astype(np.float32)),     # [T, 21, 3]
        "rhand_pos": torch.from_numpy(rhand_n.astype(np.float32)),     # [T, 21, 3]
        "body_vel": torch.from_numpy(vel(pose_n).astype(np.float32)),
        "face_vel": torch.from_numpy(vel(face_n).astype(np.float32)),
        "lhand_vel": torch.from_numpy(vel(lhand_n).astype(np.float32)),
        "rhand_vel": torch.from_numpy(vel(rhand_n).astype(np.float32)),
    }


def features_to_graph(feat_dict, T_max: int):
    """Per-part feature dict → [C=6, T_max, V=90] graph tensor (pads / truncates time)."""
    parts = []
    for pos_key, vel_key in [
        ("body_pos", "body_vel"),
        ("face_pos", "face_vel"),
        ("lhand_pos", "lhand_vel"),
        ("rhand_pos", "rhand_vel"),
    ]:
        pos = feat_dict[pos_key]                # [T, n, 3]
        vel = feat_dict[vel_key]                # [T, n, 3]
        parts.append(torch.cat([pos, vel], dim=-1))  # [T, n, 6]
    x = torch.cat(parts, dim=1)                  # [T, 90, 6]
    T = x.shape[0]
    if T < T_max:
        x = F.pad(x, (0, 0, 0, 0, 0, T_max - T))
    elif T > T_max:
        x = x[:T_max]
    return x.permute(2, 0, 1).contiguous()       # [6, T_max, 90]


# ---------------------------------------------------------------------------
# Adjacency
# ---------------------------------------------------------------------------
def build_adjacency():
    """Return symmetric, self-looped, D^-1/2-normalized A of shape [90, 90]."""
    edges: List[Tuple[int, int]] = []

    POSE_EDGES = [
        (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
        (9, 10), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
        (11, 23), (12, 24), (23, 24), (23, 25), (25, 27), (27, 29), (27, 31),
        (24, 26), (26, 28), (28, 30), (28, 32),
    ]
    edges += [(OFFSET_BODY + a, OFFSET_BODY + b) for a, b in POSE_EDGES]

    FACE_LOCAL = [
        (0, 1), (2, 3), (4, 5), (6, 7),
        (8, 9), (10, 11), (11, 12), (12, 13), (13, 14),
    ]
    edges += [(OFFSET_FACE + a, OFFSET_FACE + b) for a, b in FACE_LOCAL]

    HAND_EDGES = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12),
        (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20),
        (5, 9), (9, 13), (13, 17),
    ]
    edges += [(OFFSET_LHAND + a, OFFSET_LHAND + b) for a, b in HAND_EDGES]
    edges += [(OFFSET_RHAND + a, OFFSET_RHAND + b) for a, b in HAND_EDGES]

    # Cross-part bridges so the graph is connected.
    edges += [
        (15, OFFSET_LHAND),                 # pose left wrist ↔ left hand wrist
        (16, OFFSET_RHAND),                 # pose right wrist ↔ right hand wrist
        (0, OFFSET_FACE + 8),               # pose nose ↔ face nose tip (idx 8 of FACE_LANDMARK_IDXS = MP 4)
    ]

    A = torch.zeros(N_JOINTS, N_JOINTS)
    for a, b in edges:
        A[a, b] = 1.0
        A[b, a] = 1.0
    A = A + torch.eye(N_JOINTS)
    deg = A.sum(-1).clamp(min=1.0).rsqrt()
    return deg[:, None] * A * deg[None, :]


# ---------------------------------------------------------------------------
# Model — CTR-GCN building blocks
# ---------------------------------------------------------------------------
class CTRGC(nn.Module):
    """Channel-wise Topology Refinement Graph Conv.

    For each output channel, learn an offset to a shared base topology A
    from query/key projections of the input.
    """

    def __init__(self, in_c: int, out_c: int, A, mid_c: int = 8):
        super().__init__()
        self.A = nn.Parameter(A.clone(), requires_grad=True)  # [V, V]
        self.q = nn.Conv2d(in_c, mid_c, 1)
        self.k = nn.Conv2d(in_c, mid_c, 1)
        self.refine = nn.Conv2d(mid_c, out_c, 1)
        self.value = nn.Conv2d(in_c, out_c, 1)
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x):                              # x: [B, C, T, V]
        q = self.q(x).mean(dim=2)                      # [B, mid, V]
        k = self.k(x).mean(dim=2)                      # [B, mid, V]
        diff = q.unsqueeze(-1) - k.unsqueeze(-2)       # [B, mid, V, V]
        delta = torch.tanh(self.refine(diff))          # [B, out, V, V]
        A = self.A[None, None] + self.alpha * delta    # [B, out, V, V]
        v = self.value(x)                              # [B, out, T, V]
        return torch.einsum("boTv,bovw->boTw", v, A)


class MSTCN(nn.Module):
    """Multi-scale temporal conv: parallel branches over (kernel, dilation) + 1x1 skip."""

    def __init__(self, c: int, kernels=(3, 5), dilations=(1, 2)):
        super().__init__()
        branches = []
        for k in kernels:
            for d in dilations:
                pad = d * (k - 1) // 2
                branches.append(
                    nn.Sequential(
                        nn.Conv2d(c, c // 4, 1),
                        nn.BatchNorm2d(c // 4),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(
                            c // 4, c // 4, (k, 1),
                            padding=(pad, 0), dilation=(d, 1),
                        ),
                    )
                )
        self.branches = nn.ModuleList(branches)
        self.skip = nn.Conv2d(c, c // 4, 1)
        merged = c // 4 * (len(branches) + 1)
        self.out = nn.Sequential(
            nn.BatchNorm2d(merged),
            nn.ReLU(inplace=True),
            nn.Conv2d(merged, c, 1),
            nn.BatchNorm2d(c),
        )

    def forward(self, x):
        ys = [b(x) for b in self.branches] + [self.skip(x)]
        return self.out(torch.cat(ys, dim=1))


class CTRGCBlock(nn.Module):
    """One layer: CTR-GC → BN → ReLU → MS-TCN, with residual.

    Temporal stride is applied via uniform subsampling after the temporal conv,
    matching common skeleton-action recipes.
    """

    def __init__(self, in_c: int, out_c: int, A, stride: int = 1):
        super().__init__()
        self.gc = CTRGC(in_c, out_c, A)
        self.bn = nn.BatchNorm2d(out_c)
        self.tcn = MSTCN(out_c)
        self.stride = stride
        if in_c == out_c and stride == 1:
            self.res = nn.Identity()
        else:
            self.res = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1),
                nn.BatchNorm2d(out_c),
            )

    def forward(self, x):
        y = F.relu(self.bn(self.gc(x)), inplace=True)
        y = self.tcn(y)
        if self.stride > 1:
            y = y[:, :, :: self.stride]
            x = x[:, :, :: self.stride]
        return F.relu(y + self.res(x), inplace=True)


class CTRGCNEncoder(nn.Module):
    """Stacked CTR-GC blocks. Input [B, 6, T, 90] → [B, C_out, T', 90] (T' = T/4)."""

    def __init__(self, A, channels: Tuple[int, ...] = (64, 128, 256), in_c: int = IN_CHANNELS):
        super().__init__()
        self.input_bn = nn.BatchNorm1d(in_c * N_JOINTS)
        # Per-joint learnable bias to compensate for the per-part normalization
        # used in feature engineering (each part lives in its own coord frame).
        self.part_embed = nn.Parameter(torch.zeros(1, in_c, 1, N_JOINTS))

        layers: List[nn.Module] = []
        c_prev = in_c
        for i, c in enumerate(channels):
            stage_stride = 1 if i == 0 else 2
            for j in range(2):                                  # 2 blocks per stage
                stride = stage_stride if j == 0 else 1
                layers.append(CTRGCBlock(c_prev, c, A, stride=stride))
                c_prev = c
        self.blocks = nn.ModuleList(layers)
        self.out_channels = c_prev

    def forward(self, x):
        B, C, T, V = x.shape
        x = self.input_bn(x.reshape(B, C * V, T)).reshape(B, C, T, V)
        x = x + self.part_embed
        for blk in self.blocks:
            x = blk(x)
        return x


class SkeletonMAE(nn.Module):
    """Masked autoencoder over (joint × time) tubes for self-supervised pretraining."""

    def __init__(
        self,
        A,
        enc_channels: Tuple[int, ...] = (64, 128, 256),
        dec_channels: Tuple[int, ...] = (128, 64),
        in_c: int = IN_CHANNELS,
        mask_ratio_joints: float = 0.4,
        mask_ratio_time: float = 0.3,
    ):
        super().__init__()
        self.encoder = CTRGCNEncoder(A, enc_channels, in_c=in_c)
        self.mask_ratio_joints = mask_ratio_joints
        self.mask_ratio_time = mask_ratio_time
        self.mask_token = nn.Parameter(torch.zeros(1, in_c, 1, 1))

        c0 = enc_channels[-1]
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(c0, dec_channels[0], (4, 1), stride=(2, 1), padding=(1, 0)),
            nn.BatchNorm2d(dec_channels[0]),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(dec_channels[0], dec_channels[1], (4, 1), stride=(2, 1), padding=(1, 0)),
            nn.BatchNorm2d(dec_channels[1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(dec_channels[1], in_c, 1),
        )

    def random_mask(self, x):
        """Joint-tube + time-tube random masking. Returns (masked_x, mask[B,T,V])."""
        B, C, T, V = x.shape
        joint_keep = torch.rand(B, V, device=x.device) > self.mask_ratio_joints
        time_keep = torch.rand(B, T, device=x.device) > self.mask_ratio_time
        keep = joint_keep[:, None, :] & time_keep[:, :, None]    # [B, T, V]
        mask = ~keep
        x_m = torch.where(mask[:, None], self.mask_token, x)
        return x_m, mask

    def forward(self, x):
        x_m, mask = self.random_mask(x)
        z = self.encoder(x_m)
        recon = self.dec(z)
        if recon.shape[-2:] != x.shape[-2:]:
            recon = F.interpolate(
                recon, size=x.shape[-2:], mode="bilinear", align_corners=False
            )
        per_pos = ((recon - x) ** 2).mean(dim=1)                  # [B, T, V]
        denom = mask.float().sum().clamp(min=1.0)
        loss = (per_pos * mask.float()).sum() / denom
        return loss, recon


class CTRGCNClassifier(nn.Module):
    """CTR-GCN encoder + global pool + linear head for isolated-word classification."""

    def __init__(
        self,
        A,
        num_classes: int,
        enc_channels: Tuple[int, ...] = (64, 128, 256),
        in_c: int = IN_CHANNELS,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.encoder = CTRGCNEncoder(A, enc_channels, in_c=in_c)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(self.encoder.out_channels, num_classes)

    def forward(self, x):
        z = self.encoder(x)               # [B, C_out, T', V]
        z = z.mean(dim=(2, 3))            # global average pool
        return self.head(self.dropout(z))

    @classmethod
    def from_pretrained(cls, ckpt_path: str, A, num_classes: int, **kw):
        m = cls(A, num_classes, **kw)
        sd = torch.load(ckpt_path, map_location="cpu")["encoder_state"]
        missing, unexpected = m.encoder.load_state_dict(sd, strict=False)
        print(f"[load_pretrained] missing={len(missing)} unexpected={len(unexpected)}")
        return m


# ---------------------------------------------------------------------------
# Pretraining loader (How2Sign streaming → graph windows)
# ---------------------------------------------------------------------------
def _utterance_to_graph_windows(raw, win: int, stride: int):
    """Yield [6, win, 90] tensors from one utterance's raw [T, 543, 3] landmarks."""
    feat = engineer_features_multistream(raw)
    if feat is None:
        return
    T = feat["body_pos"].shape[0]
    if T < win:
        # Pad and emit a single window.
        yield features_to_graph(feat, win)
        return
    start = 0
    while start + win <= T:
        sub = {k: v[start : start + win] for k, v in feat.items()}
        yield features_to_graph(sub, win)
        start += stride


def iter_pretrain_batches(cfg: PretrainConfig):
    """Generator yielding [B, 6, T, V] batches over the streaming How2Sign dataset."""
    import numpy as np
    from datasets import load_dataset

    ds = load_dataset(cfg.landmarks_repo, split=cfg.split, streaming=True)
    if cfg.split.startswith("train"):
        ds = ds.shuffle(seed=cfg.seed, buffer_size=1024)

    count, batch = 0, []
    for sample in ds:
        if cfg.max_samples and count >= cfg.max_samples:
            break
        count += 1
        try:
            raw = np.frombuffer(sample["features"], dtype=np.float32).reshape(sample["shape"])
        except Exception:
            continue
        for clip in _utterance_to_graph_windows(raw, cfg.window_size, cfg.stride):
            batch.append(clip)
            if len(batch) == cfg.batch_size:
                yield torch.stack(batch, dim=0)
                batch = []
    if batch:
        yield torch.stack(batch, dim=0)


# ---------------------------------------------------------------------------
# WLASL extraction
# ---------------------------------------------------------------------------
def _download_clip(url: str, frame_start: int, frame_end: int, out_dir: str) -> Optional[str]:
    """Download a YouTube clip via yt-dlp; cut frame range with ffmpeg. Returns local path or None.

    WLASL stores frame indices assuming the source video's native FPS; we keep that
    frame range and rely on cv2 frame indexing in _extract_holistic.
    """
    import subprocess
    import uuid

    os.makedirs(out_dir, exist_ok=True)
    raw_path = os.path.join(out_dir, f"{uuid.uuid4().hex}.mp4")
    try:
        subprocess.run(
            ["yt-dlp", "-q", "-f", "mp4", "-o", raw_path, url],
            check=True, timeout=180,
        )
    except Exception:
        if os.path.exists(raw_path):
            os.remove(raw_path)
        return None
    return raw_path  # caller selects frame range during extraction


def _extract_holistic(video_path: str, frame_start: int, frame_end: int):
    """Run MediaPipe Holistic over [frame_start, frame_end). Returns [T, 543, 3] np.float32 or None."""
    import cv2
    import mediapipe as mp
    import numpy as np

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    holistic = mp.solutions.holistic.Holistic(
        static_image_mode=False, model_complexity=1, refine_face_landmarks=False
    )

    frames: List[np.ndarray] = []
    fi = 0
    end = max(frame_end, frame_start + 1)
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if fi >= frame_start and fi < end:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            res = holistic.process(rgb)

            def _to_arr(landmark_list, n):
                if landmark_list is None:
                    return np.zeros((n, 3), dtype=np.float32)
                return np.array(
                    [(lm.x, lm.y, lm.z) for lm in landmark_list.landmark],
                    dtype=np.float32,
                )

            pose = _to_arr(res.pose_landmarks, 33)
            face = _to_arr(res.face_landmarks, 468)
            lh = _to_arr(res.left_hand_landmarks, 21)
            rh = _to_arr(res.right_hand_landmarks, 21)
            frames.append(np.concatenate([pose, face, lh, rh], axis=0))
        fi += 1
        if fi >= end:
            break

    cap.release()
    holistic.close()
    if not frames:
        return None
    return np.stack(frames, axis=0).astype(np.float32)   # [T, 543, 3]


@app.function(
    image=extract_image,
    volumes={"/wlasl": wlasl_vol},
    timeout=60 * 60 * 8,
    cpu=4.0,
)
def extract_wlasl(subset: int = 100, wlasl_json_url: str = WLASL_JSON_URL):
    """Run once per subset (100 / 300). Produces /wlasl/wlasl{subset}.parquet."""
    import shutil
    import tempfile
    import urllib.request

    import numpy as np
    import pandas as pd
    from tqdm import tqdm

    print(f"[extract] fetching WLASL JSON from {wlasl_json_url}")
    meta = json.loads(urllib.request.urlopen(wlasl_json_url, timeout=30).read())
    classes = sorted({entry["gloss"] for entry in meta})[:subset]
    label2id = {g: i for i, g in enumerate(classes)}
    print(f"[extract] using {len(classes)} classes (subset={subset})")

    tmp_root = tempfile.mkdtemp(prefix="wlasl_dl_")
    rows: List[dict] = []
    n_seen, n_ok, n_fail = 0, 0, 0
    try:
        for entry in tqdm(meta, desc="classes"):
            gloss = entry["gloss"]
            if gloss not in label2id:
                continue
            for inst in entry["instances"]:
                n_seen += 1
                video = _download_clip(
                    inst["url"], inst["frame_start"], inst["frame_end"], tmp_root
                )
                if video is None:
                    n_fail += 1
                    continue
                try:
                    raw = _extract_holistic(video, inst["frame_start"], inst["frame_end"])
                finally:
                    if os.path.exists(video):
                        os.remove(video)
                if raw is None or raw.shape[0] < 8:
                    n_fail += 1
                    continue
                rows.append(
                    {
                        "features": raw.astype(np.float32).tobytes(),
                        "shape": list(raw.shape),
                        "label": label2id[gloss],
                        "gloss": gloss,
                        "split": inst.get("split", "train"),
                    }
                )
                n_ok += 1
                if n_seen % 50 == 0:
                    print(f"[extract] seen={n_seen} ok={n_ok} fail={n_fail}")
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    print(f"[extract] done: ok={n_ok} fail={n_fail} (recovery {n_ok / max(1, n_seen):.1%})")
    out = f"/wlasl/wlasl{subset}.parquet"
    pd.DataFrame(rows).to_parquet(out)
    wlasl_vol.commit()
    print(f"[extract] wrote {out} ({len(rows)} rows)")
    return {"rows": len(rows), "ok": n_ok, "fail": n_fail, "path": out}


# ---------------------------------------------------------------------------
# WLASL fine-tune loaders
# ---------------------------------------------------------------------------
class _WLASLDataset:
    """Map-style dataset over a parquet produced by extract_wlasl."""

    def __init__(self, parquet_path: str, split_filter: List[str], window_size: int):
        import numpy as np
        import pandas as pd

        df = pd.read_parquet(parquet_path)
        df = df[df["split"].isin(split_filter)].reset_index(drop=True)
        self.df = df
        self.window_size = window_size
        self._np = np

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        raw = self._np.frombuffer(row["features"], dtype=self._np.float32).reshape(row["shape"])
        feat = engineer_features_multistream(raw)
        if feat is None:
            # Defensive fallback — shouldn't happen since extractor enforces T>=8.
            x = torch.zeros(IN_CHANNELS, self.window_size, N_JOINTS)
        else:
            x = features_to_graph(feat, self.window_size)
        return x, int(row["label"])


def _collate(batch):
    xs, ys = zip(*batch)
    return torch.stack(xs, dim=0), torch.tensor(ys, dtype=torch.long)


def _build_wlasl_loaders(cfg: FinetuneConfig):
    from torch.utils.data import DataLoader

    train_ds = _WLASLDataset(cfg.wlasl_parquet, ["train"], cfg.window_size)
    val_ds = _WLASLDataset(cfg.wlasl_parquet, ["val", "test"], cfg.window_size)
    print(f"[data] wlasl train={len(train_ds)} val+test={len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=_collate,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=_collate,
        pin_memory=True,
    )
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Pretrain entrypoint
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    gpu="A10G",
    volumes={"/ckpt": ckpt_vol},
    secrets=[modal.Secret.from_name(HF_SECRET)],
    timeout=60 * 60 * 12,
)
def pretrain(cfg_json: Optional[str] = None):
    cfg = PretrainConfig(**(json.loads(cfg_json) if cfg_json else {}))
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    A = build_adjacency().to(device)
    model = SkeletonMAE(
        A,
        enc_channels=tuple(cfg.encoder_channels),
        dec_channels=tuple(cfg.decoder_channels),
        mask_ratio_joints=cfg.mask_ratio_joints,
        mask_ratio_time=cfg.mask_ratio_time,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[pretrain] model params = {n_params/1e6:.2f}M, device={device}")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)

    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    best = float("inf")
    for ep in range(cfg.epochs):
        model.train()
        running, n_clips, step = 0.0, 0, 0
        for x in iter_pretrain_batches(cfg):
            x = x.to(device, non_blocking=True)
            loss, _ = model(x)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            running += loss.item() * x.size(0)
            n_clips += x.size(0)
            step += 1
            if step % cfg.log_every == 0:
                print(f"[pretrain] ep{ep} step{step} loss={loss.item():.4f}")
        sched.step()
        avg = running / max(1, n_clips)
        peak_mb = torch.cuda.max_memory_allocated() / 1024**2 if device == "cuda" else 0
        print(f"[pretrain] epoch {ep} done  avg_mse={avg:.4f}  peak_vram_MB={peak_mb:.0f}")
        if avg < best:
            best = avg
            ckpt_path = os.path.join(cfg.ckpt_dir, "best.pt")
            torch.save(
                {
                    "encoder_state": model.encoder.state_dict(),
                    "mae_state": model.state_dict(),
                    "cfg": cfg.__dict__,
                    "epoch": ep,
                    "loss": best,
                },
                ckpt_path,
            )
            ckpt_vol.commit()
            print(f"[pretrain] saved best to {ckpt_path}")
    return {"best_loss": best}


# ---------------------------------------------------------------------------
# Fine-tune entrypoint
# ---------------------------------------------------------------------------
def _evaluate(model, loader, device):
    model.eval()
    n, c1, c5 = 0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            top5 = logits.topk(min(5, logits.size(-1)), dim=-1).indices
            c1 += (top5[:, 0] == y).sum().item()
            c5 += (top5 == y[:, None]).any(-1).sum().item()
            n += y.size(0)
    return c1 / max(1, n), c5 / max(1, n)


@app.function(
    image=image,
    gpu="A10G",
    volumes={"/ckpt": ckpt_vol, "/wlasl": wlasl_vol},
    timeout=60 * 60 * 8,
)
def finetune(cfg_json: Optional[str] = None):
    cfg = FinetuneConfig(**(json.loads(cfg_json) if cfg_json else {}))
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    A = build_adjacency().to(device)
    if os.path.exists(cfg.pretrained_ckpt):
        model = CTRGCNClassifier.from_pretrained(
            cfg.pretrained_ckpt, A,
            num_classes=cfg.num_classes,
            enc_channels=tuple(cfg.encoder_channels),
        ).to(device)
        print(f"[finetune] loaded pretrained encoder from {cfg.pretrained_ckpt}")
    else:
        print(f"[finetune] no pretrained ckpt at {cfg.pretrained_ckpt}; training from scratch")
        model = CTRGCNClassifier(
            A, cfg.num_classes, enc_channels=tuple(cfg.encoder_channels),
        ).to(device)

    opt = torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": cfg.lr_encoder},
            {"params": model.head.parameters(), "lr": cfg.lr_head},
        ],
        weight_decay=cfg.weight_decay,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

    train_loader, val_loader = _build_wlasl_loaders(cfg)

    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    best_top1 = 0.0
    for ep in range(cfg.epochs):
        model.train()
        running, n_obs = 0.0, 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item() * x.size(0)
            n_obs += x.size(0)
        sched.step()
        top1, top5 = _evaluate(model, val_loader, device)
        peak_mb = torch.cuda.max_memory_allocated() / 1024**2 if device == "cuda" else 0
        print(
            f"[finetune] ep{ep} train_loss={running/max(1,n_obs):.4f} "
            f"val_top1={top1:.3f} val_top5={top5:.3f} peak_vram_MB={peak_mb:.0f}"
        )
        if top1 > best_top1:
            best_top1 = top1
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "cfg": cfg.__dict__,
                    "epoch": ep,
                    "top1": top1,
                    "top5": top5,
                },
                os.path.join(cfg.ckpt_dir, "best.pt"),
            )
            ckpt_vol.commit()
            print(f"[finetune] saved new best (top1={top1:.3f})")
    return {"best_top1": best_top1}


# ---------------------------------------------------------------------------
# Local CLI dispatch
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(stage: str = "pretrain", subset: int = 100, **kw):
    """Stages: extract | pretrain | finetune."""
    if stage == "extract":
        result = extract_wlasl.remote(subset=subset)
        print(json.dumps(result, indent=2))
    elif stage == "pretrain":
        cfg = PretrainConfig(**kw).__dict__
        # tuples come back as lists from JSON; pretrain entrypoint normalizes via tuple(...).
        cfg.pop("encoder_channels", None)
        cfg.pop("decoder_channels", None)
        kw_clean = {k: v for k, v in cfg.items() if not isinstance(v, (tuple, list))}
        result = pretrain.remote(json.dumps(kw_clean))
        print(json.dumps(result, indent=2))
    elif stage == "finetune":
        kw.setdefault("wlasl_parquet", f"/wlasl/wlasl{subset}.parquet")
        kw.setdefault("num_classes", subset)
        kw.setdefault("ckpt_dir", f"/ckpt/wlasl{subset}")
        result = finetune.remote(json.dumps(kw))
        print(json.dumps(result, indent=2))
    else:
        raise SystemExit(f"unknown stage: {stage} (expected: extract | pretrain | finetune)")


# ---------------------------------------------------------------------------
# Local self-test
# ---------------------------------------------------------------------------
def _self_test():
    """Run without Modal: adjacency checks + classifier overfit + MAE forward.

    Usage:  python modal_skeleton_mae_app.py
    """
    print("[selftest] building adjacency...")
    A = build_adjacency()
    assert A.shape == (N_JOINTS, N_JOINTS), A.shape
    assert torch.allclose(A, A.T, atol=1e-6), "adjacency must be symmetric"
    assert A.sum(-1).gt(0).all(), "every joint must have at least one connection"
    print(f"[selftest]   A.shape={tuple(A.shape)} symmetric, all degrees>0  ✓")

    torch.manual_seed(0)
    model = CTRGCNClassifier(A, num_classes=10)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[selftest] CTRGCNClassifier params = {n_params/1e6:.2f}M")

    x = torch.randn(4, IN_CHANNELS, 32, N_JOINTS)
    y = torch.randint(0, 10, (4,))
    logits = model(x)
    assert logits.shape == (4, 10), logits.shape
    print(f"[selftest]   forward OK, logits.shape={tuple(logits.shape)}  ✓")

    print("[selftest] overfitting 4 random clips...")
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    last = None
    for step in range(300):
        loss = F.cross_entropy(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 50 == 0:
            print(f"[selftest]   step {step:3d}  loss={loss.item():.4f}")
        last = loss.item()
    assert last is not None and last < 0.1, f"failed to overfit (final loss={last})"
    print(f"[selftest]   final overfit loss = {last:.4f}  ✓")

    print("[selftest] SkeletonMAE forward...")
    mae = SkeletonMAE(A)
    mae_loss, recon = mae(x)
    assert recon.shape == x.shape, (recon.shape, x.shape)
    print(f"[selftest]   recon.shape={tuple(recon.shape)} loss={mae_loss.item():.4f}  ✓")

    print("[selftest] all checks passed.")


if __name__ == "__main__":
    _self_test()
