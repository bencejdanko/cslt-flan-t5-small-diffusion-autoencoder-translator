from __future__ import annotations

import json
import os
import random
from dataclasses import asdict, dataclass
from typing import Any

import modal


APP_NAME = "asl-motion-conformer-t5"
HF_SECRET = "huggingface-secret"
CHECKPOINT_VOLUME = "asl-motion-conformer-checkpoints"

LANDMARKS_REPO = "bdanko/how2sign-landmarks-front-raw-parquet"
RGB_REPO = "bdanko/how2sign-rgb-front-clips"

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "datasets",
        "huggingface_hub",
        "transformers",
        "sentencepiece",
        "numpy",
        "tqdm",
        "sacrebleu",
    )
)

volume = modal.Volume.from_name(CHECKPOINT_VOLUME, create_if_missing=True)


@dataclass
class TrainConfig:
    landmarks_repo: str = LANDMARKS_REPO
    split: str = "train"
    max_samples: int | None = 100

    window_size: int = 60
    stride: int = 30
    batch_size: int = 4

    lr: float = 1e-4
    ctc_weight: float = 0.1
    reconstruction_weight: float = 1.0

    t5_model: str = "google/flan-t5-small"
    freeze_t5: bool = True

    ckpt_dir: str = "/checkpoints/motion_conformer_t5_smoke"
    seed: int = 15179996


@dataclass
class EvalConfig:
    landmarks_repo: str = LANDMARKS_REPO
    split: str = "validation"
    max_samples: int | None = None
    batch_size: int = 4

    window_size: int = 60
    stride: int = 30

    t5_model: str = "google/flan-t5-small"
    checkpoint_path: str = "/checkpoints/motion_conformer_t5_full/motion_conformer_t5_smoke.pt"
    output_dir: str = "/checkpoints/motion_conformer_t5_full/eval_validation"

    num_beams: int = 4
    max_new_tokens: int = 64
    seed: int = 15179996


@dataclass
class Phase1EvalConfig:
    landmarks_repo: str = LANDMARKS_REPO
    split: str = "validation"
    max_samples: int | None = None
    batch_size: int = 16

    window_size: int = 60
    stride: int = 30

    t5_model: str = "google/flan-t5-small"
    checkpoint_path: str = "/checkpoints/motion_conformer_t5_full/motion_conformer_t5_smoke.pt"
    output_dir: str = "/checkpoints/motion_conformer_t5_full/eval_phase1_validation"

    seed: int = 15179996


@dataclass
class Phase1WordsTrainConfig:
    landmarks_repo: str = LANDMARKS_REPO
    split: str = "train"
    max_samples: int | None = None

    window_size: int = 60
    stride: int = 30
    max_windows_per_utterance: int = 12
    batch_size: int = 2

    top_k_words: int = 512
    min_word_len: int = 3
    pos_weight_clip: float = 20.0

    lr: float = 1e-4
    epochs: int = 1
    reconstruction_weight: float = 1.0
    word_weight: float = 1.0
    latent_smooth_weight: float = 0.01
    latent_reg_weight: float = 0.01
    grad_clip: float = 1.0

    ckpt_dir: str = "/checkpoints/motion_conformer_phase1_words_full"
    seed: int = 15179996


@dataclass
class Phase1WordsEvalConfig:
    landmarks_repo: str = LANDMARKS_REPO
    split: str = "validation"
    max_samples: int | None = None
    batch_size: int = 4

    window_size: int = 60
    stride: int = 30
    max_windows_per_utterance: int = 12

    checkpoint_path: str = "/checkpoints/motion_conformer_phase1_words_full/phase1_words.pt"
    vocab_path: str = "/checkpoints/motion_conformer_phase1_words_full/word_vocab.json"
    output_dir: str = "/checkpoints/motion_conformer_phase1_words_full/eval_validation"

    seed: int = 15179996


def _normalize_max_samples(max_samples: int) -> int | None:
    """Modal CLI cannot parse Optional[int]; use 0 to mean full dataset."""
    return None if max_samples <= 0 else max_samples


FACE_LANDMARK_IDXS = [70, 105, 336, 300, 33, 133, 362, 263, 4, 61, 291, 13, 14, 17, 0]
POSE_SLICE = slice(0, 33)
FACE_SLICE = slice(33, 501)
LHAND_SLICE = slice(501, 522)
RHAND_SLICE = slice(522, 543)

WORD_STOPWORDS = {
    "about",
    "above",
    "after",
    "again",
    "against",
    "ain",
    "all",
    "also",
    "am",
    "an",
    "and",
    "any",
    "are",
    "aren",
    "around",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "below",
    "between",
    "both",
    "but",
    "by",
    "can",
    "could",
    "couldn",
    "did",
    "didn",
    "do",
    "does",
    "doesn",
    "doing",
    "don",
    "down",
    "during",
    "each",
    "few",
    "for",
    "from",
    "further",
    "had",
    "hadn",
    "has",
    "hasn",
    "have",
    "haven",
    "having",
    "he",
    "her",
    "here",
    "hers",
    "herself",
    "him",
    "himself",
    "his",
    "how",
    "if",
    "in",
    "into",
    "is",
    "isn",
    "it",
    "its",
    "itself",
    "just",
    "ll",
    "m",
    "ma",
    "me",
    "more",
    "most",
    "mustn",
    "my",
    "myself",
    "need",
    "no",
    "nor",
    "not",
    "now",
    "of",
    "off",
    "on",
    "once",
    "only",
    "or",
    "other",
    "our",
    "ours",
    "ourselves",
    "out",
    "over",
    "own",
    "re",
    "s",
    "same",
    "shan",
    "she",
    "should",
    "shouldn",
    "so",
    "some",
    "such",
    "t",
    "than",
    "that",
    "the",
    "their",
    "theirs",
    "them",
    "themselves",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "to",
    "too",
    "under",
    "until",
    "up",
    "ve",
    "very",
    "was",
    "wasn",
    "we",
    "were",
    "weren",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "whom",
    "why",
    "will",
    "with",
    "won",
    "would",
    "wouldn",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
}


def _check_data_source_impl(repo_id: str = LANDMARKS_REPO, split: str = "train") -> dict[str, Any]:
    import numpy as np
    from datasets import load_dataset

    try:
        ds = load_dataset(repo_id, split=split, streaming=True)
        sample = next(iter(ds))
    except Exception as exc:
        raise RuntimeError(
            f"Could not load dataset {repo_id!r} split {split!r}. "
            f"If the dataset is private, ensure Modal secret {HF_SECRET!r} exposes HF_TOKEN."
        ) from exc

    required = {"features", "shape", "sentence"}
    missing = sorted(required - set(sample.keys()))
    if missing:
        raise ValueError(
            f"Dataset {repo_id} is missing required fields: {missing}. "
            f"Keys: {sorted(sample.keys())}"
        )

    raw = np.frombuffer(sample["features"], dtype=np.float32).reshape(sample["shape"])
    if raw.ndim != 3 or raw.shape[1:] != (543, 3):
        raise ValueError(f"Expected raw landmark shape [T, 543, 3], got {raw.shape}")

    return {
        "repo_id": repo_id,
        "split": split,
        "keys": sorted(sample.keys()),
        "shape": tuple(int(v) for v in raw.shape),
        "sentence_preview": sample["sentence"][:160],
        "video_id": sample.get("video_id"),
    }


@app.function(image=image)
def check_data_source(
    repo_id: str = LANDMARKS_REPO,
    split: str = "train",
) -> dict[str, Any]:
    result = _check_data_source_impl(repo_id=repo_id, split=split)
    print(json.dumps(result, indent=2))
    return result


def engineer_features(raw: "np.ndarray") -> "torch.Tensor | None":
    import numpy as np
    import torch

    t = raw.shape[0]
    if t < 2:
        return None

    pose = raw[:, POSE_SLICE, :]
    face = raw[:, FACE_SLICE, :]
    lhand = raw[:, LHAND_SLICE, :]
    rhand = raw[:, RHAND_SLICE, :]

    center = pose.mean(axis=1, keepdims=True)
    pose, face, lhand, rhand = (part - center for part in (pose, face, lhand, rhand))

    keypoints = np.concatenate(
        [pose, face[:, FACE_LANDMARK_IDXS, :], lhand, rhand],
        axis=1,
    )

    delta = np.zeros_like(keypoints)
    delta[1:] = keypoints[1:] - keypoints[:-1]

    features = np.concatenate(
        [keypoints.reshape(t, -1), delta.reshape(t, -1)],
        axis=1,
    )
    return torch.from_numpy(features.astype(np.float32))


def sliding_windows(features: "torch.Tensor", window: int = 60, stride: int = 30):
    import torch.nn.functional as F

    t = features.shape[0]
    if t < window:
        yield F.pad(features, (0, 0, 0, window - t))
        return

    start = 0
    while start + window <= t:
        yield features[start : start + window]
        start += stride


def tokenize_content_words(sentence: str, min_word_len: int = 3) -> list[str]:
    import re

    words = []
    for match in re.findall(r"[a-z]+(?:'[a-z]+)?", sentence.lower()):
        token = match.strip("'")
        if len(token) < min_word_len:
            continue
        if token in WORD_STOPWORDS:
            continue
        words.append(token)
    return words


def build_word_vocab(
    repo_id: str,
    split: str,
    max_samples: int | None,
    top_k_words: int,
    min_word_len: int,
) -> tuple[list[str], list[int], int]:
    from collections import Counter

    from datasets import load_dataset

    counts: Counter[str] = Counter()
    ds = load_dataset(repo_id, split=split, streaming=True)
    sample_count = 0
    for sample in ds:
        if max_samples is not None and sample_count >= max_samples:
            break
        words = set(tokenize_content_words(sample.get("sentence", ""), min_word_len=min_word_len))
        counts.update(words)
        sample_count += 1

    vocab = [word for word, _ in counts.most_common(top_k_words)]
    if not vocab:
        raise ValueError("Could not build a word vocabulary from the dataset sentences.")
    doc_counts = [counts[word] for word in vocab]
    return vocab, doc_counts, sample_count


def labels_from_sentence(sentence: str, word_to_idx: dict[str, int], min_word_len: int) -> "torch.Tensor":
    import torch

    labels = torch.zeros(len(word_to_idx), dtype=torch.float32)
    for word in set(tokenize_content_words(sentence, min_word_len=min_word_len)):
        idx = word_to_idx.get(word)
        if idx is not None:
            labels[idx] = 1.0
    return labels


def select_utterance_windows(
    features: "torch.Tensor",
    window: int,
    stride: int,
    max_windows: int,
) -> "torch.Tensor":
    import torch

    chunks = list(sliding_windows(features, window=window, stride=stride))
    if not chunks:
        raise ValueError("Utterance produced no windows.")

    if max_windows > 0 and len(chunks) > max_windows:
        if max_windows == 1:
            indices = [len(chunks) // 2]
        else:
            indices = [
                round(i * (len(chunks) - 1) / (max_windows - 1))
                for i in range(max_windows)
            ]
        chunks = [chunks[i] for i in indices]

    return torch.stack(chunks)


class How2SignWindowDataset:
    def __init__(
        self,
        repo_id: str,
        split: str = "train",
        max_samples: int | None = 100,
        window: int = 60,
        stride: int = 30,
    ) -> None:
        self.repo_id = repo_id
        self.split = split
        self.max_samples = max_samples
        self.window = window
        self.stride = stride

    def __iter__(self):
        import numpy as np
        from datasets import load_dataset

        ds = load_dataset(self.repo_id, split=self.split, streaming=True)
        sample_count = 0
        for sample in ds:
            if self.max_samples is not None and sample_count >= self.max_samples:
                break

            raw = np.frombuffer(sample["features"], dtype=np.float32).reshape(sample["shape"])
            features = engineer_features(raw)
            if features is None:
                continue

            sentence = sample.get("sentence", "")
            video_id = sample.get("video_id", "")
            for chunk in sliding_windows(features, self.window, self.stride):
                yield {
                    "features": chunk,
                    "sentence": sentence,
                    "video_id": video_id,
                }

            sample_count += 1


class How2SignUtteranceWordDataset:
    def __init__(
        self,
        repo_id: str,
        split: str,
        word_to_idx: dict[str, int],
        max_samples: int | None,
        window: int,
        stride: int,
        max_windows_per_utterance: int,
        min_word_len: int,
    ) -> None:
        self.repo_id = repo_id
        self.split = split
        self.word_to_idx = word_to_idx
        self.max_samples = max_samples
        self.window = window
        self.stride = stride
        self.max_windows_per_utterance = max_windows_per_utterance
        self.min_word_len = min_word_len

    def __iter__(self):
        import numpy as np
        from datasets import load_dataset

        ds = load_dataset(self.repo_id, split=self.split, streaming=True)
        sample_count = 0
        for sample in ds:
            if self.max_samples is not None and sample_count >= self.max_samples:
                break

            raw = np.frombuffer(sample["features"], dtype=np.float32).reshape(sample["shape"])
            features = engineer_features(raw)
            if features is None:
                continue

            sentence = sample.get("sentence", "")
            labels = labels_from_sentence(
                sentence,
                word_to_idx=self.word_to_idx,
                min_word_len=self.min_word_len,
            )
            try:
                windows = select_utterance_windows(
                    features,
                    window=self.window,
                    stride=self.stride,
                    max_windows=self.max_windows_per_utterance,
                )
            except ValueError:
                continue

            yield {
                "windows": windows,
                "labels": labels,
                "sentence": sentence,
                "video_id": sample.get("video_id", ""),
            }
            sample_count += 1


def batched(iterable, batch_size: int):
    batch = []
    for row in iterable:
        batch.append(row)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _load_torch_modules():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    return torch, nn, F


torch, nn, F = _load_torch_modules()


class ModalityProjection(nn.Module):
    """Project pose, face, and hands separately before temporal modeling."""

    def __init__(self, d_model: int = 256):
        super().__init__()
        self.pose = nn.Sequential(nn.Linear(198, 64), nn.GELU())
        self.face = nn.Sequential(nn.Linear(90, 32), nn.GELU())
        self.lhand = nn.Sequential(nn.Linear(126, 96), nn.GELU())
        self.rhand = nn.Sequential(nn.Linear(126, 96), nn.GELU())
        self.fuse = nn.Sequential(nn.Linear(288, d_model), nn.LayerNorm(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pos = x[:, :, :270]
        delta = x[:, :, 270:]

        pose = torch.cat([pos[:, :, 0:99], delta[:, :, 0:99]], dim=-1)
        face = torch.cat([pos[:, :, 99:144], delta[:, :, 99:144]], dim=-1)
        lhand = torch.cat([pos[:, :, 144:207], delta[:, :, 144:207]], dim=-1)
        rhand = torch.cat([pos[:, :, 207:270], delta[:, :, 207:270]], dim=-1)

        projected = torch.cat(
            [
                self.pose(pose),
                self.face(face),
                self.lhand(lhand),
                self.rhand(rhand),
            ],
            dim=-1,
        )
        return self.fuse(projected)


class ConformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 4,
        ff_dim: int = 1024,
        conv_kernel: int = 7,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.ff1 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout),
        )
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.conv_norm = nn.LayerNorm(d_model)
        self.depthwise = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=conv_kernel,
            padding=conv_kernel // 2,
            groups=d_model,
        )
        self.pointwise = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.conv_dropout = nn.Dropout(dropout)
        self.ff2 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout),
        )
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + 0.5 * self.ff1(x)

        attn_in = self.attn_norm(x)
        attn_out, _ = self.attn(attn_in, attn_in, attn_in, need_weights=False)
        x = x + attn_out

        conv_in = self.conv_norm(x).transpose(1, 2)
        conv_out = self.depthwise(conv_in)
        conv_out = self.pointwise(F.gelu(conv_out)).transpose(1, 2)
        x = x + self.conv_dropout(conv_out)

        x = x + 0.5 * self.ff2(x)
        return self.out_norm(x)


class MotionConformerEncoder(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        latent_dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 4,
        ff_dim: int = 1024,
        conv_kernel: int = 7,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.project = ModalityProjection(d_model=d_model)
        self.blocks = nn.ModuleList(
            [
                ConformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    ff_dim=ff_dim,
                    conv_kernel=conv_kernel,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.downsample = nn.Sequential(
            nn.Conv1d(d_model, 384, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv1d(384, latent_dim, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )
        self.out_norm = nn.LayerNorm(latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.project(x)
        for block in self.blocks:
            x = block(x)
        x = self.downsample(x.transpose(1, 2)).transpose(1, 2)
        return self.out_norm(x)


class ReconstructionDecoder(nn.Module):
    def __init__(self, latent_dim: int = 512, output_dim: int = 540):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose1d(latent_dim, 384, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose1d(384, 256, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv1d(256, output_dim, kernel_size=3, padding=1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z.transpose(1, 2)).transpose(1, 2)


class T5Adapter(nn.Module):
    def __init__(self, dim: int = 512, max_len: int = 15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.pos = nn.Parameter(torch.zeros(1, max_len, dim))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z) + self.pos[:, : z.shape[1]]


class MotionConformerT5(nn.Module):
    def __init__(self, t5: nn.Module, freeze_t5: bool = True):
        super().__init__()
        self.encoder = MotionConformerEncoder()
        self.reconstruction_decoder = ReconstructionDecoder()
        self.ctc_blank = t5.config.vocab_size
        self.ctc_head = nn.Linear(512, t5.config.vocab_size + 1)
        self.adapter = T5Adapter(dim=512, max_len=15)
        self.t5 = t5

        if freeze_t5:
            for parameter in self.t5.parameters():
                parameter.requires_grad = False

    def forward(self, features: torch.Tensor, labels: torch.Tensor | None = None) -> dict[str, Any]:
        z = self.encoder(features)
        adapted = self.adapter(z)
        t5_outputs = self.t5(inputs_embeds=adapted, labels=labels) if labels is not None else None

        return {
            "z": z,
            "reconstruction": self.reconstruction_decoder(z),
            "ctc_logits": self.ctc_head(z),
            "t5_outputs": t5_outputs,
        }


class MotionConformerPhase1Words(nn.Module):
    """Phase 1 model with window reconstruction and utterance-level word logits."""

    def __init__(self, num_words: int, latent_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.encoder = MotionConformerEncoder(latent_dim=latent_dim)
        self.reconstruction_decoder = ReconstructionDecoder(latent_dim=latent_dim)
        self.classifier = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, num_words),
        )

    def forward(self, windows: torch.Tensor, window_counts: list[int]) -> dict[str, torch.Tensor]:
        z_windows = self.encoder(windows)
        reconstruction = self.reconstruction_decoder(z_windows)

        utterance_embeddings = []
        offset = 0
        for count in window_counts:
            z_slice = z_windows[offset : offset + count]
            utterance_embeddings.append(z_slice.mean(dim=(0, 1)))
            offset += count
        pooled = torch.stack(utterance_embeddings)
        logits = self.classifier(pooled)

        return {
            "z": z_windows,
            "reconstruction": reconstruction,
            "utterance_embedding": pooled,
            "word_logits": logits,
        }


def pack_utterance_batch(batch: list[dict[str, Any]], device: str) -> tuple[torch.Tensor, list[int], torch.Tensor]:
    windows = [row["windows"] for row in batch]
    window_counts = [int(item.shape[0]) for item in windows]
    flat_windows = torch.cat(windows, dim=0).to(device)
    labels = torch.stack([row["labels"] for row in batch]).to(device)
    return flat_windows, window_counts, labels


def compute_phase1_word_losses(
    model_outputs: dict[str, torch.Tensor],
    windows: torch.Tensor,
    labels: torch.Tensor,
    bce_loss_fn: nn.Module,
    cfg: Phase1WordsTrainConfig | Phase1WordsEvalConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    reconstruction = model_outputs["reconstruction"]
    z = model_outputs["z"]
    logits = model_outputs["word_logits"]

    recon_loss = F.mse_loss(reconstruction, windows)
    pos_loss = F.mse_loss(reconstruction[:, :, :270], windows[:, :, :270])
    vel_loss = F.mse_loss(reconstruction[:, :, 270:], windows[:, :, 270:])
    word_bce = bce_loss_fn(logits, labels)
    latent_smooth = (
        ((z[:, 1:] - z[:, :-1]) ** 2).mean()
        if z.shape[1] > 1
        else torch.tensor(0.0, device=z.device)
    )
    latent_reg = z.mean().pow(2) + (z.var(unbiased=False) - 1.0).pow(2)

    reconstruction_weight = getattr(cfg, "reconstruction_weight", 1.0)
    word_weight = getattr(cfg, "word_weight", 1.0)
    latent_smooth_weight = getattr(cfg, "latent_smooth_weight", 0.01)
    latent_reg_weight = getattr(cfg, "latent_reg_weight", 0.01)

    total = (
        reconstruction_weight * recon_loss
        + word_weight * word_bce
        + latent_smooth_weight * latent_smooth
        + latent_reg_weight * latent_reg
    )

    return total, {
        "total_loss": float(total.detach().cpu()),
        "full_recon_loss": float(recon_loss.detach().cpu()),
        "pos_recon_loss": float(pos_loss.detach().cpu()),
        "vel_recon_loss": float(vel_loss.detach().cpu()),
        "word_bce": float(word_bce.detach().cpu()),
        "latent_smooth_loss": float(latent_smooth.detach().cpu()),
        "latent_reg_loss": float(latent_reg.detach().cpu()),
    }


def word_classification_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    k: int = 5,
) -> dict[str, float]:
    probs = torch.sigmoid(logits)
    predicted = probs >= 0.5
    truth = labels > 0.5

    tp = float((predicted & truth).sum().detach().cpu())
    fp = float((predicted & ~truth).sum().detach().cpu())
    fn = float((~predicted & truth).sum().detach().cpu())

    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    top_k = min(k, logits.shape[1])
    top_indices = torch.topk(probs, k=top_k, dim=1).indices
    batch_precision_at_k = []
    batch_recall_at_k = []
    hits = []
    for row_idx in range(labels.shape[0]):
        predicted_set = set(int(v) for v in top_indices[row_idx].detach().cpu().tolist())
        true_set = set(int(v) for v in torch.nonzero(truth[row_idx], as_tuple=False).flatten().detach().cpu().tolist())
        overlap = len(predicted_set & true_set)
        batch_precision_at_k.append(overlap / max(top_k, 1))
        batch_recall_at_k.append(overlap / max(len(true_set), 1))
        hits.append(float(overlap > 0))

    return {
        "micro_precision@0.5": precision,
        "micro_recall@0.5": recall,
        "micro_f1@0.5": f1,
        "precision@5": sum(batch_precision_at_k) / max(len(batch_precision_at_k), 1),
        "recall@5": sum(batch_recall_at_k) / max(len(batch_recall_at_k), 1),
        "hit@5": sum(hits) / max(len(hits), 1),
    }


def decode_top_words(logits: torch.Tensor, vocab: list[str], k: int = 5) -> list[list[str]]:
    probs = torch.sigmoid(logits)
    top_k = min(k, len(vocab))
    top_indices = torch.topk(probs, k=top_k, dim=1).indices.detach().cpu().tolist()
    return [[vocab[idx] for idx in row] for row in top_indices]


def compute_t5_subword_ctc_loss(
    tokenizer: Any,
    ctc_loss_fn: nn.Module,
    ctc_logits: torch.Tensor,
    sentences: list[str],
    device: str,
) -> torch.Tensor:
    encoded = tokenizer(
        sentences,
        add_special_tokens=False,
        padding=False,
        truncation=True,
        max_length=ctc_logits.shape[1],
    )

    flat_targets: list[int] = []
    target_lengths: list[int] = []
    for token_ids in encoded.input_ids:
        if not token_ids:
            token_ids = [tokenizer.pad_token_id]
        flat_targets.extend(token_ids)
        target_lengths.append(len(token_ids))

    targets = torch.tensor(flat_targets, dtype=torch.long, device=device)
    target_lengths_tensor = torch.tensor(target_lengths, dtype=torch.long, device=device)
    input_lengths = torch.full(
        size=(ctc_logits.shape[0],),
        fill_value=ctc_logits.shape[1],
        dtype=torch.long,
        device=device,
    )

    log_probs = F.log_softmax(ctc_logits, dim=-1).transpose(0, 1)
    return ctc_loss_fn(log_probs, targets, input_lengths, target_lengths_tensor)


def rouge_l_score(prediction: str, reference: str) -> float:
    pred_tokens = prediction.split()
    ref_tokens = reference.split()
    if not pred_tokens or not ref_tokens:
        return 0.0

    dp = [[0] * (len(ref_tokens) + 1) for _ in range(len(pred_tokens) + 1)]
    for i, pred_token in enumerate(pred_tokens, start=1):
        for j, ref_token in enumerate(ref_tokens, start=1):
            if pred_token == ref_token:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs = dp[-1][-1]
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def exact_match(prediction: str, reference: str) -> float:
    return float(prediction.strip().lower() == reference.strip().lower())


def run_shape_test() -> dict[str, tuple[int, ...]]:
    class _FakeConfig:
        vocab_size = 32128

    class _FakeT5(nn.Module):
        config = _FakeConfig()

        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(512, 1)

        def forward(self, inputs_embeds: torch.Tensor, labels: torch.Tensor | None = None):
            _ = self.proj(inputs_embeds)
            return type("FakeT5Output", (), {"loss": torch.tensor(0.0, device=inputs_embeds.device)})()

    model = MotionConformerT5(_FakeT5())
    model.eval()
    x = torch.randn(2, 60, 540)
    with torch.no_grad():
        out = model(x)

    assert out["z"].shape == (2, 15, 512)
    assert out["reconstruction"].shape == (2, 60, 540)
    assert out["ctc_logits"].shape[0:2] == (2, 15)

    phase1_words = MotionConformerPhase1Words(num_words=128)
    phase1_words.eval()
    word_x = torch.randn(5, 60, 540)
    with torch.no_grad():
        word_out = phase1_words(word_x, [2, 3])

    assert word_out["z"].shape == (5, 15, 512)
    assert word_out["reconstruction"].shape == (5, 60, 540)
    assert word_out["word_logits"].shape == (2, 128)

    labels = torch.zeros(2, 128)
    labels[0, :3] = 1.0
    labels[1, 4:8] = 1.0
    loss, _ = compute_phase1_word_losses(
        word_out,
        windows=word_x,
        labels=labels,
        bce_loss_fn=nn.BCEWithLogitsLoss(),
        cfg=Phase1WordsTrainConfig(top_k_words=128),
    )
    assert loss.item() > 0

    return {
        "z": tuple(out["z"].shape),
        "reconstruction": tuple(out["reconstruction"].shape),
        "ctc_logits": tuple(out["ctc_logits"].shape),
        "phase1_words_z": tuple(word_out["z"].shape),
        "phase1_words_reconstruction": tuple(word_out["reconstruction"].shape),
        "phase1_words_logits": tuple(word_out["word_logits"].shape),
    }


def train_motion_conformer_smoke(cfg: TrainConfig) -> dict[str, Any]:
    import numpy as np
    import torch
    import torch.nn as nn
    from transformers import T5ForConditionalGeneration, T5Tokenizer

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = T5Tokenizer.from_pretrained(cfg.t5_model)
    t5 = T5ForConditionalGeneration.from_pretrained(cfg.t5_model).to(device)
    model = MotionConformerT5(t5, freeze_t5=cfg.freeze_t5).to(device)

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=cfg.lr,
    )
    reconstruction_loss_fn = nn.MSELoss()
    ctc_loss_fn = nn.CTCLoss(blank=model.ctc_blank, zero_infinity=True)

    dataset = How2SignWindowDataset(
        repo_id=cfg.landmarks_repo,
        split=cfg.split,
        max_samples=cfg.max_samples,
        window=cfg.window_size,
        stride=cfg.stride,
    )

    stats: dict[str, Any] = {
        "device": device,
        "samples": 0,
        "steps": 0,
        "reconstruction_loss": 0.0,
        "ctc_loss": 0.0,
        "translation_loss": 0.0,
    }

    batch_features: list[torch.Tensor] = []
    batch_sentences: list[str] = []

    model.train()
    for row in dataset:
        batch_features.append(row["features"])
        batch_sentences.append(row["sentence"])

        if len(batch_features) < cfg.batch_size:
            continue

        x = torch.stack(batch_features).to(device)
        labels = tokenizer(
            batch_sentences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        ).input_ids.to(device)
        labels[labels == tokenizer.pad_token_id] = -100

        outputs = model(x, labels=labels)
        reconstruction_loss = reconstruction_loss_fn(outputs["reconstruction"], x)
        translation_loss = outputs["t5_outputs"].loss
        ctc_loss = compute_t5_subword_ctc_loss(
            tokenizer=tokenizer,
            ctc_loss_fn=ctc_loss_fn,
            ctc_logits=outputs["ctc_logits"],
            sentences=batch_sentences,
            device=device,
        )

        loss = (
            cfg.reconstruction_weight * reconstruction_loss
            + translation_loss
            + cfg.ctc_weight * ctc_loss
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        stats["samples"] += len(batch_features)
        stats["steps"] += 1
        stats["reconstruction_loss"] += float(reconstruction_loss.detach().cpu())
        stats["ctc_loss"] += float(ctc_loss.detach().cpu())
        stats["translation_loss"] += float(translation_loss.detach().cpu())

        batch_features = []
        batch_sentences = []

    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    checkpoint_path = os.path.join(cfg.ckpt_dir, "motion_conformer_t5_smoke.pt")
    config_path = os.path.join(cfg.ckpt_dir, "smoke_config.json")

    torch.save(model.state_dict(), checkpoint_path)

    summary = {
        **stats,
        "avg_reconstruction_loss": stats["reconstruction_loss"] / max(stats["steps"], 1),
        "avg_ctc_loss": stats["ctc_loss"] / max(stats["steps"], 1),
        "avg_translation_loss": stats["translation_loss"] / max(stats["steps"], 1),
        "checkpoint": checkpoint_path,
        "config": asdict(cfg),
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    volume.commit()
    return summary


def train_phase1_words(cfg: Phase1WordsTrainConfig) -> dict[str, Any]:
    import numpy as np
    import torch

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    vocab, doc_counts, observed_docs = build_word_vocab(
        repo_id=cfg.landmarks_repo,
        split=cfg.split,
        max_samples=cfg.max_samples,
        top_k_words=cfg.top_k_words,
        min_word_len=cfg.min_word_len,
    )
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    observed_docs = max(observed_docs, max(doc_counts))
    pos_weight = torch.tensor(
        [
            min(max((observed_docs - count) / max(count, 1), 1.0), cfg.pos_weight_clip)
            for count in doc_counts
        ],
        dtype=torch.float32,
        device=device,
    )

    model = MotionConformerPhase1Words(num_words=len(vocab)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    bce_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    stats: dict[str, Any] = {
        "device": device,
        "utterances": 0,
        "windows": 0,
        "steps": 0,
        "total_loss": 0.0,
        "full_recon_loss": 0.0,
        "pos_recon_loss": 0.0,
        "vel_recon_loss": 0.0,
        "word_bce": 0.0,
        "latent_smooth_loss": 0.0,
        "latent_reg_loss": 0.0,
    }

    model.train()
    for epoch in range(cfg.epochs):
        dataset = How2SignUtteranceWordDataset(
            repo_id=cfg.landmarks_repo,
            split=cfg.split,
            word_to_idx=word_to_idx,
            max_samples=cfg.max_samples,
            window=cfg.window_size,
            stride=cfg.stride,
            max_windows_per_utterance=cfg.max_windows_per_utterance,
            min_word_len=cfg.min_word_len,
        )

        for batch in batched(dataset, cfg.batch_size):
            windows, window_counts, labels = pack_utterance_batch(batch, device=device)
            outputs = model(windows, window_counts)
            loss, loss_dict = compute_phase1_word_losses(
                outputs,
                windows=windows,
                labels=labels,
                bce_loss_fn=bce_loss_fn,
                cfg=cfg,
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

            stats["utterances"] += len(batch)
            stats["windows"] += int(windows.shape[0])
            stats["steps"] += 1
            for key in (
                "total_loss",
                "full_recon_loss",
                "pos_recon_loss",
                "vel_recon_loss",
                "word_bce",
                "latent_smooth_loss",
                "latent_reg_loss",
            ):
                stats[key] += loss_dict[key]

    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    checkpoint_path = os.path.join(cfg.ckpt_dir, "phase1_words.pt")
    vocab_path = os.path.join(cfg.ckpt_dir, "word_vocab.json")
    summary_path = os.path.join(cfg.ckpt_dir, "train_summary.json")

    checkpoint = {
        "model_state": model.state_dict(),
        "vocab": vocab,
        "doc_counts": doc_counts,
        "observed_docs": observed_docs,
        "pos_weight": pos_weight.detach().cpu().tolist(),
        "config": asdict(cfg),
    }
    torch.save(checkpoint, checkpoint_path)

    vocab_payload = {
        "vocab": vocab,
        "doc_counts": doc_counts,
        "observed_docs": observed_docs,
        "top_k_words": cfg.top_k_words,
        "min_word_len": cfg.min_word_len,
    }
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab_payload, f, indent=2)

    summary = {
        **stats,
        "avg_total_loss": stats["total_loss"] / max(stats["steps"], 1),
        "avg_full_recon_loss": stats["full_recon_loss"] / max(stats["steps"], 1),
        "avg_pos_recon_loss": stats["pos_recon_loss"] / max(stats["steps"], 1),
        "avg_vel_recon_loss": stats["vel_recon_loss"] / max(stats["steps"], 1),
        "avg_word_bce": stats["word_bce"] / max(stats["steps"], 1),
        "avg_latent_smooth_loss": stats["latent_smooth_loss"] / max(stats["steps"], 1),
        "avg_latent_reg_loss": stats["latent_reg_loss"] / max(stats["steps"], 1),
        "checkpoint": checkpoint_path,
        "word_vocab": vocab_path,
        "config": asdict(cfg),
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    volume.commit()
    return summary


def evaluate_phase1_words(cfg: Phase1WordsEvalConfig) -> dict[str, Any]:
    import numpy as np
    import torch

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.exists(cfg.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {cfg.checkpoint_path}")
    if not os.path.exists(cfg.vocab_path):
        raise FileNotFoundError(f"Word vocabulary not found: {cfg.vocab_path}")

    checkpoint = torch.load(cfg.checkpoint_path, map_location=device)
    with open(cfg.vocab_path, encoding="utf-8") as f:
        vocab_payload = json.load(f)

    vocab = vocab_payload["vocab"]
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    min_word_len = int(vocab_payload.get("min_word_len", 3))

    model = MotionConformerPhase1Words(num_words=len(vocab)).to(device)
    model.load_state_dict(checkpoint["model_state"] if "model_state" in checkpoint else checkpoint)
    model.eval()

    baseline = MotionConformerPhase1Words(num_words=len(vocab)).to(device)
    baseline.eval()

    pos_weight_values = checkpoint.get("pos_weight") if isinstance(checkpoint, dict) else None
    pos_weight = (
        torch.tensor(pos_weight_values, dtype=torch.float32, device=device)
        if pos_weight_values is not None
        else torch.ones(len(vocab), dtype=torch.float32, device=device)
    )
    bce_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    dataset = How2SignUtteranceWordDataset(
        repo_id=cfg.landmarks_repo,
        split=cfg.split,
        word_to_idx=word_to_idx,
        max_samples=cfg.max_samples,
        window=cfg.window_size,
        stride=cfg.stride,
        max_windows_per_utterance=cfg.max_windows_per_utterance,
        min_word_len=min_word_len,
    )

    totals = {
        "full_recon_loss": 0.0,
        "pos_recon_loss": 0.0,
        "vel_recon_loss": 0.0,
        "word_bce": 0.0,
        "latent_smooth_loss": 0.0,
        "latent_reg_loss": 0.0,
    }
    logits_all: list[torch.Tensor] = []
    baseline_logits_all: list[torch.Tensor] = []
    labels_all: list[torch.Tensor] = []
    prediction_rows: list[dict[str, Any]] = []
    utterances = 0
    windows_seen = 0
    steps = 0

    with torch.no_grad():
        for batch in batched(dataset, cfg.batch_size):
            windows, window_counts, labels = pack_utterance_batch(batch, device=device)
            outputs = model(windows, window_counts)
            _, loss_dict = compute_phase1_word_losses(
                outputs,
                windows=windows,
                labels=labels,
                bce_loss_fn=bce_loss_fn,
                cfg=cfg,
            )
            baseline_outputs = baseline(windows, window_counts)

            for key in totals:
                totals[key] += loss_dict[key]

            logits_all.append(outputs["word_logits"].detach().cpu())
            baseline_logits_all.append(baseline_outputs["word_logits"].detach().cpu())
            labels_all.append(labels.detach().cpu())

            top_words = decode_top_words(outputs["word_logits"], vocab=vocab, k=5)
            baseline_top_words = decode_top_words(baseline_outputs["word_logits"], vocab=vocab, k=5)
            for row, predicted_words, base_words, label_row in zip(batch, top_words, baseline_top_words, labels):
                true_indices = torch.nonzero(label_row > 0.5, as_tuple=False).flatten().detach().cpu().tolist()
                prediction_rows.append(
                    {
                        "video_id": row["video_id"],
                        "sentence": row["sentence"],
                        "true_words": [vocab[idx] for idx in true_indices],
                        "predicted_words": predicted_words,
                        "baseline_predicted_words": base_words,
                    }
                )

            utterances += len(batch)
            windows_seen += int(windows.shape[0])
            steps += 1

    if logits_all:
        logits = torch.cat(logits_all)
        baseline_logits = torch.cat(baseline_logits_all)
        labels = torch.cat(labels_all)
    else:
        logits = torch.empty(0, len(vocab))
        baseline_logits = torch.empty(0, len(vocab))
        labels = torch.empty(0, len(vocab))

    metric_values = word_classification_metrics(logits, labels)
    baseline_metrics = word_classification_metrics(baseline_logits, labels)
    predicted_sets = {tuple(row["predicted_words"]) for row in prediction_rows}
    baseline_sets = {tuple(row["baseline_predicted_words"]) for row in prediction_rows}

    result = {
        "device": device,
        "split": cfg.split,
        "checkpoint": cfg.checkpoint_path,
        "vocab_size": len(vocab),
        "utterances": utterances,
        "windows": windows_seen,
        "steps": steps,
        "val/full_recon_loss": totals["full_recon_loss"] / max(steps, 1),
        "val/pos_recon_loss": totals["pos_recon_loss"] / max(steps, 1),
        "val/vel_recon_loss": totals["vel_recon_loss"] / max(steps, 1),
        "val/latent_smooth_loss": totals["latent_smooth_loss"] / max(steps, 1),
        "val/latent_reg_loss": totals["latent_reg_loss"] / max(steps, 1),
        "val/word_bce": totals["word_bce"] / max(steps, 1),
        **{f"val/{key}": value for key, value in metric_values.items()},
        **{f"baseline/{key}": value for key, value in baseline_metrics.items()},
        "unique_predicted_word_sets": len(predicted_sets),
        "baseline_unique_predicted_word_sets": len(baseline_sets),
        "beats_untrained_baseline": (
            metric_values["micro_f1@0.5"] > baseline_metrics["micro_f1@0.5"]
            or metric_values["hit@5"] > baseline_metrics["hit@5"]
        ),
        "predictions_vary": len(predicted_sets) > 1,
        "config": asdict(cfg),
    }

    os.makedirs(cfg.output_dir, exist_ok=True)
    with open(os.path.join(cfg.output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    with open(os.path.join(cfg.output_dir, "predictions.jsonl"), "w", encoding="utf-8") as f:
        for row in prediction_rows:
            f.write(json.dumps(row) + "\n")

    volume.commit()
    return result


def evaluate_motion_conformer_checkpoint(cfg: EvalConfig) -> dict[str, Any]:
    import numpy as np
    import sacrebleu
    import torch
    from transformers import T5ForConditionalGeneration, T5Tokenizer

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.exists(cfg.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {cfg.checkpoint_path}")

    tokenizer = T5Tokenizer.from_pretrained(cfg.t5_model)
    t5 = T5ForConditionalGeneration.from_pretrained(cfg.t5_model).to(device)
    model = MotionConformerT5(t5, freeze_t5=True).to(device)

    state = torch.load(cfg.checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    dataset = How2SignWindowDataset(
        repo_id=cfg.landmarks_repo,
        split=cfg.split,
        max_samples=cfg.max_samples,
        window=cfg.window_size,
        stride=cfg.stride,
    )

    loss_total = 0.0
    steps = 0
    windows = 0
    predictions: list[str] = []
    references: list[str] = []
    video_ids: list[str] = []

    batch_features: list[torch.Tensor] = []
    batch_sentences: list[str] = []
    batch_video_ids: list[str] = []

    with torch.no_grad():
        for row in dataset:
            batch_features.append(row["features"])
            batch_sentences.append(row["sentence"])
            batch_video_ids.append(row["video_id"])

            if len(batch_features) < cfg.batch_size:
                continue

            x = torch.stack(batch_features).to(device)

            labels = tokenizer(
                batch_sentences,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128,
            ).input_ids.to(device)
            labels[labels == tokenizer.pad_token_id] = -100

            outputs = model(x, labels=labels)
            loss_total += float(outputs["t5_outputs"].loss.detach().cpu())
            steps += 1
            windows += len(batch_features)

            generated = model.t5.generate(
                inputs_embeds=model.adapter(outputs["z"]),
                num_beams=cfg.num_beams,
                max_new_tokens=cfg.max_new_tokens,
            )

            decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
            predictions.extend(decoded)
            references.extend(batch_sentences)
            video_ids.extend(batch_video_ids)

            batch_features = []
            batch_sentences = []
            batch_video_ids = []

    bleu = sacrebleu.corpus_bleu(predictions, [references]).score if predictions else 0.0
    chrf = sacrebleu.corpus_chrf(predictions, [references]).score if predictions else 0.0
    rouge_l = (
        sum(rouge_l_score(prediction, reference) for prediction, reference in zip(predictions, references))
        / max(len(predictions), 1)
    )
    exact = (
        sum(exact_match(prediction, reference) for prediction, reference in zip(predictions, references))
        / max(len(predictions), 1)
    )

    result = {
        "device": device,
        "split": cfg.split,
        "checkpoint": cfg.checkpoint_path,
        "windows": windows,
        "steps": steps,
        "validation_loss": loss_total / max(steps, 1),
        "bleu": bleu,
        "chrf": chrf,
        "rouge_l": rouge_l,
        "exact_match": exact,
        "config": asdict(cfg),
    }

    os.makedirs(cfg.output_dir, exist_ok=True)
    with open(os.path.join(cfg.output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    with open(os.path.join(cfg.output_dir, "predictions.jsonl"), "w", encoding="utf-8") as f:
        for video_id, prediction, reference in zip(video_ids, predictions, references):
            f.write(
                json.dumps(
                    {
                        "video_id": video_id,
                        "prediction": prediction,
                        "reference": reference,
                    }
                )
                + "\n"
            )

    volume.commit()
    return result


def evaluate_phase1_reconstruction_checkpoint(cfg: Phase1EvalConfig) -> dict[str, Any]:
    import numpy as np
    import torch
    import torch.nn as nn
    from transformers import T5ForConditionalGeneration

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.exists(cfg.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {cfg.checkpoint_path}")

    t5 = T5ForConditionalGeneration.from_pretrained(cfg.t5_model).to(device)
    model = MotionConformerT5(t5, freeze_t5=True).to(device)
    state = torch.load(cfg.checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    dataset = How2SignWindowDataset(
        repo_id=cfg.landmarks_repo,
        split=cfg.split,
        max_samples=cfg.max_samples,
        window=cfg.window_size,
        stride=cfg.stride,
    )

    mse = nn.MSELoss(reduction="mean")
    total_recon_loss = 0.0
    total_pos_loss = 0.0
    total_vel_loss = 0.0
    total_latent_smooth_loss = 0.0
    total_latent_reg_loss = 0.0
    z_sum = 0.0
    z_sq_sum = 0.0
    z_count = 0
    windows = 0
    steps = 0

    batch_features: list[torch.Tensor] = []

    with torch.no_grad():
        for row in dataset:
            batch_features.append(row["features"])
            if len(batch_features) < cfg.batch_size:
                continue

            x = torch.stack(batch_features).to(device)
            z = model.encoder(x)
            reconstruction = model.reconstruction_decoder(z)

            pos = x[:, :, :270]
            vel = x[:, :, 270:]
            recon_pos = reconstruction[:, :, :270]
            recon_vel = reconstruction[:, :, 270:]

            total_recon_loss += float(mse(reconstruction, x).detach().cpu())
            total_pos_loss += float(mse(recon_pos, pos).detach().cpu())
            total_vel_loss += float(mse(recon_vel, vel).detach().cpu())
            if z.shape[1] > 1:
                total_latent_smooth_loss += float(((z[:, 1:] - z[:, :-1]) ** 2).mean().detach().cpu())
            total_latent_reg_loss += float((z**2).mean().detach().cpu())

            z_sum += float(z.sum().detach().cpu())
            z_sq_sum += float((z**2).sum().detach().cpu())
            z_count += z.numel()
            windows += len(batch_features)
            steps += 1
            batch_features = []

    z_mean = z_sum / max(z_count, 1)
    z_variance = max((z_sq_sum / max(z_count, 1)) - (z_mean**2), 0.0)
    z_std = z_variance**0.5

    result = {
        "device": device,
        "split": cfg.split,
        "checkpoint": cfg.checkpoint_path,
        "windows": windows,
        "steps": steps,
        "val/full_recon_loss": total_recon_loss / max(steps, 1),
        "val/masked_pos_loss": total_pos_loss / max(steps, 1),
        "val/masked_vel_loss": total_vel_loss / max(steps, 1),
        "val/latent_smooth_loss": total_latent_smooth_loss / max(steps, 1),
        "val/latent_reg_loss": total_latent_reg_loss / max(steps, 1),
        "z_mean": z_mean,
        "z_std": z_std,
        "config": asdict(cfg),
    }

    os.makedirs(cfg.output_dir, exist_ok=True)
    with open(os.path.join(cfg.output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    volume.commit()
    return result


@app.function(
    image=image,
    gpu="L4",
    timeout=2 * 60 * 60,
    volumes={"/checkpoints": volume},
)
def train_smoke(
    max_samples: int = 100,
    landmarks_repo: str = LANDMARKS_REPO,
    split: str = "train",
    batch_size: int = 4,
) -> dict[str, Any]:
    cfg = TrainConfig(
        landmarks_repo=landmarks_repo,
        split=split,
        max_samples=max_samples,
        batch_size=batch_size,
    )
    data_source_check = _check_data_source_impl(cfg.landmarks_repo, cfg.split)
    result = train_motion_conformer_smoke(cfg)
    result["data_source_check"] = data_source_check
    print(json.dumps(result, indent=2))
    return result


@app.function(
    image=image,
    gpu="L4",
    timeout=24 * 60 * 60,
    volumes={"/checkpoints": volume},
)
def train_full(
    landmarks_repo: str = LANDMARKS_REPO,
    split: str = "train",
    batch_size: int = 4,
) -> dict[str, Any]:
    cfg = TrainConfig(
        landmarks_repo=landmarks_repo,
        split=split,
        max_samples=None,
        batch_size=batch_size,
        ckpt_dir="/checkpoints/motion_conformer_t5_full",
    )
    data_source_check = _check_data_source_impl(cfg.landmarks_repo, cfg.split)
    result = train_motion_conformer_smoke(cfg)
    result["data_source_check"] = data_source_check
    print(json.dumps(result, indent=2))
    return result


@app.function(
    image=image,
    gpu="L4",
    timeout=12 * 60 * 60,
    volumes={"/checkpoints": volume},
)
def evaluate_full_checkpoint(
    split: str = "validation",
    batch_size: int = 4,
    max_samples: int = 0,
) -> dict[str, Any]:
    output_dir = f"/checkpoints/motion_conformer_t5_full/eval_{split}"
    cfg = EvalConfig(
        split=split,
        batch_size=batch_size,
        max_samples=_normalize_max_samples(max_samples),
        output_dir=output_dir,
    )
    result = evaluate_motion_conformer_checkpoint(cfg)
    print(json.dumps(result, indent=2))
    return result


@app.function(
    image=image,
    gpu="L4",
    timeout=4 * 60 * 60,
    volumes={"/checkpoints": volume},
)
def evaluate_phase1_checkpoint(
    split: str = "validation",
    batch_size: int = 16,
    max_samples: int = 0,
) -> dict[str, Any]:
    output_dir = f"/checkpoints/motion_conformer_t5_full/eval_phase1_{split}"
    cfg = Phase1EvalConfig(
        split=split,
        batch_size=batch_size,
        max_samples=_normalize_max_samples(max_samples),
        output_dir=output_dir,
    )
    result = evaluate_phase1_reconstruction_checkpoint(cfg)
    print(json.dumps(result, indent=2))
    return result


@app.function(
    image=image,
    gpu="L4",
    timeout=2 * 60 * 60,
    volumes={"/checkpoints": volume},
)
def train_phase1_words_smoke(
    max_samples: int = 20,
    landmarks_repo: str = LANDMARKS_REPO,
    split: str = "train",
    batch_size: int = 2,
    top_k_words: int = 128,
    max_windows_per_utterance: int = 4,
) -> dict[str, Any]:
    cfg = Phase1WordsTrainConfig(
        landmarks_repo=landmarks_repo,
        split=split,
        max_samples=max_samples,
        batch_size=batch_size,
        top_k_words=top_k_words,
        max_windows_per_utterance=max_windows_per_utterance,
        ckpt_dir="/checkpoints/motion_conformer_phase1_words_smoke",
    )
    data_source_check = _check_data_source_impl(cfg.landmarks_repo, cfg.split)
    result = train_phase1_words(cfg)
    result["data_source_check"] = data_source_check
    print(json.dumps(result, indent=2))
    return result


@app.function(
    image=image,
    gpu="L4",
    timeout=24 * 60 * 60,
    volumes={"/checkpoints": volume},
)
def train_phase1_words_full(
    landmarks_repo: str = LANDMARKS_REPO,
    split: str = "train",
    batch_size: int = 2,
    top_k_words: int = 512,
    max_windows_per_utterance: int = 12,
) -> dict[str, Any]:
    cfg = Phase1WordsTrainConfig(
        landmarks_repo=landmarks_repo,
        split=split,
        max_samples=None,
        batch_size=batch_size,
        top_k_words=top_k_words,
        max_windows_per_utterance=max_windows_per_utterance,
        ckpt_dir="/checkpoints/motion_conformer_phase1_words_full",
    )
    data_source_check = _check_data_source_impl(cfg.landmarks_repo, cfg.split)
    result = train_phase1_words(cfg)
    result["data_source_check"] = data_source_check
    print(json.dumps(result, indent=2))
    return result


@app.function(
    image=image,
    gpu="L4",
    timeout=6 * 60 * 60,
    volumes={"/checkpoints": volume},
)
def evaluate_phase1_words_checkpoint(
    split: str = "validation",
    batch_size: int = 4,
    max_samples: int = 0,
    checkpoint_dir: str = "/checkpoints/motion_conformer_phase1_words_full",
    max_windows_per_utterance: int = 12,
) -> dict[str, Any]:
    output_dir = os.path.join(checkpoint_dir, f"eval_{split}")
    cfg = Phase1WordsEvalConfig(
        split=split,
        batch_size=batch_size,
        max_samples=_normalize_max_samples(max_samples),
        max_windows_per_utterance=max_windows_per_utterance,
        checkpoint_path=os.path.join(checkpoint_dir, "phase1_words.pt"),
        vocab_path=os.path.join(checkpoint_dir, "word_vocab.json"),
        output_dir=output_dir,
    )
    result = evaluate_phase1_words(cfg)
    print(json.dumps(result, indent=2))
    return result


@app.local_entrypoint()
def main(max_samples: int = 100, full: bool = False):
    cfg = Phase1WordsTrainConfig(max_samples=_normalize_max_samples(max_samples))
    print(check_data_source.remote(cfg.landmarks_repo, cfg.split))
    if full:
        print(
            train_phase1_words_full.remote(
                landmarks_repo=cfg.landmarks_repo,
                split=cfg.split,
                batch_size=cfg.batch_size,
                top_k_words=cfg.top_k_words,
                max_windows_per_utterance=cfg.max_windows_per_utterance,
            )
        )
    else:
        print(
            train_phase1_words_smoke.remote(
                max_samples=max_samples,
                landmarks_repo=cfg.landmarks_repo,
                split=cfg.split,
                batch_size=cfg.batch_size,
                top_k_words=128,
                max_windows_per_utterance=4,
            )
        )
