"""
data.py — Dataset classes, feature engineering, and collation for CSLT.

Key design decisions:
  - UtteranceLevelDataset: loads full utterances (no chunking).
  - Handles both Streaming (Iterable) and Local (Map-style) modes.
  - Multi-worker support for Local mode.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, IterableDataset

from config import (
    FACE_LANDMARK_IDXS,
    FACE_SLICE,
    LHAND_SLICE,
    PART_KEYS,
    POSE_SLICE,
    RHAND_SLICE,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
def engineer_features_multistream(raw: np.ndarray) -> Optional[Dict[str, torch.Tensor]]:
    """
    Convert raw [T, 543, 3] MediaPipe landmarks into structured feature dict.

    Processing:
      - Downsample face mesh from 468 → 15 key points
      - Normalize each part relative to its anchor
      - Compute temporal velocity (delta)

    Returns:
        dict with keys: body_pos, body_vel, face_pos, face_vel,
                        lhand_pos, lhand_vel, rhand_pos, rhand_vel
        Each tensor: [T, part_dim]
        Returns None if sequence is too short.
    """
    T = raw.shape[0]
    if T < 2:
        return None

    pose = raw[:, POSE_SLICE, :]
    face = raw[:, FACE_SLICE, :][:, FACE_LANDMARK_IDXS, :]
    lhand = raw[:, LHAND_SLICE, :]
    rhand = raw[:, RHAND_SLICE, :]

    # Spatial normalization
    shoulder_center = (pose[:, 11, :] + pose[:, 12, :]) / 2.0
    pose_norm = pose - shoulder_center[:, None, :]

    face_center = face.mean(axis=1, keepdims=True)
    face_norm = face - face_center

    lhand_norm = lhand - lhand[:, 0:1, :]
    rhand_norm = rhand - rhand[:, 0:1, :]

    def calc_vel(x: np.ndarray) -> np.ndarray:
        vel = np.zeros_like(x)
        vel[1:] = x[1:] - x[:-1]
        return vel

    return {
        "body_pos": torch.from_numpy(pose_norm.reshape(T, -1).astype(np.float32)),
        "face_pos": torch.from_numpy(face_norm.reshape(T, -1).astype(np.float32)),
        "lhand_pos": torch.from_numpy(lhand_norm.reshape(T, -1).astype(np.float32)),
        "rhand_pos": torch.from_numpy(rhand_norm.reshape(T, -1).astype(np.float32)),
        "body_vel": torch.from_numpy(calc_vel(pose_norm).reshape(T, -1).astype(np.float32)),
        "face_vel": torch.from_numpy(calc_vel(face_norm).reshape(T, -1).astype(np.float32)),
        "lhand_vel": torch.from_numpy(calc_vel(lhand_norm).reshape(T, -1).astype(np.float32)),
        "rhand_vel": torch.from_numpy(calc_vel(rhand_norm).reshape(T, -1).astype(np.float32)),
    }

# ---------------------------------------------------------------------------
# Dataset Implementations
# ---------------------------------------------------------------------------

def _process_sample(sample):
    """Common sample processing logic."""
    raw = np.frombuffer(sample["features"], dtype=np.float32).reshape(sample["shape"])
    feat_dict = engineer_features_multistream(raw)
    if feat_dict is None:
        return None
    return {
        "features": feat_dict,
        "seq_len": feat_dict["body_pos"].shape[0],
        "sentence": sample.get("sentence", ""),
    }

class UtteranceLevelStreamingDataset(IterableDataset):
    """Streaming version (Iterable)."""
    def __init__(self, split, repo_id, max_samples, shuffle_buffer):
        self.split = split
        self.repo_id = repo_id
        self.max_samples = max_samples
        self.shuffle_buffer = shuffle_buffer

    def __iter__(self):
        from datasets import load_dataset
        ds = load_dataset(self.repo_id, split=self.split, streaming=True)
        if "train" in self.split and self.shuffle_buffer > 0:
            ds = ds.shuffle(seed=None, buffer_size=self.shuffle_buffer)
        
        count = 0
        for sample in ds:
            if self.max_samples and count >= self.max_samples:
                break
            out = _process_sample(sample)
            if out:
                yield out
                count += 1

class UtteranceLevelMapDataset(Dataset):
    """Local version (Map-style)."""
    def __init__(self, split, repo_id, max_samples):
        from datasets import load_dataset
        logger.info(f"Loading/Downloading dataset split '{split}'...")
        self.ds = load_dataset(repo_id, split=split, streaming=False)
        if max_samples:
            self.ds = self.ds.select(range(min(max_samples, len(self.ds))))
        logger.info(f"Dataset loaded: {len(self.ds)} samples.")

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        out = _process_sample(self.ds[idx])
        if out is None:
            # Return a small dummy or handle error
            return self.__getitem__((idx + 1) % len(self.ds))
        return out

# ---------------------------------------------------------------------------
# Collator
# ---------------------------------------------------------------------------
class SignLanguageCollator:
    def __init__(self, tokenizer=None, max_target_length=128, phase=1):
        self.tokenizer = tokenizer
        self.max_target_length = max_target_length
        self.phase = phase

    def __call__(self, batch: List[dict]) -> dict:
        batch = [b for b in batch if b is not None]
        if not batch: return {}

        seq_lens = [item["seq_len"] for item in batch]
        max_len = max(seq_lens)

        padded_features = {}
        for key in PART_KEYS:
            tensors = []
            for item in batch:
                t = item["features"][key]
                pad_len = max_len - t.shape[0]
                if pad_len > 0:
                    t = F.pad(t, (0, 0, 0, pad_len))
                tensors.append(t)
            padded_features[key] = torch.stack(tensors)

        padding_mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
        for i, sl in enumerate(seq_lens):
            padding_mask[i, sl:] = True

        result = {
            "features": padded_features,
            "padding_mask": padding_mask,
            "seq_lens": torch.tensor(seq_lens, dtype=torch.long),
        }

        if self.phase == 2 and self.tokenizer is not None:
            sentences = [item["sentence"] for item in batch]
            result["sentences"] = sentences
            tokenized = self.tokenizer(
                sentences, return_tensors="pt", padding=True, truncation=True,
                max_length=self.max_target_length
            )
            labels = tokenized.input_ids.clone()
            labels[labels == self.tokenizer.pad_token_id] = -100
            result["labels"] = labels

        return result

# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------
def create_dataloader(
    split: str,
    batch_size: int,
    max_samples: Optional[int] = None,
    repo_id: str = "bdanko/how2sign-landmarks-front-raw-parquet",
    tokenizer=None,
    max_target_length: int = 128,
    phase: int = 1,
    shuffle_buffer: int = 1000,
    streaming: bool = True,
) -> DataLoader:
    if streaming:
        dataset = UtteranceLevelStreamingDataset(split, repo_id, max_samples, shuffle_buffer)
        num_workers = 0
        shuffle = False # IterableDataset handles shuffle internally
    else:
        dataset = UtteranceLevelMapDataset(split, repo_id, max_samples)
        num_workers = 4 # Parallel preprocessing!
        shuffle = (split == "train")

    collator = SignLanguageCollator(tokenizer=tokenizer, max_target_length=max_target_length, phase=phase)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collator,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=True if torch.cuda.is_available() else False,
    )
