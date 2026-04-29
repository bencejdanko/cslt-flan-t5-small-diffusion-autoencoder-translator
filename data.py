"""
data.py — Dataset classes, feature engineering, and collation for CSLT.

Key design decisions:
  - UtteranceLevelDataset: loads full utterances (no chunking) so the model
    sees the complete sign sequence paired with its sentence.  This eliminates
    the noisy "each chunk gets the whole sentence" supervision.
  - SignLanguageCollator: variable-length padding, mask creation, and optional
    text tokenization — all in one place.
  - Feature engineering is identical to the v2 notebooks but factored into
    reusable functions.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset

from config import (
    FACE_LANDMARK_IDXS,
    FACE_SLICE,
    LHAND_SLICE,
    PART_KEYS,
    POSE_SLICE,
    RHAND_SLICE,
)


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
# Utterance-level dataset (no chunking)
# ---------------------------------------------------------------------------
class UtteranceLevelDataset(IterableDataset):
    """
    Streams full utterances from the HuggingFace dataset.

    Each sample is a complete sign sequence paired with its sentence.
    No sliding window / chunking — variable lengths are handled by the collator.
    """

    def __init__(
        self,
        split: str = "train",
        max_samples: Optional[int] = None,
        repo_id: str = "bdanko/how2sign-landmarks-front-raw-parquet",
    ):
        self.split = split
        self.max_samples = max_samples
        self.repo_id = repo_id

    def __iter__(self):
        from datasets import load_dataset

        ds = load_dataset(self.repo_id, split=self.split, streaming=True)
        count = 0
        for sample in ds:
            if self.max_samples and count >= self.max_samples:
                break
            raw = np.frombuffer(
                sample["features"], dtype=np.float32
            ).reshape(sample["shape"])
            feat_dict = engineer_features_multistream(raw)
            if feat_dict is None:
                continue

            seq_len = feat_dict["body_pos"].shape[0]
            sentence = sample.get("sentence", "")

            yield {
                "features": feat_dict,
                "seq_len": seq_len,
                "sentence": sentence,
            }
            count += 1


# ---------------------------------------------------------------------------
# Collator
# ---------------------------------------------------------------------------
class SignLanguageCollator:
    """
    Batches variable-length sign sequences with proper padding and masks.

    Handles:
      - Padding each part tensor to the max length in the batch
      - Creating src_key_padding_mask (True = padding position)
      - Optional text tokenization (for Phase 2)
      - Setting label pad tokens to -100 for CrossEntropy
    """

    def __init__(
        self,
        tokenizer=None,
        max_target_length: int = 128,
        phase: int = 1,
    ):
        self.tokenizer = tokenizer
        self.max_target_length = max_target_length
        self.phase = phase

    def __call__(
        self, batch: List[dict]
    ) -> dict:
        """
        Args:
            batch: list of dicts from UtteranceLevelDataset

        Returns:
            dict with:
              - features: dict of padded tensors [B, T_max, dim]
              - padding_mask: [B, T_max] bool
              - seq_lens: [B] int tensor
              - sentences: list of str (if phase 2)
              - labels: [B, L] token ids with -100 for padding (if phase 2)
        """
        seq_lens = [item["seq_len"] for item in batch]
        max_len = max(seq_lens)

        # Pad features
        padded_features = {}
        for key in PART_KEYS:
            tensors = []
            for item in batch:
                t = item["features"][key]  # [T_i, dim]
                pad_len = max_len - t.shape[0]
                if pad_len > 0:
                    t = F.pad(t, (0, 0, 0, pad_len))  # pad time dim
                tensors.append(t)
            padded_features[key] = torch.stack(tensors)  # [B, T_max, dim]

        # Padding mask: True where padded
        padding_mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
        for i, sl in enumerate(seq_lens):
            padding_mask[i, sl:] = True

        result = {
            "features": padded_features,
            "padding_mask": padding_mask,
            "seq_lens": torch.tensor(seq_lens, dtype=torch.long),
        }

        # Phase 2: tokenize sentences
        if self.phase == 2 and self.tokenizer is not None:
            sentences = [item["sentence"] for item in batch]
            result["sentences"] = sentences

            tokenized = self.tokenizer(
                sentences,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_target_length,
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
) -> DataLoader:
    """Create a DataLoader with the utterance-level dataset and collator."""
    dataset = UtteranceLevelDataset(
        split=split, max_samples=max_samples, repo_id=repo_id
    )
    collator = SignLanguageCollator(
        tokenizer=tokenizer,
        max_target_length=max_target_length,
        phase=phase,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collator,
        num_workers=0,  # IterableDataset with streaming
    )
