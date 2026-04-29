"""
models.py — All model architectures for the CSLT pipeline.

Contains:
  - PositionalEncoding: sinusoidal positional encoding
  - PartSpatialEncoder: per-body-part spatial projection
  - MultiStreamSemanticEncoder: multi-stream encoder with temporal transformer
  - DDPMNoiseSchedule: proper DDPM beta/alpha schedule
  - StructuredDiffusionDecoder: epsilon-predicting denoising decoder
  - LatentAlignmentHead: CTC-style auxiliary alignment loss
  - AttentionPooling: temporal aggregation via cross-attention
  - LatentToT5Adapter: projection from encoder latent to T5 hidden space
  - SignToTextModel: full Phase 2 encoder-to-text model
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import PART_DIMS, PART_KEYS, TOTAL_FEAT_DIM


# ---------------------------------------------------------------------------
# Positional encoding
# ---------------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        return x + self.pe[:, : x.size(1), :]


# ---------------------------------------------------------------------------
# Part-level spatial encoder
# ---------------------------------------------------------------------------
class PartSpatialEncoder(nn.Module):
    """Per-body-part spatial MLP with optional learned part embedding."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.LayerNorm(out_dim),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Multi-stream semantic encoder
# ---------------------------------------------------------------------------
class MultiStreamSemanticEncoder(nn.Module):
    """
    Multi-stream encoder that processes body parts independently then fuses.

    Supports:
      - Variable-length inputs via src_key_padding_mask
      - Optional learned part embeddings
      - Temporal downsampling via strided convolutions
    """

    def __init__(
        self,
        d_model: int = 384,
        latent_dim: int = 512,
        num_layers: int = 3,
        nhead: int = 8,
        dropout: float = 0.1,
        use_part_embeddings: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.latent_dim = latent_dim
        part_dim = d_model // 4

        # Per-part spatial encoders (input = pos + vel concatenated)
        self.body_encoder = PartSpatialEncoder(33 * 3 * 2, part_dim)
        self.face_encoder = PartSpatialEncoder(15 * 3 * 2, part_dim)
        self.lhand_encoder = PartSpatialEncoder(21 * 3 * 2, part_dim)
        self.rhand_encoder = PartSpatialEncoder(21 * 3 * 2, part_dim)

        # Optional learned part embeddings added before fusion
        self.use_part_embeddings = use_part_embeddings
        if use_part_embeddings:
            self.part_embeddings = nn.ParameterDict({
                "body": nn.Parameter(torch.randn(1, 1, part_dim) * 0.02),
                "face": nn.Parameter(torch.randn(1, 1, part_dim) * 0.02),
                "lhand": nn.Parameter(torch.randn(1, 1, part_dim) * 0.02),
                "rhand": nn.Parameter(torch.randn(1, 1, part_dim) * 0.02),
            })

        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(part_dim * 4, d_model), nn.GELU(), nn.LayerNorm(d_model)
        )

        # Temporal transformer
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.temporal = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Temporal downsampling (stride 2 twice → 4x compression)
        self.downsample = nn.Conv1d(
            d_model, latent_dim, kernel_size=3, stride=2, padding=1
        )
        self.downsample2 = nn.Conv1d(
            latent_dim, latent_dim, kernel_size=3, stride=2, padding=1
        )

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            inputs: dict with keys body_pos, body_vel, face_pos, face_vel, etc.
                    Each tensor: [B, T, feat_dim]
            src_key_padding_mask: [B, T] bool tensor, True = padding

        Returns:
            z: [B, T', latent_dim] where T' ≈ T/4
        """
        # Concatenate pos + vel for each part
        body = torch.cat([inputs["body_pos"], inputs["body_vel"]], dim=-1)
        face = torch.cat([inputs["face_pos"], inputs["face_vel"]], dim=-1)
        lhand = torch.cat([inputs["lhand_pos"], inputs["lhand_vel"]], dim=-1)
        rhand = torch.cat([inputs["rhand_pos"], inputs["rhand_vel"]], dim=-1)

        # Part-level encoding
        b_feat = self.body_encoder(body)
        f_feat = self.face_encoder(face)
        l_feat = self.lhand_encoder(lhand)
        r_feat = self.rhand_encoder(rhand)

        # Add part embeddings
        if self.use_part_embeddings:
            b_feat = b_feat + self.part_embeddings["body"]
            f_feat = f_feat + self.part_embeddings["face"]
            l_feat = l_feat + self.part_embeddings["lhand"]
            r_feat = r_feat + self.part_embeddings["rhand"]

        # Fusion
        fused = self.fusion(torch.cat([b_feat, f_feat, l_feat, r_feat], dim=-1))
        fused = self.pos_encoder(fused)

        # Temporal modeling
        temp_out = self.temporal(fused, src_key_padding_mask=src_key_padding_mask)

        # Temporal downsampling
        x = temp_out.transpose(1, 2)  # [B, D, T]
        x = F.gelu(self.downsample(x))
        x = F.gelu(self.downsample2(x))
        return x.transpose(1, 2)  # [B, T', latent_dim]


# ---------------------------------------------------------------------------
# DDPM noise schedule
# ---------------------------------------------------------------------------
class DDPMNoiseSchedule(nn.Module):
    """
    Proper DDPM noise schedule with precomputed alpha/beta tensors.

    Supports linear and cosine schedules.
    Training: predict epsilon (noise) from noisy input.
    """

    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        schedule_type: str = "linear",
    ):
        super().__init__()
        self.num_timesteps = num_timesteps

        if schedule_type == "linear":
            betas = torch.linspace(beta_start, beta_end, num_timesteps)
        elif schedule_type == "cosine":
            steps = torch.arange(num_timesteps + 1, dtype=torch.float64) / num_timesteps
            alpha_bar = torch.cos((steps + 0.008) / 1.008 * math.pi / 2) ** 2
            alpha_bar = alpha_bar / alpha_bar[0]
            betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
            betas = torch.clamp(betas, min=1e-5, max=0.999).float()
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer(
            "sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )

    def q_sample(
        self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None
    ) -> tuple:
        """
        Forward diffusion: add noise to x0.

        Args:
            x0: [B, T, D] clean signal
            t: [B] integer timesteps
            noise: optional pre-sampled noise

        Returns:
            (noisy_x, noise): tuple of noisy signal and the noise that was added
        """
        if noise is None:
            noise = torch.randn_like(x0)

        sqrt_alpha = self.sqrt_alphas_cumprod[t]  # [B]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t]  # [B]

        # Reshape for broadcasting: [B, 1, 1]
        while sqrt_alpha.dim() < x0.dim():
            sqrt_alpha = sqrt_alpha.unsqueeze(-1)
            sqrt_one_minus_alpha = sqrt_one_minus_alpha.unsqueeze(-1)

        noisy_x = sqrt_alpha * x0 + sqrt_one_minus_alpha * noise
        return noisy_x, noise


# ---------------------------------------------------------------------------
# Structured diffusion decoder (epsilon-predicting)
# ---------------------------------------------------------------------------
class StructuredDiffusionDecoder(nn.Module):
    """
    Denoising decoder that predicts the noise (epsilon) added to each body part.

    Architecture: shared conv stem → per-part prediction heads.
    Conditioned on encoder latent z and diffusion timestep t.
    """

    def __init__(self, latent_dim: int = 512):
        super().__init__()
        self.latent_dim = latent_dim

        # Timestep embedding (sinusoidal → MLP)
        self.time_dim = 128
        self.time_mlp = nn.Sequential(
            nn.Linear(self.time_dim, 256),
            nn.GELU(),
            nn.Linear(256, 128),
        )

        # Upsample z from T' to T
        self.upsample_z = nn.ConvTranspose1d(
            latent_dim, latent_dim, kernel_size=4, stride=4
        )

        # Shared convolutional stem
        self.shared_net = nn.Sequential(
            nn.Conv1d(TOTAL_FEAT_DIM + latent_dim + 128, 512, kernel_size=3, padding=1),
            nn.GELU(),
            nn.GroupNorm(8, 512),
            nn.Conv1d(512, 512, kernel_size=3, padding=1),
            nn.GELU(),
            nn.GroupNorm(8, 512),
        )

        # Per-part noise prediction heads
        self.heads = nn.ModuleDict({
            "body_pos": nn.Conv1d(512, 33 * 3, kernel_size=3, padding=1),
            "body_vel": nn.Conv1d(512, 33 * 3, kernel_size=3, padding=1),
            "face_pos": nn.Conv1d(512, 15 * 3, kernel_size=3, padding=1),
            "face_vel": nn.Conv1d(512, 15 * 3, kernel_size=3, padding=1),
            "lhand_pos": nn.Conv1d(512, 21 * 3, kernel_size=3, padding=1),
            "lhand_vel": nn.Conv1d(512, 21 * 3, kernel_size=3, padding=1),
            "rhand_pos": nn.Conv1d(512, 21 * 3, kernel_size=3, padding=1),
            "rhand_vel": nn.Conv1d(512, 21 * 3, kernel_size=3, padding=1),
        })

    def _sinusoidal_embedding(self, t: torch.Tensor) -> torch.Tensor:
        """Sinusoidal timestep embedding. t: [B] integer timesteps → [B, time_dim]."""
        half_dim = self.time_dim // 2
        emb = math.log(10000.0) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=torch.float) * -emb)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)  # [B, half_dim]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)  # [B, time_dim]

    def forward(
        self,
        noisy_flat: torch.Tensor,
        z: torch.Tensor,
        t: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Predict epsilon (noise) for each body part.

        Args:
            noisy_flat: [B, T, 540] concatenated noisy features
            z: [B, T', latent_dim] encoder latent
            t: [B] integer diffusion timesteps

        Returns:
            dict of predicted noise per part, each [B, T, part_dim]
        """
        B, T, _ = noisy_flat.shape
        x = noisy_flat.transpose(1, 2)  # [B, 540, T]

        # Timestep conditioning
        t_emb = self._sinusoidal_embedding(t)  # [B, time_dim]
        t_emb = self.time_mlp(t_emb)  # [B, 128]
        t_emb = t_emb.unsqueeze(-1).expand(-1, -1, T)  # [B, 128, T]

        # Upsample z
        z_up = self.upsample_z(z.transpose(1, 2))  # [B, latent_dim, ~T]
        if z_up.shape[-1] != T:
            z_up = F.interpolate(z_up, size=T, mode="linear", align_corners=False)

        # Shared stem
        shared_out = self.shared_net(
            torch.cat([x, z_up, t_emb], dim=1)
        )  # [B, 512, T]

        # Per-part heads
        pred = {}
        for key, head in self.heads.items():
            pred[key] = head(shared_out).transpose(1, 2)  # [B, T, part_dim]

        return pred


# ---------------------------------------------------------------------------
# CTC-style latent alignment head
# ---------------------------------------------------------------------------
class LatentAlignmentHead(nn.Module):
    """
    Auxiliary CTC alignment head over the encoder's temporal output.

    Since we do not have gloss annotations, this head learns a latent
    alignment vocabulary — providing a gradient signal for temporal structure.
    The CTC loss encourages the encoder to produce monotonic, segmented
    representations even without explicit boundary supervision.
    """

    def __init__(self, latent_dim: int = 512, vocab_size: int = 256):
        super().__init__()
        self.vocab_size = vocab_size
        # +1 for CTC blank token
        self.projection = nn.Linear(latent_dim, vocab_size + 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: [B, T', latent_dim] encoder output

        Returns:
            log_probs: [T', B, vocab_size+1] for CTC loss
        """
        logits = self.projection(z)  # [B, T', V+1]
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs.permute(1, 0, 2)  # [T', B, V+1] for CTC


# ---------------------------------------------------------------------------
# Attention pooling
# ---------------------------------------------------------------------------
class AttentionPooling(nn.Module):
    """
    Attention-based temporal pooling/aggregation.

    Uses a set of learned query vectors to attend over the encoder sequence,
    producing a fixed or reduced-length output. This is important for
    handling long sign sequences before feeding into T5.
    """

    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 4,
        num_queries: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.query = nn.Parameter(torch.randn(1, num_queries, d_model) * 0.02)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        z: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            z: [B, T', d_model] encoder output
            key_padding_mask: [B, T'] padding mask

        Returns:
            pooled: [B, num_queries, d_model]
        """
        B = z.size(0)
        queries = self.query.expand(B, -1, -1)  # [B, num_queries, d_model]
        attn_out, _ = self.attn(
            queries, z, z, key_padding_mask=key_padding_mask
        )
        return self.norm(attn_out)


# ---------------------------------------------------------------------------
# Latent-to-T5 adapter
# ---------------------------------------------------------------------------
class LatentToT5Adapter(nn.Module):
    """Projects encoder latent space into T5's embedding space."""

    def __init__(
        self,
        latent_dim: int = 512,
        t5_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, t5_dim),
            nn.GELU(),
            nn.LayerNorm(t5_dim),
            nn.Dropout(dropout),
            nn.Linear(t5_dim, t5_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


# ---------------------------------------------------------------------------
# Full sign-to-text model (Phase 2)
# ---------------------------------------------------------------------------
class SignToTextModel(nn.Module):
    """
    Full translation model: encoder → (optional pooling) → adapter → T5.

    Supports:
      - Frozen / unfrozen encoder
      - Optional attention pooling for long sequences
      - Optional CTC alignment head
    """

    def __init__(
        self,
        encoder: MultiStreamSemanticEncoder,
        latent_dim: int = 512,
        t5_name: str = "google/flan-t5-small",
        t5_dim: int = 512,
        adapter_dropout: float = 0.1,
        use_attention_pooling: bool = True,
        pool_num_heads: int = 4,
        pool_num_queries: int = 32,
        use_ctc_head: bool = False,
        ctc_vocab_size: int = 256,
    ):
        super().__init__()
        self.encoder = encoder
        self.adapter = LatentToT5Adapter(
            latent_dim=latent_dim, t5_dim=t5_dim, dropout=adapter_dropout
        )

        self.use_attention_pooling = use_attention_pooling
        if use_attention_pooling:
            self.attention_pool = AttentionPooling(
                d_model=latent_dim,
                num_heads=pool_num_heads,
                num_queries=pool_num_queries,
            )

        self.use_ctc_head = use_ctc_head
        if use_ctc_head:
            self.ctc_head = LatentAlignmentHead(
                latent_dim=latent_dim, vocab_size=ctc_vocab_size
            )

        from transformers import T5ForConditionalGeneration
        self.t5 = T5ForConditionalGeneration.from_pretrained(t5_name)

    def forward(
        self,
        batch_inputs: Dict[str, torch.Tensor],
        labels: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass for training.

        Returns:
            T5 model output (with .loss if labels provided)
            Optionally also returns CTC log probs for alignment loss.
        """
        z = self.encoder(batch_inputs, src_key_padding_mask=padding_mask)

        ctc_log_probs = None
        if self.use_ctc_head:
            ctc_log_probs = self.ctc_head(z)

        if self.use_attention_pooling:
            z = self.attention_pool(z)

        z = self.adapter(z)
        t5_out = self.t5(inputs_embeds=z, labels=labels)

        if ctc_log_probs is not None:
            return t5_out, ctc_log_probs
        return t5_out

    def generate(
        self,
        batch_inputs: Dict[str, torch.Tensor],
        padding_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Generate text tokens from sign input."""
        z = self.encoder(batch_inputs, src_key_padding_mask=padding_mask)
        if self.use_attention_pooling:
            z = self.attention_pool(z)
        z = self.adapter(z)
        return self.t5.generate(inputs_embeds=z, **kwargs)
