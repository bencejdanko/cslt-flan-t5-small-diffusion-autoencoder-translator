# Implemented Models and Architectures

This document catalogs all the neural network models, layers, and full architectures currently implemented in the repository. The codebase contains two distinct sets of model definitions:
1. The original Transformer/DDPM architecture (`models.py`)
2. The experimental Conformer architecture (`modal_motion_conformer_app.py`)

---

## 1. Original Architecture (`models.py`)

This file contains the models used in the baseline two-phase training approach (Phase 1: DDPM Autoencoder, Phase 2: Translation).

### 1.1 Encoders & Projections
- **`PartSpatialEncoder`**: A simple MLP (Linear $\rightarrow$ GELU $\rightarrow$ LayerNorm $\rightarrow$ Linear) used to independently project the concatenated position and velocity features of specific body parts. Input dimensions are 198 (body), 90 (face), 126 (left hand), and 126 (right hand), which are all projected to a common `part_dim` (default 96).
- **`MultiStreamSemanticEncoder`**: The core representation learning model. It processes the body parts using `PartSpatialEncoder`s. It optionally adds learned part embeddings, fuses the streams via an MLP into a 384-dimensional space, applies sinusoidal `PositionalEncoding`, and runs the sequence through a standard multi-layer `TransformerEncoder` (default 3 layers, 8 heads). Finally, it applies two strided 1D convolutions (kernel size 3, stride 2) to compress the temporal dimension by a factor of 4, producing the 512-dimensional latent representation $Z$.

### 1.2 Phase 1: Diffusion Components
- **`DDPMNoiseSchedule`**: Implements the forward diffusion process. Supports both `linear` and `cosine` beta schedules. Precomputes alphas and betas for injecting noise into the clean motion signal over 1000 timesteps.
- **`StructuredDiffusionDecoder`**: The denoising model used to train the semantic encoder. It takes noisy motion features, the diffusion timestep $t$, and the encoder latent $Z$. The latent $Z$ is first upsampled 4x via `ConvTranspose1d` to match the temporal resolution. Then, a shared 1D convolutional stem with GroupNorm(8) is followed by separate 1D convolutional heads to predict the noise ($\epsilon$) added to each specific body part's position and velocity.

### 1.3 Phase 2: Translation Components
- **`LatentAlignmentHead`**: An optional auxiliary Continuous Temporal Classification (CTC) head. It uses a single Linear layer to project the encoder latent $Z$ to a vocabulary space (default 256 + 1 blank token) to encourage monotonic, segmented representations, even without explicit gloss-level supervision.
- **`AttentionPooling`**: An attention-based temporal aggregator. It uses a set of learned query vectors (default 32) to cross-attend over the variable-length encoder sequence $Z$, producing a fixed-length output sequence.
- **`LatentToT5Adapter`**: A multi-layer projection network (Linear $\rightarrow$ LayerNorm $\rightarrow$ GELU $\rightarrow$ Dropout $\rightarrow$ Linear) that maps the encoder's 512-dimensional latent space into the specific hidden dimension expected by `FLAN-T5`.
- **`SignToTextModel`**: The full Phase 2 architecture. It chains the `MultiStreamSemanticEncoder` $\rightarrow$ `AttentionPooling` (optional) $\rightarrow$ `LatentToT5Adapter` $\rightarrow$ `FLAN-T5`. It handles frozen/unfrozen encoder states and optional CTC alignment losses during training, and implements custom generation methods.

---

## 2. Experimental Conformer Architecture (`modal_motion_conformer_app.py`)

This file contains an alternative set of models built around the Conformer architecture, designed to be trained in a unified multi-task setup on Modal.

### 2.1 Encoders & Projections
- **`ModalityProjection`**: Projects raw 540-dim concatenated position and velocity features into a 256-dim space. Similar to `MultiStreamSemanticEncoder`, it uses separate MLPs for pose (198 $\rightarrow$ 64), face (90 $\rightarrow$ 32), and hands (126 $\rightarrow$ 96 each) before concatenating and fusing them via a LayerNorm-activated Linear layer.
- **`ConformerBlock`**: A standard Conformer block consisting of: a Feed-Forward network (dim 1024) $\rightarrow$ Multi-Head Self-Attention (4 heads) $\rightarrow$ Depthwise 1D Convolution (kernel size 7) $\rightarrow$ Pointwise Convolution $\rightarrow$ a second Feed-Forward network.
- **`MotionConformerEncoder`**: The core Conformer-based representation learner. It chains `ModalityProjection` $\rightarrow$ four `ConformerBlock`s $\rightarrow$ two strided 1D convolutions for 4x temporal downsampling, yielding a 512-dimensional latent representation $Z$.

### 2.2 Reconstructors & Adapters
- **`ReconstructionDecoder`**: A simpler decoder than the Phase 1 DDPM. It uses two transposed 1D convolutions (`ConvTranspose1d` with kernel size 4, stride 2) to upsample the latent representation back to the original temporal resolution, followed by a final `Conv1d` to reconstruct the original 540-dim features directly (optimizing for MSE), bypassing the iterative diffusion process.
- **`T5Adapter`**: A projection network (Linear $\rightarrow$ LayerNorm $\rightarrow$ GELU $\rightarrow$ Linear) to match the T5 hidden dimension, importantly featuring an added learned positional embedding (max length 15).

### 2.3 Full Experimental Architectures
- **`MotionConformerT5`**: The full translation model evaluated in the Modal experiment. It processes inputs through the `MotionConformerEncoder` to produce latent $Z$. It simultaneously passes $Z$ to the `ReconstructionDecoder` (for a multi-task reconstruction loss), a CTC head (for alignment loss), and the `T5Adapter` $\rightarrow$ `FLAN-T5` (for translation loss).
- **`MotionConformerPhase1Words`**: An architecture designed to test if the model can perform utterance-level word classification instead of full text generation. It uses the `MotionConformerEncoder` to encode sliding windows of an utterance, and the `ReconstructionDecoder` to compute window-level reconstruction loss. It then applies temporal mean-pooling across all window latents within an utterance and passes the aggregated embedding to a multi-label classification MLP (`LayerNorm` $\rightarrow$ `Linear` $\rightarrow$ `GELU` $\rightarrow$ `Dropout` $\rightarrow$ `Linear(num_words)`) to predict the presence of specific words.
