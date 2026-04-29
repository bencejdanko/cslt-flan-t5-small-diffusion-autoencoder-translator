# Repository Analysis Summary

## Overview

This repository is a compact research and experimentation scaffold for **Continuous Sign Language Translation (CSLT)**. Its goal is to translate **continuous sign language sequences** into **English text** using MediaPipe-style landmark inputs and a two-stage training setup built around a diffusion autoencoder and `FLAN-T5-small`.

In practical terms, the project is trying to solve:

- **Input:** sequences of human pose, hand, and face landmarks over time
- **Output:** sentence-level English text
- **Use case:** efficient inference for edge deployment, especially on devices such as the NVIDIA Jetson AGX Orin

The repository is small and mainly consists of:

- `README.md`
- `colab_phase1_diffusion.ipynb`
- `colab_phase2_translation.ipynb`

So this is not a large packaged application. It is primarily a **project description plus two runnable Colab notebooks** that demonstrate the training flow.

---

## Main Purpose of the Code

The code implements a **two-phase architecture** for sign language translation:

1. **Phase 1:** Learn a compact latent representation of motion using a **semantic encoder** trained together with a **diffusion decoder**
2. **Phase 2:** Freeze that encoder and train **`google/flan-t5-small`** to translate the encoder’s latent representation into English text

The key idea is that diffusion is used during training to help shape a good latent space, but **diffusion is not used at runtime**. At inference time, the intended pipeline is much lighter:

**MediaPipe landmarks → engineered motion features → trained semantic encoder → FLAN-T5-small → English text**

---

## Data and Input Representation

The repo references two Hugging Face datasets:

- `bdanko/how2sign-rgb-front-clips`
- `bdanko/how2sign-landmarks-front-raw-parquet`

The training code uses the **landmark dataset** rather than RGB video frames directly.

### Landmark format

The input starts as **MediaPipe Holistic landmarks** with:

- 33 pose landmarks
- 468 face landmarks
- 21 left-hand landmarks
- 21 right-hand landmarks

That gives **543 keypoints per frame**.

### Feature engineering

The notebooks reduce this raw landmark data to a more compact motion representation:

- Keep all **pose** landmarks
- Keep all **hand** landmarks
- Downsample the face mesh to **15 selected face landmarks**
- Normalize all landmarks relative to body center
- Compute temporal deltas \(`\Delta x, \Delta y, \Delta z`\)
- Concatenate positions and deltas into a feature tensor

This produces:

- **90 keypoints total**
- **540 features per frame**

The data is then sliced into temporal windows:

- `T_WINDOW = 60`
- `T_STRIDE = 30`

So each training chunk is effectively a **60-frame motion segment** represented as a `[T, 540]` tensor.

---

## Model Architecture

## 1. Semantic Encoder

The semantic encoder is a small **1D convolutional network**. It takes motion features shaped like:

- Input: `[B, T=60, F=540]`

and compresses them into a latent representation:

- Output: `[B, D=512, T'=15]`

This latent `Z` is the central learned representation in the system.

### Role of `Z`

The repo frames `Z` as a compact representation of continuous signing—effectively a learned intermediate motion embedding that the language model can condition on.

---

## 2. Diffusion Decoder

The diffusion component is a **1D UNet-like decoder** conditioned on the latent `Z`.

Its role is:

- take noisy motion features
- condition on the encoder latent
- reconstruct the original motion sequence

Important detail: the diffusion model is used as a **training regularizer**, not as part of the intended real-time inference stack.

So the architecture is **not**:

**MediaPipe → diffusion model → text**

It is better described as:

**MediaPipe → semantic encoder trained with diffusion supervision → text model**

---

## 3. Translation Model

The translation stage uses:

- `google/flan-t5-small`

In Phase 2, the notebook loads the trained semantic encoder from Phase 1, freezes it, computes latent `Z`, and feeds `Z` directly into T5 through `inputs_embeds`.

That means the language model is being used as a translator from a continuous latent motion space into tokenized English text.

So yes, the repo does describe a model very close to:

**MediaPipe coords → trained encoder → LLM**

with the nuance that the encoder was trained inside a diffusion-autoencoder setup.

---

## Phase 1 Details: Diffusion Autoencoder Training

The first notebook, `colab_phase1_diffusion.ipynb`, trains:

- `SemanticEncoder`
- `DiffusionDecoder`

### Declared notebook settings

- **Dataset:** streamed How2Sign landmark data
- **Max samples:** `100`
- **Batch size:** `32`
- **Learning rate:** `1e-4`
- **Epochs:** `3`
- **Runtime target:** GPU \(T4 or better\)
- **Estimated runtime:** about **2 minutes** for 100 clips and 3 epochs

### Objective

The notebook computes:

- latent `Z = encoder(batch)`
- noisy input by mixing batch features with random noise
- decoder prediction from noisy input, latent `Z`, and timestep `t`
- MSE loss against the original batch

So the reported optimization target is:

- **Phase 1 loss:** Mean Squared Error \(MSE\)

The notebook prints only:

- `avg_loss` per epoch

There are no saved validation metrics or benchmark tables inside the repo.

---

## Phase 2 Details: FLAN-T5 Translation Training

The second notebook, `colab_phase2_translation.ipynb`, performs the text-training step.

### Declared notebook settings

- **Encoder source:** `bdanko/continuous-sign-language-translation`
- **Translation model:** `google/flan-t5-small`
- **Max samples:** `100`
- **Batch size:** `8`
- **Learning rate:** `5e-5`
- **Epochs:** `2`
- **Runtime target:** GPU \(T4 or better\)
- **Estimated runtime:** about **4 minutes** for 100 clips and 2 epochs

### Objective

The notebook:

1. Downloads the trained semantic encoder checkpoint
2. Freezes encoder parameters
3. Computes latent `Z` for each feature chunk
4. Tokenizes the paired sentence text
5. Trains FLAN-T5 using standard sequence-to-sequence loss

So the reported optimization target is:

- **Phase 2 loss:** language-model cross-entropy via `outputs.loss`

Again, the notebook only prints:

- `avg_loss` per epoch

There are no evaluation metrics beyond training loss.

---

## What the Repo Says About Inference

The README’s intended deployment story is:

1. Capture webcam or live input
2. Run MediaPipe
3. Build a `[1, 60, 540]` feature buffer
4. Run the frozen semantic encoder
5. Feed latent `Z` into FLAN-T5-small
6. Generate English text

The README explicitly says the diffusion decoder is **discarded after Phase 1** and that there is **no iterative diffusion at runtime**.

That is the main performance-oriented architectural claim: use diffusion only to shape the latent during training, then remove it for faster inference.

---

## Current Performance: What Is Actually Available

Based on the contents of this repository, the available performance information is limited.

### What is explicitly present

- Rough Colab runtime estimates:
  - **Phase 1:** ~2 minutes for 100 clips / 3 epochs
  - **Phase 2:** ~4 minutes for 100 clips / 2 epochs
- Hardware guidance:
  - **GPU:** T4 or better
- Training settings:
  - small sample count
  - short runs
  - lightweight demo-scale hyperparameters
- Training losses:
  - **Phase 1:** MSE
  - **Phase 2:** T5 cross-entropy loss

### What is missing

The repo does **not** provide:

- BLEU
- ROUGE
- WER
- exact match
- validation loss curves
- held-out test set results
- benchmark comparisons
- measured FPS
- latency numbers
- Jetson AGX Orin timing results
- memory use measurements
- ablation studies

So there is currently **no evidence in the repo of actual translation quality or real-world inference speed**.

---

## Best Interpretation of “Current Performance”

If someone asks, “How well does this model currently perform?”, the honest answer from this repo alone is:

- It has a **clear prototype architecture**
- It has **small runnable training notebooks**
- It has **estimated training times**
- It has **loss functions and epoch logging**
- It does **not** yet include rigorous evaluation results

In other words, this repository appears to be a **proof-of-concept research scaffold**, not a polished benchmarked system.

---

## Final Takeaway

This repository is best understood as an experimental CSLT pipeline built around the idea:

**MediaPipe landmark sequences → compact learned motion latent → FLAN-T5-small translation**

with diffusion used during training to make that latent representation more useful.

The code strongly suggests the intended architecture is:

- lightweight at inference
- suitable for edge deployment
- based on landmark features rather than raw video
- designed for sentence-level continuous sign language translation

But in its current state, the repo provides **architecture and training scaffolding**, not strong empirical proof of model quality. The available “performance” information is limited to:

- rough notebook runtime estimates
- training hyperparameters
- per-epoch loss logging

There are no in-repo results showing how accurate, robust, or fast the final system actually is.
