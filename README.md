# Continuous Sign Language Translation (CSLT) with FLAN-T5-small and a Diffusion Autoencoder

Continuous sign language translation (sentence-level translation) remains a relatively niche and unexplored research area. Semantically rich spatio-temporal videos must be modelled to generate text from meaningful features in longer form clips.

## Data Sources

### [bdanko/how2sign-rgb-front-clips](https://huggingface.co/datasets/bdanko/how2sign-rgb-front-clips)

A WebDataset-formatted repository of the frontal RGB clips, sharded for high-speed streaming.

```python
from datasets import load_dataset

# Access the video bytes directly via streaming
ds = load_dataset("bdanko/how2sign-rgb-front-clips", split="train", streaming=True)
sample = next(iter(ds))

print(f"ID: {sample['__key__']}")
print(f"Video bytes: {len(sample['mp4'])} bytes")
```

### [bdanko/how2sign-landmarks-front-raw-parquet](https://huggingface.co/datasets/bdanko/how2sign-landmarks-front-raw-parquet)

Optimized Parquet shards containing raw MediaPipe Holistic landmarks (543 keypoints per frame). This allows for on-the-fly feature engineering without the overhead of massive `.npy` file extractions.

```python
from datasets import load_dataset
import numpy as np

dataset = load_dataset("bdanko/how2sign-landmarks-front-raw-parquet", split="train", streaming=True)
sample = next(iter(dataset))

# Landmarks are stored as binary float32 to save space
landmarks = np.frombuffer(sample['features'], dtype=np.float32).reshape(sample['shape'])

print(f"Video ID: {sample['video_id']}")
print(f"Sentence: {sample['sentence']}")
print(f"Shape: {landmarks.shape}") # [Frames, 543, 3]
```


Notes
* **Synchronization**: The `video_id` in landmarks matches the `__key__` in the RGB clips.
* **Keypoints**: 543 points (33 Pose, 468 Face, 21 Left Hand, 21 Right Hand).
* **Spatial Data**: Coordinates are normalized (0.0 to 1.0) relative to the original frame dimensions.
Citation

## Approach: Diffusion-Regularized Continuous Translation

This project implements a novel Continuous Sign Language Translation (CSLT) architecture optimized for edge deployment on an NVIDIA Jetson AGX Orin. The AGX Orin is unique, in that it features native Tensor Core support for 2:4 Structured Sparsity, which we can leverage.

To bypass the latency of real-time diffusion while retaining its massive representational power, we utilize a **Two-Phase Diffusion Autoencoder Paradigm**. We use a 1D Latent Diffusion model strictly as a training regularizer to learn a smooth, continuous manifold of human motion ($Z$). During real-time inference, the heavy diffusion decoder is discarded, allowing a frozen semantic encoder and a lightweight Transformer (FLAN-T5-small) to achieve high-FPS sequence-to-sequence translation.

### System Architecture & Tensor Flow

Below is the modular breakdown of the pipeline, including explicit tensor dimensions to guide implementation.
*Assumed hyperparameters for demonstration: Temporal Window $T=60$ frames, Feature Dim $F=540$ (90 keypoints $\times$ 6 kinematic features), Latent Dim $D=512$, Compressed Time $T'=15$.*

#### Module 1: Preprocessing & Feature Engineering
Raw `.npy` Mediapipe holistic landmarks are spatially normalized to the body center. 

**Critical Downsampling Step:** ASL grammar relies heavily on Non-Manual Markers (e.g., raised eyebrows for yes/no questions). However, the full Mediapipe face mesh provides 468 points, dominating the feature array (~75%) over the hands. To prevent the network from over-indexing on the face while ignoring the hands, we downsample the face mesh to just 15 keypoints (e.g., corners of the mouth, eyebrow arches, and tip of the nose). Combining this with 33 pose points and 42 hand points yields 90 total keypoints.

We calculate the temporal derivatives $(\Delta x, \Delta y, \Delta z)$ to explicitly provide motion vectors.
* **Input:** Raw coordinate sequences.
* **Process:** Face mesh downsampling, sliding window chunking, normalization, delta concatenation.
* **Output Shape:** `[Batch, T, F]` $\rightarrow$ e.g., `[B, 60, 540]`.

#### Module 2: Phase 1 - Latent Diffusion Autoencoder (Self-Supervised)
This network is trained to compress spatio-temporal keypoints into a rich, continuous latent representation ($Z$), effectively acting as a "continuous pseudo-gloss."
* **Sub-module 2A: Semantic Encoder (1D-CNN)**
    * **Status:** *Trainable* (during Phase 1).
    * **Input Shape:** `[B, 60, 540]`.
    * **Process:** 1D convolutions across the time dimension to temporally compress and project the features.
    * **Output Shape ($Z$):** `[B, 15, 512]`.
* **Sub-module 2B: Diffusion Decoder (1D-UNet)**
    * **Status:** *Trainable* (during Phase 1), *Discarded* (after Phase 1).
    * **Input Shape:** Pure Noise `[B, 60, 540]` + Conditioned on $Z$ `[B, 15, 512]` + Timestep $t$.
    * **Process:** Iterative DDPM/DDIM denoising to reconstruct the original coordinates.
    * **Target Output Shape:** `[B, 60, 540]`.
    * **Loss:** Mean Squared Error (MSE) between original and reconstructed coordinates.

#### Module 3: Phase 2 - Latent-to-Text Translation
Once Phase 1 converges, the Diffusion Decoder is deleted. The Semantic Encoder is frozen. We train a lightweight language model to map the continuous manifold $Z$ to English text.
* **Sub-module 3A: Frozen Semantic Encoder**
    * **Status:** *Frozen*.
    * **Input Shape:** `[B, 60, 540]`.
    * **Output Shape ($Z$):** `[B, 15, 512]`.
* **Sub-module 3B: FLAN-T5-small Translator (`google/flan-t5-small`)**
    * **Status:** *Trainable* (Fine-tuning only the cross-attention and decoder blocks).
    * **Input Shape:** Encoder hidden states ($Z$) `[B, 15, 512]`.
    * **Process:** Cross-attention maps the continuous motion latent to discrete text tokens.
    * **Target Output Shape:** `[B, Sequence_Length]` (English Token IDs).
    * **Loss:** Standard Cross-Entropy Language Modeling Loss.

#### Module 4: Edge Deployment Pipeline (AGX Orin)
For real-time inference, the pipeline is strictly a forward-pass sequence. No iterative diffusion occurs at runtime.
1.  **Stream:** Webcam $\rightarrow$ Mediapipe $\rightarrow$ `[1, 60, 540]` tensor buffer.
2.  **Encode:** Buffer $\rightarrow$ Frozen Semantic Encoder $\rightarrow$ $Z$ `[1, 15, 512]`.
3.  **Translate:** $Z$ $\rightarrow$ FLAN-T5-small $\rightarrow$ English String.

## Demonstration

The associated demonstration repository can be found at https://github.com/bencejdanko/continuous-sign-language-demonstration. This loads pose, hands and face Mediapipe Holistic models, and runs them in parallel to be sent to our inference server.

## Model Availability

All available at `bdanko/continuous-sign-language-translation`

## Citations

@inproceedings{Duarte_CVPR2021,
    title={{How2Sign: A Large-scale Multimodal Dataset for Continuous American Sign Language}},
    author={Duarte, Amanda and Palaskar, Shruti and Ventura, Lucas and Ghadiyaram, Deepti and DeHaan, Kenneth and
                   Metze, Florian and Torres, Jordi and Giro-i-Nieto, Xavier},
    booktitle={Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2021}
}

Duarte, A., Palaskar, S., Ventura, L., Ghadiyaram, D., DeHaan, K., Metze, F., Torres, J., & Giro-i-Nieto, X. “How2Sign: A Large-scale Multimodal Dataset for Continuous American Sign Language.” Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2021.

ASL, Sign Language, Mediapipe, Holistic, Pose Landmarks, Hand Landmarks, Face Landmarks, Keypoints, Motion Capture, Time Series, Gesture Recognition, Computer Vision, Deep Learning, Sequence Modeling
