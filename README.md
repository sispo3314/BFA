# Boundary-Flux-Aware Gated Multi-Stream Network for Robust Human Activity Recognition
<img width="1040" height="765" alt="image" src="https://github.com/user-attachments/assets/d35eb994-9aa6-40e7-96ee-b1b777d56ae6" />
This repository implements the methodology proposed in the paper "Boundary-Flux-Aware Gated Multi-Stream Network for Robust Human Activity Recognition"

## Paper Overview
Abstract: In sensor-based Human Activity Recognition (HAR),
deep learning models often struggle to balance sensitivity during
activity transitions with stability during steady-state segments.
Most HAR systems adopt fixed-length windowing and a single
temporal encoder. However, processing diverse motion regimes
with a monolithic encoder can be inefficient when handling
both transient and steady-state dynamics simultaneously. This
one-size-fits-all representation is ill-suited to competing tempo-
ral regimes, often smoothing abrupt switch points or yielding
jittery predictions. To address this challenge, we propose a role-
separated architecture that effectively disentangles boundary-
sensitive dynamics from stationary contexts. Our framework
incorporates a Boundary-Flux stream to capture rapid signal
variations and a Steady-State Representation (SSR) stream to
maintain robustness during continuous activities. Crucially, we
introduce a lightweight, Label-Free Motion Gating mechanism
that dynamically modulates the contribution of these streams. Un-
like traditional attention mechanisms requiring explicit bound-
ary annotations, our gate is trained using weak supervision
derived from signal-intrinsic motion cues via a Gaussian Mixture
Model (GMM) approach. Extensive experiments on four public
benchmarks, including UCI-HAR, WISDM, MotionSense, and
Mhealth, demonstrate that the proposed method achieves state-
of-the-art performance with F1-scores ranging from 0.9684 to
0.9993 while maintaining exceptional stability against sensor
noise. Furthermore, the model maintains high computational
efficiency with approximately 0.66M parameters and an inference
latency of less than 5 ms (on standard CPUs), making it highly
suitable for real

##Dataset
- UCI-HAR dataset is available at https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones
- Motionsense dataset is available at https://www.kaggle.com/datasets/malekzadeh/motionsense-dataset
- MHEALTH dataset is available at https://archive.ics.uci.edu/dataset/319/mhealth+dataset
- WISDM dataset is available at https://www.cis.fordham.edu/wisdm/dataset.php

