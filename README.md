# Unified Policy Learning for Multi-Task Cross-Embodiment Robot Control

This repository contains the code for my master’s thesis:

> **Unified Policy Learning for Multi-Task Cross-Embodiment Robot Control**  
> **Author:** Rongxin Wang  
> **Contact:** rongxinwang99@gmail.com  

If you have any questions, or if you would like to reproduce or extend this work, feel very welcome to reach out or open an issue.

---

## 1. Overview

Cross-embodiment transfer is still a major bottleneck for robot reinforcement learning:  
policies are expensive to train on a single robot and task, and they often fail to generalize to new embodiments.

This project proposes a **unified latent-space control framework**:

- Each robot has **lightweight, embodiment-specific encoders and decoders** trained via behavior cloning.  
  These map observations and actions into a **shared continuous latent space**.
- A **single latent policy** operates on this space and is **reused across robots and tasks**, instead of training a separate policy per embodiment.
- **Cycle-consistency** and a **latent-dynamics term** keep:
  - forward / inverse mappings coherent, and  
  - temporal evolution smooth and stable.

For cross-embodiment transfer, we cast alignment as a **generative latent distribution alignment** problem:

- We focus on a **diffusion prior** over the fused **Obs–Act latent**,  
- conditioned on low-dimensional physical anchors:
  - end-effector position \( p_{\text{eef}} \)  
  - Jacobian-induced end-effector linear velocity \( v_{\text{eef}} \)
- This keeps the conditioning informative **without over-conditioning** in the low-data regime.

The framework is implemented and evaluated within **robosuite** on:

- **Robots:** Panda, Sawyer, xArm6  
- **Tasks:** Reach, Lift, Stack  

under a **unified evaluation protocol**.  
We systematically compare **diffusion**, **adversarial (GAN)**, and **flow-matching** priors for alignment.

---

## 2. Main Contributions (What this code does)

Concretely, this repository provides:

1. **Unified latent policy for multi-task, cross-embodiment control**
   - BC-trained encoders/decoders per robot to embed observations and actions.
   - A shared latent actor reused across robots and tasks.

2. **Geometry- and dynamics-aware latent regularization**
   - Cycle-consistency losses to couple encoder/decoder in both directions.
   - Latent-dynamics constraints to enforce temporal coherence.

3. **Generative alignment with diffusion priors**
   - Diffusion prior on the **joint Obs–Act latent**.
   - Conditioning on \( p_{\text{eef}} \) and Jacobian-based \( v_{\text{eef}} \).
   - Systematic comparison against GAN- and flow-matching-based alignment.

4. **Multi-task joint training of the prior**
   - Joint alignment on **Reach + Lift** to obtain a **shared multi-task latent**.
   - Task-specific actors that operate in the same latent and:
     - keep **millimeter-level accuracy** on Reach,  
     - stabilize **contact and grasp-and-lift** behavior on Lift.

5. **Practical guidance for future work**
   - Empirical insights on:
     - how to couple the prior with the policy and encoders,
     - how to choose and scale physical conditions,
     - typical failure modes of GAN/flow matching in this setting.
   - Discussion of limitations:
     - BC-shaped latent from finite simulated data,
     - gripper–arm bottlenecks on long-horizon contact tasks,
     - reliance on privileged simulation state.

---

## 3. Environment & Hardware

The experiments in the thesis were run on the following hardware:

- **CPU:** Intel i9-14900K  
- **GPU:** NVIDIA RTX 4090 (24 GB)  
- **RAM:** 64 GB  

The code is intended to run on a recent Linux system (e.g., Ubuntu 20.04/22.04) with:

- Python ≥ 3.9  
- CUDA-compatible GPU and recent NVIDIA drivers  
- MuJoCo / robosuite stack for manipulation tasks

A minimal setup (to be refined):

```bash
# Create a fresh environment
conda create -n cross-embodiment python=3.10
conda activate cross-embodiment

# Install project dependencies
pip install -r requirements.txt