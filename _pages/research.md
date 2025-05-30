---
layout: page
permalink: /research/
title: research
description:
nav: true
nav_order: 1
---

My research spans the **theory and methodology** of the following core areas of modern machine learning (click to see):
- [**I. Contextual Understanding and Reasoning with LLMs**](#i-contextual-understanding-and-reasoning-with-llms)
- [**II. Unsupervised Representation Learning**](#ii-unsupervised-representation-learning)
- [**III. Robust Representation Learning**](#iii-robust-representation-learning)
- [**IV. Structural Representation Learning**](#iv-structural-representation-learning)

**Research Methodology.** Although people often find that deep learning relies on tons of empirical intuitions, I'd rather believe that "nothing is more intuitive than a good theory." The process of developing a theory of your intuition helps question, sharpen, validate your intuitive understanding. I follow this principle in my work and hope you find them helpful as well.

---
##### **I. Contextual Understanding and Reasoning with LLMs**

How LLMs understand, adapt to, and reason with contexts.

**I.1 In-context / Long-context Understanding: LongPPL & LongCE, Invariant ICL, OOD of ICL**
- [*What is Wrong with Perplexity for Long-context Language Modeling?*](https://openreview.net/pdf?id=fL4qWkSmtM) (ICLR 2025)  
- [*Rethinking Invariance in In-context Learning*](https://openreview.net/pdf?id=q1UyoY3MgJ) (ICLR 2025)  
- [*Can In-context Learning Really Generalize to OOD Tasks?*](https://openreview.net/pdf?id=INe4otjryz) (ICLR 2025)  

**I.2 Chain-of-thought and Reasoning: Self-correction, Optimal Chain-of-thought Length, Reasoning on Graph**
- [*A Theoretical Understanding of Self-Correction through In-context Alignment*](https://arxiv.org/pdf/2405.18634) (NeurIPS 2024)  
- [*When More is Less: Understanding Chain-of-Thought Length in LLMs*](https://arxiv.org/abs/2502.07266) (Best Paper Runner-up at ICLR 2025 Workshop)  
- [*G1: Teaching LLMs to Reason on Graphs with Reinforcement Learning*](https://www.arxiv.org/pdf/2505.18499) (arxiv 2025)


**I.3 Transformers: Position Bias, Multi-layer Attention, Dimensional Feature Collapse**
- [*On the Emergence of Position Bias in Transformers*](https://www.arxiv.org/pdf/2502.01951) (ICML 2025)  
- [*On the Role of Attention Masks and LayerNorm in Transformers*](https://arxiv.org/pdf/2405.18781) (NeurIPS 2024)  
- [*ContraNorm: A Contrastive Learning Perspective on Oversmoothing and Beyond*](https://openreview.net/pdf?id=SM7XkJouWHm) (ICLR 2023)  


---
##### **II. Unsupervised Representation Learning**
How to pretrain powerful foundation models from massive unlabeled data.

**II.1 Self-supervised Learning (SSL): Contrastive, Non-contrastive, Equivariant, Contextual**
- [*Chaos is a Ladder: A New Theoretical Understanding of Contrastive Learning via Augmentation Overlap*](http://arxiv.org/pdf/2203.13457) (ICLR 2022)  
- [*Towards a Unified Theoretical Understanding of Non-contrastive Learning via Rank Differential Mechanism*](https://openreview.net/pdf?id=cIbjyd2Vcy) (ICLR 2023)  
- [*Understanding the Role of Equivariance in Self-supervised Learning*](https://arxiv.org/pdf/2411.06508) (NeurIPS 2024)  
- [*In-Context Symmetries: Self-Supervised Learning through Contextual World Models*](https://arxiv.org/pdf/2405.18193) (NeurIPS 2024)

**II.2 Generative Models: Masked Autoencoders, Autoregressive, Energy-based Models, Generative Adversarial Networks**
- [*How Mask Matters: Towards Theoretical Understandings of Masked Autoencoders*](https://arxiv.org/pdf/2210.08344) (NeurIPS 2022)  
- [*Look Ahead or Look Around? A Theoretical Comparison Between Autoregressive and Masked Pretraining*](https://openreview.net/pdf?id=2rPoTgEmjV) (ICML 2024)  
- [*A Unified Contrastive Energy-based Model for Understanding the Generative Ability of Adversarial Training*](http://arxiv.org/pdf/2203.13455) (ICLR 2022)  
- [*Reparameterized Sampling for Generative Adversarial Networks*](https://arxiv.org/pdf/2107.00352) (ECML-PKDD 2021 Best ML Paper)  

**II.3 Key SSL Components: Predictor, Discrete Tokenization, Projector, Generated Data**  
- [*Towards a Unified Theoretical Understanding of Non-contrastive Learning via Rank Differential Mechanism*](https://openreview.net/pdf?id=cIbjyd2Vcy) (ICLR 2023)  
- [*Do Generated Data Always Help Contrastive Learning?*](https://arxiv.org/pdf/2403.12448.pdf) (ICLR 2024)  
- [*On the Role of Discrete Tokenization in Visual Representation Learning*](https://openreview.net/pdf?id=WNLAkjUm19) (ICLR 2024)  
- [*Projection Head is Secretly an Information Bottleneck*](https://openreview.net/pdf?id=L0evcuybH5) (ICLR 2025)  

**II.4 Feature Sparsity, Identifiability, Interpretability: Non-negative CL, triCL, Robustness Gains, CSR Embedding**
- [*Non-negative Contrastive Learning*](https://arxiv.org/pdf/2403.12459) (ICLR 2024)  
- [*Identifiable Contrastive Learning with Automatic Feature Importance Discovery*](https://arxiv.org/pdf/2310.18904.pdf) (NeurIPS 2023)  
- [*Beyond Interpretability: The Gains of Feature Monosemanticity on Model Robustness*](https://openreview.net/forum?id=g6Qc3p7JH5) (ICLR 2025)  
- [*Beyond Matryoshka: Revisiting Sparse Coding for Adaptive Representation*](https://www.arxiv.org/pdf/2503.01776) (ICML 2025)  
- [*The Multi-faceted Monosemanticity in Multimodal Representations*](https://openreview.net/pdf?id=9NLRpwfLnT) (NeurIPS Workshop 2024)


---

##### **III. Robust Representation Learning**

How to build models robust to adversarial attacks and reliable across distribution shifts.

**III.1 Adversarial Attack and Defense: Adversarial Examples, Unsupervised Adversarial Training, Robust Overfitting**
- [*Adversarial Examples Are Not Real Features*](https://arxiv.org/pdf/2310.18936.pdf) (NeurIPS 2023)  
- [*Rethinking the Effect of Data Augmentation in Adversarial Contrastive Learning*](https://openreview.net/pdf?id=0qmwFNJyxCL) (ICLR 2023)  
- [*Balance, Imbalance, and Rebalance: Understanding Robust Overfitting from a Minimax Game Perspective*](https://arxiv.org/pdf/2310.19360.pdf) (NeurIPS 2023)

**III.2 LLM Jailbreak: In-context Attack and Defense, Safety of Chain-of-thought Reasoning**
- [*Jailbreak and guard aligned language models with only few in-context demonstrations*](https://arxiv.org/pdf/2310.06387) (arxiv 2023, 250+ citations)  
- [*Are Smarter LLMs Safer? Exploring Safety-Reasoning Trade-offs in Prompting and Fine-Tuning*](https://arxiv.org/pdf/2502.09673) (arxiv 2025)

**III.3 Out-of-distribution (OOD) Generalization: OODRobustBench, Adversarial Training for OOD**
- [*OODRobustBench: Benchmarking Adversarial Robustness under Distribution Shift*](https://arxiv.org/pdf/2310.12793) (ICML 2024)  
- [*Improving Out-of-distribution Robustness by Adversarial Training with Structured Priors*](https://arxiv.org/pdf/2210.06807) (NeurIPS 2022)  
- [*On the Connection between Invariant Learning and Adversarial Training for Out-of-Distribution Generalization*](https://arxiv.org/pdf/2212.09082.pdf) (AAAI 2023 Oral)

---

##### **IV. Structural Representation Learning**

How to learn structured data (e.g., graphs) efficiently with structured models, such as, Graph Neural Networks (GNNs).

**IV.1 Feature Dynamics of GNNs: oversmoothing, graph equilibrium, unbiased graph sampling, GraphSSL**
- [*Dissecting the Diffusion Process in Linear Graph Convolutional Networks*](https://arxiv.org/pdf/2102.10739) (NeurIPS 2021)  
- [*Optimization-Induced Graph Implicit Nonlinear Diffusion*](https://proceedings.mlr.press/v162/chen22z/chen22z.pdf) (ICML 2022)  
- [*Unbiased Stochastic Proximal Solver for Graph Neural Networks with Equilibrium States*](https://openreview.net/pdf?id=j3cUWIMsFBN) (ICLR 2023)  
- [*Architecture Matters: Uncovering Implicit Mechanisms in Graph Contrastive Learning*](https://arxiv.org/pdf/2311.02687.pdf) (NeurIPS 2023)  

**IV.2 Learning with Symmetry: Laplacian canonicalization, theory of canonicalization**
- [*Laplacian Canonization: A Minimalist Approach to Sign and Basis Invariant Spectral Embedding*](https://arxiv.org/pdf/2310.18716.pdf) (NeurIPS 2023)  
- [*A Canonization Perspective on Invariant and Equivariant Learning*](https://arxiv.org/pdf/2405.18378) (NeurIPS 2024)  


<!-- **I. unsupervised learning** <br>
- Self-supervised Learning: Contrastive (ICLR'22; ICLR'23), Non-contrastive (ICLR'23), Equivariant (NeurIPS'24), ContextSSL (ours, NeurIPS'24, CSAIL News)
- Generative Models: Masked Autoencoders (NeurIPS'22), Autoregressive (ICML'24), Energy-based Model (ICLR'22), GAN (ECML-PKDD'21 Best ML Paper)
- Key SSL Components: Predictor/asymmetry module (DINO, BYOL, SimSiam, SwAV) (ICLR'23), Discrete Tokenization (ICLR'24), Projector (ICLR'25), Generated Data (ICLR'25)
- Sparsity and Interpretability: Non-negative CL (ICLR'24), triCL (NeurIPS'23), Beyond Interpretability (ICLR'25), CSR (ICML'25), 

**II. Understanding and Reasoning with Large Language Models:**
- In-context/long-context learning: LongPPL & LongCE (ICLR'25), position bias (ICML'25), invariance of ICL (ICLR'25), OOD of ICL (ICLR'25)
- Reasonsing: Self-correction (NeurIPS'25), CoT Length (ICLR-W'25 Award), reasoning on graph (G1, arxiv)
- Transformers: position bias (ICML'25), dimensional collapse (ICLR'23)

**III. Structured Representation Learning:**
- Feature dynamics of Graph Neural Networks: DGC (NeurIPS'21), GIND (ICML'22), G2CN (ICML'22), GraphSSL (NeurIPS'23, )
- Learning with symmetry: Laplacian canonicalization (NeurIPS'23), theory of canonicalization (NeurIPS'24)

**IV. Robust and Trustworthy Machine Learning:**
- Out-of-distribution generalzaition: OODRobustBench (ICML'23), Adversarial Training (NeurIPS'22, AAAI'23)
- Adversarial Attack and Defense: Not Real Features (NeurIPS'23), Robust overfitting (NeurIPS'23)
- LLM safety: in-context attack and defense (arxiv, featured by Anthropic), reasoning safety (arxiv)  -->

