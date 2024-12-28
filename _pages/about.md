---
layout: about
title: about
permalink: /
subtitle: Postdoctoral researcher in machine learning at <a href='https://www.csail.mit.edu/'>MIT CSAIL</a>, advised by <a href="https://people.csail.mit.edu/stefje/">Stefanie Jegelka</a>.

profile:
  align: right
  image: avatar.JPG
  image_circular: false # crops the image to make it circular
social: true # includes social icons at the bottom of the page
news: true # includes a list of news items
selected_papers: true # includes a list of papers marked as "selected={true}"
---

My long-term goal is to develop foundation models that can self-learn from massive data and reliably self-adapt to novel tasks. My research has contributed to unveiling the key principles underlying foundation models and designing algorithms to enhance their capabilities and trustworthiness in challenging scenarios:
<!-- I work on theoretical principles of foundation models (generative models & representation models) and practical algorithms for improving their capabilities and safety: -->
- **Mathematical Principles of Foundation Models**. I established theoretical foundations for a broad spectrum of *Self-Supervised Learning (SSL)* methods that are at the heart of foundation models, from contrastive [[1](http://arxiv.org/pdf/2203.13457), [2](https://openreview.net/pdf?id=VBTJqqWjxMv)], non-contrastive [[3](https://openreview.net/pdf?id=cIbjyd2Vcy)], autoregressive [[4](https://openreview.net/pdf?id=2rPoTgEmjV)], reconstructive [[5](https://arxiv.org/pdf/2210.08344)], to predictive [[6](https://openreview.net/pdf?id=yLpuruMZHE)] approaches. My recent work further pioneered the first rigorous theory [[7](https://arxiv.org/pdf/2405.18634)] for the test-time self-correction ability of LLMs, a key mechanism for scaling reasoning during inference.
- **Improving Model Capabilities**. I leveraged these principles to "debug" foundation models. Notably, I generalized self-supervised learning to be able to self-adapt to new tasks without retraining [[8](https://arxiv.org/pdf/2405.18193)] ([featured by MIT](https://www.csail.mit.edu/news/machines-self-adapt-new-tasks-without-re-training)), proposed adaptive training with AI-generated data to circumvent data shortage [[9](https://arxiv.org/pdf/2403.12448.pdf)], and significantly enhanced LLMs' long-context understanding by identifying key relevant tokens [[10](https://arxiv.org/pdf/2410.23771)].
- **Trustworthy Foundation Models**. I developed efficient and principled algorithms for improving model trustworthiness, spanning adversarial robustness [[11](http://arxiv.org/pdf/2203.13455), [12](https://arxiv.org/pdf/2210.07540.pdf), [13](https://arxiv.org/pdf/2310.19360.pdf), [14](https://arxiv.org/pdf/2310.18936.pdf)], feature interpretability [[15](https://arxiv.org/pdf/2310.18904.pdf), [16](https://arxiv.org/pdf/2403.12459)], and domain generalization [[17](https://arxiv.org/pdf/2210.06807), [18](https://arxiv.org/pdf/2212.09082.pdf), [19](https://arxiv.org/pdf/2310.12793)]. In particular, I showed that LLMs' own emergent abilities, like in-context learning [[20](https://arxiv.org/pdf/2310.06387)] and self-correction [[21](https://arxiv.org/pdf/2405.18634)], can be used to jailbreak and defend LLMs, which was [featured and scaled up in Anthropic's blog](https://www.anthropic.com/research/many-shot-jailbreaking).
 
My first-author papers received the Best ML Paper Award at ECML-PKDD 2021, the Silver Best Paper Award at ICML 2021 AdvML workshop, and the Best Paper Award at ICML 2024 ICL workshop. My thesis won the CAAI Outstanding Ph.D. Dissertation Runner-Up Award. I published 33 papers at NeurIPS, ICLR, and ICML, and I am a (co)-first author on 22 of them. 

I served as an area chair for ICLR 2024 and 2025, and as a reviewer for main AI conferences (NeurIPS, ICML, ECML, AISTATS, LoG, CVPR, ACL). I organized the [NeurIPS 2024 Workshop on Red Teaming GenAI](https://redteaming-gen-ai.github.io/) and the [ML Tea Seminar at MIT](https://projects.csail.mit.edu/ml-tea/).

I obtained my PhD in Applied Mathematics from Peking University in 2023, advised by [Yisen Wang](https://yisenwang.github.io), [Zhouchen Lin](https://zhouchenlin.github.io/), [Jiansheng Yang](https://www.math.pku.edu.cn/jsdw/js_20180628175159671361/y_20180628175159671361/69984.htm). Prior to that, I did my undergraduate at School of Mathematical Sciences, Peking University.

**I am on the job market 2024-2025. Links: [CV](assets/pdf/CV-Yifei Wang-MIT-Dec2024.pdf) \|  [Research Statement](assets/pdf/Research Statement Yifei Wang.pdf)**
