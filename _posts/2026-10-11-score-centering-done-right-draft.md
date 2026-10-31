---
layout: post
title: Score Centering Done Right
description: A policy-gradient loss for the sampler, a local Jacobian repair, and what the correction buys in gradient estimation and asynchronous learning.
date: 2026-10-11 12:00:00 -0700
last_updated: 2026-10-12
author: Yifei Wang
tags: reinforcement-learning policy-gradient score-centering
categories: research
featured: true
related_posts: false
meta: 12-minute read
toc:
  beginning: true
---

A recent work, [score centering](https://arxiv.org/abs/2609.20807), introduces a new interesting approach to address the train-inference mismatch in RL training. In the paper, this technique is motivated from a seemingly arbitrary drift-removal perspective. In this blog, we will re-derive this technique from a even simpler  perspective of straight-through estimate (STE) (PART I). Based on this perspective, we will reveal the bias introduced by score centering and how to further alleviate this bias via a second-order correction term (PART II). We then show the derived method, called score centering done right (DRSC), obtained lower variance compared to TIS and lower bias than score centering from both synthetic and real-world experiments (PART III). The first part of this work is heavily inspired by the discussions with [Jiacheng You](https://x.com/YouJiacheng/status/2101580038546362759?s=20).

## 1. Score Centering A Change of Perspective
{: #perspective }

[Draw a figure to illustrate this change of perspective ]

Standard RL training optimizes for xx. Then introduce IS and TIS for off-policy. And discussion cons. Then introduce SC as an approixmation to Eq Alog q, with STE estimate. 

https://x.com/yifeiwang77/status/2102049177203323316?s=20
> Very intriguing discussion here to rethink policy gradient: we always want to optimize E_p[R], so we would need the IS techniques that introduced all variance.

Score centering is taking a different view: let’s just optimize E_q[R], as q (sampling distribution) is what we finally care about. As samples are indeed from q, no IS is needed.

And since we cannot optimize q (inference engine/old policy) directly, we just use p’s grad as a surrogate to q’s grad (yes its biased), as explained in the STE estimate here.

A change of perspective (from optimizing p obj to q obj) gives a lot of simplicity!

[your role in this part is to expand, formalize this disccussion]
===


things below might not matter.

The choice of objective determines what needs correcting. To optimize $$ \mathbb E_p[R] $$ using samples from $$ q $$, we need the ratio $$ p(a)/q(a) $$ in the policy gradient. To optimize $$ \mathbb E_q[R] $$, we already have the samples we need:

<div class="overflow-auto" markdown="1">

$$
g_q=\mathbb E_{a\sim q}\left[A(a)\nabla_{\theta_q}\log q(a)\right].
$$

</div>

Here $$ A=R-b $$ uses an action-independent baseline. The usual surrogate loss is $$ -\mathbb E_q[A\log q] $$, with the sampled actions and advantages held fixed during backpropagation. Nothing about this objective calls for importance sampling.

But a fast inference engine ordinarily supplies probabilities without a backward pass. Loading an old sampler checkpoint into the trainer would also add work. We therefore want to use $$ q $$’s probabilities with the derivatives already available in the trainer.

Write the logits as $$ z_q $$ and $$ z_p $$, and let $$ \operatorname{sg} $$ denote stop-gradient. The straight-through construction does exactly this:

<div class="overflow-auto" markdown="1">

$$
\widetilde z_q=z_p+\operatorname{sg}(z_q-z_p).
$$

</div>

Its value is $$ z_q $$. Its derivative is $$ J_p=\partial z_p/\partial\theta_p $$. Insert it into the PG loss:

**SCORE CENTERING · PG + STE**

<div class="overflow-auto" markdown="1">

$$
\mathcal L_{\mathrm{SC}}
=-\mathbb E_{a\sim q}\left[A(a)\log\operatorname{softmax}(\widetilde z_q)_a\right].
$$

</div>

Differentiating gives the unweighted score-centering update. Comparing it with the oracle makes the approximation visible:

<div class="overflow-auto" markdown="1">

$$
\begin{aligned}
g_{\mathrm{SC}}(a)&=A(a)J_p^\top(e_a-q),\\
g_{\mathrm{oracle}}(a)&=A(a)J_q^\top(e_a-q).
\end{aligned}
$$

</div>

Here $$ e_a $$ is the one-hot vector for the sampled action, and $$ J_q $$ is the logit Jacobian of the sampler’s differentiable reference. The action, advantage and probability vector are the same. **SC substitutes $$ J_p $$ for $$ J_q $$.** That is its STE bias. Avoiding importance weights removes one possible source of noise, but ordinary sampling variance remains. [Bengio et al., 2013](https://arxiv.org/abs/1308.3432)

<details markdown="1">
<summary>Why this is score centering, and which objective it targets</summary>

For $$ p=\operatorname{softmax}(z_p) $$, the trainer’s score is $$ J_p^\top(e_a-p) $$. Its mean under $$ q $$ is $$ J_p^\top(q-p) $$. Subtracting that mean leaves $$ J_p^\top(e_a-q) $$, the update derived from the PG + STE loss above. This is the unweighted form; the original score-centering method also composes with importance sampling. [Marek & Ryabinin, 2026](https://arxiv.org/abs/2609.20807)

The two raw reward-gradient targets are

<div class="overflow-auto" markdown="1">

$$
g_p=\mathbb E_{a\sim q}\left[\frac{p(a)}{q(a)}R(a)\nabla_{\theta_p}\log p(a)\right],
\qquad
g_q=\mathbb E_{a\sim q}\left[R(a)\nabla_{\theta_q}\log q(a)\right].
$$

</div>

The sampling distribution is detached in the surrogate loss. Differentiating through the expectation itself would define a different objective. Group-standardized advantages, used in the experiments below, also change the target from the raw expected-reward gradient.

</details>

When $$ q $$ is the deployed inference policy, improving its responses is the direct objective. When $$ q $$ is an older checkpoint, $$ g_q $$ is a delayed gradient applied at today’s weights. The local correction below makes that delayed gradient more accurate; it does not guarantee ascent for the current $$ p $$ objective.


=======

2. What is Wrong with Score Centering and How to Fix it.

2.1 The bias of score centering


quantifize the bias formally.

measure the bias in synth data.

2.2 Correcting the bias

since delta update etc (quote), a simple fix is to approximate zq with a linear interpolation without requring optimizing on q directly. formalize thigns and quantify the bias reduction theoretically / formally,

add syn data to show.

2.3 SCDR and implementation.

loss / alg format and pseudo code and JVP implementation.

quantify 1. train time diff / total rl time diff

3. DrSC in Action

3.1 How good is Dr SC at gradient estimate

quantify the bias and variance compared to 1. IS / TIS 2. SC under different bsz and different train inference mismatch (eg bf 16, int 4). only the approx to gq is needed when compared to sc. 

3.2 Dr SC in actual RL training

use the seed 42 rl training result. state the setting. maybe try to explain a bit with real evidence of why dr sc is better.

4. Concluding

SC and DrSC solve off-policyness by 1/ optimizing q objectives that directly use samples from q 2/ fix off-policy with linear terms instead of mulitplication terms which reduce high variance grads and training collapse.


## 2. A better backward pass from nearby weights
{: #bias }

The comparison above locates the missing information: how the logit Jacobian changes between $$ p $$ and $$ q $$. For nearby checkpoints, the parameter displacement tells us how to approximate that change.

RL post-training provides a reason to study this local regime. Empirical work finds restricted changes to pretrained weights, including update sparsity and limited spectral drift. These observations motivate a local approximation; the relevant assumption here is that the particular sampler and trainer checkpoints are close. [Mukherjee et al., 2025](https://arxiv.org/abs/2505.11711); [Zhu et al., 2025](https://arxiv.org/abs/2511.08567)

For the same smooth model implementation, set $$ \Delta=\operatorname{sg}(\theta_q-\theta_p) $$. Taylor expansion gives

<div class="overflow-auto" markdown="1">

$$
z_q=z_p+J_p\Delta+O(\|\Delta\|^2).
$$

</div>

This supplies a better backward proxy:

<div class="overflow-auto" markdown="1">

$$
\boxed{\widehat z_q=z_p+J_p\Delta.}
$$

</div>

The rollout already gives us $$ z_q $$’s forward value. **What we need from $$ \widehat z_q $$ is its dependence on the weights.** Holding $$ \Delta $$ fixed and differentiating the added term gives $$ D J_p[\Delta] $$, the leading change in the Jacobian.

Under locally Lipschitz logit Hessians,

<div class="overflow-auto" markdown="1">

$$
J_q=J_p+D J_p[\Delta]+O(\|\Delta\|^2).
$$

</div>

SC discards the first-order change. The corrected proxy includes it. For bounded advantages, this improves the checkpoint-induced gradient bias from generally first order to second order in $$ \|\Delta\| $$, without reweighting a single sample. This conclusion follows from the derivative expansion; forward agreement alone would not establish it.

<details markdown="1">
<summary>Locality, the bias expression, and what the RL literature establishes</summary>

Writing $$ c_q=\mathbb E_q[A(a)(e_a-q)] $$, the original bias is $$ (J_p-J_q)^\top c_q $$. The corrected bias replaces $$ J_p $$ by $$ J_p+D J_p[\Delta] $$, leaving an $$ O(\|\Delta\|^2) $$ residual under the stated regularity assumptions. The displacement is between parameter vectors, not probability vectors; the approximation is local and parameterization-dependent.

Mukherjee et al. report effective changes in 5–30% of parameters. Zhu et al. find that finite precision hides small updates outside preferred parameter regions and characterize limited spectral changes. Neither sparsity nor a small policy KL alone proves that the Jacobian expansion is accurate. We validate the relevant gradient and HVP approximation directly.

</details>

## 3. DR score centering keeps the same PG loss
{: #repair }

We can now use the same STE construction, replacing its backward proxy $$ z_p $$ with $$ \widehat z_q $$:

<div class="overflow-auto" markdown="1">

$$
\widetilde z_q^{\mathrm{DR}}
=\widehat z_q+\operatorname{sg}(z_q-\widehat z_q).
$$

</div>

The forward value remains $$ z_q $$, so the loss is still the ordinary PG surrogate evaluated at the sampler:

**DERIVATIVE-REPAIRED SCORE CENTERING · DR–SC**

<div class="overflow-auto" markdown="1">

$$
\mathcal L_{\mathrm{DR}}
=-\mathbb E_{a\sim q}\left[A(a)\log\operatorname{softmax}(\widetilde z_q^{\mathrm{DR}})_a\right].
$$

</div>

Both losses evaluate $$ -\mathbb E_q[A\log q] $$. SC differentiates through $$ z_p $$; DR–SC differentiates through $$ z_p+J_p\Delta $$. We call the latter derivative-repaired score centering.

The extra backward term is practical to compute. With $$ v=e_a-q $$ held fixed, the ascent direction is

<div class="overflow-auto" markdown="1">

$$
\boxed{g_{\mathrm{DR}}(a)
=A(a)\left[J_p^\top v+H_{v^\top z_p}\Delta\right].}
$$

</div>

One Hessian–vector product computes the correction. We never need to construct a full Jacobian or Hessian. For a batch from one behavior checkpoint, form $$ F=\operatorname{mean}_i A_i v_i^\top z_p(x_i) $$, then use $$ \nabla F+H_F\Delta $$ as the ascent direction. [Pearlmutter, 1994](https://doi.org/10.1162/neco.1994.6.1.147)

<details markdown="1">
<summary>From the loss to a matrix-free implementation</summary>

Keep $$ q $$, $$ A $$, $$ v $$ and $$ \Delta $$ detached; keep the derivative of $$ J_p $$ alive. For a batch sharing one behavior checkpoint, define $$ F=\operatorname{mean}_i A_i v_i^\top z_p(x_i) $$. Compute $$ \nabla F $$ and $$ H_F\Delta $$, then send their sum as the ascent direction to the optimizer. A minimal PyTorch-style sketch is:

```python
# q, actions and advantages come from the same rollout checkpoint.
delta = tuple((old - p).detach() for old, p in zip(old_params, params))
v = (one_hot(actions, z_p.shape[-1]) - q).detach()
F = (advantages.detach() * (v * z_p).sum(-1)).mean()
g = torch.autograd.grad(F, params, create_graph=True)
directional = sum((gi * di).sum() for gi, di in zip(g, delta))
h = torch.autograd.grad(directional, params)
for p, gi, hi in zip(params, g, h):
    p.grad = -(gi + hi).detach()  # optimizer minimizes
```

This sketch assumes participating differentiable parameters, double-backward support and no existing accumulated gradients. A production implementation must preserve accumulation/scaling and handle unused parameters. For mixed-age batches, compute contributions with each sample’s actual behavior checkpoint and sum them; one average displacement is generally incorrect. This HVP computes the gradient of the PG loss above via an equivalent scalar, not a different objective.

Detaching $$ \Delta $$ is essential: differentiating $$ \theta_q-\theta_p $$ would cancel the leading $$ J_p $$ term. Conversely, detaching $$ J_p $$ would remove the repair. The equivalent logit implementation computes a JVP $$ J_p\Delta $$, keeps its graph, constructs $$ \widetilde z_q^{\mathrm{DR}} $$, and backpropagates the displayed PG loss.

If full $$ z_q $$ is unavailable, using $$ \operatorname{softmax}(\widehat z_q) $$ directly also replaces $$ e_a-q $$ by $$ e_a-\widehat q $$. Under the same local smoothness assumptions that introduces another second-order forward approximation. It is not the exact-forward variant measured here. Top-$$ k $$ or other compressed rollout distributions introduce their own approximation.

</details>

**The correction needs a parameter displacement.** At identical weights, $$ \Delta=0 $$ and the HVP is zero. Numerical mismatch from engines, kernels or precision can still remain. This is why we validate a differentiable inference reference and study checkpoint lag separately from quantization. [Training–inference mismatch study](https://yichuan-w.github.io/blog/GDN-train-inference-mismatch-asyncRL/); [He et al., 2025](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)

## 4. Does the gradient move closer to the q oracle?
{: #evidence }

The derivation predicts a specific benefit: a smaller bias relative to $$ q $$’s gradient. It also suggests when that benefit should become easier to see. Larger batches average away sampling noise, but leave a biased estimator’s error floor.

### Forward agreement is not enough
{: #forward-agreement-is-not-enough }

We first isolate the premise in a simulation where logits and their Jacobians can vary independently. Even at a small mean KL of 0.0091, increasing Jacobian mismatch raises relative STE bias from 0.149 to 0.956. Similar probabilities need not give similar gradients.

{% include figure.liquid path="assets/blog/score-centering-done-right/synthetic-mechanism.svg" class="img-fluid rounded z-depth-1" loading="lazy" alt="Synthetic results: STE bias grows as Jacobian mismatch increases at small KL; the fraction of settings where a 25-percent-residual correction improves q-target MSE rises from 24 percent at batch 4 to 95 percent at batch 256." %}

<div class="caption" markdown="1">

**FIGURE 1**  Left: vary Jacobian mismatch at fixed probability-mismatch scale. Right: add a synthetic projected-score correction with 25% residual error, across all nonzero-Jacobian-mismatch settings. It beats STE’s q-target MSE in 24.0%, 56.2%, 86.7% and 95.2% of settings at batches 4, 16, 64 and 256. Exact enumerated moments; the synthetic correction is not an HVP or a trained model.

</div>

The right panel then asks how batch size affects the value of repairing that derivative. A synthetic correction that leaves 25% of the score error improves MSE in 24% of settings at batch 4, rising to 95% at batch 256. This is a controlled illustration, not a test of the HVP: a better score can add variance, so its bias reduction becomes more useful as sampling noise falls.

### Testing the HVP on Qwen
{: #testing-the-hvp-on-qwen }

We next test the actual correction against a differentiable $$ q $$ reference in Qwen3.5-0.8B, using all 752 million text parameters. Eight SST-2 prompts define a two-action policy; four responses form a group. That small action space lets us enumerate every group and measure the gradient error exactly.

We compare the corresponding **group-advantage update directions** for SC, DR–SC, TIS, GRPO and the oracle. These are the group-normalized counterparts of the PG directions above. Every method uses the same sampling distribution and advantages; only the estimator changes. We create checkpoint lag with four isolated SGD updates, then vary statistical batch size while holding engine batch at one.

{% include figure.liquid path="assets/blog/score-centering-done-right/qwen-targets.svg" class="img-fluid rounded z-depth-1" loading="lazy" alt="Qwen full-parameter gradient MSE at lag four: DR-SC approaches oracle q as batch size grows, while TIS is closer to the distinct current-policy target gp." %}

<div class="caption" markdown="1">

**FIGURE 2**  The same estimators, judged against two explicitly different targets. Curves redraw the existing lag-4 measurements. Each point averages per-prompt conditional MSE divided by that prompt’s target norm squared. TIS and unclipped IS overlap exactly here because the cap of 2 never activates. Oracle q is a sampled oracle-score estimator, so its MSE is not zero.

</div>

The left panel shows the predicted bias floor. SC stops improving as batch size grows, while DR–SC stays much closer to the oracle. At $$ B=256 $$, normalized MSE falls from **0.052127 to 0.005127**. The table shows where that improvement comes from:

<div class="table-responsive" markdown="1">

| Method | Bias² | Variance | MSE to $$ g_q $$ |
| :--- | ---: | ---: | ---: |
| SC / STE | 0.049218 | 0.002909 | 0.052127 |
| DR–SC | 0.001989 | 0.003138 | 0.005127 |
| TIS = unclipped IS | 0.054491 | 0.003203 | 0.057694 |
| Oracle q | 0.000000 | 0.003068 | 0.003068 |

</div>

Mean of per-prompt quantities divided by that prompt’s target norm squared. Variance is the trace of the batch estimator’s covariance. References use group-standardized advantages; they are not generally raw expected-reward gradients.

The large change is in bias²: 0.049218 becomes 0.001989. The sampled oracle also has nonzero MSE because it still uses a finite batch. Correction brings the estimator closer to that limit.

The right panel tests a different target, $$ g_p $$. Against it, correction raises SC’s MSE from 0.004999 to 0.047355. This is consistent with the derivation: the added term approximates $$ q $$’s derivative. A method can improve the $$ q $$ estimate while moving away from the $$ p $$ estimate.

### Why TIS has only slightly higher variance here
{: #why-tis-has-only-slightly-higher-variance-here }

The variance column contains a smaller effect. DR–SC is about 2% below TIS and 8% above SC. The importance ratios explain why: they range only from 0.746 to 1.425, so the TIS cap of 2 never activates. There are no extreme weights to create a large noise penalty; TIS and unclipped IS therefore overlap.

Reducing batch size raises all three variances in proportion to $$ 1/B $$. It does not change their relative ordering. Clipped GRPO actually has lower variance here (0.001632), accompanied by much larger $$ g_q $$ bias² (0.325266) and MSE (0.326898). These measurements support the bias correction, while showing why a universal variance ranking would be misleading.

<details markdown="1">
<summary>Why smaller batches do not change the variance ordering here</summary>

{% include figure.liquid path="assets/blog/score-centering-done-right/variance-scan.svg" class="img-fluid rounded z-depth-1" loading="lazy" alt="Variance at batches 4, 16, 64 and 256 scales as one over batch size. At batch 4 normalized variance is 0.186163 for SC, 0.200815 for DR-SC and 0.204994 for TIS. DR-SC to TIS variance stays at 0.980 across all four batches." %}

<div class="caption" markdown="1">

**BATCH-SIZE DIAGNOSTIC**  Batch size changes the noise scale, not the variance ordering in this fixed estimator comparison. At B = 4, the normalized variances are 0.186163 (SC), 0.200815 (DR–SC) and 0.204994 (TIS); at B = 256 they are 64 times smaller. Engine batch remains one, and group size remains four.

</div>

Each batch averages $$ B/4 $$ independent four-response groups, so its variance scales exactly as $$ 1/B $$. The percentage gap between methods is unchanged.

For these eight prompts, $$ p(a)/q(a) $$ lies between 0.746 and 1.425; TIS never clips. There is an additional simplification: the two-action policy and $$ \sum_i A_i=0 $$ imply, for every shared group,

<div class="overflow-auto" markdown="1">

$$
\widehat g_{\mathrm{TIS}}
=\frac{p_1(1-p_1)}{q_1(1-q_1)}\,\widehat g_{\mathrm{SC}}.
$$

</div>

Here $$ p_1=p(a=1) $$ and $$ q_1=q(a=1) $$. The multiplier is only 0.999–1.102. Increasing sampling noise cannot create the missing importance-weight tail.

</details>

### Stronger mismatch reveals the cost of large weights
{: #stronger-mismatch-reveals-the-cost-of-large-weights }

To test the missing condition, we hold the master checkpoint fixed and lower the sampler’s precision. Alongside native FP16/BF16 autocast, we simulate INT4 and INT3 weight rounding, with the rounded weights executed in FP32.

The weight tails now change substantially. Across the full vocabulary, $$ \mathbb E_q[(p/q)^2] $$ averages 1.0007 in BF16, 1.736 at INT4 and 357.0 at INT3. We separately measure full-parameter gradient variance for the constrained A/B policy:

{% include figure.liquid path="assets/blog/score-centering-done-right/precision-stress.svg" class="img-fluid rounded z-depth-1" loading="lazy" alt="Precision stress results: full-vocabulary IS weight second moments rise sharply for simulated four-bit and three-bit weights. A separate binary-policy full-parameter diagnostic shows TIS variance 1.23 and 1.29 times SC, while unclipped IS is 2.13 and 7.05 times SC." %}

<div class="caption" markdown="1">

**FIGURE 3**  Left: full-vocabulary importance-weight moments. Right: parameter-gradient variance for a separately normalized A/B policy. These measure different objects; the left panel is not a full-vocabulary gradient-variance result. Ratios on the right divide arithmetic means of absolute conditional variances across eight prompts.

</div>

With INT4 weights, unclipped IS has **2.13× SC’s variance**; at INT3 it reaches **7.05×**. Truncation holds TIS to **1.23× and 1.29×**. Once large ratios appear, the difference between IS and TIS becomes substantial.

This explains both the appeal of an estimator without importance weights and the value of TIS’s cap. Truncation controls much of the extra noise, at the cost of changing the estimator mean. The full-vocabulary weight moments are a diagnostic of tails; the measured gradient variances belong to the A/B policy.

This experiment tests the variance mechanism. It cannot demonstrate a checkpoint correction: the master weights have not moved, so $$ \Delta=0 $$ and DR–SC equals SC. It also measures simulated weight rounding, rather than a native INT4/INT3 inference kernel.

<details markdown="1">
<summary>Precision stress: what was quantized, and what was measured?</summary>

FP16 and BF16 use native CPU autocast. The 8/4/3-bit configurations use symmetric group-128 rounding on 497,614,848 linear-weight parameters, excluding the output head; dequantized values execute in FP32. This isolates weight-rounding effects. It is neither a native low-bit kernel benchmark nor a calibrated GPTQ/AWQ quantizer. The master checkpoint, inputs and temperature remain fixed; there is no optimizer update.

At 4 bits, an average 6.27% of full-vocabulary $$ q $$ mass exceeds the TIS cap of 2. At 3 bits, weight second moments range from 27.1 to 2,153.2 across prompts, with median 126.8: the mean of 357.0 is tail-sensitive. Small FP16/BF16 differences are not guaranteed to order by bit count; FP16 has a different precision/range trade-off. [Qi et al., 2025](https://arxiv.org/abs/2510.26788)

The A/B variance uses fresh VJPs through all text parameters of the FP32 trainer and exact enumeration of four-response groups. Direct weighted-loss backpropagation was independently checked against the VJP contraction. No quantized-engine oracle Jacobian or $$ q $$-target bias is claimed. Lower variance under severe quantization is not automatically desirable: concentrated, consistently wrong groups can have zero normalized advantages. Full-vocabulary parameter-gradient variance and native low-bit inference engines remain untested.

</details>

<details markdown="1">
<summary>Targets, estimator definitions and derivative checks</summary>

The synthetic study uses 12 actions, 24 parameters, 16 seeds and three reward families. Its projected-score correction is assigned a 25% residual error; it is not an HVP implementation.

The Qwen study differentiates through 752,393,024 text parameters. Its eight A/B prompts yield 16 ordered four-response groups per prompt, which we enumerate exactly.

In this Qwen study, $$ A_i=(R_i-\overline R)/(s_R+10^{-6}) $$, using sample standard deviation with `ddof=1`. The references are explicitly $$ g_q=\mathbb E_{q^4}[\frac14\sum_i A_i\nabla\log q(a_i)] $$ and the analogous $$ g_p $$ under $$ p^4 $$. These group-standardized directions are not generally the raw gradients of expected reward used in the opening derivation.

For these single-token, same-prompt groups, $$ \sum_i A_i=0 $$, so the common SC centering term cancels within each group. This study isolates score/Jacobian transport; it does not establish an independent benefit from subtracting the centering term. Multi-token trajectories do not share one common prefix at every position.

The GRPO row uses the derivative of $$ -\operatorname{mean}\min(r_iA_i,\operatorname{clip}(r_i,0.8,1.2)A_i) $$, $$ r_i=p(a_i)/q(a_i) $$. TIS uses the detached weight $$ \min(r_i,2) $$ once on the current-policy score: the decoupled PPO ratio is one at this evaluation point. There is no duplicate IS factor. Group-dependent advantages mean per-action IS alone is not an unbiased change from $$ q^4 $$ to $$ p^4 $$; an additional full-group IS estimator is retained in the evidence. [Schulman et al., 2017](https://arxiv.org/abs/1707.06347); [Shao et al., 2024](https://arxiv.org/abs/2402.03300)

All methods share sampled groups in the Monte Carlo check (three seeds, 2,000 repetitions per seed, nested batch prefixes). The plotted moments themselves are exact enumerations computed from full-parameter gradient inner products. CPU recurrent/chunk/cache references were checked with VJPs and directional finite differences, rather than logits alone. Lag-4 HVP versus full-gradient central differences had relative error $$ 4.23\times10^{-4} $$ on the checked prompt/direction. BF16 diagnostics are excluded from these FP32 plots.

A separate full-vocabulary Qwen score diagnostic, with two prompt/direction pairs, found that correction left 1.8–2.0% of STE’s error at relative per-tensor displacement 0.001, but overshot at 0.1. That supports a local approximation, not a universal radius. In the eight-prompt CPU study, gradient-plus-HVP basis construction averaged 7.54 seconds at lag 4 versus 2.47 seconds for the same-checkpoint gradient basis; this is diagnostic construction cost, not per-online-batch throughput.

The downloadable evidence includes all Qwen methods and configurations, individual prompt statistics, plotting values and source hashes.

</details>

## 5. From a better gradient to asynchronous learning
{: #learning }

Agreement with an oracle at a fixed checkpoint does not establish better learning. Once updates begin, each method changes the next sampling distribution. We therefore test the correction in a running asynchronous system.

Qwen3.5-0.8B learns to answer $$ (a+b)\bmod10 $$ using native SLIME/Megatron training and SGLang rollouts. We compare GRPO, SC and DR–SC over 30 updates, with the same seed and training-question order. Each update contains **16 prompts × four responses = 64 samples**; the engine microbatch is one.

The queue produces genuine checkpoint lag: roughly a quarter of the responses are one update stale. This puts the correction in the setting its derivation addresses.

{% include figure.liquid path="assets/blog/score-centering-done-right/async-learning.svg" class="img-fluid rounded z-depth-1" loading="lazy" alt="One paired seed of asynchronous Qwen training at batch 64. Corrected STE rises to 100 percent on a fixed 20-question test; GRPO and STE end at 10 percent." %}

<div class="caption" markdown="1">

**FIGURE 4**  One paired seed, a fixed 20-question test, and actual asynchronous sampling. At step 30, GRPO always answers “1”, SC always answers “3”, and DR–SC answers all 80 evaluation draws correctly. Repeated draws are not 80 independent test questions. No TIS learning run is included.

</div>

On a fixed set of **20 held-out questions**, DR–SC reaches **100%** sampled accuracy, while GRPO and SC finish at **10%**. The baselines settle on a constant answer; DR–SC learns the task in this run.

The result is one paired seed on a small, single-token task. An earlier five-seed pilot provides a useful counterpoint: correction learns earlier on average at refresh interval four, but ties SC at the final checkpoint, and the early paired difference is not statistically resolved. The runs use different settings and cannot be pooled. Together they establish a successful learning case, with the size and reliability of the gain still open.

### The cost: training updates versus the full RL run
{: #the-cost-training-updates-versus-the-full-rl-run }

The HVP adds work to the trainer, but training time and full RL time measure different costs. We recovered a separate timer for each of the 30 training updates, covering the forward/backward passes, optimizer and reference diagnostics. The full-run clock also includes initialization, rollout, evaluation, weight transfer and waiting.

<div class="table-responsive" markdown="1">

| Method | Training updates (min) | Full RL run (min) |
| :--- | ---: | ---: |
| SC | 67.92 | 71.32 |
| DR–SC | 91.24 | 94.54 |
| GRPO | 68.58 | 71.85 |

</div>

Relative to SC, DR–SC increases **training-update time by 34.3%** and **full RL time by 32.6%**. Its training updates average 182.5 seconds instead of 135.8 seconds, and the full run takes 23.2 minutes longer. These are measured costs for the instrumented study, including reference diagnostics; they do not isolate the HVP kernel’s overhead.

Training accounts for most of this short-response experiment, so the two increases are close. In an asynchronous system, rollout and training can overlap: a training slowdown affects total time according to which stage limits throughput. We therefore cannot carry the 32.6% figure over to a generation-heavy workload. The practical comparison should be learning progress against elapsed time as well as optimizer steps.

<details markdown="1">
<summary>Timing boundaries and the separate CPU HVP probe</summary>

Training-update time sums the 30 unique `perf/actor_train_time` records, one for each completed update. That timer surrounds the training routine and includes the before-step reference hook, 64 forward/backward microbatches, correction/gradient diagnostics, optimizer, scheduler and logging. The reference is initialized lazily inside its first hook. The timer uses wall-clock time, rather than CUDA events for individual kernels.

Full RL time comes from `run_status.json`’s `elapsed_seconds`, measured from before worker initialization until the training driver returns. It includes initialization, rollouts, evaluation, transfers, waiting and diagnostics. Worker rollout durations cannot simply be added to training time because asynchronous work overlaps. The `seconds` field in the gradient diagnostics measures only the loss hook and is not a training-update timer.

The separate CPU probe takes 2.47 seconds to construct a gradient basis and 7.54 seconds for gradient-plus-HVP at lag four, or **3.05×**. That diagnostic omits the optimizer, sampling and scheduler and uses a different execution path. It is not the measured slowdown of native training, which is 1.34× here.

These are three runs with one paired seed, not repeated throughput benchmarks. No diagnostic-free training benchmark or isolated HVP kernel timing was recorded. The embedded evidence retains all 90 per-update timing records, clock definitions, source hashes and aggregation formulas.

</details>

<details markdown="1">
<summary>Learning-study implementation and scope</summary>

The 30-step study uses seed 42, 480 distinct training questions in the same order, Adam with learning rate $$ 10^{-6} $$, operands from 0 to 99 and one output token. Of 1,920 responses per run, 464 are one update stale and 1,456 are fresh. There is no completed synchronous control in this evidence snapshot; the result does not establish that asynchrony caused the baseline failures.

All text parameters are trained. Two RTX A6000s execute independent runs; rollout and trainer are colocated within each run. The inference path uses repaired FP32 Q/K normalization and rotary computation. A separate two-step accumulated-gradient audit matched the native correction path to its reference below $$ 9\times10^{-6} $$ relative error.

Evaluation uses 20 disjoint questions with four sampled answers each. At step 30, GRPO answers “1” on all 80 draws, SC answers “3”, and DR–SC answers all correctly. Those draws are not 80 independent test problems; this small single-token task does not establish general arithmetic ability. No TIS learning run is included.

In the separate five-seed, 24-step pilot, corrected SC reaches 30.75% at step 12 versus 18.00% for SC, but both finish at 32.00%. Additional seeds and synchronous controls for the larger-batch study were incomplete at the frozen evidence cutoff and are excluded.

</details>

## 6. Conclusion
{: #takeaway }

Starting from $$ \mathbb E_q[R] $$ leads to an ordinary PG loss on the responses we actually sample. Score centering makes that loss trainable by substituting $$ J_p $$ for the unavailable $$ J_q $$. For nearby checkpoints, $$ z_p+J_p\Delta $$ supplies a better backward proxy, and one HVP computes its correction.

The Qwen gradient measurements show the intended effect: a smaller bias floor against the $$ q $$ oracle. The precision probe explains when importance weights add substantial noise and why truncation helps. The asynchronous case shows that the correction can translate into learning, while leaving its reliability across seeds and tasks to further study.

---

## References & evidence
{: #references }

1. Marek, M. & Ryabinin, M. (2026). [Score Centering Stabilizes Off-policy Reinforcement Learning.](https://arxiv.org/abs/2609.20807) Original additive correction and empirical study.
2. Bengio, Y., Léonard, N. & Courville, A. (2013). [Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation.](https://arxiv.org/abs/1308.3432) The straight-through estimator.
3. Mukherjee, S., Yuan, L., Hakkani-Tur, D. & Peng, H. (2025). [Reinforcement Learning Finetunes Small Subnetworks in Large Language Models.](https://arxiv.org/abs/2505.11711) Empirical parameter-update sparsity.
4. Zhu, H. et al. (2025). [The Path Not Taken: RLVR Provably Learns Off the Principals.](https://arxiv.org/abs/2511.08567) Update geometry and the role of finite precision.
5. Pearlmutter, B. A. (1994). [Fast Exact Multiplication by the Hessian.](https://doi.org/10.1162/neco.1994.6.1.147) Matrix-free HVPs.
6. Schulman, J. et al. (2017). [Proximal Policy Optimization Algorithms.](https://arxiv.org/abs/1707.06347) Clipped policy surrogates.
7. Shao, Z. et al. (2024). [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models.](https://arxiv.org/abs/2402.03300) GRPO and group-relative advantages.
8. He, H. & Thinking Machines Lab (2025). [Defeating Nondeterminism in LLM Inference.](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/) Batch invariance and numerical execution.
9. Wang, Y. (2026). [Defending Against the Training–Inference Numeric Mismatch in RL.](https://yichuan-w.github.io/blog/GDN-train-inference-mismatch-asyncRL/) Systems background and motivation.
10. Qi, P. et al. (2025). [Defeating the Training-Inference Mismatch via FP16.](https://arxiv.org/abs/2510.26788) Precision/range trade-offs in RL training and inference.
11. Yao, F. et al. (2025). [Your Efficient RL Framework Secretly Brings You Off-Policy RL Training.](https://fengyao.notion.site/off-policy-rl) Learner–sampler mismatch and truncated importance sampling.

**Inspect the measurements**<a href="{{ &#39;/assets/blog/score-centering-done-right/score-centering-evidence.json&#39; | relative_url }}" download="score_centering_evidence.json">Download the evidence (JSON)</a> — figure values, per-prompt Qwen statistics, the precision probe, the complete asynchronous case audit, configurations and source hashes. The figures and evidence are available with this article. External references require an internet connection.

Evidence snapshot: October 11, 2026. Completed experiments and a separate CPU precision probe; no new training was launched for this article. The implementation measures the same update directions as the PG + STE losses derived here.
