# RKHS, kernel mean embeddings, and MMD for sim2real domain adaptation

The theory behind the feature-alignment loss in `train_seg_detr.py`
(`--align-real`). It explains *why* pulling the encoder's sim-feature
distribution toward the real distribution with a kernel discrepancy is the
right tool, how the MMD we compute relates to the underlying Hilbert-space
geometry, and where the operator view connects sim2real to a difference of
two nearby transition kernels.

See also: [`global-first-architecture.md`](global-first-architecture.md)
(where alignment sits in the backbone loop) and
[`sim2real-translation-findings.md`](sim2real-translation-findings.md) (why
pixel translation does *not* close the frozen-feature gap — validate transfer
with a held-out MMD/probe, which is exactly the object defined here).

---

## 1. The construction

Start with a positive-definite kernel `k` and its reproducing-kernel Hilbert
space `H_k`. The reproducing property is the whole engine: `k(·, x) ∈ H_k` is
the feature map `φ(x)`, and for any `f ∈ H_k`,

```
⟨f, k(·, x)⟩_H = f(x).
```

The **kernel mean embedding** of a distribution `P` is the expected feature
map,

```
μ_P = E_{X∼P}[φ(X)] = ∫ k(·, x) dP(x)  ∈  H_k,
```

a Bochner integral in the Hilbert space (it exists whenever
`E[√k(X,X)] < ∞`, automatic for bounded kernels like the Gaussian). A
probability measure becomes a single point in a function space.

This is not just notation. Combine the embedding with the reproducing
property:

```
⟨f, μ_P⟩_H = E_P[f]      for every f ∈ H_k.
```

The embedding is a lossless bookkeeping device for the expectations of *all*
RKHS functions simultaneously — one vector that knows every `H_k`-moment of
`P` at once.

---

## 2. Characteristic kernels — the "no information loss" property

The map `P ↦ μ_P` is interesting only if it is injective. Kernels for which
it is are **characteristic**; the Gaussian and Laplace kernels are the
standard examples on `R^d`. For a characteristic kernel `μ_P` determines `P`
uniquely, so the embedding is a faithful representation — nothing about the
distribution is thrown away. This is what lets us do statistics on
distributions by doing geometry on their embeddings.

The RBF kernel we use (`_mmd`, below) is characteristic, so in the population
limit MMD = 0 **iff** the two feature distributions are identical.

---

## 3. MMD: the metric that falls out

Distance between embeddings is a distance between distributions:

```
MMD(P, Q) = ‖μ_P − μ_Q‖_H.
```

Two faces of this object, and the interplay is the useful part.

**Computational face** — expand the norm and only kernel evaluations
survive:

```
MMD²(P,Q) = E_{X,X'}[k(X,X')] + E_{Y,Y'}[k(Y,Y')] − 2·E_{X,Y}[k(X,Y)],
```

with `X, X' ∼ P` and `Y, Y' ∼ Q` independent. No densities, no
normalization, no intractable integrals — given samples you plug into a U- or
V-statistic and you are done. **This is the exact expression `_mmd` computes**
(`kxx.mean() + kyy.mean() − 2·kxy.mean()`, a biased V-statistic).

**Functional-analytic face** — by Cauchy–Schwarz,
`‖μ_P − μ_Q‖ = sup_{‖f‖_H ≤ 1} ⟨f, μ_P − μ_Q⟩`, so

```
MMD(P,Q) = sup_{‖f‖_H ≤ 1} ( E_P[f] − E_Q[f] ).
```

That is an **integral probability metric**: the witness `f*` achieving the
sup is the function that maximally separates the two distributions, available
in closed form as a normalization of `μ_P − μ_Q`. Wasserstein has the same
IPM shape but constrains to 1-Lipschitz functions; MMD's constraint set is
the RKHS unit ball, which is exactly what makes it cheap while Wasserstein
generally is not.

---

## 4. Estimation and its caveats

The empirical embedding `μ̂_P = (1/n) Σ_i φ(x_i)` concentrates at

```
‖μ̂_P − μ_P‖_H = O_p(n^{-1/2}),
```

with a constant governed by the kernel rather than the ambient dimension.
That dimension-independence in the RKHS norm is often quoted as MMD "beating
the curse of dimensionality" — **but be careful**:

- The *statistical power* of an MMD two-sample test still degrades badly in
  high dimensions. Characteristic-ness guarantees separation in the limit;
  it says nothing about finite-sample detectability.
- **Bandwidth choice matters enormously.** We use the median heuristic
  (`med = median of pairwise squared distances`) and, to be robust to it,
  a *mixture* of bandwidths `0.5·med, 1.0·med, 2.0·med` summed together —
  so no single length scale has to be right.

Practical consequence for us: a small measured MMD is necessary but not
sufficient evidence of alignment; pair it with the linear-probe
separability check (PAD) from the translation findings.

---

## 5. The energy-distance bridge

Worth flagging: the **energy distance** from the statistics literature is not
a separate object — it is MMD with the distance-induced kernel
`k(x,y) = ½(‖x‖ + ‖y‖ − ‖x − y‖)`. Sejdinovic, Sriperumbudur, Gretton &
Fukumizu (2013) made the equivalence precise: distance-based and
kernel-based statistics are the same theory viewed through two charts.
Anything built on an energy functional has a kernel-mean-embedding dual.

---

## 6. Conditional mean embeddings — the operator bridge to sim2real

Embed a *conditional* distribution `P(Y | X = x)` as

```
μ_{Y|x} = E[φ(Y) | X = x] = C_{Y|X} φ(x),
```

where `C_{Y|X}: H_X → H_Y` is a linear **operator** between RKHSs, built from
the (cross-)covariance operators as `C_{YX} C_{XX}^{-1}` — with the inverse
replaced by a regularized `(C_{XX} + λI)^{-1}` in practice, since the raw
inverse is unbounded.

A Markov transition kernel `P(s' | s, a)` — the actual object that differs
between simulator and reality — is a conditional distribution, and its
embedding is one of these operators. The **sim2real dynamics gap** becomes a
difference of operators,

```
‖ C^sim_{S'|S,A} − C^real_{S'|S,A} ‖,
```

i.e. the reality gap as an operator-norm discrepancy in a Hilbert space,
which slots directly into contraction-mapping perturbation bounds. That is
the clean version of "sim2real is two nearby operators."

For our static detection problem we use the *marginal* embedding (Sections
1–4) on encoder features, not the conditional operator — but the operator
view is the right frame if/when the loop grows a dynamics/transition
component.

---

## 7. Why this is the right tool for the gap

Three concrete payoffs in the sim2real setting:

1. **Two-sample test** — a principled, kernelized way to ask whether sim and
   real rollouts come from the same distribution, *quantifying* the gap
   rather than eyeballing it. This is how we report the domain gap (e.g.
   DINOv2 sim→real_dev MMD ≈ 0.077, styled ≈ 0.056).
2. **Training objective** — minimizing MMD between sim-feature and
   real-feature distributions *is* the alignment loss in deep
   domain-adaptation methods, and is exactly `--align-real`.
3. **Interpretable witness** — `f*` points at the regions of feature/state
   space where sim and real disagree most: actionable signal for where to
   concentrate domain randomization or system identification.

**The framing in one sentence:** the embedding is a *linearization of the
space of measures*. It extends linearly to signed measures, carrying the
nonlinear simplex of distributions into a convex subset of a Hilbert space,
where mixtures become vector-space combinations and "distance between
distributions" becomes an honest norm. Differentiating through measures,
optimizing over them, comparing them — all become linear algebra on their
embeddings. It is a functor from the badly-behaved category of measures into
the well-behaved category of Hilbert spaces, and MMD plus the
conditional-embedding operators are what you compute with once you are there.

---

## 8. How the theory maps onto `train_seg_detr.py`

The alignment loss (`--align-real`, default weight `0.1`) pulls the encoder's
sim-feature distribution toward the **unlabeled** `real_dev` distribution. It
has three terms (`_align_loss`), each an instance of the above:

| Code | Theory | Notes |
|------|--------|-------|
| `_mmd(a, b)` | empirical `MMD²` biased V-statistic (§3 computational face) | `kxx.mean()+kyy.mean()−2·kxy.mean()` |
| L2-normalize features then RBF | characteristic kernel on the sphere (§2) | removes scale so only direction is compared |
| `med = median(pairwise d²)`, mix `0.5/1.0/2.0·med` | median heuristic + multi-bandwidth (§4 caveat) | robust to bandwidth misspecification |
| **global** term: `_mmd(pool(f_sim), pool(f_real))` | embedding of the pooled-feature marginal | one vector per image |
| **local** term: `_mmd` over `--align-local` sampled per-location features at level `--align-local-level` | embedding of the per-location feature marginal | aligns texture/local statistics, not just the image-level mean |
| **EMA** term: `‖ema_s − ema_t‖²` with rate `--align-ema` | first-moment (mean-embedding) match, smoothed | low-variance anchor; `μ_P − μ_Q` is the witness direction (§3) |

Discipline (from the data-split rules): the real distribution used for
alignment is `real_dev` only; `real_holdout` is never touched, so the
held-out MMD/probe remains an honest measure of transfer.

### Caveats carried into practice
- A near-zero training MMD does **not** by itself prove the gap is closed
  (§4): high-dim power is weak. Cross-check with the held-out linear-probe
  separability (PAD) — the translation findings showed pixel translation
  driving a GAN loss down while the held-out MMD/probe stayed wide open.
- Alignment weight is a balance: too high and the encoder collapses
  sim features onto real ones at the cost of the supervised seg objective.
  `0.1` is the working default.

---

## Further reading branches

- The covariance-operator formalism and its tie to Gaussian processes.
- Regularization theory for the conditional operator `(C_{XX} + λI)^{-1}`.
- Witness-function behavior for specific kernels (what `f*` looks like for
  the RBF vs. the distance kernel).
