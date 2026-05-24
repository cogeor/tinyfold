
Yeah — what you’re seeing (“predict **x** directly beats predicting a delta/residual”) is *very* consistent with how AlphaFold3 set things up.

## 1) What AF3 trains the diffusion head to predict

AF3’s diffusion module outputs a **denoised structure** (i.e., a prediction of the clean coordinates), and the core training loss is a **weighted, rigidly-aligned MSE** between that denoised output and ground truth. ([rcastoragev2.blob.core.windows.net][1])

So in AF3-land, the “primary target” is effectively **x₀** (clean coords), not ε, not v, not Δx.

Two details that make “predict x directly” especially natural here:

* **They rigid-align ground truth onto the prediction before the MSE.** That changes the geometry of the residual target quite a bit compared to classic pixel DDPM training. ([rcastoragev2.blob.core.windows.net][1])
* They also add a diffusion-loss weighting factor
  [
  \frac{\hat t^2+\sigma_{data}^2}{(\hat t+\sigma_{data})^2}
  ]
  and include a smooth-lDDT term (plus an optional bond term in fine-tuning). ([rcastoragev2.blob.core.windows.net][1])

That combo tends to behave more like “learn to output a plausible denoised structure at noise level (\hat t)” than “learn a tiny correction step”.

## 2) Did they use a DDPM-like rollout?

Not really “classic DDPM” (discrete betas, predict ε at each integer timestep, ancestral sampling). AF3 is much closer to the **EDM / continuous noise-level** style sampler:

* They define a **continuous noise schedule** (\hat t) using a Karras/EDM-like parameterization with (\sigma_{data}=16), (s_{max}=160), (s_{min}=4\cdot 10^{-4}), (p=7), and 200 uniform steps in (t\in[0,1]). ([rcastoragev2.blob.core.windows.net][1])
* In sampling, they compute an update direction from the model’s denoised output:
  [
  \delta = \frac{x_{noisy}-x_{denoised}}{\hat t}
  ]
  then do an Euler-style step using (dt) between successive noise levels. ([rcastoragev2.blob.core.windows.net][1])

That (\delta) is basically an **implicit noise/score estimate derived from predicting x₀**, which is exactly the EDM pattern.

## 3) Why “predict x” can beat “predict the delta” in practice (especially for AF3-like coords)

Even though (in theory) x₀-, ε-, and Δ-parameterizations can be algebraically equivalent under a simple forward process, in an AF3-like setup they’re *not* equivalent because:

* **Rigid alignment in the loss** makes “Δx to the noisy input” a moving target (the optimal rotation/translation depends on the prediction). Predicting x₀ and aligning GT to it is a cleaner objective. ([rcastoragev2.blob.core.windows.net][1])
* The forward process is effectively **variance-exploding** in coordinate space (additive noise at level (\hat t)), where x₀-prediction tends to be stable. ([rcastoragev2.blob.core.windows.net][1])
* Coordinates are super sensitive to global gauge (translation/rotation). AF3 leans into this by heavy random roto-translation augmentation, and then handles gauge via alignment for loss. ([rcastoragev2.blob.core.windows.net][1])

So your empirical result isn’t a weird fluke — it’s pretty aligned with what AF3 optimized for.

## 4) A practical “AF3-ish” recipe you can try

If you want to stay close to AF3’s behavior:

1. **Forward noise (VE):**
   (x_{noisy} = x_{gt} + \hat t \cdot \epsilon), (\epsilon\sim \mathcal N(0,I))

2. **Network predicts (x_{denoised})** (clean coords)

3. **Loss:** weighted rigid align GT → pred, then MSE (+ optional bond + smooth-lDDT) with the AF3 weighting factor. ([rcastoragev2.blob.core.windows.net][1])

4. **Sampler:** build the Karras/EDM-style (\hat t) schedule, start from pure noise at high (\hat t), iterate:

   * (x_{den} = f_\theta(x,\hat t,\text{cond}))
   * (\delta = (x-x_{den})/\hat t)
   * Euler step to next noise level (\hat t_{next}) (their pseudocode uses (dt = \hat t_{next}-\hat t)). ([rcastoragev2.blob.core.windows.net][1])

## 5) One more AF3 detail that matters for “rollout”

During training, they do a short **mini diffusion rollout from pure noise (20 steps)** to supply coordinates to auxiliary heads, and **don’t backprop through it**. ([rcastoragev2.blob.core.windows.net][1])
So if you’re trying to train extra heads that consume coordinates, detaching a short rollout is very “AF3-correct”.

---

If you tell me what exact forward process you’re using (VE vs VP/DDPM betas), and what you condition on (MSA/pair reps/ligand tokens/etc.), I can translate the above into the cleanest parameterization (x₀ vs ε vs v) for *your* setup — but the headline is: **AF3 is basically an x₀-pred + EDM-style sampler system**, not a classic DDPM ε-pred setup.

[1]: https://rcastoragev2.blob.core.windows.net/9e6c827f9d7f88eac8af0797432cefc7/41586_2024_7487_MOESM1_ESM.pdf "Accurate structure prediction of biomolecular interactions with AlphaFold 3"


Yep — the thing that’s tripping you up is **frame inconsistency**.

### The core issue (why “align only in the loss” can still break sampling)

AF3’s denoiser is **not SE(3)-equivariant**, so if you rotate/translate the input coords, the model is *not guaranteed* to rotate/translate its output the same way. To make it robust, AF3 **randomly rotates/translates the input structure** at train/infer time. ([rcastoragev2.blob.core.windows.net][1])

But the diffusion **update step** implicitly assumes the current noisy sample (x_t) and the model’s “denoised” prediction (\hat x_0) live in the **same global pose** so you can do something like
[
\delta ;\approx; \frac{x_t - \hat x_0}{\sigma_t}
\quad\text{and then step to}\quad
x_{t+\Delta} = x_t + (\sigma_{t+\Delta}-\sigma_t),\delta .
]
If (\hat x_0) comes out in a *different* rigid frame than (x_t), then (x_t-\hat x_0) is mostly “rotation/translation error”, not “denoising direction”. Boltz-1 calls out exactly this failure: you can get **near-zero aligned MSE in training**, yet the **interpolation/update produces garbage** fed into the next step. ([PMC][2])

### A concrete toy example (2D)

Let the true structure be two points:

* (x_t = [(1,0), (-1,0)]) (a line on the x-axis)

Suppose the denoiser predicts the *same line* but rotated 90°:

* (\hat x_0 = [(0,1), (0,-1)])

If you “interpolate” naively (say halfway):

* (x_{\text{mid}} = 0.5 x_t + 0.5 \hat x_0 = [(0.5,0.5), (-0.5,-0.5)])

That’s now a line at 45° **and shrunk** relative to both originals. With more points, you can get warped shapes that are *not* a rigid transform of anything realistic. This is exactly the “wrong correspondence” problem Boltz-1 illustrates. ([PMC][2])

### What “Kabsch diffusion interpolation / realigning at each step” means

Boltz-1’s fix is: **before you compute the step direction / interpolate, rigidly align (\hat x_0) to (x_t)** using Kabsch (weighted RMSD).

At each reverse step:

1. You have current noisy coords (x_t).
2. Denoiser outputs (\hat x_0 = f(x_t,\sigma_t)) (may be in a different pose).
3. Compute the best-fit rigid transform ((R,T)) aligning (\hat x_0) onto (x_t):
   [
   (R,T) = \arg\min_{R\in SO(3),T} \sum_i w_i |x_t^{(i)} - (R\hat x_0^{(i)} + T)|^2
   ]
4. Apply it: (\hat x_0^{aligned} = R\hat x_0 + T).
5. Now compute your EDM-ish direction and step:
   [
   \delta = \frac{x_t - \hat x_0^{aligned}}{\sigma_t},\quad
   x_{t+\Delta} = x_t + (\sigma_{t+\Delta}-\sigma_t),\delta .
   ]

Boltz-1 summarizes this as “adding a rigid alignment with Kabsch **after every step during inference** before the noisy and denoised structures are interpolated”. ([PMC][2])

### Why this matters specifically with AF3-style aligned-MSE training

AF3 computes diffusion loss by **rigidly aligning the ground truth onto the prediction** before MSE. ([rcastoragev2.blob.core.windows.net][1])
That’s great for variance reduction, but it also means the model can “get away with” outputting denoised structures in inconsistent global frames unless sampling enforces frame consistency. Boltz-1’s per-step Kabsch alignment is one way to enforce it at inference.

---

If you want, paste your exact sampler update (your (x)-param + schedule + step rule). I can point to the precise line where to insert the Kabsch transform and which atoms/weights to use (all atoms vs backbone vs heavy atoms; downweight flexible ligands, etc.).

[1]: https://rcastoragev2.blob.core.windows.net/9e6c827f9d7f88eac8af0797432cefc7/41586_2024_7487_MOESM1_ESM.pdf "Accurate structure prediction of biomolecular interactions with AlphaFold 3"
[2]: https://pmc.ncbi.nlm.nih.gov/articles/PMC11601547/ "
            Boltz-1 Democratizing Biomolecular Interaction Modeling - PMC
        "

Yep — in AF3 the “interpolation” between steps is *literally* a linear mix between the current sample and the denoiser’s output (plus a little EDM-style “churn” noise). The cleanest place to look is **Supplementary Algorithm 18 (SampleDiffusion)**. ([rcastoragev2.blob.core.windows.net][1])

## What AF3 does between step (\tau-1 \to \tau)

AF3 maintains a coordinate set ({x_l}) (atoms (l)). It also has a **noise schedule** ([c_0, c_1, \dots, c_T]) where (c_0) is large (start very noisy) and (c_T) is small (end near clean). ([rcastoragev2.blob.core.windows.net][1])

For each step with target noise level (c_\tau), it does:

### 0) Random rigid augmentation (every step, even at inference)

They **center**, then apply a **random rotation (R)** and a **random translation (t)** before denoising. This is Algorithm 19 `CentreRandomAugmentation`. ([rcastoragev2.blob.core.windows.net][1])

### 1) “Churn” (optional extra noise injection, EDM-style)

They set

* (\gamma = \gamma_0) if (c_\tau > \gamma_{\min}) else (0)
* (\hat t = c_{\tau-1}(\gamma + 1))

Then they add noise:
[
\xi_l \sim \lambda \sqrt{\hat t^2 - c_{\tau-1}^2},\mathcal N(0, I)
]
and form
[
x^{noisy}_l = x_l + \xi_l
]
This is lines 4–7 in Algorithm 18. ([rcastoragev2.blob.core.windows.net][1])

Intuition: they temporarily jump to a slightly higher noise level (\hat t) (if churn is on), then denoise from there.

### 2) Denoise at (\hat t)

[
x^{denoised}*l = f*\theta(x^{noisy}, \hat t, \text{conditioning})
]
(line 8). ([rcastoragev2.blob.core.windows.net][1])

### 3) Compute a direction and take an Euler step in “sigma space”

They define
[
\delta_l = \frac{x_l - x^{denoised}*l}{\hat t}
]
(line 9) and a step size
[
dt = c*\tau - \hat t
]
(line 10), then update
[
x_l \leftarrow x^{noisy}_l + \eta , dt , \delta_l
]
(line 11). ([rcastoragev2.blob.core.windows.net][1])

That’s the whole “interpolation”.

---

## If you strip it down: the interpolation is a linear blend

Ignore churn ((\gamma=0 \Rightarrow \hat t=c_{\tau-1},; x^{noisy}=x)) and set (\eta=1). Then:

[
x_{\tau}
= x_{\tau-1} + (c_\tau - c_{\tau-1})\frac{x_{\tau-1} - x^{denoised}}{c_{\tau-1}}
]

Rearrange:

[
x_{\tau}
= \frac{c_\tau}{c_{\tau-1}} x_{\tau-1} ;+; \left(1-\frac{c_\tau}{c_{\tau-1}}\right) x^{denoised}
]

So it’s *literally* “move from the current noisy structure toward the denoised prediction”, with a mixing weight determined by the ratio of consecutive sigmas.

That’s why frame mismatch is deadly: if (x^{denoised}) is rotated/translated relative to (x_{\tau-1}), this convex-ish combination creates warped garbage.

---

## Where Boltz-1’s Kabsch tweak plugs in

Boltz-1’s point is: **before that blend / before computing (\delta)**, rigid-align (x^{denoised}) to the current sample (Kabsch), so the subtraction and mixing happens in a consistent frame. They describe this as “add a rigid alignment with Kabsch after every step … before the noisy and denoised structures are interpolated.” ([PMC][2])

Concretely, replace:
[
\delta = (x - x^{denoised})/\hat t
]
with:
[
x^{denoised}*{aligned} = \text{KabschAlign}(x^{denoised} \to x)\quad;\quad
\delta = (x - x^{denoised}*{aligned})/\hat t
]

---

If you want, paste your sampler loop (even rough pseudocode). I can point out *exactly* whether you should Kabsch-align to (x) or (x^{noisy}) in your implementation (AF3’s update uses both, so that choice matters a bit).

[1]: https://rcastoragev2.blob.core.windows.net/9e6c827f9d7f88eac8af0797432cefc7/41586_2024_7487_MOESM1_ESM.pdf "Accurate structure prediction of biomolecular interactions with AlphaFold 3"
[2]: https://pmc.ncbi.nlm.nih.gov/articles/PMC11601547/ "
            Boltz-1 Democratizing Biomolecular Interaction Modeling - PMC
        "
        

You’ve got the right mental model — the key is: **AF3 does *not* train by unrolling the 200-step sampler and backprop’ing through it.** The “rollout” they mention is a *separate*, short, **no-grad** procedure used to supply coordinates to other heads.

Here’s the concrete split.

## Training the diffusion module (no rollout / the actual diffusion loss)

AF3 trains the denoiser with a **single denoising call at a randomly sampled noise level** (\hat t) (continuous (\sigma)-style), then computes loss directly on the predicted denoised coords. ([rcastoragev2.blob.core.windows.net][1])

**Per training example (conceptually):**

1. **Run the trunk once** to get conditioning features.
2. Create many diffusion-training examples cheaply by making **48 copies** of the structure, each with:

   * a random rigid transform (rotation+translation), and
   * independent Gaussian noise. ([rcastoragev2.blob.core.windows.net][1])
3. Sample noise level (\hat t) from their training distribution
   [
   \hat t \sim \sigma_{data}\cdot \exp(-1.2 + 1.5\cdot \mathcal N(0,1))
   ]
   with (\sigma_{data}=16). ([rcastoragev2.blob.core.windows.net][1])
4. Noise the coordinates (VE-style):
   [
   x_{\text{noisy}} = \text{Augment}(x_{GT}) + \hat t,\epsilon,\quad \epsilon\sim \mathcal N(0,I)
   ]
5. **One forward pass** through the diffusion module:
   [
   x_{\text{denoised}} = f_\theta(x_{\text{noisy}}, \hat t, \text{trunk cond})
   ]
6. Compute diffusion loss:

   * **Weighted rigid alignment** of (x_{GT}) onto (x_{\text{denoised}}) (then MSE) ([rcastoragev2.blob.core.windows.net][1])
   * plus **smooth-lDDT** (and sometimes a bond-length term during finetuning) ([rcastoragev2.blob.core.windows.net][1])
   * all wrapped in their scalar weighting factor:
     [
     L_{\text{diffusion}} = \frac{\hat t^2+\sigma_{data}^2}{(\hat t+\sigma_{data})^2}\cdot(\text{MSE}+ \alpha_{bond}L_{bond}) + L_{\text{smooth-lddt}}
     ]
     ([rcastoragev2.blob.core.windows.net][1])

✅ **Backprop happens through this single denoiser call** (and through whatever parts of the network you’ve wired to provide conditioning), *not through an unrolled sampler*.

---

## Training-time “mini diffusion rollout” (the 20-step thing)

This is **not** how they train the diffusion loss. It’s for “heads that require predicted coordinates”.

AF3 explicitly says: at training time they do a **short rollout from pure noise with 20 steps**, and **no gradients are applied** to this mini-rollout. ([rcastoragev2.blob.core.windows.net][1])

So the pattern is:

1. Run trunk → features.
2. **`with no_grad()`**: run a **20-step sampler** (starting from noise) to get a plausible coordinate guess (x_{\text{aux}}).
3. Feed (x_{\text{aux}}) into coordinate-dependent heads (confidence, etc.), compute those head losses.
4. **Backprop those head losses normally** (into trunk/head parameters), but **the coordinates (x_{\text{aux}}) are treated as constants** (the diffusion module + the unrolled 20 steps do not get gradients from these heads).

So: **you do run the model multiple times for the rollout, but you do *not* do a backward pass through those repeated diffusion steps** in AF3.

---

## Inference (actual sampling)

At inference they do the full reverse diffusion **iteratively** using a fixed **200-step** noise schedule defined by their equation (a Karras/EDM-style schedule over (t\in[0,1]) with step size (1/200)). ([rcastoragev2.blob.core.windows.net][1])
And they use the Euler-style update you were discussing (plus optional “churn” noise), i.e. repeated denoise → compute direction → step. ([rcastoragev2.blob.core.windows.net][1])

---

### The punchline

* **Diffusion loss training:** single-step denoising at random (\hat t) → loss on (x_{\text{denoised}}) → backprop.
* **Rollout during training:** 20-step sampler for auxiliary coordinate inputs → **no-grad** through the rollout.
* **Inference:** 200-step sampler → no backprop.

If you tell me how you currently implement “rollout training” (are you trying to backprop through the unrolled chain?), I can suggest the closest AF3-like variant and where to detach to match their setup.

[1]: https://rcastoragev2.blob.core.windows.net/9e6c827f9d7f88eac8af0797432cefc7/41586_2024_7487_MOESM1_ESM.pdf "Accurate structure prediction of biomolecular interactions with AlphaFold 3"



Even though AF3 talks about **continuous** noise level (\hat t), at inference it still generates structures by taking a **discrete sequence of 200 noise levels** (sigmas) and doing an Euler-style update at each one.

### 1) “Continuous (t)” → a discrete sigma schedule

They define (\hat t) (noise std / “sigma”) from a continuous (t \in [0,1]) via: ([rcastoragev2.blob.core.windows.net][1])

[
\hat t(t)= \sigma_{\text{data}}\cdot\left(s_{\max}^{1/p} + t,(s_{\min}^{1/p}-s_{\max}^{1/p})\right)^p
]
with (\sigma_{\text{data}}=16,; s_{\max}=160,; s_{\min}=4\cdot 10^{-4},; p=7), and (t) sampled **uniformly with step size (1/200)**. ([rcastoragev2.blob.core.windows.net][1])

So in practice you get a list ([c_0, c_1, \dots, c_T]) of **sigmas** (noise levels), from high → low.

### 2) Initialization from pure noise

They start coordinates (all atoms across the whole complex: proteins, nucleic acids, ligands, ions…) from a Gaussian at the highest noise level: ([rcastoragev2.blob.core.windows.net][2])

[
x_l \sim c_0 \cdot \mathcal N(0, I_3)
]

### 3) The per-step “interpolation” / update (Algorithm 18)

For each next noise level (c_\tau), they do: ([rcastoragev2.blob.core.windows.net][2])

**(a) Random rigid augmentation every step**

* mean-center, random rotation, random translation. ([rcastoragev2.blob.core.windows.net][2])

**(b) Optional “churn” noise injection**
They compute (\gamma) (0.8 if (c_\tau > \gamma_{\min}) else 0), then:
[
\hat t = c_{\tau-1}(\gamma+1)
]
Add extra noise:
[
\xi_l = \lambda\sqrt{\hat t^2 - c_{\tau-1}^2};\mathcal N(0,I_3),\quad
x^{noisy}_l = x_l + \xi_l
]
((\lambda \approx 1.003) in their pseudocode). ([rcastoragev2.blob.core.windows.net][2])

**(c) Denoise once at (\hat t)**
[
x^{denoised} = f_\theta(x^{noisy}, \hat t, \text{conditioning})
]
([rcastoragev2.blob.core.windows.net][2])

Inside the diffusion module they use EDM-style preconditioning; the output coords are a particular combination of the noisy coords and a learned update term (their Algorithm 20, line 8). ([rcastoragev2.blob.core.windows.net][2])

**(d) Euler step in “sigma space”**
They form a direction:
[
\delta_l = \frac{x_l - x^{denoised}*l}{\hat t}
]
and step size:
[
dt = c*\tau - \hat t
]
then update:
[
x_l \leftarrow x^{noisy}_l + \eta , dt , \delta_l
]
with (\eta \approx 1.5). ([rcastoragev2.blob.core.windows.net][2])

That last line is the “interpolation” you’re asking about: it moves the current sample toward the denoiser’s prediction by an amount determined by how far you’re moving in sigma.

### 4) Intuition: it’s a blend between “current noisy” and “denoised”

If you ignore churn ((\gamma=0\Rightarrow \hat t=c_{\tau-1}), so (x^{noisy}\approx x)) and take (\eta=1), you can rearrange the update into a literal convex-ish blend:

[
x_{\tau} \approx \frac{c_\tau}{c_{\tau-1}}x_{\tau-1} + \left(1-\frac{c_\tau}{c_{\tau-1}}\right)x^{denoised}
]

So: **start at pure noise → repeatedly denoise at current sigma → mix/step toward the denoised structure → decrease sigma → repeat**.

### Where your “realignment each step” plugs in

Because AF3 does random rigid transforms and the denoiser isn’t equivariant, (x^{denoised}) can come out in a different pose than the current (x). If so, the subtraction (x-;x^{denoised}) (and therefore the blend above) becomes nonsense. Boltz-style fixes do **Kabsch-align (x^{denoised}) into the frame of (x) (or (x^{noisy})) before computing (\delta)**.

If you show me your sampler loop, I can tell you the best choice in *your* code: align-to-(x) vs align-to-(x^{noisy}) (AF3 uses both in the same step, so it’s a small but real detail).

[1]: https://rcastoragev2.blob.core.windows.net/9e6c827f9d7f88eac8af0797432cefc7/41586_2024_7487_MOESM1_ESM.pdf "Accurate structure prediction of biomolecular interactions with AlphaFold 3"
[2]: https://rcastoragev2.blob.core.windows.net/9e6c827f9d7f88eac8af0797432cefc7/41586_2024_7487_MOESM1_ESM.pdf?utm_source=chatgpt.com "Accurate structure prediction of biomolecular interactions with ..."


Yeah — “I can overfit training but sampling drifts” is almost always a **train–sample mismatch** or an **integration / gauge** problem, not “model can’t represent the data”.

Here are the main **failure modes** and the best-known **drift-limiting strategies** for AF3/EDM-ish coordinate diffusion.

---

## What “drift” usually is in structure diffusion

You’re iterating
[
x \leftarrow x + \Delta\sigma \cdot \delta(x,\sigma)
]
so any small bias in (\delta) accumulates. Drift shows up as:

* slow global **translation/rotation wandering** (gauge drift)
* **shape warping** because the denoiser output is in a different frame than the current sample
* gradual **blow-up** at low (\sigma) (stiff dynamics / too-large steps)
* collapse / weird shrinkage (bad scaling / wrong preconditioning)

---

## 1) Gauge / frame mismatch (big one for AF3-like non-equivariant denoisers)

**Symptom:** even if denoiser predicts “correct” structures up to rigid motion, the iterative updates produce garbage.

**Fixes:**

* **Per-step rigid alignment (Kabsch) before computing the update direction.**
  Align (\hat x_0) to the current (x) (or (x^{noisy}) if you inject churn) then compute (\delta \propto (x-\hat x_0^{aligned})/\sigma).
  This is *the* classic way to stop pose inconsistency from accumulating.
* **Remove random roto-translation at inference** as a diagnostic.
  If drift disappears, your denoiser isn’t robust enough to the augmentation, or you need the per-step alignment.
* **Always re-center** (subtract centroid) each step.
  Even with alignment, re-centering kills pure translation drift cheaply.

---

## 2) Too-large step size / wrong solver (numerical integration error)

Even with the same 200 steps, you can effectively be taking steps that are “too big” near low noise.

**Fixes (high impact):**

* Switch from **Euler** to **Heun / predictor–corrector** (2nd order).
  Heun often dramatically reduces drift for the same number of function evals (2 per step).
* Use a **DPM-Solver / EDM 2nd-order** style sampler if you have a clean (\hat x_0) parameterization.
* Reduce your **effective step**: multiply update by a factor (<1) (your (\eta)).
  Many implementations quietly need (\eta \in [0.5, 1.2]) depending on scaling. If you copied AF3-ish constants but your coordinate scaling differs, (\eta) becomes wrong.

**Quick diagnostic:** run with 2× or 4× more steps **and** Euler.

* If drift drops a lot → it’s mostly integration error / stiffness.
* If drift persists → it’s more likely frame mismatch or model calibration.

---

## 3) Noise schedule mismatch (training distribution ≠ inference schedule)

You can overfit single-step denoising at sampled (\sigma), but if inference walks through (\sigma) regions the model didn’t train on (or at different density), error accumulates.

**Fixes:**

* Match the **sigma distribution** in training to the **inference schedule density**.
  If inference spends many steps at low (\sigma), ensure training samples low (\sigma) frequently enough.
* Use a **Karras/EDM schedule** (more resolution at low noise) rather than linear in (\sigma).
* Verify **units/scales**: angstroms vs nm, centering, per-atom weighting — these change what “sigma=1” means.

---

## 4) Parameterization / preconditioning issues (x₀ vs ε vs v, EDM scaling)

In coordinate diffusion, predicting **x₀** is usually stable — but only if your conversion to the step direction matches your forward process.

**Common gotchas:**

* Using (\delta=(x-\hat x_0)/\sigma) while your forward process isn’t actually (x = x_0 + \sigma \epsilon).
* Forgetting EDM-style preconditioning (or mixing VP-style formulas with VE-style noise).
* Feeding (\sigma) incorrectly (log σ vs σ, normalized time vs raw σ).

**Fix:** write down your exact forward noising equation and derive the exact sampler update you intend (EDM VE vs VP). Then assert it in code.

---

## 5) Stochasticity (“churn”) makes drift worse until everything else is correct

If you add extra noise at steps (EDM churn), it can explode drift if the denoiser isn’t well-calibrated.

**Fixes:**

* Turn churn **off** (γ=0) until you get stable deterministic sampling.
* If you need it later, keep it tiny and only at high σ.

---

## 6) Training that overfits but doesn’t “compose” across steps

Single-step denoising loss doesn’t guarantee multi-step rollout stability.

**Ways to make it compose better:**

* **Self-conditioning**: feed previous (\hat x_0) (or a stop-grad version) back into the denoiser input.
* Add a small “consistency across nearby sigmas” loss: sample (\sigma_1>\sigma_2), denoise both, penalize disagreements after mapping to same frame.
* Use **EMA weights** at inference (massively helps stability in diffusion models).

---

# A tight debugging playbook (do these in order)

1. **No churn, re-center every step, fixed frame** (no random rotations at inference).
2. Add **Kabsch-align (\hat x_0) to current (x)** before computing update.
3. Tune **step scale (\eta)** downward until stable.
4. Switch to **Heun** (predictor–corrector).
5. Only then re-introduce random augmentation / churn if you need it.

---

If you paste:

* your forward noising equation,
* your sampler update (one step),
* whether you random-rotate/translate each step,
* and whether your denoiser outputs x₀ or a residual,

…I can point to the *most likely* drift source in your loop and suggest the smallest code change that typically fixes it.


Heun is a **2nd-order ODE solver** (a.k.a. *improved Euler* / *explicit trapezoid*). In diffusion sampling, it’s a simple **predictor–corrector** step that usually reduces “drift” a lot compared to plain Euler, because it uses **two denoiser evaluations** to approximate the average slope over the step.

### In one line

* **Euler:** take one denoiser call, step using the slope at the start.
* **Heun:** take one denoiser call to predict the next point, take a second denoiser call there, then step using the **average** of the two slopes.

---

## How it looks in AF3/EDM-style sampling (x₀-prediction)

Assume your sampler is in “sigma space” and your model returns a denoised structure (\hat x_0 = f(x,\sigma)). Define a “slope” (direction):

[
d(x,\sigma) = \frac{x - \hat x_0}{\sigma}
]

Let current state be ((x, \sigma)) and next noise level be (\sigma') (smaller). Let (\Delta\sigma = \sigma' - \sigma) (negative).

### Euler (what you’re probably doing)

1. (\hat x_0 = f(x,\sigma))
2. (d = (x - \hat x_0)/\sigma)
3. (x \leftarrow x + \Delta\sigma , d)

### Heun (improved Euler)

1. **Slope at start**

   * (\hat x_0 = f(x,\sigma))
   * (d = (x - \hat x_0)/\sigma)

2. **Predict**

   * (x_{\text{pred}} = x + \Delta\sigma , d)

3. **Slope at predicted point**

   * (\hat x_0^{pred} = f(x_{\text{pred}},\sigma'))
   * (d_{\text{pred}} = (x_{\text{pred}} - \hat x_0^{pred})/\sigma')

4. **Correct (average slopes)**

   * (x \leftarrow x + \Delta\sigma \cdot \frac{d + d_{\text{pred}}}{2})

That’s it.

---

## Why it helps with drift

Euler can overshoot or accumulate bias because it assumes the direction (d) is constant over the step. In diffusion, (d) changes a lot as (\sigma) shrinks (especially at low noise), so Euler error compounds. Heun uses the slope at both ends, so it’s much closer to the “true” trajectory.

**Tradeoff:** ~2× compute (two denoiser calls per step), often letting you use **fewer steps** for similar or better quality.

---

## Practical notes for your AF3-like setup

* If you do **Kabsch alignment per step**, apply it consistently when computing (\hat x_0) both at ((x,\sigma)) and at ((x_{\text{pred}},\sigma')).
* Start by testing Heun with **no churn** (no extra noise injection). Add churn back only after it’s stable.

If you paste your current sampler update (5–15 lines), I’ll rewrite it into a Heun version with the right “align-to-x vs align-to-x_noisy” choice for your exact loop.

---

# Technical Requirements: Fixing TinyFold Diffusion

> Based on the above conversation with ChatGPT about AF3/Boltz-1 diffusion.
> 
> **Core Problem**: Model can overfit training (good single-step denoising) but sampling "drifts" and produces garbage structures.
>
> **Root Cause**: Frame inconsistency — the denoiser outputs x₀ in a different rigid frame than the current x_t, causing interpolation/updates to warp the structure.

---

## Summary of Current TinyFold Implementation

### Forward Noising (Training)

| Aspect | Current Implementation | AF3 Approach | Gap |
|--------|----------------------|--------------|-----|
| **Noise process** | VP-style (sqrt_alpha_bar blending) | VE-style (additive: x = x₀ + σ·ε) | ⚠️ Different |
| **Schedule** | Discrete (50 steps), linear or cosine alpha_bar | Continuous σ, Karras/EDM schedule | ⚠️ Different |
| **Timestep sampling** | Uniform over [0,T-1] | Log-normal: σ_data·exp(-1.2 + 1.5·N(0,1)) | ⚠️ Different |
| **Augmentation** | None during training | Random rotation + translation (48 copies) | ❌ **Missing** |
| **Loss** | MSE after Kabsch align | Weighted MSE + smooth-lDDT + bond term | Partial |
| **Loss weighting** | Uniform | (σ² + σ_data²)/(σ + σ_data)² | ❌ **Missing** |

**Relevant Code:**
```python
# scripts/models/diffusion.py (GaussianNoise.add_noise)
x_t = sqrt_ab * x0 + sqrt_one_minus_ab * noise  # VP-style blending

# scripts/train.py (training loop)
t = torch.randint(0, noiser.T, (batch_size,))  # Uniform sampling
x_input, target = noiser.add_noise(batch['coords'], t, ...)
pred = model(x_input, ...)
loss = compute_loss(pred, batch['coords'], batch['mask'], use_kabsch=True)
```

### Sampling (Inference)

| Aspect | Current Implementation | AF3 Approach | Gap |
|--------|----------------------|--------------|-----|
| **Initialization** | Random N(0,1) | σ_max · N(0,1) | Minor |
| **Steps** | 50 (T) | 200 | Can tune |
| **Update rule** | DDPM posterior (mean + variance) | EDM Euler: x += dt·(x-x_den)/σ | ⚠️ Different |
| **Per-step centering** | ❌ None | ✅ Re-center every step | ❌ **Missing** |
| **Per-step Kabsch** | ❌ None | ✅ Align x₀_pred to x_t (Boltz-1) | ❌ **CRITICAL MISSING** |
| **Random augmentation** | ❌ None at inference | ✅ Every step (AF3) | ⚠️ Optional |
| **Churn** | ❌ None | Optional (γ=0.8 at high σ) | Later |
| **Solver** | Euler (DDPM) | Euler or Heun | Upgrade later |

**Relevant Code:**
```python
# scripts/train.py (ddpm_sample function, lines 59-164)
for t in t_range:
    x0_pred = model(x, ...)
    x0_pred = torch.clamp(x0_pred, -clamp_val, clamp_val)
    
    # DDPM reverse step - NO ALIGNMENT HERE!
    if t > 0:
        coef1 = sqrt(ab_prev) * beta / (1 - ab_t)
        coef2 = sqrt(alpha) * (1 - ab_prev) / (1 - ab_t)
        mean = coef1 * x0_pred + coef2 * x
        var = beta * (1 - ab_prev) / (1 - ab_t)
        x = mean + sqrt(var) * randn_like(x)
    else:
        x = x0_pred
```

---

## REQUIREMENTS (Ordered by Priority)

### REQ-1: Per-Step Kabsch Alignment in Sampling (CRITICAL)

**Problem**: `x0_pred` may be in a different rigid frame than `x_t`. The DDPM update computes `coef1 * x0_pred + coef2 * x`, which produces **warped garbage** if frames don't match.

**Solution**: Before computing the update direction, Kabsch-align `x0_pred` to `x_t`.

**Implementation**:

```python
# In ddpm_sample() - ADD THIS BEFORE the update step
def ddpm_sample(...):
    ...
    for t in t_range:
        x0_pred = model(x, ...)
        x0_pred = torch.clamp(x0_pred, -clamp_val, clamp_val)
        
        # NEW: Kabsch-align x0_pred to current x
        x0_pred_aligned = kabsch_align_to_target(x0_pred, x, mask)
        
        if t > 0:
            # Use aligned prediction in update
            mean = coef1 * x0_pred_aligned + coef2 * x
            ...

def kabsch_align_to_target(pred, target, mask=None):
    """Align pred to target's frame (returns aligned pred)."""
    # Compute optimal R, T to minimize ||target - (R @ pred + T)||
    # Apply R, T to pred
    # Return aligned pred
```

**Files to modify**:
- `scripts/train.py` - `ddpm_sample()` function
- `scripts/train_resfold.py` - `sample_centroids()` function
- `scripts/models/resfold_pipeline.py` - `sample()` method

**Test**: After implementing, sampling should produce coherent structures instead of warped blobs.

---

### REQ-2: Per-Step Re-Centering

**Problem**: Even with Kabsch, pure translation drift can accumulate.

**Solution**: Re-center (subtract centroid) after each update step.

**Implementation**:

```python
# In ddpm_sample(), AFTER the update:
x = mean + sqrt(var) * randn_like(x)
x = x - x.mean(dim=1, keepdim=True)  # Re-center
```

**Files to modify**: Same as REQ-1.

---

### REQ-3: VE-Style Forward Process (Medium Priority)

**Problem**: TinyFold uses VP-style (variance-preserving) noising, but AF3 uses VE-style (variance-exploding). This affects the relationship between noise level and update direction.

**Current**:
```
x_t = sqrt(α_bar) * x0 + sqrt(1 - α_bar) * ε   # VP
```

**AF3**:
```
x_noisy = x0 + σ * ε                            # VE (additive)
```

**Solution**: Add a VE-style noise process option.

**Implementation**:

```python
# scripts/models/diffusion.py - new class
class VENoiser:
    """Variance-Exploding noise (AF3-style)."""
    
    def __init__(self, sigma_data=16.0, sigma_min=0.0004, sigma_max=160.0, n_steps=200):
        self.sigma_data = sigma_data
        # Build Karras/EDM schedule
        self.sigmas = self._build_karras_schedule(sigma_min, sigma_max, n_steps)
    
    def add_noise(self, x0, sigma):
        """Simple additive: x_noisy = x0 + sigma * noise"""
        noise = torch.randn_like(x0)
        x_noisy = x0 + sigma * noise
        return x_noisy, noise
    
    def sample_sigma(self, batch_size, device):
        """AF3-style log-normal sigma sampling for training."""
        log_sigma = -1.2 + 1.5 * torch.randn(batch_size, device=device)
        return self.sigma_data * torch.exp(log_sigma)
```

---

### REQ-4: EDM-Style Sampler Update

**Problem**: DDPM update formula is designed for VP process. With VE, use simpler EDM update.

**AF3 Update**:
```
δ = (x - x_denoised) / σ
x_next = x + dt * δ        where dt = σ_next - σ
```

**Simplified (no churn)**:
```
x_next = (σ_next/σ) * x + (1 - σ_next/σ) * x_denoised
```

**Implementation**:

```python
def edm_sample(model, ..., noiser, mask=None, clamp_val=3.0):
    """EDM-style sampling with Kabsch alignment."""
    sigmas = noiser.sigmas  # [σ_0 (high) -> σ_T (low)]
    
    # Initialize at high noise
    x = sigmas[0] * torch.randn(B, N, 3, device=device)
    
    for i in range(len(sigmas) - 1):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]
        
        # Denoise
        x0_pred = model(x, ..., sigma, ...)
        x0_pred = torch.clamp(x0_pred, -clamp_val, clamp_val)
        
        # Kabsch-align to current frame
        x0_pred = kabsch_align_to_target(x0_pred, x, mask)
        
        # EDM update
        dt = sigma_next - sigma  # negative
        delta = (x - x0_pred) / sigma
        x = x + dt * delta
        
        # Re-center
        x = x - x.mean(dim=1, keepdim=True)
    
    return x
```

---

### REQ-5: Loss Weighting Factor

**Problem**: Uniform loss across all noise levels. AF3 uses weighting that emphasizes mid-range noise.

**AF3 Weighting**:
```
w(σ) = (σ² + σ_data²) / (σ + σ_data)²
```

**Implementation**:

```python
# In training loss computation
sigma_data = 16.0  # or your coordinate std
weight = (sigma**2 + sigma_data**2) / (sigma + sigma_data)**2
loss = weight * mse_loss
```

---

### REQ-6: Heun Sampler (2nd Order)

**Problem**: Euler accumulates integration error, especially at low noise.

**Heun Algorithm**:
```
1. d1 = (x - f(x, σ)) / σ
2. x_pred = x + dt * d1
3. d2 = (x_pred - f(x_pred, σ_next)) / σ_next
4. x = x + dt * (d1 + d2) / 2
```

**Implementation**:

```python
def heun_sample(model, ..., noiser, mask=None):
    """Heun (2nd order) sampler with Kabsch alignment."""
    sigmas = noiser.sigmas
    x = sigmas[0] * torch.randn(B, N, 3, device=device)
    
    for i in range(len(sigmas) - 1):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]
        dt = sigma_next - sigma
        
        # Slope at start
        x0_pred = model(x, ..., sigma, ...)
        x0_pred = kabsch_align_to_target(x0_pred, x, mask)
        d1 = (x - x0_pred) / sigma
        
        # Predict
        x_pred = x + dt * d1
        
        if sigma_next > 0:  # Not at final step
            # Slope at predicted point
            x0_pred_next = model(x_pred, ..., sigma_next, ...)
            x0_pred_next = kabsch_align_to_target(x0_pred_next, x_pred, mask)
            d2 = (x_pred - x0_pred_next) / sigma_next
            
            # Average
            x = x + dt * (d1 + d2) / 2
        else:
            x = x + dt * d1
        
        # Re-center
        x = x - x.mean(dim=1, keepdim=True)
    
    return x
```

---

### REQ-7: Random Rigid Augmentation During Training

**Problem**: Model not robust to arbitrary coordinate frames.

**AF3 Approach**: Apply random rotation + translation to EACH training sample (48 copies).

**Implementation**:

```python
def random_rigid_augment(x, mask=None):
    """Apply random rotation and translation."""
    B, N, _ = x.shape
    device = x.device
    
    # Random rotation (per sample in batch)
    R = random_rotation_matrix(B, device)  # [B, 3, 3]
    
    # Random translation
    T = torch.randn(B, 1, 3, device=device) * 10.0  # Scale appropriately
    
    # Apply
    x_aug = torch.bmm(x, R.transpose(1, 2)) + T
    
    return x_aug

# In training loop:
x0 = random_rigid_augment(batch['coords'])
x_t, noise = noiser.add_noise(x0, t, ...)
```

---

### REQ-8: Auxiliary Loss Terms (Lower Priority)

**AF3 Losses**:
1. **smooth-lDDT**: Differentiable local distance difference test
2. **Bond length loss**: Already have in `geometry_losses.py`

**Current Status**: Bond/geometry losses exist but may not be applied to diffusion output during training.

---

## Implementation Order

| Phase | Requirements | Impact | Effort |
|-------|--------------|--------|--------|
| **1. Critical Fix** | REQ-1 (Kabsch sampling), REQ-2 (re-center) | Should fix drift immediately | 2h |
| **2. Better Sampling** | REQ-6 (Heun) | Reduces integration error | 2h |
| **3. Proper Process** | REQ-3 (VE noise), REQ-4 (EDM update) | Aligns with AF3 | 4h |
| **4. Training Improvements** | REQ-5 (weighting), REQ-7 (augmentation) | Better generalization | 3h |
| **5. Optional** | REQ-8 (auxiliary losses), churn | Refinements | 4h+ |

---

## Questions for Clarification

Before implementing, please confirm:

1. **Coordinate units**: Are your normalized coordinates in units where std ≈ 1? (This affects sigma_data choice)

2. **ResFold vs AF3-style**: Should we fix both models, or focus on one first?

3. **Step count**: Current T=50. Should we increase to 100-200 for better sampling?

4. **Kabsch weighting**: For alignment, should all atoms have equal weight, or weight by backbone/CA only?

5. **Testing approach**: Do you have specific proteins where drift is most visible? We should test the fix on those.

---

## Files to Modify

| File | Changes |
|------|---------|
| `scripts/models/diffusion.py` | Add VENoiser, Karras schedule, kabsch_align_to_target utility |
| `scripts/train.py` | Update ddpm_sample with Kabsch + re-centering, add edm_sample/heun_sample |
| `scripts/train_resfold.py` | Update sample_centroids with Kabsch + re-centering |
| `scripts/models/resfold_pipeline.py` | Update sample() method |
| `scripts/models/af3_style.py` | Verify model outputs are in consistent frame |

---

## Debugging Playbook (From Conversation)

If still seeing drift after REQ-1 and REQ-2:

1. **No churn, no random augmentation at inference** (baseline)
2. **Add Kabsch alignment** → should fix frame mismatch
3. **Tune step scale η downward** → if still drifting, may be too aggressive
4. **Switch to Heun** → reduces integration error
5. **Increase steps** (2× or 4×) → diagnostic: if this helps, it's integration error
6. **Only then** re-introduce augmentation/churn

Before I start implementing, please answer the clarifying questions at the end of the file:

Are normalized coordinates std ≈ 1?
Focus on ResFold or AF3-style first?
Increase steps from 50 to 100-200?
Kabsch weighting: all atoms or backbone only?
Specific test proteins with visible drift?
diffusion_todo.