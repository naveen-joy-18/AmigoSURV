# AmiGO-Surv — Code-Faithful Pseudo-code

This document restates the AmiGO-Surv framework at the level of the shipped
implementation (`src/amigo_surv_training.py`, `src/validation_pipeline.py`).
It supersedes the pseudo-code in the manuscript where the two deviate
(see `pseudocode_changes.md`).

---

## 1. Notation

| Symbol | Meaning |
|--------|---------|
| n | number of patients |
| p | number of raw triplet features |
| d | number of PCA components retained (≤ 64) |
| z_i ∈ ℝ^d | PCA-compressed feature vector of patient i |
| t_i, δ_i | observed survival time (months), event indicator (δ=1 death) |
| r̂_i = f_θ(z_i) | predicted log-hazard risk score |
| ⊙ | Hadamard (element-wise) product |
| τ | sparse-gate temperature, τ = 0.8 |
| k | gate bottleneck ratio, k = 4 |

Training set D = { (z_i, t_i, δ_i) }.

---

## 2. Data loading & triplet prior

For the cohort of interest (e.g. UCEC), the framework loads:

- the **STAD-derived H–M–T triplet tables** (`data/triplets/Table 1`, `Table 2`)
  — oncogenic sheets `*_H(+)_M(+)_T(-)_AdjP`, tumour-suppressor sheets
  `*_H(-)_M(-)_T(+)_AdjP`;
- per-patient **miRNA** and **mRNA** expression matrices;
- clinical records with overall-survival (months) and status, converted to
  δ ∈ {0, 1} (deceased/1 ⇒ event).

Only samples present in all three sources are kept, and subjects with
`OS_months ≤ 0` are discarded.

---

## 3. H–M–T triplet feature construction

Let T_c^{onco} and T_c^{ts} be the oncogenic and tumour-suppressor triplets.

**Layer 1 — expression features.** Per patient i collect

- mRNA of every host and target gene: x_i^{mRNA} = [e_{ig}]_{g ∈ G_c}
- miRNA of every triplet miRNA:        x_i^{miRNA} = [e_{im}]_{m ∈ M_c}

**Layer 2 — correlation-weighted bilinear interactions.** For each triplet:

```
oncogenic (H, M, T):
    φ_i^HT  = e_iH · e_iT · r_HM
    φ_i^MT  = e_iM · e_iT · r_MT
tumour-suppressor (H, M, T):
    φ_i^HT = e_iH · e_iT · r_HT
```

where r_HM, r_MT, r_HT are the pre-computed Pearson correlations from the
triplet annotation step.

**Complete raw vector**

```
x_i^raw = [ x_i^mRNA  ‖  x_i^miRNA  ‖  {φ_i^HT, φ_i^MT}_{(H,M,T)∈T_c} ]
```

**Variance filter:** drop columns with Var(x^raw) ≤ ε_v = 1e-6, producing the
p-dimensional matrix X ∈ ℝ^{n×p}.

---

## 4. Preprocessing

```
1. log-modulus transform:   x̃_ij = log(1 + |x^filt_ij|)
2. z-score standardisation: x̂_ij = (x̃_ij − μ_j) / (σ_j + δ),   δ = 1e-8
                             μ_j, σ_j fitted on the training fold ONLY
3. PCA:                     z_i = Uᵀ x̂_i ∈ ℝ^d
                             d = min(64, p_filt, n − 1)
                             U fitted on the training fold ONLY
```

For the pan-cancer validation pipeline d is chosen as the smallest number of
components with ≥ 90% explained variance (bounded by 32); for the single-cohort
pipeline d = min(64, p, n−1) is retained.

---

## 5. AmiGO-Surv architecture

### 5.1 Sparse attention gate (sigmoid, temperature-scaled)

```
h⁽¹⁾ = SiLU( LayerNorm( W₁ z + b₁ ) ),      W₁ ∈ ℝ^{(d/k)×d}
h⁽²⁾ = W₂ h⁽¹⁾ + b₂,                         W₂ ∈ ℝ^{d×(d/k)}
g    = σ( h⁽²⁾ / τ ) ∈ (0,1)^d
z_g  = z ⊙ g
```

Implementation note: the gate is a 2-layer MLP
`LayerNorm → Linear(d, d/4) → SiLU → Linear(d/4, d)`, with `bottleneck d/4`
(padded to ≥ 8) and sigmoid activation scaled by τ = 0.8. Because the gate is
sigmoid-based (not softmax), multiple features can be strongly selected
**simultaneously**.

### 5.2 Cross-attention regulatory refinement

The gated vector is used to *query* the raw (ungated) input across feature
tokens:

```
q = MeanPool( W_Q z_g )          # summary query token
k, v = W_K z, W_V z              # keys/values from raw features
z_att = MultiHeadAttention(q, k, v;  4 heads, d_model = d)
z̃     = z_att + z_g              # residual gated path
```

### 5.3 Survival encoder + Cox head

```
encoder:  h⁽ˡ⁾ = SiLU( LayerNorm( W⁽ˡ⁾ h⁽ˡ⁻¹⁾ + b⁽ˡ⁾ ) ),   l = 1..3
          hidden dims (256, 128, 64), dropout 0.25 after each SiLU
residual: h_enc = h⁽³⁾ + W_proj z̃

Cox head: r̂ = f_θ(z) = w₂ᵀ SiLU( W_head₁ h_enc + b_head₁ ) + b₂
```

### 5.4 Weight initialisation

All linear layers use Kaiming normal init; all biases are zeroed.

---

## 6. Losses

### 6.1 Cox partial likelihood (Breslow approximation)

Minibatch sorted by descending survival time:

```
L_Cox = − (1/|E|) Σ_{i∈E} [ r̂_i − log Σ_{j : t_j ≥ t_i} exp(r̂_j) ]
```

computed stably with `torch.logcumsumexp`.

### 6.2 Pairwise concordance ranking loss (auxiliary)

```
L_rank = (1/|E|) Σ_{i∈E} (1/|C_i|) Σ_{j∈C_i} max(0, r̂_j − r̂_i + ν )
C_i = { j : t_j > t_i },   margin ν = 0.1
```

### 6.3 Combined AmiGO-Surv objective

```
L = L_Cox + γ · L_rank,   γ = 0.3
```

### 6.4 Baseline losses

| Model | Loss |
|-------|------|
| DeepCox-MLP | L_Cox |
| VAE-Survival | L_Cox + 0.5·‖z−ẑ‖₂² + 0.1·D_KL(N(μ,σ²)‖N(0,1)) |
| ResCox-Net / TransSurv | L_Cox |

---

## 7. Baseline architectures

- **DeepCox-MLP:** FC 128→64→32, BatchNorm + ReLU + dropout 0.3.
- **VAE-Survival:** encoder → latent z = μ + σ⊙ε (ε∼N(0,I)), decoder
  reconstructs input, Cox head on μ (eval) / sampled z (train).
- **ResCox-Net:** projection + 2 pre-activation residual blocks
  (`LayerNorm → Linear → GELU → Dropout`, shortcut projection when dims differ).
- **TransSurv:** each PCA component is a token, projected to d_model = 64;
  prepends a trainable [CLS] token; **3** transformer encoder layers
  (4 heads, pre-norm), output = CLS token → Cox head.

---

## 8. Training protocol

- Optimiser: **AdamW** (weight decay 2e-4 for AmiGO-Surv, 1e-4 baselines).
- Scheduler: AmiGO-Surv → **OneCycleLR** (peak 5e-4, warm-up 15%, cosine);
  baselines → **CosineAnnealingLR** (lr 1e-4, T_max = epochs).
- Batch size 32, max 200 epochs, gradient clipping norm ≤ 1.
- Early stopping on validation C-index (patience 30 AmiGO-Surv, 20 baselines)
  with best-state restoration.
- **AmiGO-Surv: 4 random restarts per fold**; best validation C-index kept.

---

## 9. Evaluation

- **Harrell's C-index** (mean ± SD over folds).
- **Kaplan–Meier** for high- vs low-risk groups split at the median predicted
  log-hazard; separation tested with the **log-rank test**.
- **Tertile calibration:** risk tertiles T1/T2/T3 → ordered KM curves.
- **Interpretability:** mean gate weights ḡ_j = (1/n)Σ_i g_ij(z_i), and
  **permutation importance** ΔC-index after shuffling each PCA component.

---

## 10. TAS (Triplet Activity Score) — validation pipeline

For each triplet in the validation data, per patient

```
TAS = z(Host) + z(miRNA) − z(Target)        # z = standardised expression
```

Patients split at the TAS median promote the "regulatory axis ON vs OFF"
comparison used for per-triplet Kaplan–Meier figures.