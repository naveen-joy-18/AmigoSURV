# Manuscript Pseudo-code ↔ Implementation Reconciliation

This note lists every place where the manuscript pseudo-code (Methods section)
differs from the code actually shipped, with the corrected wording to use.

## 1. Sparse attention gate — activation and layer norm

**Manuscript (as written):**

```
h⁽¹⁾ = tanh(W₁ z + b₁),     W₁ ∈ ℝ^{(d/k)×d}
h⁽²⁾ = W₂ h⁽¹⁾ + b₂,        W₂ ∈ ℝ^{d×(d/k)}
g    = σ(h⁽²⁾/τ)
```

**Implementation** (`_SparseGate`, both pipelines):

```
h⁽¹⁾ = SiLU( LayerNorm( W₁ z + b₁ ) )      # ← LayerNorm first, SiLU, not tanh
h⁽²⁾ = W₂ h⁽¹⁾ + b₂
g    = σ( h⁽²⁾ / τ )
```

Bottleneck k = 4 and temperature τ = 0.8 are unchanged. Everything downstream
(`z_g = z ⊙ g`, cross-attention, encoder, Cox head) is identical.

> **Action:** replace `tanh` with `SiLU(LayerNorm(·))` in Eq. (13) of the
> Methods, i.e.
> `h⁽¹⁾ = SiLU(LN(W₁ z + b₁))`. Keep Eqs. (14)–(16) unchanged.

## 2. TransSurv depth

**Manuscript:** "d_model = 64, n_heads = 4, **n_layers = 2**".

**Implementation** (`build_configs` / `build_model_configs` in both pipelines):
`n_layers = 3`. Manuscript runs (Figures, Tables 1–2, Supplementary) were
produced with the 3-layer encoder.

> **Action:** update the Methods §6.4 and Table 1 row for TransSurv from
> `L = 2` to `L = 3` (i.e. 3 transformer encoder layers).

## 3. Model display name

Inside the code the proposed architecture is labelled **`Attention-Net`**;
the manuscript refers to the same model as **AmiGO-Surv**. No behavioural
difference — only a label.

> **Action:** no change required; the README documents the equivalence.

## 4. PCA retention rule (validation pipeline only)

- Single-cohort pipeline: `d = min(64, p_filt, n − 1)` — matches manuscript Eq. (12).
- Pan-cancer validation pipeline: `d` = smallest number of components explaining
  ≥ 90% variance (max 32). This is why per-cohort `d` varies (e.g. 6 in HNSC,
  64 in STAD), which the manuscript §3.7 already describes ("PCA dimensionality
  varied substantially by cohort").

> **Action:** if the Methods should reflect the validation setting, add one
> sentence in §3.2.3: *"For the pan-cancer validation runs, d was set to the
> smallest number of components retaining ≥ 90% of variance (≤ 32)."*

## 5. Single-cohort script cohort label

`src/amigo_surv_training.py` runs on the **UCEC** cohort
(n = 395, 70 events, 17.7%) using the STAD-derived H–M–T triplet sheets as the
biological prior — reproducing the UCEC row of manuscript Table 1 (AmiGO-Surv
C-index 0.679, KM p = 0.705). Plot/report titles are labelled UCEC in the
shipped script.

> **Action:** none for the manuscript text; note for reviewers that the UCEC
> analysis uses the STAD-derived triplet prior, consistent with the Methods
> ("triplet counts refer to STAD-derived triplets used as the biological prior").

## 6. File renames (working → repository)

| Working file                 | Repository location                  |
|------------------------------|--------------------------------------|
| `model.py`                   | `src/amigo_surv_training.py`         |
| `Validation pipeline.py`     | `src/validation_pipeline.py`         |
| `rebuild_atlas.py`           | `src/rebuild_atlas.py`               |
| Table 1 / Table 2 xlsx       | `data/triplets/`                     |
| Model Training *.xlsx        | `data/validation/`                   |

Raw expression/clinical matrices are intentionally **not** committed
(TCGA/GDC public data, ~45 MB). The README (§5 Data) documents the exact
expected layout.