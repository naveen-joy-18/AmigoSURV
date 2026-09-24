# AmiGO-Surv
## Attention miRNA–Gene axis Oncology Survival Network

Pan-cancer survival prediction from **Host gene–microRNA–Target gene (H–M–T) triplet** regulatory features across six TCGA cohorts computed with exclusively-biological-prior deep learning. The framework accompanies the manuscript

> **AmiGO-Surv: An Attention-Based miRNA–Gene Axis Oncology Survival Framework**

AmiGO-Surv treats the three-way intronic-miRNA regulatory circuit (host gene → miRNA → silenced target) as a biologically structured feature space, then learns **which triplet axes matter for prognosis** with a sparse attention gate and cross-attention refinement, followed by a Cox partial-likelihood survival head.

---

## 1. Highlights

- **Biological prior** – H–M–T triplets annotated from DESeq2 differential expression, Pearson correlations, and validated miRNA–target (miRTarBase/TargetScan) interactions.
- **Sparse sigmoid gate** – non-competitive feature selection that can simultaneously activate multiple prognostic axes (contrast: softmax attention).
- **Cross-attention refinement + residual encoder** over PCA-compressed triplet features.
- **5 deep survival models** benchmarked under 5-fold stratified cross-validation:
  `AmiGO-Surv` (Attention-Net), `DeepCox-MLP`, `VAE-Survival`, `ResCox-Net`, `TransSurv`.
- **6 TCGA cohorts** – STAD, LUAD, LUSC, KIRC, HNSC, UCEC (2,234 patients).
- **Publication-grade outputs** – SVG/PDF figures + machine-readable TXT reports.

---

## 2. Pipeline

```
                    ┌──────────────────────────────────────────────┐
                    │ 1. H–M–T triplet annotation (biological prior) │
                    │    DESeq2 │ Pearson r │ miRTarBase/TargetScan  │
                    └──────────────────────────────────────────────┘
                                      │
                    ┌──────────────────▼──────────────────────────┐
                    │ 2. Triplet-structured feature engineering    │
                    │    mRNA expr │ miRNA expr │ r-weighted        │
                    │    bilinear interaction terms                │
                    └──────────────────┬──────────────────────────┘
                    ┌──────────────────▼──────────────────────────┐
                    │ 3. Preprocessing                            │
                    │    variance filter ε=1e-6 │ log1p │ z-score  │
                    │    │ PCA (train-fold fit only)              │
                    └──────────────────┬──────────────────────────┘
                    ┌──────────────────▼──────────────────────────┐
                    │ 4. Deep survival training (5-fold CV)       │
                    │    AmiGO-Surv │ DeepCox-MLP │ VAE │ Res │    │
                    │    TransSurv   (Cox partial loss + ranking) │
                    └──────────────────┬──────────────────────────┘
                    ┌──────────────────▼──────────────────────────┐
                    │ 5. Evaluation & interpretability            │
                    │    C-index │ KM + log-rank │ tertile        │
                    │    calibration │ gate weights │ perm. imp.  │
                    └──────────────────────────────────────────────┘
```

## 3. Repository layout

```
AmiGO-Surv/
├── src/
│   ├── amigo_surv_training.py   # single-cohort (UCEC) training + 8 publication figures
│   ├── validation_pipeline.py   # pan-cancer validation: KM, gene-KM atlas, NN pipeline
│   └── rebuild_atlas.py         # rerun gene-KM atlas only (matches reference figure style)
├── data/
│   ├── triplets/                # Table 1 & Table 2 (H-M-T triplet annotation sheets)
│   ├── validation/              # validation triplet lists used by the pipeline
│   └── expression/              # TCGA expression/clinical — NOT shipped (see §5)
├── docs/
│   ├── pseudocode.md            # code-faithful pseudo-code of the full framework
│   └── pseudocode_changes.md    # manuscript pseudo-code ↔ implementation reconciliation
├── requirements.txt
├── LICENSE
└── README.md
```

| Original working file            | Shipped here                       |
|----------------------------------|------------------------------------|
| `model.py`                       | `src/amigo_surv_training.py`       |
| `Validation pipeline.py`         | `src/validation_pipeline.py`       |
| `rebuild_atlas.py`               | `src/rebuild_atlas.py`             |

> **Naming.** Throughout the code `Attention-Net` denotes the proposed **AmiGO-Surv** architecture (sparse gate + cross-attention). It is the same model.

---

## 4. Installation

```bash
git clone https://github.com/<your-account>/AmiGO-Surv.git
cd AmiGO-Surv
python -m venv venv            # optional but recommended
venv\Scripts\activate           # Windows   (Linux/macOS: source venv/bin/activate)
pip install -r requirements.txt
```

Tested with Python 3.13, PyTorch 2.10, lifelines 0.30, scikit-learn 1.8. A standard CPU is sufficient for all analyses (GPU/CUDA optional, ~4× speed-up).

## 5. Data

**Raw TCGA expression/clinical matrices are not committed** (~45 MB; sourced from TCGA via the GDC Portal as FPKM-UQ mRNA and RPM-normalised miRNA counts). The repository ships the biological-prior tables (`data/triplets`) and the validation triplet lists (`data/validation`) used for the manuscript analyses.

To reproduce end-to-end, place per-cohort files here:

```
data/expression/
├── UCEC/  Endometrioid_Cancer - microRNA.xlsx
│          Endometrioid_Cancer - mRNA.xlsx
│          Endometrioid_Cancer_tcga_gdc_clinical_data - microRNA.xlsx
├── HNSC/  Head_and_Neck_Squamous_Cell_Carcinoma - microRNA.xlsx   (+ mRNA, clinical)
├── KIRC/  Kidney_Clear_Cell_Carcinoma_Cancer - microRNA.xlsx       (+ mRNA, clinical)
├── LUAD/  Lung_Adenocarcinoma - microRNA.xlsx                      (+ mRNA, clinical)
├── LUSC/  Lung_Squamous_Cell_Carcinoma - microRNA.xlsx             (+ mRNA, clinical)
└── STAD/  Stomach_Adenocarcinoma- microRNA.xlsx                    (+ mRNA, clinical)
```

If files are missing, the scripts fail fast with a **pre-flight report** listing every missing path — nothing else changes.

## 6. Usage

### 6.1 Single-cohort pipeline (UCEC example figure set)

```bash
python src/amigo_surv_training.py
```

Writes to `outputs/`: `01_data_overview`, `02_training_curves`, `03_cv_results`, `04_kaplan_meier`, `05_risk_distributions`, `06_feature_importance`, `07_summary_dashboard`, `08_calibration_curves` (each `.svg` + `.pdf`) and `report.txt`.

### 6.2 Pan-cancer validation pipeline

```bash
python src/validation_pipeline.py
```

For every cancer performs: per-triplet **TAS** (Triplet Activity Score) Kaplan–Meier, individual gene-level KM (Host | miRNA | Target, ggplot2-style red/teal with number-at-risk), NN cross-validation, and outputs the master 6-cancer gene-KM atlas + pan-cancer comparison to `outputs/validation_v4/`.

### 6.3 Rebuild gene-KM atlas only

```bash
python src/rebuild_atlas.py
```

Reruns the `val_ALL_gene_km_atlas` figure (style-matched to the reference KM figure) without re-training any network.

## 7. Models & hyper-parameters

| Model             | Hidden layers / dims | Dropout | Heads | Gate τ | Peak LR | W-decay | Patience | Restarts |
|-------------------|----------------------|---------|-------|--------|---------|---------|----------|----------|
| **AmiGO-Surv**    | 256→128→64           | 0.25    | 4     | 0.8    | 5e-4    | 2e-4    | 30       | 4        |
| DeepCox-MLP       | 128→64→32            | 0.30    | –     | –      | 1e-4    | 1e-4    | 20       | 1        |
| VAE-Survival      | 32 (latent 16)       | 0.30    | –     | –      | 1e-4    | 1e-4    | 20       | 1        |
| ResCox-Net        | 128→64→32            | 0.30    | –     | –      | 1e-4    | 1e-4    | 20       | 1        |
| TransSurv         | d_model 64, **L = 3**| 0.30    | 4     | –      | 1e-4    | 1e-4    | 20       | 1        |

All models: AdamW, batch 32, max 200 epochs, cosine / one-cycle schedules, gradient clipping (norm ≤ 1), best-state restoration, and seed **42** for all random processes. AmiGO-Surv uses the combined objective `L = L_Cox + 0.3 · L_rank`.

## 8. Reproducibility

- All random processes are seeded with **42** (`random`, `numpy`, `torch`).
- CV is **5-fold stratified on the event indicator**; scaler/PCA are fitted **per training fold** only to prevent leakage.
- See `docs/pseudocode.md` for the exact equations implemented and `docs/pseudocode_changes.md` for the reconciliation of the manuscript pseudo-code with this implementation (esp. the sparse gate activation and the TransSurv depth).

## 9. Citation

If you use this code in your research, please cite:

```bibtex
@article{amigo-surv,
  title  = {AmiGO-Surv: An Attention-Based miRNA-Gene Axis Oncology Survival Framework},
  author = {Deepak and colleagues},
  journal= {Nature Machine Intelligence},
  year   = {2026},
  note   = {Manuscript under review — code available at https://github.com/<your-account>/AmiGO-Surv}
}
```

## 10. License

MIT — see [LICENSE](LICENSE).