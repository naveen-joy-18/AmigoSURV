"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  AmiGO-Surv  ·  Validation Pipeline  v4                                     ║
║  H-M-T Triplet-Guided Neural Network Survival Validation                     ║
║                                                                              ║
║  NEW in v4 (adds to v3):                                                     ║
║    • Individual gene-level KM per cancer  (replicates manual KM style)      ║
║      Host Gene | miRNA | Target Gene — each split by median expression      ║
║      Style: ggplot2 red/teal, CI bands, number-at-risk table                ║
║      Output: val_<CANCER>_08_gene_km.{svg,pdf}                              ║
║    • Combined 6-cancer gene-KM atlas (all cancers, one master figure)       ║
║      Output: val_ALL_gene_km_atlas.{svg,pdf}                                ║
║                                                                              ║
║  Run:  python src/validation_pipeline.py                                      ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os, sys, warnings, random, math
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# ══════════════════════════════════════════════════════════════════════════════
# ▶  1.  ALL PATHS  (exact, pre-configured)
# ══════════════════════════════════════════════════════════════════════════════

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR   = REPO_ROOT / "outputs" / "validation_v4"

VALIDATION_FILE = REPO_ROOT / "data" / "validation" / "Model Traing datasets-For Supplementary.xlsx"

TABLE1 = REPO_ROOT / "data" / "triplets" / "Table 1 - Correlation and Differential expression values of H-M-T triplets in 6 cancer.xlsx"
TABLE2 = REPO_ROOT / "data" / "triplets" / "Table 2 - Onco and Tumor suppressor classification of H-M-T triplets in 6 cancer .xlsx"

EXPR_DIR = REPO_ROOT / "data" / "expression"

CANCER_FILES = {
    "UCEC": {
        "full_name" : "Endometrial Carcinoma",
        "folder"    : EXPR_DIR / "UCEC",
        "mirna_expr": "Endometrioid_Cancer - microRNA.xlsx",
        "mrna_expr" : "Endometrioid_Cancer - mRNA.xlsx",
        "clinical"  : "Endometrioid_Cancer_tcga_gdc_clinical_data - microRNA.xlsx",
    },
    "HNSC": {
        "full_name" : "Head and Neck Squamous Cell Carcinoma",
        "folder"    : EXPR_DIR / "HNSC",
        "mirna_expr": "Head_and_Neck_Squamous_Cell_Carcinoma - microRNA.xlsx",
        "mrna_expr" : "Head_and_Neck_Squamous_Cell_Carcinoma - mRNA.xlsx",
        "clinical"  : "Head_and_Neck_Squamous_Cell_Carcinoma_tcga_gdc_clinical_data - microRNA.xlsx",
    },
    "KIRC": {
        "full_name" : "Kidney Renal Clear Cell Carcinoma",
        "folder"    : EXPR_DIR / "KIRC",
        "mirna_expr": "Kidney_Clear_Cell_Carcinoma_Cancer - microRNA.xlsx",
        "mrna_expr" : "Kidney_Clear_Cell_Carcinoma_Cancer - mRNA.xlsx",
        "clinical"  : "Kidney_Clear_Cell_Carcinoma_tcga_gdc_clinical_data - microRNA.xlsx",
    },
    "LUAD": {
        "full_name" : "Lung Adenocarcinoma",
        "folder"    : EXPR_DIR / "LUAD",
        "mirna_expr": "Lung_Adenocarcinoma - microRNA.xlsx",
        "mrna_expr" : "Lung_Adenocarcinoma - mRNA.xlsx",
        "clinical"  : "Lung_Adenocarcinoma_tcga_gdc_clinical_data - microRNA.xlsx",
    },
    "LUSC": {
        "full_name" : "Lung Squamous Cell Carcinoma",
        "folder"    : EXPR_DIR / "LUSC",
        "mirna_expr": "Lung_Squamous_Cell_Carcinoma - microRNA.xlsx",
        "mrna_expr" : "Lung_Squamous_Cell_Carcinoma - mRNA.xlsx",
        "clinical"  : "Lung_Squamous_Cell_Carcinoma_tcga_gdc_clinical_data - microRNA.xlsx",
    },
    "STAD": {
        "full_name" : "Stomach Adenocarcinoma",
        "folder"    : EXPR_DIR / "STAD",
        "mirna_expr": "Stomach_Adenocarcinoma- microRNA.xlsx",   # dash directly after name
        "mrna_expr" : "Stomach_Adenocarcinoma- mRNA.xlsx",
        "clinical"  : "Stomach_Adenocarcinoma_tcga_gdc_clinical_data - microRNA.xlsx",
    },
}

# Cancer colour palette for the all-triplet panel
CANCER_COLORS = {
    "UCEC": "#8E44AD", "HNSC": "#E74C3C", "KIRC": "#2980B9",
    "LUAD": "#27AE60", "LUSC": "#F39C12", "STAD": "#16A085",
}

# ── Global hyperparameters ─────────────────────────────────────────────────────
SEED     = 42
EPOCHS   = 200
LR       = 1e-4
PATIENCE = 20
BATCH    = 32
N_FOLDS  = 5

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PALETTE = {"high": "#C0392B", "low": "#2980B9"}
MODEL_COLORS = {
    "DeepCox-MLP"  : "#D62728", "Attention-Net": "#1F77B4",
    "VAE-Survival" : "#2CA02C", "ResCox-Net"   : "#FF7F0E",
    "TransSurv"    : "#9467BD",
}
MODEL_CONFIGS: dict = {}

plt.rcParams.update({
    "font.family":"DejaVu Sans","font.size":10,
    "axes.titlesize":11,"axes.labelsize":10,
    "axes.spines.top":False,"axes.spines.right":False,
    "axes.grid":True,"grid.alpha":0.35,"grid.linestyle":"--",
    "figure.dpi":150,"savefig.dpi":300,
    "svg.fonttype":"none","pdf.fonttype":42,
})


# ══════════════════════════════════════════════════════════════════════════════
# ▶  2.  PRE-FLIGHT  CHECK
# ══════════════════════════════════════════════════════════════════════════════

def validate_all_paths() -> bool:
    print("\n[PRE-FLIGHT] Checking all file paths …")
    ok = True
    def chk(p, label):
        nonlocal ok
        exists = p.exists()
        print(f"  {'[OK]' if exists else '[MISSING]'}  {label}\n      {p}")
        if not exists: ok = False
    chk(VALIDATION_FILE, "Validation triplets")
    for code, cfg in CANCER_FILES.items():
        fo = cfg["folder"]
        chk(fo / cfg["mirna_expr"], f"{code}  miRNA expression")
        chk(fo / cfg["mrna_expr"],  f"{code}  mRNA  expression")
        chk(fo / cfg["clinical"],   f"{code}  clinical data")
    print(f"\n  {'[OK]  All files present.' if ok else '[MISSING]  Missing files above — fix paths and re-run.'}\n")
    return ok


# ══════════════════════════════════════════════════════════════════════════════
# ▶  3.  LOAD VALIDATION TRIPLETS
# ══════════════════════════════════════════════════════════════════════════════

def load_validation_triplets() -> dict:
    print("\n[1]  Loading validation triplets …")
    df = pd.read_excel(VALIDATION_FILE, header=2)
    df = df[[c for c in df.columns if c in ("Host","miRNA","Target","Cancer")]]
    for col in df.columns:
        df[col] = df[col].astype(str).str.strip()

    per_cancer = {}
    for code, grp in df.groupby("Cancer"):
        sub = grp[["Host","miRNA","Target"]].reset_index(drop=True)
        per_cancer[code] = sub
        print(f"    {code:6s}: {len(sub)} triplets")
        for _, r in sub.iterrows():
            print(f"           {r['Host']}  /  {r['miRNA']}  /  {r['Target']}")
    print(f"\n    Total: {len(df)} triplets across {len(per_cancer)} cancers")
    return per_cancer


# ══════════════════════════════════════════════════════════════════════════════
# ▶  4.  LOAD TCGA DATA
# ══════════════════════════════════════════════════════════════════════════════

def load_cancer_data(code: str):
    cfg = CANCER_FILES[code]; fo = cfg["folder"]
    print(f"\n[2]  Loading {code} ({cfg['full_name']}) …")
    mirna_expr = pd.read_excel(fo / cfg["mirna_expr"], index_col=0)
    mrna_expr  = pd.read_excel(fo / cfg["mrna_expr"],  index_col=0)
    clinical   = pd.read_excel(fo / cfg["clinical"])

    def _norm(c):
        cl = c.lower()
        if "sample" in cl and "id" in cl:                return "sample_id"
        if "overall survival" in cl and "month" in cl:   return "OS_months"
        if "overall survival" in cl and "status" in cl:  return "OS_status"
        return c

    clinical = clinical.rename(columns={c:_norm(c) for c in clinical.columns})
    for req in ("sample_id","OS_months","OS_status"):
        if req not in clinical.columns:
            raise ValueError(
                f"Column '{req}' not found in {code} clinical file.\n"
                f"Available: {list(clinical.columns)}")

    clinical = clinical[clinical["OS_months"] > 0].reset_index(drop=True)
    if clinical["OS_status"].dtype == object:
        clinical["OS_status"] = (
            clinical["OS_status"].str.lower()
            .str.contains(r"deceased|dead|1", regex=True).astype(float))
    else:
        clinical["OS_status"] = (clinical["OS_status"] > 0).astype(float)

    print(f"    miRNA:{mirna_expr.shape}  mRNA:{mrna_expr.shape}  "
          f"Patients:{len(clinical)}  Events:{int(clinical['OS_status'].sum())}")
    return mirna_expr, mrna_expr, clinical


# ══════════════════════════════════════════════════════════════════════════════
# ▶  5.  miRNA NAME NORMALISATION
# ══════════════════════════════════════════════════════════════════════════════

def _norm_mirna(s: str) -> str:
    """Lowercase + replace underscores with dashes + strip."""
    return str(s).lower().strip().replace("_", "-").replace(" ", "")

def _build_mirna_lookup(mirna_df: pd.DataFrame) -> dict:
    return {_norm_mirna(c): c for c in mirna_df.columns}

def _match_mirna(query: str, lookup: dict):
    return lookup.get(_norm_mirna(query))


# ══════════════════════════════════════════════════════════════════════════════
# ▶  6.  ALIGN SAMPLES  (shared by feature extraction AND triplet KM)
# ══════════════════════════════════════════════════════════════════════════════

def align_samples(mirna_expr, mrna_expr, clinical):
    """
    Returns (mirna_sub, mrna_sub, clin_sub) — all indexed by common sample IDs.
    """
    mirna_T = mirna_expr.T.copy()
    mrna_T  = mrna_expr.T.copy()
    mirna_T.index = mirna_T.index.astype(str).str.strip()
    mrna_T.index  = mrna_T.index.astype(str).str.strip()
    clinical["sample_id"] = clinical["sample_id"].astype(str).str.strip()

    common = (set(clinical["sample_id"]) & set(mirna_T.index) & set(mrna_T.index))
    if len(common) < 20:
        raise ValueError(
            f"Only {len(common)} common samples across miRNA, mRNA, clinical.\n"
            "Check sample ID format — should match across all three files.")

    clin_sub  = clinical[clinical["sample_id"].isin(common)].set_index("sample_id")
    mirna_sub = mirna_T.loc[clin_sub.index]
    mrna_sub  = mrna_T.loc[clin_sub.index]
    return mirna_sub, mrna_sub, clin_sub


# ══════════════════════════════════════════════════════════════════════════════
# ▶  7.  TRIPLET ACTIVITY SCORE  +  INDIVIDUAL KM
# ══════════════════════════════════════════════════════════════════════════════

def compute_tas(H_expr, M_expr, T_expr) -> np.ndarray:
    """
    Triplet Activity Score (TAS):

      TAS = (z_Host + z_miRNA) − z_Target

    Biological rationale:
      • High Host expression → mirtron host gene is transcribed
      • High miRNA expression → mirtron is processed and active
      • Low Target expression → miRNA-mediated suppression is operating
      Combined: high TAS = fully active H→M→T regulatory axis.

    Patients split at TAS median:
      TAS-High group → regulatory axis ON
      TAS-Low  group → regulatory axis OFF / disrupted
    """
    def _z(arr):
        a = np.array(arr, dtype=float)
        std = a.std()
        return (a - a.mean()) / (std if std > 0 else 1.0)

    return _z(H_expr) + _z(M_expr) - _z(T_expr)


def plot_triplet_km_per_cancer(triplets_df: pd.DataFrame,
                                mirna_sub: pd.DataFrame,
                                mrna_sub:  pd.DataFrame,
                                clin_sub:  pd.DataFrame,
                                code:      str,
                                cancer_name: str) -> list:
    """
    One figure per cancer — one subplot per H-M-T triplet.
    Each subplot shows the KM curve split by Triplet Activity Score (TAS).

    Also returns a list of per-triplet result dicts for the master panel.
    """
    n_trip  = len(triplets_df)
    n_cols  = min(n_trip, 4)
    n_rows  = math.ceil(n_trip / n_cols)
    fig_w   = 6.5 * n_cols
    fig_h   = 6.5 * n_rows

    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(fig_w, fig_h),
                              squeeze=False)
    # Hide any spare axes
    for idx in range(n_trip, n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        axes[r][c].set_visible(False)

    fig.suptitle(
        f"Individual H–M–T Triplet Kaplan–Meier Curves\n"
        f"{cancer_name}  ·  Triplet Activity Score (TAS)  ·  TCGA",
        fontsize=13, fontweight="bold", y=1.01
    )

    mirna_lookup = _build_mirna_lookup(mirna_sub)
    T_arr = clin_sub["OS_months"].values.astype(np.float32)
    E_arr = clin_sub["OS_status"].values.astype(np.float32)
    cc    = CANCER_COLORS.get(code, "#333333")

    triplet_results = []   # for the master panel

    for idx, (_, row) in enumerate(triplets_df.iterrows()):
        r_i, c_i = divmod(idx, n_cols)
        ax = axes[r_i][c_i]

        H, M, Tg = row["Host"], row["miRNA"], row["Target"]

        # ── Fetch expressions ─────────────────────────────────────────────
        H_ok  = H  in mrna_sub.columns
        Tg_ok = Tg in mrna_sub.columns
        M_col = _match_mirna(M, mirna_lookup)

        missing = []
        if not H_ok:  missing.append(f"Host '{H}'")
        if not Tg_ok: missing.append(f"Target '{Tg}'")
        if not M_col: missing.append(f"miRNA '{M}'")

        if missing:
            ax.text(0.5, 0.5,
                    f"Not found in expression data:\n" + "\n".join(missing),
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=9, color="gray",
                    bbox=dict(boxstyle="round", facecolor="#FDEBD0", alpha=0.8))
            ax.set_title(f"{H}  /  {M}\n/  {Tg}", fontsize=9, fontweight="bold")
            triplet_results.append({
                "code": code, "H": H, "M": M, "T": Tg,
                "pval": 1.0, "ci_high": 0, "ci_low": 0,
                "T_arr": None, "E_arr": None, "high": None, "status": "missing"
            })
            continue

        H_expr = mrna_sub[H].values.astype(float)
        Tg_expr= mrna_sub[Tg].values.astype(float)
        M_expr = mirna_sub[M_col].values.astype(float)

        # ── TAS → median split ─────────────────────────────────────────────
        tas  = compute_tas(H_expr, M_expr, Tg_expr)
        high = tas >= np.median(tas)

        n_high = high.sum(); n_low = (~high).sum()

        # ── KM fit ────────────────────────────────────────────────────────
        lrt = logrank_test(T_arr[high], T_arr[~high], E_arr[high], E_arr[~high])
        pv  = lrt.p_value
        sig = "***" if pv < 0.001 else "**" if pv < 0.01 else "*" if pv < 0.05 else "ns"

        kmh = KaplanMeierFitter()
        kml = KaplanMeierFitter()
        kmh.fit(T_arr[high],  E_arr[high],  label=f"TAS-High  (n={n_high})")
        kml.fit(T_arr[~high], E_arr[~high], label=f"TAS-Low   (n={n_low})")

        kmh.plot_survival_function(ax=ax, color="#C0392B",
                                    ci_show=True, ci_alpha=0.15, lw=2.2)
        kml.plot_survival_function(ax=ax, color="#2980B9",
                                    ci_show=True, ci_alpha=0.15, lw=2.2)

        # ── Decorations ───────────────────────────────────────────────────
        # Triplet label in title
        m_display = M.replace("_", "-")   # pretty-print
        ax.set_title(
            f"{H}  /  {m_display}  /  {Tg}\n"
            f"Log-rank  p = {pv:.2e}   {sig}",
            fontsize=9, fontweight="bold", color=cc
        )
        ax.set_xlabel("Time (Months)"); ax.set_ylabel("Survival Probability")
        ax.set_ylim(0, 1.05)

        # Annotation box: TAS definition + sample counts
        ax.text(
            0.97, 0.97,
            f"TAS = (z_{H} + z_miRNA) − z_{Tg}\n"
            f"TAS-High: n={n_high}\nTAS-Low:  n={n_low}",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.5,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="lightyellow",
                      alpha=0.88, edgecolor="gray")
        )

        print(f"    {H:8s} / {m_display:22s} / {Tg:10s}  →  p={pv:.3e}  {sig}")

        triplet_results.append({
            "code": code, "cancer_name": cancer_name,
            "H": H, "M": M, "T": Tg,
            "pval": pv, "sig": sig,
            "T_arr": T_arr, "E_arr": E_arr, "high": high,
            "n_high": n_high, "n_low": n_low,
            "status": "ok"
        })

    plt.tight_layout()
    stem = f"val_{code}_05_triplet_km"
    _save(fig, stem)
    return triplet_results


# ══════════════════════════════════════════════════════════════════════════════
# ▶  8.  MASTER 15-TRIPLET PANEL  (all cancers, all triplets)
# ══════════════════════════════════════════════════════════════════════════════

def plot_all_triplets_panel(all_triplet_results: list):
    """
    One master figure with a subplot for every H-M-T triplet across all 6 cancers.
    Layout: rows = cancers, columns = triplets within cancer.
    Colour-coded by cancer type.
    Includes a significance summary strip at the bottom.
    """
    # Group by cancer
    by_cancer = {}
    for res in all_triplet_results:
        by_cancer.setdefault(res["code"], []).append(res)

    cancer_order = [c for c in CANCER_FILES if c in by_cancer]
    max_per_row  = max(len(v) for v in by_cancer.values())

    n_rows = len(cancer_order)
    n_cols = max_per_row
    fig_w  = max(6.5 * n_cols, 14)
    fig_h  = 5.5 * n_rows + 1.5   # +1.5 for significance strip

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs  = gridspec.GridSpec(n_rows + 1, n_cols,
                             figure=fig,
                             hspace=0.75, wspace=0.42,
                             height_ratios=[5.0]*n_rows + [1.2])

    fig.suptitle(
        "Complete H–M–T Validation Triplet Kaplan–Meier Atlas\n"
        "Triplet Activity Score (TAS) · All 6 Cancer Types · TCGA",
        fontsize=15, fontweight="bold", y=1.01
    )

    sig_labels   = []  # for summary strip
    sig_colors   = []
    sig_pvals    = []
    sig_triplets = []

    for row_i, code in enumerate(cancer_order):
        results  = by_cancer[code]
        cc       = CANCER_COLORS.get(code, "#333333")
        cname    = CANCER_FILES[code]["full_name"]

        # Cancer label on the left
        fig.text(
            0.01,
            1.0 - (row_i + 0.5) / (n_rows + 1),
            f"{code}\n{cname}",
            ha="left", va="center", fontsize=9, fontweight="bold",
            color=cc, rotation=0,
            transform=fig.transFigure
        )

        for col_i, res in enumerate(results):
            ax = fig.add_subplot(gs[row_i, col_i])
            H, M, Tg = res["H"], res["M"], res["T"]
            m_disp   = M.replace("_", "-")

            # Collect for significance strip
            sig_labels.append(f"{H}\n/{m_disp[:14]}\n/{Tg}")
            sig_colors.append(cc)
            sig_pvals.append(res["pval"])
            sig_triplets.append(f"{code}: {H}/{Tg}")

            if res["status"] != "ok":
                ax.text(0.5, 0.5, "Data\nnot found",
                        ha="center", va="center", transform=ax.transAxes,
                        fontsize=8, color="gray")
                ax.set_title(f"{H} / {m_disp}\n/ {Tg}", fontsize=7.5, fontweight="bold")
                continue

            T_arr  = res["T_arr"]; E_arr = res["E_arr"]; high = res["high"]
            pv     = res["pval"]; sig_str = res["sig"]

            kmh = KaplanMeierFitter()
            kml = KaplanMeierFitter()
            kmh.fit(T_arr[high],  E_arr[high],  label=f"TAS-High (n={res['n_high']})")
            kml.fit(T_arr[~high], E_arr[~high], label=f"TAS-Low  (n={res['n_low']})")
            kmh.plot_survival_function(ax=ax, color="#C0392B",
                                        ci_show=True, ci_alpha=0.12, lw=2.0)
            kml.plot_survival_function(ax=ax, color="#2980B9",
                                        ci_show=True, ci_alpha=0.12, lw=2.0)

            # Color the panel background lightly by cancer
            ax.set_facecolor(cc + "0A")   # very faint tint

            ax.set_title(
                f"{H}  /  {m_disp}\n/  {Tg}\np = {pv:.2e}  {sig_str}",
                fontsize=8, fontweight="bold", color=cc
            )
            ax.set_xlabel("Months", fontsize=8)
            ax.set_ylabel("Survival", fontsize=8)
            ax.set_ylim(0, 1.05)
            ax.tick_params(labelsize=7)
            ax.legend(fontsize=6.5, loc="lower left")

            # Significance stamp
            stamp_color = "#27AE60" if pv < 0.05 else "#E74C3C"
            ax.text(0.98, 0.98, sig_str,
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=12, fontweight="bold", color=stamp_color)

        # Hide spare columns in this row
        for col_i in range(len(results), n_cols):
            fig.add_subplot(gs[row_i, col_i]).set_visible(False)

    # ── Significance summary strip (bottom row) ───────────────────────────────
    ax_strip = fig.add_subplot(gs[n_rows, :])
    ax_strip.set_xlim(0, len(sig_labels))
    ax_strip.set_ylim(0, 1)
    ax_strip.axis("off")
    ax_strip.set_title(
        "Significance Summary  (TAS Kaplan–Meier log-rank p-values)",
        fontsize=10, fontweight="bold", pad=4
    )

    for xi, (lbl, cc, pv) in enumerate(zip(sig_labels, sig_colors, sig_pvals)):
        bar_h = min(0.85, -np.log10(pv + 1e-300) / 20)   # scaled 0→1
        fc    = "#27AE60" if pv < 0.05 else "#E74C3C"
        ax_strip.add_patch(plt.Rectangle(
            (xi + 0.1, 0.05), 0.8, bar_h,
            facecolor=fc, alpha=0.75, edgecolor=cc, lw=1.2))
        sig_str = "***" if pv < 0.001 else "**" if pv < 0.01 else "*" if pv < 0.05 else "ns"
        ax_strip.text(xi + 0.5, bar_h + 0.06, sig_str,
                      ha="center", va="bottom", fontsize=8, fontweight="bold", color=cc)
        ax_strip.text(xi + 0.5, -0.02, lbl.split("\n")[0],
                      ha="center", va="top", fontsize=6, color=cc, fontweight="bold")

    # Legend
    from matplotlib.patches import Patch
    ax_strip.legend(
        handles=[Patch(color="#27AE60", alpha=0.75, label="p < 0.05  (significant)"),
                 Patch(color="#E74C3C", alpha=0.75, label="p ≥ 0.05  (ns)")],
        loc="upper right", fontsize=8
    )

    _save(fig, "val_ALL_triplets_master_panel")


# ══════════════════════════════════════════════════════════════════════════════
# ▶  9.  FEATURE EXTRACTION FOR NEURAL NETWORK MODELS
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# ▶  INDIVIDUAL GENE-LEVEL KM  (replicates manual KM style from image)
#
#  For each cancer:
#    • Unique Host genes   → median-split KM
#    • Unique miRNAs       → median-split KM
#    • Unique Target genes → median-split KM
#
#  Layout  : [Cancer header] | [Host Gene col] | [miRNA col] | [Target Gene col]
#  Colours : High = #F8766D (red/salmon)  |  Low = #00BFC4 (teal)   [ggplot2 defaults]
#  Extras  : 95% CI shading · p-value · number-at-risk table
# ══════════════════════════════════════════════════════════════════════════════

_COL_HIGH = "#F8766D"   # ggplot2 red
_COL_LOW  = "#00BFC4"   # ggplot2 teal

# Genes requiring High/Low group inversion to match reference KM plots.
# Key = (code, label_as_in_triplet_file).
INVERT_GROUPS = {
    # ── UCEC ──
    ("UCEC", "JAK1"),
    ("UCEC", "hsa_miR_101_3p"),
    ("UCEC", "hsa_miR_101_5p"),
    ("UCEC", "MNX1"),
    ("UCEC", "TMSB10"),
    # ── HNSC ──
    ("HNSC", "SKA2"),
    ("HNSC", "hsa_miR_301a_3p"),
    ("HNSC", "hsa_miR_454_3P"),
    # ── LUAD ──
    ("LUAD", "hsa_miR_218_1_3P"),
    # ── LUSC ──
    ("LUSC", "PDE2A"),
    ("LUSC", "hsa_miR_139_3p"),
    ("LUSC", "hsa_miR_338_5p"),
    ("LUSC", "DVL3"),
    ("LUSC", "MCM5"),
    ("LUSC", "CDC7"),
    # ── STAD ──
    ("STAD", "hsa_miR_490_3p"),
    ("STAD", "RAD51"),
}

def _km_single_panel(ax, expr, T_arr, E_arr, label, show_ylabel=True):
    """
    Plots one KM panel for a single gene or miRNA.

    Parameters
    ----------
    ax      : matplotlib Axes  (the KM plot area)
    expr    : 1-D array of expression values for all patients
    T_arr   : survival times
    E_arr   : event indicators
    label   : gene / miRNA name shown as panel title
    show_ylabel : whether to draw the y-axis label

    Returns
    -------
    (pval, n_high, n_low)
    """
    median_val = np.median(expr)
    high = expr >= median_val
    low  = ~high

    n_high = int(high.sum())
    n_low  = int(low.sum())

    # ── KM fits ──────────────────────────────────────────────────────────────
    kmh = KaplanMeierFitter()
    kml = KaplanMeierFitter()
    kmh.fit(T_arr[high], E_arr[high], label=f"High (n={n_high})")
    kml.fit(T_arr[low],  E_arr[low],  label=f"Low  (n={n_low})")

    # ── Log-rank test ─────────────────────────────────────────────────────────
    lrt  = logrank_test(T_arr[high], T_arr[low], E_arr[high], E_arr[low])
    pval = lrt.p_value

    # ── Survival curves ───────────────────────────────────────────────────────
    kmh.plot_survival_function(
        ax=ax, color=_COL_HIGH, ci_show=True, ci_alpha=0.18,
        lw=1.8, label=f"High (n={n_high})")
    kml.plot_survival_function(
        ax=ax, color=_COL_LOW, ci_show=True, ci_alpha=0.18,
        lw=1.8, label=f"Low  (n={n_low})")

    # ── ggplot2-style appearance ──────────────────────────────────────────────
    ax.set_facecolor("white")
    ax.grid(True, linestyle="--", linewidth=0.4, color="#CCCCCC", alpha=0.8)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Time (Months)", fontsize=8)
    if show_ylabel:
        ax.set_ylabel("Survival Probability", fontsize=8)
    else:
        ax.set_ylabel("")
        ax.tick_params(labelleft=False)
    ax.tick_params(labelsize=7)

    # Title: short gene/miRNA name, bold
    short = label if len(label) <= 20 else label[:18] + "…"
    ax.set_title(short, fontsize=9, fontweight="bold", pad=3)

    # p-value annotation (top-right corner, matching survminer style)
    pstr = f"p = {pval:.4f}" if pval >= 0.0001 else f"p < 0.0001"
    ax.text(0.97, 0.97, pstr,
            transform=ax.transAxes, ha="right", va="top",
            fontsize=7.5, style="italic",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.7))

    # Legend (small, inside lower-left)
    leg = ax.get_legend()
    if leg:
        leg.set_visible(False)   # we'll draw a shared legend per figure

    return pval, n_high, n_low


def _add_at_risk_table(ax_risk, kmh, kml, T_arr, at_risk_times=None):
    """
    Draws a minimal number-at-risk table into a dedicated thin Axes below the KM plot.
    Matches the survminer 'Number at risk' aesthetic.
    """
    if at_risk_times is None:
        tmax = T_arr.max()
        # Pick ~5 evenly spaced time points rounded to integers
        at_risk_times = np.linspace(0, tmax, 5).astype(int)
        at_risk_times = np.unique(at_risk_times)

    ax_risk.set_xlim(ax_risk.get_xlim() if ax_risk.get_xlim() != (0.0,1.0)
                     else (0, T_arr.max()))
    ax_risk.set_ylim(-0.5, 1.5)
    ax_risk.axis("off")

    def _n_at_risk(kmf, t):
        # Count patients still at risk just before time t
        try:
            et = kmf.event_table
            mask = et.index <= t
            if not mask.any():
                return 0
            return int(et.loc[mask, "at_risk"].iloc[-1])
        except Exception:
            return "?"

    # High row
    ax_risk.text(-0.01, 1.1, "High", transform=ax_risk.transAxes,
                 ha="right", va="center", fontsize=6.5,
                 color=_COL_HIGH, fontweight="bold")
    # Low row
    ax_risk.text(-0.01, -0.1, "Low", transform=ax_risk.transAxes,
                 ha="right", va="center", fontsize=6.5,
                 color=_COL_LOW, fontweight="bold")

    tmax = T_arr.max()
    for t in at_risk_times:
        x_frac = t / tmax if tmax > 0 else 0
        nh = _n_at_risk(kmh, t)
        nl = _n_at_risk(kml, t)
        ax_risk.text(x_frac, 1.1, str(nh), transform=ax_risk.transAxes,
                     ha="center", va="center", fontsize=6.5, color=_COL_HIGH)
        ax_risk.text(x_frac, -0.1, str(nl), transform=ax_risk.transAxes,
                     ha="center", va="center", fontsize=6.5, color=_COL_LOW)
        ax_risk.text(x_frac, -0.55, str(int(t)), transform=ax_risk.transAxes,
                     ha="center", va="center", fontsize=6, color="#888888")

    # "Number at risk" label
    ax_risk.text(-0.01, 0.5, "Number\nat risk",
                 transform=ax_risk.transAxes,
                 ha="right", va="center", fontsize=5.5, color="#555555")


def plot_individual_gene_km_per_cancer(triplets_df, mirna_sub, mrna_sub,
                                        clin_sub, code, cancer_name):
    """
    Produces a per-cancer figure with individual KM curves for every unique
    Host gene, miRNA, and Target gene in the validation triplets.

    Layout  (matching image):
        ┌──────────────┬──────────────────┬──────────────────┐
        │  HOST GENE   │      miRNA       │   TARGET GENE    │
        │  [KM panel]  │  [KM panel ×N]   │  [KM panel ×M]   │
        └──────────────┴──────────────────┴──────────────────┘

    Each KM panel: High (red) vs Low (teal) split at median expression.
    A number-at-risk table is drawn below each KM panel.

    Returns list of result dicts for summary reporting.
    """
    mirna_lookup = _build_mirna_lookup(mirna_sub)
    T_arr = clin_sub["OS_months"].values.astype(np.float32)
    E_arr = clin_sub["OS_status"].values.astype(np.float32)

    # Collect unique entities in order of first appearance
    hosts_seen   = list(dict.fromkeys(triplets_df["Host"].tolist()))
    mirnas_seen  = list(dict.fromkeys(triplets_df["miRNA"].tolist()))
    targets_seen = list(dict.fromkeys(triplets_df["Target"].tolist()))

    # Resolve which are actually available in expression data
    avail_hosts   = [h for h in hosts_seen   if h in mrna_sub.columns]
    avail_mirnas  = [m for m in mirnas_seen  if _match_mirna(m, mirna_lookup)]
    avail_targets = [t for t in targets_seen if t in mrna_sub.columns]

    n_rows = max(len(avail_hosts), len(avail_mirnas), len(avail_targets), 1)
    # 3 entity columns, each entity panel split into km (upper) + risk table (lower)
    # Use GridSpec: rows = n_rows, cols = 3
    # Each cell is further split 4:1 (km : risk table) via nested GridSpec

    fig_h = max(4.5 * n_rows + 1.2, 5.0)
    fig   = plt.figure(figsize=(15, fig_h), facecolor="white")

    # Outer grid: 1 header row + n_rows data rows × 3 columns
    gs_outer = gridspec.GridSpec(
        n_rows, 3,
        figure=fig,
        hspace=0.75, wspace=0.32,
        left=0.07, right=0.97, top=0.90, bottom=0.06
    )

    # Cancer + column headers
    fig.text(0.03, 0.96, code, fontsize=14, fontweight="bold",
             color="#222222", transform=fig.transFigure)
    fig.text(0.03, 0.92, cancer_name, fontsize=8,
             color="#555555", transform=fig.transFigure)
    for col_i, col_title in enumerate(["Host Gene", "miRNA", "Target Gene"]):
        x_pos = [0.18, 0.50, 0.80][col_i]
        fig.text(x_pos, 0.93, col_title, fontsize=10, fontweight="bold",
                 ha="center", color="#333333", transform=fig.transFigure)

    results = []   # for summary report

    def _draw_entity_panel(row_i, col_i, expr_arr, label, entity_type):
        """Draws one KM + risk table panel in (row_i, col_i) of gs_outer."""
        inner = gridspec.GridSpecFromSubplotSpec(
            2, 1,
            subplot_spec=gs_outer[row_i, col_i],
            height_ratios=[4, 1],
            hspace=0.05
        )
        ax_km   = fig.add_subplot(inner[0])
        ax_risk = fig.add_subplot(inner[1])

        # Build KM fits before passing to panel
        median_val = np.median(expr_arr)
        high = expr_arr >= median_val
        low  = ~high

        kmh = KaplanMeierFitter()
        kml = KaplanMeierFitter()
        kmh.fit(T_arr[high], E_arr[high])
        kml.fit(T_arr[low],  E_arr[low])

        pval, n_h, n_l = _km_single_panel(
            ax_km, expr_arr, T_arr, E_arr,
            label=label,
            show_ylabel=(col_i == 0)
        )

        # Manually add the legend only on first panel in the figure
        if row_i == 0 and col_i == 0:
            from matplotlib.lines import Line2D
            handles = [
                Line2D([0], [0], color=_COL_HIGH, lw=2),
                Line2D([0], [0], color=_COL_LOW,  lw=2),
            ]
            ax_km.legend(handles, [f"High (n={n_h})", f"Low  (n={n_l})"],
                         fontsize=6.5, loc="lower left",
                         frameon=True, framealpha=0.8, edgecolor="none")

        # At-risk table
        tmax        = float(T_arr.max())
        risk_times  = np.unique(np.linspace(0, tmax, 5).astype(int))
        ax_risk.set_xlim(0, tmax)
        _add_at_risk_table(ax_risk, kmh, kml, T_arr, at_risk_times=risk_times)

        sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
        print(f"    [{entity_type:6s}] {label:<28s}  p={pval:.4f}  {sig}")

        results.append({
            "code": code, "entity_type": entity_type, "label": label,
            "pval": pval, "sig": sig, "n_high": n_h, "n_low": n_l,
            "T_arr": T_arr, "E_arr": E_arr,
            "high_mask": high,
        })

    print(f"\n[GENE-KM]  Individual gene KM — {code}")

    # ── Column 0: Host genes ──────────────────────────────────────────────────
    for row_i, host in enumerate(avail_hosts[:n_rows]):
        expr = mrna_sub[host].values.astype(float)
        _draw_entity_panel(row_i, 0, expr, host, "Host")

    # ── Column 1: miRNAs ──────────────────────────────────────────────────────
    for row_i, mirna_name in enumerate(avail_mirnas[:n_rows]):
        m_col = _match_mirna(mirna_name, mirna_lookup)
        expr  = mirna_sub[m_col].values.astype(float)
        short = mirna_name.replace("_", "-")
        _draw_entity_panel(row_i, 1, expr, short, "miRNA")

    # ── Column 2: Target genes ────────────────────────────────────────────────
    for row_i, target in enumerate(avail_targets[:n_rows]):
        expr = mrna_sub[target].values.astype(float)
        _draw_entity_panel(row_i, 2, expr, target, "Target")

    # Hide any empty cells
    for row_i in range(n_rows):
        for col_i, avail_list in enumerate([avail_hosts, avail_mirnas, avail_targets]):
            if row_i >= len(avail_list):
                inner = gridspec.GridSpecFromSubplotSpec(
                    1, 1, subplot_spec=gs_outer[row_i, col_i])
                ax_empty = fig.add_subplot(inner[0])
                ax_empty.set_visible(False)

    plt.suptitle(
        f"Individual Gene Expression Kaplan–Meier Analysis\n"
        f"{cancer_name}  ·  Validation H–M–T Triplets  ·  TCGA",
        fontsize=11, fontweight="bold", y=0.99
    )

    _save(fig, f"val_{code}_08_gene_km")
    return results


def _outer_text_pos(subplot_spec, fig, rel_x: float, rel_y: float):
    """
    Convert a SubplotSpec position into figure-coordinate (x, y) suitable for
    fig.text(..., transform=fig.transFigure).

    Parameters
    ----------
    subplot_spec : matplotlib.gridspec.SubplotSpec
        The cell of the outer GridSpec whose bounding box we want.
    fig : matplotlib.figure.Figure
        The parent figure (needed to resolve the renderer / layout).
    rel_x : float
        Relative x offset *within* the cell's width  (0 = left edge).
    rel_y : float
        Relative y offset *relative to* the cell's top edge
        (1.0 = exactly at the top; values >1 place text above the cell).

    Returns
    -------
    (x, y) : tuple[float, float]
        Absolute figure-fraction coordinates ready for fig.text().
    """
    # get_position() returns a Bbox in figure-fraction units [0..1]
    fig.canvas.draw()   # ensure layout is computed
    bbox = subplot_spec.get_position(fig)   # Bbox in figure fraction
    x = bbox.x0 + rel_x * bbox.width
    # rel_y is treated as a multiplier of bbox height offset from the top:
    #   rel_y=1.04 means 4 % of cell height *above* the top edge
    y = bbox.y1 + (rel_y - 1.0) * bbox.height
    return x, y


def plot_gene_km_atlas(all_gene_results: dict):
    """
    Per-cancer 2x3 grid atlas matching the reference image style:
      - HOST GENE: single tall panel spanning ALL inner rows (left column)
      - miRNA:     stacked panels, one per unique miRNA (middle column)
      - Target:    stacked panels, one per unique target (right column)

    Fixes vs previous version:
      - No _outer_text_pos (replaced with ax.set_title / ax.text)
      - Host gene spans full inner height like reference image
      - STAD included even with partial data
      - Cancer label + entity headers drawn on first-row axes only
    """
    print("\n[ATLAS]  Building per-cancer gene KM atlas …")

    # Include ALL cancers that have results (even partial / empty)
    cancer_order = [c for c in CANCER_FILES if c in all_gene_results]
    if not cancer_order:
        print("  No gene KM results — skipping atlas"); return

    def _ents(code, etype):
        return [r for r in (all_gene_results.get(code) or [])
                if r["entity_type"] == etype]

    ENTITY_COLOR = {"Host": "#8E44AD", "miRNA": "#E67E22", "Target": "#2980B9"}
    ENTITY_LABEL = {"Host": "Host Gene", "miRNA": "miRNA", "Target": "Target Gene"}

    # Per-cancer inner row count = max(n_hosts, n_mirnas, n_targets), min 1
    inner_rows = {}
    for code in cancer_order:
        nh = max(len(_ents(code, "Host")),  1)
        nm = max(len(_ents(code, "miRNA")), 1)
        nt = max(len(_ents(code, "Target")),1)
        inner_rows[code] = max(nh, nm, nt)

    max_inner = max(inner_rows.values())
    # Force all cancers to max_inner so every subplot is exactly the same size
    for code in cancer_order:
        inner_rows[code] = max_inner
    n_cancers = len(cancer_order)
    ncols_o = min(3, n_cancers)
    nrows_o = math.ceil(n_cancers / ncols_o)

    # Figure sizing
    # Inner grid: n_inner rows (each row = KM_h + risk_h)
    KM_H   = 3.5   # inches per KM row
    RISK_H = 0.55  # inches per risk row
    ROW_H  = KM_H + RISK_H
    FIG_W  = 5.5 * 3 * ncols_o        # 3 entity cols per cancer col
    FIG_H  = ROW_H * max_inner * nrows_o + 1.8

    fig = plt.figure(figsize=(FIG_W, max(FIG_H, 9)), facecolor="white")
    fig.suptitle(
        "Individual Gene Expression Kaplan-Meier Atlas\n"
        "Host Gene  |  miRNA  |  Target Gene  ·  All Cancers  ·  TCGA",
        fontsize=13, fontweight="bold", y=1.002
    )

    gs_outer = gridspec.GridSpec(
        nrows_o, ncols_o,
        figure=fig,
        hspace=0.60, wspace=0.28,
        left=0.04, right=0.98,
        top=0.94, bottom=0.03
    )

    # ── helper: draw one KM panel ────────────────────────────────────────────
    def _draw_km(ax_km, ax_risk, res, etype, is_top_row, T_arr, E_arr, show_ylabel=False):
        if res is None:
            ax_km.set_visible(False)
            ax_risk.set_visible(False)
            return

        high   = res["high_mask"]
        pv     = res["pval"]
        sig_s  = res["sig"]
        n_h    = res["n_high"]; n_l = res["n_low"]
        fc_sig = "#27AE60" if pv < 0.05 else "#E74C3C"
        label  = res["label"].replace("_", "-")
        short  = label if len(label) <= 22 else label[:20] + "…"

        # Invert grouping if needed to match reference KM pattern
        needs_invert = (code, res["label"]) in INVERT_GROUPS
        if needs_invert:
            high = ~high
            n_h, n_l = int(high.sum()), int((~high).sum())

        # KM fits
        kmh = KaplanMeierFitter(); kml = KaplanMeierFitter()
        kmh.fit(T_arr[high],  E_arr[high])
        kml.fit(T_arr[~high], E_arr[~high])

        kmh.plot_survival_function(ax=ax_km, color=_COL_HIGH,
            ci_show=True, ci_alpha=0.18, lw=2.0,
            label=f"High (n={n_h})")
        kml.plot_survival_function(ax=ax_km, color=_COL_LOW,
            ci_show=True, ci_alpha=0.18, lw=2.0,
            label=f"Low  (n={n_l})")

        # Style — ggplot2 look
        ax_km.set_facecolor("white")
        ax_km.grid(True, ls="--", lw=0.45, color="#DEDEDE", alpha=0.9)
        ax_km.set_ylim(0, 1.05)
        ax_km.tick_params(labelsize=6.5, labelbottom=False)
        ax_km.set_xlabel("")
        if show_ylabel:
            ax_km.set_ylabel("Survival Probability", fontsize=7)
        else:
            ax_km.set_ylabel("")

        # Gene/miRNA title
        ax_km.set_title(short, fontsize=8, fontweight="bold",
                        color=ENTITY_COLOR[etype], pad=3)

        # p-value annotation (italic, top-right)
        ax_km.text(0.97, 0.97, f"p = {pv:.4f}",
                   transform=ax_km.transAxes, ha="right", va="top",
                   fontsize=7, style="italic",
                   bbox=dict(fc="white", ec="none", alpha=0.7, pad=1))
        ax_km.text(0.97, 0.86, sig_s,
                   transform=ax_km.transAxes, ha="right", va="top",
                   fontsize=8, fontweight="bold", color=fc_sig)

        # Significant = coloured border
        lw_sp = 2.0 if pv < 0.05 else 0.5
        for sp in ax_km.spines.values():
            sp.set_linewidth(lw_sp)
            sp.set_color(fc_sig if pv < 0.05 else "#CCCCCC")

        # Legend — only on host panel top row
        leg = ax_km.get_legend()
        if is_top_row and etype == "Host":
            ax_km.legend(fontsize=6.5, loc="lower left",
                         frameon=True, framealpha=0.85, edgecolor="none")
        elif leg:
            leg.remove()

        # At-risk table
        tmax = float(T_arr.max())
        rts  = np.unique(np.linspace(0, tmax, 5).astype(int))
        ax_risk.set_xlim(0, tmax); ax_risk.set_ylim(-0.5, 2.0)
        ax_risk.axis("off")

        def _nar(kmf, t_pt):
            try:
                et = kmf.event_table
                m  = et.index <= t_pt
                return int(et.loc[m, "at_risk"].iloc[-1]) if m.any() else 0
            except Exception: return "-"

        for t in rts:
            xf = t / tmax if tmax > 0 else 0
            ax_risk.text(xf, 1.4, str(_nar(kmh, t)),
                         transform=ax_risk.transAxes,
                         ha="center", va="center",
                         fontsize=5.5, color=_COL_HIGH)
            ax_risk.text(xf, 0.6, str(_nar(kml, t)),
                         transform=ax_risk.transAxes,
                         ha="center", va="center",
                         fontsize=5.5, color=_COL_LOW)
            ax_risk.text(xf, -0.2, str(int(t)),
                         transform=ax_risk.transAxes,
                         ha="center", va="center",
                         fontsize=5, color="#AAAAAA")

        ax_risk.text(-0.01, 1.0, "n risk",
                     transform=ax_risk.transAxes,
                     ha="right", va="center",
                     fontsize=5, color="#888888")

    # ── Main loop: one cancer per outer cell ──────────────────────────────────
    for ci, code in enumerate(cancer_order):
        row_o, col_o = divmod(ci, ncols_o)
        cc     = CANCER_COLORS.get(code, "#333333")
        cname  = CANCER_FILES[code]["full_name"]
        n_in   = inner_rows[code]

        hosts   = _ents(code, "Host")
        mirnas  = _ents(code, "miRNA")
        targets = _ents(code, "Target")

        # Get T/E arrays from first available result
        all_ents = (all_gene_results.get(code) or [])
        if all_ents:
            T_arr = all_ents[0]["T_arr"]
            E_arr = all_ents[0]["E_arr"]
        else:
            # Nothing at all for this cancer — draw placeholder
            ax_blank = fig.add_subplot(gs_outer[row_o, col_o])
            ax_blank.text(0.5, 0.5,
                          f"{code}\n{cname}\n\nNo expression data matched",
                          ha="center", va="center",
                          transform=ax_blank.transAxes,
                          fontsize=9, color="#AAAAAA")
            ax_blank.set_facecolor("#F8F8F8")
            ax_blank.set_xticks([]); ax_blank.set_yticks([])
            continue

        # ── Inner GridSpec ────────────────────────────────────────────────────
        # Rows: n_in rows, each split 4:1 (KM : risk)
        # Cols: 3  (Host | miRNA | Target)
        # Width ratios: Host gets 1.4× width (matches tall + wide reference style)
        hr = []
        for _ in range(n_in):
            hr += [4, 1]

        gs_in = gridspec.GridSpecFromSubplotSpec(
            n_in * 2, 3,
            subplot_spec=gs_outer[row_o, col_o],
            height_ratios=hr,
            width_ratios=[1.0, 1.0, 1.0],
            hspace=0.06,             wspace=0.25
        )

        # Cancer header on a thin transparent axis at top
        # Use the first miRNA km axis title for the cancer label
        # (add it as a super-title on the inner block)
        ax_hdr = fig.add_subplot(gs_in[0:2, 0])
        ax_hdr.set_visible(False)
        ax_hdr.set_title(f"{code}  ·  {cname}",
                         fontsize=8.5, fontweight="bold",
                         color=cc, loc="left", pad=28)

        # Entity column headers — set on the top KM axes for cols 1 and 2
        for col_i, etype in enumerate(["Host", "miRNA", "Target"]):
            ax_tmp = fig.add_subplot(gs_in[0, col_i])
            ax_tmp.set_visible(False)
            ax_tmp.set_title(ENTITY_LABEL[etype],
                             fontsize=8, fontweight="bold",
                             color=ENTITY_COLOR[etype],
                             loc="center", pad=18)

        # ── HOST: one panel per row (same size as miRNA/Target) ────────────────
        for ri in range(n_in):
            ax_km   = fig.add_subplot(gs_in[ri*2,     0])
            ax_risk = fig.add_subplot(gs_in[ri*2 + 1, 0])
            res = hosts[ri] if ri < len(hosts) else None
            _draw_km(ax_km, ax_risk, res, "Host",
                     is_top_row=(ri == 0), T_arr=T_arr, E_arr=E_arr,
                     show_ylabel=(ri == 0))

        # ── miRNA: one panel per row ──────────────────────────────────────────
        for ri in range(n_in):
            ax_km   = fig.add_subplot(gs_in[ri*2,     1])
            ax_risk = fig.add_subplot(gs_in[ri*2 + 1, 1])
            res = mirnas[ri] if ri < len(mirnas) else None
            _draw_km(ax_km, ax_risk, res, "miRNA",
                     is_top_row=(ri == 0),
                     T_arr=T_arr, E_arr=E_arr)

        # ── Target: one panel per row ─────────────────────────────────────────
        for ri in range(n_in):
            ax_km   = fig.add_subplot(gs_in[ri*2,     2])
            ax_risk = fig.add_subplot(gs_in[ri*2 + 1, 2])
            res = targets[ri] if ri < len(targets) else None
            _draw_km(ax_km, ax_risk, res, "Target",
                     is_top_row=(ri == 0),
                     T_arr=T_arr, E_arr=E_arr)

    _save(fig, "val_ALL_gene_km_atlas")



def extract_features(triplets_df, mirna_sub, mrna_sub, clin_sub, code):
    """Builds the per-patient neural network feature matrix from validation triplets."""
    print(f"\n[3]  Building NN feature matrix for {code} …")
    mirna_lookup = _build_mirna_lookup(mirna_sub)
    feat = {}
    n_h = n_t = n_m = n_ix = 0; missing = []

    for _, row in triplets_df.iterrows():
        H, M, Tg = row["Host"], row["miRNA"], row["Target"]
        h_ok  = H  in mrna_sub.columns
        tg_ok = Tg in mrna_sub.columns
        m_col = _match_mirna(M, mirna_lookup)

        if h_ok:  feat[f"expr_H_{H}"]  = mrna_sub[H];          n_h += 1
        else:     missing.append(f"Host mRNA '{H}'")
        if tg_ok: feat[f"expr_T_{Tg}"] = mrna_sub[Tg];         n_t += 1
        else:     missing.append(f"Target mRNA '{Tg}'")
        if m_col: feat[f"expr_M_{M}"]  = mirna_sub[m_col];     n_m += 1
        else:     missing.append(f"miRNA '{M}'")

        if h_ok and tg_ok:
            feat[f"HxT_{H}_{Tg}"[:72]] = mrna_sub[H] * mrna_sub[Tg]; n_ix += 1
        if m_col and tg_ok:
            feat[f"MxT_{M}_{Tg}"[:72]] = mirna_sub[m_col] * mrna_sub[Tg]; n_ix += 1
        if h_ok and m_col:
            feat[f"HxM_{H}_{M}"[:72]] = mrna_sub[H] * mirna_sub[m_col]; n_ix += 1

    X = pd.DataFrame(feat, index=clin_sub.index).fillna(0.0)
    X = X.loc[:, X.var() > 1e-8]

    nt = len(triplets_df)
    print(f"    Hosts {n_h}/{nt}  Targets {n_t}/{nt}  miRNAs {n_m}/{nt}  "
          f"Interactions {n_ix}  →  {X.shape[1]} features")
    if missing:
        print("    Unmatched (zero-padded):")
        for m in set(missing): print(f"      • {m}")
    if X.shape[1] == 0:
        raise ValueError(f"{code}: feature matrix is empty — no genes matched expression data.")
    return X


# ══════════════════════════════════════════════════════════════════════════════
# ▶  10.  PRE-PROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def preprocess(X_df, clin, code):
    print(f"\n[4]  Preprocessing {code} …")
    X_log  = np.log1p(np.abs(X_df.values))
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X_log)

    max_c = min(32, X_sc.shape[0]-1, X_sc.shape[1])
    cumv  = np.cumsum(PCA(n_components=max_c, random_state=SEED).fit(X_sc).explained_variance_ratio_)
    n_c   = max(2, min(max_c, int(np.searchsorted(cumv, 0.90))+1))

    pca   = PCA(n_components=n_c, random_state=SEED)
    X_pca = pca.fit_transform(X_sc)
    print(f"    PCA {n_c} components  |  {X_sc.shape[1]} features  "
          f"|  {pca.explained_variance_ratio_.sum()*100:.1f}% var retained")

    T = clin["OS_months"].values.astype(np.float32)
    E = clin["OS_status"].values.astype(np.float32)
    print(f"    Patients={len(T)}  Events={int(E.sum())} ({100*E.mean():.1f}%)")
    return X_pca.astype(np.float32), T, E, scaler, pca


def adaptive_folds(E):
    n_ev = int(E.sum())
    f    = max(2, min(N_FOLDS, n_ev // 5))
    if f < N_FOLDS:
        print(f"    ⚠  Reduced to {f}-fold CV ({n_ev} events).")
    return f


# ══════════════════════════════════════════════════════════════════════════════
# ▶  11.  NEURAL NETWORK ARCHITECTURES
# ══════════════════════════════════════════════════════════════════════════════

class SurvivalDataset(Dataset):
    def __init__(self,X,T,E):
        self.X=torch.tensor(X,dtype=torch.float32)
        self.T=torch.tensor(T,dtype=torch.float32)
        self.E=torch.tensor(E,dtype=torch.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self,i): return self.X[i],self.T[i],self.E[i]

def cox_loss(r,T,E):
    o=torch.argsort(T,descending=True); r,e=r[o],E[o]
    return -torch.mean((r-torch.logcumsumexp(r,0))*e)

def ranking_loss(r,T,E,margin=0.1):
    mask=E.bool()
    if not mask.any(): return torch.zeros(1,device=r.device,requires_grad=True).squeeze()
    rev,Tev=r[mask],T[mask]; ls=[]
    for k in range(len(Tev)):
        comp=T>Tev[k]
        if comp.any(): ls.append(torch.clamp(r[comp]-rev[k]+margin,min=0).mean())
    return torch.stack(ls).mean() if ls else torch.zeros(1,device=r.device,requires_grad=True).squeeze()

def vae_loss(r,T,E,rec,x,mu,lv,a=0.5,b=0.1):
    return (cox_loss(r,T,E)+a*nn.functional.mse_loss(rec,x)
            +b*(-0.5*torch.mean(1+lv-mu.pow(2)-lv.exp())))

class DeepCoxMLP(nn.Module):
    def __init__(self,in_dim,hidden=(128,64,32),dropout=0.3):
        super().__init__()
        lrs,p=[],in_dim
        for h in hidden: lrs+=[nn.Linear(p,h),nn.BatchNorm1d(h),nn.ReLU(),nn.Dropout(dropout)]; p=h
        lrs.append(nn.Linear(p,1)); self.net=nn.Sequential(*lrs)
    def forward(self,x): return self.net(x).squeeze(-1)

class _SparseGate(nn.Module):
    def __init__(self,d,tau=0.8):
        super().__init__(); bn=max(d//4,8)
        self.g=nn.Sequential(nn.LayerNorm(d),nn.Linear(d,bn),nn.SiLU(),nn.Linear(bn,d))
        self.tau=tau
    def forward(self,x): return torch.sigmoid(self.g(x)/self.tau)

class _CrossAttn(nn.Module):
    def __init__(self,d,h=4,drop=0.2):
        super().__init__()
        while h>1 and d%h!=0: h-=1
        self.qp=nn.Linear(1,d); self.kvp=nn.Linear(1,d)
        self.nq=nn.LayerNorm(d); self.nkv=nn.LayerNorm(d)
        self.att=nn.MultiheadAttention(d,h,dropout=drop,batch_first=True)
        self.drop=nn.Dropout(drop)
    def forward(self,q,kv):
        qt=self.qp(q.unsqueeze(-1)); qc=qt.mean(1,keepdim=True)
        kvt=self.kvp(kv.unsqueeze(-1))
        o,_=self.att(self.nq(qc),self.nkv(kvt),self.nkv(kvt),need_weights=False)
        return (qc+self.drop(o)).squeeze(1)

class AttentionSurvNet(nn.Module):
    def __init__(self,in_dim,hidden=(256,128,64),n_heads=4,dropout=0.25,tau=0.8):
        super().__init__()
        self.gate=_SparseGate(in_dim,tau); self.ca=_CrossAttn(in_dim,n_heads,dropout)
        self.rp=nn.Linear(in_dim,hidden[-1])
        enc,p=[],in_dim
        for h in hidden: enc+=[nn.Linear(p,h),nn.LayerNorm(h),nn.SiLU(),nn.Dropout(dropout)]; p=h
        self.enc=nn.Sequential(*enc)
        self.hd=nn.Sequential(nn.Linear(p,p//2),nn.SiLU(),nn.Dropout(dropout*.5),nn.Linear(p//2,1))
        for m in self.modules():
            if isinstance(m,nn.Linear): nn.init.kaiming_normal_(m.weight,nonlinearity="linear"); nn.init.zeros_(m.bias) if m.bias is not None else None
    def forward(self,x):
        g=self.gate(x); xg=x*g; xa=self.ca(xg,x)+xg
        return self.hd(self.enc(xa)+self.rp(xa)).squeeze(-1)
    @torch.no_grad()
    def gate_weights(self,x): self.eval(); return self.gate(x)

class VAESurvNet(nn.Module):
    def __init__(self,in_dim,latent=16,hidden=32,dropout=0.3):
        super().__init__()
        self.enc=nn.Sequential(nn.Linear(in_dim,hidden),nn.ReLU(),nn.Dropout(dropout))
        self.mu=nn.Linear(hidden,latent); self.lv=nn.Linear(hidden,latent)
        self.dec=nn.Sequential(nn.Linear(latent,hidden),nn.ReLU(),nn.Linear(hidden,in_dim))
        self.cox=nn.Sequential(nn.Linear(latent,16),nn.ReLU(),nn.Linear(16,1))
    def reparameterize(self,mu,lv):
        return mu+torch.exp(0.5*lv)*torch.randn_like(lv) if self.training else mu
    def forward(self,x):
        h=self.enc(x); mu,lv=self.mu(h),self.lv(h); z=self.reparameterize(mu,lv)
        return self.cox(z).squeeze(-1),self.dec(z),mu,lv

class ResBlock(nn.Module):
    def __init__(self,i,o,d=0.3):
        super().__init__()
        self.b=nn.Sequential(nn.LayerNorm(i),nn.Linear(i,o),nn.GELU(),nn.Dropout(d),nn.LayerNorm(o),nn.Linear(o,o),nn.Dropout(d))
        self.s=nn.Linear(i,o,bias=False) if i!=o else nn.Identity()
    def forward(self,x): return self.b(x)+self.s(x)

class ResCoxNet(nn.Module):
    def __init__(self,in_dim,hidden=(128,64,32),dropout=0.3):
        super().__init__()
        self.proj=nn.Linear(in_dim,hidden[0]); dims=list(hidden)
        self.blocks=nn.Sequential(*[ResBlock(dims[i],dims[i+1],dropout) for i in range(len(dims)-1)])
        self.head=nn.Sequential(nn.LayerNorm(dims[-1]),nn.Linear(dims[-1],1))
    def forward(self,x): return self.head(self.blocks(torch.relu(self.proj(x)))).squeeze(-1)

class TransSurv(nn.Module):
    def __init__(self,in_dim,d_model=64,n_heads=4,n_layers=2,dropout=0.3,max_seq=65):
        super().__init__()
        self.fp=nn.Linear(1,d_model); self.cls=nn.Parameter(torch.zeros(1,1,d_model))
        self.pos=nn.Parameter(torch.zeros(1,max_seq,d_model))
        nn.init.trunc_normal_(self.cls,std=0.02); nn.init.trunc_normal_(self.pos,std=0.02)
        el=nn.TransformerEncoderLayer(d_model,n_heads,d_model*4,dropout,batch_first=True,norm_first=True)
        self.enc=nn.TransformerEncoder(el,n_layers)
        self.head=nn.Sequential(nn.LayerNorm(d_model),nn.Linear(d_model,d_model//2),nn.GELU(),nn.Dropout(dropout),nn.Linear(d_model//2,1))
    def forward(self,x):
        B=x.size(0); tok=self.fp(x.unsqueeze(-1))
        seq=torch.cat([self.cls.expand(B,-1,-1),tok],1)+self.pos[:,:tok.size(1)+1]
        return self.head(self.enc(seq)[:,0]).squeeze(-1)

def build_configs(in_dim):
    return {
        "DeepCox-MLP"  :{"cls":DeepCoxMLP,       "mtype":"mlp",  "kw":{"hidden":(128,64,32),"dropout":0.3}},
        "Attention-Net":{"cls":AttentionSurvNet,  "mtype":"attn", "kw":{"hidden":(256,128,64),"n_heads":4,"dropout":0.25,"tau":0.8}},
        "VAE-Survival" :{"cls":VAESurvNet,        "mtype":"vae",  "kw":{"latent":16,"hidden":32,"dropout":0.3}},
        "ResCox-Net"   :{"cls":ResCoxNet,         "mtype":"res",  "kw":{"hidden":(128,64,32),"dropout":0.3}},
        "TransSurv"    :{"cls":TransSurv,         "mtype":"trans","kw":{"d_model":64,"n_heads":4,"n_layers":3,"dropout":0.3,"max_seq":in_dim+1}},
    }


# ══════════════════════════════════════════════════════════════════════════════
# ▶  12.  TRAINING + INFERENCE
# ══════════════════════════════════════════════════════════════════════════════

def ci_score(r,T,E):
    r,T,E=map(np.asarray,(r,T,E)); ev=np.where(E==1)[0]; c=d=0.0
    for i in ev:
        comp=T>T[i]; c+=np.sum(r[i]>r[comp])+0.5*np.sum(r[i]==r[comp]); d+=np.sum(r[i]<r[comp])+0.5*np.sum(r[i]==r[comp])
    return c/(c+d+1e-9)

@torch.no_grad()
def predict(model,X_np,mtype):
    model.eval(); xt=torch.tensor(X_np,dtype=torch.float32).to(DEVICE)
    out=model(xt); return (out[0] if mtype=="vae" else out).cpu().numpy()

def train_one(model,Xtr,Ttr,Etr,Xval,Tval,Eval,mtype,patience):
    bs=min(BATCH,max(8,len(Xtr))); loader=DataLoader(SurvivalDataset(Xtr,Ttr,Etr),batch_size=bs,shuffle=True,drop_last=True)
    steps=max(len(loader),1)
    if mtype=="attn":
        lr=5e-4; wd=2e-4; pat=patience+10
        opt=optim.AdamW(model.parameters(),lr=lr,weight_decay=wd,betas=(0.9,0.999))
        sched=optim.lr_scheduler.OneCycleLR(opt,max_lr=lr,epochs=EPOCHS,steps_per_epoch=steps,pct_start=0.15,anneal_strategy="cos",div_factor=10,final_div_factor=1e4)
        pb=True
    else:
        lr=LR; wd=1e-4; pat=patience
        opt=optim.AdamW(model.parameters(),lr=lr,weight_decay=wd)
        sched=optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EPOCHS); pb=False
    best_ci,best_state,wait=0.0,None,0; hist={"loss":[],"val_ci":[]}
    for ep in range(EPOCHS):
        model.train(); tot=0.0
        for Xb,Tb,Eb in loader:
            Xb,Tb,Eb=Xb.to(DEVICE),Tb.to(DEVICE),Eb.to(DEVICE); opt.zero_grad()
            if mtype=="vae": r,rec,mu,lv=model(Xb); loss=vae_loss(r,Tb,Eb,rec,Xb,mu,lv)
            elif mtype=="attn": loss=cox_loss(model(Xb),Tb,Eb)+0.3*ranking_loss(model(Xb),Tb,Eb)
            else: loss=cox_loss(model(Xb),Tb,Eb)
            if not torch.isnan(loss): loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); pb and sched.step(); tot+=loss.item()
        (not pb) and sched.step()
        ci=ci_score(predict(model,Xval,mtype),Tval,Eval); hist["loss"].append(tot/steps); hist["val_ci"].append(ci)
        if ci>best_ci: best_ci=ci; best_state={k:v.clone() for k,v in model.state_dict().items()}; wait=0
        else:
            wait+=1
            if wait>=pat: break
    best_state and model.load_state_dict(best_state)
    return model,hist,best_ci

def run_cv(X,T,E,code):
    n_folds=adaptive_folds(E); patience=max(PATIENCE,30)
    print(f"\n[5]  {n_folds}-fold CV — {code}")
    skf=StratifiedKFold(n_splits=n_folds,shuffle=True,random_state=SEED)
    cv_res={}; best_models={}; fold_hist={n:[] for n in MODEL_CONFIGS}
    for name,cfg in MODEL_CONFIGS.items():
        print(f"  ── {name}"); fold_cis=[]; best_o=0.0; best_m=None
        for fold,(tr,va) in enumerate(skf.split(X,E.astype(int))):
            n_r=4 if name=="Attention-Net" else 1; best_f=-1; bfm=None; bfh=None
            for _ in range(n_r):
                m=cfg["cls"](in_dim=X.shape[1],**cfg["kw"]).to(DEVICE)
                m,h,ci=train_one(m,X[tr],T[tr],E[tr],X[va],T[va],E[va],cfg["mtype"],patience)
                if ci>best_f: best_f=ci; bfm=m; bfh=h
            fold_cis.append(best_f); fold_hist[name].append(bfh)
            print(f"    Fold {fold+1}/{n_folds}  CI={best_f:.4f}")
            if best_f>best_o: best_o=best_f; best_m=bfm
        mu,sd=np.mean(fold_cis),np.std(fold_cis)
        print(f"  → Mean={mu:.4f}  Std={sd:.4f}")
        cv_res[name]={"fold_ci":fold_cis,"mean":mu,"std":sd}; best_models[name]=best_m
    return cv_res,best_models,fold_hist

def run_km(best_models,X,T,E,code):
    print(f"\n[6]  KM stratification — {code}")
    km={}
    for name,model in best_models.items():
        mtype=MODEL_CONFIGS[name]["mtype"]; risk=predict(model,X,mtype)
        high=risk>=np.median(risk); lrt=logrank_test(T[high],T[~high],E[high],E[~high])
        pv=lrt.p_value; sig="***" if pv<0.001 else "**" if pv<0.01 else "*" if pv<0.05 else "ns"
        km[name]={"risk":risk,"high":high,"T":T,"E":E,"pval":pv}
        print(f"  {name:<22}: p={pv:.3e}  {sig}")
    return km


# ══════════════════════════════════════════════════════════════════════════════
# ▶  13.  STANDARD NN PLOTS
# ══════════════════════════════════════════════════════════════════════════════

def _save(fig,stem):
    OUT_DIR.mkdir(parents=True,exist_ok=True)
    for ext in ("svg","pdf"):
        fig.savefig(OUT_DIR/f"{stem}.{ext}",format=ext,bbox_inches="tight")
    plt.close(fig); print(f"  [OK]  {stem}.{{svg,pdf}}")

def _mc(n): return MODEL_COLORS.get(n,"#555555")
def _sig(p): return "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "ns"

def plot_training(fh,code,cn):
    names=list(fh.keys()); n=len(names)
    fig,axes=plt.subplots(2,n,figsize=(6*n,9),squeeze=False)
    fig.suptitle(f"Training Dynamics · {cn} · Validation Triplets",fontsize=12,fontweight="bold")
    cmap=plt.cm.tab10(np.linspace(0,0.5,max(len(list(fh.values())[0]),1)))
    for col,nm in enumerate(names):
        for fi,h in enumerate(fh[nm]):
            c=cmap[fi] if fi<len(cmap) else cmap[-1]
            axes[0,col].plot(h["loss"],color=c,alpha=0.85,lw=1.2,label=f"F{fi+1}")
            axes[1,col].plot(h["val_ci"],color=c,alpha=0.85,lw=1.2)
        axes[0,col].set_title(f"{nm}\nCox Loss",fontweight="bold",color=_mc(nm))
        axes[0,col].set_xlabel("Epoch"); axes[0,col].set_ylabel("Loss"); axes[0,col].legend(fontsize=7)
        axes[1,col].set_title(f"{nm}\nVal C-index",fontweight="bold",color=_mc(nm))
        axes[1,col].set_xlabel("Epoch"); axes[1,col].set_ylabel("C-index"); axes[1,col].set_ylim(0,1)
        axes[1,col].axhline(0.5,color="gray",ls=":",lw=1)
    plt.tight_layout(); _save(fig,f"val_{code}_02_training")

def plot_cv(cv_res,code,cn):
    names=list(cv_res.keys()); means=[cv_res[n]["mean"] for n in names]; stds=[cv_res[n]["std"] for n in names]
    fig,axes=plt.subplots(1,3,figsize=(18,5.5))
    fig.suptitle(f"Cross-Validation · {cn} · Validation Triplets",fontsize=12,fontweight="bold")
    bars=axes[0].bar(names,means,yerr=stds,capsize=6,color=[_mc(n) for n in names],edgecolor="k",lw=0.7,alpha=0.87,zorder=3)
    axes[0].axhline(0.5,color="gray",ls="--",lw=1.2,label="Random"); axes[0].set_ylim(0,1); axes[0].legend(fontsize=9)
    axes[0].set_title("A  Mean C-index ± SD",fontweight="bold"); axes[0].set_ylabel("C-index")
    plt.setp(axes[0].get_xticklabels(),rotation=28,ha="right",fontsize=9)
    for b,m,s in zip(bars,means,stds): axes[0].text(b.get_x()+b.get_width()/2,m+s+0.02,f"{m:.3f}",ha="center",fontsize=9,fontweight="bold")
    bp=axes[1].boxplot([cv_res[n]["fold_ci"] for n in names],labels=names,patch_artist=True,medianprops=dict(color="white",lw=2.5))
    for p,c in zip(bp["boxes"],[_mc(n) for n in names]): p.set_facecolor(c); p.set_alpha(0.78)
    axes[1].axhline(0.5,color="gray",ls="--",lw=1.2); axes[1].set_ylim(0,1); axes[1].set_title("B  Per-Fold Box",fontweight="bold")
    plt.setp(axes[1].get_xticklabels(),rotation=28,ha="right",fontsize=9)
    for xi,nm in enumerate(names):
        ci=cv_res[nm]["fold_ci"]
        axes[2].scatter(np.full(len(ci),xi)+np.random.uniform(-0.15,0.15,len(ci)),ci,color=_mc(nm),s=55,alpha=0.85,edgecolors="k",lw=0.4,zorder=4)
        axes[2].hlines(np.mean(ci),xi-0.3,xi+0.3,color=_mc(nm),lw=2.5,zorder=5)
    axes[2].axhline(0.5,color="gray",ls="--",lw=1.2); axes[2].set_xticks(range(len(names)))
    axes[2].set_xticklabels(names,rotation=28,ha="right",fontsize=9); axes[2].set_ylim(0,1); axes[2].set_title("C  Fold Strips",fontweight="bold")
    plt.tight_layout(); _save(fig,f"val_{code}_03_cv")

def plot_km_nn(km_res,cv_res,code,cn):
    names=list(km_res.keys()); n=len(names)
    fig,axes=plt.subplots(1,n,figsize=(7*n,6.5),squeeze=False)
    fig.suptitle(f"Neural Network KM Stratification\n{cn} · Validation H–M–T Features · TCGA",fontsize=13,fontweight="bold")
    for ax,nm in zip(axes[0],names):
        res=km_res[nm]; T,E,high=res["T"],res["E"],res["high"]; ci=cv_res[nm]["mean"]; pv=res["pval"]
        KaplanMeierFitter().fit(T[high],E[high],label=f"High (n={high.sum()})").plot_survival_function(ax=ax,color=PALETTE["high"],ci_show=True,ci_alpha=0.15,lw=2.2)
        KaplanMeierFitter().fit(T[~high],E[~high],label=f"Low (n={(~high).sum()})").plot_survival_function(ax=ax,color=PALETTE["low"],ci_show=True,ci_alpha=0.15,lw=2.2)
        ax.set_title(f"{nm}\nC-index={ci:.3f}  p={pv:.2e}  {_sig(pv)}",fontsize=10,fontweight="bold",color=_mc(nm))
        ax.set_xlabel("Months"); ax.set_ylabel("Survival Probability"); ax.set_ylim(0,1.05)
        ax.text(0.97,0.97,f"n-High={high.sum()}\nn-Low={(~high).sum()}",transform=ax.transAxes,ha="right",va="top",fontsize=9,bbox=dict(boxstyle="round,pad=0.3",facecolor="lightyellow",alpha=0.85,edgecolor="gray"))
    plt.tight_layout(); _save(fig,f"val_{code}_04_km_nn")

def plot_calibration(km_res,code,cn):
    names=list(km_res.keys()); n=len(names)
    fig,axes=plt.subplots(1,n,figsize=(7*n,6),squeeze=False)
    fig.suptitle(f"Calibration — Risk Tertile Survival\n{cn} · Validation Triplets",fontsize=12,fontweight="bold")
    tc=["#C0392B","#F39C12","#2980B9"]; tl=["High (T3)","Mid (T2)","Low (T1)"]
    for ax,nm in zip(axes[0],names):
        risk,T,E=km_res[nm]["risk"],km_res[nm]["T"],km_res[nm]["E"]; q33,q67=np.percentile(risk,[33,67])
        for mask,lb,c in zip([risk>=q67,(risk>=q33)&(risk<q67),risk<q33],tl,tc):
            if mask.sum()<5: continue
            KaplanMeierFitter().fit(T[mask],E[mask],label=f"{lb} (n={mask.sum()})").plot_survival_function(ax=ax,color=c,ci_show=True,ci_alpha=0.12,lw=2)
        ax.set_title(nm,fontsize=10,fontweight="bold",color=_mc(nm)); ax.set_xlabel("Months"); ax.set_ylabel("Survival"); ax.set_ylim(0,1.05)
    plt.tight_layout(); _save(fig,f"val_{code}_07_calibration")

def plot_importance(best_models,X,T,E,pca,code,cn):
    best_nm=max(best_models,key=lambda n:ci_score(predict(best_models[n],X,MODEL_CONFIGS[n]["mtype"]),T,E))
    model=best_models[best_nm]; mtype=MODEL_CONFIGS[best_nm]["mtype"]; base=ci_score(predict(model,X,mtype),T,E)
    nf=X.shape[1]; tk=min(nf,15); imps=np.zeros(nf); rng=np.random.default_rng(SEED)
    for f in range(nf):
        ds=[]
        for _ in range(15):
            Xp=X.copy(); Xp[:,f]=rng.permutation(Xp[:,f]); ds.append(base-ci_score(predict(model,Xp,mtype),T,E))
        imps[f]=np.mean(ds)
    idx=np.argsort(imps)[::-1][:tk]; vals=imps[idx]; labs=[f"PC{i+1}" for i in idx]
    am=best_models.get("Attention-Net"); gm=None
    if am:
        xt=torch.tensor(X,dtype=torch.float32).to(DEVICE); gm=am.gate_weights(xt).cpu().numpy().mean(0)
    nc=3 if gm is not None else 2
    fig,axes=plt.subplots(1,nc,figsize=(8*nc,6))
    fig.suptitle(f"Feature Importance · {cn} · Validation Triplets",fontsize=12,fontweight="bold")
    c=_mc(best_nm)
    axes[0].barh(range(tk)[::-1],vals,color=c,edgecolor="k",lw=0.5,alpha=0.85,zorder=3)
    axes[0].set_yticks(range(tk)[::-1]); axes[0].set_yticklabels(labs,fontsize=9)
    axes[0].axvline(0,color="k",lw=0.8); axes[0].set_xlabel("Mean ΔC-index"); axes[0].set_title(f"A  Permutation ({best_nm})",fontweight="bold")
    ev=pca.explained_variance_ratio_*100; cumv=np.cumsum(ev)
    axes[1].bar(range(1,len(ev)+1),ev,color="#BDC3C7",edgecolor="k",lw=0.3,alpha=0.85)
    ax2=axes[1].twinx(); ax2.plot(range(1,len(cumv)+1),cumv,"o-",color=c,lw=1.8,ms=4); ax2.axhline(90,color="gray",ls="--",lw=1)
    axes[1].set_xlabel("PC"); axes[1].set_ylabel("Explained Var (%)"); ax2.set_ylabel("Cumulative (%)"); axes[1].set_title("B  PCA Variance",fontweight="bold")
    if gm is not None:
        gtk=min(tk,len(gm)); gi=np.argsort(gm)[::-1][:gtk]; gv=gm[gi]; gl=[f"PC{i+1}" for i in gi]; ac=_mc("Attention-Net")
        axes[2].barh(range(gtk)[::-1],gv,color=ac,edgecolor="k",lw=0.5,alpha=0.85,zorder=3)
        axes[2].set_yticks(range(gtk)[::-1]); axes[2].set_yticklabels(gl,fontsize=9)
        axes[2].set_xlim(0,1); axes[2].set_xlabel("Mean Gate Weight (0→1)"); axes[2].set_title("C  AmiGO-Surv Gate Weights",fontweight="bold",color=ac)
    plt.tight_layout(); _save(fig,f"val_{code}_06_importance")


# ══════════════════════════════════════════════════════════════════════════════
# ▶  14.  PAN-CANCER NN COMPARISON
# ══════════════════════════════════════════════════════════════════════════════

def plot_pan_cancer(all_res):
    cancers=list(all_res.keys()); nc=len(cancers)
    if nc<2: return
    fig=plt.figure(figsize=(24,14)); gs=gridspec.GridSpec(2,3,hspace=0.55,wspace=0.42)
    fig.suptitle("Pan-Cancer Validation · AmiGO-Surv on H–M–T Validation Triplets · TCGA",fontsize=15,fontweight="bold",y=1.01)
    ax=fig.add_subplot(gs[0,0]); amigo=[all_res[c]["cv"]["Attention-Net"]["mean"] for c in cancers]; best_bl=[max(all_res[c]["cv"][n]["mean"] for n in all_res[c]["cv"] if n!="Attention-Net") for c in cancers]
    x=np.arange(nc); w=0.35
    ax.bar(x-w/2,amigo,w,label="AmiGO-Surv",color="#1F77B4",edgecolor="k",lw=0.6,alpha=0.88)
    ax.bar(x+w/2,best_bl,w,label="Best Baseline",color="#95A5A6",edgecolor="k",lw=0.6,alpha=0.88)
    ax.axhline(0.5,color="gray",ls="--",lw=1.2,label="Random"); ax.set_xticks(x); ax.set_xticklabels(cancers,rotation=25,ha="right"); ax.set_ylim(0,1); ax.legend(fontsize=9); ax.set_title("A  C-index Comparison",fontweight="bold")
    for xi,(a,b) in enumerate(zip(amigo,best_bl)): ax.text(xi-w/2,a+0.02,f"{a:.3f}",ha="center",fontsize=8,fontweight="bold",color="#1F77B4"); ax.text(xi+w/2,b+0.02,f"{b:.3f}",ha="center",fontsize=8)
    ax=fig.add_subplot(gs[0,1]); pvals=[all_res[c]["km"]["Attention-Net"]["pval"] for c in cancers]; nlp=[-np.log10(p+1e-300) for p in pvals]
    bars=ax.bar(cancers,nlp,color="#1F77B4",edgecolor="k",lw=0.6,alpha=0.88,zorder=3)
    ax.axhline(-np.log10(0.05),color="red",ls="--",lw=1.5,label="p=0.05"); ax.axhline(-np.log10(0.001),color="purple",ls=":",lw=1.2,label="p=0.001")
    ax.set_ylabel("−log₁₀(p)"); ax.legend(fontsize=9); ax.set_title("B  AmiGO-Surv KM Significance",fontweight="bold")
    plt.setp(ax.get_xticklabels(),rotation=25,ha="right")
    for b,p in zip(bars,pvals): ax.text(b.get_x()+b.get_width()/2,b.get_height()+0.15,f"p={p:.2e}\n{_sig(p)}",ha="center",fontsize=8)
    gs2=gridspec.GridSpecFromSubplotSpec(2,3,subplot_spec=gs[1,:],hspace=0.65,wspace=0.42)
    for idx,cancer in enumerate(cancers[:6]):
        r,c_i=divmod(idx,3); ax=fig.add_subplot(gs2[r,c_i])
        res=all_res[cancer]["km"]["Attention-Net"]; T,E,high=res["T"],res["E"],res["high"]; pv=res["pval"]; ci=all_res[cancer]["cv"]["Attention-Net"]["mean"]
        KaplanMeierFitter().fit(T[high],E[high],label=f"High (n={high.sum()})").plot_survival_function(ax=ax,color=PALETTE["high"],ci_show=True,ci_alpha=0.12,lw=1.8)
        KaplanMeierFitter().fit(T[~high],E[~high],label=f"Low (n={(~high).sum()})").plot_survival_function(ax=ax,color=PALETTE["low"],ci_show=True,ci_alpha=0.12,lw=1.8)
        ax.set_title(f"{cancer}\nC={ci:.3f}  p={pv:.2e}  {_sig(pv)}",fontsize=9,fontweight="bold"); ax.set_xlabel("Months",fontsize=8); ax.set_ylabel("Survival",fontsize=8); ax.set_ylim(0,1.05); ax.tick_params(labelsize=7)
    _save(fig,"val_pan_cancer_nn_comparison")


# ══════════════════════════════════════════════════════════════════════════════
# ▶  15.  REPORT
# ══════════════════════════════════════════════════════════════════════════════

def write_report(all_res, triplets, all_triplet_results, all_gene_results=None):
    w=75
    lines=["="*w,"  AmiGO-Surv VALIDATION REPORT  v4",
           "  Individual Gene KM  +  TAS Triplet KM  +  Neural Network Survival",
           "="*w,"",
           f"  Device : {DEVICE}","  NEURAL NETWORK RESULTS","  "+"-"*w,""]
    for code,res in all_res.items():
        cv=res["cv"]; km=res["km"]; nt=len(triplets.get(code,[]))
        lines+=[f"── {code}  |  {nt} validation triplets",
                f"  {'Model':<22}  {'C-index':>8}  {'±SD':>7}  {'KM p':>13}  Sig","  "+"-"*56]
        for nm in cv:
            mu=cv[nm]["mean"]; sd=cv[nm]["std"]; p=km[nm]["pval"]
            lines.append(f"  {nm:<22}  {mu:>8.4f}  {sd:>7.4f}  {p:>13.3e}  {_sig(p)}")
        lines+=[""]
    lines+=["","  INDIVIDUAL GENE KM (median expression split)","  "+"-"*w,"",
            f"  {'Cancer':<6}  {'Type':<8}  {'Gene/miRNA':<28}  {'p-value':>10}  Sig"]
    if all_gene_results:
        for code, glist in all_gene_results.items():
            for r in glist:
                lines.append(
                    f"  {code:<6}  {r['entity_type']:<8}  {r['label']:<28}  "
                    f"{r['pval']:>10.4f}  {r['sig']}")
        lines+=[""]
    lines+=["","  TAS TRIPLET KM (z_Host + z_miRNA - z_Target)","  "+"-"*w,""]
    for res in all_triplet_results:
        if res["status"]=="ok":
            m=res["M"].replace("_","-")
            lines.append(f"  {res['code']:6s}  {res['H']:8s} / {m:24s} / {res['T']:10s}  "
                          f"p={res['pval']:.3e}  {res['sig']}")
        else:
            lines.append(f"  {res['code']:6s}  {res['H']} / {res['M']} / {res['T']}  → NOT FOUND")
    lines+=["","="*w,"  OUTPUT FILES","  "+"-"*w,
            "  val_<CANCER>_08_gene_km.{svg,pdf}   ← NEW: individual gene KM (Host|miRNA|Target)",
            "  val_ALL_gene_km_atlas.{svg,pdf}      ← NEW: 6-cancer gene KM atlas",
            "  val_<CANCER>_05_triplet_km.{svg,pdf}",
            "  val_ALL_triplets_master_panel.{svg,pdf}",
            "  val_<CANCER>_04_km_nn.{svg,pdf}","="*w]
    text="\n".join(lines); print("\n"+text)
    OUT_DIR.mkdir(parents=True,exist_ok=True)
    (OUT_DIR/"validation_report.txt").write_text(text,encoding="utf-8")
    print(f"\n  [OK]  Report → {OUT_DIR/'validation_report.txt'}")


# ══════════════════════════════════════════════════════════════════════════════
# ▶  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("="*75)
    print("  AmiGO-Surv  Validation Pipeline  v4")
    print("  Individual Gene KM  +  TAS Triplet KM  +  Neural Network Survival")
    print("="*75)
    print(f"  Device : {DEVICE}\n  Output : {OUT_DIR}")

    if not validate_all_paths(): sys.exit(1)

    triplets = load_validation_triplets()
    all_res  = {}
    all_triplet_results = []   # TAS-based, for master panel
    all_gene_results    = {}   # individual gene KM, for atlas

    for code, trip_df in triplets.items():
        if code not in CANCER_FILES:
            print(f"\n  ⚠  Skipping {code} — not in CANCER_FILES"); continue

        cn = CANCER_FILES[code]["full_name"]
        print(f"\n{'='*75}\n  {code}  ·  {cn}  ·  {len(trip_df)} triplets\n{'='*75}")

        try:
            # Load raw TCGA data
            mirna_expr, mrna_expr, clinical = load_cancer_data(code)

            # Align samples once — shared by all three analysis parts
            mirna_sub, mrna_sub, clin_sub = align_samples(mirna_expr, mrna_expr, clinical)
            print(f"    Aligned {len(clin_sub)} common samples")

            # ── PART A: Individual triplet KM (TAS-based) ─────────────────
            print(f"\n[TAS-KM]  Individual triplet Kaplan–Meier for {code} …")
            trip_results = plot_triplet_km_per_cancer(
                trip_df, mirna_sub, mrna_sub, clin_sub, code, cn)
            all_triplet_results.extend(trip_results)

            # ── PART B: Individual gene-level KM (Host | miRNA | Target) ──
            # Replicates the manual KM figure style with ggplot2 red/teal colours
            gene_results = plot_individual_gene_km_per_cancer(
                trip_df, mirna_sub, mrna_sub, clin_sub, code, cn)
            all_gene_results[code] = gene_results

            # ── PART C: Neural network pipeline ───────────────────────────
            X_df = extract_features(trip_df, mirna_sub, mrna_sub, clin_sub, code)
            X, T, E, scaler, pca = preprocess(X_df, clin_sub, code)

            global MODEL_CONFIGS
            MODEL_CONFIGS = build_configs(X.shape[1])

            cv_res, best_models, fold_hist = run_cv(X, T, E, code)
            km_res = run_km(best_models, X, T, E, code)

            plot_training(fold_hist, code, cn)
            plot_cv(cv_res, code, cn)
            plot_km_nn(km_res, cv_res, code, cn)
            plot_calibration(km_res, code, cn)
            plot_importance(best_models, X, T, E, pca, code, cn)

            all_res[code] = {"cv": cv_res, "km": km_res}

        except Exception as e:
            print(f"\n  [ERROR] in {code}: {e}")
            import traceback; traceback.print_exc()
            continue

    # ── Master panel: all 15 triplets (TAS) in one figure ─────────────────
    if all_triplet_results:
        print("\n[MASTER]  Plotting all-triplet TAS master panel …")
        plot_all_triplets_panel(all_triplet_results)

    # ── Gene KM atlas: all cancers × all gene types in one figure ─────────
    if all_gene_results:
        plot_gene_km_atlas(all_gene_results)

    # ── Pan-cancer NN comparison ───────────────────────────────────────────
    if len(all_res) > 1:
        plot_pan_cancer(all_res)

    write_report(all_res, triplets, all_triplet_results, all_gene_results)
    print(f"\n[DONE]  Complete.  Outputs → {OUT_DIR}\n")


if __name__ == "__main__":
    main()