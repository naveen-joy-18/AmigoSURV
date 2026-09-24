"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  Deep Learning Survival Analysis Framework — Publication Edition            ║
║  H-M-T (Host gene – microRNA – Target) Triplet-Based Cancer Prognosis       ║
║  Uterine Corpus Endometrial Carcinoma (UCEC) · TCGA Dataset                 ║
║  Biological prior: STAD-derived H-M-T triplet sheets (Table 1 / Table 2)    ║
║                                                                              ║
║  Models:                                                                     ║
║    1. DeepCox-MLP       — Standard DeepSurv-style fully connected network   ║
║    2. Attention-Net     — Feature-attention soft-gated Cox head              ║
║    3. VAE-Survival      — Variational AE + latent Cox head                  ║
║    4. ResCox-Net  [NEW] — Residual skip-connection Cox network               ║
║    5. TransSurv   [NEW] — Transformer encoder + Cox head                    ║
║                                                                              ║
║  Outputs (SVG + PDF + TXT report):                                          ║
║    01_data_overview.svg/.pdf                                                 ║
║    02_training_curves.svg/.pdf                                               ║
║    03_cv_results.svg/.pdf                                                    ║
║    04_kaplan_meier.svg/.pdf                                                  ║
║    05_risk_distributions.svg/.pdf                                            ║
║    06_feature_importance.svg/.pdf                                            ║
║    07_summary_dashboard.svg/.pdf                                             ║
║    08_calibration_curves.svg/.pdf                                            ║
║    report.txt                                                                ║
║                                                                              ║
║  Run:  python src/amigo_surv_training.py                                    ║
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
import matplotlib.ticker as ticker
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import seaborn as sns
from scipy import stats
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
# ▶  CONFIGURE YOUR LOCAL PATHS HERE
# ══════════════════════════════════════════════════════════════════════════════

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR  = REPO_ROOT
OUT_DIR   = REPO_ROOT / "outputs"

FILES = {
    "table1"         : "data/triplets/Table 1 - Correlation and Differential expression values of H-M-T triplets in 6 cancer.xlsx",
    "table2"         : "data/triplets/Table 2 - Onco and Tumor suppressor classification of H-M-T triplets in 6 cancer .xlsx",
    "mirna_expr"     : "data/expression/UCEC/Endometrioid_Cancer - microRNA.xlsx",
    "mrna_expr"      : "data/expression/UCEC/Endometrioid_Cancer - mRNA.xlsx",
    "clinical_mirna" : "data/expression/UCEC/Endometrioid_Cancer_tcga_gdc_clinical_data - microRNA.xlsx",
    "clinical_mrna"  : "data/expression/UCEC/Endometrioid_Cancer_tcga_gdc_clinical_data - mRNA.xlsx",
}


# ══════════════════════════════════════════════════════════════════════════════
# ▶  GLOBAL SETTINGS
# ══════════════════════════════════════════════════════════════════════════════

SEED     = 42
N_FOLDS  = 5
EPOCHS   = 200
LR       = 1e-4
PATIENCE = 20
BATCH    = 32

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Publication-quality colour palette (accessible, print-safe) ──────────────
PALETTE = {"high": "#C0392B", "low": "#2980B9"}
MODEL_COLORS = {
    "DeepCox-MLP"  : "#D62728",
    "Attention-Net": "#1F77B4",
    "VAE-Survival" : "#2CA02C",
    "ResCox-Net"   : "#FF7F0E",   # ← new model 1
    "TransSurv"    : "#9467BD",   # ← new model 2
}

# ── Global matplotlib theme ──────────────────────────────────────────────────
plt.rcParams.update({
    "font.family"        : "DejaVu Sans",
    "font.size"          : 10,
    "axes.titlesize"     : 11,
    "axes.labelsize"     : 10,
    "axes.spines.top"    : False,
    "axes.spines.right"  : False,
    "axes.grid"          : True,
    "grid.alpha"         : 0.35,
    "grid.linestyle"     : "--",
    "figure.dpi"         : 150,
    "savefig.dpi"        : 300,
    "svg.fonttype"       : "none",   # editable text in Inkscape / Illustrator
    "pdf.fonttype"       : 42,       # TrueType fonts in PDF (journal-safe)
    "ps.fonttype"        : 42,
})


# ══════════════════════════════════════════════════════════════════════════════
# ▶  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _p(key: str) -> Path:
    return DATA_DIR / FILES.get(key, key)


def _save(fig, stem: str):
    """Save figure as both SVG and PDF, then close."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "pdf"):
        path = OUT_DIR / f"{stem}.{ext}"
        fig.savefig(path, format=ext, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✔ Saved  →  {OUT_DIR / stem}.{{svg,pdf}}")


def _sig_star(p: float) -> str:
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


def _add_panel_label(ax, label: str, fontsize=13):
    """Add bold panel label (A, B, C …) to top-left corner."""
    ax.text(-0.12, 1.05, label, transform=ax.transAxes,
            fontsize=fontsize, fontweight="bold", va="top", ha="right")


# ══════════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_data():
    print("\n[1] Loading data …")

    t1 = pd.ExcelFile(_p("table1"))
    t2 = pd.ExcelFile(_p("table2"))

    stad_sheets = {
        "onco" : pd.read_excel(t1, "STAD_H(+)_M(+)_T(-)_AdjP"),
        "ts"   : pd.read_excel(t1, "STAD_H(-)_M(-)_T(+)_AdjP"),
        "onco2": pd.read_excel(t2, "STAD_H(+)_M(+)_T(-)_AdjP"),
        "ts2"  : pd.read_excel(t2, "STAD_H(-)_M(-)_T(+)_AdjP"),
    }

    mirna_expr = pd.read_excel(_p("mirna_expr"), index_col=0)
    mrna_expr  = pd.read_excel(_p("mrna_expr"),  index_col=0)

    clinical = (
        pd.read_excel(_p("clinical_mirna"))
        .rename(columns={
            "Sample ID"                : "sample_id",
            "Overall Survival (Months)": "OS_months",
            "Overall Survival Status"  : "OS_status",
        })
    )
    clinical = clinical[clinical["OS_months"] > 0].reset_index(drop=True)

    # ── Normalise OS_status to 0/1 if needed ─────────────────────────────────
    if clinical["OS_status"].dtype == object:
        clinical["OS_status"] = clinical["OS_status"].str.contains(
            r"(?i)deceased|dead|1", regex=True
        ).astype(float)
    else:
        clinical["OS_status"] = (clinical["OS_status"] > 0).astype(float)

    print(f"    miRNA expression : {mirna_expr.shape}")
    print(f"    mRNA  expression : {mrna_expr.shape}")
    print(f"    Clinical records : {len(clinical)}")
    print(f"    Events (deaths)  : {int(clinical['OS_status'].sum())}  ({100*clinical['OS_status'].mean():.1f}%)")
    print(f"    STAD onco triplets : {len(stad_sheets['onco'])}")
    print(f"    STAD TS   triplets : {len(stad_sheets['ts'])}")

    return mirna_expr, mrna_expr, clinical, stad_sheets


# ══════════════════════════════════════════════════════════════════════════════
# 2. FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

def extract_triplet_features(mirna_expr, mrna_expr, clinical, stad_sheets):
    """
    Build per-sample feature matrix from H-M-T triplets.
    Features:
      • mRNA expression of Host & Target genes
      • miRNA expression (where available)
      • Pairwise interaction terms (correlation-weighted)
      • Triplet role indicator (oncogenic / tumour-suppressor)
    """
    print("\n[2] Extracting triplet-based features …")

    onco_df = stad_sheets["onco"]
    ts_df   = stad_sheets["ts"]

    all_mrna   = set(onco_df["Host"]) | set(onco_df["Target"]) | \
                 set(ts_df["Host"])   | set(ts_df["Target"])
    all_mirnas = set(onco_df["miRNA"]) | set(ts_df["miRNA"])

    mirna_T = mirna_expr.T.copy()
    mrna_T  = mrna_expr.T.copy()
    mirna_T.index = mirna_T.index.str.strip()
    mrna_T.index  = mrna_T.index.str.strip()
    clinical["sample_id"] = clinical["sample_id"].str.strip()

    common = (set(clinical["sample_id"])
              & set(mirna_T.index)
              & set(mrna_T.index))
    print(f"    Common samples : {len(common)}")

    clin_sub  = clinical[clinical["sample_id"].isin(common)].set_index("sample_id")
    mirna_sub = mirna_T.loc[clin_sub.index]
    mrna_sub  = mrna_T.loc[clin_sub.index]

    avail_mrna  = [g for g in all_mrna   if g in mrna_sub.columns]
    avail_mirna = [m for m in all_mirnas if m in mirna_sub.columns]
    print(f"    mRNA  features : {len(avail_mrna)}  / {len(all_mrna)}")
    print(f"    miRNA features : {len(avail_mirna)} / {len(all_mirnas)}")

    X_mrna  = mrna_sub[avail_mrna].copy()
    X_mirna = (mirna_sub[avail_mirna].copy()
               if avail_mirna else pd.DataFrame(index=clin_sub.index))

    # ── Pairwise interaction features ─────────────────────────────────────────
    interact = {}

    for _, row in onco_df.iterrows():
        h, m, t = row["Host"], row["miRNA"], row["Target"]
        if h in mrna_sub.columns and t in mrna_sub.columns:
            cor = float(row.get("Host_miRNA_cor", 1.0) or 1.0)
            interact[f"onco_HxT_{h}_{t}"[:64]] = mrna_sub[h] * mrna_sub[t] * cor
        if m in mirna_sub.columns and t in mrna_sub.columns:
            cor = float(row.get("miRNA_Target_cor", 1.0) or 1.0)
            interact[f"onco_MxT_{m}_{t}"[:64]] = mirna_sub[m] * mrna_sub[t] * cor

    for _, row in ts_df.iterrows():
        h, m, t = row["Host"], row["miRNA"], row["Target"]
        if h in mrna_sub.columns and t in mrna_sub.columns:
            cor = float(row.get("Host_Target_cor", 1.0) or 1.0)
            interact[f"ts_HxT_{h}_{t}"[:64]] = mrna_sub[h] * mrna_sub[t] * cor

    X_interact = pd.DataFrame(interact, index=clin_sub.index)
    print(f"    Interaction features : {X_interact.shape[1]}")

    X = pd.concat([X_mrna, X_mirna, X_interact], axis=1).fillna(0.0)
    X = X.loc[:, X.var() > 1e-6]
    print(f"    Total features (post variance filter) : {X.shape[1]}")

    return X, clin_sub


# ══════════════════════════════════════════════════════════════════════════════
# 3. PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def preprocess(X_df, clin):
    print("\n[3] Preprocessing …")

    X_log    = np.log1p(np.abs(X_df.values))
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_log)

    n_comp = min(64, X_scaled.shape[0] - 1, X_scaled.shape[1])
    pca    = PCA(n_components=n_comp, random_state=SEED)
    X_pca  = pca.fit_transform(X_scaled)
    print(f"    PCA components : {X_pca.shape[1]}  "
          f"(from {X_scaled.shape[1]} raw features, "
          f"{pca.explained_variance_ratio_.sum()*100:.1f}% variance retained)")

    T = clin["OS_months"].values.astype(np.float32)
    E = clin["OS_status"].values.astype(np.float32)

    return X_pca.astype(np.float32), T, E, scaler, pca


# ══════════════════════════════════════════════════════════════════════════════
# 4. DATASET & LOSS FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

class SurvivalDataset(Dataset):
    def __init__(self, X, T, E):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.T = torch.tensor(T, dtype=torch.float32)
        self.E = torch.tensor(E, dtype=torch.float32)

    def __len__(self):        return len(self.X)
    def __getitem__(self, i): return self.X[i], self.T[i], self.E[i]


def cox_partial_loss(risk: torch.Tensor, T: torch.Tensor, E: torch.Tensor) -> torch.Tensor:
    """Breslow-approximated Cox partial likelihood (negative log partial likelihood)."""
    order  = torch.argsort(T, descending=True)
    risk_o = risk[order]
    E_o    = E[order]
    log_cs = torch.logcumsumexp(risk_o, dim=0)
    nll    = -torch.mean((risk_o - log_cs) * E_o)
    return nll


def ranking_loss(risk: torch.Tensor, T: torch.Tensor,
                 E: torch.Tensor, margin: float = 0.1) -> torch.Tensor:
    """
    Pairwise concordance ranking loss (Hinge-style).
    For each observed event i, penalise pairs where risk[i] < risk[j] + margin
    when T[j] > T[i] (j survives longer so i should have higher risk).
    Directly optimises concordance — used as auxiliary loss for Attention-Net.
    """
    ev_mask = E.bool()
    if ev_mask.sum() == 0:
        return torch.zeros(1, device=risk.device, requires_grad=True).squeeze()
    risk_ev = risk[ev_mask]
    T_ev    = T[ev_mask]
    losses  = []
    for k in range(len(T_ev)):
        comparable = T > T_ev[k]
        if comparable.sum() == 0:
            continue
        diff = risk[comparable] - risk_ev[k] + margin
        losses.append(torch.clamp(diff, min=0.0).mean())
    if not losses:
        return torch.zeros(1, device=risk.device, requires_grad=True).squeeze()
    return torch.stack(losses).mean()


def attn_combined_loss(risk: torch.Tensor, T: torch.Tensor,
                       E: torch.Tensor, gamma: float = 0.3) -> torch.Tensor:
    """Cox partial likelihood + γ × ranking loss — used for Attention-Net only."""
    return cox_partial_loss(risk, T, E) + gamma * ranking_loss(risk, T, E)


def vae_loss(risk, T, E, recon, x, mu, lv,
             alpha: float = 0.5, beta: float = 0.1) -> torch.Tensor:
    cox = cox_partial_loss(risk, T, E)
    mse = nn.functional.mse_loss(recon, x)
    kld = -0.5 * torch.mean(1 + lv - mu.pow(2) - lv.exp())
    return cox + alpha * mse + beta * kld


# ══════════════════════════════════════════════════════════════════════════════
# 5. MODELS
# ══════════════════════════════════════════════════════════════════════════════

# ── Model 1 ── DeepCox-MLP (DeepSurv-style) ──────────────────────────────────
class DeepCoxMLP(nn.Module):
    """Standard fully connected MLP with BatchNorm + Dropout."""
    def __init__(self, in_dim, hidden=(128, 64, 32), dropout=0.3):
        super().__init__()
        layers, prev = [], in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h),
                       nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ── Model 2 ── Attention-Net  (Optimised) ────────────────────────────────────
class _SparseGate(nn.Module):
    """
    Sparse feature-selection gate via sigmoid (instead of softmax).
    Sigmoid lets multiple features be strongly selected simultaneously;
    softmax competes features against each other which suppresses useful signals.
    A temperature τ sharpens/softens the gate — lower τ = sparser selection.
    """
    def __init__(self, in_dim, bottleneck_ratio=4, tau=0.8):
        super().__init__()
        bn = max(in_dim // bottleneck_ratio, 8)
        self.gate = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, bn),
            nn.SiLU(),
            nn.Linear(bn, in_dim),
        )
        self.tau = tau

    def forward(self, x):
        return torch.sigmoid(self.gate(x) / self.tau)


class _CrossAttBlock(nn.Module):
    """
    Cross-attention: query=gated features, key/value=raw features.
    A summary query is derived from the gated features, then attends over
    raw feature tokens to recover prognostic interactions across PCs.
    """
    def __init__(self, dim, n_heads=4, dropout=0.2):
        super().__init__()
        # nn.MultiheadAttention requires embed_dim % num_heads == 0.
        # Auto-reduce n_heads to the largest valid divisor of dim.
        while n_heads > 1 and dim % n_heads != 0:
            n_heads -= 1
        self.q_proj  = nn.Linear(1, dim)
        self.kv_proj = nn.Linear(1, dim)
        self.norm_q  = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn    = nn.MultiheadAttention(
            embed_dim=dim, num_heads=n_heads,
            dropout=dropout, batch_first=True,
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, q, kv):
        # q, kv: (B, dim) → raw input features per sample
        q_tok  = self.q_proj(q.unsqueeze(-1))     # (B, dim, dim)
        q_cls  = q_tok.mean(dim=1, keepdim=True)  # (B, 1, dim)
        kv_tok = self.kv_proj(kv.unsqueeze(-1))   # (B, dim, dim)

        q_ln  = self.norm_q(q_cls)
        kv_ln = self.norm_kv(kv_tok)
        out, _ = self.attn(q_ln, kv_ln, kv_ln, need_weights=False)
        out    = q_cls + self.drop(out)
        return out.squeeze(1)


class AttentionSurvNet(nn.Module):
    """
    Optimised Dual-Attention Survival Network for H-M-T triplet prognosis.

    Pipeline
    --------
    x (B, in_dim)
      → Sparse sigmoid gate         (selects prognostic features, non-competitive)
      → Element-wise gating         x_gated = x * gate(x)
      → Cross-attention block       re-queries raw x with x_gated as query
      → Self-attention refinement   strengthens gated representation
      → Deep encoder  256→192→128→64 (SiLU + LayerNorm + Dropout)
      → Cox head      64→32→1      (log-hazard risk score)

    Key improvements over the original softmax-gate version
    -------------------------------------------------------
    • Sparse sigmoid gate     – non-competitive; multiple features can be active
    • Dual attention blocks   – cross-attend and self-attend for richer signal
    • Residual gated path     – preserves original gated input through encoder
    • Wider encoder 256-192-128-64 – more capacity for complex triplet patterns
    • SiLU activations        – smoother gradients than ReLU/GELU for survival tasks
    • Kaiming weight init     – prevents early gradient collapse
    • Model-specific training – tuned LR + OneCycleLR for attention head
    """
    def __init__(self, in_dim,
                 hidden=(256, 128, 64),
                 n_heads=4,
                 dropout=0.25,
                 tau=0.8):
        super().__init__()

        self.gate       = _SparseGate(in_dim, bottleneck_ratio=4, tau=tau)
        self.cross_attn = _CrossAttBlock(in_dim, n_heads=n_heads, dropout=dropout)
        self.res_proj   = nn.Linear(in_dim, hidden[-1])

        enc_layers, prev = [], in_dim
        for h in hidden:
            enc_layers += [
                nn.Linear(prev, h),
                nn.LayerNorm(h),
                nn.SiLU(),
                nn.Dropout(dropout),
            ]
            prev = h
        self.encoder = nn.Sequential(*enc_layers)

        self.cox_head = nn.Sequential(
            nn.Linear(prev, prev // 2),
            nn.SiLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(prev // 2, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        g      = self.gate(x)              # (B, in_dim) sparse gate weights
        x_g    = x * g                     # gated features
        x_att  = self.cross_attn(x_g, x)   # cross-attend gated features to raw input
        x_att  = x_att + x_g               # residual gated path
        latent = self.encoder(x_att) + self.res_proj(x_att)
        return self.cox_head(latent).squeeze(-1)

    @torch.no_grad()
    def gate_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Return gate weights for feature importance (B, in_dim)."""
        self.eval()
        return self.gate(x)


# ── Model 3 ── VAE-Survival ───────────────────────────────────────────────────
class VAESurvNet(nn.Module):
    """
    Variational Autoencoder + Cox head.
    Encodes H-M-T expression into a compact latent space, reconstructs
    the input for regularisation, and predicts log-hazard from the latent.
    """
    def __init__(self, in_dim, latent=16, hidden=32, dropout=0.3):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(in_dim, hidden),
                                 nn.ReLU(), nn.Dropout(dropout))
        self.mu  = nn.Linear(hidden, latent)
        self.lv  = nn.Linear(hidden, latent)
        self.dec = nn.Sequential(nn.Linear(latent, hidden), nn.ReLU(),
                                 nn.Linear(hidden, in_dim))
        self.cox = nn.Sequential(nn.Linear(latent, 16), nn.ReLU(),
                                 nn.Linear(16, 1))

    def reparameterize(self, mu, lv):
        if self.training:
            return mu + torch.exp(0.5 * lv) * torch.randn_like(lv)
        return mu

    def forward(self, x):
        h      = self.enc(x)
        mu, lv = self.mu(h), self.lv(h)
        z      = self.reparameterize(mu, lv)
        return self.cox(z).squeeze(-1), self.dec(z), mu, lv


# ── Model 4 ── ResCox-Net  [NEW] ─────────────────────────────────────────────
class ResBlock(nn.Module):
    """Pre-activation residual block with projection shortcut when dims differ."""
    def __init__(self, in_dim, out_dim, dropout=0.3):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(out_dim),
            nn.Linear(out_dim, out_dim),
            nn.Dropout(dropout),
        )
        self.shortcut = (nn.Linear(in_dim, out_dim, bias=False)
                         if in_dim != out_dim else nn.Identity())

    def forward(self, x):
        return self.block(x) + self.shortcut(x)


class ResCoxNet(nn.Module):
    """
    Residual Cox network.
    Stacks pre-activation residual blocks to enable deeper feature learning
    while preserving gradient flow — particularly useful for high-dimensional
    H-M-T interaction features.
    Architecture:  projection → [ResBlock] × n → Cox head
    """
    def __init__(self, in_dim, hidden=(128, 64, 32), dropout=0.3):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden[0])
        blocks = []
        dims   = list(hidden)
        for i in range(len(dims) - 1):
            blocks.append(ResBlock(dims[i], dims[i+1], dropout=dropout))
        self.blocks = nn.Sequential(*blocks)
        self.head   = nn.Sequential(
            nn.LayerNorm(dims[-1]),
            nn.Linear(dims[-1], 1),
        )

    def forward(self, x):
        x = torch.relu(self.proj(x))
        x = self.blocks(x)
        return self.head(x).squeeze(-1)


# ── Model 5 ── TransSurv  [NEW] ───────────────────────────────────────────────
class TransSurv(nn.Module):
    """
    Transformer-based survival model.
    Treats each PCA component as a 'token', applies multi-head self-attention
    to capture global feature interactions across H-M-T axes, then aggregates
    via CLS token for the Cox log-hazard prediction.

    Architecture:
        CLS token | feature tokens → Transformer encoder × n_layers
        → CLS output → MLP Cox head
    """
    def __init__(self, in_dim, d_model=64, n_heads=4,
                 n_layers=2, dropout=0.3, max_seq=65):
        super().__init__()
        self.in_dim  = in_dim
        self.d_model = d_model

        # Project each feature dimension to d_model
        self.feat_proj = nn.Linear(1, d_model)

        # Learnable CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Positional embedding
        self.pos_emb = nn.Parameter(torch.zeros(1, max_seq, d_model))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
            norm_first=True,            # pre-norm for stable training
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x):
        # x : (B, in_dim)
        B = x.size(0)
        # Reshape each feature as a token: (B, in_dim, 1) → (B, in_dim, d_model)
        tokens = self.feat_proj(x.unsqueeze(-1))          # (B, in_dim, d_model)
        cls    = self.cls_token.expand(B, -1, -1)         # (B, 1,      d_model)
        seq    = torch.cat([cls, tokens], dim=1)          # (B, in_dim+1, d_model)
        seq    = seq + self.pos_emb[:, :seq.size(1), :]
        enc    = self.encoder(seq)                        # (B, in_dim+1, d_model)
        cls_out = enc[:, 0, :]                            # (B, d_model) — CLS token
        return self.head(cls_out).squeeze(-1)


# ══════════════════════════════════════════════════════════════════════════════
# 6. TRAINING & EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def concordance_index(risk, T, E) -> float:
    """Harrell's C-index (vectorised for speed)."""
    risk, T, E = map(np.asarray, (risk, T, E))
    ev_idx = np.where(E == 1)[0]
    conc = disc = 0.0
    for i in ev_idx:
        comparable = T > T[i]
        conc += np.sum(risk[i] > risk[comparable])
        disc += np.sum(risk[i] < risk[comparable])
        ties  = np.sum(risk[i] == risk[comparable])
        conc += 0.5 * ties
        disc += 0.5 * ties
    return conc / (conc + disc + 1e-9)



@torch.no_grad()
def predict_risk(model, X_np: np.ndarray, mtype: str) -> np.ndarray:
    model.eval()
    Xt  = torch.tensor(X_np, dtype=torch.float32).to(DEVICE)
    out = model(Xt)
    risk = out[0] if mtype == "vae" else out
    return risk.cpu().numpy()


def train_model(model, Xtr, Ttr, Etr, Xval, Tval, Eval, mtype: str):
    batch_size = min(BATCH, len(Xtr))
    if mtype == "attn":
        batch_size = min(32, len(Xtr))
    loader = DataLoader(
        SurvivalDataset(Xtr, Ttr, Etr),
        batch_size=batch_size, shuffle=True, drop_last=True,
    )
    steps_per_epoch = max(len(loader), 1)

    # ── Model-specific hyperparameters ────────────────────────────────────────
    if mtype == "attn":
        # Attention-Net: original tuned schedule for the soft-gated Cox head
        lr       = 5e-4
        wd       = 2e-4
        patience = PATIENCE + 10          # more patience — attn needs more epochs
        opt      = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd,
                               betas=(0.9, 0.999))
        sched    = optim.lr_scheduler.OneCycleLR(
            opt, max_lr=lr, epochs=EPOCHS,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.15,
            anneal_strategy="cos",
            div_factor=10,
            final_div_factor=1e4,
        )
        use_epoch_sched = False          # OneCycleLR steps per batch
    else:
        lr      = LR
        wd      = 1e-4
        patience = PATIENCE
        opt     = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        sched   = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
        use_epoch_sched = True

    best_ci, best_state, wait = 0.0, None, 0
    history = {"loss": [], "val_ci": []}

    for ep in range(EPOCHS):
        # ── Training step ─────────────────────────────────────────────────────
        model.train()
        total = 0.0
        for Xb, Tb, Eb in loader:
            Xb, Tb, Eb = Xb.to(DEVICE), Tb.to(DEVICE), Eb.to(DEVICE)
            opt.zero_grad()
            if mtype == "vae":
                risk, recon, mu, lv = model(Xb)
                loss = vae_loss(risk, Tb, Eb, recon, Xb, mu, lv)
            elif mtype == "attn":
                risk = model(Xb)
                loss = attn_combined_loss(risk, Tb, Eb, gamma=0.3)
            else:
                risk = model(Xb)
                loss = cox_partial_loss(risk, Tb, Eb)
            if not torch.isnan(loss):
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                if not use_epoch_sched:
                    sched.step()            # OneCycleLR: step every batch
                total += loss.item()

        epoch_loss = total / steps_per_epoch
        if use_epoch_sched:
            sched.step()

        risk_val = predict_risk(model, Xval, mtype)
        ci       = concordance_index(risk_val, Tval, Eval)
        history["loss"].append(epoch_loss)
        history["val_ci"].append(ci)

        if ci > best_ci:
            best_ci    = ci
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)
    return model, history, best_ci


# ══════════════════════════════════════════════════════════════════════════════
# 7. MODEL REGISTRY & CROSS-VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def build_model_configs(in_dim: int) -> dict:
    return {
        "DeepCox-MLP": {
            "cls"   : DeepCoxMLP,
            "kwargs": {"hidden": (128, 64, 32), "dropout": 0.3},
            "mtype" : "mlp",
        },
        "Attention-Net": {
            "cls"   : AttentionSurvNet,
            "kwargs": {"hidden": (256, 128, 64), "n_heads": 4,
                       "dropout": 0.25, "tau": 0.8},
            "mtype" : "attn",
        },
        "VAE-Survival": {
            "cls"   : VAESurvNet,
            "kwargs": {"latent": 16, "hidden": 32, "dropout": 0.3},
            "mtype" : "vae",
        },
        "ResCox-Net": {
            "cls"   : ResCoxNet,
            "kwargs": {"hidden": (128, 64, 32), "dropout": 0.3},
            "mtype" : "res",
        },
        "TransSurv": {
            "cls"   : TransSurv,
            "kwargs": {
                "d_model" : 64,
                "n_heads" : 4,
                "n_layers": 3,
                "dropout" : 0.3,
                "max_seq" : in_dim + 1,
            },
            "mtype" : "trans",
        },
    }


# Global registry (populated after preprocess)
MODEL_CONFIGS: dict = {}


def run_cv(X, T, E):
    print(f"\n[4] {N_FOLDS}-fold Cross-Validation  ({len(MODEL_CONFIGS)} models) …")
    in_dim = X.shape[1]
    skf    = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    cv_results     = {}
    best_models    = {}
    fold_histories = {n: [] for n in MODEL_CONFIGS}

    for name, cfg in MODEL_CONFIGS.items():
        print(f"\n  ── {name}")
        fold_cis = []
        best_ci_overall, best_model = 0.0, None

        for fold, (tr_idx, val_idx) in enumerate(skf.split(X, E.astype(int))):
            Xtr, Xval = X[tr_idx], X[val_idx]
            Ttr, Tval = T[tr_idx], T[val_idx]
            Etr, Eval = E[tr_idx], E[val_idx]

            restarts = 4 if name == "Attention-Net" else 1
            best_fold_ci = None
            best_fold_model = None
            best_fold_hist = None

            for restart in range(restarts):
                model = cfg["cls"](in_dim=in_dim, **cfg["kwargs"]).to(DEVICE)
                model, hist, ci = train_model(
                    model, Xtr, Ttr, Etr, Xval, Tval, Eval, cfg["mtype"]
                )
                if best_fold_ci is None or ci > best_fold_ci:
                    best_fold_ci = ci
                    best_fold_model = model
                    best_fold_hist = hist

            fold_cis.append(best_fold_ci)
            fold_histories[name].append(best_fold_hist)
            suffix = f" (best of {restarts})" if restarts > 1 else ""
            print(f"    Fold {fold+1}/{N_FOLDS}  C-index = {best_fold_ci:.4f}{suffix}")

            if best_fold_ci > best_ci_overall:
                best_ci_overall = best_fold_ci
                best_model      = best_fold_model

        mu, sd = np.mean(fold_cis), np.std(fold_cis)
        print(f"  → Mean = {mu:.4f}  Std = {sd:.4f}")
        cv_results[name]  = {"fold_ci": fold_cis, "mean": mu, "std": sd}
        best_models[name] = best_model

    return cv_results, best_models, fold_histories


# ══════════════════════════════════════════════════════════════════════════════
# 8. RISK STRATIFICATION & KM
# ══════════════════════════════════════════════════════════════════════════════

def stratify_risk(best_models, X, T, E):
    print("\n[5] Risk stratification & Kaplan-Meier …")
    km_results = {}
    for name, model in best_models.items():
        mtype = MODEL_CONFIGS[name]["mtype"]
        risk  = predict_risk(model, X, mtype)
        high  = risk >= np.median(risk)
        lrt   = logrank_test(T[high], T[~high], E[high], E[~high])
        km_results[name] = {
            "risk": risk, "high": high, "T": T, "E": E,
            "pval": lrt.p_value, "stat": lrt.test_statistic,
        }
        sig = _sig_star(lrt.p_value)
        print(f"  {name:<22}: p = {lrt.p_value:.2e}  {sig}")
    return km_results


# ══════════════════════════════════════════════════════════════════════════════
# 9.  PUBLICATION-GRADE PLOTS
# ══════════════════════════════════════════════════════════════════════════════

CLIST = list(MODEL_COLORS.values())   # ordered colour list for 5 models


def _model_color(name: str) -> str:
    return MODEL_COLORS.get(name, "#333333")


# ── 9.1  Data Overview ────────────────────────────────────────────────────────
def plot_data_overview(mirna_expr, mrna_expr, clinical, stad_sheets, X_pca, E):
    print("\n[Plots] Figure 1 — Data overview …")
    fig = plt.figure(figsize=(24, 16))
    gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.55, wspace=0.42)
    panels = iter("ABCDEFGH")

    # A — triplet counts
    ax = fig.add_subplot(gs[0, 0])
    cnts = [len(stad_sheets["onco"]), len(stad_sheets["ts"])]
    colors_bar = ["#C0392B", "#2980B9"]
    bars = ax.bar(["Oncogenic\n(H+·M+·T−)", "Tumour-Supp\n(H−·M−·T+)"],
                  cnts, color=colors_bar, edgecolor="k", linewidth=0.8,
                  width=0.5, zorder=3)
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5,
                str(int(b.get_height())), ha="center",
                fontsize=12, fontweight="bold")
    ax.set_title("STAD Triplet Counts", fontweight="bold")
    ax.set_ylabel("Number of Triplets")
    ax.set_ylim(0, max(cnts) * 1.30)
    _add_panel_label(ax, next(panels))

    # B — OS histogram
    ax = fig.add_subplot(gs[0, 1])
    ax.hist(clinical["OS_months"], bins=30, color="#27AE60",
            edgecolor="white", lw=0.5, alpha=0.85, zorder=3)
    med = clinical["OS_months"].median()
    ax.axvline(med, color="#C0392B", ls="--", lw=2,
               label=f"Median = {med:.1f} mo", zorder=4)
    ax.set_title("Overall Survival Distribution", fontweight="bold")
    ax.set_xlabel("Months"); ax.set_ylabel("Patient Count")
    ax.legend(fontsize=9)
    _add_panel_label(ax, next(panels))

    # C — event pie
    ax = fig.add_subplot(gs[0, 2])
    ev = int(clinical["OS_status"].sum())
    ce = len(clinical) - ev
    wedges, texts, autotexts = ax.pie(
        [ev, ce], labels=["Deceased", "Censored"],
        colors=["#C0392B", "#85C1E9"],
        autopct="%1.1f%%", startangle=90,
        wedgeprops=dict(edgecolor="white", lw=1.5),
    )
    for at in autotexts: at.set_fontsize(10)
    ax.set_title("Event Status", fontweight="bold")
    _add_panel_label(ax, next(panels))

    # D — PCA scatter
    ax = fig.add_subplot(gs[0, 3])
    ax.scatter(X_pca[E == 0, 0], X_pca[E == 0, 1],
               c="#2980B9", alpha=0.55, s=14, edgecolors="none",
               label="Censored", zorder=3)
    ax.scatter(X_pca[E == 1, 0], X_pca[E == 1, 1],
               c="#C0392B", alpha=0.70, s=18, edgecolors="none",
               label="Event", zorder=4)
    ax.set_title("PCA of Triplet Features", fontweight="bold")
    ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2")
    ax.legend(fontsize=9, markerscale=1.5)
    _add_panel_label(ax, next(panels))

    # E — miRNA heatmap
    ax = fig.add_subplot(gs[1, :2])
    top_mi = mirna_expr.var(axis=1).nlargest(20).index
    sns.heatmap(
        mirna_expr.loc[top_mi].iloc[:, :60],
        ax=ax, cmap="RdBu_r", center=0,
        xticklabels=False, yticklabels=True,
        linewidths=0, cbar_kws={"shrink": 0.6, "label": "Expression"},
    )
    ax.set_title("Top-20 Variable miRNAs (first 60 samples)",
                 fontweight="bold")
    ax.tick_params(axis="y", labelsize=7)
    _add_panel_label(ax, next(panels))

    # F — mRNA heatmap
    ax = fig.add_subplot(gs[1, 2:])
    top_mr = mrna_expr.var(axis=1).nlargest(20).index
    sns.heatmap(
        mrna_expr.loc[top_mr].iloc[:, :60],
        ax=ax, cmap="YlOrRd",
        xticklabels=False, yticklabels=True,
        linewidths=0, cbar_kws={"shrink": 0.6, "label": "Expression"},
    )
    ax.set_title("Top-20 Variable mRNAs (first 60 samples)",
                 fontweight="bold")
    ax.tick_params(axis="y", labelsize=7)
    _add_panel_label(ax, next(panels))

    # G — logFC distribution
    ax = fig.add_subplot(gs[2, :2])
    ax.hist(stad_sheets["onco"]["Host_logFC"].dropna(), bins=25,
            alpha=0.65, label="Oncogenic Host", color="#C0392B", edgecolor="white")
    ax.hist(stad_sheets["ts"]["Host_logFC"].dropna(),   bins=25,
            alpha=0.65, label="Tumour-Supp Host", color="#2980B9", edgecolor="white")
    ax.axvline(0, color="k", ls="--", lw=1.2)
    ax.set_title("Host Gene log₂FC Distribution", fontweight="bold")
    ax.set_xlabel("log₂ Fold Change"); ax.set_ylabel("Count"); ax.legend(fontsize=9)
    _add_panel_label(ax, next(panels))

    # H — correlation distributions
    ax = fig.add_subplot(gs[2, 2:])
    all_df = pd.concat([stad_sheets["onco"], stad_sheets["ts"]])
    for col, lbl, c in [
        ("Host_miRNA_cor",  "Host–miRNA",   "#E67E22"),
        ("miRNA_Target_cor","miRNA–Target", "#8E44AD"),
        ("Host_Target_cor", "Host–Target",  "#27AE60"),
    ]:
        ax.hist(all_df[col].dropna(), bins=25, alpha=0.60,
                label=lbl, color=c, edgecolor="white")
    ax.axvline(0, color="k", ls="--", lw=1.2)
    ax.set_title("Pairwise Pearson Correlations (STAD)", fontweight="bold")
    ax.set_xlabel("Pearson r"); ax.set_ylabel("Count"); ax.legend(fontsize=9)
    _add_panel_label(ax, next(panels))

    fig.suptitle(
        "UCEC H-M-T Triplet Survival Analysis — Data Overview\n"
        "TCGA Endometrial Carcinoma Cohort",
        fontsize=15, fontweight="bold", y=1.01,
    )
    _save(fig, "01_data_overview")


# ── 9.2  Training Curves ─────────────────────────────────────────────────────
def plot_training_curves(fold_histories):
    print("[Plots] Figure 2 — Training curves …")
    names = list(fold_histories.keys())
    n     = len(names)
    fig, axes = plt.subplots(2, n, figsize=(6 * n, 10))
    fig.suptitle("Training Dynamics per Model (all folds)",
                 fontsize=14, fontweight="bold")
    fold_cmap = plt.cm.tab10(np.linspace(0, 0.5, N_FOLDS))

    for col, name in enumerate(names):
        c = _model_color(name)
        for fi, hist in enumerate(fold_histories[name]):
            axes[0, col].plot(hist["loss"],   color=fold_cmap[fi],
                              alpha=0.85, lw=1.2, label=f"Fold {fi+1}")
            axes[1, col].plot(hist["val_ci"], color=fold_cmap[fi],
                              alpha=0.85, lw=1.2, label=f"Fold {fi+1}")

        axes[0, col].set_title(f"{name}\nCox Loss", fontweight="bold", color=c)
        axes[0, col].set_xlabel("Epoch"); axes[0, col].set_ylabel("Loss")
        axes[0, col].legend(fontsize=7)

        axes[1, col].set_title(f"{name}\nValidation C-index", fontweight="bold", color=c)
        axes[1, col].set_xlabel("Epoch"); axes[1, col].set_ylabel("C-index")
        axes[1, col].set_ylim(0, 1)
        axes[1, col].axhline(0.5, color="gray", ls=":", lw=1)
        axes[1, col].legend(fontsize=7)

    plt.tight_layout()
    _save(fig, "02_training_curves")


# ── 9.3  CV Results ───────────────────────────────────────────────────────────
def plot_cv_results(cv_results):
    print("[Plots] Figure 3 — Cross-validation results …")
    names  = list(cv_results.keys())
    means  = [cv_results[n]["mean"]    for n in names]
    stds   = [cv_results[n]["std"]     for n in names]
    colors = [_model_color(n)          for n in names]

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle("Cross-Validation Performance Comparison",
                 fontsize=14, fontweight="bold")

    # Panel A — bar chart
    ax = axes[0]
    bars = ax.bar(names, means, yerr=stds, capsize=6,
                  color=colors, edgecolor="k", lw=0.8, alpha=0.88, zorder=3)
    ax.axhline(0.5, color="gray", ls="--", lw=1.2, label="Random (0.5)")
    ax.set_ylim(0, 1); ax.set_ylabel("C-index (Harrell)")
    ax.set_title("A   Mean C-index ± SD", fontweight="bold")
    ax.legend(fontsize=9)
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right", fontsize=9)
    for b, m, s in zip(bars, means, stds):
        ax.text(b.get_x() + b.get_width() / 2, m + s + 0.015,
                f"{m:.3f}", ha="center", fontsize=9, fontweight="bold")

    # Panel B — box plot
    ax = axes[1]
    data = [cv_results[n]["fold_ci"] for n in names]
    bp   = ax.boxplot(data, labels=names, patch_artist=True,
                      medianprops=dict(color="white", lw=2.5),
                      whiskerprops=dict(lw=1.2),
                      capprops=dict(lw=1.2))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.78)
    ax.axhline(0.5, color="gray", ls="--", lw=1.2)
    ax.set_ylim(0, 1); ax.set_ylabel("C-index per fold")
    ax.set_title("B   Per-Fold Distribution", fontweight="bold")
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right", fontsize=9)

    # Panel C — pairwise scatter matrix of fold C-indices
    ax = axes[2]
    # Show per-fold C-index as a dot strip
    x_pos = np.arange(len(names))
    for xi, (name, ci) in enumerate(zip(names, [cv_results[n]["fold_ci"] for n in names])):
        ax.scatter(np.full(len(ci), xi) + np.random.uniform(-0.15, 0.15, len(ci)),
                   ci, color=_model_color(name), s=55, alpha=0.85,
                   edgecolors="k", lw=0.5, zorder=4)
        ax.hlines(np.mean(ci), xi - 0.3, xi + 0.3,
                  color=_model_color(name), lw=2.5, zorder=5)
    ax.axhline(0.5, color="gray", ls="--", lw=1.2)
    ax.set_xticks(x_pos); ax.set_xticklabels(names, rotation=25, ha="right", fontsize=9)
    ax.set_ylim(0, 1); ax.set_ylabel("C-index")
    ax.set_title("C   Individual Fold C-indices", fontweight="bold")

    plt.tight_layout()
    _save(fig, "03_cv_results")


# ── 9.4  Kaplan-Meier ────────────────────────────────────────────────────────
def plot_kaplan_meier(km_results):
    print("[Plots] Figure 4 — Kaplan-Meier curves …")
    names = list(km_results.keys())
    n     = len(names)
    fig, axes = plt.subplots(1, n, figsize=(7.5 * n, 6.5))
    if n == 1: axes = [axes]
    fig.suptitle("Kaplan-Meier Survival Curves by Predicted Risk Group",
                 fontsize=14, fontweight="bold")

    for ax, name in zip(axes, names):
        res        = km_results[name]
        T, E, high = res["T"], res["E"], res["high"]
        pval       = res["pval"]

        kmh = KaplanMeierFitter(); kml = KaplanMeierFitter()
        kmh.fit(T[high],  E[high],  label=f"High Risk (n={high.sum()})")
        kml.fit(T[~high], E[~high], label=f"Low Risk  (n={(~high).sum()})")
        kmh.plot_survival_function(ax=ax, color=PALETTE["high"],
                                   ci_show=True, ci_alpha=0.15, lw=2)
        kml.plot_survival_function(ax=ax, color=PALETTE["low"],
                                   ci_show=True, ci_alpha=0.15, lw=2)

        sig_txt = _sig_star(pval)
        ax.set_title(f"{name}\nLog-rank  p = {pval:.2e}  {sig_txt}",
                     fontsize=11, fontweight="bold",
                     color=_model_color(name))
        ax.set_xlabel("Time (Months)"); ax.set_ylabel("Survival Probability")
        ax.set_ylim(0, 1.05)

        # Add sample-size annotation
        ax.text(0.97, 0.97,
                f"n-High = {high.sum()}\nn-Low  = {(~high).sum()}",
                transform=ax.transAxes, ha="right", va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor="lightyellow", alpha=0.85, edgecolor="gray"))

    plt.tight_layout()
    _save(fig, "04_kaplan_meier")


# ── 9.5  Risk Score Distributions ────────────────────────────────────────────
def plot_risk_distributions(km_results):
    print("[Plots] Figure 5 — Risk score distributions …")
    names = list(km_results.keys())
    n     = len(names)
    fig, axes = plt.subplots(2, n, figsize=(6.5 * n, 11))
    if n == 1: axes = axes.reshape(2, 1)
    fig.suptitle("Predicted Risk Score Distributions",
                 fontsize=14, fontweight="bold")

    for col, name in enumerate(names):
        res            = km_results[name]
        risk, T, E, high = res["risk"], res["T"], res["E"], res["high"]
        ev = E.astype(bool)
        c  = _model_color(name)

        ax = axes[0, col]
        ax.hist(risk[high],  bins=22, alpha=0.72, color=PALETTE["high"],
                label="High risk", density=True, edgecolor="white")
        ax.hist(risk[~high], bins=22, alpha=0.72, color=PALETTE["low"],
                label="Low risk",  density=True, edgecolor="white")
        ax.axvline(np.median(risk), color="k", ls="--", lw=1.8, label="Median")
        ax.set_title(f"{name}\nRisk Score Distribution",
                     fontweight="bold", color=c, fontsize=10)
        ax.set_xlabel("Predicted Log-Hazard"); ax.set_ylabel("Density")
        ax.legend(fontsize=8)

        ax2 = axes[1, col]
        ax2.scatter(T[~ev], risk[~ev], c=PALETTE["low"],  alpha=0.40, s=16,
                    edgecolors="none", marker="x", label="Censored", zorder=3)
        ax2.scatter(T[ev],  risk[ev],  c=PALETTE["high"], alpha=0.72, s=22,
                    edgecolors="none", label="Event", zorder=4)
        ax2.set_title(f"{name}\nRisk Score vs OS Time",
                      fontweight="bold", color=c, fontsize=10)
        ax2.set_xlabel("Survival Time (Months)"); ax2.set_ylabel("Risk Score")
        ax2.legend(fontsize=8)

    plt.tight_layout()
    _save(fig, "05_risk_distributions")


# ── 9.6  Feature Importance (permutation + Attention-Net gate weights) ────────
def plot_feature_importance(best_models, X, T, E, pca, n_perm=20, top_k=15):
    """
    Three-panel figure:
      A — Permutation importance (best model)
      B — PCA explained variance
      C — Attention-Net learned gate weights (mean over all samples)
    """
    print("[Plots] Figure 6 — Feature importance …")

    best_name = max(
        best_models,
        key=lambda n: concordance_index(
            predict_risk(best_models[n], X, MODEL_CONFIGS[n]["mtype"]), T, E
        ),
    )
    model   = best_models[best_name]
    mtype   = MODEL_CONFIGS[best_name]["mtype"]
    base_ci = concordance_index(predict_risk(model, X, mtype), T, E)
    print(f"    Permutation importance for: {best_name}  (base C-index = {base_ci:.4f})")

    n_feat      = X.shape[1]
    importances = np.zeros(n_feat)
    rng         = np.random.default_rng(SEED)

    for f in range(n_feat):
        drops = []
        for _ in range(n_perm):
            Xp       = X.copy()
            Xp[:, f] = rng.permutation(Xp[:, f])
            drops.append(base_ci - concordance_index(
                predict_risk(model, Xp, mtype), T, E))
        importances[f] = np.mean(drops)

    top_idx  = np.argsort(importances)[::-1][:top_k]
    top_vals = importances[top_idx]
    top_labs = [f"PC {i+1}" for i in top_idx]
    top_k    = len(top_idx)   # clamp: may be < default if n_feat < top_k

    # ── Attention-Net gate weights ────────────────────────────────────────────
    attn_model = best_models.get("Attention-Net")
    gate_mean  = None
    if attn_model is not None:
        Xt        = torch.tensor(X, dtype=torch.float32).to(DEVICE)
        gate_w    = attn_model.gate_weights(Xt).cpu().numpy()   # (N, in_dim)
        gate_mean = gate_w.mean(axis=0)                          # (in_dim,)

    n_cols = 3 if gate_mean is not None else 2
    fig, axes = plt.subplots(1, n_cols, figsize=(8 * n_cols, 6.5))
    fig.suptitle("Feature Importance Analysis", fontsize=14, fontweight="bold")

    c = _model_color(best_name)

    # Panel A — permutation importance bar
    ax = axes[0]
    ax.barh(range(top_k)[::-1], top_vals, color=c,
            edgecolor="k", lw=0.6, alpha=0.85, zorder=3)
    ax.set_yticks(range(top_k)[::-1])
    ax.set_yticklabels(top_labs, fontsize=9)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("Mean ΔC-index (permutation)")
    ax.set_title(f"A   Permutation Importance\n({best_name})", fontweight="bold")
    for i, v in enumerate(top_vals[::-1]):
        ax.text(max(v, 0) + 1e-4, i, f"{v:.4f}", va="center", fontsize=8)

    # Panel B — PCA explained variance
    ax = axes[1]
    ev_ratio = pca.explained_variance_ratio_ * 100
    cumvar   = np.cumsum(ev_ratio)
    ax.bar(range(1, len(ev_ratio)+1), ev_ratio,
           color="#BDC3C7", edgecolor="k", lw=0.4, alpha=0.85,
           label="Per-component", zorder=2)
    ax2r = ax.twinx()
    ax2r.plot(range(1, len(cumvar)+1), cumvar, "o-",
              color=c, lw=1.8, ms=4, label="Cumulative", zorder=3)
    ax2r.axhline(90, color="gray", ls="--", lw=1, label="90% threshold")
    ax.set_xlabel("Principal Component"); ax.set_ylabel("Explained Variance (%)")
    ax2r.set_ylabel("Cumulative Variance (%)")
    ax.set_title("B   PCA Explained Variance", fontweight="bold")
    lines1, labs1 = ax.get_legend_handles_labels()
    lines2, labs2 = ax2r.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labs1 + labs2, fontsize=8, loc="center right")

    # Panel C — Attention-Net gate weights (top-k)
    if gate_mean is not None:
        ax = axes[2]
        ac = _model_color("Attention-Net")
        gate_top_idx  = np.argsort(gate_mean)[::-1][:top_k]
        gate_top_vals = gate_mean[gate_top_idx]
        gate_top_labs = [f"PC {i+1}" for i in gate_top_idx]
        gate_k        = len(gate_top_idx)   # clamp gate panel independently
        ax.barh(range(gate_k)[::-1], gate_top_vals, color=ac,
                edgecolor="k", lw=0.6, alpha=0.85, zorder=3)
        ax.set_yticks(range(gate_k)[::-1])
        ax.set_yticklabels(gate_top_labs, fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Mean Gate Weight  (sigmoid, 0→1)")
        ax.set_title("C   Attention-Net Sparse Gate Weights\n"
                     "(learned feature selection)", fontweight="bold",
                     color=ac)
        for i, v in enumerate(gate_top_vals[::-1]):
            ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=8)

    plt.tight_layout()
    _save(fig, "06_feature_importance")


# ── 9.7  Summary Dashboard ───────────────────────────────────────────────────
def plot_summary_dashboard(cv_results, km_results):
    print("[Plots] Figure 7 — Summary dashboard …")
    names  = list(cv_results.keys())
    means  = [cv_results[n]["mean"] for n in names]
    stds   = [cv_results[n]["std"]  for n in names]
    pvals  = [km_results[n]["pval"] for n in names]
    colors = [_model_color(n)       for n in names]
    n      = len(names)

    fig = plt.figure(figsize=(24, 14))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.55, wspace=0.42)

    # A — horizontal bar
    ax = fig.add_subplot(gs[0, 0])
    y  = np.arange(n)
    ax.barh(y, means, xerr=stds, color=colors, edgecolor="k", lw=0.6,
            height=0.55, capsize=5, alpha=0.87, zorder=3)
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=10)
    ax.axvline(0.5, color="gray", ls="--", lw=1.2)
    ax.set_xlim(0, 1); ax.set_xlabel("C-index (Harrell)")
    ax.set_title("A   Model Performance (C-index)", fontweight="bold")
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(m + s + 0.01, i, f"{m:.3f}", va="center", fontsize=10)

    # B — −log10 p-values
    ax = fig.add_subplot(gs[0, 1])
    nlp  = [-np.log10(p + 1e-300) for p in pvals]
    bars = ax.bar(names, nlp, color=colors, edgecolor="k", lw=0.6, alpha=0.87, zorder=3)
    ax.axhline(-np.log10(0.05),  color="red",    ls="--", lw=1.5, label="p = 0.05")
    ax.axhline(-np.log10(0.01),  color="orange", ls=":",  lw=1.2, label="p = 0.01")
    ax.axhline(-np.log10(0.001), color="purple", ls=":",  lw=1.2, label="p = 0.001")
    ax.set_ylabel("−log₁₀(p)"); ax.set_title("B   Log-rank Significance", fontweight="bold")
    ax.legend(fontsize=8)
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right", fontsize=9)
    for b, p in zip(bars, pvals):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.15,
                f"p={p:.1e}", ha="center", fontsize=7.5, rotation=20)

    # C — best model KM
    ax = fig.add_subplot(gs[0, 2])
    best = max(cv_results, key=lambda n: cv_results[n]["mean"])
    res  = km_results[best]
    T, E, high = res["T"], res["E"], res["high"]
    kmh = KaplanMeierFitter(); kml = KaplanMeierFitter()
    kmh.fit(T[high],  E[high],  label="High Risk")
    kml.fit(T[~high], E[~high], label="Low Risk")
    kmh.plot_survival_function(ax=ax, color=PALETTE["high"], ci_show=True, ci_alpha=0.12, lw=2)
    kml.plot_survival_function(ax=ax, color=PALETTE["low"],  ci_show=True, ci_alpha=0.12, lw=2)
    ax.set_title(f"C   Best Model: {best}\nLog-rank p = {res['pval']:.2e}  {_sig_star(res['pval'])}",
                 fontweight="bold", color=_model_color(best))
    ax.set_xlabel("Months"); ax.set_ylabel("Survival Probability")
    ax.set_ylim(0, 1.05)

    # D-F — per-fold CI bar for each model (3 per row)
    for idx, name in enumerate(names[:3]):
        ax = fig.add_subplot(gs[1, idx])
        fci = cv_results[name]["fold_ci"]
        ax.bar(range(1, len(fci)+1), fci, color=_model_color(name),
               edgecolor="k", lw=0.6, alpha=0.85, zorder=3)
        ax.axhline(np.mean(fci), color="k", ls="--", lw=1.8,
                   label=f"Mean = {np.mean(fci):.3f}", zorder=4)
        ax.axhline(0.5, color="gray", ls=":", lw=1)
        ax.set_ylim(0, 1); ax.set_xlabel("Fold"); ax.set_ylabel("C-index")
        ax.set_title(f"{'DEF'[idx]}   {name} · Per-Fold C-index", fontweight="bold",
                     color=_model_color(name))
        ax.set_xticks(range(1, len(fci)+1)); ax.legend(fontsize=8)

    fig.suptitle(
        "Deep Learning Survival Analysis — Summary Dashboard\n"
        "UCEC  ·  H-M-T Triplet Features  ·  TCGA",
        fontsize=16, fontweight="bold", y=1.02,
    )
    _save(fig, "07_summary_dashboard")


# ── 9.8  Calibration-style curves (risk quantile survival) ────────────────────
def plot_calibration_curves(km_results):
    """
    Divide predicted risk into tertiles and show KM curves for each.
    Provides a richer calibration view than binary median split.
    """
    print("[Plots] Figure 8 — Calibration curves (risk tertiles) …")
    names = list(km_results.keys())
    n     = len(names)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 6.5))
    if n == 1: axes = [axes]
    fig.suptitle("Survival by Predicted Risk Tertile (Calibration View)",
                 fontsize=14, fontweight="bold")

    tertile_colors = ["#C0392B", "#F39C12", "#2980B9"]
    tertile_labels = ["High (T3)", "Mid (T2)", "Low (T1)"]

    for ax, name in zip(axes, names):
        res = km_results[name]
        risk, T, E = res["risk"], res["T"], res["E"]

        q33, q67 = np.percentile(risk, [33, 67])
        masks = [risk >= q67, (risk >= q33) & (risk < q67), risk < q33]

        for mask, lbl, c in zip(masks, tertile_labels, tertile_colors):
            if mask.sum() < 5: continue
            kmf = KaplanMeierFitter()
            kmf.fit(T[mask], E[mask], label=f"{lbl} (n={mask.sum()})")
            kmf.plot_survival_function(ax=ax, color=c, ci_show=True,
                                       ci_alpha=0.12, lw=2)

        ax.set_title(f"{name}", fontsize=11, fontweight="bold",
                     color=_model_color(name))
        ax.set_xlabel("Time (Months)"); ax.set_ylabel("Survival Probability")
        ax.set_ylim(0, 1.05)

    plt.tight_layout()
    _save(fig, "08_calibration_curves")


# ══════════════════════════════════════════════════════════════════════════════
# 10. TEXT REPORT
# ══════════════════════════════════════════════════════════════════════════════

def write_report(cv_results, km_results, X, T, E):
    print("\n[6] Writing report …")
    best = max(cv_results, key=lambda n: cv_results[n]["mean"])
    w    = 72

    lines = [
        "=" * w,
        "  DEEP LEARNING SURVIVAL ANALYSIS",
        "  UCEC H-M-T Triplet Framework  ·  TCGA Cohort",
        "=" * w,
        "",
        "DATASET SUMMARY",
        f"  Samples              : {len(T)}",
        f"  Events (deaths)      : {int(E.sum())}  ({100*E.mean():.1f}%)",
        f"  Median OS            : {np.median(T):.1f} months",
        f"  Feature dims (PCA)   : {X.shape[1]}",
        "",
        f"CROSS-VALIDATION  ({N_FOLDS}-fold Stratified  ·  Harrell C-index)",
        "-" * w,
    ]
    for name in cv_results:
        r   = cv_results[name]
        fci = "  ".join(f"{v:.3f}" for v in r["fold_ci"])
        lines.append(f"  {name:<22}: {r['mean']:.4f} ± {r['std']:.4f}  "
                     f"[{fci}]")

    lines += ["", "KAPLAN-MEIER  (median risk split · log-rank test)", "-" * w]
    for name in km_results:
        p   = km_results[name]["pval"]
        sig = _sig_star(p)
        lines.append(f"  {name:<22}: p = {p:.2e}  {sig}")

    lines += [
        "",
        f"BEST MODEL  →  {best}",
        f"  C-index = {cv_results[best]['mean']:.4f} ± {cv_results[best]['std']:.4f}",
        f"  KM p-value = {km_results[best]['pval']:.2e}  "
        f"{_sig_star(km_results[best]['pval'])}",
        "",
        "OUTPUT FILES  (SVG + PDF in outputs/)",
        "  01_data_overview.{svg,pdf}      Data stats & expression heatmaps",
        "  02_training_curves.{svg,pdf}    Loss & C-index curves (all folds)",
        "  03_cv_results.{svg,pdf}         CV performance comparison",
        "  04_kaplan_meier.{svg,pdf}       KM survival curves (all models)",
        "  05_risk_distributions.{svg,pdf} Risk histograms & scatter",
        "  06_feature_importance.{svg,pdf} Permutation importance & PCA",
        "  07_summary_dashboard.{svg,pdf}  Full summary dashboard",
        "  08_calibration_curves.{svg,pdf} Tertile calibration curves",
        "  report.txt                      This report",
        "=" * w,
    ]
    text = "\n".join(lines)
    print(text)
    path = OUT_DIR / "report.txt"
    path.write_text(text, encoding="utf-8")
    print(f"\n  ✔ Report →  {path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 72)
    print("  Deep Learning Survival Analysis — UCEC H-M-T Triplet Framework")
    print("=" * 72)
    print(f"  Device  : {DEVICE}")
    print(f"  Data    : {DATA_DIR}")
    print(f"  Outputs : {OUT_DIR}")
    print(f"  Models  : {', '.join(build_model_configs(1).keys())}")

    # ── Validate paths ────────────────────────────────────────────────────────
    missing = [v for k, v in FILES.items() if not (DATA_DIR / v).exists()]
    if missing:
        print("\n[ERROR] Files not found:")
        for m in missing: print(f"  ✗  {DATA_DIR / m}")
        print("\nPlease update DATA_DIR and/or FILES at the top of this script.")
        sys.exit(1)

    # ── Pipeline ──────────────────────────────────────────────────────────────
    mirna_expr, mrna_expr, clinical, stad_sheets = load_data()
    X_df, clin_sub = extract_triplet_features(
        mirna_expr, mrna_expr, clinical, stad_sheets
    )
    X, T, E, scaler, pca = preprocess(X_df, clin_sub)

    # Populate global model registry with correct in_dim
    global MODEL_CONFIGS
    MODEL_CONFIGS = build_model_configs(X.shape[1])

    plot_data_overview(mirna_expr, mrna_expr, clinical, stad_sheets, X, E)

    cv_results, best_models, fold_histories = run_cv(X, T, E)

    plot_training_curves(fold_histories)
    plot_cv_results(cv_results)

    km_results = stratify_risk(best_models, X, T, E)
    plot_kaplan_meier(km_results)
    plot_risk_distributions(km_results)
    plot_feature_importance(best_models, X, T, E, pca)
    plot_summary_dashboard(cv_results, km_results)
    plot_calibration_curves(km_results)

    write_report(cv_results, km_results, X, T, E)

    print(f"\n✅  Pipeline complete.  All outputs saved to:\n    {OUT_DIR}\n")


if __name__ == "__main__":
    main()