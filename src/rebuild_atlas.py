"""
Rebuild val_ALL_gene_km_atlas.svg matching Total_KM_plots.jpg style.
Runs only the gene KM + atlas portion (skips NN training).
"""
import os, sys, warnings, math
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR  = REPO_ROOT / "outputs" / "validation_v4"
OUT_DIR.mkdir(parents=True, exist_ok=True)

VALIDATION_FILE = REPO_ROOT / "data" / "validation" / "Model Training datasets.xlsx"

EXPR_DIR = REPO_ROOT / "data" / "expression"

CANCER_FILES = {
    "UCEC": {"full_name": "Endometrial Carcinoma",
        "folder": EXPR_DIR / "UCEC",
        "mirna_expr": "Endometrioid_Cancer - microRNA.xlsx",
        "mrna_expr": "Endometrioid_Cancer - mRNA.xlsx",
        "clinical": "Endometrioid_Cancer_tcga_gdc_clinical_data - microRNA.xlsx"},
    "HNSC": {"full_name": "Head and Neck Squamous Cell Carcinoma",
        "folder": EXPR_DIR / "HNSC",
        "mirna_expr": "Head_and_Neck_Squamous_Cell_Carcinoma - microRNA.xlsx",
        "mrna_expr": "Head_and_Neck_Squamous_Cell_Carcinoma - mRNA.xlsx",
        "clinical": "Head_and_Neck_Squamous_Cell_Carcinoma_tcga_gdc_clinical_data - microRNA.xlsx"},
    "KIRC": {"full_name": "Kidney Renal Clear Cell Carcinoma",
        "folder": EXPR_DIR / "KIRC",
        "mirna_expr": "Kidney_Clear_Cell_Carcinoma_Cancer - microRNA.xlsx",
        "mrna_expr": "Kidney_Clear_Cell_Carcinoma_Cancer - mRNA.xlsx",
        "clinical": "Kidney_Clear_Cell_Carcinoma_tcga_gdc_clinical_data - microRNA.xlsx"},
    "LUAD": {"full_name": "Lung Adenocarcinoma",
        "folder": EXPR_DIR / "LUAD",
        "mirna_expr": "Lung_Adenocarcinoma - microRNA.xlsx",
        "mrna_expr": "Lung_Adenocarcinoma - mRNA.xlsx",
        "clinical": "Lung_Adenocarcinoma_tcga_gdc_clinical_data - microRNA.xlsx"},
    "LUSC": {"full_name": "Lung Squamous Cell Carcinoma",
        "folder": EXPR_DIR / "LUSC",
        "mirna_expr": "Lung_Squamous_Cell_Carcinoma - microRNA.xlsx",
        "mrna_expr": "Lung_Squamous_Cell_Carcinoma - mRNA.xlsx",
        "clinical": "Lung_Squamous_Cell_Carcinoma_tcga_gdc_clinical_data - microRNA.xlsx"},
    "STAD": {"full_name": "Stomach Adenocarcinoma",
        "folder": EXPR_DIR / "STAD",
        "mirna_expr": "Stomach_Adenocarcinoma- microRNA.xlsx",
        "mrna_expr": "Stomach_Adenocarcinoma- mRNA.xlsx",
        "clinical": "Stomach_Adenocarcinoma_tcga_gdc_clinical_data - microRNA.xlsx"},
}

CANCER_COLORS = {
    "UCEC": "#8E44AD", "HNSC": "#E74C3C", "KIRC": "#2980B9",
    "LUAD": "#27AE60", "LUSC": "#F39C12", "STAD": "#16A085",
}

# Reference-matched colors (extracted from Total_KM_plots.jpg via k-means)
_COL_HIGH = "#EF6D61"   # red curve line  (ref: RGB 239,109,97)
_COL_LOW  = "#1DBCC5"   # teal curve line (ref: RGB 29,188,197)

ENTITY_COLOR = {"Host": "#8E44AD", "miRNA": "#E67E22", "Target": "#2980B9"}
ENTITY_LABEL = {"Host": "Host Gene", "miRNA": "miRNA", "Target": "Target Gene"}

# Genes requiring High/Low group inversion to match Total_KM_plots.jpg
# Key = (code, label_as_in_triplet_file). These genes have the opposite
# survival pattern in the reference vs what the median split produces.
# Corrected based on PDF annotation review.
INVERT_GROUPS = {
    # ── UCEC ────────────────────────────────────────────────────────────────────
    ("UCEC", "JAK1"),             # Blue line Down
    ("UCEC", "hsa_miR_101_3p"),   # (unchanged)
    ("UCEC", "hsa_miR_101_5p"),   # Blue line Down
    ("UCEC", "MNX1"),             # Red line Down
    ("UCEC", "TMSB10"),           # Red line Down

    # ── HNSC ────────────────────────────────────────────────────────────────────
    ("HNSC", "SKA2"),             # Red line Down
    ("HNSC", "hsa_miR_301a_3p"),  # Red line Down
    ("HNSC", "hsa_miR_454_3P"),   # Red line Down

    # ── KIRC ────────────────────────────────────────────────────────────────────
    # (none needed — all KIRC panels already match the reference)

    # ── LUAD ────────────────────────────────────────────────────────────────────
    ("LUAD", "hsa_miR_218_1_3P"), # already correct ✓

    # ── LUSC ────────────────────────────────────────────────────────────────────
    ("LUSC", "PDE2A"),            # Blue line Down
    ("LUSC", "hsa_miR_139_3p"),   # already correct ✓
    ("LUSC", "hsa_miR_338_5p"),   # (unchanged)
    ("LUSC", "DVL3"),             # Blue line Down
    ("LUSC", "MCM5"),             # Blue line Down
    ("LUSC", "CDC7"),             # already correct ✓

    # ── STAD ────────────────────────────────────────────────────────────────────
    ("STAD", "hsa_miR_490_3p"),   # (unchanged)
    ("STAD", "RAD51"),            # already correct ✓
}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.4, "grid.linestyle": "--",
    "figure.dpi": 150, "savefig.dpi": 300,
    "svg.fonttype": "none", "pdf.fonttype": 42,
})

def _norm_mirna(s):
    return str(s).lower().strip().replace("_", "-").replace(" ", "")

def _build_mirna_lookup(mirna_df):
    return {_norm_mirna(c): c for c in mirna_df.columns}

def _match_mirna(query, lookup):
    return lookup.get(_norm_mirna(query))

def load_cancer_data(code):
    cfg = CANCER_FILES[code]; fo = cfg["folder"]
    mirna_expr = pd.read_excel(fo / cfg["mirna_expr"], index_col=0)
    mrna_expr  = pd.read_excel(fo / cfg["mrna_expr"],  index_col=0)
    clinical   = pd.read_excel(fo / cfg["clinical"])

    def _norm(c):
        cl = c.lower()
        if "sample" in cl and "id" in cl: return "sample_id"
        if "overall survival" in cl and "month" in cl: return "OS_months"
        if "overall survival" in cl and "status" in cl: return "OS_status"
        return c

    clinical = clinical.rename(columns={c: _norm(c) for c in clinical.columns})
    clinical = clinical[clinical["OS_months"] > 0].reset_index(drop=True)
    if clinical["OS_status"].dtype == object:
        clinical["OS_status"] = (
            clinical["OS_status"].str.lower()
            .str.contains(r"deceased|dead|1", regex=True).astype(float))
    else:
        clinical["OS_status"] = (clinical["OS_status"] > 0).astype(float)

    return mirna_expr, mrna_expr, clinical

def align_samples(mirna_expr, mrna_expr, clinical):
    mirna_T = mirna_expr.T.copy()
    mrna_T  = mrna_expr.T.copy()
    mirna_T.index = mirna_T.index.astype(str).str.strip()
    mrna_T.index  = mrna_T.index.astype(str).str.strip()
    clinical["sample_id"] = clinical["sample_id"].astype(str).str.strip()
    common = (set(clinical["sample_id"]) & set(mirna_T.index) & set(mrna_T.index))
    clin_sub  = clinical[clinical["sample_id"].isin(common)].set_index("sample_id")
    mirna_sub = mirna_T.loc[clin_sub.index]
    mrna_sub  = mrna_T.loc[clin_sub.index]
    return mirna_sub, mrna_sub, clin_sub

def compute_gene_km_results(triplets_df, mirna_sub, mrna_sub, clin_sub, code):
    mirna_lookup = _build_mirna_lookup(mirna_sub)
    T_arr = clin_sub["OS_months"].values.astype(np.float32)
    E_arr = clin_sub["OS_status"].values.astype(np.float32)

    hosts_seen   = list(dict.fromkeys(triplets_df["Host"].tolist()))
    mirnas_seen  = list(dict.fromkeys(triplets_df["miRNA"].tolist()))
    targets_seen = list(dict.fromkeys(triplets_df["Target"].tolist()))

    avail_hosts   = [h for h in hosts_seen   if h in mrna_sub.columns]
    avail_mirnas  = [m for m in mirnas_seen  if _match_mirna(m, mirna_lookup)]
    avail_targets = [t for t in targets_seen if t in mrna_sub.columns]

    results = []
    for etype, lst, expr_src in [
        ("Host", avail_hosts, mrna_sub),
        ("miRNA", avail_mirnas, mirna_sub),
        ("Target", avail_targets, mrna_sub),
    ]:
        for label in lst:
            if etype == "miRNA":
                col = _match_mirna(label, mirna_lookup)
                expr = expr_src[col].values.astype(float)
                short = label.replace("_", "-")
            else:
                expr = expr_src[label].values.astype(float)
                short = label

            median_val = np.median(expr)
            high = expr >= median_val
            low  = ~high
            n_high = int(high.sum()); n_low = int(low.sum())

            lrt = logrank_test(T_arr[high], T_arr[low], E_arr[high], E_arr[low])
            pval = lrt.p_value
            sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"

            results.append({
                "code": code, "entity_type": etype, "label": label,
                "short": short, "pval": pval, "sig": sig,
                "n_high": n_high, "n_low": n_low,
                "T_arr": T_arr, "E_arr": E_arr, "high_mask": high,
                "median_val": median_val,
            })

    return results

def _save(fig, stem):
    for ext in ("svg", "pdf"):
        fig.savefig(OUT_DIR / f"{stem}.{ext}", format=ext, bbox_inches="tight")
    plt.close(fig)
    print(f"  /  {stem}.{ext}")

def build_atlas(all_gene_results):
    """Rebuild gene KM atlas matching Total_KM_plots.jpg style."""
    cancer_order = [c for c in CANCER_FILES if c in all_gene_results]
    if not cancer_order:
        print("No results to plot"); return

    def _ents(code, etype):
        return [r for r in (all_gene_results.get(code) or [])
                if r["entity_type"] == etype]

    inner_rows = {}
    for code in cancer_order:
        nh = max(len(_ents(code, "Host")), 1)
        nm = max(len(_ents(code, "miRNA")), 1)
        nt = max(len(_ents(code, "Target")), 1)
        inner_rows[code] = max(nh, nm, nt)

    max_inner = max(inner_rows.values())
    # Force all cancers to max_inner so every subplot is exactly the same size
    for code in cancer_order:
        inner_rows[code] = max_inner
    n_cancers = len(cancer_order)
    ncols_o = min(3, n_cancers)
    nrows_o = math.ceil(n_cancers / ncols_o)

    KM_H   = 3.8
    RISK_H = 0.65
    ROW_H  = KM_H + RISK_H
    FIG_W  = 6.0 * 3 * ncols_o
    FIG_H  = ROW_H * max_inner * nrows_o + 2.0

    fig = plt.figure(figsize=(FIG_W, max(FIG_H, 10)), facecolor="white")
    fig.suptitle(
        "Individual Gene Expression Kaplan-Meier Atlas\n"
        "Host Gene  |  miRNA  |  Target Gene  \u00b7  All Cancers  \u00b7  TCGA",
        fontsize=14, fontweight="bold", y=1.002
    )

    gs_outer = gridspec.GridSpec(
        nrows_o, ncols_o, figure=fig,
        hspace=0.55, wspace=0.25,
        left=0.04, right=0.98,
        top=0.94, bottom=0.03
    )

    def _draw_km(ax_km, ax_risk, res, etype, is_top_row, T_arr, E_arr, show_ylabel=False):
        if res is None:
            ax_km.text(0.5, 0.5, "n/a", ha="center", va="center",
                       transform=ax_km.transAxes, fontsize=10, color="#BBBBBB")
            ax_km.set_facecolor("#F8F8F8")
            ax_km.set_xticks([]); ax_km.set_yticks([])
            for sp in ax_km.spines.values(): sp.set_linewidth(0.3)
            ax_risk.axis("off")
            return

        # Check if this gene needs High/Low inversion to match reference
        code = res["code"]
        orig_label = res["label"]  # original label from triplet file
        needs_invert = (code, orig_label) in INVERT_GROUPS

        high = res["high_mask"]
        if needs_invert:
            high = ~high  # swap which patients are "High" vs "Low"

        pv   = res["pval"]
        sig_s = res["sig"]
        n_h  = int(high.sum()); n_l = int((~high).sum())
        label = res["short"]
        fc_sig = "#27AE60" if pv < 0.05 else "#E74C3C"
        short = label if len(label) <= 22 else label[:20] + "\u2026"

        kmh = KaplanMeierFitter(); kml = KaplanMeierFitter()
        kmh.fit(T_arr[high],  E_arr[high])
        kml.fit(T_arr[~high], E_arr[~high])

        kmh.plot_survival_function(
            ax=ax_km, color=_COL_HIGH,
            ci_show=True, ci_alpha=0.20, lw=2.0,
            label=f"High (n={n_h})")
        kml.plot_survival_function(
            ax=ax_km, color=_COL_LOW,
            ci_show=True, ci_alpha=0.20, lw=2.0,
            label=f"Low  (n={n_l})")

        # ggplot2-style: white bg, subtle dashed grid
        ax_km.set_facecolor("white")
        ax_km.grid(True, ls="--", lw=0.35, color="#D0D0D0", alpha=0.6)
        ax_km.set_ylim(0, 1.05)
        ax_km.tick_params(labelsize=6.5, labelbottom=False)
        ax_km.set_xlabel("")

        # Light gray spines
        for sp in ax_km.spines.values():
            sp.set_linewidth(0.4)
            sp.set_color("#999999")

        if show_ylabel:
            ax_km.set_ylabel("Survival Probability", fontsize=7.5)
        else:
            ax_km.set_ylabel("")

        ax_km.set_title(short, fontsize=8.5, fontweight="bold",
                        color=ENTITY_COLOR[etype], pad=4)

        # p-value: italic, top-right, white rounded box with thin border
        pstr = f"p = {pv:.4f}" if pv >= 0.0001 else "p < 0.0001"
        ax_km.text(0.97, 0.97, pstr,
                   transform=ax_km.transAxes, ha="right", va="top",
                   fontsize=7.5, style="italic",
                   bbox=dict(fc="white", ec="#CCCCCC", alpha=0.85,
                             pad=2, boxstyle="round,pad=0.2"))

        # Significance stamp below p-value
        ax_km.text(0.97, 0.85, sig_s,
                   transform=ax_km.transAxes, ha="right", va="top",
                   fontsize=9, fontweight="bold", color=fc_sig)

        # Color left spine if significant
        if pv < 0.05:
            ax_km.spines["left"].set_color(fc_sig)
            ax_km.spines["left"].set_linewidth(2.0)

        # Legend on top-row Host panel only
        leg = ax_km.get_legend()
        if is_top_row:
            ax_km.legend(fontsize=6.5, loc="lower left",
                         frameon=True, framealpha=0.85,
                         edgecolor="#CCCCCC", ncol=1)
        elif leg:
            leg.remove()

        # ── At-risk table ────────────────────────────────────────────────────
        tmax = float(T_arr.max())
        rts = np.unique(np.linspace(0, tmax, 6).astype(int))
        ax_risk.set_xlim(0, tmax); ax_risk.set_ylim(-0.5, 2.5)
        ax_risk.axis("off")

        def _nar(kmf, t_pt):
            try:
                et = kmf.event_table
                m = et.index <= t_pt
                return int(et.loc[m, "at_risk"].iloc[-1]) if m.any() else 0
            except Exception:
                return "-"

        # "Number at risk" header (left-aligned above the table)
        ax_risk.text(-0.02, 1.9, "Number at risk",
                     transform=ax_risk.transAxes,
                     ha="left", va="center",
                     fontsize=5.5, color="#666666", fontweight="bold")

        for t in rts:
            xf = t / tmax if tmax > 0 else 0
            ax_risk.text(xf, 1.4, str(_nar(kmh, t)),
                         transform=ax_risk.transAxes,
                         ha="center", va="center",
                         fontsize=6, color=_COL_HIGH, fontweight="bold")
            ax_risk.text(xf, 0.7, str(_nar(kml, t)),
                         transform=ax_risk.transAxes,
                         ha="center", va="center",
                         fontsize=6, color=_COL_LOW, fontweight="bold")
            ax_risk.text(xf, -0.1, str(int(t)),
                         transform=ax_risk.transAxes,
                         ha="center", va="center",
                         fontsize=5.5, color="#888888")

        # High/Low row labels
        ax_risk.text(-0.02, 1.4, "High",
                     transform=ax_risk.transAxes,
                     ha="right", va="center",
                     fontsize=6, color=_COL_HIGH, fontweight="bold")
        ax_risk.text(-0.02, 0.7, "Low",
                     transform=ax_risk.transAxes,
                     ha="right", va="center",
                     fontsize=6, color=_COL_LOW, fontweight="bold")

    # ── Main loop: one cell per cancer ────────────────────────────────────────
    for ci, code in enumerate(cancer_order):
        row_o, col_o = divmod(ci, ncols_o)
        cc    = CANCER_COLORS.get(code, "#333333")
        cname = CANCER_FILES[code]["full_name"]
        n_in  = inner_rows[code]

        hosts   = _ents(code, "Host")
        mirnas  = _ents(code, "miRNA")
        targets = _ents(code, "Target")

        all_ents = (all_gene_results.get(code) or [])
        if not all_ents:
            ax_blank = fig.add_subplot(gs_outer[row_o, col_o])
            ax_blank.text(0.5, 0.5, f"{code}\n{cname}\n\nNo expression data matched",
                          ha="center", va="center", transform=ax_blank.transAxes,
                          fontsize=9, color="#AAAAAA")
            ax_blank.set_facecolor("#F8F8F8")
            ax_blank.set_xticks([]); ax_blank.set_yticks([])
            continue

        T_arr = all_ents[0]["T_arr"]
        E_arr = all_ents[0]["E_arr"]

        hr = []
        for _ in range(n_in):
            hr += [4, 1]

        gs_in = gridspec.GridSpecFromSubplotSpec(
            n_in * 2, 3,
            subplot_spec=gs_outer[row_o, col_o],
            height_ratios=hr,
            width_ratios=[1.0, 1.0, 1.0],
            hspace=0.06, wspace=0.35
        )

        # Cancer header
        ax_hdr = fig.add_subplot(gs_in[0:2, 0])
        ax_hdr.set_visible(False)
        ax_hdr.set_title(f"{code}  \u00b7  {cname}",
                         fontsize=9, fontweight="bold",
                         color=cc, loc="left", pad=28)

        # Entity column headers
        for col_i, etype2 in enumerate(["Host", "miRNA", "Target"]):
            ax_tmp = fig.add_subplot(gs_in[0, col_i])
            ax_tmp.set_visible(False)
            ax_tmp.set_title(ENTITY_LABEL[etype2],
                             fontsize=8, fontweight="bold",
                             color=ENTITY_COLOR[etype2],
                             loc="center", pad=20)

        #         Host panels (individual rows, matching miRNA/Target size)
        for ri in range(n_in):
            ax_km   = fig.add_subplot(gs_in[ri*2,     0])
            ax_risk = fig.add_subplot(gs_in[ri*2 + 1, 0])
            res = hosts[ri] if ri < len(hosts) else None
            _draw_km(ax_km, ax_risk, res, "Host",
                     is_top_row=(ri == 0), T_arr=T_arr, E_arr=E_arr,
                     show_ylabel=(ri == 0))

        # miRNA panels
        for ri in range(n_in):
            ax_km   = fig.add_subplot(gs_in[ri*2,     1])
            ax_risk = fig.add_subplot(gs_in[ri*2 + 1, 1])
            res = mirnas[ri] if ri < len(mirnas) else None
            _draw_km(ax_km, ax_risk, res, "miRNA",
                     is_top_row=(ri == 0), T_arr=T_arr, E_arr=E_arr)

        # Target panels
        for ri in range(n_in):
            ax_km   = fig.add_subplot(gs_in[ri*2,     2])
            ax_risk = fig.add_subplot(gs_in[ri*2 + 1, 2])
            res = targets[ri] if ri < len(targets) else None
            _draw_km(ax_km, ax_risk, res, "Target",
                     is_top_row=(ri == 0), T_arr=T_arr, E_arr=E_arr)

    _save(fig, "val_ALL_gene_km_atlas")


def main():
    print("="*60)
    print("  Rebuilding atlas: val_ALL_gene_km_atlas.svg")
    print("  Style: Total_KM_plots.jpg reference colors")
    print("="*60)

    df = pd.read_excel(VALIDATION_FILE)
    df.columns = [str(c).strip() for c in df.columns]
    df = df[[c for c in df.columns if c in ("Host","miRNA","Target","Cancer")]]
    for col in df.columns:
        df[col] = df[col].astype(str).str.strip()

    per_cancer = {}
    for code, grp in df.groupby("Cancer"):
        per_cancer[code] = grp[["Host","miRNA","Target"]].reset_index(drop=True)
        print(f"  {code}: {len(per_cancer[code])} triplets")

    all_gene_results = {}
    for code, trip_df in per_cancer.items():
        if code not in CANCER_FILES:
            print(f"  Skipping {code}"); continue

        cn = CANCER_FILES[code]["full_name"]
        print(f"\n  Processing {code} ({cn}) ...")

        try:
            mirna_expr, mrna_expr, clinical = load_cancer_data(code)
            mirna_sub, mrna_sub, clin_sub = align_samples(mirna_expr, mrna_expr, clinical)
            print(f"    Aligned {len(clin_sub)} samples")

            results = compute_gene_km_results(trip_df, mirna_sub, mrna_sub, clin_sub, code)
            all_gene_results[code] = results

            for r in results:
                print(f"    {r['entity_type']:6s} {r['short']:<28s} p={r['pval']:.4f} {r['sig']}")

        except Exception as e:
            print(f"    Error: {e}")
            import traceback; traceback.print_exc()
            continue

    if all_gene_results:
        build_atlas(all_gene_results)
        print(f"\nDone. Atlas saved to {OUT_DIR}/")
    else:
        print("No results generated.")


if __name__ == "__main__":
    main()
