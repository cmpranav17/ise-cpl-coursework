"""
generate_figures.py
--------------------
Run AFTER Evaluate.py has produced results/raw_results.csv and results/summary.csv.

Usage:
    python generate_figures.py

Outputs (saved to results/):
    fig1_mape_bar.png      — grouped bar chart LR vs RF (all datasets, log scale)
    fig2_pred_actual.png   — predicted vs actual scatter (4 representative datasets)
    fig3_error_analysis.png — relative improvement + LR-vs-RF log-log scatter
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

RESULTS_DIR = "results"
DATASETS_DIR = "datasets"
OUT_DIR = RESULTS_DIR
SEED = 0  # single representative split for scatter plots

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})


# ── Load results ─────────────────────────────────────────────────────────────

def load_summary():
    path = os.path.join(RESULTS_DIR, "summary.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Run Evaluate.py first — {path} not found.")
    return pd.read_csv(path)


def load_raw():
    path = os.path.join(RESULTS_DIR, "raw_results.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Run Evaluate.py first — {path} not found.")
    return pd.read_csv(path)


# ── Figure 1: Grouped bar chart LR vs RF, all datasets (log scale) ───────────

def plot_fig1(summary_df):
    lr = summary_df[summary_df["model"] == "LinearRegression"][["dataset", "MAPE_mean", "MAPE_std"]].rename(
        columns={"MAPE_mean": "lr_mean", "MAPE_std": "lr_std"})
    rf = summary_df[summary_df["model"] == "RandomForest"][["dataset", "MAPE_mean", "MAPE_std"]].rename(
        columns={"MAPE_mean": "rf_mean", "MAPE_std": "rf_std"})
    merged = pd.merge(lr, rf, on="dataset").sort_values("dataset")

    systems = [d.split("/")[0] for d in merged["dataset"]]
    labels  = [d.split("/")[1] for d in merged["dataset"]]
    n = len(merged)
    x = np.arange(n)
    w = 0.38

    fig, ax = plt.subplots(figsize=(max(14, n * 0.55), 5))
    ax.bar(x - w/2, merged["lr_mean"], w, yerr=merged["lr_std"], capsize=2,
           color="#4472C4", alpha=0.85, label="Linear Regression",
           error_kw={"linewidth": 0.8, "ecolor": "#1F3864"})
    ax.bar(x + w/2, merged["rf_mean"], w, yerr=merged["rf_std"], capsize=2,
           color="#ED7D31", alpha=0.85, label="Random Forest",
           error_kw={"linewidth": 0.8, "ecolor": "#843C0C"})

    ax.set_yscale("log")
    ax.set_ylabel("Mean MAPE (%, log scale)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=7)
    ax.set_title(
        "Figure 1: Mean MAPE (±std) — Linear Regression vs Random Forest across all datasets\n"
        "(log scale; error bars = ±1 std over 30 repeats)", pad=8)
    ax.legend(loc="upper right", fontsize=9)
    ax.yaxis.grid(True, linestyle="--", alpha=0.45, which="both")

    # System boundary lines + labels
    prev, start = None, 0
    boundaries = []
    for i, sys in enumerate(systems):
        if sys != prev:
            if prev is not None:
                boundaries.append((start, i - 1, prev))
            prev, start = sys, i
    boundaries.append((start, n - 1, prev))

    for (s, e, sys_name) in boundaries:
        if s > 0:
            ax.axvline(s - 0.5, color="#BBBBBB", linewidth=0.8, linestyle="--")
        mid = (s + e) / 2
        ax.text(mid, ax.get_ylim()[1] * 2.5, sys_name,
                ha="center", va="bottom", fontsize=7, color="#444", style="italic")

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "fig1_mape_bar.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


# ── Figure 2: Predicted vs Actual (real model outputs, one split) ─────────────

def load_dataset(key):
    system, workload = key.split("/", 1)
    system_path = os.path.join(DATASETS_DIR, system)

    print(f"\n[DEBUG] Looking for dataset:")
    print(f"  system: {system}")
    print(f"  workload: {workload}")
    print(f"  system_path: {system_path}")

    if not os.path.exists(system_path):
        print("  ❌ system folder not found")
        return None, None

    print("  files in folder:", os.listdir(system_path))

    for fname in os.listdir(system_path):
        if workload in fname:   
            print(f"  ✅ Found match: {fname}")
            path = os.path.join(system_path, fname)
            df = pd.read_csv(path)
            df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
            X = df.iloc[:, :-1]
            y = df.iloc[:, -1]
            return X, y

    print("  ❌ No matching file found")
    return None, None


def get_preds_for_dataset(key, model_names=("LinearRegression", "RandomForest")):
    """
    Fit models on one 70/30 split (seed=SEED) and return y_test, preds dict.
    Returns (y_test, {model_name: y_pred}) or None if dataset unavailable.
    """
    from models import get_models
    X, y = load_dataset(key)
    if X is None:
        return None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=SEED)

    models = get_models()
    preds = {}
    for name in model_names:
        if name not in models:
            continue
        try:
            models[name].fit(X_train, y_train)
            preds[name] = models[name].predict(X_test)
        except Exception as e:
            print(f"    [WARN] {name} on {key}: {e}")

    return y_test.values, preds


def plot_fig2(summary_df):
    """
    4-panel predicted vs actual from real model outputs.
    Picks 4 datasets illustrating different cases (same ones as report Table 3).
    Falls back gracefully if a dataset CSV is unavailable.
    """
    # (key, description)
    target_datasets = [
        ("batlik/village",      "High LR error, strong RF gain"),
        ("h2/tpcc-8",           "Both models similar"),
        ("xz/ambivert.wav.tar", "Large non-linear improvement"),
        ("z3/QF_NRA_hong_9",    "Near-linear; RF slightly worse"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle(
        "Figure 2: Predicted vs. Actual Performance — Representative Datasets\n"
        "(Real model outputs, single 70/30 split; LR = blue ○, RF = orange △)",
        fontsize=10, y=1.01)

    for ax, (key, desc) in zip(axes.flatten(), target_datasets):
        result = get_preds_for_dataset(key)
        if result is None:
            ax.text(0.5, 0.5, f"Dataset unavailable:\n{key}",
                    ha="center", va="center", transform=ax.transAxes, fontsize=8)
            ax.set_title(f"{key}\n({desc})", fontsize=8)
            continue

        y_test, preds = result
        lr_pred = preds.get("LinearRegression")
        rf_pred = preds.get("RandomForest")

        from sklearn.metrics import mean_absolute_percentage_error as mape_fn
        lr_mape = mape_fn(y_test, lr_pred) * 100 if lr_pred is not None else None
        rf_mape = mape_fn(y_test, rf_pred) * 100 if rf_pred is not None else None

        all_vals = [y_test]
        if lr_pred is not None: all_vals.append(lr_pred)
        if rf_pred is not None: all_vals.append(rf_pred)
        lo = min(v.min() for v in all_vals)
        hi = max(v.max() for v in all_vals)

        ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.5, label="Perfect")
        if lr_pred is not None:
            ax.scatter(y_test, lr_pred, s=10, alpha=0.4, color="#4472C4",
                       marker="o", label=f"LR (MAPE={lr_mape:.1f}%)")
        if rf_pred is not None:
            ax.scatter(y_test, rf_pred, s=10, alpha=0.4, color="#ED7D31",
                       marker="^", label=f"RF (MAPE={rf_mape:.1f}%)")

        ax.set_xlabel("Actual", fontsize=8)
        ax.set_ylabel("Predicted", fontsize=8)
        ax.set_title(f"{key}\n({desc})", fontsize=8.5)
        ax.legend(fontsize=7, markerscale=1.3)
        ax.tick_params(labelsize=7)

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "fig2_pred_actual.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


# ── Figure 3: Improvement bar + LR-vs-RF log scatter ─────────────────────────

def plot_fig3(summary_df):
    lr = summary_df[summary_df["model"] == "LinearRegression"][["dataset", "MAPE_mean"]].rename(
        columns={"MAPE_mean": "lr"})
    rf = summary_df[summary_df["model"] == "RandomForest"][["dataset", "MAPE_mean"]].rename(
        columns={"MAPE_mean": "rf"})
    merged = pd.merge(lr, rf, on="dataset")
    merged["improvement"] = (merged["lr"] - merged["rf"]) / merged["lr"] * 100
    merged["label"] = merged["dataset"].str.split("/").str[1]
    merged["system"] = merged["dataset"].str.split("/").str[0]
    merged = merged.sort_values("improvement")

    fig, axes = plt.subplots(1, 2, figsize=(13, max(5, len(merged) * 0.13 + 1)))

    # ── Left: horizontal improvement bar ──
    ax = axes[0]
    colors = ["#ED7D31" if v > 0 else "#4472C4" for v in merged["improvement"]]
    y_pos = np.arange(len(merged))
    ax.barh(y_pos, merged["improvement"], color=colors, alpha=0.85, height=0.7)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(merged["label"], fontsize=6.5)
    ax.set_xlabel("Relative MAPE Improvement over LR (%)")
    ax.set_title("Figure 3a: RF vs LR Relative MAPE Improvement\n(orange = RF better; blue = LR better)", fontsize=9)
    ax.xaxis.grid(True, linestyle="--", alpha=0.5)

    # ── Right: log-log scatter ──
    ax = axes[1]
    unique_sys = list(dict.fromkeys(merged["system"]))
    cmap = matplotlib.colormaps.get_cmap("tab10")
    sys_color = {s: cmap(i / max(len(unique_sys) - 1, 1)) for i, s in enumerate(unique_sys)}

    for _, row in merged.iterrows():
        ax.scatter(row["lr"], row["rf"], color=sys_color[row["system"]], s=45, zorder=3)

    lims = [merged[["lr", "rf"]].min().min() * 0.5,
            merged[["lr", "rf"]].max().max() * 2]
    lims = [max(lims[0], 0.05), lims[1]]
    ax.plot(lims, lims, "k--", lw=1, alpha=0.5)
    ax.fill_between(lims, [lims[0], lims[0]], lims, alpha=0.06, color="green")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("LR MAPE (%)")
    ax.set_ylabel("RF MAPE (%)")
    ax.set_title("Figure 3b: LR MAPE vs RF MAPE (log–log)\nBelow diagonal = RF outperforms LR", fontsize=9)
    patches = [mpatches.Patch(color=sys_color[s], label=s) for s in unique_sys]
    ax.legend(handles=patches, fontsize=7, ncol=2, loc="lower right")
    ax.xaxis.grid(True, linestyle="--", alpha=0.4, which="both")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, which="both")

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "fig3_error_analysis.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading results CSVs...")
    summary_df = load_summary()
    raw_df = load_raw()

    print("Generating Figure 1 (MAPE bar chart)...")
    plot_fig1(summary_df)

    print("Generating Figure 2 (predicted vs actual — real model outputs)...")
    plot_fig2(summary_df)

    print("Generating Figure 3 (improvement analysis)...")
    plot_fig3(summary_df)

    print("\nDone. Figures saved to results/")
    print("  fig1_mape_bar.png")
    print("  fig2_pred_actual.png")
    print("  fig3_error_analysis.png")


if __name__ == "__main__":
    main()