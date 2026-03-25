
import os
import pandas as pd
import numpy as np


RESULTS_DIR = "results"


def print_mape_table(summary_df):
    """
    Prints a pivot table: rows = datasets, columns = model MAPE mean ± std
    """
    print("\n" + "=" * 80)
    print("TABLE: Mean MAPE (%) ± Std across 30 repeats")
    print("=" * 80)

    models = summary_df["model"].unique()
    pivot_rows = []

    for dataset in summary_df["dataset"].unique():
        row = {"Dataset": dataset}
        for model in models:
            sub = summary_df[
                (summary_df["dataset"] == dataset) &
                (summary_df["model"] == model)
            ]
            if sub.empty:
                row[model] = "N/A"
            else:
                mean = sub["MAPE_mean"].values[0]
                std = sub["MAPE_std"].values[0]
                row[model] = f"{mean:.2f} ± {std:.2f}"
        pivot_rows.append(row)

    pivot = pd.DataFrame(pivot_rows)
    print(pivot.to_string(index=False))


def print_stats_table(stats_df):
    """
    Prints the Wilcoxon test summary.
    """
    print("\n" + "=" * 80)
    print("TABLE: Wilcoxon Signed-Rank Test (proposed vs LinearRegression baseline)")
    print("Significant = p < 0.05  |  W=Wins  T=Ties  L=Losses  (per repeat, MAPE)")
    print("=" * 80)

    display_cols = [
        "dataset", "proposed_model",
        "baseline_mean_MAPE", "proposed_mean_MAPE",
        "wins", "ties", "losses",
        "wilcoxon_p", "significant"
    ]
    print(stats_df[display_cols].to_string(index=False))


def print_overall_summary(stats_df):
    """
    Prints an aggregate summary per model.
    """
    print("\n" + "=" * 80)
    print("OVERALL SUMMARY per proposed model vs LinearRegression")
    print("=" * 80)

    for model_name in stats_df["proposed_model"].unique():
        sub = stats_df[stats_df["proposed_model"] == model_name]
        n = len(sub)
        beats = (sub["proposed_mean_MAPE"] < sub["baseline_mean_MAPE"]).sum()
        sig = sub["significant"].sum()
        avg_improve = (
            (sub["baseline_mean_MAPE"] - sub["proposed_mean_MAPE"])
            / sub["baseline_mean_MAPE"] * 100
        ).mean()

        print(f"\n  Model: {model_name}")
        print(f"    Datasets evaluated  : {n}")
        print(f"    Better MAPE than LR : {beats}/{n}")
        print(f"    Statistically sig.  : {sig}/{n}  (Wilcoxon p<0.05)")
        print(f"    Avg MAPE improvement: {avg_improve:.1f}%")


def main():
    summary_path = os.path.join(RESULTS_DIR, "summary.csv")
    stats_path = os.path.join(RESULTS_DIR, "stats.csv")

    if not os.path.exists(summary_path) or not os.path.exists(stats_path):
        print("Results not found. Please run evaluate.py first.")
        return

    summary_df = pd.read_csv(summary_path)
    stats_df = pd.read_csv(stats_path)

    print_mape_table(summary_df)
    print_stats_table(stats_df)
    print_overall_summary(stats_df)


if __name__ == "__main__":
    main()