

import os
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_percentage_error,
    mean_absolute_error,
    root_mean_squared_error,
)
from scipy.stats import wilcoxon

from data_loader import load_all_datasets
from models import get_models

warnings.filterwarnings("ignore")

REPEATS = 30
TEST_SIZE = 0.30
RESULTS_DIR = "results"
# Gaussian Process can be slow on large datasets; skip it if >500 rows
GP_MAX_ROWS = 500



def evaluate_dataset(key, X, y, models, repeats=REPEATS):
 
    records = []
    n_rows = len(y)

    for repeat in range(repeats):
        seed = repeat  # deterministic but different each repeat
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=seed
        )

        for model_name, pipeline in models.items():
            # Skip GP on large datasets to keep runtime reasonable
            if model_name == "GaussianProcess" and n_rows > GP_MAX_ROWS:
                continue

            try:
                pipeline.fit(X_train, y_train)
                y_pred = pipeline.predict(X_test)

                mape = mean_absolute_percentage_error(y_test, y_pred) * 100
                mae = mean_absolute_error(y_test, y_pred)
                rmse = root_mean_squared_error(y_test, y_pred)

                records.append({
                    "dataset": key,
                    "model": model_name,
                    "repeat": repeat,
                    "MAPE": mape,
                    "MAE": mae,
                    "RMSE": rmse,
                })
            except Exception as e:
                print(f"  [WARN] {model_name} failed on {key} repeat {repeat}: {e}")

    return pd.DataFrame(records)


def run_wilcoxon(raw_df, baseline="LinearRegression"):

    stats_records = []
    proposed = [m for m in raw_df["model"].unique() if m != baseline]

    for dataset in raw_df["dataset"].unique():
        df_ds = raw_df[raw_df["dataset"] == dataset]
        base_mape = df_ds[df_ds["model"] == baseline]["MAPE"].values

        for model_name in proposed:
            prop_mape = df_ds[df_ds["model"] == model_name]["MAPE"].values

            if len(prop_mape) < 2 or len(base_mape) < 2:
                continue

            # Align lengths (GP may have been skipped)
            min_len = min(len(base_mape), len(prop_mape))
            b = base_mape[:min_len]
            p = prop_mape[:min_len]

            diff = b - p  # positive = proposed is better
            wins = int(np.sum(diff > 0))
            losses = int(np.sum(diff < 0))
            ties = int(np.sum(diff == 0))

            try:
                _, pval = wilcoxon(b, p)
            except ValueError:
                pval = 1.0  # identical arrays

            stats_records.append({
                "dataset": dataset,
                "proposed_model": model_name,
                "baseline_mean_MAPE": round(np.mean(b), 4),
                "proposed_mean_MAPE": round(np.mean(p), 4),
                "wins": wins,
                "ties": ties,
                "losses": losses,
                "wilcoxon_p": round(pval, 4),
                "significant": pval < 0.05,
            })

    return pd.DataFrame(stats_records)


def build_summary(raw_df):
    """
    Aggregate raw results to mean ± std per dataset × model.
    """
    summary = (
        raw_df.groupby(["dataset", "model"])[["MAPE", "MAE", "RMSE"]]
        .agg(["mean", "std"])
        .round(4)
    )
    summary.columns = ["_".join(c) for c in summary.columns]
    return summary.reset_index()


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    datasets = load_all_datasets()
    models = get_models()

    all_raw = []
    total = len(datasets)

    for i, (key, (X, y)) in enumerate(datasets.items(), 1):
        print(f"[{i}/{total}] Evaluating: {key}  (rows={len(y)})")
        df = evaluate_dataset(key, X, y, models)
        all_raw.append(df)

    raw_df = pd.concat(all_raw, ignore_index=True)
    raw_path = os.path.join(RESULTS_DIR, "raw_results.csv")
    raw_df.to_csv(raw_path, index=False)
    print(f"\nRaw results saved → {raw_path}")

    summary_df = build_summary(raw_df)
    summary_path = os.path.join(RESULTS_DIR, "summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary saved      → {summary_path}")

    stats_df = run_wilcoxon(raw_df)
    stats_path = os.path.join(RESULTS_DIR, "stats.csv")
    stats_df.to_csv(stats_path, index=False)
    print(f"Stats saved        → {stats_path}")

    # Quick console overview
    print("\n── Overall Win/Tie/Loss vs LinearRegression (MAPE) ──")
    for model_name in stats_df["proposed_model"].unique():
        sub = stats_df[stats_df["proposed_model"] == model_name]
        total_wins = sub["wins"].sum()
        total_losses = sub["losses"].sum()
        total_ties = sub["ties"].sum()
        sig = sub["significant"].sum()
        print(f"  {model_name}: W={total_wins} T={total_ties} L={total_losses}  "
              f"(significant on {sig}/{len(sub)} datasets)")

    print("\nDone! Check the 'results/' folder.")


if __name__ == "__main__":
    main()