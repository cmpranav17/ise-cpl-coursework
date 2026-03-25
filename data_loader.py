
import os
import pandas as pd


def load_all_datasets(datasets_path="datasets"):
    """
    Walks the datasets directory and returns a dict of:
        { "system/workload": (X, y) }
    e.g. { "batlik/corona": (DataFrame, Series) }
    """
    data = {}
    for system in sorted(os.listdir(datasets_path)):
        system_path = os.path.join(datasets_path, system)
        if not os.path.isdir(system_path):
            continue
        for fname in sorted(os.listdir(system_path)):
            if not fname.endswith(".csv"):
                continue
            fpath = os.path.join(system_path, fname)
            df = pd.read_csv(fpath)
            # Drop any unnamed index columns
            df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
            # Last column = performance (y), rest = config options (X)
            X = df.iloc[:, :-1]
            y = df.iloc[:, -1]
            key = f"{system}/{fname.replace('.csv', '')}"
            data[key] = (X, y)
    print(f"Loaded {len(data)} datasets from '{datasets_path}'")
    return data


if __name__ == "__main__":
    datasets = load_all_datasets()
    for k, (X, y) in list(datasets.items())[:3]:
        print(f"{k}: X={X.shape}, y={y.shape}")