import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

DATA_DIR = os.getenv("DATA_DIR", "/app/data")
CACHE_DIR = os.path.join(DATA_DIR, ".cache")
COLUMNS_FILE = os.path.join(CACHE_DIR, "columns.json")
NUM_PARTITIONS = int(os.getenv("NUM_PARTITIONS", "3"))

METADATA_COLS = ["ts", "src_ip", "dst_ip", "src_mac", "dst_mac", "Unnamed: 0", "type"]

all_csvs = sorted(glob.glob(os.path.join(DATA_DIR, "**", "*.csv"), recursive=True))
if not all_csvs:
    raise FileNotFoundError(f"No CSV files found in {DATA_DIR}")

os.makedirs(CACHE_DIR, exist_ok=True)

partition_caches = [
    os.path.join(CACHE_DIR, f"partition_{i}_of_{NUM_PARTITIONS}.npz")
    for i in range(NUM_PARTITIONS)
]
if os.path.exists(COLUMNS_FILE) and all(os.path.exists(p) for p in partition_caches):
    print("All caches already exist. Nothing to do.")
    sys.exit(0)

print(f"=== Step 1: Finding common numeric columns across {len(all_csvs)} CSV files ===")
common_cols = None
for f in all_csvs:
    df = pd.read_csv(f, low_memory=False)
    df.drop(columns=[c for c in METADATA_COLS + ["label"] if c in df.columns], inplace=True)
    numeric = set(df.select_dtypes(include=[np.number]).columns)
    common_cols = numeric if common_cols is None else common_cols.intersection(numeric)

common_cols = sorted(common_cols)
print(f"Found {len(common_cols)} common numeric columns: {common_cols}")

with open(COLUMNS_FILE, "w") as f:
    json.dump(common_cols, f, indent=2)

print(f"Column list saved to {COLUMNS_FILE}")

for partition_id in range(NUM_PARTITIONS):
    print(f"\n=== Step 2: Preprocessing partition {partition_id + 1}/{NUM_PARTITIONS} ===")
    cache_path = os.path.join(CACHE_DIR, f"partition_{partition_id}_of_{NUM_PARTITIONS}.npz")

    assigned = np.array_split(all_csvs, NUM_PARTITIONS)[partition_id]
    print(f"  Files: {[os.path.basename(f) for f in assigned]}")

    df = pd.concat(
        [pd.read_csv(f, low_memory=False) for f in assigned],
        ignore_index=True,
    )
    df.drop(columns=[c for c in METADATA_COLS if c in df.columns], inplace=True)
    y = df.pop("label").values.astype(np.int64)

    available = [c for c in common_cols if c in df.columns]
    X = StandardScaler().fit_transform(df[available].values).astype(np.float32)

    np.savez_compressed(cache_path, X=X, y=y)
    print(f"  Saved: {cache_path}")

print("\nAll partitions cached. Ready for FL run.")
