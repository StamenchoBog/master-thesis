"""Pre-processes all data partitions and saves them as .npz cache files.

Run by the `preprocessor` docker-compose service before the FL federation starts.

Step 1 — scans all CSV files to find the numeric columns present in every file.
         Saves that agreed-upon column list to columns.json so all nodes use
         identical features and the global model weights are always compatible.

Step 2 — preprocesses each partition and saves a compressed .npz cache so each
         round loads in ~1 second instead of re-reading CSV files.
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

DATA_DIR = os.getenv("DATA_DIR", "/app/data")
CACHE_DIR = os.path.join(DATA_DIR, ".cache")
COLUMNS_FILE = os.path.join(CACHE_DIR, "columns.json")
NUM_PARTITIONS = int(os.getenv("NUM_PARTITIONS", "3"))

DROP_COLS = ["ts", "src_ip", "dst_ip", "src_mac", "dst_mac", "Unnamed: 0", "type", "label"]

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
    df.drop(columns=[c for c in DROP_COLS if c in df.columns], inplace=True)
    numeric = set(df.select_dtypes(include=[np.number]).columns)
    common_cols = numeric if common_cols is None else common_cols.intersection(numeric)

common_cols = sorted(common_cols)
print(f"Found {len(common_cols)} common numeric columns: {common_cols}")

with open(COLUMNS_FILE, "w") as f:
    json.dump(common_cols, f, indent=2)

print(f"Column list saved to {COLUMNS_FILE}")

from data_loader import load_data  # noqa: E402

for partition_id in range(NUM_PARTITIONS):
    print(f"\n=== Step 2: Preprocessing partition {partition_id + 1}/{NUM_PARTITIONS} ===")
    load_data(partition_id, NUM_PARTITIONS)

print("\nAll partitions cached. Ready for FL run.")
