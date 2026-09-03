"""Extract rich features from raw MAT files. Run before run_all_experiments.py"""
import sys; sys.path.insert(0,'.')
from config import DATA_DIR, BATTERIES, RESULTS_DIR
from src.feature_engineering import extract_rich_features
print("Extracting rich features...")
for bid, mf in BATTERIES.items():
    df = extract_rich_features(DATA_DIR/mf, bid)
    df.to_csv(RESULTS_DIR/f"{bid}_rich.csv", index=False)
    print(f"  {bid}: {len(df)} cycles, SOH {df.SOH.min():.2f}–{df.SOH.max():.2f}%")
print("Done. Rich feature CSVs saved to results/")
