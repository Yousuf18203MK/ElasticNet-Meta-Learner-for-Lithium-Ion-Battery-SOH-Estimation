"""
run_all_experiments.py — Authoritative experiment runner.

Architecture:  GRU + LSTM + XGBoost  →  ElasticNet stacking meta-learner

Execution
---------
Run from the project root directory:

    python experiments/run_all_experiments.py

Or open in PyCharm and run directly — the sys.path setup handles both cases.

Output files (all written to results/)
--------------------------------------
{cell}_rich.csv                    – Rich feature CSV (extracted if missing)
{cell}_predictions_s{seed}.csv     – Per-seed test-set predictions (seed=42)
{cell}_s{seed}_experiment.json     – Per-seed experiment metadata
final_results_raw.csv              – All metrics, all cells, all seeds
final_overall.csv                  – Mean ± std across cells and seeds
logs/{cell}_s{seed}_{model}.csv    – Training history per model
"""

import sys
import os
import json
import time
import random
import warnings
from pathlib import Path

# ── resolve project root so imports work whether the script is launched from
#    the project root, from the experiments/ subfolder, or via PyCharm ────────
_SCRIPT_DIR  = Path(__file__).resolve().parent          # .../experiments/
PROJECT_ROOT = _SCRIPT_DIR.parent                       # .../authoritative/
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"]   = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"]  = "0"

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.callbacks import EarlyStopping

# ── project imports ───────────────────────────────────────────────────────────
from config import (
    DATA_DIR, RESULTS_DIR, LOGS_DIR,
    BATTERIES, SEEDS, FEATURE_COLS, SEQ_LEN,
    FRAC_BASE,
    GRU_UNITS, LSTM_UNITS, DROPOUT, LR, EPOCHS, BATCH_SIZE, PATIENCE,
    XGB_N_LAGS, XGB_N_ESTIMATORS, XGB_MAX_DEPTH,
    XGB_LEARNING_RATE, XGB_SUBSAMPLE, XGB_COLSAMPLE,
    RIDGE_ALPHA, EN_ALPHA, EN_L1_RATIO, EN_MAX_ITER,
)
from src.feature_engineering import (
    extract_rich_features,
    build_lag_features,
    build_sequences,
)
from src.models import BUILDERS, build_xgboost
from src.metrics import all_metrics


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(s: int) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(s)
    np.random.seed(s)
    tf.random.set_seed(s)


def inverse_transform(scaler: MinMaxScaler, arr: np.ndarray) -> np.ndarray:
    """Inverse-transform a 1-D scaled array back to SOH %."""
    return scaler.inverse_transform(arr.reshape(-1, 1)).reshape(-1)


def chronological_masks(cycle_index: np.ndarray, total_cycles: int):
    """
    Four-way chronological split.

    base  : cycles 1 – 60%          (base training)
    early : cycles 60% – 70%        (early-stopping validation — NEVER metrics)
    meta  : cycles 70% – 85%        (meta-learner training)
    test  : cycles 85% – 100%       (final untouched evaluation)
    """
    b = int(total_cycles * 0.60)
    e = int(total_cycles * 0.70)
    m = int(total_cycles * 0.85)
    return (
        cycle_index <= b,
        (cycle_index > b) & (cycle_index <= e),
        (cycle_index > e) & (cycle_index <= m),
        cycle_index > m,
    )


def train_keras_model(model, Xb, yb_s, Xv, yv_s):
    """Train a Keras model with EarlyStopping on the early-val partition."""
    cb = EarlyStopping(
        monitor="val_loss",
        patience=PATIENCE,
        restore_best_weights=True,
        verbose=0,
    )
    history = model.fit(
        Xb, yb_s,
        validation_data=(Xv, yv_s),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[cb],
        verbose=0,
    )
    return model, history


def ensure_rich_features(battery_id: str, mat_path: Path) -> pd.DataFrame:
    """Load rich features from CSV cache, or extract and cache them."""
    cache_path = RESULTS_DIR / f"{battery_id}_rich.csv"
    if cache_path.exists():
        return pd.read_csv(str(cache_path))
    print(f"  Extracting rich features for {battery_id} …")
    df = extract_rich_features(mat_path, battery_id)
    df.to_csv(str(cache_path), index=False)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Per-battery experiment
# ─────────────────────────────────────────────────────────────────────────────

def run_battery(battery_id: str, mat_path: Path, all_rows: list) -> None:
    """
    Run the full GRU + LSTM + XGBoost → EN Stack pipeline for one battery
    across all evaluation seeds. Appends result dicts to all_rows.
    """
    # ── 1. load rich features ─────────────────────────────────────────────────
    df  = ensure_rich_features(battery_id, mat_path)
    tot = len(df)
    be  = int(tot * FRAC_BASE)

    print(f"\n  {battery_id}: {tot} discharge cycles")

    # ── 2. scalers (fitted on base partition ONLY) ────────────────────────────
    fsc = MinMaxScaler()
    fsc.fit(df.iloc[:be][FEATURE_COLS].values.astype(np.float32))

    tsc = MinMaxScaler()
    tsc.fit(df.iloc[:be][["SOH"]].values.astype(np.float32))

    # ── 3. XGBoost lag features (computed once, all cycles) ───────────────────
    lag_raw = build_lag_features(df, FEATURE_COLS, n_lags=XGB_N_LAGS)
    lag_sc  = MinMaxScaler()
    lag_sc.fit(lag_raw[:be])
    lag_norm = lag_sc.transform(lag_raw)
    soh_arr  = df["SOH"].values.astype(np.float32)
    cyc_arr  = df["Cycle"].values.astype(np.int32)

    bm_x, em_x, mm_x, tm_x = chronological_masks(cyc_arr, tot)

    # ── 4. LSTM / GRU sequences ───────────────────────────────────────────────
    X_seq, y_seq, c_seq = build_sequences(df, fsc, FEATURE_COLS, SEQ_LEN)
    bm_s, em_s, mm_s, tm_s = chronological_masks(c_seq, tot)

    Xb_s = X_seq[bm_s];  yb_s_raw = y_seq[bm_s]
    Xv_s = X_seq[em_s];  yv_s_raw = y_seq[em_s]

    yb_s = tsc.transform(yb_s_raw.reshape(-1, 1)).astype(np.float32)
    yv_s = tsc.transform(yv_s_raw.reshape(-1, 1)).astype(np.float32)

    input_shape = (X_seq.shape[1], X_seq.shape[2])

    # ── 5. Align meta / test cycles across XGB and sequence branches ──────────
    meta_cycles = sorted(set(cyc_arr[mm_x].tolist()) & set(c_seq[mm_s].tolist()))
    test_cycles = sorted(set(cyc_arr[tm_x].tolist()) & set(c_seq[tm_s].tolist()))

    if len(meta_cycles) < 3:
        print(f"  WARNING: {battery_id} has only {len(meta_cycles)} aligned "
              f"meta cycles — results may be unstable.")

    y_meta = soh_arr[np.isin(cyc_arr, meta_cycles)]
    y_test = soh_arr[np.isin(cyc_arr, test_cycles)]

    print(f"    Partition sizes — base:{bm_s.sum()}  early:{em_s.sum()}  "
          f"meta:{len(meta_cycles)}  test:{len(test_cycles)}")

    # ── 6. Loop over seeds ────────────────────────────────────────────────────
    first_seed_preds = None   # saved for seed=SEEDS[0] predictions CSV

    for seed in SEEDS:
        set_seed(seed)
        t0 = time.time()

        meta_preds = {}   # model_name → predictions on meta cycles
        test_preds = {}   # model_name → predictions on test cycles

        # ── XGBoost ──────────────────────────────────────────────────────────
        xgb_model = build_xgboost(
            seed             = seed,
            n_estimators     = XGB_N_ESTIMATORS,
            max_depth        = XGB_MAX_DEPTH,
            learning_rate    = XGB_LEARNING_RATE,
            subsample        = XGB_SUBSAMPLE,
            colsample_bytree = XGB_COLSAMPLE,
        )
        xgb_model.fit(lag_norm[bm_x], soh_arr[bm_x])
        meta_preds["XGB"] = xgb_model.predict(lag_norm[np.isin(cyc_arr, meta_cycles)])
        test_preds["XGB"] = xgb_model.predict(lag_norm[np.isin(cyc_arr, test_cycles)])

        # ── Keras base learners (GRU, LSTM) ──────────────────────────────────
        for name, builder in BUILDERS.items():
            model = builder(input_shape, units=GRU_UNITS, dropout=DROPOUT, lr=LR)
            model, history = train_keras_model(model, Xb_s, yb_s, Xv_s, yv_s)

            # Save training history
            hist_path = LOGS_DIR / f"{battery_id}_s{seed}_{name}.csv"
            pd.DataFrame(history.history).to_csv(str(hist_path), index=False)

            meta_preds[name] = inverse_transform(
                tsc, model.predict(X_seq[np.isin(c_seq, meta_cycles)], verbose=0)
            )
            test_preds[name] = inverse_transform(
                tsc, model.predict(X_seq[np.isin(c_seq, test_cycles)], verbose=0)
            )

        # ── Ensemble variants ─────────────────────────────────────────────────
        names    = list(meta_preds)   # ["XGB", "GRU", "LSTM"]
        meta_mat = np.column_stack([meta_preds[n] for n in names])
        test_mat = np.column_stack([test_preds[n]  for n in names])

        # ElasticNet stacking (primary method)
        en_clf = ElasticNet(alpha=EN_ALPHA, l1_ratio=EN_L1_RATIO, max_iter=EN_MAX_ITER)
        en_clf.fit(meta_mat, y_meta)
        en_pred = en_clf.predict(test_mat)

        # Ridge stacking (comparison)
        ri_clf = Ridge(alpha=RIDGE_ALPHA)
        ri_clf.fit(meta_mat, y_meta)
        ri_pred = ri_clf.predict(test_mat)

        # Linear Regression stacking (baseline — unstable on small meta blocks)
        lr_clf = LinearRegression()
        lr_clf.fit(meta_mat, y_meta)
        lr_pred = lr_clf.predict(test_mat)

        # Weighted ensemble (inverse-RMSE weights from meta block)
        meta_rmse = np.array([
            float(np.sqrt(np.mean((y_meta - meta_preds[n]) ** 2))) for n in names
        ])
        inv_w = 1.0 / np.maximum(meta_rmse, 1e-8)
        w     = inv_w / inv_w.sum()
        wp    = sum(w[i] * test_preds[n] for i, n in enumerate(names))

        # ── Collect all metrics ───────────────────────────────────────────────
        row = {"Battery": battery_id, "Seed": seed}
        for name in names:
            m = all_metrics(y_test, test_preds[name])
            row.update({f"{name}_{k}": v for k, v in m.items()})
        row.update({f"Weighted_{k}":  v for k, v in all_metrics(y_test, wp).items()})
        row.update({f"Stack_LR_{k}":  v for k, v in all_metrics(y_test, lr_pred).items()})
        row.update({f"Stack_Ridge_{k}": v for k, v in all_metrics(y_test, ri_pred).items()})
        row.update({f"Stack_EN_{k}":  v for k, v in all_metrics(y_test, en_pred).items()})
        all_rows.append(row)

        elapsed = time.time() - t0
        print(f"    seed={seed}: GRU={row['GRU_RMSE']:.4f}  "
              f"LSTM={row['LSTM_RMSE']:.4f}  "
              f"XGB={row['XGB_RMSE']:.4f}  "
              f"EN={row['Stack_EN_RMSE']:.4f}  ({elapsed:.0f}s)")

        # ── Save seed-42 prediction CSV ───────────────────────────────────────
        if seed == SEEDS[0]:
            pred_df = pd.DataFrame({
                "Cycle":         test_cycles,
                "Actual_SOH":    y_test,
                "GRU":           test_preds["GRU"],
                "LSTM":          test_preds["LSTM"],
                "XGB":           test_preds["XGB"],
                "Weighted":      wp,
                "Stack_LR":      lr_pred,
                "Stack_Ridge":   ri_pred,
                "Stack_EN":      en_pred,
            })
            pred_path = RESULTS_DIR / f"{battery_id}_predictions_s{seed}.csv"
            pred_df.to_csv(str(pred_path), index=False)
            first_seed_preds = pred_df

            # Experiment metadata JSON
            meta_json = {
                "battery":        battery_id,
                "seed":           seed,
                "total_cycles":   tot,
                "seq_len":        SEQ_LEN,
                "n_features":     len(FEATURE_COLS),
                "xgb_n_lags":     XGB_N_LAGS,
                "partition": {
                    "base":  int(bm_s.sum()),
                    "early": int(em_s.sum()),
                    "meta":  len(meta_cycles),
                    "test":  len(test_cycles),
                },
                "meta_learner":       "ElasticNet",
                "en_alpha":           EN_ALPHA,
                "en_l1_ratio":        EN_L1_RATIO,
                "en_coefficients":    en_clf.coef_.tolist(),
                "en_intercept":       float(en_clf.intercept_),
                "weighted_weights":   dict(zip(names, w.tolist())),
                "base_meta_rmse":     dict(zip(names, meta_rmse.tolist())),
            }
            json_path = RESULTS_DIR / f"{battery_id}_s{seed}_experiment.json"
            with open(str(json_path), "w", encoding="utf-8") as jf:
                json.dump(meta_json, jf, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 70)
    print("  Battery SOH Experiment: GRU + LSTM + XGBoost → ElasticNet Stack")
    print(f"  Seeds: {SEEDS}   Batteries: {list(BATTERIES)}")
    print("=" * 70)

    # Validate that MAT files exist before spending time on anything else
    missing = []
    for bid, mf in BATTERIES.items():
        p = DATA_DIR / mf
        if not p.exists():
            missing.append(str(p))
    if missing:
        raise FileNotFoundError(
            "Missing NASA MAT files:\n" + "\n".join(missing) +
            "\n\nPlace B0005.mat, B0006.mat, B0007.mat, B0018.mat "
            "in the project data/ directory."
        )

    all_rows = []
    t_start  = time.time()

    for bid, mf in BATTERIES.items():
        run_battery(bid, DATA_DIR / mf, all_rows)

    # ── Aggregate results ─────────────────────────────────────────────────────
    full = pd.DataFrame(all_rows)
    full.to_csv(str(RESULTS_DIR / "final_results_raw.csv"), index=False)

    # Overall model summary (mean ± std across all cells and seeds)
    models = ["GRU", "LSTM", "XGB", "Weighted", "Stack_LR", "Stack_Ridge", "Stack_EN"]
    ovrl_rows = []
    for mdl in models:
        col = f"{mdl}_RMSE"
        if col not in full.columns:
            continue
        ovrl_rows.append({
            "Model":      mdl,
            "RMSE_mean":  round(float(full[col].mean()),               4),
            "RMSE_std":   round(float(full[col].std()),                4),
            "MAE_mean":   round(float(full[f"{mdl}_MAE"].mean()),      4),
            "MAE_std":    round(float(full[f"{mdl}_MAE"].std()),       4),
            "MAPE_mean":  round(float(full[f"{mdl}_MAPE"].mean()),     4),
            "MAPE_std":   round(float(full[f"{mdl}_MAPE"].std()),      4),
            "R2_mean":    round(float(full[f"{mdl}_R2"].mean()),       4),
            "R2_std":     round(float(full[f"{mdl}_R2"].std()),        4),
        })
    ovrl = pd.DataFrame(ovrl_rows)
    ovrl.to_csv(str(RESULTS_DIR / "final_overall.csv"), index=False)

    # Per-battery multi-seed summary
    smry_rows = []
    for bid in BATTERIES:
        for mdl in models:
            col = f"{mdl}_RMSE"
            if col not in full.columns:
                continue
            sub = full[full.Battery == bid]
            smry_rows.append({
                "Battery":    bid,
                "Model":      mdl,
                "RMSE_mean":  round(float(sub[col].mean()),            4),
                "RMSE_std":   round(float(sub[col].std()),             4),
                "MAE_mean":   round(float(sub[f"{mdl}_MAE"].mean()),   4),
                "MAE_std":    round(float(sub[f"{mdl}_MAE"].std()),    4),
                "MAPE_mean":  round(float(sub[f"{mdl}_MAPE"].mean()),  4),
                "MAPE_std":   round(float(sub[f"{mdl}_MAPE"].std()),   4),
                "R2_mean":    round(float(sub[f"{mdl}_R2"].mean()),    4),
                "R2_std":     round(float(sub[f"{mdl}_R2"].std()),     4),
            })
    smry = pd.DataFrame(smry_rows)
    smry.to_csv(str(RESULTS_DIR / "final_summary_per_battery.csv"), index=False)

    # ── Print summary ─────────────────────────────────────────────────────────
    elapsed = (time.time() - t_start) / 60
    print(f"\n{'=' * 70}")
    print(f"  OVERALL RESULTS  (mean ± std across {len(BATTERIES)} cells × {len(SEEDS)} seeds)")
    print(f"{'=' * 70}")
    print(ovrl[["Model", "RMSE_mean", "RMSE_std", "MAE_mean", "MAPE_mean", "R2_mean"]]
          .to_string(index=False))
    print(f"\n  Total runtime: {elapsed:.1f} min")
    print(f"  Results saved to: {RESULTS_DIR}")
    print(f"  Next step: python visualization/generate_figures.py")


if __name__ == "__main__":
    main()
