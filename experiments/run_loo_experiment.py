"""
run_loo_experiment.py — Leave-One-Battery-Out (LOO) cross-battery generalization.

Evaluation protocol
-------------------
Four LOO configurations are executed sequentially.
In each configuration one battery is the completely held-out test battery
and the remaining three batteries are used exclusively for training.

Critical leakage prevention rules (all enforced by design)
----------------------------------------------------------
  1. Feature scaler fitted ONLY on base portion (70%) of training batteries.
  2. Target scaler fitted ONLY on base portion (70%) of training batteries.
  3. XGBoost lag-feature normalizer fitted ONLY on training battery base data.
  4. Sequences are built WITHIN each battery; no sequences cross battery boundaries.
  5. ElasticNet meta-learner receives predictions generated on meta portions of
     TRAINING batteries — never on the held-out test battery.
  6. EarlyStopping validation set is the early-val portion of TRAINING batteries.
  7. The held-out battery is used once, only for the final reported metrics.

Training split (within each training battery)
---------------------------------------------
  base     : 0 – 70%   (base model training)
  early-val: 70% – 85% (EarlyStopping validation)
  meta     : 85% – 100% (ElasticNet meta-learner training)

Test
----
  held-out battery : ALL sequences (entire discharge trajectory)

Run from the project root:   python experiments/run_loo_experiment.py
"""

import sys
import os
import json
import time
import random
import warnings
from pathlib import Path

_SCRIPT_DIR  = Path(__file__).resolve().parent
PROJECT_ROOT = _SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, GRU, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from config import (
    RESULTS_DIR, BATTERIES, SEEDS, FEATURE_COLS, SEQ_LEN,
    XGB_N_LAGS, XGB_N_ESTIMATORS, XGB_MAX_DEPTH,
    XGB_LEARNING_RATE, XGB_SUBSAMPLE, XGB_COLSAMPLE,
    EN_ALPHA, EN_L1_RATIO, EN_MAX_ITER, RIDGE_ALPHA,
)
from src.feature_engineering import build_lag_features, build_sequences
from src.metrics import all_metrics

LOO_DIR = RESULTS_DIR / "loo"
LOO_DIR.mkdir(parents=True, exist_ok=True)

CELLS = list(BATTERIES.keys())


# ─── helpers ─────────────────────────────────────────────────────────────────

def set_seed(s):
    random.seed(s); np.random.seed(s); tf.random.set_seed(s)


def inv(scaler, arr):
    return scaler.inverse_transform(arr.reshape(-1, 1)).reshape(-1)


def loo_split(cycle_idx: np.ndarray, total_cycles: int):
    """
    Three-way chronological split for training batteries under LOO.
    No within-battery test partition is needed because the entire
    held-out battery serves as the external test set.

    base     : cycles 1 – 70%
    early-val: 70% – 85%
    meta     : 85% – 100%
    """
    b = int(total_cycles * 0.70)
    e = int(total_cycles * 0.85)
    return (
        cycle_idx <= b,
        (cycle_idx > b) & (cycle_idx <= e),
        cycle_idx > e,
    )


def build_keras_model(shape, kind="GRU", units=64, dropout=0.20, lr=1e-3):
    if kind == "GRU":
        m = Sequential([Input(shape), GRU(units, return_sequences=True), Dropout(dropout),
                        GRU(units), Dropout(dropout), Dense(1)], name="GRU")
    else:
        m = Sequential([Input(shape), LSTM(units, return_sequences=True), Dropout(dropout),
                        LSTM(units), Dropout(dropout), Dense(1)], name="LSTM")
    m.compile(optimizer=Adam(lr), loss="mse")
    return m


def train_keras(model, Xb, ybs, Xv, yvs, patience=8, epochs=60, batch=32):
    cb = EarlyStopping("val_loss", patience=patience, restore_best_weights=True, verbose=0)
    model.fit(Xb, ybs, validation_data=(Xv, yvs), epochs=epochs,
              batch_size=batch, callbacks=[cb], verbose=0)
    return model


def load_rich(battery_id: str) -> pd.DataFrame:
    path = RESULTS_DIR / f"{battery_id}_rich.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found.\nRun experiments/run_all_experiments.py first "
            "to generate rich feature CSVs."
        )
    return pd.read_csv(str(path))


# ─── one LOO configuration ────────────────────────────────────────────────────

def run_one_loo(test_bid: str, all_rows: list, pred_store: dict) -> None:
    """
    Execute one LOO configuration (test_bid = held-out battery).
    Results appended to all_rows; seed-42 predictions stored in pred_store.
    """
    train_bids = [b for b in CELLS if b != test_bid]
    print(f"\n  TEST (held out): {test_bid}   TRAIN: {train_bids}")

    # ── 1. Load data ──────────────────────────────────────────────────────────
    train_dfs = {b: load_rich(b) for b in train_bids}
    test_df   = load_rich(test_bid)

    # ── 2. Fit scalers on base (70%) of training cells ONLY ──────────────────
    base_frames = [d.iloc[:int(len(d) * 0.70)] for d in train_dfs.values()]
    base_concat = pd.concat(base_frames, ignore_index=True)

    fsc = MinMaxScaler(); fsc.fit(base_concat[FEATURE_COLS].values.astype(np.float32))
    tsc = MinMaxScaler(); tsc.fit(base_concat[["SOH"]].values.astype(np.float32))

    # ── 3. Build sequences (per-cell, then concatenate) ───────────────────────
    #     "Sequences must not cross battery boundaries" → build within each cell
    Xb_l, yb_l, Xv_l, yv_l, Xm_l, ym_l = [], [], [], [], [], []
    for bid, df in train_dfs.items():
        tot = len(df)
        X, y, c = build_sequences(df, fsc, FEATURE_COLS, SEQ_LEN)
        bm, em, mm = loo_split(c, tot)
        Xb_l.append(X[bm]); yb_l.append(y[bm])
        Xv_l.append(X[em]); yv_l.append(y[em])
        Xm_l.append(X[mm]); ym_l.append(y[mm])

    Xb = np.concatenate(Xb_l); yb = np.concatenate(yb_l)
    Xv = np.concatenate(Xv_l); yv = np.concatenate(yv_l)
    Xm = np.concatenate(Xm_l); ym = np.concatenate(ym_l)
    ybs = tsc.transform(yb.reshape(-1, 1)).astype(np.float32)
    yvs = tsc.transform(yv.reshape(-1, 1)).astype(np.float32)

    # Test: ALL sequences from the held-out battery
    Xt, yt, ct = build_sequences(test_df, fsc, FEATURE_COLS, SEQ_LEN)
    shape = (Xb.shape[1], Xb.shape[2])

    # ── 4. XGBoost lag features (per-cell, then concatenate) ─────────────────
    lag_base_parts, lag_base_y_parts = [], []
    lag_meta_parts, lag_meta_y_parts = [], []

    for bid, df in train_dfs.items():
        tot = len(df)
        lr  = build_lag_features(df, FEATURE_COLS, XGB_N_LAGS)
        C   = df["Cycle"].values.astype(np.int32)
        bm2, _, mm2 = loo_split(C, tot)
        lag_base_parts.append(lr[bm2]);    lag_base_y_parts.append(df["SOH"].values[bm2])
        lag_meta_parts.append(lr[mm2]);    lag_meta_y_parts.append(df["SOH"].values[mm2])

    lag_base_raw = np.concatenate(lag_base_parts)
    lag_base_y   = np.concatenate(lag_base_y_parts)
    lag_meta_raw = np.concatenate(lag_meta_parts)
    lag_meta_y   = np.concatenate(lag_meta_y_parts)

    lag_sc = MinMaxScaler(); lag_sc.fit(lag_base_raw)
    lag_base_n = lag_sc.transform(lag_base_raw)
    lag_meta_n = lag_sc.transform(lag_meta_raw)

    # Held-out battery lag features (fitted scaler from training only)
    lag_test_n = lag_sc.transform(build_lag_features(test_df, FEATURE_COLS, XGB_N_LAGS))

    print(f"    base={len(Xb)}  early_val={len(Xv)}  meta={len(Xm)}  test={len(Xt)}")

    # ── 5. Per-seed training and evaluation ───────────────────────────────────
    for seed in SEEDS:
        set_seed(seed)
        t0 = time.time()

        meta_preds, test_preds = {}, {}

        # XGBoost
        xgb_model = xgb.XGBRegressor(
            n_estimators=XGB_N_ESTIMATORS, max_depth=XGB_MAX_DEPTH,
            learning_rate=XGB_LEARNING_RATE, subsample=XGB_SUBSAMPLE,
            colsample_bytree=XGB_COLSAMPLE, random_state=seed,
            verbosity=0, n_jobs=2,
        )
        xgb_model.fit(lag_base_n, lag_base_y)
        meta_preds["XGB"] = xgb_model.predict(lag_meta_n)
        xgb_test_all      = xgb_model.predict(lag_test_n)
        test_preds["XGB"] = xgb_test_all[-len(yt):]  # align with sequence test targets

        # GRU
        mg = build_keras_model(shape, "GRU")
        train_keras(mg, Xb, ybs, Xv, yvs)
        meta_preds["GRU"] = inv(tsc, mg.predict(Xm, verbose=0))
        test_preds["GRU"] = inv(tsc, mg.predict(Xt, verbose=0))

        # LSTM
        ml = build_keras_model(shape, "LSTM")
        train_keras(ml, Xb, ybs, Xv, yvs)
        meta_preds["LSTM"] = inv(tsc, ml.predict(Xm, verbose=0))
        test_preds["LSTM"] = inv(tsc, ml.predict(Xt, verbose=0))

        # Align sizes
        n_meta = min(len(ym), len(meta_preds["XGB"]), len(meta_preds["GRU"]))
        n_test = len(yt)
        mm_mat = np.column_stack([meta_preds["XGB"][-n_meta:],
                                  meta_preds["GRU"][-n_meta:],
                                  meta_preds["LSTM"][-n_meta:]])
        tt_mat = np.column_stack([test_preds["XGB"][-n_test:],
                                  test_preds["GRU"][-n_test:],
                                  test_preds["LSTM"][-n_test:]])
        ym_al  = ym[-n_meta:]

        # Ensemble variants
        en_clf  = ElasticNet(alpha=EN_ALPHA, l1_ratio=EN_L1_RATIO, max_iter=EN_MAX_ITER)
        ri_clf  = Ridge(alpha=RIDGE_ALPHA)
        lr_clf  = LinearRegression()
        en_clf.fit(mm_mat, ym_al); ri_clf.fit(mm_mat, ym_al); lr_clf.fit(mm_mat, ym_al)

        en_pred = en_clf.predict(tt_mat)
        ri_pred = ri_clf.predict(tt_mat)
        lr_pred = lr_clf.predict(tt_mat)

        mr  = np.array([float(np.sqrt(np.mean((ym_al - mm_mat[:, i]) ** 2))) for i in range(3)])
        inv_w = 1.0 / np.maximum(mr, 1e-8); w = inv_w / inv_w.sum()
        wp  = tt_mat @ w

        # Collect
        row = {"Test_Battery": test_bid, "Train_Batteries": "+".join(train_bids), "Seed": seed}
        for nm, pr in [("GRU",  test_preds["GRU"]),  ("LSTM", test_preds["LSTM"]),
                       ("XGB",  test_preds["XGB"][-n_test:]), ("Weighted", wp),
                       ("Stack_LR", lr_pred), ("Stack_Ridge", ri_pred), ("Stack_EN", en_pred)]:
            m = all_metrics(yt, pr)
            row.update({f"{nm}_{k}": v for k, v in m.items()})
        all_rows.append(row)

        # Seed-42 predictions → CSV
        if seed == SEEDS[0]:
            pdf = pd.DataFrame({
                "Cycle": ct, "Actual_SOH": yt,
                "GRU": test_preds["GRU"], "LSTM": test_preds["LSTM"],
                "XGB": test_preds["XGB"][-n_test:], "Weighted": wp,
                "Stack_LR": lr_pred, "Stack_Ridge": ri_pred, "Stack_EN": en_pred,
            })
            pdf.to_csv(str(LOO_DIR / f"{test_bid}_loo_predictions_s42.csv"), index=False)
            pred_store[test_bid] = pdf

            with open(str(LOO_DIR / f"{test_bid}_loo_s42_info.json"), "w") as jf:
                json.dump({
                    "test_battery": test_bid, "train_batteries": train_bids,
                    "seed": seed, "n_base": len(Xb), "n_early": len(Xv),
                    "n_meta": n_meta, "n_test": n_test,
                    "en_coefficients": en_clf.coef_.tolist(),
                    "en_intercept": float(en_clf.intercept_),
                    "weighted_weights": w.tolist(),
                    "meta_rmse_per_model": mr.tolist(),
                }, jf, indent=2)

        elapsed = time.time() - t0
        print(f"    seed={seed}: GRU={row['GRU_RMSE']:.4f}  LSTM={row['LSTM_RMSE']:.4f}"
              f"  XGB={row['XGB_RMSE']:.4f}  EN={row['Stack_EN_RMSE']:.4f}  ({elapsed:.0f}s)")


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  Leave-One-Battery-Out (LOO) Cross-Battery Generalization Experiment")
    print(f"  Seeds: {SEEDS}   Configurations: {len(CELLS)}")
    print("=" * 70)

    # Validate rich CSVs exist
    missing = [bid for bid in CELLS if not (RESULTS_DIR / f"{bid}_rich.csv").exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing rich feature CSV(s) for: {missing}\n"
            "Run experiments/run_all_experiments.py first."
        )

    all_rows  = []
    pred_store = {}
    t_start   = time.time()

    for test_bid in CELLS:
        run_one_loo(test_bid, all_rows, pred_store)

    # ── Aggregate ─────────────────────────────────────────────────────────────
    loo_full = pd.DataFrame(all_rows)
    loo_full.to_csv(str(LOO_DIR / "loo_results_raw.csv"), index=False)

    MODS = ["GRU","LSTM","XGB","Weighted","Stack_LR","Stack_Ridge","Stack_EN"]

    smry_rows = []
    for test_bid in CELLS:
        sub = loo_full[loo_full.Test_Battery == test_bid]
        for mdl in MODS:
            col = f"{mdl}_RMSE"
            if col not in sub.columns:
                continue
            smry_rows.append({
                "Test_Battery":  test_bid, "Model": mdl,
                "RMSE_mean":  round(sub[col].mean(), 4),
                "RMSE_std":   round(sub[col].std(),  4),
                "MAE_mean":   round(sub[f"{mdl}_MAE"].mean(), 4),
                "MAE_std":    round(sub[f"{mdl}_MAE"].std(),  4),
                "MAPE_mean":  round(sub[f"{mdl}_MAPE"].mean(), 4),
                "MAPE_std":   round(sub[f"{mdl}_MAPE"].std(),  4),
                "R2_mean":    round(sub[f"{mdl}_R2"].mean(), 4),
                "R2_std":     round(sub[f"{mdl}_R2"].std(),  4),
            })
    pd.DataFrame(smry_rows).to_csv(str(LOO_DIR / "loo_summary_per_battery.csv"), index=False)

    ovrl_rows = []
    for mdl in MODS:
        col = f"{mdl}_RMSE"
        if col not in loo_full.columns:
            continue
        ovrl_rows.append({
            "Model":      mdl,
            "RMSE_mean":  round(loo_full[col].mean(), 4),
            "RMSE_std":   round(loo_full[col].std(),  4),
            "MAE_mean":   round(loo_full[f"{mdl}_MAE"].mean(), 4),
            "MAE_std":    round(loo_full[f"{mdl}_MAE"].std(),  4),
            "MAPE_mean":  round(loo_full[f"{mdl}_MAPE"].mean(), 4),
            "MAPE_std":   round(loo_full[f"{mdl}_MAPE"].std(),  4),
            "R2_mean":    round(loo_full[f"{mdl}_R2"].mean(), 4),
            "R2_std":     round(loo_full[f"{mdl}_R2"].std(),  4),
        })
    ovrl = pd.DataFrame(ovrl_rows)
    ovrl.to_csv(str(LOO_DIR / "loo_overall.csv"), index=False)

    import pickle
    with open(str(LOO_DIR / "loo_preds_s42.pkl"), "wb") as f:
        pickle.dump(pred_store, f)

    elapsed = (time.time() - t_start) / 60
    print(f"\n{'=' * 70}")
    print(f"  LOO RESULTS  (mean ± std across 4 held-out cells × {len(SEEDS)} seeds)")
    print(f"{'=' * 70}")
    print(ovrl[["Model","RMSE_mean","RMSE_std","MAE_mean","MAPE_mean","R2_mean"]].to_string(index=False))
    print(f"\n  Total runtime: {elapsed:.1f} min")
    print(f"  Results saved to: {LOO_DIR}")
    print(f"  Next step: python visualization/generate_loo_figures.py")


if __name__ == "__main__":
    main()
