"""
feature_engineering.py — Rich per-cycle feature extraction from raw NASA MAT files.

Extracts 20 within-cycle statistical and electrochemical features from
the raw V, I, T time-series measurements within each discharge cycle.

LEAKAGE PREVENTION:
  - Capacity is extracted ONLY to compute the SOH target; it is NEVER
    placed in the predictive feature matrix.
  - Cycle index is NEVER used as a predictive feature.
  - All scalers must be fitted on training-partition data only (enforced
    in run_all_experiments.py, not here).
"""

import numpy as np
import pandas as pd
from scipy.io import loadmat
from pathlib import Path

# numpy >= 2.0 renamed trapz → trapezoid; support both
try:
    _trapz = np.trapezoid
except AttributeError:
    _trapz = np.trapz

FEATURE_COLS = [
    "V_mean", "V_std", "V_min", "V_max", "V_range",
    "V_p10", "V_p90", "V_slope", "V_integ",
    "I_mean", "I_std", "I_min", "I_max",
    "T_mean", "T_std", "T_rise", "T_max", "T_slope",
    "power", "dur",
]


def extract_rich_features(mat_path: Path, battery_id: str) -> pd.DataFrame:
    """
    Parse one NASA battery MAT file and return a cycle-level DataFrame
    with 20 engineered features plus Capacity (for SOH computation) and SOH.

    Columns
    -------
    Cycle   : 1-based discharge cycle index
    V_mean  : mean discharge voltage (V)
    V_std   : std of discharge voltage
    V_min   : minimum discharge voltage (V)
    V_max   : maximum discharge voltage (V)
    V_range : max - min voltage (V)
    V_p10   : 10th-percentile voltage (V)
    V_p90   : 90th-percentile voltage (V)
    V_slope : linear regression slope of V vs sample index
    V_integ : normalised trapezoidal integral of V curve
    I_mean  : mean discharge current (A)
    I_std   : std of discharge current
    I_min   : minimum discharge current (A)
    I_max   : maximum discharge current (A)
    T_mean  : mean discharge temperature (°C)
    T_std   : std of discharge temperature
    T_rise  : T_final - T_initial within the cycle (°C)
    T_max   : maximum discharge temperature (°C)
    T_slope : linear regression slope of T vs sample index
    power   : mean discharge power  V × |I| (W)
    dur     : number of valid measurement samples (discharge duration proxy)
    Capacity: measured discharge capacity (Ah)  — bookkeeping only, NOT a feature
    SOH     : State of Health (%)               — prediction target only
    """
    mat_path = Path(mat_path)
    if not mat_path.exists():
        raise FileNotFoundError(
            f"MAT file not found: {mat_path}\n"
            f"Place {battery_id}.mat in the project data/ directory."
        )

    mat    = loadmat(str(mat_path), squeeze_me=False, struct_as_record=True)
    cycles = mat[battery_id][0, 0]["cycle"][0]
    rows, idx = [], 0

    for cyc in cycles:
        if str(cyc["type"][0]).strip().lower() != "discharge":
            continue

        d   = cyc["data"][0, 0]
        def _g(field):
            return np.asarray(d[field], dtype=float).squeeze().reshape(-1)

        V   = _g("Voltage_measured")
        I   = _g("Current_measured")
        T   = _g("Temperature_measured")
        Cap = _g("Capacity")

        n   = min(len(V), len(I), len(T))
        c0  = float(Cap[0])
        if n < 4 or not np.isfinite(c0):
            continue

        V2, I2, T2 = V[:n], I[:n], T[:n]
        ok = np.isfinite(V2) & np.isfinite(I2) & np.isfinite(T2)
        if ok.sum() < 4:
            continue

        Vf, If, Tf = V2[ok], I2[ok], T2[ok]
        idx += 1
        t   = np.arange(len(Vf), dtype=float)

        rows.append({
            "Cycle":   idx,
            "V_mean":  float(np.mean(Vf)),
            "V_std":   float(np.std(Vf)),
            "V_min":   float(Vf.min()),
            "V_max":   float(Vf.max()),
            "V_range": float(Vf.max() - Vf.min()),
            "V_p10":   float(np.percentile(Vf, 10)),
            "V_p90":   float(np.percentile(Vf, 90)),
            "V_slope": float(np.polyfit(t, Vf, 1)[0]),
            "V_integ": float(_trapz(Vf) / len(Vf)),
            "I_mean":  float(np.mean(If)),
            "I_std":   float(np.std(If)),
            "I_min":   float(If.min()),
            "I_max":   float(If.max()),
            "T_mean":  float(np.mean(Tf)),
            "T_std":   float(np.std(Tf)),
            "T_rise":  float(Tf[-1] - Tf[0]),
            "T_max":   float(Tf.max()),
            "T_slope": float(np.polyfit(t, Tf, 1)[0]),
            "power":   float(np.mean(Vf * np.abs(If))),
            "dur":     float(len(Vf)),
            "Capacity": c0,
        })

    if not rows:
        raise ValueError(f"No valid discharge cycles found for {battery_id}")

    df = pd.DataFrame(rows)
    df["SOH"] = df["Capacity"] / df["Capacity"].iloc[0] * 100.0
    return df.reset_index(drop=True)


def build_lag_features(
    df: pd.DataFrame,
    feat_cols: list,
    n_lags: int = 5,
) -> np.ndarray:
    """
    Build a causal lag feature matrix for XGBoost.

    Row i contains:  current-cycle features  +  lag-1 features  +  … +  lag-n_lags features.
    Lags that extend before cycle 1 are zero-padded.
    No future information is included.

    Returns ndarray of shape (n_cycles, len(feat_cols) * (1 + n_lags)).
    """
    X = df[feat_cols].values.copy()
    rows = []
    for i in range(len(X)):
        row = list(X[i])
        for lag in range(1, n_lags + 1):
            row.extend(list(X[i - lag]) if i - lag >= 0 else [0.0] * len(feat_cols))
        rows.append(row)
    return np.array(rows, dtype=np.float32)


def build_sequences(
    df: pd.DataFrame,
    feature_scaler,
    feat_cols: list,
    seq_len: int = 32,
):
    """
    Build overlapping sliding-window sequences for GRU / LSTM input.

    The feature_scaler must already be fitted on the training partition.

    Returns
    -------
    X            : (n_samples, seq_len, n_features)  float32
    y            : (n_samples,)                       float32  — SOH in original %
    cycle_index  : (n_samples,)                       int32    — target cycle number
    """
    F = feature_scaler.transform(df[feat_cols].values.astype(np.float32))
    S = df["SOH"].values.astype(np.float32)
    C = df["Cycle"].values.astype(np.int32)

    X, y, c = [], [], []
    for end in range(seq_len - 1, len(df)):
        X.append(F[end - seq_len + 1 : end + 1])
        y.append(S[end])
        c.append(C[end])

    return (
        np.array(X, dtype=np.float32),
        np.array(y, dtype=np.float32),
        np.array(c, dtype=np.int32),
    )
