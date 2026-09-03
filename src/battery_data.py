"""
battery_data.py — NASA MAT parsing, SOH computation, feature scaling,
                  sliding-window sequence construction.

Predictive features : Voltage, Current, Temperature (cycle-mean)
Excluded features   : Capacity (defines SOH), Cycle (temporal proxy), SOH itself
SOH definition      : SOH_i = Capacity_i / Capacity_1 × 100 (%)
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.preprocessing import MinMaxScaler


# ── helpers ───────────────────────────────────────────────────────────────────
def _vec(v):
    return np.asarray(v, dtype=np.float64).squeeze().reshape(-1)


def _field(struct, name):
    if name not in (struct.dtype.names or ()):
        raise KeyError(f"Field '{name}' not in MAT struct")
    return _vec(struct[name])


# ── public ────────────────────────────────────────────────────────────────────
def load_discharge_cycles(mat_path: Path, battery_id: str) -> pd.DataFrame:
    """
    Load one NASA battery MAT file and return a cycle-level DataFrame.

    Columns returned
    ----------------
    Cycle       : 1-based discharge index
    Voltage     : mean discharge voltage (V)
    Current     : mean discharge current (A)
    Temperature : mean discharge temperature (°C)
    Capacity    : measured discharge capacity (Ah) — bookkeeping only
    SOH         : State of Health (%)               — target only
    """
    mat_path = Path(mat_path)
    if not mat_path.exists():
        raise FileNotFoundError(
            f"MAT file not found: {mat_path}\n"
            f"Place {battery_id}.mat in the project data/ directory."
        )

    mat     = loadmat(str(mat_path), squeeze_me=False, struct_as_record=True)
    battery = mat[battery_id][0, 0]
    cycles  = battery["cycle"][0]

    rows, idx = [], 0
    for cyc in cycles:
        if str(cyc["type"][0]).strip().lower() != "discharge":
            continue
        d   = cyc["data"][0, 0]
        V   = _field(d, "Voltage_measured")
        I   = _field(d, "Current_measured")
        T   = _field(d, "Temperature_measured")
        Cap = _field(d, "Capacity")
        n   = min(len(V), len(I), len(T))
        if n < 2 or not np.isfinite(Cap[0]):
            continue
        V, I, T = V[:n], I[:n], T[:n]
        ok = np.isfinite(V) & np.isfinite(I) & np.isfinite(T)
        if ok.sum() < 2:
            continue
        idx += 1
        rows.append({
            "Cycle":       idx,
            "Voltage":     float(np.mean(V[ok])),
            "Current":     float(np.mean(I[ok])),
            "Temperature": float(np.mean(T[ok])),
            "Capacity":    float(Cap[0]),
        })

    if not rows:
        raise ValueError(f"No valid discharge cycles found for {battery_id}")

    df = pd.DataFrame(rows)
    df["SOH"] = df["Capacity"] / df["Capacity"].iloc[0] * 100.0
    return df.reset_index(drop=True)


def make_scalers(train_df: pd.DataFrame):
    """Fit feature and target scalers on training partition ONLY."""
    fsc = MinMaxScaler()
    fsc.fit(train_df[["Voltage", "Current", "Temperature"]].values.astype(np.float32))
    tsc = MinMaxScaler()
    tsc.fit(train_df[["SOH"]].values.astype(np.float32))
    return fsc, tsc


def build_sequences(df: pd.DataFrame, fsc: MinMaxScaler, seq_len: int):
    """
    Sliding-window sequences of length seq_len over scaled features.

    Returns
    -------
    X            : (n, seq_len, 3) float32
    y            : (n,)           float32  — SOH targets in original % scale
    cycle_index  : (n,)           int32    — target cycle number
    """
    F = fsc.transform(df[["Voltage", "Current", "Temperature"]].values.astype(np.float32))
    S = df["SOH"].values.astype(np.float32)
    C = df["Cycle"].values.astype(np.int32)
    X, y, c = [], [], []
    for e in range(seq_len - 1, len(df)):
        X.append(F[e - seq_len + 1 : e + 1])
        y.append(S[e])
        c.append(C[e])
    return np.array(X, np.float32), np.array(y, np.float32), np.array(c, np.int32)


def chronological_masks(cycle_idx: np.ndarray, total_cycles: int):
    """
    Four-way chronological split based on cycle number.

    base  : cycles 1 – 60%          (base training)
    early : cycles 60% – 70%        (early-stopping validation)
    meta  : cycles 70% – 85%        (stacking meta-learner training)
    test  : cycles 85% – 100%       (final untouched evaluation)
    """
    b = int(total_cycles * 0.60)
    e = int(total_cycles * 0.70)
    m = int(total_cycles * 0.85)
    return (
        cycle_idx <= b,
        (cycle_idx > b) & (cycle_idx <= e),
        (cycle_idx > e) & (cycle_idx <= m),
        cycle_idx > m,
    )
