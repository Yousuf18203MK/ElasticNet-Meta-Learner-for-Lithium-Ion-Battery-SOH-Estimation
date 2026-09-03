"""
config.py — Single authoritative configuration for the entire project.

All paths are computed relative to this file so the project works on
any machine (Windows, Linux, macOS) without editing paths manually.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR     = PROJECT_ROOT / "data"
RESULTS_DIR  = PROJECT_ROOT / "results"
FIGURES_DIR  = PROJECT_ROOT / "figures"
TABLES_DIR   = PROJECT_ROOT / "tables"
MODELS_DIR   = PROJECT_ROOT / "models"
LOGS_DIR     = PROJECT_ROOT / "logs"

for _d in [RESULTS_DIR, FIGURES_DIR, TABLES_DIR, MODELS_DIR, LOGS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ── Battery cells ─────────────────────────────────────────────────────────────
BATTERIES = {
    "B0005": "B0005.mat",
    "B0006": "B0006.mat",
    "B0007": "B0007.mat",
    "B0018": "B0018.mat",
}

# ── Evaluation seeds ──────────────────────────────────────────────────────────
SEEDS = [42, 123, 2024]

# ── Feature engineering ───────────────────────────────────────────────────────
# 20 within-cycle statistical features extracted from raw V, I, T time series.
# Capacity and Cycle are NEVER included as predictive features.
FEATURE_COLS = [
    "V_mean", "V_std", "V_min", "V_max", "V_range",
    "V_p10", "V_p90", "V_slope", "V_integ",
    "I_mean", "I_std", "I_min", "I_max",
    "T_mean", "T_std", "T_rise", "T_max", "T_slope",
    "power", "dur",
]

XGB_N_LAGS = 5   # causal lag depth for XGBoost feature matrix

# ── Sequence construction ─────────────────────────────────────────────────────
SEQ_LEN = 32     # sliding window length (cycles) for GRU / LSTM input

# ── Chronological split fractions ────────────────────────────────────────────
FRAC_BASE      = 0.60   # base training block
FRAC_EARLY_VAL = 0.10   # early-stopping validation (NEVER used for metrics)
FRAC_META      = 0.15   # stacking meta-learner training
FRAC_TEST      = 0.15   # final untouched test set

# ── GRU / LSTM hyperparameters ────────────────────────────────────────────────
GRU_UNITS  = 64
LSTM_UNITS = 64
DROPOUT    = 0.20
LR         = 0.001
EPOCHS     = 100
BATCH_SIZE = 16
PATIENCE   = 15   # EarlyStopping patience on early-val loss

# ── XGBoost hyperparameters ───────────────────────────────────────────────────
XGB_N_ESTIMATORS   = 300
XGB_MAX_DEPTH      = 4
XGB_LEARNING_RATE  = 0.05
XGB_SUBSAMPLE      = 0.8
XGB_COLSAMPLE      = 0.8

# ── Ensemble meta-learner hyperparameters ─────────────────────────────────────
RIDGE_ALPHA  = 10.0
EN_ALPHA     = 0.05   # ElasticNet alpha (chosen for small meta-block robustness)
EN_L1_RATIO  = 0.50
EN_MAX_ITER  = 5000
