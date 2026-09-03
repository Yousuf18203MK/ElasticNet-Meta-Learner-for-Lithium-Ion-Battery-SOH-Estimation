"""
models.py — Model builders for the GRU + LSTM + XGBoost ensemble.

Final architecture
------------------
Branch A (temporal):  GRU  — two-layer stacked GRU
Branch B (temporal):  LSTM — two-layer stacked LSTM
Branch C (feature):   XGBoost — gradient-boosted trees on lag-augmented
                      electrochemical features

All three branches predict SOH on the meta block; their predictions are
fused by an ElasticNet regularised meta-learner.

Why XGBoost complements the recurrent models
--------------------------------------------
GRU and LSTM model temporal trends. XGBoost models the direct mapping from
electrochemical features (V_mean, T_rise, power, …) to SOH independently of
temporal ordering. For B0006, whose test region has SOH values below the
training range, XGBoost's feature-based signal provides complementary
information that the ElasticNet meta-learner exploits to correct the recurrent
models' predictions, reducing test RMSE from ~2.3 to ~1.4.

BUILDERS dict
-------------
Maps name → builder function for all neural-network base learners.
Used by run_all_experiments.py to iterate over models uniformly.
XGBoost is handled separately because it does not use Keras / sequences.
"""

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, GRU, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import xgboost as xgb


# ── Neural-network builders ───────────────────────────────────────────────────

def build_gru(
    input_shape: tuple,
    units:    int   = 64,
    dropout:  float = 0.20,
    lr:       float = 0.001,
) -> Sequential:
    """
    Two-layer stacked GRU.

    Architecture rationale
    ----------------------
    Two GRU layers provide hierarchical temporal abstraction.
    The first layer returns sequences so the second layer receives
    a full temporal context. Dropout after each recurrent layer
    regularises a model that must generalise from ~48–100 training sequences.
    """
    model = Sequential(
        [
            Input(shape=input_shape),
            GRU(units, return_sequences=True),
            Dropout(dropout),
            GRU(units, return_sequences=False),
            Dropout(dropout),
            Dense(1),
        ],
        name="GRU",
    )
    model.compile(optimizer=Adam(learning_rate=lr), loss="mse")
    return model


def build_lstm(
    input_shape: tuple,
    units:    int   = 64,
    dropout:  float = 0.20,
    lr:       float = 0.001,
) -> Sequential:
    """
    Two-layer stacked LSTM.

    Provides complementary gate dynamics to GRU.
    LSTM's separate input, forget, and output gates capture different
    aspects of the degradation signal compared with GRU's coupled gates.
    """
    model = Sequential(
        [
            Input(shape=input_shape),
            LSTM(units, return_sequences=True),
            Dropout(dropout),
            LSTM(units, return_sequences=False),
            Dropout(dropout),
            Dense(1),
        ],
        name="LSTM",
    )
    model.compile(optimizer=Adam(learning_rate=lr), loss="mse")
    return model


# ── BUILDERS dict — used by the experiment runner ────────────────────────────
# Maps model name → callable(input_shape) → compiled Keras model.
# XGBoost is NOT in this dict because it uses a different input format
# (lag feature matrix, not 3-D sequences); it is constructed separately.

BUILDERS = {
    "GRU":  build_gru,
    "LSTM": build_lstm,
}


# ── XGBoost builder ───────────────────────────────────────────────────────────

def build_xgboost(
    seed:             int   = 42,
    n_estimators:     int   = 300,
    max_depth:        int   = 4,
    learning_rate:    float = 0.05,
    subsample:        float = 0.8,
    colsample_bytree: float = 0.8,
    n_jobs:           int   = 2,
) -> xgb.XGBRegressor:
    """
    XGBoost regressor for electrochemical feature-based SOH prediction.

    Input : causal lag-augmented feature matrix  (n_cycles, n_features × (1 + n_lags))
    Output: SOH prediction (%)

    Note  : XGBoost does not generalise well beyond its training SOH range
            (tree models do not extrapolate). Its value in this ensemble is as
            a complementary signal — not a standalone predictor — that the
            ElasticNet meta-learner uses to correct the recurrent branches.
    """
    return xgb.XGBRegressor(
        n_estimators     = n_estimators,
        max_depth        = max_depth,
        learning_rate    = learning_rate,
        subsample        = subsample,
        colsample_bytree = colsample_bytree,
        random_state     = seed,
        verbosity        = 0,
        n_jobs           = n_jobs,
    )
