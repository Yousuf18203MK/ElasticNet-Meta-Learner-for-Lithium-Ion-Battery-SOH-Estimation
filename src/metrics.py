"""
metrics.py — Evaluation metrics computed on the final test set only.
"""
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def rmse(t, p):
    return float(np.sqrt(mean_squared_error(t, p)))

def mae(t, p):
    return float(mean_absolute_error(t, p))

def mape(t, p):
    t, p = np.asarray(t, float), np.asarray(p, float)
    return float(np.mean(np.abs((t - p) / np.maximum(np.abs(t), 1e-8))) * 100.0)

def r2(t, p):
    return float(r2_score(t, p))

def all_metrics(t, p):
    return {"RMSE": rmse(t,p), "MAE": mae(t,p), "MAPE": mape(t,p), "R2": r2(t,p)}
