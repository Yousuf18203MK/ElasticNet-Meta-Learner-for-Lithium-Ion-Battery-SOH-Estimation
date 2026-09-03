"""
ensemble.py — Ensemble combination methods.

Three stacking meta-learners compared:
  LinearRegression — unconstrained; fails on small meta blocks (collinear inputs)
  Ridge            — L2 regularisation; stable coefficients
  ElasticNet       — L1+L2; sparse + stable; selected as final method

Weighted ensemble uses inverse-RMSE weights computed on the META block,
so the test set is never used for weight determination.
"""
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet


def weighted_ensemble(meta_preds: dict, test_preds: dict, y_meta: np.ndarray):
    """Inverse-RMSE weighted average of base-learner test predictions."""
    names = list(meta_preds)
    rmse_arr = np.array([
        float(np.sqrt(np.mean((y_meta - meta_preds[n]) ** 2))) for n in names
    ])
    inv = 1.0 / np.maximum(rmse_arr, 1e-8)
    w   = inv / inv.sum()
    pred = sum(w[i] * test_preds[n] for i, n in enumerate(names))
    return pred, dict(zip(names, w.tolist()))


def stack_lr(meta_preds: dict, test_preds: dict, y_meta: np.ndarray):
    meta_m = np.column_stack([meta_preds[n] for n in meta_preds])
    test_m = np.column_stack([test_preds[n]  for n in meta_preds])
    clf = LinearRegression()
    clf.fit(meta_m, y_meta)
    return clf.predict(test_m), clf


def stack_ridge(meta_preds: dict, test_preds: dict, y_meta: np.ndarray, alpha=10.0):
    meta_m = np.column_stack([meta_preds[n] for n in meta_preds])
    test_m = np.column_stack([test_preds[n]  for n in meta_preds])
    clf = Ridge(alpha=alpha)
    clf.fit(meta_m, y_meta)
    return clf.predict(test_m), clf


def stack_en(meta_preds: dict, test_preds: dict, y_meta: np.ndarray,
             alpha=0.1, l1_ratio=0.5, max_iter=5000):
    meta_m = np.column_stack([meta_preds[n] for n in meta_preds])
    test_m = np.column_stack([test_preds[n]  for n in meta_preds])
    clf = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=max_iter)
    clf.fit(meta_m, y_meta)
    return clf.predict(test_m), clf
