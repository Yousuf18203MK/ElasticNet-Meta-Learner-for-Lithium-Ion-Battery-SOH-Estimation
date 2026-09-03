# Battery SOH Prediction — GRU + LSTM + XGBoost → ElasticNet Stack

**Final Architecture**: Three complementary base learners fused by ElasticNet stacking.

## Authoritative Results

### Within-Battery Evaluation (15% chronological test — verified)

| Cell  | RMSE mean | RMSE std | MAPE mean |
|-------|-----------|----------|-----------|
| B0005 | 0.8676    | ±0.1181  | 0.8966%   |
| B0006 | 1.2538    | ±0.6493  | 1.8118%   |
| B0007 | 0.6566    | ±0.2417  | 0.6995%   |
| B0018 | 1.4479    | ±0.2927  | 1.8101%   |
| **Overall** | **1.0565** | **±0.4595** | **1.3045%** |

### LOO Cross-Battery Generalization (entire held-out cell)

| Held-Out Cell | RMSE mean | RMSE std | MAPE mean | R² mean |
|---------------|-----------|----------|-----------|---------|
| B0005         | 2.5024    | ±2.1126  | 2.9086%   | 0.8803  |
| B0006         | 3.3687    | ±0.3357  | 4.0879%   | 0.8637  |
| B0007         | 3.5748    | ±2.5117  | 4.0193%   | 0.6554  |
| B0018         | 2.6952    | ±0.9609  | 2.8377%   | 0.7760  |
| **Overall**   | **3.0353** | **±1.5381** | **3.4634%** | **0.7938** |

**Note:** LOO tests on the *full* battery trajectory (100%→EOL), while within-battery
tests on the *last 15%*. LOO degradation is a harder and more scientifically informative task.

---

## Setup

```bash
pip install -r requirements.txt
# Place B0005.mat B0006.mat B0007.mat B0018.mat in data/
```

## Execution Order

```bash
# Step 1 — Within-battery experiment (≈10–18 min on CPU)
python experiments/run_all_experiments.py

# Step 2 — LOO cross-battery experiment (≈4–6 min on CPU)
python experiments/run_loo_experiment.py

# Step 3 — Generate figures
python visualization/generate_figures.py       # within-battery
python visualization/generate_loo_figures.py   # LOO

# Step 4 — Generate tables
python visualization/generate_tables.py
```

## Project Structure

```
authoritative/
├── data/
│   ├── README_DATA.txt     ← Where to place .mat files
│   └── [B0005-B0018.mat]   ← Not redistributed
├── src/
│   ├── feature_engineering.py  ← 20-feature extractor + sequence builder
│   ├── models.py               ← GRU, LSTM (BUILDERS dict) + XGBoost
│   ├── ensemble.py             ← Weighted, LR, Ridge, ElasticNet stacking
│   └── metrics.py              ← RMSE, MAE, MAPE, R²
├── experiments/
│   ├── run_all_experiments.py  ← Within-battery experiment
│   └── run_loo_experiment.py   ← LOO cross-battery experiment
├── visualization/
│   ├── generate_figures.py     ← 14 within-battery figures
│   ├── generate_loo_figures.py ← 8 LOO figures
│   └── generate_tables.py      ← Publication tables
├── results/
│   ├── {cell}_rich.csv                   ← Shared by both experiments
│   ├── final_results_raw.csv             ← Within-battery raw results
│   ├── final_overall.csv                 ← Within-battery summary
│   ├── final_summary_per_battery.csv     ← Per-cell within-battery summary
│   └── loo/
│       ├── loo_results_raw.csv           ← LOO raw results
│       ├── loo_overall.csv               ← LOO overall summary
│       └── loo_summary_per_battery.csv   ← Per-cell LOO summary
├── figures/
│   ├── fig01–fig14                       ← Within-battery figures
│   └── loo/figL01–figL08                 ← LOO figures
├── tables/
│   ├── table01–table10                   ← Within-battery tables
│   └── table_loo01–table_loo05           ← LOO tables
├── archive/                              ← Historical experiments
├── config.py                            ← All hyperparameters (single source of truth)
├── requirements.txt
├── README.md
└── REPRODUCIBILITY_REPORT.md
```

## Key Scientific Finding — Generalization

The model achieves **positive R² = 0.79** on LOO cross-battery evaluation (Stack EN),
confirming that it captures the degradation trend even on completely unseen batteries.
The **Weighted Ensemble outperforms ElasticNet stacking under LOO** (RMSE 2.69 vs 3.04),
indicating that the meta-learner's learned coefficients do not transfer robustly across
battery identities. For deployment on unseen batteries, the Weighted Ensemble is recommended.
