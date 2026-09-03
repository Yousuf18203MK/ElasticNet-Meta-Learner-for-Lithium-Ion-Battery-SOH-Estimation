Below is a publication oriented `README.md` based directly on the revised manuscript and the project structure you provided. I have used the manuscript values as the authoritative results, including the distinction between within battery and LOO performance. 

````markdown
# ElasticNet Meta Learner for Lithium Ion Battery SOH Estimation

## Heterogeneous Stacking Ensemble with ElasticNet Meta Learner for Lithium Ion Battery State of Health Estimation

A rigorous machine learning framework for Lithium Ion Battery State of Health estimation using a heterogeneous stacking ensemble composed of GRU, LSTM, and XGBoost base learners with an ElasticNet meta learner.

The framework is evaluated using the NASA Prognostics Center of Excellence battery degradation dataset under strict chronological data splitting, leakage free preprocessing, and three independent random seeds.

## Overview

Accurate State of Health estimation is essential for battery management, lifecycle assessment, predictive maintenance, electric vehicles, and stationary energy storage.

This project investigates whether combining complementary temporal and feature based models through a regularized stacking architecture can improve SOH estimation and whether the resulting learned fusion strategy transfers effectively to previously unseen battery cells.

The proposed architecture combines:

• GRU for sequential degradation modelling

• LSTM for alternative temporal representation learning

• XGBoost for nonlinear feature based modelling

• ElasticNet for regularized prediction fusion

The study evaluates both within battery estimation and Leave One Battery Out cross battery generalization.

## Proposed Architecture

```text
                    NASA Battery Measurements
                              │
                              ▼
             20 Electrochemical Cycle Features
                    Voltage, Current, Temperature
                              │
             ┌────────────────┼────────────────┐
             │                │                │
             ▼                ▼                ▼
            GRU              LSTM           XGBoost
             │                │                │
             └────────────────┼────────────────┘
                              │
                              ▼
                 Meta Training Predictions
                              │
                              ▼
                    ElasticNet Meta Learner
                              │
                              ▼
                       SOH Prediction
````

The recurrent models process 32 consecutive discharge cycles using the 20 dimensional feature representation.

XGBoost receives a causal lag augmented representation containing the current cycle and five preceding cycles, resulting in 120 input features.

The ElasticNet meta learner is trained exclusively on predictions generated from a disjoint meta training block.

## Key Contributions

• A leakage free SOH estimation pipeline using 20 electrochemical features derived from voltage, current, and temperature measurements.

• Strict chronological data splitting with preprocessing parameters fitted only on training partitions.

• Three seed evaluation using seeds 42, 123, and 2024.

• Heterogeneous ensemble learning through GRU, LSTM, and XGBoost.

• ElasticNet regularization for adaptive base learner selection and coefficient shrinkage.

• Explicit Leave One Battery Out evaluation for cross battery generalization.

• Direct comparison between learned stacking and inverse RMSE weighted fusion.

• Cell specific diagnostics for distribution shift and limited meta training samples.

## Dataset

The study uses four lithium ion battery cells from the NASA Prognostics Center of Excellence battery aging dataset:

| Battery | Cycles | Initial SOH | EOL SOH |
| ------- | ------ | ----------- | ------- |
| B0005   | 168    | 100.0%      | 69.4%   |
| B0006   | 168    | 100.0%      | 56.7%   |
| B0007   | 168    | 100.0%      | 74.1%   |
| B0018   | 132    | 100.0%      | 72.3%   |

The battery cells are 18650 format LiCoO₂ cells evaluated under controlled laboratory cycling conditions.

The original `.mat` battery files are not redistributed with this repository. Place the required data files inside the `data` directory according to the instructions provided in `data/README_DATA.txt`.

## SOH Definition

State of Health is defined as:

```text
SOHₖ = (Cₖ / C₀) × 100
```

where `Cₖ` is the measured discharge capacity at cycle `k` and `C₀` is the initial measured capacity.

Capacity is used exclusively to construct the SOH target and is excluded from the predictive feature set to prevent direct target leakage.

Cycle index is also excluded from predictive features.

## Feature Engineering

Each discharge cycle is represented by 20 statistical and electrochemical descriptors.

### Voltage Features

• Mean

• Standard deviation

• Minimum

• Maximum

• Range

• 10th percentile

• 90th percentile

• Slope

• Integral

### Current Features

• Mean

• Standard deviation

• Minimum

• Maximum

### Temperature Features

• Mean

• Standard deviation

• Temperature rise

• Maximum

• Slope

### Derived Features

• Mean power

• Valid sample duration

Capacity and cycle index are explicitly excluded from the predictive feature set.

## Data Preprocessing

A MinMaxScaler is fitted exclusively on the base training partition and then applied unchanged to later partitions.

For recurrent models, the SOH target is independently scaled using training data and inverse transformed before evaluation.

No test information is used for:

• Preprocessing parameter estimation

• Model training

• Early stopping decisions

• Meta learner fitting

• Ensemble coefficient selection

• Hyperparameter selection

## Experimental Protocol

### Within Battery Evaluation

The chronological split is:

```text
60% Base Training
10% Early Stopping Validation
15% Meta Training
15% Final Test
```

The test region is accessed only once for final evaluation.

### Leave One Battery Out Evaluation

For each experiment, one battery is completely held out.

The remaining three batteries are used for:

```text
70% Base Training
15% Early Stopping Validation
15% Meta Training
```

The entire held out battery trajectory is used as the external test set.

No sequence crosses a battery boundary.

## Base Models

### GRU

Two layer stacked Gated Recurrent Unit network.

```text
Layers: 2
Units per layer: 64
Dropout: 0.20
Optimizer: Adam
Learning rate: 0.001
Sequence length: 32
Batch size: 16
Maximum epochs: 100
Early stopping patience: 15
```

### LSTM

Two layer stacked Long Short Term Memory network.

```text
Layers: 2
Units per layer: 64
Dropout: 0.20
Optimizer: Adam
Learning rate: 0.001
Sequence length: 32
Batch size: 16
Maximum epochs: 100
Early stopping patience: 15
```

### XGBoost

```text
Estimators: 300
Maximum depth: 4
Learning rate: 0.05
Subsample: 0.80
Column sampling: 0.80
Lag depth: 5 cycles
Input dimension: 120
```

## ElasticNet Meta Learner

The proposed stacking model uses ElasticNet as the second level learner.

```text
Alpha: 0.05
L1 ratio: 0.50
Maximum iterations: 5000
```

ElasticNet combines L1 sparsity with L2 coefficient shrinkage.

This allows the meta learner to selectively suppress weak base learner contributions while stabilizing coefficients when meta training samples are limited and base learner predictions are highly correlated.

## Ensemble Methods Compared

The study evaluates seven configurations:

| Configuration     | Description                   |
| ----------------- | ----------------------------- |
| GRU               | Individual recurrent model    |
| LSTM              | Individual recurrent model    |
| XGBoost           | Individual tree based model   |
| Weighted Ensemble | Inverse RMSE weighted fusion  |
| Stack LR          | Linear stacking               |
| Stack Ridge       | Ridge regularized stacking    |
| Stack ElasticNet  | Proposed regularized stacking |

## Authoritative Within Battery Results

Mean results across four battery cells and three random seeds:

| Model                | RMSE              | MAE       | MAPE       | R²        |
| -------------------- | ----------------- | --------- | ---------- | --------- |
| GRU                  | 2.405             | 2.249     | 3.454%     | −2.747    |
| LSTM                 | 3.527             | 3.375     | 5.094%     | −8.758    |
| XGBoost              | 8.820             | 8.670     | 12.658%    | −74.140   |
| Weighted Ensemble    | 3.223             | 3.089     | 4.707%     | −6.846    |
| Stack LR             | 1.947             | 1.739     | 2.640%     | −2.157    |
| Stack Ridge          | 1.662             | 1.502     | 2.304%     | −0.767    |
| **Stack ElasticNet** | **1.058 ± 0.466** | **0.894** | **1.304%** | **0.058** |

Stack ElasticNet achieves the lowest aggregate within battery RMSE and MAPE.

## Per Battery Within Battery Results

| Battery     | RMSE      | RMSE Std  | MAE       | MAPE       | R²        |
| ----------- | --------- | --------- | --------- | ---------- | --------- |
| B0005       | 0.867     | 0.117     | 0.639     | 0.895%     | 0.071     |
| B0006       | 1.245     | 0.653     | 1.061     | 1.797%     | 0.529     |
| B0007       | 0.664     | 0.254     | 0.538     | 0.710%     | 0.356     |
| B0018       | 1.458     | 0.329     | 1.340     | 1.815%     | −0.722    |
| **Overall** | **1.058** | **0.466** | **0.894** | **1.304%** | **0.058** |

B0018 has a narrow test SOH range, making R² particularly sensitive to target variance. RMSE and MAE provide more stable interpretation for this cell.

## Leave One Battery Out Results

Mean results across four held out batteries and three random seeds:

| Model                 | RMSE              | RMSE Std  | MAE       | MAPE       | R²        |
| --------------------- | ----------------- | --------- | --------- | ---------- | --------- |
| GRU                   | 3.838             | 1.111     | 3.394     | 4.419%     | 0.735     |
| LSTM                  | 3.655             | 1.195     | 2.986     | 3.673%     | 0.745     |
| XGBoost               | 4.679             | 3.031     | 3.852     | 5.384%     | 0.587     |
| **Weighted Ensemble** | **2.696 ± 0.512** | **0.512** | **2.206** | **2.812%** | **0.871** |
| Stack LR              | 3.490             | 1.646     | 3.091     | 3.884%     | 0.740     |
| Stack Ridge           | 2.936             | 1.430     | 2.609     | 3.292%     | 0.807     |
| Stack ElasticNet      | 3.075             | 1.553     | 2.762     | 3.496%     | 0.788     |

The Weighted Ensemble is the strongest configuration for cross battery deployment under the evaluated LOO protocol.

## Deployment Interpretation

The central finding of this study is that the best model depends on the deployment scenario.

### Target Battery Observed During Training

**Recommended model: Stack ElasticNet**

```text
RMSE = 1.058 ± 0.466 SOH percentage points
MAPE = 1.304%
```

The meta learner can exploit battery specific prediction patterns when historical observations from the target battery are available.

### Target Battery Not Observed During Training

**Recommended model: Weighted Ensemble**

```text
RMSE = 2.696 ± 0.512 SOH percentage points
MAPE = 2.812%
R² = 0.871
```

The simpler inverse RMSE fusion strategy transfers more reliably to unseen batteries.

## Key Scientific Finding

The study demonstrates a ranking inversion between within battery estimation and cross battery generalization.

Stack ElasticNet is the strongest model when the target battery contributes data to training.

Weighted Ensemble becomes the strongest model when the target battery is completely unseen.

This indicates that learned meta coefficients can capture battery specific relationships that are effective for within battery estimation but may transfer less reliably across battery identities.

## Cell Specific Findings

### B0006

B0006 presents a significant distribution shift.

Its test SOH range extends from approximately 56.7% to 63.4%, while the training region reaches only down to approximately 63.4%.

This creates an extrapolation challenge for models evaluated on the late life degradation region.

### B0018

B0018 contains only 132 discharge cycles compared with 168 cycles for B0005, B0006, and B0007.

Its chronological within battery protocol produces only 20 meta training sequences.

The limited meta training sample size and strong correlation between base learner predictions can increase coefficient instability.

## Reproducibility

The complete workflow is designed for reproducible experimentation.

Three independent random seeds are used:

```text
42
123
2024
```

All evaluation metrics are computed only on final held out test data.

Metrics include:

• RMSE

• MAE

• MAPE

• R²

Results are reported as mean and standard deviation across evaluation seeds.

## Installation

Clone the repository:

```bash
git clone https://github.com/Yousuf18203MK/ElasticNet-Meta-Learner-for-Lithium-Ion-Battery-SOH-Estimation.git
cd ElasticNet-Meta-Learner-for-Lithium-Ion-Battery-SOH-Estimation
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Place the NASA battery data files inside:

```text
data/
```

Required files:

```text
B0005.mat
B0006.mat
B0007.mat
B0018.mat
```

The original dataset files are not redistributed in this repository.

## Execution

### Step 1: Within Battery Experiment

```bash
python experiments/run_all_experiments.py
```

### Step 2: Leave One Battery Out Experiment

```bash
python experiments/run_loo_experiment.py
```

### Step 3: Generate Within Battery Figures

```bash
python visualization/generate_figures.py
```

### Step 4: Generate LOO Figures

```bash
python visualization/generate_loo_figures.py
```

### Step 5: Generate Publication Tables

```bash
python visualization/generate_tables.py
```

## Repository Structure

```text
authoritative/
│
├── data/
│   └── README_DATA.txt
│
├── src/
│   ├── feature_engineering.py
│   ├── models.py
│   ├── ensemble.py
│   └── metrics.py
│
├── experiments/
│   ├── run_all_experiments.py
│   └── run_loo_experiment.py
│
├── visualization/
│   ├── generate_figures.py
│   ├── generate_loo_figures.py
│   └── generate_tables.py
│
├── results/
│   ├── final_results_raw.csv
│   ├── final_overall.csv
│   ├── final_summary_per_battery.csv
│   └── loo/
│       ├── loo_results_raw.csv
│       ├── loo_overall.csv
│       └── loo_summary_per_battery.csv
│
├── figures/
│
├── tables/
│
├── archive/
│
├── config.py
├── requirements.txt
├── README.md
└── REPRODUCIBILITY_REPORT.md
```

## Project Outputs

The repository provides reproducible generation of:

• Model predictions

• Evaluation metrics

• Within battery performance tables

• LOO cross battery performance tables

• Actual versus predicted SOH plots

• Residual analysis

• RMSE comparison visualizations

• Training convergence plots

• Cell specific diagnostic figures

• Publication ready tables

## Research Limitations

The current evaluation has several limitations.

• All four batteries share the same chemistry, nominal capacity, and controlled laboratory cycling conditions.

• Generalization to other battery chemistries and operating conditions is not demonstrated.

• The LOO evaluation contains only four held out battery cells.

• B0018 has a narrow SOH range, making its R² sensitive to target variance.

• XGBoost shows limitations when the evaluation SOH falls below its effective training range.

• Capacity regeneration artifacts are not reproduced reliably by the evaluated models, and their physical cause is not established by this study.

## Future Research Directions

Potential extensions include:

• Online adaptation of the meta learner using streaming observations from the target battery

• Evaluation across multi chemistry datasets such as CALCE, Oxford, and TRI

• Uncertainty quantified SOH estimation using conformal prediction or Bayesian approaches

• Feature importance analysis for identifying the most influential electrochemical descriptors

• More extensive cross battery validation with larger numbers of battery cells

## Citation

If you use this repository, methodology, or experimental results in academic work, please cite the corresponding research article.

```text
Heterogeneous Stacking Ensemble with ElasticNet Meta Learner for
Lithium Ion Battery State of Health Estimation:
A Rigorous Multi Seed, Within Battery and Cross Battery Evaluation
on NASA Degradation Data
```

## Research Keywords

```text
Lithium Ion Battery
Battery State of Health
SOH Estimation
Machine Learning
Deep Learning
GRU
LSTM
XGBoost
ElasticNet
Stacking Ensemble
Battery Degradation
NASA PCoE
Cross Battery Generalization
Leave One Battery Out
Predictive Maintenance
Battery Management Systems
```

## License

Add the license applicable to your research and repository before public distribution.

## Acknowledgment

This project was developed as part of research on machine learning based battery health estimation and focuses on reproducible evaluation, leakage prevention, ensemble learning, and cross battery generalization.

```

The manuscript specifically establishes the main scientific message as deployment dependent model selection: Stack ElasticNet is strongest for within battery estimation, whereas the Weighted Ensemble is stronger for previously unseen batteries under LOO evaluation. :contentReference[oaicite:1]{index=1}

The reported conclusion is also consistent with the final manuscript results, where Stack ElasticNet reaches `1.058 ± 0.466` RMSE within battery, while the Weighted Ensemble reaches `2.696 ± 0.512` RMSE under LOO. :contentReference[oaicite:2]{index=2}
```
