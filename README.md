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
