# AI-Based On-Chain Cryptocurrency Fraud & Market Manipulation Detection System

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A **production-grade, patent-ready AI system** for detecting fraudulent cryptocurrency transactions and market manipulation on-chain. Combines Graph Neural Networks, LSTM temporal analysis, ensemble machine learning, and Explainable AI across **10 novel components**.

---

## Table of Contents

1. [Overview](#overview)
2. [Novelties](#novelties)
3. [Architecture](#architecture)
4. [Project Structure](#project-structure)
5. [Installation](#installation)
6. [Quick Start](#quick-start)
7. [Dataset](#dataset)
8. [Models](#models)
9. [Results & Visualizations](#results--visualizations)
10. [Configuration](#configuration)
11. [License](#license)

---

## Overview

This system addresses the growing challenge of detecting fraud and market manipulation in decentralized cryptocurrency ecosystems. It synthesizes 50,000 realistic on-chain transactions across 2,000 wallets and applies a multi-model hybrid approach:

- **Graph Neural Networks (GNN)** capture wallet relationship patterns
- **LSTM Networks** detect temporal transaction anomalies
- **Random Forest / XGBoost** provide robust ensemble classification
- **Isolation Forest** performs unsupervised anomaly detection
- **SHAP Explainable AI** delivers human-readable fraud explanations
- **HMM-based Adaptive Thresholds** adjust to market conditions

---

## Novelties

| # | Novelty | Description |
|---|---------|-------------|
| 1 | **Hybrid Multi-Model Dynamic Risk Scoring** | Dynamically weights model outputs based on blockchain state |
| 2 | **LSTM Temporal Pattern Detection** | Captures sequential transaction patterns per wallet |
| 3 | **GNN Wallet Clustering** | Message-passing GCN for wallet-level fraud classification |
| 4 | **Liquidity Pool Anomaly Detection** | Detects rug-pull patterns from liquidity drain velocity |
| 5 | **Behavioral Fingerprinting** | GMM-based wallet behavior signatures |
| 6 | **Adaptive Threshold (HMM)** | Regime-aware thresholding (Bull/Bear/Volatile) |
| 7 | **Hybrid Scoring Engine** | Production-ready ensemble with dynamic weighting |
| 8 | **Explainable AI (SHAP)** | Human-readable fraud explanations |
| 9 | *RL-Based Policy Learning* | *(Future work)* |
| 10 | *Federated Learning* | *(Future work)* |

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                 Synthetic Data Generator             │
│        (50K txs, 2K wallets, 400 fraud wallets)     │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│           Data Preprocessor & Feature Engineering    │
│  • Transaction features  • Graph features            │
│  • Temporal features     • Risk indicators           │
│  • Rug-pull risk (N#4)   • Fingerprinting (N#5)     │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│              Transaction Graph Builder               │
│  • NetworkX digraph  • Centrality metrics            │
│  • Cycle detection   • Hub identification            │
└──────────────────────┬──────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        │              │              │
   ┌────▼────┐   ┌────▼────┐   ┌────▼────┐
   │   RF    │   │  LSTM   │   │   GNN   │
   │ XGBoost │   │ Temporal│   │  Wallet │
   │  IsoFor │   │ Pattern │   │ Cluster │
   └────┬────┘   └────┬────┘   └────┬────┘
        │              │              │
┌───────▼──────────────▼──────────────▼───────────────┐
│          Hybrid Risk Scoring Engine (N#1/7)          │
│     Dynamic weighting + Adaptive threshold (N#6)     │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│            Explainable AI — SHAP (N#8)              │
│     Human-readable fraud explanations                │
└─────────────────────────────────────────────────────┘
```

---

## Project Structure

```
crypto-fraud-detection/
├── data/
│   ├── raw/
│   │   └── synthetic_dataset.csv       # 50,000 transactions
│   └── processed/
│       ├── train_data.csv              # 70% train split
│       ├── test_data.csv               # 30% test split
│       └── graph_data.pkl              # Pickled NetworkX graph
├── models/
│   ├── gnn_model.pth                   # GNN (PyTorch)
│   ├── lstm_model.pth                  # LSTM (PyTorch)
│   ├── rf_model.pkl                    # Random Forest (sklearn)
│   ├── xgb_model.pkl                   # XGBoost
│   ├── isolation_forest_model.pkl      # Isolation Forest
│   └── scaler.pkl                      # StandardScaler
├── results/
│   ├── evaluation_metrics.json         # All metrics
│   ├── roc_curve.png                   # ROC-AUC curve
│   ├── confusion_matrix.png            # Confusion matrix
│   ├── risk_score_distribution.png     # Score histograms
│   ├── feature_importance.png          # Top-15 features
│   ├── network_graph_visualization.png # Network graph
│   ├── temporal_pattern_detection.png  # LSTM patterns
│   ├── shap_explanation.png            # SHAP importance
│   ├── fraud_cluster_visualization.png # t-SNE clusters
│   ├── adaptive_threshold_performance.png
│   └── model_comparison.png            # F1 comparison
├── notebooks/
│   └── crypto_fraud_detection.ipynb    # Full interactive analysis
├── src/
│   ├── __init__.py
│   ├── data_generator.py               # Phase 1: Synthetic data
│   ├── data_preprocessor.py            # Phase 2: Feature engineering
│   ├── graph_builder.py                # Phase 3: Graph construction
│   ├── model_training.py               # Phases 4–6, 11–16: Training & eval
│   ├── risk_scoring_engine.py          # Phases 9–10: Scoring & explainability
│   └── utils.py                        # Shared utilities
└── README.md
```

---

## Installation

### Prerequisites

- Python 3.9+
- CUDA-capable GPU (optional, for faster training)

### Install Dependencies

```bash
pip install numpy pandas scikit-learn matplotlib seaborn networkx xgboost shap torch hmmlearn
```

For GNN support with PyTorch Geometric (optional):

```bash
pip install torch-geometric
```

---

## Quick Start

### Option 1: Run the Jupyter Notebook (Recommended)

```bash
jupyter notebook notebooks/crypto_fraud_detection.ipynb
```

The notebook runs the complete pipeline end-to-end with detailed explanations.

### Option 2: Run CLI Scripts

```bash
# Step 1: Generate synthetic dataset (50K transactions)
python -m src.data_generator

# Step 2: Preprocess & engineer features
python -m src.data_preprocessor

# Step 3: Build transaction graph
python -m src.graph_builder

# Step 4: Train all models & evaluate
python -m src.model_training
```

---

## Dataset

The synthetic dataset contains **50,000 transactions** across **2,000 wallets** (20% fraudulent):

| Feature | Description |
|---------|-------------|
| `wallet_id` | Unique wallet identifier |
| `sender_wallet` | Source wallet address |
| `receiver_wallet` | Destination wallet address |
| `amount` | Transaction amount (0.01–1000 ETH) |
| `timestamp` | Unix timestamp (past 1 year) |
| `gas_fee` | Gas price in Gwei |
| `token_type` | ERC-20 token contract address |
| `token_age_days` | Days since token creation |
| `transaction_frequency` | Txs per hour for sender |
| `average_amount` | Mean tx amount for sender |
| `volume_spike_indicator` | Sudden volume increase flag |
| `liquidity_pool_change` | % change in liquidity pool |
| `is_contract` | Receiver is a smart contract |
| `fraud_label` | 0 (normal) or 1 (fraud) |

### Embedded Fraud Patterns

- **Pump & Dump**: Coordinated high-volume buys → sell-off
- **Rug Pulls**: Liquidity injection → abrupt drain
- **Wash Trading**: Self-trading across linked wallets
- **Abnormal Sequences**: Irregular timing & amount spikes

---

## Models

| Model | Type | Purpose |
|-------|------|---------|
| Random Forest | Supervised | Baseline classification |
| XGBoost | Supervised | Advanced gradient boosting |
| Isolation Forest | Unsupervised | Anomaly detection |
| LSTM | Deep Learning | Temporal pattern detection |
| GNN (GCN) | Deep Learning | Wallet graph analysis |

---

## Results & Visualizations

After running the pipeline, the `results/` folder contains:

| File | Description |
|------|-------------|
| `evaluation_metrics.json` | ROC-AUC, F1 scores, classification report |
| `roc_curve.png` | ROC curve with AUC |
| `confusion_matrix.png` | True/predicted label heatmap |
| `risk_score_distribution.png` | Score distribution by class |
| `feature_importance.png` | Top-15 Random Forest features |
| `network_graph_visualization.png` | Transaction network (fraud highlighted) |
| `temporal_pattern_detection.png` | LSTM temporal patterns |
| `shap_explanation.png` | SHAP feature importance |
| `fraud_cluster_visualization.png` | t-SNE cluster visualization |
| `adaptive_threshold_performance.png` | Static vs adaptive thresholds |
| `model_comparison.png` | F1 score comparison across models |

---

## Configuration

Key parameters can be adjusted in each module:

- **Dataset size**: `NUM_TRANSACTIONS`, `NUM_WALLETS` in `data_generator.py`
- **Feature set**: `FEATURE_COLUMNS` in `utils.py`
- **Model hyperparameters**: See each training function in `model_training.py`
- **Threshold regimes**: `regime_adjustments` in `risk_scoring_engine.py`
- **LSTM sequence length**: `seq_len=30` in `_build_wallet_sequences()`

---

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

---

*Built for research and patent-filing purposes. The synthetic dataset does not represent real blockchain data.*
