# Data Directory

This directory contains the datasets used for training and evaluation. The actual data files are not included in the repository due to their large size.

## Structure

```
data/
├── raw/
│   ├── elliptic/              # Elliptic Bitcoin Transaction Dataset
│   │   ├── elliptic_txs_features.csv
│   │   ├── elliptic_txs_classes.csv
│   │   └── elliptic_txs_edgelist.csv
│   └── synthetic_dataset.csv  # Legacy synthetic dataset
└── processed/
    ├── train_data.csv         # Preprocessed training split
    └── test_data.csv          # Preprocessed testing split
```

## How to Generate

Run the notebook `notebooks/crypto_fraud_detection.ipynb` from the beginning. The Elliptic Bitcoin dataset will be automatically downloaded via `kagglehub` and preprocessed into the `processed/` directory.

### Elliptic Bitcoin Dataset

- **Source**: [Kaggle - Elliptic Bitcoin Dataset](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set)
- **Size**: 46,564 labelled transactions (4,545 illicit / 42,019 licit)
- **Features**: 165 features per transaction (94 local + 72 aggregate)
- **Edges**: 36,624 directed edges representing Bitcoin flows
- **Timesteps**: 49 temporal snapshots
