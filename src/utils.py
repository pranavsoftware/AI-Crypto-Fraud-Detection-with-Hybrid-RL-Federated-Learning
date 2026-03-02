"""
Utility functions for the Crypto Fraud Detection System.

Provides logging configuration, entropy calculation, directory management,
and common helper functions used across all modules.
"""

import os
import logging
import numpy as np
import random
import json
from typing import Any, Dict, List, Optional, Union
from pathlib import Path


# ==============================================================================
# Project Paths
# ==============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

ALL_DIRS = [DATA_RAW_DIR, DATA_PROCESSED_DIR, MODELS_DIR, RESULTS_DIR, NOTEBOOKS_DIR]


def ensure_directories() -> None:
    """Create all required project directories if they don't exist."""
    for d in ALL_DIRS:
        d.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# Logging
# ==============================================================================
def setup_logging(name: str = "crypto_fraud", level: int = logging.INFO) -> logging.Logger:
    """
    Configure and return a logger with console output.

    Parameters
    ----------
    name : str
        Logger name.
    level : int
        Logging level (default INFO).

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        handler = logging.StreamHandler()
        handler.setLevel(level)
        fmt = logging.Formatter(
            "[%(asctime)s] %(levelname)s - %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger


# ==============================================================================
# Reproducibility
# ==============================================================================
def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility across numpy, random, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


# ==============================================================================
# Mathematical Helpers
# ==============================================================================
def calculate_entropy(series: np.ndarray) -> float:
    """
    Calculate Shannon entropy of a numeric series.

    Discretises continuous values into 20 bins and computes the
    information-theoretic entropy of the resulting distribution.

    Parameters
    ----------
    series : array-like
        Numeric values.

    Returns
    -------
    float
        Shannon entropy (nats).
    """
    series = np.asarray(series, dtype=float)
    if len(series) < 2:
        return 0.0
    # Discretise into bins
    counts, _ = np.histogram(series, bins=min(20, len(series)))
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log(probs + 1e-12)))


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    """Min-max normalize an array to [0, 1]."""
    scores = np.asarray(scores, dtype=float)
    s_min, s_max = scores.min(), scores.max()
    if s_max - s_min < 1e-12:
        return np.zeros_like(scores)
    return (scores - s_min) / (s_max - s_min)


# ==============================================================================
# I/O Helpers
# ==============================================================================
def save_json(data: Dict[str, Any], filepath: Union[str, Path]) -> None:
    """Save a dictionary as a pretty-printed JSON file."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, default=str)


def load_json(filepath: Union[str, Path]) -> Dict[str, Any]:
    """Load a JSON file and return as dictionary."""
    with open(filepath, "r") as f:
        return json.load(f)


def get_device() -> "torch.device":
    """Return the best available torch device (CUDA > CPU)."""
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ==============================================================================
# Feature Name Registry
# ==============================================================================
# Canonical ordered list of model features (used across modules)
FEATURE_COLUMNS = [
    # Transaction-level
    "amount", "gas_fee", "token_age_days",
    "tx_frequency_per_hour", "avg_transaction_amount",
    "amount_std_dev", "max_min_ratio", "inter_transaction_time",
    # Graph-based
    "degree_centrality", "pagerank_score", "betweenness_centrality",
    "clustering_coefficient", "average_neighbor_degree",
    # Temporal
    "transaction_recency", "account_age_days",
    "transaction_entropy", "behavioral_shift_indicator",
    # Risk indicators
    "liquidity_volatility", "contract_interaction_ratio",
    "new_token_interaction", "circular_transaction_indicator",
    # Engineered
    "rugpull_risk", "volume_spike_indicator",
]

FEATURE_DISPLAY_NAMES = {
    "amount": "Transaction Amount",
    "gas_fee": "Gas Fee",
    "token_age_days": "Token Age (Days)",
    "tx_frequency_per_hour": "Transaction Frequency / Hour",
    "avg_transaction_amount": "Average Tx Amount",
    "amount_std_dev": "Amount Std Dev",
    "max_min_ratio": "Max/Min Amount Ratio",
    "inter_transaction_time": "Inter-Tx Time (s)",
    "degree_centrality": "Degree Centrality",
    "pagerank_score": "PageRank Score",
    "betweenness_centrality": "Betweenness Centrality",
    "clustering_coefficient": "Clustering Coefficient",
    "average_neighbor_degree": "Avg Neighbor Degree",
    "transaction_recency": "Transaction Recency (days)",
    "account_age_days": "Account Age (days)",
    "transaction_entropy": "Transaction Entropy",
    "behavioral_shift_indicator": "Behavioral Shift",
    "liquidity_volatility": "Liquidity Volatility",
    "contract_interaction_ratio": "Contract Interaction Ratio",
    "new_token_interaction": "New Token Interaction",
    "circular_transaction_indicator": "Circular Tx Indicator",
    "rugpull_risk": "Rug-Pull Risk",
    "volume_spike_indicator": "Volume Spike",
}


# ==============================================================================
# Elliptic Dataset Feature Registry
# ==============================================================================
# Column 0 of raw CSV = txId, column 1 = timestep (renamed from feat_0)
# Remaining 165 columns: feat_1 … feat_165
# Local features: feat_1 – feat_93 (93 cols, since feat_0 became timestep)
# Aggregated features: feat_94 – feat_165 (72 cols)
ELLIPTIC_LOCAL_FEATURES = [f"feat_{i}" for i in range(1, 94)]   # 93 features
ELLIPTIC_AGG_FEATURES   = [f"feat_{i}" for i in range(94, 166)] # 72 features
ELLIPTIC_ALL_FEATURES   = ELLIPTIC_LOCAL_FEATURES + ELLIPTIC_AGG_FEATURES  # 165 total

ELLIPTIC_FEATURE_DISPLAY_NAMES = {
    f"feat_{i}": f"Local Feature {i}" for i in range(1, 94)
}
ELLIPTIC_FEATURE_DISPLAY_NAMES.update({
    f"feat_{i}": f"Aggregated Feature {i}" for i in range(94, 166)
})
