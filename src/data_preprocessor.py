"""
Data Preprocessor & Feature Engineering for Crypto Fraud Detection.

Computes transaction-level, graph-based, temporal, and risk-indicator
features.  Includes Novelty #4 (Liquidity Pool Anomaly / Rug-Pull Risk)
and Novelty #5 (Behavioral Fingerprinting with GMM).

Usage
-----
    python -m src.data_preprocessor
"""

import sys
import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture

try:
    from src.utils import (
        setup_logging, set_seed, ensure_directories, calculate_entropy,
        DATA_RAW_DIR, DATA_PROCESSED_DIR, MODELS_DIR, FEATURE_COLUMNS,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.utils import (
        setup_logging, set_seed, ensure_directories, calculate_entropy,
        DATA_RAW_DIR, DATA_PROCESSED_DIR, MODELS_DIR, FEATURE_COLUMNS,
    )

warnings.filterwarnings("ignore", category=FutureWarning)
logger = setup_logging("data_preprocessor")

SEED = 42


# ==============================================================================
# 1. Transaction-Level Features
# ==============================================================================
def compute_transaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-wallet transaction-level statistics and merge back.

    New columns: tx_frequency_per_hour, avg_transaction_amount,
    amount_std_dev, max_min_ratio, inter_transaction_time.
    """
    logger.info("Computing transaction-level features...")
    df = df.sort_values(["sender_wallet", "timestamp"]).reset_index(drop=True)

    grp = df.groupby("sender_wallet")

    # Tx frequency per hour
    time_span = grp["timestamp"].transform(lambda s: max((s.max() - s.min()) / 3600.0, 1.0))
    tx_count = grp["timestamp"].transform("count")
    df["tx_frequency_per_hour"] = tx_count / time_span

    # Average & std of amounts
    df["avg_transaction_amount"] = grp["amount"].transform("mean")
    df["amount_std_dev"] = grp["amount"].transform("std").fillna(0)

    # Max / min ratio
    amt_max = grp["amount"].transform("max")
    amt_min = grp["amount"].transform("min").clip(lower=0.01)
    df["max_min_ratio"] = amt_max / amt_min

    # Inter-transaction time (seconds)
    df["inter_transaction_time"] = grp["timestamp"].diff().fillna(0).abs()

    return df


# ==============================================================================
# 2. Graph-Based Features
# ==============================================================================
def compute_graph_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a lightweight transaction graph and compute centrality metrics
    per wallet: degree_centrality, pagerank_score, betweenness_centrality,
    clustering_coefficient, average_neighbor_degree.
    """
    logger.info("Computing graph-based features...")

    G = nx.DiGraph()
    edge_agg = (
        df.groupby(["sender_wallet", "receiver_wallet"])
        .agg(weight=("amount", "sum"), count=("amount", "size"))
        .reset_index()
    )
    for _, row in edge_agg.iterrows():
        G.add_edge(row["sender_wallet"], row["receiver_wallet"],
                    weight=row["weight"], count=row["count"])

    G_undirected = G.to_undirected()

    # Centrality metrics
    deg_cent = nx.degree_centrality(G)
    pr = nx.pagerank(G, weight="weight", max_iter=100)
    # Betweenness on a sample for speed (full graph can be slow)
    if G.number_of_nodes() > 500:
        bc = nx.betweenness_centrality(G, k=min(200, G.number_of_nodes()), seed=SEED)
    else:
        bc = nx.betweenness_centrality(G)
    cc = nx.clustering(G_undirected)
    avg_nd = nx.average_neighbor_degree(G_undirected)

    # Map to DataFrame
    df["degree_centrality"] = df["sender_wallet"].map(deg_cent).fillna(0)
    df["pagerank_score"] = df["sender_wallet"].map(pr).fillna(0)
    df["betweenness_centrality"] = df["sender_wallet"].map(bc).fillna(0)
    df["clustering_coefficient"] = df["sender_wallet"].map(cc).fillna(0)
    df["average_neighbor_degree"] = df["sender_wallet"].map(avg_nd).fillna(0)

    return df


# ==============================================================================
# 3. Temporal Features
# ==============================================================================
def compute_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute temporal / behavioral features per wallet.
    """
    logger.info("Computing temporal features...")
    grp = df.groupby("sender_wallet")

    max_ts = df["timestamp"].max()

    # Recency: days since last transaction
    last_ts = grp["timestamp"].transform("max")
    df["transaction_recency"] = (max_ts - last_ts) / 86400.0

    # Account age: first tx to last tx in days
    first_ts = grp["timestamp"].transform("min")
    df["account_age_days"] = (last_ts - first_ts) / 86400.0

    # Entropy of transaction timing
    df["transaction_entropy"] = grp["timestamp"].transform(
        lambda s: calculate_entropy(s.values)
    )

    # Behavioral shift: compare first-half mean amount vs second-half
    def _behavioral_shift(group: pd.DataFrame) -> float:
        if len(group) < 4:
            return 0.0
        mid = len(group) // 2
        first_half = group["amount"].iloc[:mid].mean()
        second_half = group["amount"].iloc[mid:].mean()
        if first_half < 1e-6:
            return 0.0
        return abs(second_half - first_half) / (first_half + 1e-9)

    shift_map = df.groupby("sender_wallet").apply(_behavioral_shift)
    df["behavioral_shift_indicator"] = df["sender_wallet"].map(shift_map).fillna(0)

    return df


# ==============================================================================
# 4. Risk Indicators
# ==============================================================================
def compute_risk_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute risk-focused features and Novelty #4 (Rug-Pull Risk).
    """
    logger.info("Computing risk indicators...")
    grp = df.groupby("sender_wallet")

    # Liquidity volatility (std of pool changes)
    df["liquidity_volatility"] = grp["liquidity_pool_change"].transform("std").fillna(0)

    # Contract interaction ratio
    total_tx = grp["is_contract"].transform("count")
    contract_tx = grp["is_contract"].transform("sum")
    df["contract_interaction_ratio"] = contract_tx / total_tx.clip(lower=1)

    # New token interaction (token_age_days < 7)
    df["new_token_interaction"] = (df["token_age_days"] < 7).astype(int)

    # Circular transaction indicator (simplified: sender == receiver anywhere)
    circular = df.groupby("sender_wallet", group_keys=False).apply(
        lambda g: pd.Series(
            int(g.name in g["receiver_wallet"].values),
            index=[g.name],
        )
    )
    circular_map = {idx: val for idx, val in zip(circular.index, circular.values)}
    df["circular_transaction_indicator"] = df["sender_wallet"].map(circular_map).fillna(0).astype(int)

    # --- Novelty #4: Rug-Pull Risk ---
    df = calculate_rugpull_risk(df)

    return df


def calculate_rugpull_risk(df: pd.DataFrame) -> pd.DataFrame:
    """
    Novelty #4 — Liquidity Pool Anomaly / Rug-Pull Risk Detection.

    Combines:
      * Liquidity drain velocity (rate of negative pool changes)
      * Activity magnitude (rolling transaction sum)
      * Time proximity (transactions within 1 hour flag)
    """
    logger.info("Computing rug-pull risk scores (Novelty #4)...")

    df = df.sort_values(["sender_wallet", "timestamp"]).reset_index(drop=True)

    # Liquidity drain velocity
    df["_liq_diff"] = df.groupby("sender_wallet")["liquidity_pool_change"].diff().abs().fillna(0)

    # Activity magnitude (rolling 100-tx window sum, per wallet)
    df["_act_mag"] = (
        df.groupby("sender_wallet")["amount"]
        .transform(lambda s: s.rolling(window=min(100, len(s)), min_periods=1).sum())
    )
    act_max = df["_act_mag"].max()
    if act_max > 0:
        df["_act_mag_norm"] = df["_act_mag"] / act_max
    else:
        df["_act_mag_norm"] = 0

    # Time proximity flag (gap < 1 hour)
    df["_time_prox"] = (
        df.groupby("sender_wallet")["timestamp"].diff().abs().fillna(9999) < 3600
    ).astype(int)

    # Composite score
    df["rugpull_risk"] = (
        df["_liq_diff"] * 0.4
        + df["_act_mag_norm"] * 0.3
        + df["_time_prox"] * 0.3
    )
    # Normalize to [0, 1]
    rr_max = df["rugpull_risk"].max()
    if rr_max > 0:
        df["rugpull_risk"] = df["rugpull_risk"] / rr_max

    df.drop(columns=["_liq_diff", "_act_mag", "_act_mag_norm", "_time_prox"], inplace=True)
    return df


# ==============================================================================
# 5. Novelty #5: Behavioral Fingerprinting (GMM)
# ==============================================================================
def generate_wallet_fingerprints(
    df: pd.DataFrame, n_components: int = 5
) -> Tuple[pd.DataFrame, Dict]:
    """
    Novelty #5 — Create a behavioural signature for each wallet and
    cluster wallets using a Gaussian Mixture Model.

    Returns
    -------
    df : pd.DataFrame
        With added 'fingerprint_cluster' column.
    fingerprints : dict
        Mapping wallet_id → fingerprint feature dict.
    """
    logger.info("Generating wallet fingerprints (Novelty #5)...")

    fingerprint_records = []
    wallet_ids = df["sender_wallet"].unique()

    for wid in wallet_ids:
        wdata = df[df["sender_wallet"] == wid]
        fp = {
            "wallet_id": wid,
            "inter_tx_delay_mean": wdata["inter_transaction_time"].mean(),
            "inter_tx_delay_std": wdata["inter_transaction_time"].std() if len(wdata) > 1 else 0,
            "amount_distribution_mean": wdata["amount"].mean(),
            "amount_distribution_std": wdata["amount"].std() if len(wdata) > 1 else 0,
            "gas_fee_preference_mean": wdata["gas_fee"].mean(),
            "transaction_time_entropy": calculate_entropy(wdata["timestamp"].values),
        }
        fingerprint_records.append(fp)

    fp_df = pd.DataFrame(fingerprint_records)
    fp_features = fp_df.drop(columns=["wallet_id"]).fillna(0).values

    gmm = GaussianMixture(n_components=n_components, random_state=SEED)
    clusters = gmm.fit_predict(fp_features)
    fp_df["fingerprint_cluster"] = clusters

    cluster_map = dict(zip(fp_df["wallet_id"], fp_df["fingerprint_cluster"]))
    df["fingerprint_cluster"] = df["sender_wallet"].map(cluster_map).fillna(0).astype(int)

    fingerprints = {row["wallet_id"]: row.to_dict() for _, row in fp_df.iterrows()}

    logger.info("Wallet fingerprints generated for %d wallets, %d clusters.",
                len(wallet_ids), n_components)
    return df, fingerprints


# ==============================================================================
# 6. Scaling & Train/Test Split
# ==============================================================================
def scale_and_split(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    test_size: float = 0.3,
    seed: int = SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Scale features with StandardScaler, split 70/30, and save artefacts.

    Returns
    -------
    train_df, test_df, scaler
    """
    logger.info("Scaling features and splitting data (%.0f%% test)...", test_size * 100)

    if feature_cols is None:
        # Use all FEATURE_COLUMNS that exist in df
        feature_cols = [c for c in FEATURE_COLUMNS if c in df.columns]

    # Ensure label & wallet columns are preserved
    non_feature_cols = ["sender_wallet", "receiver_wallet", "wallet_id",
                        "timestamp", "token_type", "fraud_label"]
    non_feature_cols = [c for c in non_feature_cols if c in df.columns]

    # Fill any NaN in features
    df[feature_cols] = df[feature_cols].fillna(0)

    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    # Shuffle and split
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    split_idx = int(len(df) * (1 - test_size))
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    test_df = df.iloc[split_idx:].reset_index(drop=True)

    # Save
    ensure_directories()
    train_df.to_csv(DATA_PROCESSED_DIR / "train_data.csv", index=False)
    test_df.to_csv(DATA_PROCESSED_DIR / "test_data.csv", index=False)

    scaler_path = MODELS_DIR / "scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    logger.info("Train: %d rows | Test: %d rows | Scaler → %s",
                len(train_df), len(test_df), scaler_path)

    return train_df, test_df, scaler


# ==============================================================================
# Master Pipeline
# ==============================================================================
def preprocess_pipeline(
    csv_path: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler, Dict]:
    """
    Run the full preprocessing pipeline:
      1. Load raw CSV
      2. Transaction features
      3. Graph features
      4. Temporal features
      5. Risk indicators (incl. Novelty #4)
      6. Behavioral fingerprinting (Novelty #5)
      7. Scale & split

    Returns
    -------
    train_df, test_df, scaler, wallet_fingerprints
    """
    set_seed(SEED)
    ensure_directories()

    if csv_path is None:
        csv_path = DATA_RAW_DIR / "synthetic_dataset.csv"

    logger.info("Loading raw data from %s ...", csv_path)
    df = pd.read_csv(csv_path)
    logger.info("Loaded %d rows, %d columns.", len(df), len(df.columns))

    df = compute_transaction_features(df)
    df = compute_graph_features(df)
    df = compute_temporal_features(df)
    df = compute_risk_indicators(df)
    df, fingerprints = generate_wallet_fingerprints(df)

    train_df, test_df, scaler = scale_and_split(df)

    return train_df, test_df, scaler, fingerprints


# ==============================================================================
# CLI
# ==============================================================================
if __name__ == "__main__":
    train_df, test_df, scaler, fps = preprocess_pipeline()
    print(f"\nTrain shape: {train_df.shape}")
    print(f"Test  shape: {test_df.shape}")
    print(f"Feature columns used: {[c for c in FEATURE_COLUMNS if c in train_df.columns]}")
