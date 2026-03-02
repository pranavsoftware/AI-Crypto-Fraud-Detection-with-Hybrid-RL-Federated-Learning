"""
Elliptic Dataset Preprocessor for Crypto Fraud Detection.

Since the Elliptic dataset has 166 *anonymised* features (no raw amounts,
gas fees, etc.), the preprocessing is different from the synthetic pipeline:

  1. Load labelled transactions (illicit / licit only)
  2. Temporal train/test split (timesteps 1-34 train, 35-49 test)
  3. StandardScaler normalisation
  4. Graph-based feature augmentation (degree, PageRank, etc.)
  5. Behavioral fingerprinting via GMM (Novelty #5)
  6. Rug-pull risk proxy from feature variance (Novelty #4)

Usage
-----
    from src.elliptic_preprocessor import preprocess_elliptic_pipeline
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
        DATA_RAW_DIR, DATA_PROCESSED_DIR, MODELS_DIR,
        ELLIPTIC_ALL_FEATURES,
    )
    from src.elliptic_loader import (
        load_elliptic_dataset, get_temporal_split, get_elliptic_feature_columns,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.utils import (
        setup_logging, set_seed, ensure_directories, calculate_entropy,
        DATA_RAW_DIR, DATA_PROCESSED_DIR, MODELS_DIR,
        ELLIPTIC_ALL_FEATURES,
    )
    from src.elliptic_loader import (
        load_elliptic_dataset, get_temporal_split, get_elliptic_feature_columns,
    )

warnings.filterwarnings("ignore", category=FutureWarning)
logger = setup_logging("elliptic_preprocessor")
SEED = 42


# ==============================================================================
# 1. Graph-Based Feature Augmentation
# ==============================================================================
def augment_graph_features(
    df: pd.DataFrame,
    edges_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a transaction graph from the edge list and compute per-node
    centrality features to augment the base 166 features.

    New columns: degree_centrality, pagerank_score, in_degree, out_degree
    """
    logger.info("Computing graph-based features for Elliptic transactions...")

    G = nx.DiGraph()
    # Add all transaction nodes
    for txId in df["txId"].values:
        G.add_node(txId)

    # Add edges
    for _, row in edges_df.iterrows():
        if row["txId1"] in G and row["txId2"] in G:
            G.add_edge(row["txId1"], row["txId2"])

    logger.info("  Graph: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())

    # Degree centrality
    deg_c = nx.degree_centrality(G)
    df["degree_centrality"] = df["txId"].map(deg_c).fillna(0).astype(np.float32)

    # PageRank
    pr = nx.pagerank(G, max_iter=100)
    df["pagerank_score"] = df["txId"].map(pr).fillna(0).astype(np.float32)

    # In-degree and out-degree
    in_deg = dict(G.in_degree())
    out_deg = dict(G.out_degree())
    df["in_degree"] = df["txId"].map(in_deg).fillna(0).astype(np.float32)
    df["out_degree"] = df["txId"].map(out_deg).fillna(0).astype(np.float32)

    logger.info("  Graph features added: degree_centrality, pagerank_score, in_degree, out_degree")

    return df


# ==============================================================================
# 2. Temporal / Behavioral Features
# ==============================================================================
def compute_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-timestep statistics to capture temporal dynamics.
    """
    logger.info("Computing temporal features...")

    # Per-timestep fraud rate (label leakage–free: we use train stats only later)
    # For now just compute timestep-level aggregates of base features
    grp = df.groupby("timestep")

    # Timestep-level: fraction of high-activity transactions
    # Use feat_1 (first local feature) as a proxy for transaction value
    df["timestep_mean_feat1"] = grp["feat_1"].transform("mean")
    df["timestep_std_feat1"] = grp["feat_1"].transform("std").fillna(0)

    # Transaction position within timestep (early vs late)
    df["timestep_tx_count"] = grp["txId"].transform("count")

    # Deviation from timestep mean (anomaly signal)
    df["feat1_deviation"] = (df["feat_1"] - df["timestep_mean_feat1"]).abs()

    logger.info("  Temporal features added: timestep_mean_feat1, timestep_std_feat1, "
                "timestep_tx_count, feat1_deviation")
    return df


# ==============================================================================
# 3. Risk Proxy Features (Novelty #4 adaptation)
# ==============================================================================
def compute_risk_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute risk indicator proxies adapted from the Novelty #4
    (rug-pull risk) concept.  Since Elliptic features are anonymised,
    we use statistical anomaly indicators instead.
    """
    logger.info("Computing risk indicator features...")

    # Feature variance per transaction (how unusual is this tx?)
    feature_cols = get_elliptic_feature_columns(df)
    feat_matrix = df[feature_cols].values

    # Row-wise variance (transactions with extreme variance are suspicious)
    df["feature_variance"] = np.var(feat_matrix, axis=1)

    # Row-wise absolute mean (high absolute values = outlier)
    df["feature_abs_mean"] = np.mean(np.abs(feat_matrix), axis=1)

    # Local outlier proxy: how far is each feature from its column mean
    col_means = feat_matrix.mean(axis=0)
    col_stds = feat_matrix.std(axis=0)
    col_stds[col_stds < 1e-8] = 1.0
    z_scores = (feat_matrix - col_means) / col_stds
    df["max_zscore"] = np.max(np.abs(z_scores), axis=1)
    df["mean_zscore"] = np.mean(np.abs(z_scores), axis=1)

    logger.info("  Risk features added: feature_variance, feature_abs_mean, max_zscore, mean_zscore")
    return df


# ==============================================================================
# 4. Behavioral Fingerprinting (Novelty #5)
# ==============================================================================
def compute_behavioral_fingerprints(
    df: pd.DataFrame,
    n_components: int = 5,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Novelty #5 — Cluster transactions using GMM on a subset of features
    to create behavioral signatures.
    """
    logger.info("Computing behavioral fingerprints (Novelty #5)...")

    # Use first 10 local features as fingerprint basis
    fp_cols = [f"feat_{i}" for i in range(1, 11)]
    fp_data = df[fp_cols].fillna(0).values

    gmm = GaussianMixture(n_components=n_components, random_state=SEED, max_iter=200)
    clusters = gmm.fit_predict(fp_data)
    df["fingerprint_cluster"] = clusters

    # Compute per-cluster statistics
    fingerprints = {}
    for c in range(n_components):
        mask = clusters == c
        fingerprints[c] = {
            "cluster_id": c,
            "count": int(mask.sum()),
            "fraud_rate": float(df.loc[mask, "fraud_label"].mean()) if mask.sum() > 0 else 0,
        }

    logger.info("  %d clusters created. Fraud rates per cluster:", n_components)
    for c, info in fingerprints.items():
        logger.info("    Cluster %d: %d txs, fraud_rate=%.3f", c, info["count"], info["fraud_rate"])

    return df, fingerprints


# ==============================================================================
# 5. Scaling & Split
# ==============================================================================
def scale_and_split(
    df: pd.DataFrame,
    feature_cols: List[str],
    train_timesteps: int = 34,
) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Temporal split + StandardScaler.  Fits scaler on train only.
    """
    logger.info("Scaling and splitting (temporal: timesteps 1-%d train)...", train_timesteps)

    train_df = df[df["timestep"] <= train_timesteps].reset_index(drop=True)
    test_df = df[df["timestep"] > train_timesteps].reset_index(drop=True)

    # Fill NaN
    train_df[feature_cols] = train_df[feature_cols].fillna(0)
    test_df[feature_cols] = test_df[feature_cols].fillna(0)

    # Fit scaler on train, transform both
    scaler = StandardScaler()
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols].values)
    test_df[feature_cols] = scaler.transform(test_df[feature_cols].values)

    # Save
    ensure_directories()
    train_df.to_csv(DATA_PROCESSED_DIR / "train_data.csv", index=False)
    test_df.to_csv(DATA_PROCESSED_DIR / "test_data.csv", index=False)

    scaler_path = MODELS_DIR / "scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    logger.info(
        "Train: %d (fraud=%.2f%%) | Test: %d (fraud=%.2f%%) | Scaler → %s",
        len(train_df), train_df["fraud_label"].mean() * 100,
        len(test_df), test_df["fraud_label"].mean() * 100,
        scaler_path,
    )
    return train_df, test_df, scaler


# ==============================================================================
# Master Pipeline
# ==============================================================================
def preprocess_elliptic_pipeline(
    data_dir: Optional[Path] = None,
    train_timesteps: int = 34,
) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler, Dict, pd.DataFrame, List[str]]:
    """
    Full Elliptic preprocessing pipeline:
      1. Load dataset
      2. Graph-based features
      3. Temporal features
      4. Risk features (Novelty #4 proxy)
      5. Behavioral fingerprinting (Novelty #5)
      6. Scale & temporal split

    Returns
    -------
    train_df, test_df, scaler, fingerprints, edges_df, feature_cols
    """
    set_seed(SEED)
    ensure_directories()

    # Load
    df, edges_df = load_elliptic_dataset(data_dir)

    # Feature engineering
    df = augment_graph_features(df, edges_df)
    df = compute_temporal_features(df)
    df = compute_risk_features(df)
    df, fingerprints = compute_behavioral_fingerprints(df)

    # Define feature columns (all 166 base + engineered)
    base_features = get_elliptic_feature_columns()
    engineered_features = [
        "degree_centrality", "pagerank_score", "in_degree", "out_degree",
        "timestep_mean_feat1", "timestep_std_feat1", "timestep_tx_count",
        "feat1_deviation", "feature_variance", "feature_abs_mean",
        "max_zscore", "mean_zscore", "fingerprint_cluster",
    ]
    feature_cols = base_features + engineered_features
    feature_cols = [c for c in feature_cols if c in df.columns]

    logger.info("Total features: %d (base=%d + engineered=%d)",
                len(feature_cols), len(base_features), len(engineered_features))

    # Scale & split
    train_df, test_df, scaler = scale_and_split(df, feature_cols, train_timesteps)

    return train_df, test_df, scaler, fingerprints, edges_df, feature_cols


# ==============================================================================
# CLI
# ==============================================================================
if __name__ == "__main__":
    train_df, test_df, scaler, fps, edges_df, feat_cols = preprocess_elliptic_pipeline()
    print(f"\nTrain: {train_df.shape}  |  Test: {test_df.shape}")
    print(f"Features: {len(feat_cols)}")
    print(f"Fingerprint clusters: {len(fps)}")
