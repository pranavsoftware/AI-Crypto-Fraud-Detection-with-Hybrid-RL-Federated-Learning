"""
Elliptic Bitcoin Dataset Loader for Crypto Fraud Detection.

Downloads (via kagglehub) or loads the Elliptic Bitcoin dataset and
converts it into the format expected by the rest of the pipeline.

The Elliptic dataset contains:
  - 203,769 Bitcoin transactions across 49 timesteps
  - 166 anonymised features per transaction (94 local + 72 aggregated)
  - Labels: 1 = illicit (4,545), 2 = licit (42,019), unknown (157,205)
  - 234,355 directed edges (payment flows)

Reference
---------
Weber et al., "Anti-Money Laundering in Bitcoin: Experimenting with
Graph Convolutional Networks for Financial Forensics", KDD 2019.

Usage
-----
    from src.elliptic_loader import load_elliptic_dataset
"""

import sys
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd

try:
    from src.utils import (
        setup_logging, set_seed, ensure_directories,
        DATA_RAW_DIR, DATA_PROCESSED_DIR,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.utils import (
        setup_logging, set_seed, ensure_directories,
        DATA_RAW_DIR, DATA_PROCESSED_DIR,
    )

logger = setup_logging("elliptic_loader")

# Default path inside the project
ELLIPTIC_DIR = DATA_RAW_DIR / "elliptic"

# Feature column names for the 166 anonymised features
# Col 0 = txId, Col 1 = timestep, Cols 2-95 = local features, Cols 96-166 = aggregated
# After loading, feat_0 is renamed to 'timestep', so actual features are feat_1..feat_165
LOCAL_FEATURE_NAMES = [f"feat_{i}" for i in range(1, 94)]   # 93 local features
AGG_FEATURE_NAMES   = [f"feat_{i}" for i in range(94, 166)] # 72 aggregated features
ALL_FEATURE_NAMES   = LOCAL_FEATURE_NAMES + AGG_FEATURE_NAMES  # 165 features


def download_elliptic(target_dir: Optional[Path] = None) -> Path:
    """
    Download the Elliptic dataset using kagglehub and copy it into
    the project's data/raw/elliptic/ directory.

    Returns the path to the directory containing the three CSV files.
    """
    import shutil

    if target_dir is None:
        target_dir = ELLIPTIC_DIR
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    # Check if already present
    expected_files = [
        "elliptic_txs_features.csv",
        "elliptic_txs_classes.csv",
        "elliptic_txs_edgelist.csv",
    ]
    if all((target_dir / f).exists() for f in expected_files):
        logger.info("Elliptic dataset already present at %s", target_dir)
        return target_dir

    logger.info("Downloading Elliptic dataset via kagglehub...")
    import kagglehub
    cache_path = Path(kagglehub.dataset_download("ellipticco/elliptic-data-set"))

    # The download may have a nested folder
    for candidate in [cache_path, cache_path / "elliptic_bitcoin_dataset"]:
        if (candidate / "elliptic_txs_features.csv").exists():
            for f in expected_files:
                shutil.copy2(str(candidate / f), str(target_dir / f))
            logger.info("Elliptic dataset copied to %s", target_dir)
            return target_dir

    raise FileNotFoundError(
        f"Could not find Elliptic CSV files in {cache_path}. "
        "Please download manually from Kaggle."
    )


def load_elliptic_dataset(
    data_dir: Optional[Path] = None,
    include_unknown: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the Elliptic dataset and return a merged DataFrame of
    features + labels, and an edge DataFrame.

    Parameters
    ----------
    data_dir : Path, optional
        Directory containing the three CSV files.
    include_unknown : bool
        If True, keep transactions with unknown labels (unlabeled).
        Default False — only keep labelled transactions.

    Returns
    -------
    df : pd.DataFrame
        Columns: txId, timestep, 166 feature columns, fraud_label (0/1).
    edges_df : pd.DataFrame
        Columns: txId1, txId2.
    """
    if data_dir is None:
        data_dir = ELLIPTIC_DIR
    data_dir = Path(data_dir)

    logger.info("Loading Elliptic features from %s ...", data_dir)
    features_path = data_dir / "elliptic_txs_features.csv"
    classes_path = data_dir / "elliptic_txs_classes.csv"
    edges_path = data_dir / "elliptic_txs_edgelist.csv"

    # --- Features (no header) ---
    # CSV has 167 columns: col0=txId, col1=timestep (counted as first of the 166 features)
    # We separate timestep explicitly and keep 165 remaining feature columns
    features_df = pd.read_csv(features_path, header=None)
    raw_feature_names = [f"feat_{i}" for i in range(features_df.shape[1] - 1)]  # 166 names
    features_df.columns = ["txId"] + raw_feature_names
    # The first feature column is the timestep
    features_df.rename(columns={"feat_0": "timestep"}, inplace=True)
    features_df["timestep"] = features_df["timestep"].astype(int)
    logger.info("  Features loaded: %d transactions, %d columns",
                len(features_df), features_df.shape[1])

    # --- Classes ---
    classes_df = pd.read_csv(classes_path)
    classes_df.columns = ["txId", "class"]

    # Map: 1 = illicit → fraud_label=1,  2 = licit → fraud_label=0
    classes_df["fraud_label"] = classes_df["class"].map({"1": 1, "2": 0, 1: 1, 2: 0})
    # Handle string vs int class column
    classes_df["fraud_label"] = classes_df["fraud_label"].where(
        classes_df["class"].isin([1, 2, "1", "2"]), other=np.nan
    )

    # --- Merge ---
    df = features_df.merge(classes_df[["txId", "fraud_label"]], on="txId", how="left")

    if not include_unknown:
        n_before = len(df)
        df = df.dropna(subset=["fraud_label"]).reset_index(drop=True)
        df["fraud_label"] = df["fraud_label"].astype(int)
        logger.info("  Filtered to labelled only: %d → %d transactions", n_before, len(df))
    else:
        df["fraud_label"] = df["fraud_label"].fillna(-1).astype(int)

    # --- Edges ---
    edges_df = pd.read_csv(edges_path)
    edges_df.columns = ["txId1", "txId2"]
    logger.info("  Edges loaded: %d directed edges", len(edges_df))

    # Filter edges to only include labelled transactions
    valid_txIds = set(df["txId"].values)
    edges_df = edges_df[
        edges_df["txId1"].isin(valid_txIds) & edges_df["txId2"].isin(valid_txIds)
    ].reset_index(drop=True)
    logger.info("  Edges after filtering to labelled txs: %d", len(edges_df))

    logger.info(
        "Elliptic dataset ready: %d transactions (%d illicit, %d licit), %d edges",
        len(df),
        (df["fraud_label"] == 1).sum(),
        (df["fraud_label"] == 0).sum(),
        len(edges_df),
    )

    return df, edges_df


def get_elliptic_feature_columns(df: pd.DataFrame = None) -> list:
    """Return feature column names (excluding txId, timestep, fraud_label)."""
    if df is not None:
        return [c for c in df.columns if c not in ("txId", "timestep", "fraud_label")]
    return ALL_FEATURE_NAMES.copy()


def get_temporal_split(
    df: pd.DataFrame,
    train_timesteps: int = 34,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the dataset temporally: first N timesteps for training,
    the rest for testing.  This is the recommended split for Elliptic
    (paper uses timesteps 1-34 for train, 35-49 for test).

    Parameters
    ----------
    df : pd.DataFrame with 'timestep' column.
    train_timesteps : int
        Number of timesteps to use for training (default 34 per paper).

    Returns
    -------
    train_df, test_df
    """
    train_df = df[df["timestep"] <= train_timesteps].reset_index(drop=True)
    test_df = df[df["timestep"] > train_timesteps].reset_index(drop=True)

    logger.info(
        "Temporal split: train=%d (timesteps 1-%d), test=%d (timesteps %d-%d)",
        len(train_df), train_timesteps,
        len(test_df), train_timesteps + 1, int(df["timestep"].max()),
    )
    logger.info(
        "  Train fraud rate: %.2f%% (%d/%d)  |  Test fraud rate: %.2f%% (%d/%d)",
        train_df["fraud_label"].mean() * 100,
        train_df["fraud_label"].sum(), len(train_df),
        test_df["fraud_label"].mean() * 100,
        test_df["fraud_label"].sum(), len(test_df),
    )

    return train_df, test_df


# ==============================================================================
# CLI
# ==============================================================================
if __name__ == "__main__":
    ensure_directories()
    download_elliptic()
    df, edges = load_elliptic_dataset()
    print(f"\nDataset: {df.shape}")
    print(f"Fraud distribution:\n{df['fraud_label'].value_counts()}")
    print(f"\nEdges: {edges.shape}")

    train_df, test_df = get_temporal_split(df)
    print(f"\nTrain: {train_df.shape}, Test: {test_df.shape}")
