"""
Model Training Module for Crypto Fraud Detection.

Implements:
  - Random Forest (baseline)
  - XGBoost (advanced)
  - Isolation Forest (anomaly detection)
  - LSTM Temporal Pattern Detection (Novelty #2)
  - GNN Wallet Clustering (Novelty #3)
  - Comprehensive Model Evaluation (Phase 11)

Usage
-----
    python -m src.model_training
"""

import sys
import json
import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve, f1_score, accuracy_score,
)

try:
    from src.utils import (
        setup_logging, set_seed, ensure_directories, normalize_scores,
        save_json, get_device, FEATURE_COLUMNS, FEATURE_DISPLAY_NAMES,
        DATA_PROCESSED_DIR, MODELS_DIR, RESULTS_DIR,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.utils import (
        setup_logging, set_seed, ensure_directories, normalize_scores,
        save_json, get_device, FEATURE_COLUMNS, FEATURE_DISPLAY_NAMES,
        DATA_PROCESSED_DIR, MODELS_DIR, RESULTS_DIR,
    )

warnings.filterwarnings("ignore", category=FutureWarning)
logger = setup_logging("model_training")
SEED = 42


# ==============================================================================
# Helper: resolve feature columns available in a DataFrame
# ==============================================================================
def _resolve_features(df: pd.DataFrame) -> List[str]:
    return [c for c in FEATURE_COLUMNS if c in df.columns]


# ==============================================================================
# 1. Random Forest
# ==============================================================================
def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    save_path: Optional[Path] = None,
) -> RandomForestClassifier:
    """Train a Random-Forest classifier and persist it."""
    logger.info("Training Random Forest (n_estimators=200, max_depth=15)...")

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        random_state=SEED,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    if save_path is None:
        save_path = MODELS_DIR / "rf_model.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(rf, f)
    logger.info("Random Forest saved → %s", save_path)

    return rf


# ==============================================================================
# 2. XGBoost
# ==============================================================================
def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    save_path: Optional[Path] = None,
):
    """Train an XGBoost classifier and persist it."""
    import xgboost as xgb

    logger.info("Training XGBoost (n_estimators=200, max_depth=7)...")

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=7,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=SEED,
        use_label_encoder=False,
        eval_metric="logloss",
    )
    model.fit(X_train, y_train)

    if save_path is None:
        save_path = MODELS_DIR / "xgb_model.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(model, f)
    logger.info("XGBoost saved → %s", save_path)
    return model


# ==============================================================================
# 3. Isolation Forest
# ==============================================================================
def train_isolation_forest(
    X_train: np.ndarray,
    save_path: Optional[Path] = None,
) -> IsolationForest:
    """Train an Isolation Forest for unsupervised anomaly detection."""
    logger.info("Training Isolation Forest (contamination=0.2)...")

    iso = IsolationForest(
        contamination=0.2,
        random_state=SEED,
        n_jobs=-1,
    )
    iso.fit(X_train)

    if save_path is None:
        save_path = MODELS_DIR / "isolation_forest_model.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(iso, f)
    logger.info("Isolation Forest saved → %s", save_path)
    return iso


def isolation_forest_scores(iso: IsolationForest, X: np.ndarray) -> np.ndarray:
    """Convert Isolation Forest decision scores to [0, 1] risk scores."""
    raw = -iso.score_samples(X)
    return normalize_scores(raw)


# ==============================================================================
# 4. LSTM Temporal Pattern Detection  (Novelty #2)
# ==============================================================================
def _build_wallet_sequences(
    df: pd.DataFrame,
    feature_cols: List[str],
    seq_len: int = 30,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract fixed-length sequences per wallet for LSTM training.

    For each wallet, take the last *seq_len* transactions. If a wallet has
    fewer, pad with zeros.

    Returns
    -------
    X_seq : np.ndarray  shape (n_wallets, seq_len, n_features)
    y_seq : np.ndarray  shape (n_wallets,)   — majority label per wallet
    wallet_ids : np.ndarray
    """
    df = df.sort_values(["sender_wallet", "timestamp"])
    wallets = df["sender_wallet"].unique()
    n_features = len(feature_cols)
    X_list, y_list, w_list = [], [], []

    for wid in wallets:
        wdata = df[df["sender_wallet"] == wid]
        feats = wdata[feature_cols].values
        label = int(wdata["fraud_label"].mode().iloc[0])

        if len(feats) >= seq_len:
            seq = feats[-seq_len:]
        else:
            pad = np.zeros((seq_len - len(feats), n_features))
            seq = np.vstack([pad, feats])

        X_list.append(seq)
        y_list.append(label)
        w_list.append(wid)

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32), np.array(w_list)


def _build_timestep_sequences(
    df: pd.DataFrame,
    feature_cols: List[str],
    seq_len: int = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build fixed-length sequences for LSTM using Elliptic timestep ordering.

    For each transaction, we use the previous *seq_len* transactions
    within the same timestep as context (or pad with zeros).

    Returns
    -------
    X_seq : np.ndarray  shape (n_samples, seq_len, n_features)
    y_seq : np.ndarray  shape (n_samples,)
    tx_ids : np.ndarray
    """
    df = df.sort_values(["timestep", "txId"]).reset_index(drop=True)
    n_features = len(feature_cols)
    feat_values = df[feature_cols].fillna(0).values
    labels = df["fraud_label"].values
    tx_ids = df["txId"].values

    X_list = []
    for i in range(len(df)):
        start = max(0, i - seq_len + 1)
        seq = feat_values[start:i + 1]
        if len(seq) < seq_len:
            pad = np.zeros((seq_len - len(seq), n_features), dtype=np.float32)
            seq = np.vstack([pad, seq])
        X_list.append(seq)

    return (
        np.array(X_list, dtype=np.float32),
        labels.astype(np.float32),
        tx_ids,
    )


class TransactionLSTM:
    """
    Novelty #2 — LSTM for temporal pattern detection in transaction sequences.

    Wraps a PyTorch LSTM model with train / predict helpers.
    """

    class _LSTMModule:
        """Inner PyTorch module (defined without top-level torch import)."""
        pass  # replaced dynamically in __init__

    def __init__(
        self,
        input_size: int = 20,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        lr: float = 0.001,
        epochs: int = 20,
        batch_size: int = 64,
    ):
        import torch
        import torch.nn as nn

        self.device = get_device()
        self.epochs = epochs
        self.batch_size = batch_size
        self.input_size = input_size

        class LSTMNet(nn.Module):
            def __init__(self_m):
                super().__init__()
                self_m.lstm = nn.LSTM(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=dropout,
                )
                self_m.fc = nn.Linear(hidden_size, 1)
                self_m.sigmoid = nn.Sigmoid()

            def forward(self_m, x):
                lstm_out, _ = self_m.lstm(x)
                last_hidden = lstm_out[:, -1, :]
                return self_m.sigmoid(self_m.fc(last_hidden))

        self.model = LSTMNet().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.BCELoss()

    # ----- training -----
    def fit(self, X: np.ndarray, y: np.ndarray) -> "TransactionLSTM":
        import torch
        from torch.utils.data import TensorDataset, DataLoader

        logger.info("Training LSTM (%d epochs, batch=%d)...", self.epochs, self.batch_size)
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(self.device)

        ds = TensorDataset(X_t, y_t)
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for epoch in range(1, self.epochs + 1):
            total_loss = 0.0
            for xb, yb in dl:
                self.optimizer.zero_grad()
                pred = self.model(xb)
                loss = self.criterion(pred, yb)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item() * len(xb)
            avg_loss = total_loss / len(ds)
            if epoch % 5 == 0 or epoch == 1:
                logger.info("  LSTM epoch %d/%d  loss=%.4f", epoch, self.epochs, avg_loss)
        return self

    # ----- prediction -----
    def predict(self, X: np.ndarray) -> np.ndarray:
        import torch
        self.model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
            preds = self.model(X_t).cpu().numpy().flatten()
        return preds

    # ----- persistence -----
    def save(self, path: Optional[Path] = None) -> Path:
        import torch
        if path is None:
            path = MODELS_DIR / "lstm_model.pth"
        torch.save(self.model.state_dict(), path)
        logger.info("LSTM saved → %s", path)
        return path

    def load(self, path: Optional[Path] = None) -> "TransactionLSTM":
        import torch
        if path is None:
            path = MODELS_DIR / "lstm_model.pth"
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        logger.info("LSTM loaded ← %s", path)
        return self


# ==============================================================================
# 5. GNN Wallet Clustering  (Novelty #3)
# ==============================================================================
class WalletFraudGNN:
    """
    Novelty #3 — Graph Neural Network for wallet-level fraud classification.

    Uses a simplified GCN implementation that works with adjacency matrices
    when torch_geometric is available, or falls back to a manual implementation.
    """

    def __init__(
        self,
        num_features: int = 20,
        hidden_channels: int = 64,
        lr: float = 0.01,
        epochs: int = 50,
    ):
        self.num_features = num_features
        self.hidden_channels = hidden_channels
        self.lr = lr
        self.epochs = epochs
        self.device = get_device()
        self._model = None
        self._use_pyg = False

        # Try PyTorch Geometric first
        try:
            from torch_geometric.nn import GCNConv
            self._use_pyg = True
            logger.info("Using PyTorch Geometric backend for GNN.")
        except ImportError:
            logger.info("torch_geometric not available — using manual GCN implementation.")

    def _build_model_pyg(self):
        import torch
        import torch.nn as nn
        from torch_geometric.nn import GCNConv

        nf, hc = self.num_features, self.hidden_channels

        class GCNNet(nn.Module):
            def __init__(self_m):
                super().__init__()
                self_m.conv1 = GCNConv(nf, hc)
                self_m.conv2 = GCNConv(hc, hc)
                self_m.conv3 = GCNConv(hc, 2)

            def forward(self_m, x, edge_index):
                x = self_m.conv1(x, edge_index).relu()
                x = self_m.conv2(x, edge_index).relu()
                x = self_m.conv3(x, edge_index)
                return torch.softmax(x, dim=1)

        self._model = GCNNet().to(self.device)

    def _build_model_manual(self):
        """Fallback: two-layer linear GCN using adjacency-matrix multiplication."""
        import torch
        import torch.nn as nn

        nf, hc = self.num_features, self.hidden_channels

        class ManualGCN(nn.Module):
            def __init__(self_m):
                super().__init__()
                self_m.w1 = nn.Linear(nf, hc)
                self_m.w2 = nn.Linear(hc, hc)
                self_m.w3 = nn.Linear(hc, 2)
                self_m.relu = nn.ReLU()

            def forward(self_m, x, adj):
                # GCN propagation: A * X * W
                x = self_m.relu(self_m.w1(torch.sparse.mm(adj, x) if adj.is_sparse else adj @ x))
                x = self_m.relu(self_m.w2(torch.sparse.mm(adj, x) if adj.is_sparse else adj @ x))
                x = self_m.w3(torch.sparse.mm(adj, x) if adj.is_sparse else adj @ x)
                return torch.softmax(x, dim=1)

        self._model = ManualGCN().to(self.device)

    # ----- build edge structures -----
    @staticmethod
    def prepare_graph_data(
        G, feature_cols: List[str], df: pd.DataFrame,
    ) -> Tuple:
        """
        Convert a NetworkX graph + DataFrame into tensors for training.

        Returns
        -------
        node_features : np.ndarray  (N, F)
        edge_index : np.ndarray     (2, E)  — for PyG
        adj_matrix : sp.csr_matrix  — for manual fallback
        labels : np.ndarray         (N,)
        wallet_order : list          wallet ids in row order
        """
        import scipy.sparse as sp

        wallet_order = list(G.nodes())
        wallet_to_idx = {w: i for i, w in enumerate(wallet_order)}
        N = len(wallet_order)

        # Node features: average of transaction features per wallet
        feat_cols = [c for c in feature_cols if c in df.columns]
        wallet_feats = df.groupby("sender_wallet")[feat_cols].mean().reindex(wallet_order).fillna(0).values

        # Labels
        labels = np.array([
            int(G.nodes[w].get("is_fraud", False)) for w in wallet_order
        ])

        # Edges
        src_list, dst_list = [], []
        for u, v in G.edges():
            if u in wallet_to_idx and v in wallet_to_idx:
                src_list.append(wallet_to_idx[u])
                dst_list.append(wallet_to_idx[v])

        edge_index = np.array([src_list, dst_list], dtype=np.int64)

        # Normalised adjacency (D^-0.5 * A * D^-0.5 + I)
        row = np.array(src_list + dst_list + list(range(N)))
        col = np.array(dst_list + src_list + list(range(N)))
        data = np.ones(len(row))
        adj = sp.csr_matrix((data, (row, col)), shape=(N, N))
        deg = np.array(adj.sum(axis=1)).flatten()
        deg_inv_sqrt = np.power(deg, -0.5, where=deg > 0)
        deg_inv_sqrt[deg == 0] = 0
        D_inv_sqrt = sp.diags(deg_inv_sqrt)
        adj_norm = D_inv_sqrt @ adj @ D_inv_sqrt

        return wallet_feats, edge_index, adj_norm, labels, wallet_order

    # ----- Elliptic: nodes = transactions -----
    @staticmethod
    def prepare_elliptic_graph_data(
        df: pd.DataFrame,
        edges_df: pd.DataFrame,
        feature_cols: List[str],
    ) -> Tuple:
        """
        Prepare graph data for the Elliptic dataset where nodes are
        transactions (not wallets).

        Returns
        -------
        node_features : np.ndarray  (N, F)
        edge_index : np.ndarray     (2, E)
        adj_matrix : sp.csr_matrix
        labels : np.ndarray         (N,)
        tx_order : list              txIds in row order
        """
        import scipy.sparse as sp

        tx_order = df["txId"].tolist()
        tx_to_idx = {tx: i for i, tx in enumerate(tx_order)}
        N = len(tx_order)

        # Node features
        feat_cols = [c for c in feature_cols if c in df.columns]
        node_feats = df[feat_cols].fillna(0).values.astype(np.float32)

        # Labels
        labels = df["fraud_label"].values.astype(np.int64)

        # Edges (filter to nodes in tx_order)
        src_list, dst_list = [], []
        for _, row in edges_df.iterrows():
            s = row["txId1"]
            d = row["txId2"]
            if s in tx_to_idx and d in tx_to_idx:
                src_list.append(tx_to_idx[s])
                dst_list.append(tx_to_idx[d])

        edge_index = np.array([src_list, dst_list], dtype=np.int64)

        # Normalised adjacency
        row = np.array(src_list + dst_list + list(range(N)))
        col = np.array(dst_list + src_list + list(range(N)))
        data = np.ones(len(row))
        adj = sp.csr_matrix((data, (row, col)), shape=(N, N))
        deg = np.array(adj.sum(axis=1)).flatten()
        deg_inv_sqrt = np.power(deg, -0.5, where=deg > 0)
        deg_inv_sqrt[deg == 0] = 0
        D_inv_sqrt = sp.diags(deg_inv_sqrt)
        adj_norm = D_inv_sqrt @ adj @ D_inv_sqrt

        return node_feats, edge_index, adj_norm, labels, tx_order

    # ----- training -----
    def fit(
        self,
        node_features: np.ndarray,
        edge_index: np.ndarray,
        adj_norm,
        labels: np.ndarray,
        train_mask: Optional[np.ndarray] = None,
    ) -> "WalletFraudGNN":
        import torch
        import torch.nn as nn
        import scipy.sparse as sp

        if self._use_pyg:
            self._build_model_pyg()
        else:
            self._build_model_manual()

        device = self.device
        X = torch.tensor(node_features, dtype=torch.float32).to(device)
        y = torch.tensor(labels, dtype=torch.long).to(device)

        if train_mask is None:
            train_mask = np.ones(len(labels), dtype=bool)
        mask = torch.tensor(train_mask, dtype=torch.bool).to(device)

        if self._use_pyg:
            ei = torch.tensor(edge_index, dtype=torch.long).to(device)
        else:
            # Convert sparse scipy to torch sparse
            adj_coo = sp.coo_matrix(adj_norm)
            indices = torch.tensor(np.vstack([adj_coo.row, adj_coo.col]), dtype=torch.long)
            values = torch.tensor(adj_coo.data, dtype=torch.float32)
            adj_t = torch.sparse_coo_tensor(indices, values, adj_coo.shape).to(device)

        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr, weight_decay=5e-4)
        criterion = nn.CrossEntropyLoss()

        logger.info("Training GNN (%d epochs)...", self.epochs)
        self._model.train()
        for epoch in range(1, self.epochs + 1):
            optimizer.zero_grad()
            if self._use_pyg:
                out = self._model(X, ei)
            else:
                out = self._model(X, adj_t)
            loss = criterion(out[mask], y[mask])
            loss.backward()
            optimizer.step()
            if epoch % 10 == 0 or epoch == 1:
                acc = (out[mask].argmax(dim=1) == y[mask]).float().mean().item()
                logger.info("  GNN epoch %d/%d  loss=%.4f  acc=%.3f", epoch, self.epochs, loss.item(), acc)

        return self

    # ----- prediction -----
    def predict(self, node_features: np.ndarray, edge_index: np.ndarray, adj_norm=None) -> np.ndarray:
        import torch
        import scipy.sparse as sp

        self._model.eval()
        device = self.device
        X = torch.tensor(node_features, dtype=torch.float32).to(device)

        with torch.no_grad():
            if self._use_pyg:
                ei = torch.tensor(edge_index, dtype=torch.long).to(device)
                out = self._model(X, ei)
            else:
                adj_coo = sp.coo_matrix(adj_norm)
                indices = torch.tensor(np.vstack([adj_coo.row, adj_coo.col]), dtype=torch.long)
                values = torch.tensor(adj_coo.data, dtype=torch.float32)
                adj_t = torch.sparse_coo_tensor(indices, values, adj_coo.shape).to(device)
                out = self._model(X, adj_t)
        # Return fraud probability (class-1)
        return out[:, 1].cpu().numpy()

    # ----- persistence -----
    def save(self, path: Optional[Path] = None) -> Path:
        import torch
        if path is None:
            path = MODELS_DIR / "gnn_model.pth"
        torch.save(self._model.state_dict(), path)
        logger.info("GNN saved → %s", path)
        return path


# ==============================================================================
# 6. Model Evaluator  (Phase 11)
# ==============================================================================
class ModelEvaluator:
    """Comprehensive evaluation with metrics and publication-quality plots."""

    def __init__(self, results_dir: Optional[Path] = None):
        self.results_dir = Path(results_dir or RESULTS_DIR)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    # ----- main entry -----
    def evaluate_all_models(
        self,
        y_test: np.ndarray,
        rf_pred: np.ndarray,
        xgb_pred: np.ndarray,
        lstm_pred: np.ndarray,
        gnn_pred: np.ndarray,
        final_pred: np.ndarray,
        rf_model: Optional[RandomForestClassifier] = None,
        feature_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        metrics: Dict[str, Any] = {}

        # -- Classification report --
        y_binary = (final_pred > 0.5).astype(int)
        report = classification_report(y_test, y_binary, output_dict=True)
        metrics["final_model"] = report
        logger.info("=== FINAL MODEL METRICS ===")
        logger.info("\n%s", classification_report(y_test, y_binary))

        # -- ROC Curve --
        self._plot_roc(y_test, final_pred, metrics)

        # -- Confusion Matrix --
        self._plot_confusion_matrix(y_test, y_binary)

        # -- Risk Score Distribution --
        self._plot_risk_distribution(y_test, final_pred)

        # -- Feature Importance --
        if rf_model is not None:
            self._plot_feature_importance(rf_model, feature_names)

        # -- Model Comparison --
        model_names = ["Random Forest", "XGBoost", "LSTM", "GNN", "Hybrid"]
        preds = [rf_pred, xgb_pred, lstm_pred, gnn_pred, final_pred]
        f1_scores = []
        for p in preds:
            f1_scores.append(f1_score(y_test, (p > 0.5).astype(int), zero_division=0))
        self._plot_model_comparison(model_names, f1_scores)
        metrics["f1_scores"] = dict(zip(model_names, [float(s) for s in f1_scores]))
        metrics["roc_auc"] = metrics.get("roc_auc", 0)

        # -- Save JSON --
        save_json(
            {
                "roc_auc": metrics["roc_auc"],
                "f1_scores": metrics["f1_scores"],
                "final_model_metrics": metrics["final_model"],
            },
            self.results_dir / "evaluation_metrics.json",
        )
        logger.info("Evaluation metrics saved → %s", self.results_dir / "evaluation_metrics.json")
        return metrics

    # ---- individual plots ----

    def _plot_roc(self, y_test, final_pred, metrics):
        fpr, tpr, _ = roc_curve(y_test, final_pred)
        roc_auc = auc(fpr, tpr)
        metrics["roc_auc"] = float(roc_auc)

        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate", fontsize=12)
        plt.ylabel("True Positive Rate", fontsize=12)
        plt.title("ROC Curve — Fraud Detection Model", fontsize=14)
        plt.legend(loc="lower right", fontsize=11)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.results_dir / "roc_curve.png", dpi=300, bbox_inches="tight")
        plt.close()

    def _plot_confusion_matrix(self, y_test, y_binary):
        cm = confusion_matrix(y_test, y_binary)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=True,
                    xticklabels=["Licit", "Illicit"], yticklabels=["Licit", "Illicit"])
        plt.title("Confusion Matrix", fontsize=14)
        plt.ylabel("True Label", fontsize=12)
        plt.xlabel("Predicted Label", fontsize=12)
        plt.tight_layout()
        plt.savefig(self.results_dir / "confusion_matrix.png", dpi=300, bbox_inches="tight")
        plt.close()

    def _plot_risk_distribution(self, y_test, final_pred):
        plt.figure(figsize=(12, 6))
        plt.hist(final_pred[y_test == 0], bins=50, alpha=0.7, label="Normal", color="green")
        plt.hist(final_pred[y_test == 1], bins=50, alpha=0.7, label="Fraud", color="red")
        plt.xlim([0.0, 1.0])
        plt.xlabel("Risk Score", fontsize=12)
        plt.ylabel("Number of Wallets", fontsize=12)
        plt.title("Risk Score Distribution: Normal vs Fraudulent Wallets", fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.results_dir / "risk_score_distribution.png", dpi=300, bbox_inches="tight")
        plt.close()

    def _plot_feature_importance(self, rf_model, feature_names):
        importance = rf_model.feature_importances_
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(len(importance))]
        else:
            feature_names = [FEATURE_DISPLAY_NAMES.get(n, n) for n in feature_names]

        top_k = min(15, len(importance))
        sorted_idx = np.argsort(importance)[-top_k:]

        plt.figure(figsize=(10, 8))
        plt.barh(range(top_k), importance[sorted_idx], color="steelblue")
        plt.yticks(range(top_k), [feature_names[i] for i in sorted_idx])
        plt.xlabel("Importance Score", fontsize=12)
        plt.ylabel("Feature", fontsize=12)
        plt.title("Top 15 Feature Importance (Random Forest)", fontsize=14)
        plt.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.results_dir / "feature_importance.png", dpi=300, bbox_inches="tight")
        plt.close()

    def _plot_model_comparison(self, model_names, f1_scores):
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
        plt.figure(figsize=(10, 6))
        bars = plt.bar(model_names, f1_scores, color=colors[:len(model_names)])
        for bar, score in zip(bars, f1_scores):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f"{score:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
        plt.xlabel("Model", fontsize=12)
        plt.ylabel("F1 Score", fontsize=12)
        plt.title("Model Comparison: F1 Scores", fontsize=14)
        plt.ylim([0, 1.1])
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.results_dir / "model_comparison.png", dpi=300, bbox_inches="tight")
        plt.close()


# ==============================================================================
# 7. Visualization Helpers (Temporal, SHAP, Clusters, Adaptive)
# ==============================================================================
def visualize_temporal_patterns(
    wallet_sequences: np.ndarray,
    lstm_predictions: np.ndarray,
    output_path: Optional[str] = None,
) -> None:
    """Plot LSTM temporal patterns (Phase 13)."""
    if output_path is None:
        output_path = str(RESULTS_DIR / "temporal_pattern_detection.png")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Normal wallet sequences
    normal_mask = lstm_predictions < 0.3
    if normal_mask.sum() > 0:
        seqs = wallet_sequences[normal_mask][:5]
        # plot amount column (first feature in sequence)
        for s in seqs:
            axes[0, 0].plot(s[:, 0], alpha=0.7)
    axes[0, 0].set_title("Normal Wallet: Transaction Sequences", fontsize=12)
    axes[0, 0].set_xlabel("Sequence Step", fontsize=10)
    axes[0, 0].set_ylabel("Feature Value", fontsize=10)
    axes[0, 0].grid(alpha=0.3)

    # Fraudulent wallet sequences
    fraud_mask = lstm_predictions > 0.7
    if fraud_mask.sum() > 0:
        seqs = wallet_sequences[fraud_mask][:5]
        for s in seqs:
            axes[0, 1].plot(s[:, 0], alpha=0.7, color="red")
    axes[0, 1].set_title("Fraudulent Wallet: Transaction Sequences", fontsize=12)
    axes[0, 1].set_xlabel("Sequence Step", fontsize=10)
    axes[0, 1].set_ylabel("Feature Value", fontsize=10)
    axes[0, 1].grid(alpha=0.3)

    # LSTM prediction distribution
    axes[1, 0].hist(lstm_predictions, bins=50, alpha=0.7, color="steelblue")
    axes[1, 0].set_title("LSTM Risk Score Distribution", fontsize=12)
    axes[1, 0].set_xlabel("Risk Score", fontsize=10)
    axes[1, 0].set_ylabel("Frequency", fontsize=10)
    axes[1, 0].grid(alpha=0.3)

    # Scatter of anomaly scores
    axes[1, 1].scatter(
        range(len(lstm_predictions)), lstm_predictions,
        c=lstm_predictions, cmap="RdYlGn_r", alpha=0.6, s=10,
    )
    axes[1, 1].axhline(y=0.5, color="red", linestyle="--", label="Decision Threshold")
    axes[1, 1].set_title("Temporal Anomaly Scores Over Time", fontsize=12)
    axes[1, 1].set_xlabel("Transaction Index", fontsize=10)
    axes[1, 1].set_ylabel("Risk Score", fontsize=10)
    axes[1, 1].set_ylim([0.0, 1.05])
    axes[1, 1].legend(fontsize=9)
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Temporal patterns saved → %s", output_path)


def visualize_shap_explanations(
    xgb_model,
    X_test: np.ndarray,
    feature_names: Optional[List[str]] = None,
    output_path: Optional[str] = None,
) -> None:
    """SHAP feature-importance bar chart (Phase 14)."""
    import shap

    if output_path is None:
        output_path = str(RESULTS_DIR / "shap_explanation.png")

    logger.info("Computing SHAP values (may take a moment)...")
    explainer = shap.TreeExplainer(xgb_model)
    sample = X_test[:min(200, len(X_test))]
    shap_values = explainer.shap_values(sample)

    plt.figure(figsize=(12, 8))
    if feature_names is not None:
        display_names = [FEATURE_DISPLAY_NAMES.get(n, n) for n in feature_names]
        shap.summary_plot(shap_values, sample, feature_names=display_names, plot_type="bar", show=False)
    else:
        shap.summary_plot(shap_values, sample, plot_type="bar", show=False)
    plt.title("SHAP Feature Importance: Fraud Detection Model")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("SHAP explanation saved → %s", output_path)


def visualize_fraud_clusters(
    X_test: np.ndarray,
    y_test: np.ndarray,
    output_path: Optional[str] = None,
    max_samples: int = 1500,
) -> None:
    """t-SNE cluster visualization (Phase 15)."""
    from sklearn.manifold import TSNE

    if output_path is None:
        output_path = str(RESULTS_DIR / "fraud_cluster_visualization.png")

    n = min(max_samples, len(X_test))
    idx = np.random.RandomState(SEED).choice(len(X_test), size=n, replace=False)
    X_sub = X_test[idx]
    y_sub = y_test[idx]

    logger.info("Running t-SNE on %d samples...", n)
    tsne = TSNE(n_components=2, random_state=SEED, perplexity=30)
    X_2d = tsne.fit_transform(X_sub)

    plt.figure(figsize=(12, 10))
    normal_mask = y_sub == 0
    fraud_mask = y_sub == 1
    plt.scatter(X_2d[normal_mask, 0], X_2d[normal_mask, 1],
                c="green", alpha=0.5, label="Normal", s=40)
    plt.scatter(X_2d[fraud_mask, 0], X_2d[fraud_mask, 1],
                c="red", alpha=0.6, label="Fraudulent", s=50)
    plt.xlabel("t-SNE Component 1", fontsize=12)
    plt.ylabel("t-SNE Component 2", fontsize=12)
    plt.title("Fraud Cluster Visualization (t-SNE)", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Fraud clusters saved → %s", output_path)


def visualize_adaptive_threshold_performance(
    y_test: np.ndarray,
    final_predictions: np.ndarray,
    output_path: Optional[str] = None,
) -> None:
    """
    Compare static vs adaptive threshold across simulated market regimes (Phase 16).

    We simulate regime assignments proportionally and show F1 improvement.
    """
    if output_path is None:
        output_path = str(RESULTS_DIR / "adaptive_threshold_performance.png")

    rng = np.random.RandomState(SEED)
    regimes = rng.choice([0, 1, 2], size=len(y_test), p=[0.4, 0.35, 0.25])

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    regime_names = ["Bull Market", "Bear Market", "Volatile Market"]
    static_thresholds = [0.5, 0.5, 0.5]
    adaptive_thresholds = [0.4, 0.6, 0.75]

    for idx, regime in enumerate([0, 1, 2]):
        mask = regimes == regime
        y_r = y_test[mask]
        p_r = final_predictions[mask]

        if len(y_r) == 0 or y_r.sum() == 0:
            static_f1 = adaptive_f1 = 0.0
        else:
            static_f1 = f1_score(y_r, (p_r > static_thresholds[idx]).astype(int), zero_division=0)
            adaptive_f1 = f1_score(y_r, (p_r > adaptive_thresholds[idx]).astype(int), zero_division=0)

        bars = axes[idx].bar(
            [0, 1], [static_f1, adaptive_f1],
            color=["steelblue", "orange"], alpha=0.7,
        )
        axes[idx].set_ylim([0, 1])
        axes[idx].set_title(
            f"{regime_names[idx]}\n(Static: {static_thresholds[idx]:.2f}, "
            f"Adaptive: {adaptive_thresholds[idx]:.2f})"
        )
        axes[idx].set_ylabel("F1 Score")
        axes[idx].set_xticks([0, 1])
        axes[idx].set_xticklabels(["Static", "Adaptive"])
        axes[idx].grid(axis="y", alpha=0.3)
        # Annotate bars
        for bar, val in zip(bars, [static_f1, adaptive_f1]):
            axes[idx].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                           f"{val:.3f}", ha="center", fontsize=10)

    plt.suptitle("Adaptive Threshold Performance Across Market Regimes", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Adaptive threshold plot saved → %s", output_path)


# ==============================================================================
# CLI — Full training pipeline
# ==============================================================================
def run_full_training_pipeline() -> Dict[str, Any]:
    """
    End-to-end training pipeline:
      1. Load processed data
      2. Train all models
      3. Generate predictions
      4. Evaluate & visualise
    """
    set_seed(SEED)
    ensure_directories()

    # --- Load data ---
    train_path = DATA_PROCESSED_DIR / "train_data.csv"
    test_path = DATA_PROCESSED_DIR / "test_data.csv"
    if not train_path.exists() or not test_path.exists():
        logger.error("Processed data not found. Run data_preprocessor first.")
        sys.exit(1)

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    feature_cols = _resolve_features(train_df)

    X_train = train_df[feature_cols].fillna(0).values
    y_train = train_df["fraud_label"].values
    X_test = test_df[feature_cols].fillna(0).values
    y_test = test_df["fraud_label"].values

    logger.info("Training data: %s | Test data: %s | Features: %d",
                X_train.shape, X_test.shape, len(feature_cols))

    # --- 1. Random Forest ---
    rf = train_random_forest(X_train, y_train)
    rf_pred = rf.predict_proba(X_test)[:, 1]

    # --- 2. XGBoost ---
    xgb_model = train_xgboost(X_train, y_train)
    xgb_pred = xgb_model.predict_proba(X_test)[:, 1]

    # --- 3. Isolation Forest ---
    iso = train_isolation_forest(X_train)
    iso_pred = isolation_forest_scores(iso, X_test)

    # --- 4. LSTM ---
    all_df = pd.concat([train_df, test_df], ignore_index=True)
    X_seq_all, y_seq_all, w_ids_all = _build_wallet_sequences(all_df, feature_cols, seq_len=30)

    # Split by wallets present in train vs test
    train_wallets = set(train_df["sender_wallet"].unique())
    train_mask_seq = np.array([w in train_wallets for w in w_ids_all])

    X_seq_train = X_seq_all[train_mask_seq]
    y_seq_train = y_seq_all[train_mask_seq]
    X_seq_test = X_seq_all[~train_mask_seq] if (~train_mask_seq).sum() > 0 else X_seq_all

    lstm = TransactionLSTM(input_size=len(feature_cols), epochs=20, batch_size=64)
    lstm.fit(X_seq_train, y_seq_train)
    lstm.save()

    # For per-transaction LSTM scores, assign wallet-level score
    lstm_wallet_scores = lstm.predict(X_seq_all)
    wallet_score_map = dict(zip(w_ids_all, lstm_wallet_scores))
    lstm_pred_test = np.array([
        wallet_score_map.get(w, 0.5) for w in test_df["sender_wallet"]
    ])

    # --- 5. GNN ---
    from src.graph_builder import load_graph
    try:
        G = load_graph()
    except FileNotFoundError:
        logger.warning("Graph not found — building from raw data.")
        from src.graph_builder import build_transaction_graph, save_graph
        raw_df = pd.read_csv(DATA_PROCESSED_DIR / ".." / "raw" / "synthetic_dataset.csv")
        G = build_transaction_graph(raw_df)
        save_graph(G)

    node_feats, edge_index, adj_norm, node_labels, wallet_order = \
        WalletFraudGNN.prepare_graph_data(G, feature_cols, all_df)

    gnn = WalletFraudGNN(num_features=node_feats.shape[1], hidden_channels=64, epochs=50)
    gnn.fit(node_feats, edge_index, adj_norm, node_labels)
    gnn.save()

    gnn_wallet_scores = gnn.predict(node_feats, edge_index, adj_norm)
    gnn_wallet_map = dict(zip(wallet_order, gnn_wallet_scores))
    gnn_pred_test = np.array([
        gnn_wallet_map.get(w, 0.5) for w in test_df["sender_wallet"]
    ])

    # --- 6. Hybrid final score (Novelty #1) ---
    # Dynamic weights (simplified: balanced)
    w_rf, w_xgb, w_lstm, w_gnn, w_iso = 0.25, 0.25, 0.2, 0.15, 0.15
    final_pred = (
        w_rf * rf_pred
        + w_xgb * xgb_pred
        + w_lstm * lstm_pred_test
        + w_gnn * gnn_pred_test
        + w_iso * iso_pred
    )

    # --- 7. Evaluate ---
    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate_all_models(
        y_test=y_test,
        rf_pred=rf_pred,
        xgb_pred=xgb_pred,
        lstm_pred=lstm_pred_test,
        gnn_pred=gnn_pred_test,
        final_pred=final_pred,
        rf_model=rf,
        feature_names=feature_cols,
    )

    # --- 8. Additional visualisations ---
    visualize_temporal_patterns(X_seq_all, lstm_wallet_scores)
    visualize_fraud_clusters(X_test, y_test)
    visualize_adaptive_threshold_performance(y_test, final_pred)

    try:
        visualize_shap_explanations(xgb_model, X_test, feature_names=feature_cols)
    except Exception as e:
        logger.warning("SHAP visualization failed (non-critical): %s", e)

    logger.info("=== Full training pipeline complete ===")
    return metrics


if __name__ == "__main__":
    run_full_training_pipeline()
