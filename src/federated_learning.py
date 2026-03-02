"""
Novelty #10 — Federated Learning for Privacy-Preserving Fraud Detection.

Simulates a Federated Averaging (FedAvg) protocol where multiple
"exchange nodes" (banks / exchanges) each hold a private partition of
the Elliptic dataset, train local neural-network models, and periodically
send model *parameters* (not data) to a central server for aggregation.

Key Design Choices
------------------
- **Data partitioning**: Transactions are split by timestep groups to
  simulate geographically / temporally separated data silos.
- **Local model**: A small MLP classifier (same architecture across nodes).
- **Aggregation**: Weighted FedAvg — parameters averaged proportionally
  to each node's dataset size.
- **Privacy guarantee**: Raw transaction data never leaves the node;
  only model weight tensors are exchanged.
- **Differential privacy** (optional): Gaussian noise can be added to
  local gradients before aggregation.

Usage
-----
    from src.federated_learning import (
        FederatedNode, FederatedServer, run_federated_training,
    )
"""

import sys
import copy
import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

try:
    from src.utils import (
        setup_logging, set_seed, ensure_directories,
        MODELS_DIR, RESULTS_DIR,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.utils import (
        setup_logging, set_seed, ensure_directories,
        MODELS_DIR, RESULTS_DIR,
    )

warnings.filterwarnings("ignore")
logger = setup_logging("federated_learning")
SEED = 42


# ==============================================================================
# 1. Local Model (shared architecture across all nodes)
# ==============================================================================
class FraudDetectorMLP(nn.Module):
    """
    Lightweight MLP for binary fraud classification.
    Shared architecture ensures parameter compatibility during FedAvg.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return sigmoid probabilities."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
        self.train()
        return torch.sigmoid(logits)


# ==============================================================================
# 2. Federated Node (client)
# ==============================================================================
class FederatedNode:
    """
    Represents one exchange / data silo in the federated network.

    Each node:
    - Holds a private data partition (features + labels).
    - Trains a local FraudDetectorMLP.
    - Returns model parameters (no raw data) to the server.
    """

    def __init__(
        self,
        node_id: str,
        X: np.ndarray,
        y: np.ndarray,
        input_dim: int,
        hidden_dim: int = 128,
        lr: float = 1e-3,
        batch_size: int = 128,
        device: str = "cpu",
        dp_noise_scale: float = 0.0,
    ):
        self.node_id = node_id
        self.n_samples = len(y)
        self.device = torch.device(device)
        self.dp_noise_scale = dp_noise_scale

        # Local data
        self.dataset = TensorDataset(
            torch.FloatTensor(X),
            torch.FloatTensor(y),
        )
        self.loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)

        # Local model
        self.model = FraudDetectorMLP(input_dim, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # Class imbalance — compute pos_weight
        n_pos = y.sum()
        n_neg = len(y) - n_pos
        pos_weight = n_neg / max(n_pos, 1)
        self.criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight], device=self.device)
        )

    def set_parameters(self, global_state_dict: dict):
        """Replace local model parameters with the global aggregated model."""
        self.model.load_state_dict(copy.deepcopy(global_state_dict))

    def get_parameters(self) -> dict:
        """Return local model parameters (optionally with DP noise)."""
        state = copy.deepcopy(self.model.state_dict())
        if self.dp_noise_scale > 0:
            for key in state:
                noise = torch.randn_like(state[key]) * self.dp_noise_scale
                state[key] = state[key] + noise
        return state

    def local_train(self, epochs: int = 3) -> Dict:
        """
        Train the local model for a few epochs on private data.

        Returns
        -------
        dict with 'loss', 'accuracy', 'n_samples'.
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for _ in range(epochs):
            for X_batch, y_batch in self.loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                self.optimizer.zero_grad()
                logits = self.model(X_batch)
                loss = self.criterion(logits, y_batch)
                loss.backward()

                # Gradient clipping for stability + DP
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                total_loss += loss.item() * len(y_batch)
                preds = (torch.sigmoid(logits) > 0.5).float()
                correct += (preds == y_batch).sum().item()
                total += len(y_batch)

        return {
            "loss": total_loss / max(total, 1),
            "accuracy": correct / max(total, 1),
            "n_samples": self.n_samples,
        }

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Evaluate the local model on arbitrary data."""
        self.model.eval()
        X_t = torch.FloatTensor(X).to(self.device)
        y_t = torch.FloatTensor(y).to(self.device)
        with torch.no_grad():
            logits = self.model(X_t)
            preds = (torch.sigmoid(logits) > 0.5).float()
            probs = torch.sigmoid(logits).cpu().numpy()

        correct = (preds == y_t).sum().item()
        tp = ((preds == 1) & (y_t == 1)).sum().item()
        fp = ((preds == 1) & (y_t == 0)).sum().item()
        fn = ((preds == 0) & (y_t == 1)).sum().item()
        tn = ((preds == 0) & (y_t == 0)).sum().item()

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)

        return {
            "accuracy": correct / len(y),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "probs": probs,
        }


# ==============================================================================
# 3. Federated Server (aggregator)
# ==============================================================================
class FederatedServer:
    """
    Central aggregation server implementing FedAvg.

    Steps per round:
    1. Broadcast global model to all nodes.
    2. Nodes train locally for E epochs.
    3. Collect updated parameters from each node.
    4. Weighted average of parameters → new global model.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.global_model = FraudDetectorMLP(input_dim, hidden_dim).to(self.device)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.round_metrics: List[Dict] = []

    def get_global_parameters(self) -> dict:
        return copy.deepcopy(self.global_model.state_dict())

    def aggregate(self, node_params: List[Tuple[dict, int]]):
        """
        FedAvg: weighted average of node parameters.

        Parameters
        ----------
        node_params : list of (state_dict, n_samples) tuples.
        """
        total_samples = sum(n for _, n in node_params)
        if total_samples == 0:
            return

        global_dict = self.global_model.state_dict()

        # Zero out global model (only float params)
        for key in global_dict:
            if global_dict[key].dtype in (torch.int64, torch.int32, torch.long):
                continue
            global_dict[key] = torch.zeros_like(global_dict[key])

        # Weighted sum
        for state_dict, n_samples in node_params:
            weight = n_samples / total_samples
            for key in global_dict:
                param = state_dict[key].to(self.device)
                # Skip integer params (e.g. BatchNorm num_batches_tracked)
                if param.dtype in (torch.int64, torch.int32, torch.long):
                    global_dict[key] = param
                else:
                    global_dict[key] += param * weight

        self.global_model.load_state_dict(global_dict)

    def evaluate_global(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Evaluate the global model on a test set."""
        self.global_model.eval()
        X_t = torch.FloatTensor(X).to(self.device)
        y_t = torch.FloatTensor(y).to(self.device)
        with torch.no_grad():
            logits = self.global_model(X_t)
            preds = (torch.sigmoid(logits) > 0.5).float()
            probs = torch.sigmoid(logits).cpu().numpy()

        correct = (preds == y_t).sum().item()
        tp = ((preds == 1) & (y_t == 1)).sum().item()
        fp = ((preds == 1) & (y_t == 0)).sum().item()
        fn = ((preds == 0) & (y_t == 1)).sum().item()
        tn = ((preds == 0) & (y_t == 0)).sum().item()

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)

        return {
            "accuracy": correct / len(y),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "probs": probs,
        }

    def save(self, path: Optional[Path] = None):
        path = path or MODELS_DIR / "federated_global_model.pth"
        torch.save({
            "model_state": self.global_model.state_dict(),
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
        }, path)
        logger.info("Global federated model saved → %s", path)


# ==============================================================================
# 4. Data Partitioning (simulates distributed exchanges)
# ==============================================================================
def partition_data_by_timestep(
    df: 'pd.DataFrame',
    feature_cols: List[str],
    n_nodes: int = 4,
) -> List[Tuple[np.ndarray, np.ndarray, str]]:
    """
    Partition transactions across federated nodes by timestep ranges.
    Simulates geographically separated exchanges each observing
    different time windows.

    Returns
    -------
    list of (X, y, node_name) tuples.
    """
    import pandas as pd

    timesteps = sorted(df["timestep"].unique())
    n_ts = len(timesteps)
    chunk_size = max(1, n_ts // n_nodes)

    partitions = []
    for i in range(n_nodes):
        start = i * chunk_size
        end = n_ts if i == n_nodes - 1 else (i + 1) * chunk_size
        node_ts = timesteps[start:end]
        node_df = df[df["timestep"].isin(node_ts)]

        X = node_df[feature_cols].fillna(0).values.astype(np.float32)
        y = node_df["fraud_label"].values.astype(np.float32)
        name = f"Exchange-{chr(65 + i)}"  # A, B, C, ...

        partitions.append((X, y, name))
        logger.info("  %s: %d transactions, fraud_rate=%.2f%%, timesteps %d-%d",
                     name, len(y), y.mean() * 100,
                     min(node_ts), max(node_ts))

    return partitions


# ==============================================================================
# 5. Federated Training Loop
# ==============================================================================
def run_federated_training(
    train_df: 'pd.DataFrame',
    test_df: 'pd.DataFrame',
    feature_cols: List[str],
    n_nodes: int = 4,
    n_rounds: int = 10,
    local_epochs: int = 3,
    hidden_dim: int = 128,
    lr: float = 1e-3,
    dp_noise_scale: float = 0.0,
    device: str = "cpu",
) -> Tuple['FederatedServer', List[Dict], List['FederatedNode']]:
    """
    Run a complete Federated Averaging training loop.

    Parameters
    ----------
    train_df : Training DataFrame with 'timestep', 'fraud_label', and feature_cols.
    test_df : Test DataFrame (used only for global model evaluation).
    feature_cols : List of feature column names.
    n_nodes : Number of simulated federated nodes.
    n_rounds : Number of communication rounds.
    local_epochs : Epochs per node per round.
    dp_noise_scale : Gaussian noise scale for differential privacy (0=off).

    Returns
    -------
    server : FederatedServer with the final global model.
    round_history : List of per-round metric dicts.
    nodes : List of FederatedNode objects.
    """
    set_seed(SEED)
    ensure_directories()
    input_dim = len(feature_cols)

    logger.info("="*60)
    logger.info("Federated Learning: %d nodes, %d rounds, %d local epochs",
                n_nodes, n_rounds, local_epochs)
    if dp_noise_scale > 0:
        logger.info("  Differential Privacy enabled: noise_scale=%.4f", dp_noise_scale)
    logger.info("="*60)

    # 1. Partition training data across nodes
    logger.info("Partitioning training data across %d nodes (by timestep)...", n_nodes)
    partitions = partition_data_by_timestep(train_df, feature_cols, n_nodes)

    # 2. Create server & nodes
    server = FederatedServer(input_dim, hidden_dim, device)
    nodes = []
    for X, y, name in partitions:
        node = FederatedNode(
            node_id=name, X=X, y=y,
            input_dim=input_dim, hidden_dim=hidden_dim,
            lr=lr, device=device,
            dp_noise_scale=dp_noise_scale,
        )
        nodes.append(node)

    # 3. Test data (central evaluation only)
    X_test = test_df[feature_cols].fillna(0).values.astype(np.float32)
    y_test = test_df["fraud_label"].values.astype(np.float32)

    # 4. Federated rounds
    round_history = []
    for rnd in range(1, n_rounds + 1):
        # Broadcast global model
        global_params = server.get_global_parameters()

        node_results = []
        node_uploads = []
        for node in nodes:
            node.set_parameters(global_params)
            result = node.local_train(epochs=local_epochs)
            node_results.append(result)
            node_uploads.append((node.get_parameters(), node.n_samples))

        # Aggregate
        server.aggregate(node_uploads)

        # Evaluate global model on test set
        global_metrics = server.evaluate_global(X_test, y_test)

        # Per-node summary
        avg_node_loss = np.mean([r["loss"] for r in node_results])
        avg_node_acc = np.mean([r["accuracy"] for r in node_results])

        round_info = {
            "round": rnd,
            "global_accuracy": global_metrics["accuracy"],
            "global_precision": global_metrics["precision"],
            "global_recall": global_metrics["recall"],
            "global_f1": global_metrics["f1"],
            "avg_node_loss": avg_node_loss,
            "avg_node_accuracy": avg_node_acc,
            "node_details": node_results,
            **{k: v for k, v in global_metrics.items() if k != "probs"},
        }
        round_history.append(round_info)

        if rnd % max(1, n_rounds // 5) == 0 or rnd == 1:
            logger.info(
                "  Round %d/%d — Global: acc=%.3f  prec=%.3f  recall=%.3f  "
                "F1=%.3f  |  Avg node loss=%.4f",
                rnd, n_rounds,
                global_metrics["accuracy"], global_metrics["precision"],
                global_metrics["recall"], global_metrics["f1"],
                avg_node_loss,
            )

    server.save()
    logger.info("Federated training complete. Final global F1=%.3f", round_history[-1]["global_f1"])

    return server, round_history, nodes


# ==============================================================================
# 6. Visualization
# ==============================================================================
def visualize_federated_training(
    round_history: List[Dict],
    n_nodes: int = 4,
    output_path: Optional[str] = None,
) -> None:
    """Plot federated learning training curves."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = output_path or str(RESULTS_DIR / "federated_learning_curves.png")

    rounds = [h["round"] for h in round_history]
    global_acc = [h["global_accuracy"] for h in round_history]
    global_f1 = [h["global_f1"] for h in round_history]
    global_prec = [h["global_precision"] for h in round_history]
    global_recall = [h["global_recall"] for h in round_history]
    avg_loss = [h["avg_node_loss"] for h in round_history]
    avg_node_acc = [h["avg_node_accuracy"] for h in round_history]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 1. Global accuracy over rounds
    axes[0, 0].plot(rounds, global_acc, marker='o', color='#1f77b4', linewidth=2, markersize=5)
    axes[0, 0].set_title('Global Model Accuracy', fontsize=12)
    axes[0, 0].set_xlabel('Communication Round')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_ylim(0, 1.05)
    axes[0, 0].grid(alpha=0.3)

    # 2. Global F1 over rounds
    axes[0, 1].plot(rounds, global_f1, marker='s', color='#9467bd', linewidth=2, markersize=5)
    axes[0, 1].set_title('Global F1 Score (Illicit)', fontsize=12)
    axes[0, 1].set_xlabel('Communication Round')
    axes[0, 1].set_ylabel('F1 Score')
    axes[0, 1].set_ylim(0, 1.05)
    axes[0, 1].grid(alpha=0.3)

    # 3. Precision & Recall
    axes[0, 2].plot(rounds, global_prec, marker='^', color='#ff7f0e', linewidth=2, label='Precision')
    axes[0, 2].plot(rounds, global_recall, marker='v', color='#d62728', linewidth=2, label='Recall')
    axes[0, 2].set_title('Global Precision & Recall', fontsize=12)
    axes[0, 2].set_xlabel('Communication Round')
    axes[0, 2].set_ylabel('Score')
    axes[0, 2].legend()
    axes[0, 2].set_ylim(0, 1.05)
    axes[0, 2].grid(alpha=0.3)

    # 4. Average node loss
    axes[1, 0].plot(rounds, avg_loss, marker='d', color='#e377c2', linewidth=2, markersize=5)
    axes[1, 0].set_title('Average Node Training Loss', fontsize=12)
    axes[1, 0].set_xlabel('Communication Round')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].grid(alpha=0.3)

    # 5. Node accuracy convergence
    axes[1, 1].plot(rounds, avg_node_acc, marker='o', color='#2ca02c', linewidth=2,
                    label='Avg Node Acc', markersize=5)
    axes[1, 1].plot(rounds, global_acc, marker='s', color='#1f77b4', linewidth=2,
                    label='Global Acc', markersize=5, linestyle='--')
    axes[1, 1].set_title('Node vs Global Accuracy', fontsize=12)
    axes[1, 1].set_xlabel('Communication Round')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].legend()
    axes[1, 1].set_ylim(0, 1.05)
    axes[1, 1].grid(alpha=0.3)

    # 6. Final round confusion matrix
    last = round_history[-1]
    cm = np.array([[last["tn"], last["fp"]], [last["fn"], last["tp"]]])
    im = axes[1, 2].imshow(cm, cmap='Purples', interpolation='nearest')
    axes[1, 2].set_title('Final Global Confusion Matrix', fontsize=12)
    axes[1, 2].set_xlabel('Predicted')
    axes[1, 2].set_ylabel('Actual')
    axes[1, 2].set_xticks([0, 1])
    axes[1, 2].set_xticklabels(['Licit', 'Illicit'])
    axes[1, 2].set_yticks([0, 1])
    axes[1, 2].set_yticklabels(['Licit', 'Illicit'])
    for a in range(2):
        for b in range(2):
            axes[1, 2].text(b, a, f'{cm[a, b]:,}', ha='center', va='center',
                            fontsize=14, color='white' if cm[a, b] > cm.max() / 2 else 'black')

    plt.suptitle(f'Federated Learning — {n_nodes} Nodes, FedAvg (Novelty #10)',
                 fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Federated learning visualization saved → %s", output_path)
