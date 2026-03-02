"""
Transaction Graph Builder for Crypto Fraud Detection.

Constructs a weighted, directed NetworkX graph from transaction data.
Computes graph analytics including connected components, cycle detection,
strongly connected subgraphs, and hub identification.

Usage
-----
    python -m src.graph_builder
"""

import sys
import pickle
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional

import numpy as np
import pandas as pd
import networkx as nx

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

logger = setup_logging("graph_builder")
SEED = 42


# ==============================================================================
# Graph Construction
# ==============================================================================
def build_transaction_graph(
    df: pd.DataFrame,
    fraud_wallets: Optional[Set[str]] = None,
) -> nx.DiGraph:
    """
    Build a directed, weighted transaction graph.

    Nodes
    -----
    Each unique wallet becomes a node with attributes:
      - is_fraud : bool
      - total_sent / total_received : float
      - tx_count : int

    Edges
    -----
    Each (sender, receiver) pair aggregated as one edge with:
      - amount        : total ETH transferred
      - frequency     : number of transactions
      - fraud_probability : fraction of fraud txs on this edge
      - timestamp_first / timestamp_last : temporal range
    """
    set_seed(SEED)
    logger.info("Building transaction graph from %d transactions...", len(df))

    if fraud_wallets is None:
        fraud_wallets = set(
            df.loc[df["fraud_label"] == 1, "sender_wallet"].unique()
        )

    G = nx.DiGraph()

    # ---- Aggregate edges ----
    edge_agg = (
        df.groupby(["sender_wallet", "receiver_wallet"])
        .agg(
            total_amount=("amount", "sum"),
            frequency=("amount", "size"),
            fraud_count=("fraud_label", "sum"),
            ts_first=("timestamp", "min"),
            ts_last=("timestamp", "max"),
        )
        .reset_index()
    )
    edge_agg["fraud_probability"] = edge_agg["fraud_count"] / edge_agg["frequency"]

    for _, row in edge_agg.iterrows():
        G.add_edge(
            row["sender_wallet"],
            row["receiver_wallet"],
            amount=float(row["total_amount"]),
            frequency=int(row["frequency"]),
            fraud_probability=float(row["fraud_probability"]),
            timestamp_first=int(row["ts_first"]),
            timestamp_last=int(row["ts_last"]),
        )

    # ---- Node attributes ----
    sent_agg = df.groupby("sender_wallet")["amount"].agg(["sum", "count"]).rename(
        columns={"sum": "total_sent", "count": "tx_sent_count"}
    )
    recv_agg = df.groupby("receiver_wallet")["amount"].agg(["sum", "count"]).rename(
        columns={"sum": "total_received", "count": "tx_recv_count"}
    )

    for node in G.nodes():
        G.nodes[node]["is_fraud"] = node in fraud_wallets
        G.nodes[node]["total_sent"] = float(sent_agg.loc[node, "total_sent"]) if node in sent_agg.index else 0.0
        G.nodes[node]["total_received"] = float(recv_agg.loc[node, "total_received"]) if node in recv_agg.index else 0.0
        G.nodes[node]["tx_count"] = int(
            (sent_agg.loc[node, "tx_sent_count"] if node in sent_agg.index else 0)
            + (recv_agg.loc[node, "tx_recv_count"] if node in recv_agg.index else 0)
        )

    logger.info(
        "Graph built: %d nodes, %d edges, %d fraud nodes.",
        G.number_of_nodes(), G.number_of_edges(),
        sum(1 for n in G.nodes() if G.nodes[n].get("is_fraud", False)),
    )
    return G


# ==============================================================================
# Graph Analytics
# ==============================================================================
def compute_graph_analytics(G: nx.DiGraph) -> Dict:
    """
    Run graph-level analytics and return a summary dict.

    Analytics
    ---------
    * Connected components (weakly connected in digraph)
    * Strongly connected components (coordinated activity)
    * Cycle detection (circular transactions — money-laundering indicator)
    * Hub identification (top-k by in-degree + out-degree)
    """
    logger.info("Computing graph analytics...")
    analytics: Dict = {}

    # --- Weakly connected components ---
    wcc = list(nx.weakly_connected_components(G))
    analytics["num_weakly_connected_components"] = len(wcc)
    analytics["largest_wcc_size"] = max(len(c) for c in wcc) if wcc else 0

    # Fraud clusters: components containing ≥ 1 fraud node
    fraud_clusters = []
    for comp in wcc:
        fraud_in_comp = [n for n in comp if G.nodes[n].get("is_fraud", False)]
        if fraud_in_comp:
            fraud_clusters.append({
                "size": len(comp),
                "fraud_count": len(fraud_in_comp),
                "nodes": list(comp)[:20],  # keep first 20 for display
            })
    analytics["fraud_clusters"] = fraud_clusters[:20]
    analytics["num_fraud_clusters"] = len(fraud_clusters)

    # --- Strongly connected components ---
    scc = list(nx.strongly_connected_components(G))
    scc_nontrivial = [c for c in scc if len(c) > 1]
    analytics["num_strongly_connected_components"] = len(scc_nontrivial)
    analytics["largest_scc_size"] = max(len(c) for c in scc_nontrivial) if scc_nontrivial else 0

    # --- Cycle detection (limited for performance on dense graphs) ---
    logger.info("Detecting cycles (money-laundering indicators)...")
    cycles: List[List[str]] = []
    try:
        import time as _time
        _start = _time.time()
        for cycle in nx.simple_cycles(G):
            if len(cycle) <= 6:
                cycles.append(cycle)
            if len(cycles) >= 100 or (_time.time() - _start) > 30:
                break
    except Exception:
        pass
    analytics["num_cycles_detected"] = len(cycles)
    analytics["sample_cycles"] = cycles[:10]

    # --- Hub wallets (top-20 by total degree) ---
    degree_dict = dict(G.degree())
    top_hubs = sorted(degree_dict.items(), key=lambda x: x[1], reverse=True)[:20]
    analytics["hub_wallets"] = [
        {"wallet": w, "degree": d, "is_fraud": G.nodes[w].get("is_fraud", False)}
        for w, d in top_hubs
    ]

    logger.info(
        "Analytics complete — WCC: %d, SCC: %d, Cycles: %d, Hubs identified: %d",
        analytics["num_weakly_connected_components"],
        analytics["num_strongly_connected_components"],
        analytics["num_cycles_detected"],
        len(analytics["hub_wallets"]),
    )
    return analytics


# ==============================================================================
# Save / Load
# ==============================================================================
def save_graph(G: nx.DiGraph, path: Optional[Path] = None) -> Path:
    """Pickle the graph to disk."""
    ensure_directories()
    if path is None:
        path = DATA_PROCESSED_DIR / "graph_data.pkl"
    path = Path(path)
    with open(path, "wb") as f:
        pickle.dump(G, f)
    logger.info("Graph saved → %s", path)
    return path


def load_graph(path: Optional[Path] = None) -> nx.DiGraph:
    """Load a pickled graph from disk."""
    if path is None:
        path = DATA_PROCESSED_DIR / "graph_data.pkl"
    with open(path, "rb") as f:
        G = pickle.load(f)
    logger.info("Graph loaded ← %s (%d nodes, %d edges)", path, G.number_of_nodes(), G.number_of_edges())
    return G


# ==============================================================================
# Visualization Helper
# ==============================================================================
def visualize_fraud_network(
    G: nx.DiGraph,
    fraud_wallets: Optional[Set[str]] = None,
    output_path: str = "results/network_graph_visualization.png",
    max_nodes: int = 500,
) -> None:
    """
    Visualize the transaction network with fraud clusters highlighted.

    Parameters
    ----------
    G : nx.DiGraph
    fraud_wallets : set of wallet IDs marked as fraud
    output_path : str
    max_nodes : int
        Subsample graph for readability if too large.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if fraud_wallets is None:
        fraud_wallets = {n for n in G.nodes() if G.nodes[n].get("is_fraud", False)}

    # Subsample if needed
    if G.number_of_nodes() > max_nodes:
        # Keep all fraud nodes + random normal nodes
        fraud_nodes = [n for n in G.nodes() if n in fraud_wallets]
        normal_nodes = [n for n in G.nodes() if n not in fraud_wallets]
        np.random.seed(SEED)
        keep_normal = list(np.random.choice(
            normal_nodes, size=min(max_nodes - len(fraud_nodes), len(normal_nodes)), replace=False
        ))
        subgraph_nodes = set(fraud_nodes + keep_normal)
        G_vis = G.subgraph(subgraph_nodes).copy()
    else:
        G_vis = G

    logger.info("Visualizing graph with %d nodes...", G_vis.number_of_nodes())

    plt.figure(figsize=(16, 12))
    pos = nx.spring_layout(G_vis, k=0.5, iterations=50, seed=SEED)

    node_colors = [
        "red" if n in fraud_wallets else "lightblue" for n in G_vis.nodes()
    ]
    node_sizes = [
        150 if n in fraud_wallets else 60 for n in G_vis.nodes()
    ]

    nx.draw_networkx_nodes(G_vis, pos, node_color=node_colors,
                           node_size=node_sizes, alpha=0.7)
    nx.draw_networkx_edges(G_vis, pos, alpha=0.15, width=0.4,
                           edge_color="gray", arrows=True, arrowsize=5)

    plt.title(
        "Transaction Network: Fraud Detection\n"
        f"(Red = Fraudulent [{len(fraud_wallets)}], Blue = Normal "
        f"[{G_vis.number_of_nodes() - len([n for n in G_vis.nodes() if n in fraud_wallets])}])",
        fontsize=16,
    )
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Network visualization saved → %s", output_path)


# ==============================================================================
# CLI entry point
# ==============================================================================
if __name__ == "__main__":
    ensure_directories()
    csv_path = DATA_RAW_DIR / "synthetic_dataset.csv"
    if not csv_path.exists():
        logger.error("Raw dataset not found at %s. Run data_generator first.", csv_path)
        sys.exit(1)

    df = pd.read_csv(csv_path)
    G = build_transaction_graph(df)
    analytics = compute_graph_analytics(G)
    save_graph(G)

    # Quick summary
    print("\n--- Graph Analytics Summary ---")
    for key in ["num_weakly_connected_components", "largest_wcc_size",
                "num_strongly_connected_components", "largest_scc_size",
                "num_cycles_detected", "num_fraud_clusters"]:
        print(f"  {key}: {analytics[key]}")

    print("\nTop-5 Hub Wallets:")
    for hub in analytics["hub_wallets"][:5]:
        print(f"  {hub['wallet'][:16]}...  degree={hub['degree']}  fraud={hub['is_fraud']}")


# ==============================================================================
# Elliptic Dataset: Graph Construction from Edge List
# ==============================================================================
def build_elliptic_graph(
    df: pd.DataFrame,
    edges_df: pd.DataFrame,
) -> nx.DiGraph:
    """
    Build a directed graph from the Elliptic dataset where:
      - Nodes = transactions (txId)
      - Edges = payment flows (txId1 → txId2)
      - Node attributes include fraud_label and timestep

    Parameters
    ----------
    df : pd.DataFrame
        Labelled transactions with columns: txId, timestep, fraud_label, ...
    edges_df : pd.DataFrame
        Edge list with columns: txId1, txId2.

    Returns
    -------
    nx.DiGraph
    """
    logger.info("Building Elliptic transaction graph...")

    G = nx.DiGraph()

    # Add nodes with attributes
    fraud_set = set(df.loc[df["fraud_label"] == 1, "txId"].values)
    tx_to_timestep = dict(zip(df["txId"], df["timestep"]))

    for txId in df["txId"].values:
        G.add_node(
            txId,
            is_fraud=txId in fraud_set,
            timestep=int(tx_to_timestep.get(txId, 0)),
        )

    # Add edges (only between known nodes)
    valid_nodes = set(G.nodes())
    added = 0
    for _, row in edges_df.iterrows():
        if row["txId1"] in valid_nodes and row["txId2"] in valid_nodes:
            G.add_edge(row["txId1"], row["txId2"])
            added += 1

    logger.info(
        "Elliptic graph built: %d nodes, %d edges, %d illicit nodes",
        G.number_of_nodes(), G.number_of_edges(),
        sum(1 for n in G.nodes() if G.nodes[n].get("is_fraud", False)),
    )
    return G


def visualize_elliptic_network(
    G: nx.DiGraph,
    output_path: str = "results/elliptic_network_visualization.png",
    max_nodes: int = 800,
) -> None:
    """
    Visualize the Elliptic transaction network with fraud nodes highlighted.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fraud_nodes = {n for n in G.nodes() if G.nodes[n].get("is_fraud", False)}

    # Subsample
    if G.number_of_nodes() > max_nodes:
        fraud_list = [n for n in G.nodes() if n in fraud_nodes]
        normal_list = [n for n in G.nodes() if n not in fraud_nodes]
        np.random.seed(SEED)
        # If fraud nodes alone exceed max_nodes, subsample them too
        if len(fraud_list) > max_nodes // 2:
            keep_fraud = list(np.random.choice(fraud_list, size=max_nodes // 2, replace=False))
        else:
            keep_fraud = fraud_list
        remaining = max(max_nodes - len(keep_fraud), 0)
        if remaining > 0 and len(normal_list) > 0:
            keep_normal = list(np.random.choice(
                normal_list,
                size=min(remaining, len(normal_list)),
                replace=False,
            ))
        else:
            keep_normal = []
        G_vis = G.subgraph(set(keep_fraud + keep_normal)).copy()
    else:
        G_vis = G

    logger.info("Visualizing Elliptic graph with %d nodes...", G_vis.number_of_nodes())

    plt.figure(figsize=(16, 12))
    pos = nx.spring_layout(G_vis, k=0.8, iterations=50, seed=SEED)

    node_colors = [
        "#d62728" if n in fraud_nodes else "#1f77b4" for n in G_vis.nodes()
    ]
    node_sizes = [
        120 if n in fraud_nodes else 40 for n in G_vis.nodes()
    ]

    nx.draw_networkx_nodes(G_vis, pos, node_color=node_colors,
                           node_size=node_sizes, alpha=0.7)
    nx.draw_networkx_edges(G_vis, pos, alpha=0.1, width=0.3,
                           edge_color="gray", arrows=True, arrowsize=4)

    n_fraud_vis = sum(1 for n in G_vis.nodes() if n in fraud_nodes)
    n_licit_vis = G_vis.number_of_nodes() - n_fraud_vis
    plt.title(
        f"Elliptic Bitcoin Transaction Network\n"
        f"(Red = Illicit [{n_fraud_vis}], Blue = Licit [{n_licit_vis}])",
        fontsize=16,
    )
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Elliptic network visualization saved → %s", output_path)
