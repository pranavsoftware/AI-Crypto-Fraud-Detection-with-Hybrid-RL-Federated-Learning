"""
Synthetic Dataset Generator for Crypto Fraud Detection.

Generates 50,000 realistic on-chain transactions across 2,000 wallets
(400 fraudulent) with embedded fraud patterns including pump-and-dump,
rug pulls, wash trading, and abnormal transaction sequences.

Usage
-----
    python -m src.data_generator          # from project root
    python src/data_generator.py          # direct execution
"""

import sys
import time
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict

# Handle both direct execution and module import
try:
    from src.utils import (
        setup_logging, set_seed, ensure_directories,
        DATA_RAW_DIR, PROJECT_ROOT,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.utils import (
        setup_logging, set_seed, ensure_directories,
        DATA_RAW_DIR, PROJECT_ROOT,
    )

logger = setup_logging("data_generator")

# ==============================================================================
# Configuration
# ==============================================================================
NUM_TRANSACTIONS: int = 50_000
NUM_WALLETS: int = 2_000
NUM_FRAUD_WALLETS: int = 400  # 20 %
SEED: int = 42

TOKEN_TYPES: List[str] = [
    "0xdAC17F958D2ee523a2206206994597C13D831ec7",  # USDT
    "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",  # USDC
    "0x6B175474E89094C44Da98b954EedeAC495271d0F",  # DAI
    "0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599",  # WBTC
    "0x514910771AF9Ca656af840dff83E8264EcF986CA",  # LINK
    "0x1f9840a85d5aF5bf1D1762F925BDADdC4201F984",  # UNI
    "0x7Fc66500c84A76Ad7e9c93437bFc5Ac33E2DDaE9",  # AAVE
    "0xNewToken_SuspiciousA",                         # very-new token
    "0xNewToken_SuspiciousB",                         # very-new token
    "0xNewToken_SuspiciousC",                         # very-new token
]


# ==============================================================================
# Wallet Generation
# ==============================================================================
def _generate_wallet_ids(n: int) -> List[str]:
    """Generate *n* pseudo-Ethereum wallet addresses."""
    wallets = []
    for i in range(n):
        raw = hashlib.sha256(f"wallet_{i}_{SEED}".encode()).hexdigest()[:40]
        wallets.append(f"0x{raw}")
    return wallets


def _partition_wallets(
    wallets: List[str], n_fraud: int
) -> Tuple[List[str], List[str]]:
    """Split wallets into fraud and normal sets."""
    fraud_wallets = wallets[:n_fraud]
    normal_wallets = wallets[n_fraud:]
    return fraud_wallets, normal_wallets


# ==============================================================================
# Fraud-Pattern Generators
# ==============================================================================
def _generate_pump_and_dump(
    sender: str,
    receivers: List[str],
    rng: np.random.Generator,
    base_ts: int,
    n_txs: int = 20,
) -> List[Dict]:
    """
    Pump & Dump: coordinated high-volume buys followed by a sell-off.
    Phase-1 (pump): many small buys in a short window.
    Phase-2 (dump): one or two large sells.
    """
    txs = []
    pump_count = int(n_txs * 0.75)
    dump_count = n_txs - pump_count
    ts = base_ts

    # Pump phase — high frequency, moderate amounts
    for _ in range(pump_count):
        ts += int(rng.integers(30, 600))  # 0.5-10 min apart
        txs.append({
            "sender_wallet": sender,
            "receiver_wallet": rng.choice(receivers),
            "amount": float(rng.uniform(5, 50)),
            "timestamp": ts,
            "gas_fee": float(rng.uniform(80, 300)),
            "token_type": rng.choice(TOKEN_TYPES[-3:]),  # new tokens
            "token_age_days": int(rng.integers(0, 10)),
            "volume_spike_indicator": 1,
            "liquidity_pool_change": float(rng.uniform(5, 30)),
            "is_contract": int(rng.random() < 0.3),
            "fraud_label": 1,
            "_pattern": "pump_and_dump",
        })

    # Dump phase — few very large sells
    for _ in range(dump_count):
        ts += int(rng.integers(10, 120))
        txs.append({
            "sender_wallet": sender,
            "receiver_wallet": rng.choice(receivers),
            "amount": float(rng.uniform(200, 1000)),
            "timestamp": ts,
            "gas_fee": float(rng.uniform(150, 500)),
            "token_type": rng.choice(TOKEN_TYPES[-3:]),
            "token_age_days": int(rng.integers(0, 10)),
            "volume_spike_indicator": 1,
            "liquidity_pool_change": float(rng.uniform(-60, -20)),
            "is_contract": int(rng.random() < 0.4),
            "fraud_label": 1,
            "_pattern": "pump_and_dump",
        })
    return txs


def _generate_rug_pull(
    sender: str,
    receivers: List[str],
    rng: np.random.Generator,
    base_ts: int,
    n_txs: int = 15,
) -> List[Dict]:
    """
    Rug Pull: liquidity injection followed by abrupt drain.
    """
    txs = []
    ts = base_ts

    inject_count = int(n_txs * 0.6)
    drain_count = n_txs - inject_count

    # Injection phase
    for _ in range(inject_count):
        ts += int(rng.integers(600, 3600))
        txs.append({
            "sender_wallet": sender,
            "receiver_wallet": rng.choice(receivers),
            "amount": float(rng.uniform(10, 100)),
            "timestamp": ts,
            "gas_fee": float(rng.uniform(50, 200)),
            "token_type": rng.choice(TOKEN_TYPES[-3:]),
            "token_age_days": int(rng.integers(1, 30)),
            "volume_spike_indicator": 0,
            "liquidity_pool_change": float(rng.uniform(10, 50)),
            "is_contract": 1,
            "fraud_label": 1,
            "_pattern": "rug_pull",
        })

    # Drain phase — very fast, high amounts, negative liquidity
    for _ in range(drain_count):
        ts += int(rng.integers(10, 300))
        txs.append({
            "sender_wallet": sender,
            "receiver_wallet": rng.choice(receivers),
            "amount": float(rng.uniform(200, 1000)),
            "timestamp": ts,
            "gas_fee": float(rng.uniform(200, 600)),
            "token_type": rng.choice(TOKEN_TYPES[-3:]),
            "token_age_days": int(rng.integers(1, 30)),
            "volume_spike_indicator": 1,
            "liquidity_pool_change": float(rng.uniform(-90, -40)),
            "is_contract": 1,
            "fraud_label": 1,
            "_pattern": "rug_pull",
        })
    return txs


def _generate_wash_trading(
    wallets: List[str],
    rng: np.random.Generator,
    base_ts: int,
    n_txs: int = 20,
) -> List[Dict]:
    """
    Wash Trading: the same group of wallets trades back and forth
    with nearly identical amounts.
    """
    txs = []
    ts = base_ts
    base_amount = float(rng.uniform(10, 200))

    for _ in range(n_txs):
        ts += int(rng.integers(60, 1800))
        s, r = rng.choice(wallets, size=2, replace=False)
        txs.append({
            "sender_wallet": s,
            "receiver_wallet": r,
            "amount": base_amount + float(rng.normal(0, 0.5)),
            "timestamp": ts,
            "gas_fee": float(rng.uniform(20, 80)),
            "token_type": rng.choice(TOKEN_TYPES[:5]),
            "token_age_days": int(rng.integers(30, 365)),
            "volume_spike_indicator": 0,
            "liquidity_pool_change": float(rng.uniform(-2, 2)),
            "is_contract": 0,
            "fraud_label": 1,
            "_pattern": "wash_trading",
        })
    return txs


def _generate_abnormal_sequence(
    sender: str,
    receivers: List[str],
    rng: np.random.Generator,
    base_ts: int,
    n_txs: int = 12,
) -> List[Dict]:
    """
    Abnormal Transaction Sequences: irregular timing & amount spikes
    indicative of bot-driven or coordinated manipulation.
    """
    txs = []
    ts = base_ts

    for i in range(n_txs):
        # Alternating very short and very long gaps
        gap = int(rng.integers(5, 30)) if i % 2 == 0 else int(rng.integers(7200, 86400))
        ts += gap
        amt = float(rng.uniform(0.01, 5)) if i % 3 != 0 else float(rng.uniform(500, 1000))
        txs.append({
            "sender_wallet": sender,
            "receiver_wallet": rng.choice(receivers),
            "amount": amt,
            "timestamp": ts,
            "gas_fee": float(rng.uniform(30, 250)),
            "token_type": rng.choice(TOKEN_TYPES),
            "token_age_days": int(rng.integers(0, 180)),
            "volume_spike_indicator": int(amt > 400),
            "liquidity_pool_change": float(rng.uniform(-10, 10)),
            "is_contract": int(rng.random() < 0.5),
            "fraud_label": 1,
            "_pattern": "abnormal_sequence",
        })
    return txs


# ==============================================================================
# Normal Transaction Generator
# ==============================================================================
def _generate_normal_transactions(
    normal_wallets: List[str],
    all_wallets: List[str],
    rng: np.random.Generator,
    n_txs: int = 35_000,
    year_start_ts: int = 0,
) -> List[Dict]:
    """Generate realistic normal (non-fraudulent) transactions."""
    txs = []
    # Pre-convert to numpy arrays for fast random choice
    normal_arr = np.array(normal_wallets)
    all_arr = np.array(all_wallets)
    # Pre-generate random indices for speed
    sender_idx = rng.integers(0, len(normal_arr), size=n_txs)
    receiver_idx = rng.integers(0, len(all_arr), size=n_txs)
    timestamps = year_start_ts + rng.integers(0, 365 * 86400, size=n_txs)
    amounts = rng.lognormal(mean=1.5, sigma=1.5, size=n_txs).astype(float)
    gas_fees = rng.lognormal(mean=3.0, sigma=0.8, size=n_txs).astype(float)
    token_ages = rng.integers(30, 1000, size=n_txs)
    vol_spikes = (rng.random(size=n_txs) < 0.03).astype(int)
    liq_changes = rng.normal(0, 3, size=n_txs)
    is_contracts = (rng.random(size=n_txs) < 0.15).astype(int)
    token_indices = rng.integers(0, 7, size=n_txs)

    for i in range(n_txs):
        sender = normal_arr[sender_idx[i]]
        receiver = all_arr[receiver_idx[i]]
        if receiver == sender:
            receiver = all_arr[(receiver_idx[i] + 1) % len(all_arr)]
        txs.append({
            "sender_wallet": sender,
            "receiver_wallet": receiver,
            "amount": float(amounts[i]),
            "timestamp": int(timestamps[i]),
            "gas_fee": float(gas_fees[i]),
            "token_type": TOKEN_TYPES[int(token_indices[i])],
            "token_age_days": int(token_ages[i]),
            "volume_spike_indicator": int(vol_spikes[i]),
            "liquidity_pool_change": float(liq_changes[i]),
            "is_contract": int(is_contracts[i]),
            "fraud_label": 0,
            "_pattern": "normal",
        })
    return txs


# ==============================================================================
# Main Generator
# ==============================================================================
def generate_dataset(
    num_transactions: int = NUM_TRANSACTIONS,
    num_wallets: int = NUM_WALLETS,
    num_fraud_wallets: int = NUM_FRAUD_WALLETS,
    seed: int = SEED,
    output_path: Path = DATA_RAW_DIR / "synthetic_dataset.csv",
) -> pd.DataFrame:
    """
    Generate the full synthetic dataset and save to CSV.

    Parameters
    ----------
    num_transactions : int
        Target total number of transactions (approx).
    num_wallets : int
        Total unique wallets.
    num_fraud_wallets : int
        Number of wallets marked as fraudulent.
    seed : int
        Random seed.
    output_path : Path
        Where to save the CSV.

    Returns
    -------
    pd.DataFrame
        The generated dataset.
    """
    set_seed(seed)
    rng = np.random.default_rng(seed)
    ensure_directories()

    logger.info("Generating %d wallets (%d fraudulent)...", num_wallets, num_fraud_wallets)
    wallets = _generate_wallet_ids(num_wallets)
    fraud_wallets, normal_wallets = _partition_wallets(wallets, num_fraud_wallets)

    year_start_ts = int(time.time()) - 365 * 86400  # ~1 year ago

    all_txs: List[Dict] = []

    # --- Fraud transactions (~30 % of total comes from fraud wallets) ---
    fraud_tx_target = int(num_transactions * 0.30)  # ~15,000 fraud txs
    fraud_txs_generated = 0

    logger.info("Generating fraud patterns...")

    wallets_arr = np.array(wallets)
    fraud_arr = np.array(fraud_wallets)

    # Pump & Dump — ~100 campaigns
    for i in range(100):
        sender = fraud_wallets[i % num_fraud_wallets]
        base_ts = year_start_ts + int(rng.integers(0, 300 * 86400))
        receivers = list(rng.choice(wallets_arr, size=10, replace=False))
        txs = _generate_pump_and_dump(sender, receivers, rng, base_ts, n_txs=25)
        all_txs.extend(txs)
        fraud_txs_generated += len(txs)

    # Rug Pulls — ~80 campaigns
    for i in range(80):
        sender = fraud_wallets[(100 + i) % num_fraud_wallets]
        base_ts = year_start_ts + int(rng.integers(0, 300 * 86400))
        receivers = list(rng.choice(wallets_arr, size=8, replace=False))
        txs = _generate_rug_pull(sender, receivers, rng, base_ts, n_txs=18)
        all_txs.extend(txs)
        fraud_txs_generated += len(txs)

    # Wash Trading — ~60 rings (3-5 wallets each)
    for i in range(60):
        ring_size = int(rng.integers(3, 6))
        ring = list(rng.choice(fraud_arr, size=ring_size, replace=False))
        base_ts = year_start_ts + int(rng.integers(0, 300 * 86400))
        txs = _generate_wash_trading(ring, rng, base_ts, n_txs=25)
        all_txs.extend(txs)
        fraud_txs_generated += len(txs)

    # Abnormal Sequences — ~100 wallets
    for i in range(100):
        sender = fraud_wallets[i % num_fraud_wallets]
        base_ts = year_start_ts + int(rng.integers(0, 300 * 86400))
        receivers = list(rng.choice(wallets_arr, size=5, replace=False))
        txs = _generate_abnormal_sequence(sender, receivers, rng, base_ts, n_txs=15)
        all_txs.extend(txs)
        fraud_txs_generated += len(txs)

    logger.info("Fraud transactions generated: %d", fraud_txs_generated)

    # --- Normal transactions ---
    normal_target = num_transactions - fraud_txs_generated
    logger.info("Generating %d normal transactions...", normal_target)
    normal_txs = _generate_normal_transactions(
        normal_wallets, wallets, rng, n_txs=normal_target, year_start_ts=year_start_ts
    )
    all_txs.extend(normal_txs)

    # --- Build DataFrame ---
    df = pd.DataFrame(all_txs)

    # Clip amounts to realistic range
    df["amount"] = df["amount"].clip(0.01, 1000.0)

    # Add wallet_id (sender as primary wallet reference)
    df["wallet_id"] = df["sender_wallet"]

    # Add transaction_frequency per sender (txs per hour proxy)
    sender_counts = df.groupby("sender_wallet")["timestamp"].transform("count")
    time_spans = df.groupby("sender_wallet")["timestamp"].transform(
        lambda x: max((x.max() - x.min()) / 3600, 1)
    )
    df["transaction_frequency"] = sender_counts / time_spans

    # Add average_amount per sender
    df["average_amount"] = df.groupby("sender_wallet")["amount"].transform("mean")

    # Shuffle and reset index
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Final column ordering
    columns_order = [
        "wallet_id", "sender_wallet", "receiver_wallet", "amount",
        "timestamp", "gas_fee", "token_type", "token_age_days",
        "transaction_frequency", "average_amount", "volume_spike_indicator",
        "liquidity_pool_change", "is_contract", "fraud_label",
    ]
    df = df[[c for c in columns_order if c in df.columns]]

    # Ensure exactly num_transactions rows
    if len(df) > num_transactions:
        df = df.head(num_transactions)
    elif len(df) < num_transactions:
        # Duplicate some normal rows to fill
        shortfall = num_transactions - len(df)
        filler = df[df["fraud_label"] == 0].sample(n=shortfall, replace=True, random_state=seed)
        df = pd.concat([df, filler], ignore_index=True)

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    fraud_rate = df["fraud_label"].mean() * 100
    logger.info(
        "Dataset saved → %s  |  %d rows  |  %.1f%% fraud  |  %d unique wallets",
        output_path, len(df), fraud_rate, df["sender_wallet"].nunique(),
    )

    return df


# ==============================================================================
# CLI entry-point
# ==============================================================================
if __name__ == "__main__":
    df = generate_dataset()
    print(f"\nDataset shape: {df.shape}")
    print(f"Fraud distribution:\n{df['fraud_label'].value_counts()}")
    print(f"\nSample rows:\n{df.head()}")
