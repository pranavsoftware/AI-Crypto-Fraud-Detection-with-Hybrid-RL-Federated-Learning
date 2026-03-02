"""
Risk Scoring Engine for Crypto Fraud Detection.

Implements:
  - Novelty #1  : Hybrid Multi-Model Dynamic Risk Scoring
  - Novelty #6  : Adaptive Threshold with HMM Regime Detection
  - Novelty #7  : Hybrid Scoring Engine (integrated)
  - Novelty #8  : Explainable AI with SHAP

Usage
-----
    from src.risk_scoring_engine import HybridRiskScoringEngine, AdaptiveThresholdEngine
"""

import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from src.utils import (
        setup_logging, normalize_scores, FEATURE_DISPLAY_NAMES,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.utils import (
        setup_logging, normalize_scores, FEATURE_DISPLAY_NAMES,
    )

warnings.filterwarnings("ignore")
logger = setup_logging("risk_scoring_engine")


# ==============================================================================
# Novelty #6 — Adaptive Threshold with HMM Regime Detection
# ==============================================================================
class AdaptiveThresholdEngine:
    """
    Detect the current market regime using a Hidden Markov Model and
    dynamically adjust the fraud-detection threshold.

    States
    ------
    0 = Bull market  → lower threshold (more strict)
    1 = Bear market  → higher threshold
    2 = Volatile     → much higher threshold
    """

    def __init__(self, n_components: int = 3, random_state: int = 42):
        self.n_components = n_components
        self.random_state = random_state
        self._hmm = None
        self._fitted = False

        self.regime_adjustments: Dict[int, float] = {
            0: 0.80,   # Bull
            1: 1.20,   # Bear
            2: 1.50,   # Volatile
        }
        self.regime_names: Dict[int, str] = {
            0: "Bull",
            1: "Bear",
            2: "Volatile",
        }

    # ---- fit ----
    def fit(self, price_history: np.ndarray) -> "AdaptiveThresholdEngine":
        """
        Fit the HMM on log-returns of a price series.

        Parameters
        ----------
        price_history : 1-D array of prices (at least 50 observations).
        """
        try:
            from hmmlearn.hmm import GaussianHMM
        except ImportError:
            logger.warning("hmmlearn not installed — falling back to static thresholds.")
            self._fitted = False
            return self

        price_history = np.asarray(price_history, dtype=float).flatten()
        if len(price_history) < 10:
            logger.warning("Price history too short for HMM — using static thresholds.")
            self._fitted = False
            return self

        returns = np.diff(np.log(price_history + 1e-9)).reshape(-1, 1)
        self._hmm = GaussianHMM(
            n_components=self.n_components,
            covariance_type="full",
            n_iter=200,
            random_state=self.random_state,
        )
        self._hmm.fit(returns)
        self._fitted = True
        logger.info("HMM fitted on %d log-return observations.", len(returns))
        return self

    # ---- predict regime ----
    def detect_market_regime(self, price_history: np.ndarray) -> int:
        """
        Predict the current market regime from the last portion of the
        price series.

        Returns
        -------
        int : Regime index (0=Bull, 1=Bear, 2=Volatile).
        """
        if not self._fitted or self._hmm is None:
            return 0  # Default to Bull (strictest)

        price_history = np.asarray(price_history, dtype=float).flatten()
        returns = np.diff(np.log(price_history + 1e-9)).reshape(-1, 1)
        window = returns[-min(100, len(returns)):]
        regime = int(self._hmm.predict(window)[-1])
        logger.info("Detected market regime: %s (id=%d)", self.regime_names.get(regime, "?"), regime)
        return regime

    # ---- adaptive threshold ----
    def get_adaptive_threshold(self, base_threshold: float, regime: int) -> float:
        """
        Adjust the base threshold by the regime-specific multiplier.
        """
        multiplier = self.regime_adjustments.get(regime, 1.0)
        adapted = base_threshold * multiplier
        logger.info(
            "Threshold: base=%.3f → adapted=%.3f (regime=%s)",
            base_threshold, adapted, self.regime_names.get(regime, "?"),
        )
        return adapted

    # ---- convenience ----
    def generate_synthetic_price_history(self, n: int = 500, seed: int = 42) -> np.ndarray:
        """
        Generate a synthetic ETH-like price series for demonstration.
        """
        rng = np.random.RandomState(seed)
        log_returns = rng.normal(0.0005, 0.03, size=n)
        prices = 2000.0 * np.exp(np.cumsum(log_returns))
        return prices


# ==============================================================================
# Novelty #1 / #7 / #8 — Hybrid Risk Scoring Engine
# ==============================================================================
class HybridRiskScoringEngine:
    """
    Production-grade hybrid scoring engine combining:
      * GNN wallet-level scores
      * LSTM temporal scores
      * Random Forest / XGBoost ensemble scores
      * Isolation Forest anomaly scores

    Provides dynamic model weighting (Novelty #1 / #7) and
    SHAP-based explainability (Novelty #8).
    """

    def __init__(
        self,
        gnn_model=None,
        lstm_model=None,
        rf_model=None,
        xgb_model=None,
        iso_forest=None,
        adaptive_engine: Optional[AdaptiveThresholdEngine] = None,
    ):
        self.gnn_model = gnn_model
        self.lstm_model = lstm_model
        self.rf_model = rf_model
        self.xgb_model = xgb_model
        self.iso_forest = iso_forest
        self.adaptive_engine = adaptive_engine or AdaptiveThresholdEngine()

        self._shap_explainer = None

    # ---- dynamic weights (Novelty #1) ----
    def compute_dynamic_weights(
        self, blockchain_state: Optional[Dict[str, float]] = None
    ) -> np.ndarray:
        """
        Dynamically adjust model weights based on current blockchain
        network conditions.

        Parameters
        ----------
        blockchain_state : dict with keys 'avg_gas_price', 'recent_tx_count'.
            If None, balanced defaults are used.

        Returns
        -------
        np.ndarray of shape (5,) — weights for [RF, XGB, LSTM, GNN, ISO].
        """
        if blockchain_state is None:
            return np.array([0.25, 0.25, 0.20, 0.15, 0.15])

        gas = blockchain_state.get("avg_gas_price", 50)
        vol = blockchain_state.get("recent_tx_count", 5000)

        if gas > 100 and vol > 10000:
            # High volatility → trust anomaly detection more
            w = np.array([0.20, 0.15, 0.15, 0.15, 0.35])
        elif gas < 30:
            # Low activity → trust temporal patterns & graph more
            w = np.array([0.15, 0.15, 0.30, 0.25, 0.15])
        elif vol > 8000:
            # High volume → ensemble + GNN
            w = np.array([0.25, 0.25, 0.15, 0.25, 0.10])
        else:
            w = np.array([0.25, 0.25, 0.20, 0.15, 0.15])

        return w / w.sum()

    # ---- main scoring ----
    def compute_final_risk_score(
        self,
        rf_scores: np.ndarray,
        xgb_scores: np.ndarray,
        lstm_scores: np.ndarray,
        gnn_scores: np.ndarray,
        iso_scores: np.ndarray,
        blockchain_state: Optional[Dict[str, float]] = None,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Combine individual model scores into a single risk score
        using dynamic weighting.

        Returns
        -------
        final_risk : np.ndarray  (N,)
        component_scores : dict mapping model name → np.ndarray
        """
        weights = self.compute_dynamic_weights(blockchain_state)

        # Normalise each to [0, 1]
        rf_n = normalize_scores(rf_scores)
        xgb_n = normalize_scores(xgb_scores)
        lstm_n = normalize_scores(lstm_scores)
        gnn_n = normalize_scores(gnn_scores)
        iso_n = normalize_scores(iso_scores)

        final_risk = (
            weights[0] * rf_n
            + weights[1] * xgb_n
            + weights[2] * lstm_n
            + weights[3] * gnn_n
            + weights[4] * iso_n
        )

        component_scores = {
            "random_forest": rf_n,
            "xgboost": xgb_n,
            "lstm": lstm_n,
            "gnn": gnn_n,
            "isolation_forest": iso_n,
            "weights": weights,
        }

        return final_risk, component_scores

    # ---- explainability (Novelty #8) ----
    def explain_fraud_decision(
        self,
        wallet_features: np.ndarray,
        final_risk_score: float,
        feature_names: Optional[List[str]] = None,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Use SHAP to explain why a specific wallet was flagged.

        Returns
        -------
        dict with keys: risk_score, top_risk_features,
        human_readable_explanation, shap_values.
        """
        import shap

        if self._shap_explainer is None and self.xgb_model is not None:
            self._shap_explainer = shap.TreeExplainer(self.xgb_model)

        explanation: Dict[str, Any] = {
            "risk_score": float(final_risk_score),
            "top_risk_features": [],
            "human_readable_explanation": "",
            "shap_values": None,
        }

        if self._shap_explainer is None:
            explanation["human_readable_explanation"] = (
                "SHAP explainer not available (XGBoost model not loaded)."
            )
            return explanation

        if wallet_features.ndim == 1:
            wallet_features = wallet_features.reshape(1, -1)

        shap_values = self._shap_explainer.shap_values(wallet_features)
        explanation["shap_values"] = shap_values

        mean_abs = np.abs(shap_values).mean(axis=0)
        top_idx = np.argsort(mean_abs)[-top_k:][::-1]

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(mean_abs))]

        top_features = [
            (feature_names[i], float(mean_abs[i])) for i in top_idx
        ]
        explanation["top_risk_features"] = top_features

        # Human-readable
        reasons = []
        for fname, importance in top_features[:3]:
            display = FEATURE_DISPLAY_NAMES.get(fname, fname)
            reasons.append(display)

        risk_level = (
            "very high" if final_risk_score > 0.8
            else "high" if final_risk_score > 0.6
            else "moderate" if final_risk_score > 0.4
            else "low"
        )

        explanation["human_readable_explanation"] = (
            f"This wallet has a {risk_level} fraud risk (score: {final_risk_score:.3f}) "
            f"primarily due to: {', '.join(reasons)}."
        )

        return explanation

    # ---- classify with adaptive threshold ----
    def classify_with_adaptive_threshold(
        self,
        risk_scores: np.ndarray,
        price_history: Optional[np.ndarray] = None,
        base_threshold: float = 0.5,
    ) -> Tuple[np.ndarray, float, int]:
        """
        Apply adaptive threshold (Novelty #6) to risk scores.

        Returns
        -------
        labels : np.ndarray of {0, 1}
        threshold : float  (adapted)
        regime : int
        """
        if price_history is not None and len(price_history) > 10:
            if not self.adaptive_engine._fitted:
                self.adaptive_engine.fit(price_history)
            regime = self.adaptive_engine.detect_market_regime(price_history)
        else:
            regime = 0

        threshold = self.adaptive_engine.get_adaptive_threshold(base_threshold, regime)
        labels = (risk_scores > threshold).astype(int)

        logger.info(
            "Classification: regime=%s  threshold=%.3f  flagged=%d/%d",
            self.adaptive_engine.regime_names.get(regime, "?"),
            threshold, labels.sum(), len(labels),
        )
        return labels, threshold, regime


# ==============================================================================
# Standalone demo
# ==============================================================================
if __name__ == "__main__":
    logger.info("=== Risk Scoring Engine Demo ===")

    # Adaptive threshold demo
    ate = AdaptiveThresholdEngine()
    prices = ate.generate_synthetic_price_history(500)
    ate.fit(prices)
    regime = ate.detect_market_regime(prices)
    threshold = ate.get_adaptive_threshold(0.5, regime)
    print(f"Regime: {ate.regime_names[regime]} | Adapted threshold: {threshold:.3f}")

    # Hybrid scoring demo (with random scores)
    rng = np.random.RandomState(42)
    n = 100
    engine = HybridRiskScoringEngine(adaptive_engine=ate)
    final, components = engine.compute_final_risk_score(
        rf_scores=rng.rand(n),
        xgb_scores=rng.rand(n),
        lstm_scores=rng.rand(n),
        gnn_scores=rng.rand(n),
        iso_scores=rng.rand(n),
    )
    labels, thr, reg = engine.classify_with_adaptive_threshold(final, prices)
    print(f"Final risk scores: mean={final.mean():.3f}, std={final.std():.3f}")
    print(f"Flagged {labels.sum()}/{len(labels)} wallets as fraudulent.")
