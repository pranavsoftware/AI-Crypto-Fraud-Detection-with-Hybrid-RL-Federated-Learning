"""
Novelty #9 — RL-Based Policy Learning for Fraud Detection.

Implements a Deep Q-Network (DQN) agent that learns an optimal fraud
detection policy by interacting with a transaction environment.

Environment
-----------
- **State**: Per-transaction feature vector augmented with the 5 model
  scores (RF, XGB, ISO, LSTM, GNN) → dim = n_features + 5.
- **Actions**: Discrete actions controlling the detection decision:
    0 = Flag as licit (low risk)
    1 = Flag as illicit (high risk)
- **Reward**: +1 for correct classification, −2 for missed fraud (FN),
  −0.5 for false alarm (FP).  The asymmetric penalty encourages the
  agent to prioritise recall over precision.

Architecture
------------
- 2-layer MLP Q-Network with ReLU activations.
- Epsilon-greedy exploration with linear decay.
- Experience replay buffer (uniform sampling).
- Target network with periodic hard updates.

Usage
-----
    from src.rl_policy_learning import FraudDetectionEnvironment, DQNAgent, train_rl_agent
"""

import sys
import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import deque
import random

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

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
logger = setup_logging("rl_policy_learning")
SEED = 42


# ==============================================================================
# 1. Fraud Detection Environment
# ==============================================================================
class FraudDetectionEnvironment:
    """
    Gym-style environment that presents transactions one at a time.
    The agent decides whether each transaction is illicit (action=1)
    or licit (action=0).

    Parameters
    ----------
    features : np.ndarray, shape (N, D)
        Transaction feature matrix.
    model_scores : np.ndarray, shape (N, 5)
        Columns: [rf, xgb, iso, lstm, gnn] scores for each transaction.
    labels : np.ndarray, shape (N,)
        Ground-truth fraud labels (0 or 1).
    reward_tp : float   Reward for true positive.
    reward_tn : float   Reward for true negative.
    reward_fp : float   Penalty for false positive.
    reward_fn : float   Penalty for false negative (missed fraud).
    """

    def __init__(
        self,
        features: np.ndarray,
        model_scores: np.ndarray,
        labels: np.ndarray,
        reward_tp: float = 2.0,
        reward_tn: float = 1.0,
        reward_fp: float = -0.5,
        reward_fn: float = -2.0,
    ):
        self.features = np.asarray(features, dtype=np.float32)
        self.model_scores = np.asarray(model_scores, dtype=np.float32)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.n_samples = len(labels)
        self.state_dim = self.features.shape[1] + self.model_scores.shape[1]

        # Reward matrix: rows=true label, cols=action
        self._reward = np.array([
            [reward_tn, reward_fp],   # true label = 0 (licit)
            [reward_fn, reward_tp],   # true label = 1 (illicit)
        ], dtype=np.float32)

        self._idx = 0
        self._order = np.arange(self.n_samples)

    def reset(self) -> np.ndarray:
        """Shuffle transactions and return the first state."""
        np.random.shuffle(self._order)
        self._idx = 0
        return self._get_state()

    def _get_state(self) -> np.ndarray:
        i = self._order[self._idx]
        return np.concatenate([self.features[i], self.model_scores[i]])

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Take an action (0=licit, 1=illicit) and return
        (next_state, reward, done, info).
        """
        i = self._order[self._idx]
        true_label = self.labels[i]
        reward = float(self._reward[true_label, action])

        info = {
            "true_label": int(true_label),
            "action": int(action),
            "correct": int(action == true_label),
        }

        self._idx += 1
        done = self._idx >= self.n_samples
        next_state = self._get_state() if not done else np.zeros(self.state_dim, dtype=np.float32)

        return next_state, reward, done, info

    @property
    def n_actions(self) -> int:
        return 2


# ==============================================================================
# 2. Q-Network
# ==============================================================================
class QNetwork(nn.Module):
    """2-layer MLP Q-Network."""

    def __init__(self, state_dim: int, n_actions: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ==============================================================================
# 3. Experience Replay Buffer
# ==============================================================================
class ReplayBuffer:
    def __init__(self, capacity: int = 50_000):
        self.buffer: deque = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


# ==============================================================================
# 4. DQN Agent
# ==============================================================================
class DQNAgent:
    """
    Deep Q-Network agent for fraud detection policy learning.

    Parameters
    ----------
    state_dim : int
        Dimension of the state vector.
    n_actions : int
        Number of discrete actions (2: licit / illicit).
    hidden_dim : int
        Hidden layer width.
    lr : float
        Learning rate.
    gamma : float
        Discount factor.
    epsilon_start : float
        Initial exploration rate.
    epsilon_end : float
        Final exploration rate.
    epsilon_decay : int
        Number of steps over which epsilon decays linearly.
    target_update : int
        Frequency (in optimisation steps) of target network updates.
    buffer_size : int
        Replay buffer capacity.
    batch_size : int
        Mini-batch size for Q-learning updates.
    device : str
        'cpu' or 'cuda'.
    """

    def __init__(
        self,
        state_dim: int,
        n_actions: int = 2,
        hidden_dim: int = 128,
        lr: float = 1e-3,
        gamma: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: int = 5000,
        target_update: int = 500,
        buffer_size: int = 50_000,
        batch_size: int = 64,
        device: str = "cpu",
    ):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update = target_update
        self.batch_size = batch_size
        self.device = torch.device(device)

        # Networks
        self.q_net = QNetwork(state_dim, n_actions, hidden_dim).to(self.device)
        self.target_net = QNetwork(state_dim, n_actions, hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size)

        self._step_count = 0
        self._optim_count = 0

        # Metrics
        self.episode_rewards: List[float] = []
        self.episode_accuracies: List[float] = []
        self.episode_precisions: List[float] = []
        self.episode_recalls: List[float] = []
        self.losses: List[float] = []

    @property
    def epsilon(self) -> float:
        """Current epsilon (linear decay)."""
        frac = min(1.0, self._step_count / max(self.epsilon_decay, 1))
        return self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)

    def select_action(self, state: np.ndarray, greedy: bool = False) -> int:
        """Epsilon-greedy action selection."""
        if not greedy and random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)

        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state_t)
        return int(q_values.argmax(dim=1).item())

    def _optimise(self):
        """One step of DQN optimisation."""
        if len(self.buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)

        # Current Q values
        q_values = self.q_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Target Q values (Double DQN: use q_net to select action, target_net to evaluate)
        with torch.no_grad():
            next_actions = self.q_net(next_states_t).argmax(dim=1)
            next_q = self.target_net(next_states_t).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q = rewards_t + self.gamma * next_q * (1.0 - dones_t)

        loss = nn.MSELoss()(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()

        self.losses.append(loss.item())
        self._optim_count += 1

        # Hard target update
        if self._optim_count % self.target_update == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

    def train_episode(self, env: FraudDetectionEnvironment, max_steps: Optional[int] = None) -> Dict:
        """Run one full episode through the environment."""
        state = env.reset()
        total_reward = 0.0
        correct = 0
        tp, fp, fn, tn = 0, 0, 0, 0
        steps = 0
        limit = max_steps or env.n_samples

        while steps < limit:
            action = self.select_action(state)
            next_state, reward, done, info = env.step(action)

            self.buffer.push(state, action, reward, next_state, done)
            self._optimise()
            self._step_count += 1

            total_reward += reward
            correct += info["correct"]
            if info["true_label"] == 1 and action == 1:
                tp += 1
            elif info["true_label"] == 0 and action == 1:
                fp += 1
            elif info["true_label"] == 1 and action == 0:
                fn += 1
            else:
                tn += 1

            state = next_state
            steps += 1
            if done:
                break

        accuracy = correct / max(steps, 1)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)

        self.episode_rewards.append(total_reward)
        self.episode_accuracies.append(accuracy)
        self.episode_precisions.append(precision)
        self.episode_recalls.append(recall)

        return {
            "total_reward": total_reward,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "steps": steps,
            "epsilon": self.epsilon,
        }

    def predict(self, features: np.ndarray, model_scores: np.ndarray) -> np.ndarray:
        """
        Greedy policy inference: predict fraud label for each transaction.

        Returns
        -------
        actions : np.ndarray of {0, 1}
        """
        self.q_net.eval()
        states = np.concatenate([features, model_scores], axis=1).astype(np.float32)
        states_t = torch.FloatTensor(states).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(states_t)
        self.q_net.train()
        return q_values.argmax(dim=1).cpu().numpy()

    def predict_scores(self, features: np.ndarray, model_scores: np.ndarray) -> np.ndarray:
        """
        Return the Q-value difference (action=1 minus action=0) as a
        continuous risk score normalised to [0, 1] via sigmoid.
        """
        self.q_net.eval()
        states = np.concatenate([features, model_scores], axis=1).astype(np.float32)
        states_t = torch.FloatTensor(states).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(states_t)
        self.q_net.train()
        q_diff = q_values[:, 1] - q_values[:, 0]
        scores = torch.sigmoid(q_diff).cpu().numpy()
        return scores

    def save(self, path: Optional[Path] = None):
        path = path or MODELS_DIR / "rl_dqn_agent.pth"
        torch.save({
            "q_net": self.q_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "state_dim": self.state_dim,
            "n_actions": self.n_actions,
        }, path)
        logger.info("DQN agent saved → %s", path)

    def load(self, path: Optional[Path] = None):
        path = path or MODELS_DIR / "rl_dqn_agent.pth"
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.q_net.load_state_dict(checkpoint["q_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        logger.info("DQN agent loaded ← %s", path)


# ==============================================================================
# 5. Training Loop
# ==============================================================================
def train_rl_agent(
    X_train: np.ndarray,
    model_scores_train: np.ndarray,
    y_train: np.ndarray,
    n_episodes: int = 15,
    max_steps_per_episode: int = 5000,
    hidden_dim: int = 128,
    lr: float = 1e-3,
    gamma: float = 0.95,
    epsilon_decay: int = 30_000,
    device: str = "cpu",
) -> Tuple[DQNAgent, List[Dict]]:
    """
    Train a DQN agent on the fraud detection environment.

    Parameters
    ----------
    X_train : (N, D) feature matrix.
    model_scores_train : (N, 5) scores from [RF, XGB, ISO, LSTM, GNN].
    y_train : (N,) binary labels.
    n_episodes : number of training episodes.
    max_steps_per_episode : max transactions per episode.

    Returns
    -------
    agent : trained DQNAgent
    history : list of per-episode metric dicts
    """
    set_seed(SEED)
    ensure_directories()

    env = FraudDetectionEnvironment(
        features=X_train,
        model_scores=model_scores_train,
        labels=y_train,
    )

    agent = DQNAgent(
        state_dim=env.state_dim,
        n_actions=env.n_actions,
        hidden_dim=hidden_dim,
        lr=lr,
        gamma=gamma,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=epsilon_decay,
        batch_size=64,
        device=device,
    )

    logger.info("Training DQN agent: %d episodes, max %d steps each, state_dim=%d",
                n_episodes, max_steps_per_episode, env.state_dim)

    history = []
    for ep in range(1, n_episodes + 1):
        metrics = agent.train_episode(env, max_steps=max_steps_per_episode)
        history.append(metrics)

        if ep % max(1, n_episodes // 5) == 0 or ep == 1:
            logger.info(
                "  Episode %d/%d  reward=%.1f  acc=%.3f  prec=%.3f  "
                "recall=%.3f  f1=%.3f  eps=%.3f",
                ep, n_episodes, metrics["total_reward"], metrics["accuracy"],
                metrics["precision"], metrics["recall"], metrics["f1"],
                metrics["epsilon"],
            )

    agent.save()
    logger.info("RL training complete. Final episode: acc=%.3f, F1=%.3f",
                history[-1]["accuracy"], history[-1]["f1"])

    return agent, history


# ==============================================================================
# 6. Visualization
# ==============================================================================
def visualize_rl_training(
    history: List[Dict],
    output_path: Optional[str] = None,
) -> None:
    """Plot RL training curves: reward, accuracy, precision/recall, F1."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = output_path or str(RESULTS_DIR / "rl_training_curves.png")

    episodes = list(range(1, len(history) + 1))
    rewards = [h["total_reward"] for h in history]
    accuracies = [h["accuracy"] for h in history]
    precisions = [h["precision"] for h in history]
    recalls = [h["recall"] for h in history]
    f1s = [h["f1"] for h in history]
    epsilons = [h["epsilon"] for h in history]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 1. Cumulative reward
    axes[0, 0].plot(episodes, rewards, color='#1f77b4', linewidth=1.5)
    axes[0, 0].set_title('Episode Reward', fontsize=12)
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].grid(alpha=0.3)

    # 2. Accuracy
    axes[0, 1].plot(episodes, accuracies, color='#2ca02c', linewidth=1.5)
    axes[0, 1].set_title('Episode Accuracy', fontsize=12)
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_ylim(0, 1.05)
    axes[0, 1].grid(alpha=0.3)

    # 3. Precision & Recall
    axes[0, 2].plot(episodes, precisions, color='#ff7f0e', linewidth=1.5, label='Precision')
    axes[0, 2].plot(episodes, recalls, color='#d62728', linewidth=1.5, label='Recall')
    axes[0, 2].set_title('Precision & Recall', fontsize=12)
    axes[0, 2].set_xlabel('Episode')
    axes[0, 2].set_ylabel('Score')
    axes[0, 2].legend()
    axes[0, 2].set_ylim(0, 1.05)
    axes[0, 2].grid(alpha=0.3)

    # 4. F1 Score
    axes[1, 0].plot(episodes, f1s, color='#9467bd', linewidth=1.5)
    axes[1, 0].set_title('F1 Score', fontsize=12)
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('F1')
    axes[1, 0].set_ylim(0, 1.05)
    axes[1, 0].grid(alpha=0.3)

    # 5. Epsilon decay
    axes[1, 1].plot(episodes, epsilons, color='#8c564b', linewidth=1.5)
    axes[1, 1].set_title('Exploration Rate (ε)', fontsize=12)
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Epsilon')
    axes[1, 1].set_ylim(0, 1.05)
    axes[1, 1].grid(alpha=0.3)

    # 6. Confusion matrix of last episode
    last = history[-1]
    cm = np.array([[last["tn"], last["fp"]], [last["fn"], last["tp"]]])
    im = axes[1, 2].imshow(cm, cmap='Blues', interpolation='nearest')
    axes[1, 2].set_title('Final Episode Confusion Matrix', fontsize=12)
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

    plt.suptitle('RL-Based Policy Learning — Training Progress (Novelty #9)',
                 fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("RL training visualization saved → %s", output_path)
