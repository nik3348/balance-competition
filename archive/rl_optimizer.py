"""
rl_optimizer.py – Reinforcement learning optimizer for game parameters.

Uses REINFORCE (policy gradient) with a neural network to learn a distribution
over discrete parameter values. The reward signal comes from the game evaluation API.

The problem is framed as a one-step MDP (contextual bandit):
  - State:  a learnable context embedding (or game feature vector)
  - Action: for each parameter, select one of its discrete valid values
  - Reward: the score returned by evaluate_params
"""

from __future__ import annotations

import json
import logging
import math
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

import api_client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy torch import – fail fast with a clear message
# ---------------------------------------------------------------------------

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.distributions import Categorical
except ImportError:
    raise ImportError(
        "PyTorch is required for rl_optimizer.  Install it with:\n"
        "  pip install torch   # or: uv add torch"
    )


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class Trajectory:
    """One sampled trajectory (episode)."""

    param_names: list[str]
    action_indices: list[int]  # index into each param's value list
    log_probs: list[torch.Tensor]
    reward: float = 0.0


@dataclass
class EpisodeStats:
    """Stats for one training episode/batch."""

    episode: int
    mean_reward: float
    best_reward: float
    entropy: float
    policy_loss: float


# ---------------------------------------------------------------------------
# Policy network
# ---------------------------------------------------------------------------


class PolicyNetwork(nn.Module):
    """Outputs a categorical distribution for each parameter independently.

    Architecture: shared trunk → per-parameter heads.

    Input:  a single learnable context vector (since each episode is a fresh
            parameter selection, there is no external state).
    Output: one logit vector per parameter, each with ``len(valid_values)`` entries.
    """

    def __init__(
        self,
        n_params: int,
        param_sizes: list[int],
        hidden_dim: int = 128,
        n_hidden: int = 2,
    ) -> None:
        super().__init__()
        self.n_params = n_params
        self.param_sizes = param_sizes

        # Learnable context (replaces a fixed state input)
        self.context = nn.Parameter(torch.randn(1, hidden_dim) * 0.1)

        # Shared trunk
        layers: list[nn.Module] = []
        in_dim = hidden_dim
        for _ in range(n_hidden):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        self.trunk = nn.Sequential(*layers)

        # One head per parameter
        self.heads = nn.ModuleList(
            [nn.Linear(hidden_dim, size) for size in param_sizes]
        )

    def forward(self) -> list[Categorical]:
        """Return a list of Categorical distributions, one per parameter."""
        x = self.context  # (1, hidden_dim)
        x = self.trunk(x)  # (1, hidden_dim)
        dists = []
        for head in self.heads:
            logits = head(x).squeeze(0)  # (param_size,)
            dists.append(Categorical(logits=logits))
        return dists


# ---------------------------------------------------------------------------
# RL Optimizer
# ---------------------------------------------------------------------------


class RLOptimizer:
    """REINFORCE optimizer for game parameters.

    Parameters
    ----------
    game:           Game identifier forwarded to the API.
    run_type:       Execution mode passed to the API ("fast", "medium", "full").
    hidden_dim:     Hidden layer width in the policy network.
    n_hidden:       Number of hidden layers.
    lr:             Learning rate for Adam.
    entropy_coeff:  Weight of the entropy bonus (encourages exploration).
    baseline_decay: EMA decay for the reward baseline (variance reduction).
    batch_size:     Number of trajectories sampled per update step.
    n_episodes:     Total number of batched updates.
    n_workers:      Parallel API evaluation threads.
    timeout_ms:     Per-game server-side timeout (0 = none).
    seed:           RNG seed for reproducibility.
    device:         "cpu" or "cuda" (auto-detected if not given).
    """

    def __init__(
        self,
        game: str,
        run_type: str = "fast",
        hidden_dim: int = 128,
        n_hidden: int = 2,
        lr: float = 3e-4,
        entropy_coeff: float = 0.01,
        baseline_decay: float = 0.95,
        batch_size: int = 16,
        n_episodes: int = 200,
        n_workers: int = 4,
        timeout_ms: int = 0,
        seed: int | None = None,
        device: str | None = None,
    ) -> None:
        self.game = game
        self.run_type = run_type
        self.batch_size = batch_size
        self.n_episodes = n_episodes
        self.n_workers = n_workers
        self.timeout_ms = timeout_ms
        self.entropy_coeff = entropy_coeff
        self.baseline_decay = baseline_decay

        # Determinism
        self._seed = seed
        self._rng = random.Random(seed)
        torch_seed = seed if seed is not None else self._rng.randint(0, 2**31)
        torch.manual_seed(torch_seed)

        # Device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = torch.device(device)

        # Load parameter space
        self._game_params = self._load_game_params(game)
        self._param_names = list(self._game_params.keys())
        self._param_values = [self._game_params[k] for k in self._param_names]
        self._param_sizes = [len(v) for v in self._param_values]

        # Build policy
        self.policy = PolicyNetwork(
            n_params=len(self._param_names),
            param_sizes=self._param_sizes,
            hidden_dim=hidden_dim,
            n_hidden=n_hidden,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.baseline: float | None = None

        self._history: list[EpisodeStats] = []
        self._best_params: dict | None = None
        self._best_score: float = float("-inf")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_game_params(game: str) -> dict:
        with open("valid_params.json") as f:
            all_params = json.load(f)
        return all_params[game]

    def _actions_to_params(self, action_indices: list[int]) -> dict:
        """Convert action indices to a concrete parameter dict."""
        params = {}
        for name, values, idx in zip(
            self._param_names, self._param_values, action_indices
        ):
            params[name] = values[idx]
        return params

    def _evaluate_batch(self, batch: list[Trajectory]) -> None:
        """Fill in rewards for a batch of trajectories via parallel API calls."""

        def _eval(traj: Trajectory) -> Trajectory:
            params = self._actions_to_params(traj.action_indices)
            try:
                score = api_client.run_game(
                    game=self.game,
                    params=params,
                    run_type=self.run_type,
                    timeout_ms=self.timeout_ms,
                    http_timeout=300.0,
                )
                traj.reward = score
            except api_client.APIError as exc:
                logger.warning("APIError: %s", exc)
                traj.reward = float("-inf")
            return traj

        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            futures = [executor.submit(_eval, t) for t in batch]
            for future in as_completed(futures):
                future.result()  # propagate unexpected exceptions

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def _sample_trajectories(self, n: int) -> tuple[list[Trajectory], list[Categorical]]:
        """Sample *n* trajectories from the current policy."""
        dists = self.policy()  # list of Categorical
        trajectories: list[Trajectory] = []

        for _ in range(n):
            action_indices: list[int] = []
            log_probs: list[torch.Tensor] = []
            for dist in dists:
                action = dist.sample()
                action_indices.append(action.item())
                log_probs.append(dist.log_prob(action))
            trajectories.append(
                Trajectory(
                    param_names=self._param_names,
                    action_indices=action_indices,
                    log_probs=log_probs,
                )
            )
        return trajectories, dists

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def _update(self, batch: list[Trajectory]) -> tuple[float, float]:
        """Run one REINFORCE update on *batch*.  Returns (policy_loss, entropy)."""
        rewards = torch.tensor(
            [t.reward for t in batch], dtype=torch.float32, device=self.device
        )

        # Replace -inf rewards (failed evaluations) with the worst valid reward
        valid_mask = rewards > float("-inf")
        if valid_mask.any():
            worst = rewards[valid_mask].min()
            rewards[~valid_mask] = worst
        else:
            # All failed – nothing to learn
            return 0.0, 0.0

        # Update baseline (EMA of mean reward)
        mean_reward = rewards.mean().item()
        if self.baseline is None:
            self.baseline = mean_reward
        else:
            self.baseline = (
                self.baseline_decay * self.baseline
                + (1 - self.baseline_decay) * mean_reward
            )

        advantages = rewards - self.baseline

        # Policy loss: -E[advantage * log_prob]
        policy_loss = torch.tensor(0.0, device=self.device)
        for i, traj in enumerate(batch):
            for lp in traj.log_probs:
                policy_loss += -lp * advantages[i]
        policy_loss /= len(batch)

        # Entropy bonus (for exploration)
        dists = self.policy()
        entropy = sum(d.entropy().mean() for d in dists) / len(dists)

        loss = policy_loss - self.entropy_coeff * entropy

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        self.optimizer.step()

        return policy_loss.item(), entropy.item()

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> tuple[dict, float]:
        """Run the RL optimization.  Returns (best_params, best_score)."""
        logger.info(
            "Starting RL optimization — game=%s | episodes=%d | batch_size=%d"
            " | device=%s",
            self.game,
            self.n_episodes,
            self.batch_size,
            self.device,
        )
        logger.info(
            "Parameters (%d): %s", len(self._param_names), self._param_names
        )

        for ep in range(1, self.n_episodes + 1):
            # 1. Sample trajectories
            batch, dists = self._sample_trajectories(self.batch_size)

            # 2. Evaluate
            self._evaluate_batch(batch)

            # 3. Track best
            for traj in batch:
                if traj.reward > self._best_score:
                    self._best_score = traj.reward
                    self._best_params = self._actions_to_params(traj.action_indices)

            # 4. Policy update
            policy_loss, entropy = self._update(batch)

            # 5. Log
            rewards = [t.reward for t in batch if t.reward > float("-inf")]
            mean_r = sum(rewards) / len(rewards) if rewards else 0.0
            best_r = max(rewards) if rewards else 0.0

            stats = EpisodeStats(
                episode=ep,
                mean_reward=mean_r,
                best_reward=best_r,
                entropy=entropy,
                policy_loss=policy_loss,
            )
            self._history.append(stats)

            if ep % 10 == 0 or ep == 1:
                logger.info(
                    "Episode %4d/%d | mean=%10.4f | best=%10.4f"
                    " | loss=%10.4f | entropy=%7.4f | baseline=%10.4f",
                    ep,
                    self.n_episodes,
                    mean_r,
                    best_r,
                    policy_loss,
                    entropy,
                    self.baseline or 0.0,
                )

        logger.info("RL optimization complete. Best score: %.4f", self._best_score)
        return self._best_params, self._best_score


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main():
    import argparse

    parser = argparse.ArgumentParser(description="RL game parameter optimizer")
    parser.add_argument(
        "--game",
        required=True,
        choices=["Dominion", "ExplodingKittens", "Wonders7", "CantStop"],
    )
    parser.add_argument("--run-type", default="fast", choices=["fast", "medium", "full"])
    parser.add_argument("--episodes", type=int, default=200, help="Number of update steps")
    parser.add_argument("--batch-size", type=int, default=16, help="Trajectories per update")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--n-hidden", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--entropy-coeff", type=float, default=0.01)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout-ms", type=int, default=0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None, choices=["cpu", "cuda", "mps"])

    args = parser.parse_args()

    optimizer = RLOptimizer(
        game=args.game,
        run_type=args.run_type,
        hidden_dim=args.hidden_dim,
        n_hidden=args.n_hidden,
        lr=args.lr,
        entropy_coeff=args.entropy_coeff,
        batch_size=args.batch_size,
        n_episodes=args.episodes,
        n_workers=args.workers,
        timeout_ms=args.timeout_ms,
        seed=args.seed,
        device=args.device,
    )

    best_params, best_score = optimizer.run()

    # Print results
    print("\n" + "=" * 60)
    print(f"BEST PARAMETERS FOR {args.game} (RL)")
    print("=" * 60)
    for k, v in best_params.items():
        print(f"  {k}: {v}")
    print("-" * 60)
    print(f"  SCORE: {best_score:.6f}")
    print("=" * 60)

    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"results_rl_{args.game}_{timestamp}.json"

    results = {
        "game": args.game,
        "optimizer": "rl_reinforce",
        "best_score": best_score,
        "best_params": best_params,
        "config": {
            "run_type": args.run_type,
            "episodes": args.episodes,
            "batch_size": args.batch_size,
            "hidden_dim": args.hidden_dim,
            "n_hidden": args.n_hidden,
            "lr": args.lr,
            "entropy_coeff": args.entropy_coeff,
            "seed": args.seed,
        },
        "history": [
            {
                "episode": s.episode,
                "mean_reward": s.mean_reward,
                "best_reward": s.best_reward,
                "entropy": s.entropy,
                "policy_loss": s.policy_loss,
            }
            for s in optimizer._history
        ],
    }

    with open(filename, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {filename}")


if __name__ == "__main__":
    main()
