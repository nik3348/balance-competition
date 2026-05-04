"""
fast_optimizer.py – Fast greedy optimizer for game parameters.

Uses random search + hill climbing for quick exploration.
"""

import json
import logging
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import api_client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_game_params(game: str) -> dict:
    """Load valid parameters for a game."""
    with open("valid_params.json") as f:
        all_params = json.load(f)
    return all_params[game]


def sample_params(game_params: dict, rng: random.Random) -> dict:
    """Sample random parameters from valid values."""
    params = {}
    for name, values in game_params.items():
        params[name] = rng.choice(values)
    return params


def evaluate_params(game: str, params: dict, run_type: str = "fast") -> float | None:
    """Evaluate a parameter set via the API."""
    try:
        score = api_client.run_game(
            game=game,
            params=params,
            run_type=run_type,
            timeout_ms=0,
            http_timeout=300.0,
        )
        return score
    except api_client.APIError as exc:
        logger.warning("APIError: %s", exc)
        return None


def mutate_params(
    params: dict, game_params: dict, rng: random.Random, n_changes: int = 1
) -> dict:
    """Mutate a parameter set by changing n random parameters."""
    mutated = dict(params)
    param_names = list(game_params.keys())
    for _ in range(n_changes):
        name = rng.choice(param_names)
        mutated[name] = rng.choice(game_params[name])
    return mutated


def run_optimization(
    game: str,
    n_iterations: int = 100,
    n_workers: int = 4,
    seed: int | None = None,
) -> tuple[dict, float]:
    """Run fast optimization and return best params and score."""
    rng = random.Random(seed)
    game_params = load_game_params(game)

    best_params = None
    best_score = float("-inf")
    all_results = []

    logger.info(
        "Starting fast optimization for %s | iterations=%d | workers=%d",
        game,
        n_iterations,
        n_workers,
    )

    # Phase 1: Random search
    logger.info("=== Phase 1: Random Search ===")
    n_random = min(n_iterations // 2, 50)

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {}
        for i in range(n_random):
            params = sample_params(game_params, rng)
            futures[executor.submit(evaluate_params, game, params)] = (i, params)

        for future in as_completed(futures):
            idx, params = futures[future]
            score = future.result()
            if score is not None:
                all_results.append((params, score))
                if score > best_score:
                    best_score = score
                    best_params = dict(params)
                    logger.info("New best: %.4f (iteration %d)", best_score, idx)

    logger.info("Random search complete. Best so far: %.4f", best_score)

    # Phase 2: Hill climbing from best
    logger.info("=== Phase 2: Hill Climbing ===")
    n_climb = n_iterations - n_random

    current_params = dict(best_params)
    current_score = best_score

    for i in range(n_climb):
        # Try mutating 1-3 parameters
        for n_changes in [1, 2, 3]:
            candidate = mutate_params(current_params, game_params, rng, n_changes)
            score = evaluate_params(game, candidate)

            if score is not None and score > current_score:
                current_params = candidate
                current_score = score
                if score > best_score:
                    best_score = score
                    best_params = dict(candidate)
                    logger.info(
                        "Hill climb improved: %.4f (changes=%d)", best_score, n_changes
                    )
                break  # Restart from improved solution

    logger.info("Optimization complete. Best score: %.4f", best_score)
    return best_params, best_score


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Fast game parameter optimizer")
    parser.add_argument(
        "--game",
        required=True,
        choices=["Dominion", "ExplodingKittens", "Wonders7", "CantStop"],
    )
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=None)

    args = parser.parse_args()

    best_params, best_score = run_optimization(
        game=args.game,
        n_iterations=args.iterations,
        n_workers=args.workers,
        seed=args.seed,
    )

    # Print results
    print("\n" + "=" * 60)
    print(f"BEST PARAMETERS FOR {args.game}")
    print("=" * 60)
    for k, v in best_params.items():
        print(f"  {k}: {v}")
    print("-" * 60)
    print(f"  SCORE: {best_score:.6f}")
    print("=" * 60)

    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"results_fast_{args.game}_{timestamp}.json"

    results = {
        "game": args.game,
        "optimizer": "fast_greedy",
        "best_score": best_score,
        "best_params": best_params,
    }

    with open(filename, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {filename}")


if __name__ == "__main__":
    main()
