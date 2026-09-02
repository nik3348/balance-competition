"""
optimize_all.py – Exhaustive coordinate-descent optimizer for all games.

Strategy:
1. Start from best known params (or random if none)
2. For each parameter, try ALL valid values, evaluate N times each, pick best
3. Repeat until convergence
4. Use "full" run_type for more accurate scores
"""

import json
import logging
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import api_client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# Best known starting points
BEST_KNOWN = {
    "CantStop": {
        "TWO_MAX": 4, "THREE_MAX": 2, "FOUR_MAX": 7, "FIVE_MAX": 8,
        "SIX_MAX": 11, "SEVEN_MAX": 12, "EIGHT_MAX": 11, "NINE_MAX": 6,
        "TEN_MAX": 4, "ELEVEN_MAX": 2, "TWELVE_MAX": 4,
        "COLUMNS_TO_WIN": 2, "MARKERS": 2,
    },
    "ExplodingKittens": {
        "nCardsPerPlayer": 5, "nopeOwnCards": False,
        "ATTACK_count": 1, "SKIP_count": 3, "FAVOR_count": 8,
        "SHUFFLE_count": 3, "SEETHEFUTURE_count": 4, "TACOCAT_count": 4,
        "MELONCAT_count": 2, "BEARDCAT_count": 10, "RAINBOWCAT_count": 5,
        "FURRYCAT_count": 8, "NOPE_count": 9, "DEFUSE_count": 6,
    },
}


def load_valid_params(game: str) -> dict:
    with open("valid_params.json") as f:
        return json.load(f)[game]


def evaluate_mean(params: dict, game: str, n_reps: int = 3, run_type: str = "full") -> float | None:
    """Evaluate params n_reps times and return the mean score."""
    scores = []
    for _ in range(n_reps):
        try:
            score = api_client.run_game(
                game=game, params=params, run_type=run_type,
                timeout_ms=0, http_timeout=300.0,
            )
            scores.append(score)
        except api_client.APIError as e:
            logger.warning("APIError: %s", e)
    if not scores:
        return None
    return sum(scores) / len(scores)


def evaluate_parallel_mean(params: dict, game: str, n_reps: int = 3, run_type: str = "full", n_workers: int = 4) -> float | None:
    """Evaluate params n_reps times in parallel and return the mean."""
    scores = []
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = []
        for _ in range(n_reps):
            futures.append(executor.submit(
                api_client.run_game, game=game, params=params,
                run_type=run_type, timeout_ms=0, http_timeout=300.0,
            ))
        for f in as_completed(futures):
            try:
                scores.append(f.result())
            except api_client.APIError as e:
                logger.warning("APIError: %s", e)
    if not scores:
        return None
    return sum(scores) / len(scores)


def coordinate_descent(
    game: str,
    start_params: dict,
    valid_params: dict,
    n_reps: int = 3,
    run_type: str = "full",
    max_rounds: int = 20,
    n_workers: int = 4,
) -> tuple[dict, float]:
    """Exhaustive coordinate descent: try ALL values for each param."""
    current = dict(start_params)

    # Evaluate starting point
    current_score = evaluate_parallel_mean(current, game, n_reps, run_type, n_workers)
    if current_score is None:
        logger.error("Could not evaluate starting params!")
        return current, float("-inf")
    logger.info("Starting score: %.4f", current_score)

    for round_num in range(1, max_rounds + 1):
        improved = False
        logger.info("=== Round %d (current best: %.4f) ===", round_num, current_score)

        for param_name, valid_values in valid_params.items():
            current_val = current[param_name]
            best_val = current_val
            best_val_score = current_score

            # Try each value in parallel
            candidates = [(v, {**current, param_name: v}) for v in valid_values if v != current_val]
            if not candidates:
                continue

            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                future_map = {}
                for val, params in candidates:
                    # Evaluate each candidate n_reps times
                    f = executor.submit(evaluate_parallel_mean, params, game, n_reps, run_type, max(1, n_workers // 2))
                    future_map[f] = val

                for f in as_completed(future_map):
                    val = future_map[f]
                    score = f.result()
                    if score is not None and score > best_val_score:
                        best_val_score = score
                        best_val = val
                        logger.info("  %s=%s -> %.4f (improved!)", param_name, val, score)

            if best_val != current_val:
                current[param_name] = best_val
                current_score = best_val_score
                improved = True
                logger.info("  Updated %s=%s (score=%.4f)", param_name, best_val, current_score)

        logger.info("Round %d done: best=%.4f", round_num, current_score)
        if not improved:
            logger.info("Converged after %d rounds.", round_num)
            break

    return current, current_score


def random_search(
    game: str,
    valid_params: dict,
    n_samples: int = 50,
    n_reps: int = 2,
    run_type: str = "full",
    n_workers: int = 4,
    seed: int = 42,
) -> list[tuple[dict, float]]:
    """Random search to find good starting points."""
    rng = random.Random(seed)
    results = []

    logger.info("Random search: %d samples", n_samples)
    for i in range(n_samples):
        params = {name: rng.choice(values) for name, values in valid_params.items()}
        score = evaluate_parallel_mean(params, game, n_reps, run_type, n_workers)
        if score is not None:
            results.append((params, score))
            logger.info("  Sample %d: %.4f", i + 1, score)

    results.sort(key=lambda x: x[1], reverse=True)
    return results


def optimize_game(
    game: str,
    n_reps: int = 3,
    run_type: str = "full",
    n_workers: int = 4,
    n_random: int = 30,
) -> tuple[dict, float]:
    """Full optimization pipeline for a single game."""
    logger.info("=" * 60)
    logger.info("OPTIMIZING %s", game)
    logger.info("=" * 60)

    valid_params = load_valid_params(game)
    best_params = None
    best_score = float("-inf")

    # Phase 1: Evaluate best known params
    if game in BEST_KNOWN:
        logger.info("--- Phase 1: Evaluating best known params ---")
        score = evaluate_parallel_mean(BEST_KNOWN[game], game, n_reps, run_type, n_workers)
        if score is not None:
            logger.info("Best known score: %.4f", score)
            if score > best_score:
                best_score = score
                best_params = dict(BEST_KNOWN[game])

    # Phase 2: Random search
    logger.info("--- Phase 2: Random search ---")
    random_results = random_search(game, valid_params, n_random, 2, run_type, n_workers)
    if random_results and random_results[0][1] > best_score:
        best_score = random_results[0][1]
        best_params = dict(random_results[0][0])
        logger.info("Random search best: %.4f", best_score)

    # Phase 3: Coordinate descent from top candidates
    logger.info("--- Phase 3: Coordinate descent ---")
    starts = []
    if best_params:
        starts.append(best_params)
    for params, score in random_results[:3]:
        starts.append(params)

    for i, start_params in enumerate(starts):
        logger.info("--- Coord descent %d/%d ---", i + 1, len(starts))
        improved_params, improved_score = coordinate_descent(
            game, start_params, valid_params, n_reps, run_type, max_rounds=15, n_workers=n_workers,
        )
        if improved_score > best_score:
            best_score = improved_score
            best_params = dict(improved_params)
            logger.info("New global best: %.4f", best_score)

    logger.info("FINAL BEST for %s: %.4f", game, best_score)
    return best_params, best_score


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", default=None, choices=["CantStop", "Dominion", "ExplodingKittens", "Wonders7"])
    parser.add_argument("--n-reps", type=int, default=3, help="Evaluations per candidate")
    parser.add_argument("--run-type", default="full", choices=["fast", "medium", "full", "competition"])
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--n-random", type=int, default=30)
    args = parser.parse_args()

    games = [args.game] if args.game else ["CantStop", "Dominion", "ExplodingKittens", "Wonders7"]

    all_results = {}
    for game in games:
        best_params, best_score = optimize_game(
            game, n_reps=args.n_reps, run_type=args.run_type,
            n_workers=args.workers, n_random=args.n_random,
        )
        all_results[game] = {"score": best_score, "params": best_params}

        # Save per-game results
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"results_optimized_{game}_{timestamp}.json"
        with open(filename, "w") as f:
            json.dump({"game": game, "best_score": best_score, "best_params": best_params}, f, indent=2)
        logger.info("Saved to %s", filename)

    # Print summary
    print("\n" + "=" * 60)
    print("OPTIMIZATION SUMMARY")
    print("=" * 60)
    for game, result in all_results.items():
        print(f"  {game}: {result['score']:.4f}")
        for k, v in result["params"].items():
            print(f"    {k}: {v}")
    print("=" * 60)


if __name__ == "__main__":
    main()
