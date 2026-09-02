"""
fast_coord.py – Fast coordinate descent optimizer with caching.

Uses single evaluations for speed, multi-rep for final verification.
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


def params_key(params: dict) -> tuple:
    return tuple(sorted((k, str(v)) for k, v in params.items()))


class FastCoordOptimizer:
    def __init__(self, game: str, run_type: str = "fast", n_workers: int = 6):
        self.game = game
        self.run_type = run_type
        self.n_workers = n_workers
        self.valid_params = load_valid_params(game)
        self.cache: dict[tuple, float] = {}
        self.n_evals = 0
        self.n_cache_hits = 0
        self.best_params = None
        self.best_score = float("-inf")

    def evaluate(self, params: dict) -> float | None:
        key = params_key(params)
        if key in self.cache:
            self.n_cache_hits += 1
            return self.cache[key]
        try:
            score = api_client.run_game(
                game=self.game, params=params, run_type=self.run_type,
                timeout_ms=0, http_timeout=600.0,
            )
            self.cache[key] = score
            self.n_evals += 1
            if score > self.best_score:
                self.best_score = score
                self.best_params = dict(params)
                logger.info("New best: %.4f (evals=%d)", score, self.n_evals)
            return score
        except api_client.APIError as e:
            logger.warning("APIError: %s", e)
            return None

    def evaluate_mean(self, params: dict, n_reps: int = 5) -> float | None:
        """Evaluate multiple times and return mean."""
        scores = []
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            futures = []
            for _ in range(n_reps):
                futures.append(executor.submit(
                    api_client.run_game, game=self.game, params=params,
                    run_type=self.run_type, timeout_ms=0, http_timeout=600.0,
                ))
            for f in as_completed(futures):
                try:
                    scores.append(f.result())
                except api_client.APIError:
                    pass
        if not scores:
            return None
        mean = sum(scores) / len(scores)
        logger.info("Mean over %d reps: %.4f (scores: %s)", len(scores), mean, [f"{s:.1f}" for s in scores])
        return mean

    def coord_descent(self, start_params: dict, max_rounds: int = 20) -> tuple[dict, float]:
        """Exhaustive coordinate descent."""
        current = dict(start_params)
        current_score = self.evaluate(current)
        if current_score is None:
            return current, float("-inf")
        logger.info("Coord descent start: %.4f", current_score)

        for round_num in range(1, max_rounds + 1):
            improved = False
            logger.info("=== Round %d (best: %.4f) ===", round_num, current_score)

            for param_name, valid_values in self.valid_params.items():
                # Evaluate all values in parallel
                candidates = [(v, {**current, param_name: v}) for v in valid_values if v != current[param_name]]
                if not candidates:
                    continue

                best_val = current[param_name]
                best_val_score = current_score

                with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
                    future_map = {
                        executor.submit(self.evaluate, params): val
                        for val, params in candidates
                    }
                    for f in as_completed(future_map):
                        val = future_map[f]
                        score = f.result()
                        if score is not None and score > best_val_score:
                            best_val_score = score
                            best_val = val

                if best_val != current[param_name]:
                    current[param_name] = best_val
                    current_score = best_val_score
                    improved = True
                    logger.info("  %s=%s -> %.4f", param_name, best_val, current_score)

            if not improved:
                logger.info("Converged after %d rounds.", round_num)
                break

        return current, current_score

    def random_start(self, rng: random.Random) -> dict:
        return {name: rng.choice(values) for name, values in self.valid_params.items()}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", required=True, choices=["CantStop", "Dominion", "ExplodingKittens", "Wonders7"])
    parser.add_argument("--run-type", default="fast")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--n-random", type=int, default=5)
    parser.add_argument("--n-verify", type=int, default=10, help="Reps for final verification")
    args = parser.parse_args()

    optimizer = FastCoordOptimizer(game=args.game, run_type=args.run_type, n_workers=args.workers)
    rng = random.Random(42)

    best_params = None
    best_score = float("-inf")

    # Start from best known
    if args.game in BEST_KNOWN:
        logger.info("--- Evaluating best known params ---")
        score = optimizer.evaluate(BEST_KNOWN[args.game])
        if score is not None and score > best_score:
            best_score = score
            best_params = dict(BEST_KNOWN[args.game])
        logger.info("Best known: %.4f", best_score)

    # Coord descent from best known
    if best_params:
        logger.info("--- Coord descent from best known ---")
        params, score = optimizer.coord_descent(best_params)
        if score > best_score:
            best_score = score
            best_params = dict(params)

    # Random restarts + coord descent
    for i in range(args.n_random):
        logger.info("--- Random restart %d/%d ---", i + 1, args.n_random)
        start = optimizer.random_start(rng)
        params, score = optimizer.coord_descent(start)
        if score > best_score:
            best_score = score
            best_params = dict(params)

    # Final verification with multiple reps
    logger.info("--- Final verification (%d reps) ---", args.n_verify)
    mean_score = optimizer.evaluate_mean(best_params, args.n_verify)

    logger.info("FINAL: %.4f (verified mean: %.4f)", best_score, mean_score)
    logger.info("Params: %s", json.dumps(best_params, indent=2))
    logger.info("Total evals: %d, cache hits: %d", optimizer.n_evals, optimizer.n_cache_hits)

    # Save
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"results_coord_{args.game}_{timestamp}.json"
    with open(filename, "w") as f:
        json.dump({
            "game": args.game, "best_score": best_score,
            "verified_mean": mean_score, "best_params": best_params,
            "n_evals": optimizer.n_evals,
        }, f, indent=2)
    logger.info("Saved to %s", filename)


if __name__ == "__main__":
    main()
