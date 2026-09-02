"""
smart_optimizer.py – Aggressive optimizer targeting score >= 950.

Strategy:
1. Wide parallel random search
2. Systematic coordinate descent (try every valid value per param)
3. Multi-restart from top-K candidates
4. Result caching to avoid redundant API calls
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
    with open("valid_params.json") as f:
        return json.load(f)[game]


def params_key(params: dict) -> tuple:
    return tuple(sorted((k, str(v)) for k, v in params.items()))


class SmartOptimizer:
    def __init__(self, game: str, n_workers: int = 8, seed: int | None = None):
        self.game = game
        self.game_params = load_game_params(game)
        self.n_workers = n_workers
        self.rng = random.Random(seed)
        self.cache: dict[tuple, float] = {}
        self.best_params: dict | None = None
        self.best_score: float = float("-inf")
        self.n_evals: int = 0
        self.n_cache_hits: int = 0

    def _evaluate(self, params: dict, run_type: str = "fast") -> float | None:
        key = params_key(params)
        if key in self.cache:
            self.n_cache_hits += 1
            return self.cache[key]

        try:
            score = api_client.run_game(
                game=self.game,
                params=params,
                run_type=run_type,
                timeout_ms=0,
                http_timeout=300.0,
            )
            self.cache[key] = score
            self.n_evals += 1
            if score > self.best_score:
                self.best_score = score
                self.best_params = dict(params)
                logger.info(
                    "New best: %.4f (evals=%d, cache_hits=%d)",
                    score, self.n_evals, self.n_cache_hits,
                )
            return score
        except api_client.APIError as e:
            logger.warning("APIError: %s", e)
            return None

    def _sample_random(self) -> dict:
        return {
            name: self.rng.choice(values)
            for name, values in self.game_params.items()
        }

    def random_search(self, n: int, run_type: str = "fast") -> list[tuple[dict, float]]:
        """Evaluate n random parameter sets in parallel."""
        logger.info("=== Random Search: %d evaluations ===", n)
        candidates = [self._sample_random() for _ in range(n)]
        results: list[tuple[dict, float]] = []

        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            futures = {executor.submit(self._evaluate, p, run_type): p for p in candidates}
            for future in as_completed(futures):
                score = future.result()
                if score is not None:
                    results.append((futures[future], score))

        results.sort(key=lambda x: x[1], reverse=True)
        if results:
            logger.info("Random search done. Best: %.4f", results[0][1])
        return results

    def coordinate_descent(
        self, start_params: dict, max_rounds: int = 20, run_type: str = "fast"
    ) -> tuple[dict, float]:
        """For each parameter in turn, try all valid values and keep the best.

        Repeats until a full round produces no improvement.
        """
        current = dict(start_params)
        current_score = self._evaluate(current, run_type) or float("-inf")
        logger.info("Coordinate descent start: %.4f", current_score)

        for round_num in range(1, max_rounds + 1):
            improved = False

            for param_name, valid_values in self.game_params.items():
                # Evaluate all values for this param in parallel
                candidates = [
                    (val, {**current, param_name: val})
                    for val in valid_values
                    if val != current[param_name]
                ]
                if not candidates:
                    continue

                best_val = current[param_name]
                best_val_score = current_score

                with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
                    futures = {
                        executor.submit(self._evaluate, p, run_type): val
                        for val, p in candidates
                    }
                    for future in as_completed(futures):
                        val = futures[future]
                        score = future.result()
                        if score is not None and score > best_val_score:
                            best_val_score = score
                            best_val = val

                if best_val != current[param_name]:
                    current[param_name] = best_val
                    current_score = best_val_score
                    improved = True

            logger.info("Round %d done: %.4f", round_num, current_score)
            if not improved:
                logger.info("No improvement in round %d, stopping.", round_num)
                break

        return current, current_score

    def run(
        self,
        n_random: int = 300,
        top_k: int = 5,
        run_type: str = "fast",
    ) -> tuple[dict, float]:
        """Full optimization run."""
        # Phase 1: Wide random search
        random_results = self.random_search(n_random, run_type)

        # Phase 2: Coordinate descent from top-K starting points
        logger.info("=== Coordinate Descent from top-%d ===", top_k)
        for i, (params, score) in enumerate(random_results[:top_k]):
            logger.info(
                "Starting coordinate descent %d/%d from %.4f", i + 1, top_k, score
            )
            self.coordinate_descent(params, run_type=run_type)

        logger.info(
            "Optimization complete. Best=%.4f | evals=%d | cache_hits=%d",
            self.best_score, self.n_evals, self.n_cache_hits,
        )
        return self.best_params, self.best_score

    def save_results(self, filename: str | None = None) -> str:
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"results_smart_{self.game}_{timestamp}.json"
        with open(filename, "w") as f:
            json.dump(
                {
                    "game": self.game,
                    "optimizer": "smart_coordinate_descent",
                    "best_score": self.best_score,
                    "best_params": self.best_params,
                    "n_evaluations": self.n_evals,
                    "n_cache_hits": self.n_cache_hits,
                },
                f,
                indent=2,
            )
        return filename


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Smart game parameter optimizer")
    parser.add_argument(
        "--game",
        required=True,
        choices=["Dominion", "ExplodingKittens", "Wonders7", "CantStop"],
    )
    parser.add_argument("--random", type=int, default=300, help="Random search evaluations")
    parser.add_argument("--top-k", type=int, default=5, help="Hill climbing starting points")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--run-type", default="fast", help="API run type (fast/medium/full/competition)")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--start-params", type=str, default=None,
        help="JSON string of starting params for coordinate descent only"
    )
    args = parser.parse_args()

    optimizer = SmartOptimizer(game=args.game, n_workers=args.workers, seed=args.seed)

    if args.start_params:
        import json as _json
        start = _json.loads(args.start_params)
        logger.info("Skipping random search; coordinate descent from provided params")
        best_params, best_score = optimizer.coordinate_descent(start, run_type=args.run_type)
    else:
        best_params, best_score = optimizer.run(
            n_random=args.random,
            top_k=args.top_k,
            run_type=args.run_type,
        )

    print("\n" + "=" * 60)
    print(f"BEST PARAMETERS FOR {args.game}")
    print("=" * 60)
    for k, v in best_params.items():
        print(f"  {k}: {v}")
    print("-" * 60)
    print(f"  SCORE: {best_score:.6f}")
    print("=" * 60)

    filename = optimizer.save_results()
    print(f"\nResults saved to: {filename}")


if __name__ == "__main__":
    main()
