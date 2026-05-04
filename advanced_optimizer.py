"""
advanced_optimizer.py – Advanced hybrid optimizer for game-parameter optimization.

Combines multiple strategies:
1. Adaptive evolutionary algorithm with self-tuning mutation
2. Island model for population diversity
3. Parallel local search (hill climbing) for fine-tuning
4. Multi-phase optimization (explore → refine → exploit)
5. Thompson sampling for parameter importance learning
"""

from __future__ import annotations

import copy
import json
import logging
import math
import random
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import api_client
from param_space import ParameterSpace, CategoricalParam, MultiCategoricalParam

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class Individual:
    """A single candidate solution with extended metadata."""
    params: dict
    score: float | None = None
    evaluated: bool = False
    age: int = 0
    evaluations: int = 0


@dataclass
class Island:
    """An island in the island model."""
    population: List[Individual]
    best_score: float = float("-inf")
    best_params: dict = field(default_factory=dict)
    stagnation_count: int = 0
    mutation_rate: float = 0.3
    mutation_strength: float = 0.2


# ---------------------------------------------------------------------------
# Parameter importance tracker using Thompson sampling
# ---------------------------------------------------------------------------


class ParameterImportanceTracker:
    """Tracks which parameter values tend to produce high scores.

    Uses Beta distributions for categorical parameters to learn
    which values are more likely to be in high-scoring individuals.
    """

    def __init__(self, param_space: ParameterSpace):
        self.param_space = param_space
        self.alpha: Dict[str, Dict[Any, float]] = {}
        self.beta: Dict[str, Dict[Any, float]] = {}

        for name, param in param_space.params.items():
            if isinstance(param, (CategoricalParam, MultiCategoricalParam)):
                self.alpha[name] = {v: 1.0 for v in param.choices}
                self.beta[name] = {v: 1.0 for v in param.choices}

    def update(self, params: dict, score: float, is_best: bool):
        """Update distributions based on observed score."""
        for name, value in params.items():
            if name not in self.alpha:
                continue
            if isinstance(value, list):
                for v in value:
                    if v in self.alpha[name]:
                        if is_best:
                            self.alpha[name][v] += 1.0
                        else:
                            self.beta[name][v] += 1.0
            elif value in self.alpha[name]:
                if is_best:
                    self.alpha[name][value] += 1.0
                else:
                    self.beta[name][value] += 1.0

    def update_all(self, individuals: List[Individual]):
        """Update from a list of scored individuals, weighting by rank."""
        scored = [ind for ind in individuals if ind.score is not None]
        if not scored:
            return
        scored.sort(key=lambda x: x.score, reverse=True)
        n = len(scored)
        for rank, ind in enumerate(scored):
            # Top 20% get positive updates, bottom 20% get negative
            is_top = rank < max(1, n * 0.2)
            is_bottom = rank >= n * 0.8
            if is_top:
                weight = 1.0 + (n - rank) / n  # stronger for higher ranks
                for name, value in ind.params.items():
                    if name not in self.alpha:
                        continue
                    if isinstance(value, list):
                        for v in value:
                            if v in self.alpha[name]:
                                self.alpha[name][v] += weight
                    elif value in self.alpha[name]:
                        self.alpha[name][value] += weight
            elif is_bottom:
                for name, value in ind.params.items():
                    if name not in self.alpha:
                        continue
                    if isinstance(value, list):
                        for v in value:
                            if v in self.beta[name]:
                                self.beta[name][v] += 0.5
                    elif value in self.beta[name]:
                        self.beta[name][value] += 0.5

    def sample_value(self, param_name: str, rng: random.Random) -> Any:
        """Sample a value using Thompson sampling."""
        param = self.param_space.params[param_name]

        if param_name not in self.alpha:
            if isinstance(param, CategoricalParam):
                return rng.choice(param.choices)
            elif isinstance(param, MultiCategoricalParam):
                return [rng.choice(param.choices) for _ in range(param.count)]
            return None

        if isinstance(param, MultiCategoricalParam):
            result = []
            for _ in range(param.count):
                samples = {}
                for value in self.alpha[param_name]:
                    a = self.alpha[param_name][value]
                    b = self.beta[param_name][value]
                    samples[value] = rng.betavariate(a, b)
                result.append(max(samples, key=samples.get))
            return result

        samples = {}
        for value in self.alpha[param_name]:
            a = self.alpha[param_name][value]
            b = self.beta[param_name][value]
            samples[value] = rng.betavariate(a, b)

        return max(samples, key=samples.get)

    def get_greedy_value(self, param_name: str) -> Any:
        """Get the value with highest estimated probability."""
        if param_name not in self.alpha:
            return None

        best_value = None
        best_prob = -1
        for value in self.alpha[param_name]:
            a = self.alpha[param_name][value]
            b = self.beta[param_name][value]
            prob = a / (a + b)
            if prob > best_prob:
                best_prob = prob
                best_value = value
        return best_value

    def get_ranked_values(self, param_name: str) -> list:
        """Get values sorted by estimated probability (best first)."""
        if param_name not in self.alpha:
            return []

        scored = []
        for value in self.alpha[param_name]:
            a = self.alpha[param_name][value]
            b = self.beta[param_name][value]
            scored.append((value, a / (a + b)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [v for v, _ in scored]


# ---------------------------------------------------------------------------
# Advanced Optimizer
# ---------------------------------------------------------------------------


class AdvancedOptimizer:
    """Hybrid optimizer combining multiple search strategies.

    Parameters
    ----------
    game : str
        Game identifier.
    param_space : ParameterSpace
        The search space.
    run_type : str
        API execution mode.
    n_islands : int
        Number of independent islands.
    island_size : int
        Population per island.
    n_generations : int
        Total generations.
    migration_interval : int
        Generations between migrations.
    migration_size : int
        Number of individuals to migrate.
    local_search_rounds : int
        Hill-climbing rounds after evolutionary phase.
    n_workers : int
        Parallel API calls.
    timeout_ms : int
        Server-side timeout.
    seed : int
        RNG seed.
    target_score : float | None
        Stop early if this score is reached.
    """

    def __init__(
        self,
        game: str,
        param_space: ParameterSpace,
        run_type: str = "fast",
        n_islands: int = 4,
        island_size: int = 15,
        n_generations: int = 50,
        migration_interval: int = 10,
        migration_size: int = 2,
        local_search_rounds: int = 20,
        n_workers: int = 8,
        timeout_ms: int = 0,
        seed: int | None = None,
        target_score: float | None = None,
    ) -> None:
        self.game = game
        self.param_space = param_space
        self.run_type = run_type
        self.n_islands = n_islands
        self.island_size = island_size
        self.n_generations = n_generations
        self.migration_interval = migration_interval
        self.migration_size = migration_size
        self.local_search_rounds = local_search_rounds
        self.n_workers = n_workers
        self.timeout_ms = timeout_ms
        self.target_score = target_score

        self._rng = random.Random(seed)
        self._importance = ParameterImportanceTracker(param_space)
        self._history: list[dict] = []
        self._all_evaluated: List[Individual] = []
        self._eval_count = 0
        self._global_best_score = float("-inf")
        self._global_best_params: dict = {}

    # ------------------------------------------------------------------
    # API evaluation
    # ------------------------------------------------------------------

    def _evaluate_individual(self, individual: Individual) -> Individual:
        """Evaluate an individual via the API."""
        try:
            score = api_client.run_game(
                game=self.game,
                params=individual.params,
                run_type=self.run_type,
                timeout_ms=self.timeout_ms,
                http_timeout=300.0,
            )
            individual.score = score
            individual.evaluated = True
            individual.evaluations += 1
            self._eval_count += 1
            if score > self._global_best_score:
                self._global_best_score = score
                self._global_best_params = copy.deepcopy(individual.params)
            logger.debug("Evaluated -> score=%.4f", score)
        except api_client.APIError as exc:
            logger.warning("APIError: %s", exc)
            individual.score = None
            individual.evaluated = True
        return individual

    def _evaluate_batch(self, individuals: List[Individual]) -> List[Individual]:
        """Evaluate a batch of individuals in parallel."""
        unevaluated = [ind for ind in individuals if not ind.evaluated]
        if not unevaluated:
            return individuals

        evaluated = []
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            futures = {
                executor.submit(self._evaluate_individual, ind): ind
                for ind in unevaluated
            }
            for future in as_completed(futures):
                result = future.result()
                evaluated.append(result)
                self._all_evaluated.append(result)

        return [ind for ind in individuals if ind.evaluated] + evaluated

    # ------------------------------------------------------------------
    # Island operations
    # ------------------------------------------------------------------

    def _create_island(self) -> Island:
        """Create a new island with random population."""
        population = [
            Individual(params=self.param_space.sample(self._rng))
            for _ in range(self.island_size)
        ]
        return Island(population=population)

    def _evolve_island(self, island: Island, generation: int) -> None:
        """One generation of evolution on an island."""
        island.population = self._evaluate_batch(island.population)

        scored = [ind for ind in island.population if ind.score is not None]
        if not scored:
            return

        scored.sort(key=lambda x: x.score, reverse=True)

        if scored[0].score > island.best_score:
            island.best_score = scored[0].score
            island.best_params = copy.deepcopy(scored[0].params)
            island.stagnation_count = 0
        else:
            island.stagnation_count += 1

        # Adaptive mutation
        if island.stagnation_count > 5:
            island.mutation_rate = min(0.8, island.mutation_rate + 0.05)
            island.mutation_strength = min(0.5, island.mutation_strength + 0.02)
        elif island.stagnation_count == 0:
            island.mutation_rate = max(0.1, island.mutation_rate - 0.02)
            island.mutation_strength = max(0.05, island.mutation_strength - 0.01)

        # Elitism - keep top 20%
        elite_count = max(1, int(self.island_size * 0.2))
        elites = [
            Individual(params=copy.deepcopy(ind.params), score=ind.score, evaluated=True)
            for ind in scored[:elite_count]
        ]

        # Create offspring
        offspring = []
        while len(offspring) < self.island_size - elite_count:
            parent_a = self._tournament_select(scored, k=3)
            parent_b = self._tournament_select(scored, k=3)
            child_params = self.param_space.crossover(parent_a.params, parent_b.params, self._rng)
            child_params = self._importance_guided_mutation(child_params, island)
            offspring.append(Individual(params=child_params))

        island.population = elites + offspring

        for ind in island.population:
            ind.age += 1

    def _tournament_select(self, population: List[Individual], k: int = 3) -> Individual:
        """Tournament selection."""
        candidates = self._rng.sample(population, min(k, len(population)))
        return max(candidates, key=lambda x: x.score if x.score is not None else float("-inf"))

    def _importance_guided_mutation(self, params: dict, island: Island) -> dict:
        """Mutate parameters with bias toward historically good values."""
        mutated = copy.deepcopy(params)
        param_names = list(mutated.keys())

        for name in param_names:
            if self._rng.random() < island.mutation_rate:
                # 50% chance to use importance sampling
                if self._rng.random() < 0.5:
                    mutated[name] = self._importance.sample_value(name, self._rng)
                else:
                    param = self.param_space.params[name]
                    if isinstance(param, CategoricalParam):
                        mutated[name] = self._rng.choice(param.choices)
                    elif isinstance(param, MultiCategoricalParam):
                        current_list = list(mutated[name])
                        for i in range(len(current_list)):
                            if self._rng.random() < 0.3:
                                current_list[i] = self._rng.choice(param.choices)
                        mutated[name] = current_list

        return mutated

    def _migrate(self, islands: List[Island]) -> None:
        """Migrate individuals between islands."""
        for i in range(len(islands)):
            scored = [ind for ind in islands[i].population if ind.score is not None]
            if not scored:
                continue
            scored.sort(key=lambda x: x.score, reverse=True)

            next_i = (i + 1) % len(islands)
            migrants = [
                Individual(params=copy.deepcopy(ind.params), score=ind.score, evaluated=True)
                for ind in scored[:self.migration_size]
            ]

            target_scored = [ind for ind in islands[next_i].population if ind.score is not None]
            if target_scored:
                target_scored.sort(key=lambda x: x.score)
                for j, migrant in enumerate(migrants):
                    if j < len(target_scored):
                        idx = islands[next_i].population.index(target_scored[j])
                        islands[next_i].population[idx] = migrant

    # ------------------------------------------------------------------
    # Local search (parallel hill climbing)
    # ------------------------------------------------------------------

    def _local_search(self, individual: Individual, max_rounds: int = 20) -> Individual:
        """Parallel hill climbing to refine a solution.

        For each parameter, generates a small set of candidate neighbors
        using importance-ranked values, evaluates them in parallel, then
        picks the best.  Limits to 2-3 candidates per parameter to keep
        evaluation time manageable.
        """
        current = Individual(
            params=copy.deepcopy(individual.params),
            score=individual.score,
            evaluated=True
        )

        for round_num in range(max_rounds):
            improved = False
            param_names = list(current.params.keys())
            self._rng.shuffle(param_names)

            all_candidates = []  # (param_name, candidate_individual)

            for name in param_names:
                param = self.param_space.params[name]

                if isinstance(param, CategoricalParam):
                    values_to_try = []
                    # Top value from importance tracker
                    ranked = self._importance.get_ranked_values(name)
                    for v in ranked:
                        if v != current.params[name]:
                            values_to_try.append(v)
                            break
                    # 1-2 random values
                    for _ in range(2):
                        v = self._rng.choice(param.choices)
                        if v not in values_to_try and v != current.params[name]:
                            values_to_try.append(v)
                    # Limit to 2 candidates max
                    values_to_try = values_to_try[:2]

                    for v in values_to_try:
                        neighbor_params = copy.deepcopy(current.params)
                        neighbor_params[name] = v
                        all_candidates.append((name, Individual(params=neighbor_params)))

                elif isinstance(param, MultiCategoricalParam):
                    # Generate 2 mutated variants
                    for _ in range(2):
                        neighbor_params = copy.deepcopy(current.params)
                        current_list = list(neighbor_params[name])
                        idx = self._rng.randint(0, len(current_list) - 1)
                        current_list[idx] = self._rng.choice(param.choices)
                        neighbor_params[name] = current_list
                        all_candidates.append((name, Individual(params=neighbor_params)))

            if not all_candidates:
                break

            # Evaluate ALL candidates in one parallel batch
            candidate_inds = [c[1] for c in all_candidates]
            evaluated = self._evaluate_batch(candidate_inds)

            # For each parameter, find the best candidate
            param_best: dict[str, Individual] = {}
            for (name, _), cand in zip(all_candidates, evaluated):
                if cand.score is None:
                    continue
                if name not in param_best or cand.score > param_best[name].score:
                    param_best[name] = cand

            # Apply improvements greedily (best first)
            improvements = []
            for name, best_cand in param_best.items():
                if best_cand.score > current.score:
                    improvements.append((name, best_cand))

            improvements.sort(key=lambda x: x[1].score, reverse=True)
            for name, best_cand in improvements:
                current.params[name] = best_cand.params[name]
                current.score = best_cand.score
                improved = True
                logger.debug(
                    "Local search round %d: improved %s -> score=%.4f",
                    round_num + 1, name, current.score,
                )

            if not improved:
                break

            logger.info("Local search round %d: score=%.4f", round_num + 1, current.score)

        return current

    # ------------------------------------------------------------------
    # Main optimization loop
    # ------------------------------------------------------------------

    def run(self) -> Individual:
        """Run the full optimization pipeline.

        Phases:
        1. Island-based evolutionary search
        2. Local search refinement on top candidates
        3. Final evaluation with "full" run_type
        """
        start_time = time.time()
        logger.info(
            "Starting advanced optimizer — game=%s | islands=%d | island_size=%d | generations=%d | target=%s",
            self.game, self.n_islands, self.island_size, self.n_generations,
            self.target_score if self.target_score else "none",
        )

        # Phase 1: Island-based evolution
        logger.info("=== Phase 1: Island-based Evolutionary Search ===")
        islands = [self._create_island() for _ in range(self.n_islands)]

        for gen in range(1, self.n_generations + 1):
            for island in islands:
                self._evolve_island(island, gen)

            if gen % self.migration_interval == 0:
                self._migrate(islands)
                logger.info("Generation %d: Migration completed", gen)

            # Update importance tracker with all scored individuals
            all_individuals = []
            for island in islands:
                for ind in island.population:
                    if ind.score is not None:
                        all_individuals.append(ind)
            self._importance.update_all(all_individuals)

            # Record stats
            best_scores = [island.best_score for island in islands if island.best_score > float("-inf")]
            if best_scores:
                gen_best = max(best_scores)
                gen_mean = statistics.mean(best_scores)
                self._history.append({
                    "generation": gen,
                    "best": gen_best,
                    "mean": gen_mean,
                    "island_bests": best_scores,
                    "evals": self._eval_count,
                })
                logger.info(
                    "Generation %3d | best=%.4f | mean=%.4f | evals=%d",
                    gen, gen_best, gen_mean, self._eval_count,
                )

                # Early termination check
                if self.target_score and gen_best >= self.target_score:
                    logger.info("Target score %.1f reached! (%.4f)", self.target_score, gen_best)
                    break

        # Collect best from all islands
        all_best = []
        for island in islands:
            scored = [ind for ind in island.population if ind.score is not None]
            if scored:
                all_best.extend(scored)

        if not all_best:
            logger.error("No successful evaluations!")
            return Individual(params=self.param_space.sample(self._rng), score=None)

        all_best.sort(key=lambda x: x.score, reverse=True)

        # Phase 2: Local search refinement
        logger.info("=== Phase 2: Local Search Refinement ===")
        top_candidates = all_best[:min(5, len(all_best))]
        refined = []

        for i, candidate in enumerate(top_candidates):
            logger.info("Refining candidate %d (score=%.4f)...", i + 1, candidate.score)
            refined_ind = self._local_search(candidate, max_rounds=self.local_search_rounds)
            refined.append(refined_ind)
            logger.info("Refined to score=%.4f", refined_ind.score)

            # Early termination check
            if self.target_score and refined_ind.score and refined_ind.score >= self.target_score:
                logger.info("Target score %.1f reached during local search!", self.target_score)
                break

        # Phase 3: Final evaluation with full run_type
        logger.info("=== Phase 3: Final Evaluation ===")
        best_refined = max(refined, key=lambda x: x.score if x.score is not None else float("-inf"))

        # Also consider global best from evolution
        if self._global_best_score > (best_refined.score or float("-inf")):
            logger.info(
                "Global best from evolution (%.4f) beats local search (%.4f), using it",
                self._global_best_score, best_refined.score,
            )
            best_refined = Individual(
                params=copy.deepcopy(self._global_best_params),
                score=self._global_best_score,
                evaluated=True,
            )

        # Re-evaluate best with full run_type for accuracy
        final_individual = Individual(
            params=copy.deepcopy(best_refined.params),
            evaluated=False
        )
        original_run_type = self.run_type
        self.run_type = "full"
        final_individual = self._evaluate_individual(final_individual)
        self.run_type = original_run_type

        if final_individual.score is not None:
            logger.info("Final score (full evaluation): %.4f", final_individual.score)
        else:
            logger.warning("Final evaluation failed, using best from local search")
            final_individual = best_refined

        elapsed = time.time() - start_time
        logger.info("Optimization complete in %.1f seconds. Total evaluations: %d", elapsed, self._eval_count)

        return final_individual

    def save_results(self, best: Individual, filename: str | None = None) -> str:
        """Save optimization results to JSON."""
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"results_advanced_{self.game}_{timestamp}.json"

        results = {
            "game": self.game,
            "optimizer": "advanced_hybrid",
            "best_score": best.score,
            "best_params": best.params,
            "total_evaluations": self._eval_count,
            "history": self._history,
            "importance": {
                name: {str(v): float(a / (a + b))
                       for v, a in self._importance.alpha[name].items()
                       for b in [self._importance.beta[name][v]]}
                for name in self._importance.alpha
            },
            "settings": {
                "n_islands": self.n_islands,
                "island_size": self.island_size,
                "n_generations": self.n_generations,
                "migration_interval": self.migration_interval,
                "local_search_rounds": self.local_search_rounds,
                "n_workers": self.n_workers,
                "run_type": self.run_type,
                "target_score": self.target_score,
            }
        }

        with open(filename, "w") as f:
            json.dump(results, f, indent=2)

        logger.info("Results saved to %s", filename)
        return filename


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main():
    import argparse
    from main import load_param_space_from_valid_params

    parser = argparse.ArgumentParser(
        description="Advanced hybrid optimizer for game parameters",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--game", required=True, choices=["Dominion", "ExplodingKittens", "Wonders7", "CantStop"])
    parser.add_argument("--islands", type=int, default=4, help="Number of islands")
    parser.add_argument("--island-size", type=int, default=15, help="Population per island")
    parser.add_argument("--generations", type=int, default=50, help="Number of generations")
    parser.add_argument("--migration-interval", type=int, default=10, help="Generations between migrations")
    parser.add_argument("--local-search", type=int, default=20, help="Local search rounds")
    parser.add_argument("--workers", type=int, default=8, help="Parallel API calls")
    parser.add_argument("--timeout-ms", type=int, default=0, help="Server timeout (ms)")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed")
    parser.add_argument("--target-score", type=float, default=None, help="Stop early if this score is reached")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    param_space = load_param_space_from_valid_params("valid_params.json", args.game)

    optimizer = AdvancedOptimizer(
        game=args.game,
        param_space=param_space,
        n_islands=args.islands,
        island_size=args.island_size,
        n_generations=args.generations,
        migration_interval=args.migration_interval,
        local_search_rounds=args.local_search,
        n_workers=args.workers,
        timeout_ms=args.timeout_ms,
        seed=args.seed,
        target_score=args.target_score,
    )

    best = optimizer.run()

    print("\n" + "=" * 60)
    print("BEST PARAMETERS FOUND")
    print("=" * 60)
    for k, v in best.params.items():
        print(f"  {k}: {v}")
    print("-" * 60)
    print(f"  SCORE: {best.score:.6f}" if best.score else "  SCORE: N/A")
    print("=" * 60)

    filename = optimizer.save_results(best)
    print(f"\nResults saved to: {filename}")


if __name__ == "__main__":
    main()
