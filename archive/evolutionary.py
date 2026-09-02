"""Evolutionary algorithm for game-parameter optimisation."""

from __future__ import annotations

import heapq
import logging
import random
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import api_client
from param_space import ParameterSpace

logger = logging.getLogger(__name__)


def _freeze(v):
    return tuple(v) if isinstance(v, list) else v


def _copy_params(params: dict) -> dict:
    return {k: (list(v) if isinstance(v, list) else v) for k, v in params.items()}


@dataclass
class Individual:
    params: dict
    score: float | None = None

    def is_evaluated(self) -> bool:
        return self.score is not None

    def cache_key(self) -> tuple:
        return tuple(sorted((k, _freeze(v)) for k, v in self.params.items()))


class EvolutionaryAlgorithm:
    """(mu + lambda) evolutionary algorithm with tournament selection, score caching,
    adaptive mutation decay, and stagnation-triggered diversity injection."""

    def __init__(
        self,
        game: str,
        param_space: ParameterSpace,
        run_type: str = "fast",
        pop_size: int = 20,
        n_generations: int = 30,
        mutation_rate: float = 0.3,
        mutation_strength: float = 0.2,
        elite_frac: float = 0.1,
        tournament_size: int = 3,
        n_workers: int = 2,
        timeout_ms: int = 0,
        seed: int | None = None,
        maximize: bool = True,
        stagnation_limit: int = 10,
        mutation_decay: float = 0.98,
    ) -> None:
        self.game = game
        self.param_space = param_space
        self.run_type = run_type
        self.pop_size = pop_size
        self.n_generations = n_generations
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.elite_frac = elite_frac
        self.tournament_size = tournament_size
        self.n_workers = n_workers
        self.timeout_ms = timeout_ms
        self.maximize = maximize
        self.stagnation_limit = stagnation_limit
        self.mutation_decay = mutation_decay

        self._rng = random.Random(seed)
        self._history: list[dict] = []
        self._score_cache: dict[tuple, float] = {}

    @property
    def history(self) -> list[dict]:
        return self._history

    def _is_better(self, score: float, reference: float) -> bool:
        return score > reference if self.maximize else score < reference

    def _worst_score(self) -> float:
        return float("-inf") if self.maximize else float("inf")

    def _score_key(self, individual: Individual) -> float:
        if individual.score is None:
            return self._worst_score()
        return individual.score

    def _select_best(self, population: list[Individual]) -> Individual:
        if self.maximize:
            return max(population, key=self._score_key)
        return min(population, key=self._score_key)

    def _evaluate_individual(self, individual: Individual, executor: ThreadPoolExecutor | None = None) -> Individual:
        try:
            score = api_client.run_game(
                game=self.game,
                params=individual.params,
                run_type=self.run_type,
                timeout_ms=self.timeout_ms,
            )
            individual.score = score
            self._score_cache[individual.cache_key()] = score
            logger.debug("Evaluated params %s -> score=%.4f", individual.params, score)
        except api_client.APIError as exc:
            logger.warning("API error for %s: %s", individual.params, exc)
            individual.score = None
        return individual

    def _evaluate_population(
        self, individuals: list[Individual], executor: ThreadPoolExecutor
    ) -> list[Individual]:
        unevaluated = [ind for ind in individuals if not ind.is_evaluated()]
        already_done = [ind for ind in individuals if ind.is_evaluated()]

        if not unevaluated:
            return individuals

        cached = []
        to_evaluate = []
        for ind in unevaluated:
            cache_key = ind.cache_key()
            if cache_key in self._score_cache:
                ind.score = self._score_cache[cache_key]
                cached.append(ind)
            else:
                to_evaluate.append(ind)

        if cached:
            logger.info("Found %d cached scores.", len(cached))

        if not to_evaluate:
            return already_done + cached

        logger.info("Evaluating %d individual(s) in parallel (workers=%d)", len(to_evaluate), self.n_workers)

        evaluated: list[Individual] = []
        futures = {executor.submit(self._evaluate_individual, ind): ind for ind in to_evaluate}
        for future in as_completed(futures):
            evaluated.append(future.result())

        return already_done + cached + evaluated

    def _tournament_select(self, population: list[Individual]) -> Individual:
        k = min(self.tournament_size, len(population))
        candidates = self._rng.sample(population, k)
        return self._select_best(candidates)

    def _next_generation(
        self, population: list[Individual], current_mutation_strength: float
    ) -> list[Individual]:
        elite_count = max(1, int(self.pop_size * self.elite_frac))

        if self.maximize:
            elites = heapq.nlargest(elite_count, population, key=self._score_key)
        else:
            elites = heapq.nsmallest(elite_count, population, key=self._score_key)

        elites = [
            Individual(params=_copy_params(ind.params), score=ind.score)
            for ind in elites
        ]

        offspring: list[Individual] = []
        while len(offspring) < self.pop_size - elite_count:
            parent_a = self._tournament_select(population)
            parent_b = self._tournament_select(population)
            child_params = self.param_space.crossover(parent_a.params, parent_b.params, self._rng)
            child_params = self.param_space.mutate(child_params, self._rng, self.mutation_rate, current_mutation_strength)
            offspring.append(Individual(params=child_params))

        return elites + offspring

    def _inject_diversity(self, population: list[Individual]) -> list[Individual]:
        keep_count = max(1, int(self.pop_size * 0.3))

        if self.maximize:
            kept = heapq.nlargest(keep_count, population, key=self._score_key)
        else:
            kept = heapq.nsmallest(keep_count, population, key=self._score_key)

        kept = [
            Individual(params=_copy_params(ind.params), score=ind.score)
            for ind in kept
        ]
        fresh = [
            Individual(params=self.param_space.sample(self._rng))
            for _ in range(self.pop_size - keep_count)
        ]
        logger.info("Stagnation: keeping top %d, injecting %d fresh individuals.", keep_count, len(fresh))
        return kept + fresh

    def _record_generation(self, generation: int, population: list[Individual]) -> None:
        scored = [ind.score for ind in population if ind.score is not None]

        if not scored:
            logger.warning("Generation %3d: no scored individuals.", generation)
            return

        best = max(scored) if self.maximize else min(scored)
        mean = statistics.mean(scored)
        std = statistics.stdev(scored) if len(scored) > 1 else 0.0

        logger.info(
            "Generation %3d | best=%10.4f | mean=%10.4f | std=%9.4f | scored=%d/%d | cache=%d",
            generation, best, mean, std, len(scored), len(population), len(self._score_cache),
        )

        self._history.append({"generation": generation, "best": best, "mean": mean, "std": std})

    def run(self) -> Individual:
        logger.info(
            "Starting EA: game=%s run_type=%s pop=%d gens=%d maximize=%s",
            self.game, self.run_type, self.pop_size, self.n_generations, self.maximize,
        )

        population = [
            Individual(params=self.param_space.sample(self._rng))
            for _ in range(self.pop_size)
        ]

        global_best_score = self._worst_score()
        global_best: Individual | None = None
        stagnation_counter = 0
        current_strength = self.mutation_strength

        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            for generation in range(1, self.n_generations + 1):
                population = self._evaluate_population(population, executor)
                self._record_generation(generation, population)

                gen_best = self._select_best(population)
                if gen_best.score is not None:
                    if self._is_better(gen_best.score, global_best_score):
                        global_best_score = gen_best.score
                        global_best = Individual(params=_copy_params(gen_best.params), score=gen_best.score)
                        stagnation_counter = 0
                        logger.info("New global best: %.4f at generation %d", global_best_score, generation)
                    else:
                        stagnation_counter += 1

                current_strength = max(0.01, current_strength * self.mutation_decay)

                if stagnation_counter >= self.stagnation_limit:
                    population = self._inject_diversity(population)
                    stagnation_counter = 0
                    current_strength = self.mutation_strength
                else:
                    population = self._next_generation(population, current_strength)

            logger.info("Running final evaluation pass...")
            population = self._evaluate_population(population, executor)

        best = self._select_best(population)

        if global_best is not None and global_best.score is not None:
            if self._is_better(global_best.score, best.score if best.score is not None else self._worst_score()):
                best = global_best

        logger.info(
            "EA complete. Best score=%.4f | params=%s",
            best.score if best.score is not None else float("nan"),
            best.params,
        )
        return best
