"""
evolutionary.py – evolutionary algorithm for game-parameter optimisation.

Uses only the standard library plus the local api_client and param_space modules.
"""

from __future__ import annotations

import copy
import logging
import random
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import List, Optional

import api_client
from param_space import ParameterSpace

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Individual
# ---------------------------------------------------------------------------


@dataclass
class Individual:
    """A single candidate solution."""

    params: dict
    score: float | None = None  # None = unevaluated

    def is_evaluated(self) -> bool:
        return self.score is not None


# ---------------------------------------------------------------------------
# Evolutionary algorithm
# ---------------------------------------------------------------------------


class EvolutionaryAlgorithm:
    """A (μ + λ)-style evolutionary algorithm with tournament selection.

    Parameters
    ----------
    game:               Game identifier forwarded to the API.
    param_space:        Defines the search space (sampling, mutation, crossover).
    run_type:           Execution mode passed to the API ("fast", "medium", "full").
    pop_size:           Number of individuals in the population.
    n_generations:      How many generations to evolve.
    mutation_rate:      Probability that each parameter is mutated (0–1).
    mutation_strength:  Magnitude of mutations relative to the parameter range.
    elite_frac:         Fraction of the population kept unchanged each generation.
    tournament_size:    Number of candidates drawn in each tournament selection.
    n_workers:          Maximum number of parallel API calls.
    timeout_ms:         Per-game server-side timeout in milliseconds (0 = none).
    seed:               Optional RNG seed for reproducibility.
    maximize:           If True (default), higher scores are better.
    """

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
        n_workers: int = 4,
        timeout_ms: int = 0,
        seed: int | None = None,
        maximize: bool = True,
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

        self._rng = random.Random(seed)
        self._history: list[dict] = []

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def history(self) -> list[dict]:
        """Generation-by-generation stats populated during :meth:`run`.

        Each entry is a dict with keys ``generation``, ``best``, ``mean``,
        and ``std``.
        """
        return self._history

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _score_key(self, individual: Individual) -> float:
        """Numeric sort key that places unevaluated individuals last.

        For maximisation returns the score directly (−∞ for *None*).
        For minimisation returns the score directly (+∞ for *None*).
        When used with ``reverse=self.maximize``, the *best* individual
        always sorts to the front.
        """
        if individual.score is None:
            return float("-inf") if self.maximize else float("inf")
        return individual.score

    def _evaluate_individual(self, individual: Individual) -> Individual:
        """Call the API for *individual* and store the result on the object.

        On :class:`api_client.APIError` the score is left as *None* and a
        warning is logged; the individual will simply lose every tournament.
        """
        try:
            score = api_client.run_game(
                game=self.game,
                params=individual.params,
                run_type=self.run_type,
                timeout_ms=self.timeout_ms,
            )
            individual.score = score
            logger.debug("Evaluated params %s → score=%.4f", individual.params, score)
        except api_client.APIError as exc:
            logger.warning(
                "APIError evaluating params %s: %s — individual will be treated as worst.",
                individual.params,
                exc,
            )
            individual.score = None
        return individual

    # ------------------------------------------------------------------
    # Core EA operations
    # ------------------------------------------------------------------

    def _evaluate_population(self, individuals: list[Individual]) -> list[Individual]:
        """Evaluate every unevaluated individual in *individuals* in parallel.

        Already-evaluated individuals are returned as-is.  Evaluation is
        dispatched via :class:`~concurrent.futures.ThreadPoolExecutor` with
        up to ``self.n_workers`` concurrent API calls.
        """
        unevaluated = [ind for ind in individuals if not ind.is_evaluated()]
        already_done = [ind for ind in individuals if ind.is_evaluated()]

        if not unevaluated:
            logger.debug(
                "All %d individuals already evaluated; skipping.", len(individuals)
            )
            return individuals

        logger.info(
            "Evaluating %d individual(s) in parallel (workers=%d) …",
            len(unevaluated),
            self.n_workers,
        )

        evaluated: list[Individual] = []
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            future_to_ind = {
                executor.submit(self._evaluate_individual, ind): ind
                for ind in unevaluated
            }
            for future in as_completed(future_to_ind):
                result = future.result()  # _evaluate_individual never raises
                evaluated.append(result)

        return already_done + evaluated

    def _tournament_select(self, population: list[Individual]) -> Individual:
        """Draw *tournament_size* candidates at random; return the best one.

        Individuals with ``score=None`` are treated as the worst possible
        competitor and will only win if every candidate is unevaluated.
        """
        k = min(self.tournament_size, len(population))
        candidates = self._rng.sample(population, k)
        if self.maximize:
            return max(candidates, key=self._score_key)
        # minimise: lowest real score wins; None is worst (+inf)
        return min(candidates, key=self._score_key)

    def _next_generation(self, population: list[Individual]) -> list[Individual]:
        """Produce the next generation from *population*.

        Strategy
        --------
        1. Sort by score, best first (``None`` last).
        2. Preserve the top ``elite_count`` individuals unchanged (deep copy).
        3. Fill the remaining slots with offspring created by:
           tournament selection → uniform crossover → parameter mutation.
        """
        # Sort best-first; reverse=True for maximise, False for minimise
        sorted_pop = sorted(population, key=self._score_key, reverse=self.maximize)

        elite_count = max(1, int(self.pop_size * self.elite_frac))
        elites = [
            Individual(params=copy.deepcopy(ind.params), score=ind.score)
            for ind in sorted_pop[:elite_count]
        ]

        offspring: list[Individual] = []
        while len(offspring) < self.pop_size - elite_count:
            parent_a = self._tournament_select(population)
            parent_b = self._tournament_select(population)
            child_params = self.param_space.crossover(
                parent_a.params, parent_b.params, self._rng
            )
            child_params = self.param_space.mutate(
                child_params,
                self._rng,
                self.mutation_rate,
                self.mutation_strength,
            )
            offspring.append(Individual(params=child_params))

        next_pop = elites + offspring
        logger.debug(
            "Next generation: %d elites + %d offspring = %d total.",
            len(elites),
            len(offspring),
            len(next_pop),
        )
        return next_pop

    def _record_generation(self, generation: int, population: list[Individual]) -> None:
        """Compute and log per-generation stats; append an entry to history."""
        scored = [ind.score for ind in population if ind.score is not None]

        if not scored:
            logger.warning(
                "Generation %3d: no successfully evaluated individuals — "
                "skipping stats.",
                generation,
            )
            return

        best = max(scored) if self.maximize else min(scored)
        mean = statistics.mean(scored)
        std = statistics.stdev(scored) if len(scored) > 1 else 0.0

        logger.info(
            "Generation %3d | best=%10.4f | mean=%10.4f | std=%9.4f | scored=%d/%d",
            generation,
            best,
            mean,
            std,
            len(scored),
            len(population),
        )

        self._history.append(
            {
                "generation": generation,
                "best": best,
                "mean": mean,
                "std": std,
            }
        )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self) -> Individual:
        """Run the full evolutionary algorithm and return the best individual.

        Flow
        ----
        1. Initialise a random population of size ``pop_size``.
        2. For each generation:

           a. Evaluate all unevaluated individuals (parallel API calls).
           b. Log/record generation statistics.
           c. Produce the next generation via elitism + crossover + mutation.

        3. Perform one final evaluation pass on the last offspring cohort.
        4. Return the individual with the best score.

        The :attr:`history` list is populated with one entry per generation.
        """
        logger.info(
            "Starting evolutionary algorithm — game=%s | run_type=%s"
            " | pop_size=%d | generations=%d | maximize=%s",
            self.game,
            self.run_type,
            self.pop_size,
            self.n_generations,
            self.maximize,
        )

        # Initialise population
        population: list[Individual] = [
            Individual(params=self.param_space.sample(self._rng))
            for _ in range(self.pop_size)
        ]
        logger.info("Initialised population with %d random individuals.", self.pop_size)

        for generation in range(1, self.n_generations + 1):
            # Step 1 – evaluate
            population = self._evaluate_population(population)

            # Step 2 – log stats
            self._record_generation(generation, population)

            # Step 3 – breed next generation
            population = self._next_generation(population)

        # Final evaluation pass on the last offspring cohort
        logger.info("Running final evaluation pass on last offspring cohort …")
        population = self._evaluate_population(population)

        # Select the best individual
        if self.maximize:
            best = max(population, key=self._score_key)
        else:
            best = min(population, key=self._score_key)

        logger.info(
            "Evolutionary algorithm complete. Best score=%.4f | params=%s",
            best.score if best.score is not None else float("nan"),
            best.params,
        )
        return best
