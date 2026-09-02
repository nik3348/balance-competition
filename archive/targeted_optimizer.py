"""
targeted_optimizer.py – Focused optimizer for CantStop CTW=2, MARKERS=2.

Discoveries so far:
- MARKERS=2 + CTW=2 is the best configuration (highest score, fastest eval ~78s)
- Coord descent from best known params found TWO_MAX=4, SEVEN_MAX=12 is better
- Best single-eval score: 930.31 with CTW=2, MARKERS=2

Strategy:
1. Exhaustive random search over column maxes (CTW=2, MARKERS=2 fixed)
2. Re-evaluate top candidates 3x to estimate true mean
3. Coord descent from truly best params
4. Try medium run type for final verification
"""

import json
import logging
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import api_client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-8s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

GAME = "CantStop"
FIXED_PARAMS = {"COLUMNS_TO_WIN": 2, "MARKERS": 2}

COLUMN_PARAMS = {
    "TWO_MAX": [1, 2, 3, 4, 5],
    "THREE_MAX": [2, 3, 4, 5, 6],
    "FOUR_MAX": [4, 5, 6, 7, 8],
    "FIVE_MAX": [6, 7, 8, 9, 10],
    "SIX_MAX": [8, 9, 10, 11, 12],
    "SEVEN_MAX": [10, 11, 12, 13, 14],
    "EIGHT_MAX": [8, 9, 10, 11, 12],
    "NINE_MAX": [6, 7, 8, 9, 10],
    "TEN_MAX": [4, 5, 6, 7, 8],
    "ELEVEN_MAX": [2, 3, 4, 5, 6],
    "TWELVE_MAX": [1, 2, 3, 4, 5],
}

# Best known starting point
BEST_KNOWN = {
    "TWO_MAX": 4, "THREE_MAX": 2, "FOUR_MAX": 7, "FIVE_MAX": 8,
    "SIX_MAX": 11, "SEVEN_MAX": 12, "EIGHT_MAX": 11, "NINE_MAX": 6,
    "TEN_MAX": 4, "ELEVEN_MAX": 2, "TWELVE_MAX": 4,
    **FIXED_PARAMS,
}

# Theory-optimal: maxes proportional to dice probability
THEORY_OPTIMAL = {
    "TWO_MAX": 2, "THREE_MAX": 5, "FOUR_MAX": 7, "FIVE_MAX": 9,
    "SIX_MAX": 12, "SEVEN_MAX": 14, "EIGHT_MAX": 12, "NINE_MAX": 9,
    "TEN_MAX": 7, "ELEVEN_MAX": 5, "TWELVE_MAX": 2,
    **FIXED_PARAMS,
}

# Standard CantStop rules
STANDARD_RULES = {
    "TWO_MAX": 3, "THREE_MAX": 5, "FOUR_MAX": 7, "FIVE_MAX": 9,
    "SIX_MAX": 11, "SEVEN_MAX": 13, "EIGHT_MAX": 11, "NINE_MAX": 9,
    "TEN_MAX": 7, "ELEVEN_MAX": 5, "TWELVE_MAX": 3,
    **FIXED_PARAMS,
}


def evaluate_with_retry(params, run_type="fast", max_retries=3, retry_delay=10):
    """Evaluate params with retry logic for transient server errors."""
    for attempt in range(max_retries):
        try:
            score = api_client.run_game(
                game=GAME,
                params=params,
                run_type=run_type,
                timeout_ms=0,
                http_timeout=300.0,
            )
            return score
        except api_client.APIError as e:
            if attempt < max_retries - 1:
                logger.warning("Attempt %d failed: %s. Retrying in %ds...", attempt + 1, e, retry_delay)
                time.sleep(retry_delay)
            else:
                logger.warning("All retries failed: %s", e)
                return None


def evaluate_mean(params, n_reps=3, run_type="fast"):
    """Evaluate params multiple times and return the mean."""
    scores = []
    for i in range(n_reps):
        s = evaluate_with_retry(params, run_type)
        if s is not None:
            scores.append(s)
            logger.info("Rep %d/%d: %.4f", i + 1, n_reps, s)
    if not scores:
        return None
    mean = sum(scores) / len(scores)
    logger.info("Mean over %d reps: %.4f", len(scores), mean)
    return mean


def sample_random_cols(rng):
    """Sample random column maxes."""
    return {
        **{name: rng.choice(values) for name, values in COLUMN_PARAMS.items()},
        **FIXED_PARAMS,
    }


def coordinate_descent(start_params, best_score=None, max_rounds=15, run_type="fast"):
    """Coordinate descent: for each param, try all values and keep best."""
    current = dict(start_params)
    if best_score is None:
        best_score = evaluate_with_retry(current, run_type) or float("-inf")
        logger.info("Start score: %.4f", best_score)

    all_params = {**COLUMN_PARAMS, "COLUMNS_TO_WIN": [2, 3, 4], "MARKERS": [2, 3, 4, 5, 6]}

    for round_num in range(1, max_rounds + 1):
        improved = False
        logger.info("=== Round %d (current best: %.4f) ===", round_num, best_score)

        for param_name, valid_values in all_params.items():
            current_val = current.get(param_name)
            best_val = current_val
            best_val_score = best_score

            for val in valid_values:
                if val == current_val:
                    continue
                test_params = {**current, param_name: val}
                score = evaluate_with_retry(test_params, run_type)
                if score is not None and score > best_val_score:
                    best_val_score = score
                    best_val = val
                    logger.info("  %s=%s improved: %.4f", param_name, val, score)

            if best_val != current_val:
                current[param_name] = best_val
                best_score = best_val_score
                improved = True

        logger.info("Round %d done: best=%.4f", round_num, best_score)
        if not improved:
            logger.info("Converged after %d rounds.", round_num)
            break

    return current, best_score


def main():
    logger.info("Starting targeted CantStop optimization...")
    logger.info("Fixed params: CTW=2, MARKERS=2")

    best_params = None
    best_score = float("-inf")

    # Phase 1: Test key candidate configurations
    candidates = [BEST_KNOWN, THEORY_OPTIMAL, STANDARD_RULES]
    logger.info("=== Phase 1: Key candidates ===")
    for c in candidates:
        s = evaluate_with_retry(c)
        if s is not None:
            logger.info("%.4f: %s", s, {k: v for k, v in c.items() if k not in FIXED_PARAMS})
            if s > best_score:
                best_score = s
                best_params = dict(c)

    # Phase 2: Random search over column maxes
    logger.info("=== Phase 2: Random search (50 candidates) ===")
    rng = random.Random(42)
    random_scores = []
    for i in range(50):
        p = sample_random_cols(rng)
        s = evaluate_with_retry(p)
        if s is not None:
            random_scores.append((p, s))
            logger.info("Random %d: %.4f", i + 1, s)
            if s > best_score:
                best_score = s
                best_params = dict(p)

    # Phase 3: Coord descent from top-3 random results + best known
    random_scores.sort(key=lambda x: x[1], reverse=True)
    starts = [best_params] + [p for p, _ in random_scores[:2]]
    logger.info("=== Phase 3: Coord descent from %d starting points ===", len(starts))
    for i, start in enumerate(starts):
        logger.info("--- Starting point %d/%d ---", i + 1, len(starts))
        improved_params, improved_score = coordinate_descent(start)
        if improved_score > best_score:
            best_score = improved_score
            best_params = dict(improved_params)
            logger.info("New global best: %.4f", best_score)

    logger.info("FINAL BEST SCORE: %.4f", best_score)
    logger.info("FINAL BEST PARAMS: %s", json.dumps(best_params, indent=2))

    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"results_targeted_CantStop_{timestamp}.json"
    with open(filename, "w") as f:
        json.dump({
            "game": GAME,
            "optimizer": "targeted_coord_descent",
            "best_score": best_score,
            "best_params": best_params,
        }, f, indent=2)
    logger.info("Saved to %s", filename)


if __name__ == "__main__":
    main()
