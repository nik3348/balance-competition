"""
opt.py – Parallel coordinate descent optimizer.

Key improvements:
- Evaluates all candidate values in parallel per parameter
- Handles multi-categorical params (Dominion CARDS, Wonders7 wonders)
- Uses retry logic for timeouts
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

MULTI_CATEGORICAL = {
    ("Dominion", "CARDS"): 10,
    ("Wonders7", "wonders"): 7,
}

BEST_KNOWN = {
    "CantStop": {
        "TWO_MAX": 5, "THREE_MAX": 2, "FOUR_MAX": 6, "FIVE_MAX": 7,
        "SIX_MAX": 9, "SEVEN_MAX": 11, "EIGHT_MAX": 8, "NINE_MAX": 10,
        "TEN_MAX": 4, "ELEVEN_MAX": 4, "TWELVE_MAX": 4,
        "COLUMNS_TO_WIN": 2, "MARKERS": 2,
    },
    "ExplodingKittens": {
        "nCardsPerPlayer": 3, "nopeOwnCards": False,
        "ATTACK_count": 2, "SKIP_count": 3, "FAVOR_count": 8,
        "SHUFFLE_count": 5, "SEETHEFUTURE_count": 4, "TACOCAT_count": 2,
        "MELONCAT_count": 3, "BEARDCAT_count": 5, "RAINBOWCAT_count": 6,
        "FURRYCAT_count": 8, "NOPE_count": 9, "DEFUSE_count": 7,
    },
    "Dominion": {
        "HAND_SIZE": 3, "PILES_EXHAUSTED_FOR_GAME_END": 3,
        "KINGDOM_CARDS_OF_EACH_TYPE": 10, "CURSE_CARDS_PER_PLAYER": 5,
        "STARTING_COPPER": 5, "STARTING_ESTATES": 7,
        "COPPER_SUPPLY": 40, "SILVER_SUPPLY": 30, "GOLD_SUPPLY": 50,
        "CARDS": ["WORKSHOP", "MARKET", "MILITIA", "WITCH", "WITCH",
                  "CHAPEL", "WORKSHOP", "CHAPEL", "ARTISAN", "MILITIA"],
    },
    "Wonders7": {
        "nCostNeighbourResource": 2, "nCostDiscountedResource": 2,
        "nCoinsDiscard": 2, "startingCoins": 3, "rawMaterialLow": 2,
        "rawMaterialHigh": 3, "manufacturedMaterial": 2, "victoryLow": 2,
        "victoryMed": 3, "victoryHigh": 4, "victoryVeryHigh": 5,
        "victoryPantheon": 6, "victoryPalace": 7, "tavernMoney": 4,
        "wildcardProduction": 2, "commercialMultiplierLow": 2,
        "commercialMultiplierMed": 3, "commercialMultiplierHigh": 3,
        "militaryLow": 2, "militaryMed": 3, "militaryHigh": 3,
        "scienceCompass": 2, "scienceTablet": 2, "scienceCog": 2,
        "guildMultiplierLow": 2, "guildMultiplierMed": 3,
        "builderMultiplier": 2, "decoratorVictoryPoints": 6,
        "wonders": ["TheColossusOfRhodes", "TheLighthouseOfAlexandria",
                    "TheTempleOfArtemisInEphesus", "TheHangingGardensOfBabylon",
                    "TheStatueOfZeusInOlympia", "TheMausoleumOfHalicarnassus",
                    "ThePyramidsOfGiza"],
    },
}


def load_valid_params(game: str) -> dict:
    with open("valid_params.json") as f:
        return json.load(f)[game]


def sample_random(game: str, valid_params: dict, rng: random.Random) -> dict:
    params = {}
    for name, values in valid_params.items():
        key = (game, name)
        if key in MULTI_CATEGORICAL:
            count = MULTI_CATEGORICAL[key]
            params[name] = [rng.choice(values) for _ in range(count)]
        else:
            params[name] = rng.choice(values)
    return params


def eval_with_retry(game: str, params: dict, run_type: str, max_retries: int = 2) -> float | None:
    for attempt in range(max_retries):
        try:
            return api_client.run_game(
                game=game, params=params, run_type=run_type,
                timeout_ms=0, http_timeout=600.0,
            )
        except api_client.APIError:
            if attempt < max_retries - 1:
                time.sleep(2)
    return None


def coord_descent_parallel(
    game: str, start_params: dict, valid_params: dict,
    run_type: str = "fast", max_rounds: int = 10, n_workers: int = 4,
) -> tuple[dict, float]:
    """Coordinate descent with parallel evaluation per parameter."""
    current = dict(start_params)
    current_score = eval_with_retry(game, current, run_type)
    if current_score is None:
        return current, float("-inf")
    logger.info("Coord descent start: %.4f", current_score)
    rng = random.Random(42)

    for round_num in range(1, max_rounds + 1):
        improved = False
        logger.info("=== Round %d (best: %.4f) ===", round_num, current_score)

        for param_name, valid_values in valid_params.items():
            key = (game, param_name)
            if key in MULTI_CATEGORICAL:
                # For multi-categorical, try mutating a few elements
                count = MULTI_CATEGORICAL[key]
                for idx in range(min(count, 5)):  # Only try first 5 elements per round
                    candidates = []
                    for val in rng.sample(valid_values, min(4, len(valid_values))):
                        if val == current[param_name][idx]:
                            continue
                        new_list = list(current[param_name])
                        new_list[idx] = val
                        candidates.append((val, {**current, param_name: new_list}))

                    if not candidates:
                        continue

                    best_val = current[param_name][idx]
                    best_score = current_score

                    with ThreadPoolExecutor(max_workers=n_workers) as executor:
                        future_map = {
                            executor.submit(eval_with_retry, game, p, run_type): val
                            for val, p in candidates
                        }
                        for f in as_completed(future_map):
                            val = future_map[f]
                            score = f.result()
                            if score is not None and score > best_score:
                                best_score = score
                                best_val = val

                    if best_val != current[param_name][idx]:
                        new_list = list(current[param_name])
                        new_list[idx] = best_val
                        current[param_name] = new_list
                        current_score = best_score
                        improved = True
                        logger.info("  %s[%d]=%s -> %.4f", param_name, idx, best_val, current_score)
            else:
                # Regular categorical: try all values in parallel
                candidates = [(v, {**current, param_name: v}) for v in valid_values if v != current[param_name]]
                if not candidates:
                    continue

                best_val = current[param_name]
                best_score = current_score

                with ThreadPoolExecutor(max_workers=n_workers) as executor:
                    future_map = {
                        executor.submit(eval_with_retry, game, p, run_type): val
                        for val, p in candidates
                    }
                    for f in as_completed(future_map):
                        val = future_map[f]
                        score = f.result()
                        if score is not None and score > best_score:
                            best_score = score
                            best_val = val

                if best_val != current[param_name]:
                    current[param_name] = best_val
                    current_score = best_score
                    improved = True
                    logger.info("  %s=%s -> %.4f", param_name, best_val, current_score)

        logger.info("Round %d done: best=%.4f", round_num, current_score)
        if not improved:
            logger.info("Converged after %d rounds.", round_num)
            break

    return current, current_score


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", required=True, choices=["CantStop", "Dominion", "ExplodingKittens", "Wonders7"])
    parser.add_argument("--run-type", default="fast")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--n-random", type=int, default=3)
    parser.add_argument("--n-verify", type=int, default=10)
    parser.add_argument("--max-rounds", type=int, default=5)
    args = parser.parse_args()

    game = args.game
    valid_params = load_valid_params(game)
    rng = random.Random(42)

    best_params = None
    best_score = float("-inf")

    # Evaluate best known
    if game in BEST_KNOWN:
        logger.info("--- Evaluating best known ---")
        score = eval_with_retry(game, BEST_KNOWN[game], args.run_type)
        if score is not None:
            logger.info("Best known: %.4f", score)
            best_score = score
            best_params = dict(BEST_KNOWN[game])

    # Random search
    logger.info("--- Random search (%d samples) ---", args.n_random)
    for i in range(args.n_random):
        params = sample_random(game, valid_params, rng)
        score = eval_with_retry(game, params, args.run_type)
        if score is not None:
            logger.info("  Random %d: %.4f", i + 1, score)
            if score > best_score:
                best_score = score
                best_params = dict(params)

    if best_params is None:
        logger.error("No valid params found!")
        return

    # Coordinate descent
    logger.info("--- Coord descent (max %d rounds) ---", args.max_rounds)
    params, score = coord_descent_parallel(
        game, best_params, valid_params, args.run_type,
        max_rounds=args.max_rounds, n_workers=args.workers,
    )
    if score > best_score:
        best_score = score
        best_params = dict(params)

    # Final verification
    logger.info("--- Final verification (%d reps) ---", args.n_verify)
    scores = []
    for i in range(args.n_verify):
        s = eval_with_retry(game, best_params, args.run_type)
        if s is not None:
            scores.append(s)
            logger.info("  Rep %d: %.4f", i + 1, s)
    mean_score = sum(scores) / len(scores) if scores else float("-inf")

    logger.info("FINAL: best=%.4f verified_mean=%.4f (over %d reps)", best_score, mean_score, len(scores))
    logger.info("Params: %s", json.dumps(best_params, indent=2))

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"results_final_{game}_{timestamp}.json"
    with open(filename, "w") as f:
        json.dump({
            "game": game, "best_score": best_score, "verified_mean": mean_score,
            "best_params": best_params, "all_scores": scores,
        }, f, indent=2)
    logger.info("Saved to %s", filename)


if __name__ == "__main__":
    main()
