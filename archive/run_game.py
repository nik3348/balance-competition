"""
run_game.py – Run optimization for a single game.

Usage:
  uv run python run_game.py --game Wonders7
"""

import json
import logging
import random
import sys
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
    "Dominion": {
        "HAND_SIZE": 5, "PILES_EXHAUSTED_FOR_GAME_END": 3,
        "KINGDOM_CARDS_OF_EACH_TYPE": 10, "CURSE_CARDS_PER_PLAYER": 10,
        "STARTING_COPPER": 7, "STARTING_ESTATES": 3,
        "COPPER_SUPPLY": 32, "SILVER_SUPPLY": 30, "GOLD_SUPPLY": 30,
        "CARDS": ["VILLAGE", "SMITHY", "FESTIVAL", "LABORATORY", "MARKET",
                  "MILITIA", "WITCH", "MOAT", "MERCHANT", "HARBINGER"],
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

# Multi-categorical: (game, param_name) -> count
MULTI_CATEGORICAL = {
    ("Dominion", "CARDS"): 10,
    ("Wonders7", "wonders"): 7,
}


def load_valid_params(game: str) -> dict:
    with open("valid_params.json") as f:
        return json.load(f)[game]


def sample_random(game: str, valid_params: dict, rng: random.Random) -> dict:
    """Sample random params, handling multi-categorical correctly."""
    params = {}
    for name, values in valid_params.items():
        key = (game, name)
        if key in MULTI_CATEGORICAL:
            count = MULTI_CATEGORICAL[key]
            params[name] = [rng.choice(values) for _ in range(count)]
        else:
            params[name] = rng.choice(values)
    return params


def eval_single(game: str, params: dict, run_type: str = "fast") -> float | None:
    try:
        return api_client.run_game(game=game, params=params, run_type=run_type, timeout_ms=0, http_timeout=600.0)
    except api_client.APIError:
        return None


def coord_descent(game: str, start_params: dict, valid_params: dict, run_type: str = "fast", max_rounds: int = 20) -> tuple[dict, float]:
    """Coordinate descent: for each param, try ALL values sequentially, keep best."""
    current = dict(start_params)
    current_score = eval_single(game, current, run_type)
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
                # For multi-categorical, try replacing each element
                count = MULTI_CATEGORICAL[key]
                for idx in range(count):
                    best_val = current[param_name][idx]
                    best_val_score = current_score
                    # Try a subset of values (not all, to save time)
                    vals_to_try = rng.sample(valid_values, min(5, len(valid_values)))
                    for val in vals_to_try:
                        if val == current[param_name][idx]:
                            continue
                        test_params = {**current}
                        new_list = list(current[param_name])
                        new_list[idx] = val
                        test_params[param_name] = new_list
                        score = eval_single(game, test_params, run_type)
                        if score is not None and score > best_val_score:
                            best_val_score = score
                            best_val = val
                            logger.info("  %s[%d]=%s -> %.4f", param_name, idx, val, score)
                    if best_val != current[param_name][idx]:
                        current[param_name] = list(current[param_name])
                        current[param_name][idx] = best_val
                        current_score = best_val_score
                        improved = True
            else:
                best_val = current[param_name]
                best_val_score = current_score
                for val in valid_values:
                    if val == current[param_name]:
                        continue
                    test_params = {**current, param_name: val}
                    score = eval_single(game, test_params, run_type)
                    if score is not None and score > best_val_score:
                        best_val_score = score
                        best_val = val
                        logger.info("  %s=%s -> %.4f", param_name, val, score)
                if best_val != current[param_name]:
                    current[param_name] = best_val
                    current_score = best_val_score
                    improved = True

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
    parser.add_argument("--n-random", type=int, default=5)
    parser.add_argument("--n-verify", type=int, default=10)
    args = parser.parse_args()

    game = args.game
    valid_params = load_valid_params(game)
    rng = random.Random(42)

    best_params = None
    best_score = float("-inf")

    # Start from best known
    if game in BEST_KNOWN:
        logger.info("--- Evaluating best known ---")
        score = eval_single(game, BEST_KNOWN[game], args.run_type)
        if score is not None:
            logger.info("Best known: %.4f", score)
            if score > best_score:
                best_score = score
                best_params = dict(BEST_KNOWN[game])

    # Random search
    logger.info("--- Random search (%d samples) ---", args.n_random)
    for i in range(args.n_random):
        params = sample_random(game, valid_params, rng)
        score = eval_single(game, params, args.run_type)
        if score is not None:
            logger.info("  Random %d: %.4f", i + 1, score)
            if score > best_score:
                best_score = score
                best_params = dict(params)

    if best_params is None:
        logger.error("No valid params found! Exiting.")
        return

    # Coord descent from best known
    logger.info("--- Coord descent from best ---")
    params, score = coord_descent(game, best_params, valid_params, args.run_type, max_rounds=3)
    if score > best_score:
        best_score = score
        best_params = dict(params)

    # Second coord descent pass
    logger.info("--- Coord descent pass 2 ---")
    params, score = coord_descent(game, best_params, valid_params, args.run_type, max_rounds=3)
    if score > best_score:
        best_score = score
        best_params = dict(params)

    # Final verification
    logger.info("--- Final verification (%d reps) ---", args.n_verify)
    scores = []
    for i in range(args.n_verify):
        s = eval_single(game, best_params, args.run_type)
        if s is not None:
            scores.append(s)
            logger.info("  Rep %d: %.4f", i + 1, s)
    mean_score = sum(scores) / len(scores) if scores else float("-inf")

    logger.info("FINAL BEST: %.4f (verified mean: %.4f over %d reps)", best_score, mean_score, len(scores))
    logger.info("Params: %s", json.dumps(best_params, indent=2))

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"results_final_{game}_{timestamp}.json"
    with open(filename, "w") as f:
        json.dump({
            "game": game, "best_score": best_score, "verified_mean": mean_score,
            "best_params": best_params, "n_reps": len(scores),
            "all_scores": scores,
        }, f, indent=2)
    logger.info("Saved to %s", filename)


if __name__ == "__main__":
    main()
