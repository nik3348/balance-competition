"""
final_opt.py – Final focused optimization for all games.

For each game:
1. Start from current best params
2. Run coord descent with parallel evaluation
3. Verify final params with many reps
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

# Current best params for each game
CURRENT_BEST = {
    "CantStop": {
        "TWO_MAX": 5, "THREE_MAX": 2, "FOUR_MAX": 6, "FIVE_MAX": 7,
        "SIX_MAX": 9, "SEVEN_MAX": 11, "EIGHT_MAX": 8, "NINE_MAX": 10,
        "TEN_MAX": 4, "ELEVEN_MAX": 4, "TWELVE_MAX": 4,
        "COLUMNS_TO_WIN": 2, "MARKERS": 2,
    },
    "Dominion": {
        "HAND_SIZE": 3, "PILES_EXHAUSTED_FOR_GAME_END": 3,
        "KINGDOM_CARDS_OF_EACH_TYPE": 10, "CURSE_CARDS_PER_PLAYER": 5,
        "STARTING_COPPER": 5, "STARTING_ESTATES": 7,
        "COPPER_SUPPLY": 40, "SILVER_SUPPLY": 30, "GOLD_SUPPLY": 50,
        "CARDS": ["WORKSHOP", "MARKET", "MILITIA", "WITCH", "WITCH",
                  "CHAPEL", "WORKSHOP", "CHAPEL", "ARTISAN", "MILITIA"],
    },
    "ExplodingKittens": {
        "nCardsPerPlayer": 5, "nopeOwnCards": False,
        "ATTACK_count": 1, "SKIP_count": 3, "FAVOR_count": 8,
        "SHUFFLE_count": 3, "SEETHEFUTURE_count": 4, "TACOCAT_count": 4,
        "MELONCAT_count": 2, "BEARDCAT_count": 10, "RAINBOWCAT_count": 5,
        "FURRYCAT_count": 8, "NOPE_count": 9, "DEFUSE_count": 6,
    },
    "Wonders7": {
        "nCostNeighbourResource": 3, "nCostDiscountedResource": 2,
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


def eval_with_retry(game: str, params: dict, run_type: str, max_retries: int = 2) -> float | None:
    for attempt in range(max_retries):
        try:
            return api_client.run_game(game=game, params=params, run_type=run_type, timeout_ms=0, http_timeout=600.0)
        except api_client.APIError:
            if attempt < max_retries - 1:
                time.sleep(2)
    return None


def coord_descent(game: str, start_params: dict, valid_params: dict, run_type: str = "fast", max_rounds: int = 5, n_workers: int = 3) -> tuple[dict, float]:
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
                count = MULTI_CATEGORICAL[key]
                for idx in range(min(count, 5)):
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
                        future_map = {executor.submit(eval_with_retry, game, p, run_type): val for val, p in candidates}
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
                candidates = [(v, {**current, param_name: v}) for v in valid_values if v != current[param_name]]
                if not candidates:
                    continue
                best_val = current[param_name]
                best_score = current_score
                with ThreadPoolExecutor(max_workers=n_workers) as executor:
                    future_map = {executor.submit(eval_with_retry, game, p, run_type): val for val, p in candidates}
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

        if not improved:
            logger.info("Converged after %d rounds.", round_num)
            break

    return current, current_score


def verify(game: str, params: dict, n_reps: int = 20, run_type: str = "fast") -> tuple[float, list[float]]:
    scores = []
    for i in range(n_reps):
        s = eval_with_retry(game, params, run_type)
        if s is not None:
            scores.append(s)
            logger.info("  Rep %d/%d: %.4f", i + 1, n_reps, s)
    mean = sum(scores) / len(scores) if scores else 0
    return mean, scores


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", required=True, choices=["CantStop", "Dominion", "ExplodingKittens", "Wonders7"])
    parser.add_argument("--run-type", default="fast")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--max-rounds", type=int, default=5)
    parser.add_argument("--n-verify", type=int, default=20)
    args = parser.parse_args()

    game = args.game
    valid_params = load_valid_params(game)

    logger.info("=" * 60)
    logger.info("FINAL OPTIMIZATION: %s", game)
    logger.info("=" * 60)

    # Start from current best
    start_params = CURRENT_BEST[game]
    logger.info("--- Coord descent from current best ---")
    best_params, best_score = coord_descent(game, start_params, valid_params, args.run_type, args.max_rounds, args.workers)
    logger.info("Coord descent result: %.4f", best_score)

    # Verify
    logger.info("--- Verification (%d reps) ---", args.n_verify)
    mean, scores = verify(game, best_params, args.n_verify, args.run_type)
    logger.info("Verified mean: %.4f (over %d reps)", mean, len(scores))

    # Save
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"results_final2_{game}_{timestamp}.json"
    with open(filename, "w") as f:
        json.dump({
            "game": game, "best_score": best_score, "verified_mean": mean,
            "best_params": best_params, "all_scores": scores,
        }, f, indent=2)
    logger.info("Saved to %s", filename)


if __name__ == "__main__":
    main()
