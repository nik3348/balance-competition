"""
verify.py – Multi-rep verification of best params for all games.

Tests both the best params from our optimization and the best known from previous results.
"""

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import api_client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Our best params from optimization
OUR_BEST = {
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
        "nCardsPerPlayer": 3, "nopeOwnCards": False,
        "ATTACK_count": 2, "SKIP_count": 3, "FAVOR_count": 8,
        "SHUFFLE_count": 5, "SEETHEFUTURE_count": 4, "TACOCAT_count": 2,
        "MELONCAT_count": 3, "BEARDCAT_count": 5, "RAINBOWCAT_count": 6,
        "FURRYCAT_count": 8, "NOPE_count": 9, "DEFUSE_count": 7,
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

# Previous best known
PREVIOUS_BEST = {
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


def eval_game(game: str, params: dict, run_type: str = "fast") -> float | None:
    try:
        return api_client.run_game(game=game, params=params, run_type=run_type, timeout_ms=0, http_timeout=600.0)
    except api_client.APIError:
        return None


def verify_params(game: str, params: dict, n_reps: int = 20, run_type: str = "fast") -> list[float]:
    """Evaluate params n_reps times and return all scores."""
    scores = []
    for i in range(n_reps):
        s = eval_game(game, params, run_type)
        if s is not None:
            scores.append(s)
            logger.info("  %s rep %d/%d: %.4f", game, i + 1, n_reps, s)
    return scores


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", default=None, choices=["CantStop", "Dominion", "ExplodingKittens", "Wonders7"])
    parser.add_argument("--n-reps", type=int, default=20)
    parser.add_argument("--run-type", default="fast")
    args = parser.parse_args()

    games = [args.game] if args.game else ["CantStop", "Dominion", "ExplodingKittens", "Wonders7"]

    all_results = {}
    for game in games:
        logger.info("=" * 60)
        logger.info("VERIFYING %s", game)
        logger.info("=" * 60)

        # Verify our best
        logger.info("--- Our best params ---")
        our_scores = verify_params(game, OUR_BEST[game], args.n_reps, args.run_type)
        our_mean = sum(our_scores) / len(our_scores) if our_scores else 0
        our_max = max(our_scores) if our_scores else 0
        our_min = min(our_scores) if our_scores else 0
        logger.info("Our best: mean=%.4f max=%.4f min=%.4f (over %d reps)", our_mean, our_max, our_min, len(our_scores))

        # Verify previous best if available
        prev_mean = None
        if game in PREVIOUS_BEST:
            logger.info("--- Previous best params ---")
            prev_scores = verify_params(game, PREVIOUS_BEST[game], args.n_reps, args.run_type)
            prev_mean = sum(prev_scores) / len(prev_scores) if prev_scores else 0
            prev_max = max(prev_scores) if prev_scores else 0
            prev_min = min(prev_scores) if prev_scores else 0
            logger.info("Previous best: mean=%.4f max=%.4f min=%.4f (over %d reps)", prev_mean, prev_max, prev_min, len(prev_scores))
        else:
            prev_scores = []

        all_results[game] = {
            "our_best": {"mean": our_mean, "max": our_max, "min": our_min, "scores": our_scores, "params": OUR_BEST[game]},
            "previous_best": {"mean": prev_mean, "scores": prev_scores, "params": PREVIOUS_BEST.get(game)},
        }

    # Summary
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    for game, result in all_results.items():
        our = result["our_best"]
        prev = result["previous_best"]
        print(f"\n{game}:")
        print(f"  Our best:      mean={our['mean']:.2f} max={our['max']:.2f}")
        if prev["mean"] is not None:
            print(f"  Previous best: mean={prev['mean']:.2f}")
            if our["mean"] > prev["mean"]:
                print(f"  => Our params are BETTER (+{our['mean'] - prev['mean']:.2f})")
            else:
                print(f"  => Previous params are BETTER (+{prev['mean'] - our['mean']:.2f})")
    print("=" * 60)

    # Save
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"verification_results_{timestamp}.json"
    with open(filename, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("Saved to %s", filename)


if __name__ == "__main__":
    main()
