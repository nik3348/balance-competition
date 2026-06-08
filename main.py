"""CLI entry point for the evolutionary game-parameter optimiser.

Usage:
    uv run main.py --game Dominion
    uv run main.py --game Dominion --params-file my_params.json
    uv run main.py --help
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

from evolutionary import EvolutionaryAlgorithm
from param_space import CategoricalParam, FloatParam, IntParam, ListParam, ParameterSpace

GAMES = ["Dominion", "ExplodingKittens", "Wonders7", "CantStop"]
RUN_TYPES = ["fast", "medium", "full"]
LOG_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
DEFAULT_VALID_PARAMS_FILE = "valid_params.json"

LIST_PARAM_CONFIGS = {
    "Dominion": {"CARDS": {"min_count": 10, "max_count": 10}},
    "Wonders7": {"wonders": {"min_count": 4, "max_count": 7}},
}


def load_param_space_from_custom(path: str) -> ParameterSpace:
    """Parse a custom JSON file defining int/float/categorical parameter ranges."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Params file not found: {path!r}")

    with file_path.open() as fh:
        data = json.load(fh)

    if not isinstance(data, dict):
        raise ValueError("Params file must contain a JSON object at the top level.")

    params: dict = {}
    for name, spec in data.items():
        if not isinstance(spec, dict):
            raise ValueError(f"Parameter {name!r}: expected a JSON object, got {type(spec).__name__}.")

        ptype = spec.get("type")
        try:
            if ptype == "int":
                params[name] = IntParam(low=int(spec["low"]), high=int(spec["high"]))
            elif ptype == "float":
                params[name] = FloatParam(low=float(spec["low"]), high=float(spec["high"]))
            elif ptype == "categorical":
                choices = list(spec["choices"])
                if not choices:
                    raise ValueError(f"Parameter {name!r}: 'choices' must be a non-empty list.")
                params[name] = CategoricalParam(choices=choices)
            else:
                raise ValueError(f"Parameter {name!r}: unknown type {ptype!r}. Expected 'int', 'float', or 'categorical'.")
        except KeyError as exc:
            raise ValueError(f"Parameter {name!r} (type={ptype!r}): missing required key {exc}.") from exc

    if not params:
        raise ValueError("Params file defines no parameters.")

    return ParameterSpace(params)


def load_param_space_from_valid_params(path: str, game: str) -> ParameterSpace:
    """Load the parameter space for a game from a valid_params.json style file."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Valid-params file not found: {path!r}")

    with file_path.open() as fh:
        data = json.load(fh)

    if game not in data:
        available = ", ".join(data.keys())
        raise KeyError(f"Game {game!r} not found in {path!r}. Available: {available}")

    game_params = data[game]
    if not isinstance(game_params, dict):
        raise ValueError(f"Expected a JSON object for game {game!r}, got {type(game_params).__name__}.")

    list_configs = LIST_PARAM_CONFIGS.get(game, {})

    params: dict = {}
    for name, choices in game_params.items():
        if not isinstance(choices, list):
            raise ValueError(f"Parameter {name!r}: expected a list of choices, got {type(choices).__name__}.")
        if not choices:
            raise ValueError(f"Parameter {name!r}: choices list must not be empty.")

        if name in list_configs:
            cfg = list_configs[name]
            params[name] = ListParam(choices=choices, min_count=cfg["min_count"], max_count=cfg["max_count"])
        else:
            params[name] = CategoricalParam(choices=choices)

    if not params:
        raise ValueError(f"No parameters found for game {game!r} in {path!r}.")

    return ParameterSpace(params)


def _make_table(rows: list[dict], columns: list[str]) -> str:
    if not rows:
        return "(no data)"

    widths = {col: max(len(col), max(len(str(row.get(col, ""))) for row in rows)) for col in columns}

    def fmt_row(row: dict) -> str:
        return " | ".join(str(row.get(col, "")).ljust(widths[col]) for col in columns)

    header = fmt_row({col: col for col in columns})
    separator = "-+-".join("-" * widths[col] for col in columns)
    body = "\n".join(fmt_row(row) for row in rows)
    return "\n".join([header, separator, body])


def _print_section(title: str, content: str) -> None:
    width = max(60, len(title) + 4)
    print(f"\n{'=' * width}\n{title}\n{'=' * width}\n{content}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Evolutionary algorithm for game-parameter optimisation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--game", required=True, choices=GAMES, help="Game to optimise.")

    param_group = parser.add_mutually_exclusive_group()
    param_group.add_argument("--params-file", metavar="PATH", help="Custom JSON file defining parameter space (int/float/categorical).")
    param_group.add_argument("--valid-params-file", metavar="PATH", default=DEFAULT_VALID_PARAMS_FILE, help="valid_params.json style file.")

    parser.add_argument("--run-type", default="fast", choices=RUN_TYPES, help="Execution mode passed to the game server.")
    parser.add_argument("--timeout-ms", type=int, default=0, metavar="MS", help="Per-game server-side timeout in milliseconds (0 = none).")

    parser.add_argument("--pop-size", type=int, default=20, help="Population size.")
    parser.add_argument("--generations", type=int, default=30, help="Number of generations.")
    parser.add_argument("--mutation-rate", type=float, default=0.3, metavar="RATE", help="Probability [0, 1] that each parameter is mutated per offspring.")
    parser.add_argument("--mutation-strength", type=float, default=0.2, metavar="STRENGTH", help="Mutation magnitude relative to each parameter's range.")
    parser.add_argument("--elite-frac", type=float, default=0.1, metavar="FRAC", help="Fraction of the population preserved unchanged each generation.")
    parser.add_argument("--tournament-size", type=int, default=3, help="Number of candidates drawn in each tournament selection.")
    parser.add_argument("--workers", type=int, default=2, help="Number of parallel API calls.")
    parser.add_argument("--stagnation-limit", type=int, default=10, help="Generations without improvement before partial population restart.")
    parser.add_argument("--mutation-decay", type=float, default=0.98, help="Factor to decay mutation strength each generation (0-1, 1=no decay).")

    parser.add_argument("--seed", type=int, default=None, help="RNG seed for reproducibility.")
    parser.add_argument("--minimize", action="store_true", help="Minimise the score instead of maximising it.")
    parser.add_argument("--log-level", default="INFO", choices=LOG_LEVELS, help="Logging verbosity.")

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger(__name__)

    if args.params_file:
        log.info("Loading custom parameter space from %r", args.params_file)
        try:
            param_space = load_param_space_from_custom(args.params_file)
        except (FileNotFoundError, json.JSONDecodeError, ValueError) as exc:
            print(f"Error loading --params-file {args.params_file!r}: {exc}", file=sys.stderr)
            sys.exit(1)
        source_desc = args.params_file
    else:
        vp_path = args.valid_params_file
        log.info("Loading parameter space for game %r from %r", args.game, vp_path)
        try:
            param_space = load_param_space_from_valid_params(vp_path, args.game)
        except FileNotFoundError:
            print(f"Could not find {vp_path!r}. Supply --params-file or ensure valid_params.json is present.", file=sys.stderr)
            sys.exit(1)
        except (KeyError, ValueError, json.JSONDecodeError) as exc:
            print(f"Error loading {vp_path!r} for game {args.game!r}: {exc}", file=sys.stderr)
            sys.exit(1)
        source_desc = f"{vp_path} [{args.game}]"

    log.info("Parameter space loaded: %d parameter(s) from %s.", len(param_space.params), source_desc)

    algo = EvolutionaryAlgorithm(
        game=args.game,
        param_space=param_space,
        run_type=args.run_type,
        pop_size=args.pop_size,
        n_generations=args.generations,
        mutation_rate=args.mutation_rate,
        mutation_strength=args.mutation_strength,
        elite_frac=args.elite_frac,
        tournament_size=args.tournament_size,
        n_workers=args.workers,
        timeout_ms=args.timeout_ms,
        seed=args.seed,
        maximize=not args.minimize,
        stagnation_limit=args.stagnation_limit,
        mutation_decay=args.mutation_decay,
    )

    best = algo.run()

    param_rows = [{"Parameter": k, "Value": v} for k, v in best.params.items()]
    param_rows.append({"Parameter": "─" * 12, "Value": ""})
    param_rows.append({"Parameter": "SCORE", "Value": f"{best.score:.6f}" if best.score is not None else "N/A"})
    _print_section("BEST INDIVIDUAL", _make_table(param_rows, ["Parameter", "Value"]))

    if algo.history:
        history_rows = [
            {"gen": str(h["generation"]), "best": f"{h['best']:.4f}", "mean": f"{h['mean']:.4f}", "std": f"{h['std']:.4f}"}
            for h in algo.history
        ]
        _print_section("GENERATION HISTORY", _make_table(history_rows, ["gen", "best", "mean", "std"]))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"results_{args.game}_{args.run_type}_{timestamp}.json"

    results = {
        "game": args.game,
        "run_type": args.run_type,
        "best_score": best.score,
        "best_params": best.params,
        "history": algo.history,
        "settings": {
            "param_source": source_desc,
            "pop_size": args.pop_size,
            "generations": args.generations,
            "mutation_rate": args.mutation_rate,
            "mutation_strength": args.mutation_strength,
            "elite_frac": args.elite_frac,
            "tournament_size": args.tournament_size,
            "workers": args.workers,
            "timeout_ms": args.timeout_ms,
            "seed": args.seed,
            "maximize": not args.minimize,
            "stagnation_limit": args.stagnation_limit,
            "mutation_decay": args.mutation_decay,
        },
    }

    try:
        with open(results_file, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"\nResults saved to: {results_file}")
    except OSError as exc:
        log.error("Could not save results to %r: %s", results_file, exc)


if __name__ == "__main__":
    main()
