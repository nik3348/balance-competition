"""
main.py – CLI entry point for the evolutionary game-parameter optimiser.

Usage
-----
    # Auto-loads parameters from valid_params.json for the chosen game:
    uv run main.py --game Dominion

    # Use a custom parameter-space file:
    uv run main.py --game Dominion --params-file my_params.json

Run ``uv run main.py --help`` for the full list of options.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

from evolutionary import EvolutionaryAlgorithm
from param_space import CategoricalParam, FloatParam, IntParam, ParameterSpace

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GAMES = ["Dominion", "ExplodingKittens", "Wonders7", "CantStop"]
RUN_TYPES = ["fast", "medium", "full"]
LOG_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
DEFAULT_VALID_PARAMS_FILE = "valid_params.json"

CUSTOM_PARAMS_FILE_HELP = """\
Provide a JSON file that defines the parameter search space.

Supported parameter types
-------------------------
  int         Integer in a closed range [low, high].
  float       Floating-point value in [low, high].
  categorical One element chosen from a list of discrete choices.

File format
-----------
{
  "PARAM_NAME": {"type": "int",         "low": <int>,   "high": <int>},
  "PARAM_NAME": {"type": "float",       "low": <float>, "high": <float>},
  "PARAM_NAME": {"type": "categorical", "choices": [<value>, ...]}
}

Example (save as params.json and pass with --params-file params.json)
----------------------------------------------------------------------
{
  "HAND_SIZE": {"type": "int",         "low": 1,   "high": 10},
  "DRAW_RATE": {"type": "float",       "low": 0.0, "high": 1.0},
  "STRATEGY":  {"type": "categorical", "choices": ["aggressive", "passive"]}
}
"""


# ---------------------------------------------------------------------------
# Parameter-space loaders
# ---------------------------------------------------------------------------


def load_param_space_from_custom(path: str) -> ParameterSpace:
    """Parse a custom range/categorical JSON file and return a ParameterSpace.

    Expected format::

        {
          "HAND_SIZE": {"type": "int",         "low": 1,   "high": 10},
          "DRAW_RATE": {"type": "float",       "low": 0.0, "high": 1.0},
          "STRATEGY":  {"type": "categorical", "choices": ["a", "b"]}
        }

    Raises
    ------
    FileNotFoundError    – *path* does not exist.
    json.JSONDecodeError – the file is not valid JSON.
    ValueError           – unknown parameter type or missing required key.
    """
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
            raise ValueError(
                f"Parameter {name!r}: expected a JSON object, got {type(spec).__name__}."
            )

        ptype = spec.get("type")
        try:
            if ptype == "int":
                params[name] = IntParam(low=int(spec["low"]), high=int(spec["high"]))
            elif ptype == "float":
                params[name] = FloatParam(
                    low=float(spec["low"]), high=float(spec["high"])
                )
            elif ptype == "categorical":
                choices = list(spec["choices"])
                if not choices:
                    raise ValueError(
                        f"Parameter {name!r}: 'choices' must be a non-empty list."
                    )
                params[name] = CategoricalParam(choices=choices)
            else:
                raise ValueError(
                    f"Parameter {name!r}: unknown type {ptype!r}. "
                    "Expected 'int', 'float', or 'categorical'."
                )
        except KeyError as exc:
            raise ValueError(
                f"Parameter {name!r} (type={ptype!r}): missing required key {exc}."
            ) from exc

    if not params:
        raise ValueError("Params file defines no parameters.")

    return ParameterSpace(params)


def load_param_space_from_valid_params(path: str, game: str) -> ParameterSpace:
    """Load the parameter space for *game* from a valid_params.json style file.

    Expected format – each parameter value is a list of discrete valid values,
    all of which are treated as a :class:`~param_space.CategoricalParam`::

        {
          "Dominion": {
            "HAND_SIZE": [3, 5, 7, 10],
            "CARDS": ["CELLAR", "CHAPEL", ...]
          },
          ...
        }

    Raises
    ------
    FileNotFoundError    – *path* does not exist.
    json.JSONDecodeError – the file is not valid JSON.
    KeyError             – *game* is not present in the file.
    ValueError           – a parameter's value list is empty or wrongly typed.
    """
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
        raise ValueError(
            f"Expected a JSON object for game {game!r}, "
            f"got {type(game_params).__name__}."
        )

    params: dict = {}
    for name, choices in game_params.items():
        if not isinstance(choices, list):
            raise ValueError(
                f"Parameter {name!r}: expected a list of choices, "
                f"got {type(choices).__name__}."
            )
        if not choices:
            raise ValueError(f"Parameter {name!r}: choices list must not be empty.")
        params[name] = CategoricalParam(choices=choices)

    if not params:
        raise ValueError(f"No parameters found for game {game!r} in {path!r}.")

    return ParameterSpace(params)


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def _make_table(rows: list[dict], columns: list[str]) -> str:
    """Return a plain-text ASCII table string."""
    if not rows:
        return "(no data)"

    widths = {
        col: max(len(col), max(len(str(row.get(col, ""))) for row in rows))
        for col in columns
    }

    def _fmt_row(row: dict) -> str:
        return " | ".join(str(row.get(col, "")).ljust(widths[col]) for col in columns)

    header = _fmt_row({col: col for col in columns})
    separator = "-+-".join("-" * widths[col] for col in columns)
    body = "\n".join(_fmt_row(row) for row in rows)
    return "\n".join([header, separator, body])


def _print_section(title: str, content: str) -> None:
    width = max(60, len(title) + 4)
    print()
    print("=" * width)
    print(title)
    print("=" * width)
    print(content)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Evolutionary algorithm for game-parameter optimisation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Parameter space is loaded from --params-file if given, "
            "otherwise from --valid-params-file for the chosen --game."
        ),
    )

    # Required
    parser.add_argument(
        "--game",
        required=True,
        choices=GAMES,
        metavar="{" + ",".join(GAMES) + "}",
        help="Game to optimise.",
    )

    # Parameter-space source (mutually exclusive, both optional)
    param_group = parser.add_mutually_exclusive_group()
    param_group.add_argument(
        "--params-file",
        metavar="PATH",
        help=(
            "Path to a custom JSON file defining the parameter space "
            "(int/float/categorical ranges). "
            "Takes precedence over --valid-params-file."
        ),
    )
    param_group.add_argument(
        "--valid-params-file",
        metavar="PATH",
        default=DEFAULT_VALID_PARAMS_FILE,
        help=(
            "Path to a valid_params.json style file "
            "(lists of discrete values per parameter per game)."
        ),
    )

    # API / run options
    parser.add_argument(
        "--run-type",
        default="fast",
        choices=RUN_TYPES,
        help="Execution mode passed to the game server.",
    )
    parser.add_argument(
        "--timeout-ms",
        type=int,
        default=0,
        metavar="MS",
        help="Per-game server-side timeout in milliseconds (0 = none).",
    )

    # EA hyper-parameters
    parser.add_argument("--pop-size", type=int, default=20, help="Population size.")
    parser.add_argument(
        "--generations", type=int, default=30, help="Number of generations."
    )
    parser.add_argument(
        "--mutation-rate",
        type=float,
        default=0.3,
        metavar="RATE",
        help="Probability [0, 1] that each parameter is mutated per offspring.",
    )
    parser.add_argument(
        "--mutation-strength",
        type=float,
        default=0.2,
        metavar="STRENGTH",
        help="Mutation magnitude relative to each parameter's range.",
    )
    parser.add_argument(
        "--elite-frac",
        type=float,
        default=0.1,
        metavar="FRAC",
        help="Fraction of the population preserved unchanged each generation.",
    )
    parser.add_argument(
        "--tournament-size",
        type=int,
        default=3,
        help="Number of candidates drawn in each tournament selection.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel API calls.",
    )

    # Misc
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed for reproducibility (omit for random).",
    )
    parser.add_argument(
        "--minimize",
        action="store_true",
        help="Minimise the score instead of maximising it.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=LOG_LEVELS,
        help="Logging verbosity.",
    )

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    # Configure logging before anything else so all modules pick it up.
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # Load parameter space
    # ------------------------------------------------------------------
    param_space: ParameterSpace

    if args.params_file:
        # Explicit custom-format file takes priority.
        log.info("Loading custom parameter space from %r …", args.params_file)
        try:
            param_space = load_param_space_from_custom(args.params_file)
        except (FileNotFoundError, json.JSONDecodeError, ValueError) as exc:
            print(
                f"Error loading --params-file {args.params_file!r}: {exc}\n\n"
                + CUSTOM_PARAMS_FILE_HELP,
                file=sys.stderr,
            )
            sys.exit(1)
        source_desc = args.params_file

    else:
        # Fall back to valid_params.json (or whatever --valid-params-file points to).
        vp_path = args.valid_params_file
        log.info("Loading parameter space for game %r from %r …", args.game, vp_path)
        try:
            param_space = load_param_space_from_valid_params(vp_path, args.game)
        except FileNotFoundError:
            print(
                f"Could not find {vp_path!r}. "
                "Either supply --params-file or ensure valid_params.json is present.\n\n"
                + CUSTOM_PARAMS_FILE_HELP,
                file=sys.stderr,
            )
            sys.exit(1)
        except (KeyError, ValueError, json.JSONDecodeError) as exc:
            print(
                f"Error loading {vp_path!r} for game {args.game!r}: {exc}",
                file=sys.stderr,
            )
            sys.exit(1)
        source_desc = f"{vp_path} [{args.game}]"

    log.info(
        "Parameter space loaded: %d parameter(s) from %s.",
        len(param_space.params),
        source_desc,
    )

    # ------------------------------------------------------------------
    # Build and run the evolutionary algorithm
    # ------------------------------------------------------------------
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
    )

    best = algo.run()

    # ------------------------------------------------------------------
    # Print summary – best individual
    # ------------------------------------------------------------------
    param_rows = [{"Parameter": k, "Value": v} for k, v in best.params.items()]
    param_rows.append({"Parameter": "─" * 12, "Value": ""})
    param_rows.append(
        {
            "Parameter": "SCORE",
            "Value": f"{best.score:.6f}" if best.score is not None else "N/A",
        }
    )
    _print_section("BEST INDIVIDUAL", _make_table(param_rows, ["Parameter", "Value"]))

    # ------------------------------------------------------------------
    # Print generation history
    # ------------------------------------------------------------------
    if algo.history:
        history_rows = [
            {
                "gen": str(h["generation"]),
                "best": f"{h['best']:.4f}",
                "mean": f"{h['mean']:.4f}",
                "std": f"{h['std']:.4f}",
            }
            for h in algo.history
        ]
        _print_section(
            "GENERATION HISTORY",
            _make_table(history_rows, ["gen", "best", "mean", "std"]),
        )
    else:
        print("\n(No generation history recorded.)")

    # ------------------------------------------------------------------
    # Save results to JSON
    # ------------------------------------------------------------------
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
