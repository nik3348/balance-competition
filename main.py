"""
main.py – Helpers shared across optimizers.
"""

import json

from param_space import (
    CategoricalParam,
    MultiCategoricalParam,
    ParameterSpace,
)

# Multi-categorical parameters: maps (game, param_name) → selection count
_MULTI_CATEGORICAL: dict[tuple[str, str], int] = {
    ("Dominion", "CARDS"): 10,
    ("Wonders7", "wonders"): 7,
}


def load_param_space_from_valid_params(path: str, game: str) -> ParameterSpace:
    """Load a ParameterSpace from valid_params.json for a given game."""
    with open(path) as f:
        all_params = json.load(f)

    game_params = all_params[game]
    params = {}

    for name, values in game_params.items():
        key = (game, name)
        if key in _MULTI_CATEGORICAL:
            params[name] = MultiCategoricalParam(
                choices=values, count=_MULTI_CATEGORICAL[key]
            )
        else:
            params[name] = CategoricalParam(choices=values)

    return ParameterSpace(params)
