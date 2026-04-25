import json
import random
from typing import Any

import requests

GAME = "ExplodingKittens"
PORT = 3000
URL_BASE = f"http://localhost:{PORT}/api/"
RUN_TYPE = "fast"
random.seed()

with open("valid_params.json", "r") as f:
    params = json.load(f)
    game_params = params[GAME]

max_index = [len(game_params[param]) - 1 for param in game_params]
assert len(max_index) == len(game_params)


def solution_to_params(solution: list[int]):
    chosen_parameters = {}

    for i, param_name in enumerate(game_params):
        parameter = game_params[param_name]
        value = parameter[solution[i]]

        chosen_parameters[param_name] = value
    return chosen_parameters


solution = [random.randint(0, max_i) for max_i in max_index]
print(solution)
print(solution_to_params(solution))


def run_game(
    solution: list[int], game: str = GAME, run_type: str = RUN_TYPE
) -> tuple[list[int], dict, Any]:
    params = solution_to_params(solution)
    url = f"{URL_BASE}{game}/{run_type}"
    headers = {"Content-Type": "application/json"}
    body = {
        "game": game,
        "params": params,
        "runType": run_type,
    }
    response = requests.post(url, headers=headers, json=body)
    return solution, params, response.json()["score"]


solution = [random.randint(0, max_i) for max_i in max_index]
print(solution)
print(solution_to_params(solution))
print(run_game(solution))
