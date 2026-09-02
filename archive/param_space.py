"""Parameter space definitions for game-parameter optimisation."""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class IntParam:
    low: int
    high: int


@dataclass
class FloatParam:
    low: float
    high: float


@dataclass
class CategoricalParam:
    choices: list


@dataclass
class ListParam:
    choices: list
    min_count: int
    max_count: int
    unique: bool = True


ParamType = IntParam | FloatParam | CategoricalParam | ListParam


class ParameterSpace:
    def __init__(self, params: dict[str, ParamType]) -> None:
        self.params = params

    def sample(self, rng: random.Random) -> dict:
        result = {}
        for name, param in self.params.items():
            if isinstance(param, IntParam):
                result[name] = rng.randint(param.low, param.high)
            elif isinstance(param, FloatParam):
                result[name] = rng.uniform(param.low, param.high)
            elif isinstance(param, ListParam):
                k = min(rng.randint(param.min_count, param.max_count), len(param.choices))
                result[name] = rng.sample(param.choices, k)
            elif isinstance(param, CategoricalParam):
                result[name] = rng.choice(param.choices)
            else:
                raise TypeError(f"Unknown parameter type for '{name}': {type(param)}")
        return result

    def mutate(
        self,
        individual: dict,
        rng: random.Random,
        mutation_rate: float,
        mutation_strength: float = 0.2,
    ) -> dict:
        result = dict(individual)
        for name, param in self.params.items():
            if rng.random() >= mutation_rate:
                continue

            if isinstance(param, IntParam):
                delta = max(1, int((param.high - param.low) * mutation_strength))
                result[name] = max(
                    param.low,
                    min(param.high, individual[name] + rng.randint(-delta, delta)),
                )
            elif isinstance(param, FloatParam):
                noise = rng.gauss(0.0, (param.high - param.low) * mutation_strength)
                result[name] = max(
                    param.low,
                    min(param.high, individual[name] + noise),
                )
            elif isinstance(param, ListParam):
                current = list(individual[name])
                op = rng.random()
                if op < 0.3 and len(current) > param.min_count:
                    current.pop(rng.randrange(len(current)))
                elif op < 0.6 and len(current) < param.max_count and len(current) < len(param.choices):
                    available = [c for c in param.choices if c not in current]
                    if available:
                        current.append(rng.choice(available))
                elif current:
                    pos = rng.randrange(len(current))
                    available = [c for c in param.choices if c not in current]
                    if available:
                        current[pos] = rng.choice(available)
                result[name] = current
            elif isinstance(param, CategoricalParam):
                result[name] = rng.choice(param.choices)
            else:
                raise TypeError(f"Unknown parameter type for '{name}': {type(param)}")
        return result

    def crossover(self, a: dict, b: dict, rng: random.Random) -> dict:
        result = {}
        for name, param in self.params.items():
            if isinstance(param, ListParam):
                set_a = set(a[name])
                set_b = set(b[name])
                shared = list(set_a & set_b)
                only_a = list(set_a - set_b)
                only_b = list(set_b - set_a)
                rng.shuffle(only_a)
                rng.shuffle(only_b)
                child = list(shared)
                pool = only_a + only_b
                rng.shuffle(pool)
                target_len = rng.randint(
                    max(param.min_count, len(child)),
                    min(param.max_count, len(param.choices)),
                )
                for item in pool:
                    if len(child) >= target_len:
                        break
                    child.append(item)
                while len(child) < param.min_count and pool:
                    for item in pool:
                        if item not in child:
                            child.append(item)
                            break
                    else:
                        break
                result[name] = child
            else:
                result[name] = a[name] if rng.random() < 0.5 else b[name]
        return result
