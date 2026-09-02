from __future__ import annotations

import random
from dataclasses import dataclass, field


@dataclass
class IntParam:
    low: int
    high: int  # inclusive


@dataclass
class FloatParam:
    low: float
    high: float


@dataclass
class CategoricalParam:
    choices: list  # list of any JSON-serialisable values


@dataclass
class MultiCategoricalParam:
    """Parameter that selects multiple items from a list (e.g., 10 cards from 26)."""
    choices: list  # list of any JSON-serialisable values
    count: int = 10  # number of items to select


ParamType = IntParam | FloatParam | CategoricalParam | MultiCategoricalParam


class ParameterSpace:
    def __init__(self, params: dict[str, ParamType]) -> None:
        self.params = params

    def sample(self, rng: random.Random) -> dict:
        """Randomly sample one value per parameter."""
        result = {}
        for name, param in self.params.items():
            if isinstance(param, IntParam):
                result[name] = rng.randint(param.low, param.high)
            elif isinstance(param, FloatParam):
                result[name] = rng.uniform(param.low, param.high)
            elif isinstance(param, CategoricalParam):
                result[name] = rng.choice(param.choices)
            elif isinstance(param, MultiCategoricalParam):
                # Sample count items (with replacement allowed)
                result[name] = [rng.choice(param.choices) for _ in range(param.count)]
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
        """Return a mutated copy of *individual*.

        Each parameter is independently mutated with probability *mutation_rate*.
        """
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
            elif isinstance(param, CategoricalParam):
                result[name] = rng.choice(param.choices)
            elif isinstance(param, MultiCategoricalParam):
                # Mutate each item in the list independently
                current_list = list(individual[name])
                for i in range(len(current_list)):
                    if rng.random() < mutation_rate:
                        current_list[i] = rng.choice(param.choices)
                result[name] = current_list
            else:
                raise TypeError(f"Unknown parameter type for '{name}': {type(param)}")
        return result

    def crossover(self, a: dict, b: dict, rng: random.Random) -> dict:
        """Uniform crossover: for each key, randomly pick the value from *a* or *b*.

        For MultiCategoricalParam, performs element-wise crossover.
        """
        result = {}
        for name in self.params:
            param = self.params[name]
            if isinstance(param, MultiCategoricalParam):
                # Element-wise crossover for lists
                list_a = a[name]
                list_b = b[name]
                result[name] = [
                    (list_a[i] if rng.random() < 0.5 else list_b[i])
                    for i in range(len(list_a))
                ]
            else:
                result[name] = (a[name] if rng.random() < 0.5 else b[name])
        return result
