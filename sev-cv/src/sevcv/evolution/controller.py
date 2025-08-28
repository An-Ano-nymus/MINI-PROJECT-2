from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple
import numpy as np


@dataclass
class Individual:
    prompt_tokens: List[str]
    z_seed: int
    policy: Dict[str, Any]
    micro: Dict[str, Any]


class EvolutionController:
    def __init__(self, pop_size: int = 8, z_dim: int = 128):
        self.pop_size = pop_size
        self.z_dim = z_dim
        self.rng = np.random.default_rng(42)
        self.population: List[Individual] = []
        self._init_population()

    def _init_population(self):
        self.population = [
            Individual(
                prompt_tokens=[],
                z_seed=int(self.rng.integers(0, 1_000_000)),
                policy={
                    "lr": float(10 ** self.rng.uniform(-4.5, -3.0)),
                    "beta1": 0.0,
                    "beta2": 0.99,
                    "ema": 0.999,
                },
                micro={"heads": int(self.rng.integers(2, 8)), "depth": int(self.rng.integers(2, 6))},
            )
            for _ in range(self.pop_size)
        ]

    def mutate(self) -> List[Individual]:
        children: List[Individual] = []
        for ind in self.population:
            lr = ind.policy["lr"] * (10 ** self.rng.normal(0, 0.15))
            ema = min(0.9999, max(0.9, ind.policy["ema"] + self.rng.normal(0, 0.01)))
            heads = max(2, min(8, ind.micro["heads"] + int(self.rng.integers(-1, 2))))
            depth = max(2, min(8, ind.micro["depth"] + int(self.rng.integers(-1, 2))))
            child = Individual(
                prompt_tokens=ind.prompt_tokens.copy(),
                z_seed=int(self.rng.integers(0, 1_000_000)),
                policy={"lr": float(lr), "beta1": 0.0, "beta2": 0.99, "ema": float(ema)},
                micro={"heads": int(heads), "depth": int(depth)},
            )
            children.append(child)
        return children

    def select(self, candidates: List[Tuple[Individual, float]], k: int = 8):
        # lower is better for FID-like score
        candidates_sorted = sorted(candidates, key=lambda x: x[1])
        self.population = [candidates_sorted[i][0] for i in range(min(k, len(candidates_sorted)))]
        # re-seed if needed
        while len(self.population) < self.pop_size:
            self.population.append(self.population[-1])
