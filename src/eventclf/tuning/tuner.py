from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Callable, Optional, List, Tuple
import numpy as np


@dataclass
class RandomSearchResult:
    best_params: Dict[str, Any]
    best_score: float
    trials: List[Dict[str, Any]]


def _sample(space: Dict[str, list], rng: np.random.Generator) -> Dict[str, Any]:
    params = {}
    for k, values in space.items():
        params[k] = values[int(rng.integers(0, len(values)))]
    return params


def random_search(*,space: Dict[str, list],objective_fn: Callable[[Dict[str, Any]], float],
    n_trials: int = 50,seed: int = 42,) -> RandomSearchResult:
    rng = np.random.default_rng(seed)

    best_score = -np.inf
    best_params = None
    trials: List[Dict[str, Any]] = []

    for t in range(n_trials):
        params = _sample(space, rng)
        score = float(objective_fn(params))

        trial = {"trial": t, "score": score, "params": params}
        trials.append(trial)

        if score > best_score:
            best_score = score
            best_params = params

    assert best_params is not None
    return RandomSearchResult(best_params=best_params, best_score=best_score, trials=trials)