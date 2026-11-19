"""Travelling salesman via simulated annealing with 2-opt moves evaluated in O(1)."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import NamedTuple


class City(NamedTuple):
    x: float
    y: float

    def distance_to(self, other: City) -> float:
        return math.hypot(self.x - other.x, self.y - other.y)


@dataclass(frozen=True)
class Tour:
    cities: tuple[City, ...]
    length: float


def tour_length(tour: Sequence[City]) -> float:
    return sum(tour[k].distance_to(tour[(k + 1) % len(tour)]) for k in range(len(tour)))


def two_opt_delta(tour: Sequence[City], i: int, j: int) -> float:
    """Change in length from reversing positions i..j (inclusive)."""
    n = len(tour)
    a, b, c, d = tour[(i - 1) % n], tour[i], tour[j], tour[(j + 1) % n]
    if a == c or b == d:
        return 0.0  # reversing the whole tour leaves it unchanged
    return a.distance_to(c) + b.distance_to(d) - a.distance_to(b) - c.distance_to(d)


def solve(cities: Sequence[City], rng: random.Random, *, start_temperature: float = 100.0,
          min_temperature: float = 1e-3, cooling_rate: float = 1e-4) -> Tour:
    if not start_temperature > min_temperature > 0 or not 0 < cooling_rate < 1:
        raise ValueError("invalid annealing schedule")
    current = list(cities)
    if len(current) < 4:
        return Tour(tuple(current), tour_length(current))
    rng.shuffle(current)
    length = tour_length(current)
    best, best_length = current[:], length
    n = len(current)
    t = start_temperature
    while t > min_temperature:
        i = rng.randrange(n - 1)
        j = rng.randrange(i + 1, n)
        delta = two_opt_delta(current, i, j)
        if delta < 0 or math.exp(-delta / t) > rng.random():
            current[i: j + 1] = current[i: j + 1][::-1]
            length += delta
            if length < best_length - 1e-12:
                best, best_length = current[:], length
        t *= 1 - cooling_rate
    return Tour(tuple(best), tour_length(best))
