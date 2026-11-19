"""Local search and metaheuristics. Every optimiser *minimises*; to maximise f, minimise -f."""

from __future__ import annotations

import math
import random
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass

Objective1D = Callable[[float], float]
Objective2D = Callable[[float, float], float]
ObjectiveND = Callable[[Sequence[float]], float]


@dataclass(frozen=True)
class Optimum:
    """Best point found, its objective value and the number of evaluations used."""

    point: tuple[float, ...]
    value: float
    evaluations: int

    @property
    def x(self) -> float:
        return self.point[0]

    def __str__(self) -> str:
        coords = ", ".join(f"{c:.4f}" for c in self.point)
        return f"x = ({coords}), f = {self.value:.6f}, {self.evaluations:,} evaluations"


def _check_interval(lo: float, hi: float) -> None:
    if not lo < hi:
        raise ValueError(f"require lo < hi, got [{lo}, {hi}]")


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


# ----------------------------------------------------------------------------- 1-D baselines

def brute_force(f: Objective1D, lo: float, hi: float, step: float) -> Optimum:
    """Evaluate f on an evenly spaced grid over [lo, hi]."""
    _check_interval(lo, hi)
    if step <= 0:
        raise ValueError("step must be positive")
    n = math.floor((hi - lo) / step)
    xs = [lo + i * step for i in range(n + 1)]  # index-based: no floating-point drift
    best = min(xs, key=f)
    return Optimum((best,), f(best), len(xs))


def random_search(f: Objective1D, lo: float, hi: float, iterations: int, rng: random.Random) -> Optimum:
    """Best of ``iterations`` uniformly random points."""
    _check_interval(lo, hi)
    best = min((rng.uniform(lo, hi) for _ in range(iterations)), key=f)
    return Optimum((best,), f(best), iterations)


def hill_climbing(f: Objective1D, start: float, step: float, lo: float, hi: float,
                  max_iterations: int = 100_000) -> Optimum:
    """Steepest descent: move to the better neighbour (x ± step) until neither improves."""
    _check_interval(lo, hi)
    x = _clamp(start, lo, hi)
    value, evaluations = f(x), 1
    for _ in range(max_iterations):
        left, right = _clamp(x - step, lo, hi), _clamp(x + step, lo, hi)
        f_left, f_right = f(left), f(right)
        evaluations += 2
        if f_left < value and f_left <= f_right:
            x, value = left, f_left
        elif f_right < value:
            x, value = right, f_right
        else:
            break  # local optimum
    return Optimum((x,), value, evaluations)


# ----------------------------------------------------------------------------- simulated annealing

def acceptance_probability(current: float, candidate: float, temperature: float) -> float:
    """Metropolis criterion for minimisation."""
    if candidate < current:
        return 1.0
    return math.exp((current - candidate) / temperature)


def simulated_annealing(f: Objective1D, lo: float, hi: float, rng: random.Random, *,
                        start_temperature: float = 10.0, min_temperature: float = 1e-3,
                        cooling_rate: float = 0.005) -> Optimum:
    """Simulated annealing with Gaussian neighbour proposals that narrow as the system cools."""
    _check_interval(lo, hi)
    if not start_temperature > min_temperature > 0 or not 0 < cooling_rate < 1:
        raise ValueError("invalid annealing schedule")
    width = hi - lo
    current = rng.uniform(lo, hi)
    current_value = f(current)
    best, best_value, evaluations = current, current_value, 1
    t = start_temperature
    while t > min_temperature:
        scale = width * 0.25 * (t / start_temperature) + 1e-6
        candidate = _clamp(current + rng.gauss(0, scale), lo, hi)
        candidate_value = f(candidate)
        evaluations += 1
        if acceptance_probability(current_value, candidate_value, t) > rng.random():
            current, current_value = candidate, candidate_value
            if current_value < best_value:
                best, best_value = current, current_value
        t *= 1 - cooling_rate
    return Optimum((best,), best_value, evaluations)


# ----------------------------------------------------------------------------- tabu search

def tabu_search(f: Objective2D, start: tuple[float, float], *, lo: float, hi: float, step: float,
                tenure: int = 400, iterations: int = 2_000) -> Optimum:
    """Tabu search on a square grid over [lo, hi]^2 with 4-neighbour moves.

    Always moves to the best non-tabu neighbour (even if worse), remembering the last
    ``tenure`` visited points. If every neighbour is tabu it takes the best one anyway.
    """
    _check_interval(lo, hi)
    size = round((hi - lo) / step) + 1

    def coord(i: int) -> float:
        return lo + i * step

    def to_index(x: float) -> int:
        return max(0, min(size - 1, round((x - lo) / step)))

    def value(p: tuple[int, int]) -> float:
        return f(coord(p[0]), coord(p[1]))

    current = (to_index(start[0]), to_index(start[1]))
    best, best_value, evaluations = current, value(current), 1
    tabu_queue: deque[tuple[int, int]] = deque()
    tabu_set: set[tuple[int, int]] = set()

    for _ in range(iterations):
        if current not in tabu_set:
            tabu_queue.append(current)
            tabu_set.add(current)
            if len(tabu_queue) > tenure:
                tabu_set.discard(tabu_queue.popleft())
        i, j = current
        neighbours = [(a, b) for a, b in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1))
                      if 0 <= a < size and 0 <= b < size]
        scored = [(value(p), p) for p in neighbours]
        evaluations += len(scored)
        allowed = [s for s in scored if s[1] not in tabu_set] or scored
        next_value, current = min(allowed)
        if next_value < best_value:
            best, best_value = current, next_value
    return Optimum((coord(best[0]), coord(best[1])), best_value, evaluations)


# ----------------------------------------------------------------------------- particle swarm

def particle_swarm(f: ObjectiveND, dimensions: int, lo: float, hi: float, rng: random.Random, *,
                   particles: int = 20, iterations: int = 200,
                   inertia: float = 0.729, cognitive: float = 1.49445, social: float = 1.49445) -> Optimum:
    """Particle swarm optimisation over [lo, hi]^d with velocity clamping."""
    _check_interval(lo, hi)
    width = hi - lo
    position = [[rng.uniform(lo, hi) for _ in range(dimensions)] for _ in range(particles)]
    velocity = [[rng.uniform(-0.1, 0.1) * width for _ in range(dimensions)] for _ in range(particles)]
    personal_best = [p[:] for p in position]
    personal_value = [f(p) for p in position]
    g = min(range(particles), key=personal_value.__getitem__)
    global_best, global_value = personal_best[g][:], personal_value[g]
    evaluations = particles

    for _ in range(iterations):
        for k in range(particles):
            for d in range(dimensions):
                v = (inertia * velocity[k][d]
                     + cognitive * rng.random() * (personal_best[k][d] - position[k][d])
                     + social * rng.random() * (global_best[d] - position[k][d]))
                velocity[k][d] = _clamp(v, -width, width)
                position[k][d] = _clamp(position[k][d] + velocity[k][d], lo, hi)
            value = f(position[k])
            evaluations += 1
            if value < personal_value[k]:
                personal_best[k], personal_value[k] = position[k][:], value
                if value < global_value:
                    global_best, global_value = position[k][:], value
    return Optimum(tuple(global_best), global_value, evaluations)
