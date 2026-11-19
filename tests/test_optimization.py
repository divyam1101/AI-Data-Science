import math
import random

import pytest

from ai_algorithms import optimization as opt
from ai_algorithms.demo import cubic

CUBIC_MIN_X = (-0.2 + math.sqrt(0.04 + 12 * 4.73)) / 6  # root of f'(x) = 3x^2 + 0.2x - 4.73


def test_brute_force_finds_global_minimum():
    assert opt.brute_force(cubic, -2, 2, 0.001).x == pytest.approx(CUBIC_MIN_X, abs=1e-3)


def test_random_search_gets_close():
    assert opt.random_search(cubic, -2, 2, 5_000, random.Random(1)).x == pytest.approx(CUBIC_MIN_X, abs=0.01)


@pytest.mark.parametrize("start", [0, 2])
def test_hill_climbing_works_from_either_side(start):
    assert opt.hill_climbing(cubic, start, 0.001, -2, 2).x == pytest.approx(CUBIC_MIN_X, abs=2e-3)


def test_hill_climbing_can_get_stuck_in_local_optimum():
    assert opt.hill_climbing(cubic, -2, 0.01, -2, 2).x == -2


@pytest.mark.parametrize("seed", range(5))
def test_simulated_annealing_escapes_boundary_trap(seed):
    assert opt.simulated_annealing(cubic, -2, 2, random.Random(seed)).x == pytest.approx(CUBIC_MIN_X, abs=0.05)


def test_metropolis_criterion():
    assert opt.acceptance_probability(5, 4, 1) == 1.0
    assert opt.acceptance_probability(4, 5, 1) == pytest.approx(math.exp(-1))
    assert opt.acceptance_probability(4, 5, 100) > opt.acceptance_probability(4, 5, 1)


def test_tabu_search_finds_grid_minimum():
    result = opt.tabu_search(lambda x, y: math.exp(-x * x - y * y) * math.sin(x), (0.5, 0.5), lo=-10, hi=10, step=0.1)
    assert result.point == pytest.approx((-0.7, 0.0), abs=1e-9)
    assert result.value < -0.394


def test_particle_swarm_minimises_sphere():
    result = opt.particle_swarm(lambda p: sum(v * v for v in p), 3, -10, 10, random.Random(7), iterations=300)
    assert result.value < 1e-6


def test_particle_swarm_handles_shifted_optimum():
    result = opt.particle_swarm(lambda p: (p[0] - 3) ** 2 + (p[1] + 2) ** 2, 2, -10, 10, random.Random(3),
                                particles=30, iterations=300)
    assert result.point == pytest.approx((3, -2), abs=1e-3)


def test_invalid_interval_raises():
    with pytest.raises(ValueError):
        opt.brute_force(cubic, 2, -2, 0.1)
