import random

import pytest

from ai_algorithms import genetic, tsp
from ai_algorithms.perceptron import Perceptron
from ai_algorithms.tictactoe import EMPTY, Board, opponent

# ------------------------------------------------------------------ genetic algorithm

def test_ga_evolves_target_sequence():
    target = list(range(10)) * 2
    ga = genetic.GeneticAlgorithm(population_size=100, chromosome_length=20, alleles=10, mutation_rate=0.02)
    result = ga.run(genetic.match_target(target), random.Random(11), target=20)
    assert list(result.best) == target and result.fitness == 20


def test_ga_minimises_real_function_via_binary_encoding():
    ga = genetic.GeneticAlgorithm(population_size=60, chromosome_length=16, alleles=2, mutation_rate=1 / 16,
                                  tournament_size=3)
    result = ga.run(lambda bits: -((genetic.decode(bits, 0, 10) - 2) ** 2 + 3), random.Random(5), max_generations=150)
    assert genetic.decode(result.best, 0, 10) == pytest.approx(2.0, abs=0.01)


def test_binary_decoding_covers_interval():
    assert genetic.decode([0, 0, 0, 0], 0, 10) == 0.0
    assert genetic.decode([1, 1, 1, 1], 0, 10) == 10.0


def test_ga_rejects_too_many_elites():
    with pytest.raises(ValueError):
        genetic.GeneticAlgorithm(population_size=2, chromosome_length=4, alleles=2, mutation_rate=0.1, elite=2)


# ------------------------------------------------------------------ travelling salesman

def test_annealing_solves_square_optimally():
    cities = [tsp.City(*p) for p in [(0, 0), (5, 0), (10, 0), (10, 5), (10, 10), (5, 10), (0, 10), (0, 5)]]
    tour = tsp.solve(cities, random.Random(1), start_temperature=10, cooling_rate=1e-3)
    assert tour.length == pytest.approx(40.0)
    assert sorted(tour.cities) == sorted(cities)


def test_annealing_beats_random_tour():
    rng = random.Random(2)
    cities = [tsp.City(rng.uniform(0, 100), rng.uniform(0, 100)) for _ in range(40)]
    assert tsp.solve(cities, rng).length < 0.5 * tsp.tour_length(cities)


# ------------------------------------------------------------------ perceptron

INPUTS = [(0, 0), (0, 1), (1, 0), (1, 1)]


@pytest.mark.parametrize("targets", [[0, 0, 0, 1], [0, 1, 1, 1]], ids=["AND", "OR"])
def test_perceptron_learns_linearly_separable(targets):
    p = Perceptron(2)
    assert p.train(INPUTS, targets) is not None
    assert [p.predict(x) for x in INPUTS] == targets


def test_perceptron_cannot_learn_xor():
    assert Perceptron(2).train(INPUTS, [0, 1, 1, 0], max_epochs=1_000) is None


# ------------------------------------------------------------------ tic-tac-toe

def test_ai_takes_the_winning_move():
    assert Board("XX-OO----").best_move("X") == 2


def test_ai_blocks_the_opponent():
    assert Board("OO-X-----").best_move("X") == 2


def test_perfect_play_is_a_draw():
    board, player = Board(), "X"
    while not board.is_over():
        board.play(board.best_move(player), player)
        player = opponent(player)
    assert board.winner() == EMPTY


def test_illegal_moves_are_rejected():
    board = Board("X--------")
    with pytest.raises(ValueError):
        board.play(0, "O")
    with pytest.raises(ValueError):
        board.play(9, "O")
    with pytest.raises(ValueError):
        Board("XXXXXXXXQ")
