"""Run every algorithm once on a small example problem: ``python -m ai_algorithms.demo``."""

from __future__ import annotations

import math
import random

from . import genetic, search, tsp
from . import optimization as opt
from .perceptron import Perceptron


def cubic(x: float) -> float:
    """Global minimum on [-2, 2] at x ~ 1.2228; rises in both feasible directions from x = -2."""
    return (x - 0.3) ** 3 - 5 * x + x * x - 2


def section(title: str) -> None:
    print(f"\n== {title} ==")


def main() -> None:
    rng = random.Random(42)

    section("Graph search")
    tree = search.Graph()
    for a, b in [("A", "B"), ("A", "C"), ("B", "D"), ("B", "E"), ("C", "F"), ("C", "G")]:
        tree.add_edge(a, b)
    print("BFS order:        ", search.bfs(tree, "A"))
    print("DFS order:        ", search.dfs(tree, "A"))
    print("IDDFS path A -> G:", search.iterative_deepening(tree, "A", "G", max_depth=5))

    section("A* on a grid")
    blocked = [[False] * 8 for _ in range(5)]
    blocked[1][7] = True
    for c in range(3, 8):
        blocked[2][c] = True
    path = search.GridAStar(blocked).find_path(search.Cell(3, 3), search.Cell(1, 6))
    print(f"Path ({len(path) - 1} moves):", " -> ".join(f"({r},{c})" for r, c in path))

    section("1-D optimisation of f(x) = (x - 0.3)^3 - 5x + x^2 - 2 on [-2, 2]")
    print("Brute force:          ", opt.brute_force(cubic, -2, 2, 0.001))
    print("Random search:        ", opt.random_search(cubic, -2, 2, 1_000, rng))
    print("Hill climbing (x0=0): ", opt.hill_climbing(cubic, 0, 0.001, -2, 2))
    print("Hill climbing (x0=-2):", opt.hill_climbing(cubic, -2, 0.001, -2, 2), "<- stuck at a local optimum")
    print("Simulated annealing:  ", opt.simulated_annealing(cubic, -2, 2, rng))

    section("2-D optimisation")
    print("Tabu search, exp(-x^2-y^2)*sin(x):",
          opt.tabu_search(lambda x, y: math.exp(-x * x - y * y) * math.sin(x), (0.5, 0.5), lo=-10, hi=10, step=0.1))
    print("PSO, sphere x^2+y^2:", opt.particle_swarm(lambda p: p[0] ** 2 + p[1] ** 2, 2, -10, 10, rng))

    section("Genetic algorithm: evolve a target sequence")
    target = list(range(10)) * 2
    ga = genetic.GeneticAlgorithm(population_size=100, chromosome_length=len(target), alleles=10, mutation_rate=0.02)
    print(ga.run(genetic.match_target(target), rng, target=len(target)))

    section("Travelling salesman with simulated annealing (2-opt)")
    cities = [tsp.City(rng.uniform(0, 100), rng.uniform(0, 100)) for _ in range(30)]
    tour = tsp.solve(cities, rng)
    print(f"Random tour length {tsp.tour_length(cities):.1f} -> annealed tour length {tour.length:.1f}")

    section("Perceptron learning logical AND")
    inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
    p = Perceptron(2)
    epochs = p.train(inputs, [0, 0, 0, 1])
    print(f"Converged in {epochs} epochs; predictions: {[p.predict(x) for x in inputs]}")

    section("Tic-tac-toe")
    print("Run `python -m ai_algorithms.tictactoe` to play against the minimax AI.")


if __name__ == "__main__":
    main()
