# AI Algorithms in Python

[![build](https://github.com/divyam1101/AI-Data-Science/actions/workflows/build.yml/badge.svg)](https://github.com/divyam1101/AI-Data-Science/actions/workflows/build.yml)

Clean, tested, dependency-free Python implementations of classic artificial-intelligence algorithms: graph search, A* path-finding, local search and metaheuristics, a genetic algorithm, the travelling salesman problem, a perceptron, and an unbeatable tic-tac-toe AI.

> I first wrote these in Java while following Balazs Holczer's *Artificial Intelligence* course. In 2026 I fixed the bugs in the original code, added tests, and migrated it to Python. The Java version is in the git history (commit `d408c6f`).

## Contents

| Module | Algorithm | Notes |
|---|---|---|
| `search` | Breadth-first search | Traversal and shortest path on unweighted graphs |
| `search` | Depth-first search | Explicit stack (no recursion limit); covers disconnected graphs |
| `search` | Iterative deepening DFS | Shallowest path with linear memory; cycle-safe; terminates if the goal is unreachable |
| `search` | A* on a grid | Manhattan heuristic, heap with lazy deletion, provably optimal paths |
| `optimization` | Brute force, random search, hill climbing | 1-D baselines; the hill climber moves in both directions |
| `optimization` | Simulated annealing | Gaussian neighbour proposals that narrow as it cools; Metropolis acceptance |
| `optimization` | Tabu search | 2-D grid, bounded tabu list, best-move fallback when boxed in |
| `optimization` | Particle swarm optimisation | Any dimension, velocity clamping, standard coefficients |
| `genetic` | Genetic algorithm | Tournament selection, uniform crossover, mutation, elitism; integer or binary-encoded problems |
| `tsp` | Travelling salesman | Simulated annealing with 2-opt moves evaluated in O(1) |
| `perceptron` | Perceptron | Learnable bias; stops after the first epoch with no errors |
| `tictactoe` | Tic-tac-toe | Negamax with alpha-beta pruning and memoisation; plays perfectly |

Every optimiser takes the objective function and a `random.Random` as parameters, so results can be reproduced, and returns its result instead of printing it.

```python
import random
from ai_algorithms.optimization import simulated_annealing

result = simulated_annealing(lambda x: (x - 0.3) ** 3 - 5 * x + x * x - 2, -2, 2, random.Random(0))
print(result)   # x = (1.2228), f = -5.832941, 1,839 evaluations
```

## Run it

Python 3.10+, no third-party dependencies.

```bash
pip install -e ".[dev]"
pytest -q                          # 36 tests, ~2 seconds
python -m ai_algorithms.demo       # runs every algorithm on an example problem
python -m ai_algorithms.tictactoe  # play against the AI
```

Example demo output:

```
A* path (7 moves):     (3,3) -> (3,2) -> (2,2) -> (1,2) -> (1,3) -> (1,4) -> (1,5) -> (1,6)
Hill climbing (x0=-2): x = (-2.0000)   <- stuck at a local optimum
Simulated annealing:   x = (1.2227)    (true minimum 1.2228)
Tabu search:           x = (-0.7000, 0.0000), f = -0.394665 (best grid point; true minimum -0.3967)
Genetic algorithm:     target sequence found in 20 generations
TSP (30 cities):       random tour 1552 -> annealed tour 443
```

## History of fixes

Bugs found in the original course code (fixed in the 2026 Java refactor, and carried over to Python):

- **A\*** overwrote each neighbour's cost and parent even when a cheaper route was already known. It now keeps the best known cost per cell and returns optimal paths.
- **Simulated annealing (maximisation version)** had its acceptance test inverted, so it accepted every worse move. It also proposed uniformly random points, which made it plain random search.
- **Particle swarm** never updated the global best at start-up (it assigned to a method parameter), and its `toString` read `position[2]` on a 2-D problem, which threw an exception.
- **Tabu search** permanently removed tabu states from each node's neighbour list, and relied on a dependency that wasn't in the repo.
- **TSP** added up tour lengths in an `int`, truncating every edge length.
- **Perceptron** had no bias term and stopped based only on the last sample's error.
- **Iterative deepening** searched for the start node itself and looped forever when the goal was unreachable.

Engineering changes: 13 separate Eclipse projects with committed `.class` files became one package; duplicated course variants were merged; tests and CI (ruff + pytest on Python 3.10 and 3.12) were added.
