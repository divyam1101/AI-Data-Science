# AI Algorithms in Java

[![build](https://github.com/divyam1101/AI-Data-Science/actions/workflows/build.yml/badge.svg)](https://github.com/divyam1101/AI-Data-Science/actions/workflows/build.yml)

Clean, tested Java 17 implementations of classic artificial-intelligence algorithms: graph search, A* path-finding, local search and metaheuristics, a genetic algorithm, the travelling salesman problem, a perceptron, and an unbeatable tic-tac-toe AI.

> I first wrote these while following Balazs Holczer's *Artificial Intelligence* course. In 2026 I rewrote them as one Maven project, fixed the bugs listed below, and added unit tests and CI.

## Contents

| Package | Algorithm | Notes |
|---|---|---|
| `search` | Breadth-first search | Traversal and shortest path (unweighted) |
| `search` | Depth-first search | Iterative (explicit stack), covers disconnected graphs |
| `search` | Iterative deepening DFS | Shallowest path with linear memory; cycle-safe; terminates if the goal is unreachable |
| `search` | A* on a grid | Manhattan heuristic, lazy-deletion priority queue, provably optimal paths |
| `optimization` | Brute force, random search, hill climbing | 1-D baselines; the hill climber moves in both directions |
| `optimization` | Simulated annealing | Gaussian neighbour proposals that narrow as it cools; Metropolis acceptance |
| `optimization` | Tabu search | 2-D grid, bounded tabu list (FIFO + hash set), neighbours computed on the fly |
| `optimization` | Particle swarm optimisation | Any dimension, velocity clamping, standard coefficients |
| `genetic` | Genetic algorithm | Tournament selection, uniform crossover, mutation, elitism; integer or binary-encoded problems |
| `tsp` | Travelling salesman | Simulated annealing with 2-opt moves and O(1) move evaluation |
| `neural` | Perceptron | Learnable bias; stops after the first epoch with no errors |
| `games` | Tic-tac-toe | Negamax with alpha-beta pruning; plays perfectly |

## Run it

Requires JDK 17+ and Maven.

```bash
mvn verify                                   # compile and run the 28 unit tests
java -jar target/ai-algorithms-2.0.0.jar     # run every algorithm on an example problem
java -cp target/classes io.github.divyam1101.ai.games.TicTacToeConsole   # play tic-tac-toe
```

Example demo output:

```
A* path (7 moves): (3,3) -> (3,2) -> (2,2) -> (1,2) -> (1,3) -> (1,4) -> (1,5) -> (1,6)
Hill climbing (x0=-2): x = -2.0     <- stuck at a local optimum
Simulated annealing:   x = 1.221   (true minimum 1.2228)
Tabu search:           (-0.7, 0.0), f = -0.3947 (best grid point; true minimum -0.3967 at x = -0.653)
Genetic algorithm:     target sequence found in 10 generations
TSP (30 cities):       random tour 1636 -> annealed tour 469
```

## What changed in the 2026 refactor

Bugs fixed in the original course code:

- **A\*** overwrote each neighbour's cost and parent even when a cheaper route was already known, and changed priorities of nodes already in the queue. It now keeps the best known cost per cell and returns optimal paths, which a test checks.
- **Simulated annealing (maximisation version)** had its acceptance test inverted, so it accepted *every* worse move. It also proposed uniformly random points, which made it plain random search. It now uses a correct Metropolis criterion with local moves.
- **Particle swarm**: `checkBestSolution` assigned to a method parameter, so the global best was never updated at start-up, and `toString` read `position[2]` on a 2-D problem, which throws an exception.
- **Tabu search** removed tabu states from each node's own neighbour list (`removeAll`), permanently damaging the search graph. It also needed Apache Commons Collections, which wasn't included in the repo, so it didn't build.
- **TSP** added up tour lengths in an `int`, truncating every edge length.
- **Perceptron** had no bias term and stopped based only on the last sample's error. It was also mislabelled "backpropagation".
- **Iterative deepening** searched for the start node itself and looped forever when the goal was unreachable.
- **DFS** demo had its only call commented out, and the `LocalSearch/TabuSearch.java` stub didn't compile.

Engineering changes:

- One Maven project instead of 13 separate Eclipse projects. Compiled `.class` files and IDE metadata are removed and listed in `.gitignore`.
- Duplicate copies of the same algorithms (`GeneticAlgorithms`, `NEW_CODE/GeneticAlgorithmsI`, `SimulatedAnnealing`, `NEW_CODE/SimulatedAnnealingI`, and so on) merged into single reusable classes.
- Algorithms take the objective function as a parameter (`DoubleUnaryOperator` and similar) instead of hard-coding it, accept a `Random` so results can be reproduced, and return results instead of printing them.
- 28 JUnit 5 tests and a GitHub Actions workflow that runs them on every push.
