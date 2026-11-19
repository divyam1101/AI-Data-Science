# AI Algorithms in Java

Java implementations of classic artificial-intelligence search and optimisation algorithms, written while studying AI fundamentals.

> Based on the course material of Balazs Holczer's *Artificial Intelligence* course (package `com.balazsholczer.*`). This repository is my study and practice code.

## Contents

**Graph search** (`GraphAlgorithms/`)
- Breadth-first search
- Depth-first search
- Iterative deepening search
- A* search

**Local search & metaheuristics**
- Hill climbing, random search, brute-force search (`LocalSearch/`, `NEW_CODE/`)
- Tabu search (`TabuSearch/`)
- Simulated annealing (`SimulatedAnnealing/`, `NEW_CODE/SimulatedAnnealingI|II`)
- Genetic algorithms (`GeneticAlgorithms/`, `NEW_CODE/GeneticAlgorithmsI|II`)
- Particle swarm optimisation (`PSO/`)

**Other**
- Backpropagation neural network from scratch (`NeuralNetwork/`)
- Tic-Tac-Toe game AI (`TicTacToe/`)

## Running

Each folder is a standalone Eclipse/Java project. Import it into Eclipse or IntelliJ, or compile from the command line:

```bash
javac -d out $(find GraphAlgorithms/AStarSearch/src -name "*.java")
java -cp out com.balazsholczer.astar.App
```
