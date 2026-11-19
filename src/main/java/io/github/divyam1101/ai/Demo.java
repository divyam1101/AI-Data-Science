package io.github.divyam1101.ai;

import io.github.divyam1101.ai.genetic.GeneticAlgorithm;
import io.github.divyam1101.ai.neural.Perceptron;
import io.github.divyam1101.ai.optimization.LocalSearch;
import io.github.divyam1101.ai.optimization.ParticleSwarmOptimization;
import io.github.divyam1101.ai.optimization.SimulatedAnnealing;
import io.github.divyam1101.ai.optimization.TabuSearch;
import io.github.divyam1101.ai.search.BreadthFirstSearch;
import io.github.divyam1101.ai.search.DepthFirstSearch;
import io.github.divyam1101.ai.search.Graph;
import io.github.divyam1101.ai.search.GridAStar;
import io.github.divyam1101.ai.search.GridAStar.Cell;
import io.github.divyam1101.ai.search.IterativeDeepeningSearch;
import io.github.divyam1101.ai.tsp.TravellingSalesmanAnnealing;
import io.github.divyam1101.ai.tsp.TravellingSalesmanAnnealing.City;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.function.DoubleUnaryOperator;

/** Runs every algorithm once on a small example problem and prints the results. */
public final class Demo {

    private Demo() {
    }

    public static void main(String[] args) {
        Random random = new Random(42);

        section("Graph search");
        Graph<String> tree = new Graph<String>()
                .addEdge("A", "B").addEdge("A", "C")
                .addEdge("B", "D").addEdge("B", "E")
                .addEdge("C", "F").addEdge("C", "G");
        System.out.println("BFS order:          " + BreadthFirstSearch.traverse(tree, "A"));
        System.out.println("DFS order:          " + DepthFirstSearch.traverse(tree, "A"));
        System.out.println("IDDFS path A -> G:  " + IterativeDeepeningSearch.search(tree, "A", "G", 5).orElseThrow());

        section("A* on a grid");
        boolean[][] blocked = new boolean[5][8];
        blocked[1][7] = true;
        for (int c = 3; c <= 7; c++) {
            blocked[2][c] = true;
        }
        List<Cell> path = new GridAStar(blocked).findPath(new Cell(3, 3), new Cell(1, 6)).orElseThrow();
        System.out.println("Path (" + (path.size() - 1) + " moves): " + path);

        section("1-D optimisation of f(x) = (x - 0.3)^3 - 5x + x^2 - 2 on [-2, 2]");
        DoubleUnaryOperator f = x -> Math.pow(x - 0.3, 3) - 5 * x + x * x - 2;
        System.out.println("Brute force:         " + LocalSearch.bruteForce(f, -2, 2, 0.001));
        System.out.println("Random search:       " + LocalSearch.randomSearch(f, -2, 2, 1_000, random));
        System.out.println("Hill climbing (x0=0):  " + LocalSearch.hillClimbing(f, 0, 0.001, -2, 2, 100_000));
        System.out.println("Hill climbing (x0=-2): " + LocalSearch.hillClimbing(f, -2, 0.001, -2, 2, 100_000)
                + "  <- stuck at a local optimum");
        System.out.println("Simulated annealing: " + new SimulatedAnnealing(10, 1e-3, 0.005).minimize(f, -2, 2, random));

        section("2-D optimisation");
        System.out.println("Tabu search, exp(-x^2-y^2)*sin(x): "
                + new TabuSearch(-10, 10, 0.1, 400, 2_000).minimize((x, y) -> Math.exp(-x * x - y * y) * Math.sin(x), 0.5, 0.5));
        System.out.println("PSO, sphere x^2+y^2: "
                + new ParticleSwarmOptimization(20, 200).minimize(p -> p[0] * p[0] + p[1] * p[1], 2, -10, 10, random));

        section("Genetic algorithm: evolve a target sequence");
        int[] target = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
        GeneticAlgorithm ga = new GeneticAlgorithm(100, target.length, 10, 0.02, 5, 2);
        System.out.println(ga.run(GeneticAlgorithm.matchTarget(target), target.length, 1_000, random));

        section("Travelling salesman with simulated annealing (2-opt)");
        List<City> cities = new ArrayList<>();
        for (int i = 0; i < 30; i++) {
            cities.add(new City(random.nextDouble() * 100, random.nextDouble() * 100));
        }
        var tour = new TravellingSalesmanAnnealing(100, 1e-3, 1e-4).solve(cities, random);
        System.out.printf("Random tour length %.1f -> annealed tour length %.1f%n",
                TravellingSalesmanAnnealing.length(cities), tour.length());

        section("Perceptron learning logical AND");
        double[][] inputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
        Perceptron perceptron = new Perceptron(2);
        int epochs = perceptron.train(inputs, new int[] {0, 0, 0, 1}, 0.1, 100);
        System.out.println("Converged in " + epochs + " epochs; predictions: "
                + perceptron.predict(inputs[0]) + perceptron.predict(inputs[1])
                + perceptron.predict(inputs[2]) + perceptron.predict(inputs[3]));

        section("Tic-tac-toe");
        System.out.println("Run io.github.divyam1101.ai.games.TicTacToeConsole to play against the minimax AI.");
    }

    private static void section(String title) {
        System.out.println();
        System.out.println("== " + title + " ==");
    }
}
