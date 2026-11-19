package io.github.divyam1101.ai;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.divyam1101.ai.games.TicTacToe;
import io.github.divyam1101.ai.genetic.GeneticAlgorithm;
import io.github.divyam1101.ai.neural.Perceptron;
import io.github.divyam1101.ai.tsp.TravellingSalesmanAnnealing;
import io.github.divyam1101.ai.tsp.TravellingSalesmanAnnealing.City;
import java.util.List;
import java.util.Random;
import org.junit.jupiter.api.Test;

class OtherAlgorithmsTest {

    // ------------------------------------------------------------ genetic algorithm

    @Test
    void geneticAlgorithmEvolvesTargetSequence() {
        int[] target = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
        GeneticAlgorithm ga = new GeneticAlgorithm(100, target.length, 10, 0.02, 5, 2);
        GeneticAlgorithm.Result result = ga.run(GeneticAlgorithm.matchTarget(target), target.length, 1_000, new Random(11));
        assertArrayEquals(target, result.best());
        assertEquals(target.length, result.fitness());
    }

    @Test
    void geneticAlgorithmMinimisesRealFunctionViaBinaryEncoding() {
        // minimise (x - 2)^2 + 3 on [0, 10] using 16-bit chromosomes; fitness is the negated objective
        GeneticAlgorithm ga = new GeneticAlgorithm(60, 16, 2, 1.0 / 16, 3, 2);
        GeneticAlgorithm.Result result = ga.run(
                bits -> -(Math.pow(GeneticAlgorithm.decode(bits, 0, 10) - 2, 2) + 3),
                Double.POSITIVE_INFINITY, 150, new Random(5));
        assertEquals(2.0, GeneticAlgorithm.decode(result.best(), 0, 10), 0.01);
    }

    @Test
    void binaryDecodingCoversInterval() {
        assertEquals(0.0, GeneticAlgorithm.decode(new int[] {0, 0, 0, 0}, 0, 10));
        assertEquals(10.0, GeneticAlgorithm.decode(new int[] {1, 1, 1, 1}, 0, 10));
    }

    // ------------------------------------------------------------ travelling salesman

    @Test
    void annealingSolvesPointsOnASquareOptimally() {
        List<City> cities = List.of(
                new City(0, 0), new City(5, 0), new City(10, 0), new City(10, 5),
                new City(10, 10), new City(5, 10), new City(0, 10), new City(0, 5));
        var tour = new TravellingSalesmanAnnealing(10, 1e-3, 1e-3).solve(cities, new Random(1));
        assertEquals(40.0, tour.length(), 1e-9);
        assertEquals(cities.size(), tour.cities().size());
        assertTrue(tour.cities().containsAll(cities));
    }

    @Test
    void annealingBeatsARandomTour() {
        Random random = new Random(2);
        List<City> cities = new java.util.ArrayList<>();
        for (int i = 0; i < 40; i++) {
            cities.add(new City(random.nextDouble() * 100, random.nextDouble() * 100));
        }
        double initial = TravellingSalesmanAnnealing.length(cities);
        var tour = new TravellingSalesmanAnnealing(100, 1e-3, 1e-4).solve(cities, random);
        assertTrue(tour.length() < 0.5 * initial, "annealed " + tour.length() + " vs random " + initial);
    }

    // ------------------------------------------------------------ perceptron

    private static final double[][] BOOLEAN_INPUTS = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};

    @Test
    void perceptronLearnsAndAndOr() {
        for (int[] targets : new int[][] {{0, 0, 0, 1}, {0, 1, 1, 1}}) {
            Perceptron p = new Perceptron(2);
            assertTrue(p.train(BOOLEAN_INPUTS, targets, 0.1, 100) > 0);
            for (int i = 0; i < 4; i++) {
                assertEquals(targets[i], p.predict(BOOLEAN_INPUTS[i]));
            }
        }
    }

    @Test
    void perceptronCannotLearnXor() {
        assertEquals(-1, new Perceptron(2).train(BOOLEAN_INPUTS, new int[] {0, 1, 1, 0}, 0.1, 1_000));
    }

    // ------------------------------------------------------------ tic-tac-toe

    @Test
    void aiTakesTheWinningMove() {
        assertEquals(2, TicTacToe.of("XX-OO----").bestMove('X'));
    }

    @Test
    void aiBlocksTheOpponent() {
        assertEquals(2, TicTacToe.of("OO-X-----").bestMove('X'));
    }

    @Test
    void perfectPlayEndsInADraw() {
        TicTacToe board = new TicTacToe();
        char player = 'X';
        while (!board.isOver()) {
            board.play(board.bestMove(player), player);
            player = TicTacToe.opponent(player);
        }
        assertEquals(TicTacToe.EMPTY, board.winner());
    }

    @Test
    void illegalMovesAreRejected() {
        TicTacToe board = TicTacToe.of("X--------");
        assertThrows(IllegalArgumentException.class, () -> board.play(0, 'O'));
        assertThrows(IllegalArgumentException.class, () -> board.play(9, 'O'));
    }
}
