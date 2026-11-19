package io.github.divyam1101.ai.genetic;

import java.util.Arrays;
import java.util.Comparator;
import java.util.Random;
import java.util.function.ToDoubleFunction;

/**
 * A generic genetic algorithm over fixed-length integer chromosomes.
 *
 * <p>Uses tournament selection, uniform crossover, per-gene random-reset mutation and elitism.
 * Elitism (copying the best individuals unchanged into the next generation) guarantees the best
 * fitness never gets worse from one generation to the next.
 *
 * <p>Fitness is <em>maximised</em>.
 */
public final class GeneticAlgorithm {

    /** The result of a run. */
    public record Result(int[] best, double fitness, int generations) {
        @Override
        public String toString() {
            return "Result[best=" + Arrays.toString(best) + ", fitness=" + fitness + ", generations=" + generations + "]";
        }
    }

    private final int populationSize;
    private final int chromosomeLength;
    private final int alleles;
    private final double mutationRate;
    private final int tournamentSize;
    private final int eliteCount;

    /**
     * @param populationSize   number of individuals per generation
     * @param chromosomeLength genes per individual
     * @param alleles          each gene takes a value in {@code [0, alleles)}
     * @param mutationRate     probability that each gene is reset to a random value
     * @param tournamentSize   individuals compared in each selection tournament
     * @param eliteCount       best individuals copied unchanged to the next generation
     */
    public GeneticAlgorithm(int populationSize, int chromosomeLength, int alleles,
                            double mutationRate, int tournamentSize, int eliteCount) {
        if (eliteCount >= populationSize) {
            throw new IllegalArgumentException("eliteCount must be smaller than populationSize");
        }
        this.populationSize = populationSize;
        this.chromosomeLength = chromosomeLength;
        this.alleles = alleles;
        this.mutationRate = mutationRate;
        this.tournamentSize = tournamentSize;
        this.eliteCount = eliteCount;
    }

    /**
     * Evolves until an individual reaches {@code targetFitness} or {@code maxGenerations} pass.
     */
    public Result run(ToDoubleFunction<int[]> fitness, double targetFitness, int maxGenerations, Random random) {
        int[][] population = new int[populationSize][chromosomeLength];
        for (int[] individual : population) {
            for (int g = 0; g < chromosomeLength; g++) {
                individual[g] = random.nextInt(alleles);
            }
        }

        double[] scores = evaluate(population, fitness);
        int generation = 0;
        while (generation < maxGenerations && max(scores) < targetFitness) {
            population = nextGeneration(population, scores, random);
            scores = evaluate(population, fitness);
            generation++;
        }
        int bestIndex = argMax(scores);
        return new Result(population[bestIndex].clone(), scores[bestIndex], generation);
    }

    private int[][] nextGeneration(int[][] population, double[] scores, Random random) {
        int[][] next = new int[populationSize][];
        Integer[] ranked = rankDescending(scores);
        for (int e = 0; e < eliteCount; e++) {
            next[e] = population[ranked[e]].clone();
        }
        for (int i = eliteCount; i < populationSize; i++) {
            int[] child = crossover(tournament(population, scores, random), tournament(population, scores, random), random);
            mutate(child, random);
            next[i] = child;
        }
        return next;
    }

    private int[] tournament(int[][] population, double[] scores, Random random) {
        int best = random.nextInt(populationSize);
        for (int k = 1; k < tournamentSize; k++) {
            int challenger = random.nextInt(populationSize);
            if (scores[challenger] > scores[best]) {
                best = challenger;
            }
        }
        return population[best];
    }

    private int[] crossover(int[] a, int[] b, Random random) {
        int[] child = new int[chromosomeLength];
        for (int g = 0; g < chromosomeLength; g++) {
            child[g] = random.nextBoolean() ? a[g] : b[g];
        }
        return child;
    }

    private void mutate(int[] chromosome, Random random) {
        for (int g = 0; g < chromosomeLength; g++) {
            if (random.nextDouble() < mutationRate) {
                chromosome[g] = random.nextInt(alleles);
            }
        }
    }

    private static double[] evaluate(int[][] population, ToDoubleFunction<int[]> fitness) {
        double[] scores = new double[population.length];
        for (int i = 0; i < population.length; i++) {
            scores[i] = fitness.applyAsDouble(population[i]);
        }
        return scores;
    }

    private static Integer[] rankDescending(double[] scores) {
        Integer[] idx = new Integer[scores.length];
        for (int i = 0; i < idx.length; i++) {
            idx[i] = i;
        }
        Arrays.sort(idx, Comparator.comparingDouble((Integer i) -> scores[i]).reversed());
        return idx;
    }

    private static double max(double[] values) {
        return values[argMax(values)];
    }

    private static int argMax(double[] values) {
        int best = 0;
        for (int i = 1; i < values.length; i++) {
            if (values[i] > values[best]) {
                best = i;
            }
        }
        return best;
    }

    // ---------------------------------------------------------------- ready-made problems

    /** Fitness for matching a target sequence: the number of genes equal to the target. */
    public static ToDoubleFunction<int[]> matchTarget(int[] target) {
        int[] copy = target.clone();
        return chromosome -> {
            int score = 0;
            for (int i = 0; i < copy.length; i++) {
                if (chromosome[i] == copy[i]) {
                    score++;
                }
            }
            return score;
        };
    }

    /**
     * Decodes a binary chromosome (genes in {0, 1}, most significant bit first) to a real number
     * in {@code [lo, hi]}.
     */
    public static double decode(int[] bits, double lo, double hi) {
        long value = 0;
        for (int bit : bits) {
            value = (value << 1) | bit;
        }
        long max = (1L << bits.length) - 1;
        return lo + (hi - lo) * value / max;
    }
}
