package io.github.divyam1101.ai.optimization;

import java.util.Random;
import java.util.function.ToDoubleFunction;

/**
 * Particle swarm optimisation (PSO) for minimising a function over a box {@code [lo, hi]^d}.
 *
 * <p>Uses the standard constriction-style coefficients ({@code w = 0.729}, {@code c1 = c2 = 1.494})
 * and clamps velocities to the width of the search box so particles cannot fly off.
 */
public final class ParticleSwarmOptimization {

    private static final double INERTIA = 0.729;
    private static final double COGNITIVE = 1.49445;
    private static final double SOCIAL = 1.49445;

    private final int particles;
    private final int iterations;

    public ParticleSwarmOptimization(int particles, int iterations) {
        if (particles < 1 || iterations < 1) {
            throw new IllegalArgumentException("particles and iterations must be positive");
        }
        this.particles = particles;
        this.iterations = iterations;
    }

    public Optimum minimize(ToDoubleFunction<double[]> f, int dimensions, double lo, double hi, Random random) {
        LocalSearch.requireInterval(lo, hi);
        double width = hi - lo;
        double[][] position = new double[particles][dimensions];
        double[][] velocity = new double[particles][dimensions];
        double[][] personalBest = new double[particles][];
        double[] personalBestValue = new double[particles];
        double[] globalBest = null;
        double globalBestValue = Double.POSITIVE_INFINITY;
        long evaluations = 0;

        for (int p = 0; p < particles; p++) {
            for (int d = 0; d < dimensions; d++) {
                position[p][d] = lo + width * random.nextDouble();
                velocity[p][d] = (random.nextDouble() * 2 - 1) * width * 0.1;
            }
            personalBest[p] = position[p].clone();
            personalBestValue[p] = f.applyAsDouble(position[p]);
            evaluations++;
            if (personalBestValue[p] < globalBestValue) {
                globalBestValue = personalBestValue[p];
                globalBest = position[p].clone();
            }
        }

        for (int it = 0; it < iterations; it++) {
            for (int p = 0; p < particles; p++) {
                for (int d = 0; d < dimensions; d++) {
                    double v = INERTIA * velocity[p][d]
                            + COGNITIVE * random.nextDouble() * (personalBest[p][d] - position[p][d])
                            + SOCIAL * random.nextDouble() * (globalBest[d] - position[p][d]);
                    velocity[p][d] = LocalSearch.clamp(v, -width, width);
                    position[p][d] = LocalSearch.clamp(position[p][d] + velocity[p][d], lo, hi);
                }
                double value = f.applyAsDouble(position[p]);
                evaluations++;
                if (value < personalBestValue[p]) {
                    personalBestValue[p] = value;
                    personalBest[p] = position[p].clone();
                    if (value < globalBestValue) {
                        globalBestValue = value;
                        globalBest = position[p].clone();
                    }
                }
            }
        }
        return new Optimum(globalBest, globalBestValue, evaluations);
    }
}
