package io.github.divyam1101.ai.optimization;

import java.util.Random;
import java.util.function.DoubleUnaryOperator;

/**
 * Simulated annealing for one-dimensional minimisation.
 *
 * <p>Each step proposes a neighbour drawn from a Gaussian around the current point, with a width
 * that shrinks as the temperature falls. A better neighbour is always accepted; a worse one is
 * accepted with the Metropolis probability {@code exp(-(fNew - fCurrent) / T)}. That lets the
 * search escape local minima early on and settle down as it cools.
 */
public final class SimulatedAnnealing {

    private final double startTemperature;
    private final double minTemperature;
    private final double coolingRate;

    /**
     * @param startTemperature initial temperature (positive)
     * @param minTemperature   stop once the temperature drops below this value
     * @param coolingRate      geometric cooling factor per step, in (0, 1)
     */
    public SimulatedAnnealing(double startTemperature, double minTemperature, double coolingRate) {
        if (startTemperature <= minTemperature || minTemperature <= 0) {
            throw new IllegalArgumentException("require startTemperature > minTemperature > 0");
        }
        if (coolingRate <= 0 || coolingRate >= 1) {
            throw new IllegalArgumentException("coolingRate must be in (0, 1)");
        }
        this.startTemperature = startTemperature;
        this.minTemperature = minTemperature;
        this.coolingRate = coolingRate;
    }

    public Optimum minimize(DoubleUnaryOperator f, double lo, double hi, Random random) {
        LocalSearch.requireInterval(lo, hi);
        double width = hi - lo;
        double current = lo + width * random.nextDouble();
        double currentValue = f.applyAsDouble(current);
        double best = current;
        double bestValue = currentValue;
        long evaluations = 1;

        for (double t = startTemperature; t > minTemperature; t *= 1 - coolingRate) {
            double stepScale = width * 0.25 * (t / startTemperature) + 1e-6;
            double candidate = LocalSearch.clamp(current + random.nextGaussian() * stepScale, lo, hi);
            double candidateValue = f.applyAsDouble(candidate);
            evaluations++;

            if (acceptanceProbability(currentValue, candidateValue, t) > random.nextDouble()) {
                current = candidate;
                currentValue = candidateValue;
            }
            if (currentValue < bestValue) {
                best = current;
                bestValue = currentValue;
            }
        }
        return Optimum.of1D(best, bestValue, evaluations);
    }

    /** Metropolis criterion for minimisation. */
    static double acceptanceProbability(double currentValue, double candidateValue, double temperature) {
        if (candidateValue < currentValue) {
            return 1.0;
        }
        return Math.exp((currentValue - candidateValue) / temperature);
    }
}
