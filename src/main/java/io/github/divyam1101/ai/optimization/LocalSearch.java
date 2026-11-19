package io.github.divyam1101.ai.optimization;

import java.util.Random;
import java.util.function.DoubleUnaryOperator;

/**
 * Baseline one-dimensional minimisers: exhaustive grid search, random search and hill climbing.
 *
 * <p>All methods minimise. To maximise {@code f}, minimise {@code x -> -f(x)}.
 */
public final class LocalSearch {

    private LocalSearch() {
    }

    /** Evaluates {@code f} on an evenly spaced grid over {@code [lo, hi]}. */
    public static Optimum bruteForce(DoubleUnaryOperator f, double lo, double hi, double step) {
        requireInterval(lo, hi);
        if (step <= 0) {
            throw new IllegalArgumentException("step must be positive");
        }
        long steps = (long) Math.floor((hi - lo) / step);
        double bestX = lo;
        double best = f.applyAsDouble(lo);
        for (long i = 1; i <= steps; i++) {
            // compute x from the index to avoid floating-point drift from repeated addition
            double x = lo + i * step;
            double value = f.applyAsDouble(x);
            if (value < best) {
                best = value;
                bestX = x;
            }
        }
        return Optimum.of1D(bestX, best, steps + 1);
    }

    /** Samples {@code iterations} uniformly random points in {@code [lo, hi]}. */
    public static Optimum randomSearch(DoubleUnaryOperator f, double lo, double hi, int iterations, Random random) {
        requireInterval(lo, hi);
        double bestX = lo + (hi - lo) * random.nextDouble();
        double best = f.applyAsDouble(bestX);
        for (int i = 1; i < iterations; i++) {
            double x = lo + (hi - lo) * random.nextDouble();
            double value = f.applyAsDouble(x);
            if (value < best) {
                best = value;
                bestX = x;
            }
        }
        return Optimum.of1D(bestX, best, iterations);
    }

    /**
     * Steepest-descent hill climbing: from {@code start}, move to the better of the two neighbours
     * {@code x - step} and {@code x + step} until neither improves. Unlike a one-directional climb,
     * this works whichever side of the optimum the search starts on.
     */
    public static Optimum hillClimbing(DoubleUnaryOperator f, double start, double step,
                                       double lo, double hi, int maxIterations) {
        requireInterval(lo, hi);
        double x = clamp(start, lo, hi);
        double value = f.applyAsDouble(x);
        long evaluations = 1;
        for (int i = 0; i < maxIterations; i++) {
            double left = clamp(x - step, lo, hi);
            double right = clamp(x + step, lo, hi);
            double fLeft = f.applyAsDouble(left);
            double fRight = f.applyAsDouble(right);
            evaluations += 2;
            if (fLeft < value && fLeft <= fRight) {
                x = left;
                value = fLeft;
            } else if (fRight < value) {
                x = right;
                value = fRight;
            } else {
                break; // local optimum reached
            }
        }
        return Optimum.of1D(x, value, evaluations);
    }

    static double clamp(double x, double lo, double hi) {
        return Math.max(lo, Math.min(hi, x));
    }

    static void requireInterval(double lo, double hi) {
        if (!(lo < hi)) {
            throw new IllegalArgumentException("require lo < hi, got [" + lo + ", " + hi + "]");
        }
    }
}
