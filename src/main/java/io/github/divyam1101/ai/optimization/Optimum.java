package io.github.divyam1101.ai.optimization;

import java.util.Arrays;

/**
 * The best point an optimiser found and its objective value.
 *
 * @param point       coordinates of the best point (length 1 for one-dimensional problems)
 * @param value       objective value at {@code point}
 * @param evaluations number of objective evaluations used
 */
public record Optimum(double[] point, double value, long evaluations) {

    public Optimum {
        point = point.clone();
    }

    public static Optimum of1D(double x, double value, long evaluations) {
        return new Optimum(new double[] {x}, value, evaluations);
    }

    /** Convenience accessor for one-dimensional problems. */
    public double x() {
        return point[0];
    }

    @Override
    public double[] point() {
        return point.clone();
    }

    @Override
    public String toString() {
        return String.format("Optimum[point=%s, value=%.6f, evaluations=%d]",
                Arrays.toString(point), value, evaluations);
    }
}
