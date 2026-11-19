package io.github.divyam1101.ai.optimization;

import java.util.ArrayDeque;
import java.util.Deque;
import java.util.HashSet;
import java.util.Set;
import java.util.function.DoubleBinaryOperator;

/**
 * Tabu search for minimising a function of two variables over a discretised square grid.
 *
 * <p>At every step the search moves to the best non-tabu neighbour (4-neighbourhood), even if it
 * is worse than the current point. Recently visited grid points are kept in a fixed-size tabu list,
 * which stops the search cycling back into the same local minimum.
 *
 * <p>Neighbours are computed on the fly, so no grid of objects is allocated, and the search keeps
 * no mutable state between calls.
 */
public final class TabuSearch {

    private record GridPoint(int i, int j) {
    }

    private final double lo;
    private final double step;
    private final int size;
    private final int tenure;
    private final int iterations;

    /**
     * @param lo         lower bound for both coordinates
     * @param hi         upper bound for both coordinates
     * @param step       grid spacing
     * @param tenure     maximum length of the tabu list
     * @param iterations number of moves to make
     */
    public TabuSearch(double lo, double hi, double step, int tenure, int iterations) {
        LocalSearch.requireInterval(lo, hi);
        this.lo = lo;
        this.step = step;
        this.size = (int) Math.round((hi - lo) / step) + 1;
        this.tenure = tenure;
        this.iterations = iterations;
    }

    public Optimum minimize(DoubleBinaryOperator f, double startX, double startY) {
        GridPoint current = new GridPoint(index(startX), index(startY));
        GridPoint best = current;
        double bestValue = value(f, best);
        long evaluations = 1;

        Deque<GridPoint> tabuQueue = new ArrayDeque<>();
        Set<GridPoint> tabuSet = new HashSet<>();

        for (int it = 0; it < iterations; it++) {
            remember(current, tabuQueue, tabuSet);

            GridPoint next = null;
            double nextValue = Double.POSITIVE_INFINITY;
            GridPoint fallback = null; // best neighbour ignoring the tabu list
            double fallbackValue = Double.POSITIVE_INFINITY;
            for (GridPoint candidate : neighbours(current)) {
                double v = value(f, candidate);
                evaluations++;
                if (v < fallbackValue) {
                    fallback = candidate;
                    fallbackValue = v;
                }
                if (!tabuSet.contains(candidate) && v < nextValue) {
                    next = candidate;
                    nextValue = v;
                }
            }
            if (next == null) {
                // every neighbour is tabu: rather than stopping, take the best one anyway
                next = fallback;
                nextValue = fallbackValue;
            }
            current = next;
            if (nextValue < bestValue) {
                best = next;
                bestValue = nextValue;
            }
        }
        return new Optimum(new double[] {coordinate(best.i()), coordinate(best.j())}, bestValue, evaluations);
    }

    private void remember(GridPoint p, Deque<GridPoint> queue, Set<GridPoint> set) {
        if (set.add(p)) {
            queue.addLast(p);
            if (queue.size() > tenure) {
                set.remove(queue.removeFirst());
            }
        }
    }

    private GridPoint[] neighbours(GridPoint p) {
        GridPoint[] all = {
            new GridPoint(p.i() - 1, p.j()), new GridPoint(p.i() + 1, p.j()),
            new GridPoint(p.i(), p.j() - 1), new GridPoint(p.i(), p.j() + 1)
        };
        int n = 0;
        for (GridPoint g : all) {
            if (g.i() >= 0 && g.i() < size && g.j() >= 0 && g.j() < size) {
                all[n++] = g;
            }
        }
        return java.util.Arrays.copyOf(all, n);
    }

    private double value(DoubleBinaryOperator f, GridPoint p) {
        return f.applyAsDouble(coordinate(p.i()), coordinate(p.j()));
    }

    private double coordinate(int index) {
        return lo + index * step;
    }

    private int index(double coordinate) {
        int i = (int) Math.round((coordinate - lo) / step);
        return Math.max(0, Math.min(size - 1, i));
    }
}
