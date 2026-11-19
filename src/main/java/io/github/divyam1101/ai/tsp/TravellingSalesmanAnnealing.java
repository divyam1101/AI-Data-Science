package io.github.divyam1101.ai.tsp;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

/**
 * Solves the travelling salesman problem approximately with simulated annealing and
 * 2-opt moves (reversing a segment of the tour), which is a much stronger neighbourhood than
 * swapping two random cities.
 */
public final class TravellingSalesmanAnnealing {

    /** A city on the plane. */
    public record City(double x, double y) {
        public double distanceTo(City other) {
            return Math.hypot(x - other.x, y - other.y);
        }
    }

    /** A closed tour visiting every city once. */
    public record Tour(List<City> cities, double length) {
        public Tour {
            cities = List.copyOf(cities);
        }
    }

    private final double startTemperature;
    private final double minTemperature;
    private final double coolingRate;

    public TravellingSalesmanAnnealing(double startTemperature, double minTemperature, double coolingRate) {
        if (startTemperature <= minTemperature || minTemperature <= 0 || coolingRate <= 0 || coolingRate >= 1) {
            throw new IllegalArgumentException("invalid annealing schedule");
        }
        this.startTemperature = startTemperature;
        this.minTemperature = minTemperature;
        this.coolingRate = coolingRate;
    }

    public Tour solve(List<City> cities, Random random) {
        if (cities.size() < 4) {
            return new Tour(cities, length(cities)); // every tour of < 4 cities is optimal
        }
        List<City> current = new ArrayList<>(cities);
        Collections.shuffle(current, random);
        double currentLength = length(current);
        List<City> best = new ArrayList<>(current);
        double bestLength = currentLength;
        int n = current.size();

        for (double t = startTemperature; t > minTemperature; t *= 1 - coolingRate) {
            // choose a segment [i, j] to reverse
            int i = random.nextInt(n - 1);
            int j = i + 1 + random.nextInt(n - i - 1);
            double delta = twoOptDelta(current, i, j);
            if (delta < 0 || Math.exp(-delta / t) > random.nextDouble()) {
                Collections.reverse(current.subList(i, j + 1));
                currentLength += delta;
                if (currentLength < bestLength - 1e-12) {
                    bestLength = currentLength;
                    best = new ArrayList<>(current);
                }
            }
        }
        return new Tour(best, length(best));
    }

    /** Change in tour length from reversing positions {@code i..j} (inclusive), in O(1). */
    static double twoOptDelta(List<City> tour, int i, int j) {
        int n = tour.size();
        City a = tour.get((i - 1 + n) % n);
        City b = tour.get(i);
        City c = tour.get(j);
        City d = tour.get((j + 1) % n);
        if (a == c || b == d) {
            return 0; // reversing the whole tour leaves its length unchanged
        }
        return a.distanceTo(c) + b.distanceTo(d) - a.distanceTo(b) - c.distanceTo(d);
    }

    public static double length(List<City> tour) {
        double total = 0;
        for (int k = 0; k < tour.size(); k++) {
            total += tour.get(k).distanceTo(tour.get((k + 1) % tour.size()));
        }
        return total;
    }
}
