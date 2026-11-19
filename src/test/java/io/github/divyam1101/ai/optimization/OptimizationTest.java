package io.github.divyam1101.ai.optimization;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Random;
import java.util.function.DoubleUnaryOperator;
import org.junit.jupiter.api.Test;

class OptimizationTest {

    /** Has a local maximum at x ~ -1.289 and its global minimum on [-2, 2] at x ~ 1.2228. */
    private static final DoubleUnaryOperator CUBIC = x -> Math.pow(x - 0.3, 3) - 5 * x + x * x - 2;
    private static final double CUBIC_MIN_X = (-0.2 + Math.sqrt(0.04 + 12 * 4.73)) / 6;

    @Test
    void bruteForceFindsGlobalMinimum() {
        Optimum result = LocalSearch.bruteForce(CUBIC, -2, 2, 0.001);
        assertEquals(CUBIC_MIN_X, result.x(), 1e-3);
    }

    @Test
    void randomSearchGetsClose() {
        Optimum result = LocalSearch.randomSearch(CUBIC, -2, 2, 5_000, new Random(1));
        assertEquals(CUBIC_MIN_X, result.x(), 0.01);
    }

    @Test
    void hillClimbingWorksFromEitherSide() {
        assertEquals(CUBIC_MIN_X, LocalSearch.hillClimbing(CUBIC, 0, 0.001, -2, 2, 100_000).x(), 2e-3);
        assertEquals(CUBIC_MIN_X, LocalSearch.hillClimbing(CUBIC, 2, 0.001, -2, 2, 100_000).x(), 2e-3);
    }

    @Test
    void hillClimbingCanGetStuckInLocalOptimum() {
        // from x = -2 the function rises in both feasible directions, so the climb stops at once
        assertEquals(-2, LocalSearch.hillClimbing(CUBIC, -2, 0.01, -2, 2, 1_000).x(), 1e-12);
    }

    @Test
    void simulatedAnnealingEscapesTheBoundaryTrap() {
        SimulatedAnnealing sa = new SimulatedAnnealing(10, 1e-3, 0.005);
        for (long seed = 0; seed < 5; seed++) {
            assertEquals(CUBIC_MIN_X, sa.minimize(CUBIC, -2, 2, new Random(seed)).x(), 0.05, "seed " + seed);
        }
    }

    @Test
    void metropolisCriterion() {
        assertEquals(1.0, SimulatedAnnealing.acceptanceProbability(5, 4, 1));
        assertEquals(Math.exp(-1), SimulatedAnnealing.acceptanceProbability(4, 5, 1), 1e-12);
        assertTrue(SimulatedAnnealing.acceptanceProbability(4, 5, 100) > SimulatedAnnealing.acceptanceProbability(4, 5, 1));
    }

    @Test
    void tabuSearchFindsGridMinimum() {
        // exp(-x^2 - y^2) * sin(x) has its minimum at (-0.653, 0) with value ~ -0.3967
        Optimum result = new TabuSearch(-10, 10, 0.1, 400, 2_000)
                .minimize((x, y) -> Math.exp(-x * x - y * y) * Math.sin(x), 0.5, 0.5);
        assertEquals(-0.7, result.point()[0], 1e-9);
        assertEquals(0.0, result.point()[1], 1e-9);
        assertTrue(result.value() < -0.394);
    }

    @Test
    void particleSwarmMinimisesSphere() {
        Optimum result = new ParticleSwarmOptimization(20, 300)
                .minimize(p -> p[0] * p[0] + p[1] * p[1] + p[2] * p[2], 3, -10, 10, new Random(7));
        assertTrue(result.value() < 1e-6, "value was " + result.value());
    }

    @Test
    void particleSwarmHandlesShiftedOptimum() {
        Optimum result = new ParticleSwarmOptimization(30, 300)
                .minimize(p -> Math.pow(p[0] - 3, 2) + Math.pow(p[1] + 2, 2), 2, -10, 10, new Random(3));
        assertEquals(3, result.point()[0], 1e-3);
        assertEquals(-2, result.point()[1], 1e-3);
    }
}
