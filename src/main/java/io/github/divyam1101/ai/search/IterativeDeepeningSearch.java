package io.github.divyam1101.ai.search;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Optional;
import java.util.Set;

/**
 * Iterative deepening depth-first search (IDDFS).
 *
 * <p>Runs a depth-limited DFS with limits 0, 1, 2, ... It finds the shallowest goal like BFS,
 * but only needs memory proportional to the depth, like DFS.
 */
public final class IterativeDeepeningSearch {

    private IterativeDeepeningSearch() {
    }

    /**
     * @param maxDepth upper bound on the depth limit, so the search terminates when the goal
     *                 is unreachable
     * @return the shallowest path from {@code start} to {@code goal}, or empty if none exists
     *         within {@code maxDepth} edges
     */
    public static <T> Optional<List<T>> search(Graph<T> graph, T start, T goal, int maxDepth) {
        for (int limit = 0; limit <= maxDepth; limit++) {
            List<T> path = new ArrayList<>();
            Set<T> onPath = new HashSet<>();
            if (depthLimited(graph, start, goal, limit, path, onPath)) {
                return Optional.of(path);
            }
        }
        return Optional.empty();
    }

    private static <T> boolean depthLimited(Graph<T> graph, T node, T goal, int remaining,
                                            List<T> path, Set<T> onPath) {
        path.add(node);
        onPath.add(node);
        if (node.equals(goal)) {
            return true;
        }
        if (remaining > 0) {
            for (T next : graph.neighbours(node)) {
                // skip vertices already on the current path to avoid cycles
                if (!onPath.contains(next) && depthLimited(graph, next, goal, remaining - 1, path, onPath)) {
                    return true;
                }
            }
        }
        path.remove(path.size() - 1);
        onPath.remove(node);
        return false;
    }
}
