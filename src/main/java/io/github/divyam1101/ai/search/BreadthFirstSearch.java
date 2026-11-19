package io.github.divyam1101.ai.search;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Queue;
import java.util.Set;

/** Breadth-first search: explores the graph layer by layer from a start vertex. */
public final class BreadthFirstSearch {

    private BreadthFirstSearch() {
    }

    /** Returns vertices in the order BFS visits them. */
    public static <T> List<T> traverse(Graph<T> graph, T start) {
        List<T> order = new ArrayList<>();
        Set<T> visited = new HashSet<>();
        Queue<T> queue = new ArrayDeque<>();
        visited.add(start);
        queue.add(start);
        while (!queue.isEmpty()) {
            T current = queue.remove();
            order.add(current);
            for (T next : graph.neighbours(current)) {
                if (visited.add(next)) {
                    queue.add(next);
                }
            }
        }
        return order;
    }

    /**
     * Shortest path by number of edges (BFS is optimal on unweighted graphs).
     *
     * @return the path including both endpoints, or empty if {@code goal} is unreachable
     */
    public static <T> Optional<List<T>> shortestPath(Graph<T> graph, T start, T goal) {
        Map<T, T> parent = new HashMap<>();
        Set<T> visited = new HashSet<>();
        Queue<T> queue = new ArrayDeque<>();
        visited.add(start);
        queue.add(start);
        while (!queue.isEmpty()) {
            T current = queue.remove();
            if (current.equals(goal)) {
                return Optional.of(Paths.reconstruct(parent, goal));
            }
            for (T next : graph.neighbours(current)) {
                if (visited.add(next)) {
                    parent.put(next, current);
                    queue.add(next);
                }
            }
        }
        return Optional.empty();
    }
}
