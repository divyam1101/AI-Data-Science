package io.github.divyam1101.ai.search;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.HashSet;
import java.util.List;
import java.util.ListIterator;
import java.util.Set;

/**
 * Depth-first search using an explicit stack, so deep graphs cannot overflow the call stack.
 * The visit order matches the classic recursive formulation.
 */
public final class DepthFirstSearch {

    private DepthFirstSearch() {
    }

    /** Returns vertices reachable from {@code start} in depth-first (pre-)order. */
    public static <T> List<T> traverse(Graph<T> graph, T start) {
        List<T> order = new ArrayList<>();
        Set<T> visited = new HashSet<>();
        Deque<T> stack = new ArrayDeque<>();
        stack.push(start);
        while (!stack.isEmpty()) {
            T current = stack.pop();
            if (!visited.add(current)) {
                continue;
            }
            order.add(current);
            // push in reverse so the first neighbour is explored first
            List<T> neighbours = graph.neighbours(current);
            for (ListIterator<T> it = neighbours.listIterator(neighbours.size()); it.hasPrevious(); ) {
                T next = it.previous();
                if (!visited.contains(next)) {
                    stack.push(next);
                }
            }
        }
        return order;
    }

    /** Visits every vertex of the graph, starting a new DFS tree from each unvisited vertex. */
    public static <T> List<T> traverseAll(Graph<T> graph) {
        List<T> order = new ArrayList<>();
        Set<T> seen = new HashSet<>();
        for (T vertex : graph.vertices()) {
            if (!seen.contains(vertex)) {
                for (T v : traverse(graph, vertex)) {
                    if (seen.add(v)) {
                        order.add(v);
                    }
                }
            }
        }
        return order;
    }
}
