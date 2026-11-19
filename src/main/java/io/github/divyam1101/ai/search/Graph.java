package io.github.divyam1101.ai.search;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * A small directed graph backed by an adjacency list.
 *
 * <p>Insertion order of vertices and edges is preserved, so traversals are deterministic.
 *
 * @param <T> vertex type (must implement {@code equals}/{@code hashCode})
 */
public final class Graph<T> {

    private final Map<T, List<T>> adjacency = new LinkedHashMap<>();

    public Graph<T> addVertex(T vertex) {
        adjacency.computeIfAbsent(vertex, v -> new ArrayList<>());
        return this;
    }

    /** Adds a directed edge {@code from -> to}, creating either vertex if needed. */
    public Graph<T> addEdge(T from, T to) {
        addVertex(from);
        addVertex(to);
        adjacency.get(from).add(to);
        return this;
    }

    /** Adds edges in both directions. */
    public Graph<T> addUndirectedEdge(T a, T b) {
        return addEdge(a, b).addEdge(b, a);
    }

    public List<T> neighbours(T vertex) {
        return Collections.unmodifiableList(adjacency.getOrDefault(vertex, List.of()));
    }

    public boolean contains(T vertex) {
        return adjacency.containsKey(vertex);
    }

    public List<T> vertices() {
        return List.copyOf(adjacency.keySet());
    }
}
