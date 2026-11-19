package io.github.divyam1101.ai.search;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

/** Helpers shared by the search algorithms. */
final class Paths {

    private Paths() {
    }

    /** Walks a child-to-parent map back from {@code goal} and returns the path start-first. */
    static <T> List<T> reconstruct(Map<T, T> parent, T goal) {
        List<T> path = new ArrayList<>();
        for (T node = goal; node != null; node = parent.get(node)) {
            path.add(node);
        }
        Collections.reverse(path);
        return path;
    }
}
