package io.github.divyam1101.ai.search;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.divyam1101.ai.search.GridAStar.Cell;
import java.util.List;
import org.junit.jupiter.api.Test;

class SearchTest {

    /** A -> B, C;  B -> D, E;  C -> F, G. */
    private static Graph<String> binaryTree() {
        return new Graph<String>()
                .addEdge("A", "B").addEdge("A", "C")
                .addEdge("B", "D").addEdge("B", "E")
                .addEdge("C", "F").addEdge("C", "G");
    }

    @Test
    void bfsVisitsLevelByLevel() {
        assertEquals(List.of("A", "B", "C", "D", "E", "F", "G"), BreadthFirstSearch.traverse(binaryTree(), "A"));
    }

    @Test
    void bfsFindsShortestPathAndHandlesCycles() {
        Graph<Integer> g = new Graph<Integer>()
                .addUndirectedEdge(1, 2).addUndirectedEdge(2, 3).addUndirectedEdge(3, 4)
                .addUndirectedEdge(1, 5).addUndirectedEdge(5, 4);
        assertEquals(List.of(1, 5, 4), BreadthFirstSearch.shortestPath(g, 1, 4).orElseThrow());
        assertTrue(BreadthFirstSearch.shortestPath(g.addVertex(9), 1, 9).isEmpty());
    }

    @Test
    void dfsGoesDeepFirst() {
        assertEquals(List.of("A", "B", "D", "E", "C", "F", "G"), DepthFirstSearch.traverse(binaryTree(), "A"));
    }

    @Test
    void dfsTraverseAllCoversDisconnectedComponents() {
        Graph<Integer> g = new Graph<Integer>().addEdge(1, 2).addEdge(3, 4);
        assertEquals(List.of(1, 2, 3, 4), DepthFirstSearch.traverseAll(g));
    }

    @Test
    void iddfsFindsShallowestPath() {
        Graph<String> g = binaryTree().addEdge("A", "G"); // shortcut A -> G
        assertEquals(List.of("A", "G"), IterativeDeepeningSearch.search(g, "A", "G", 5).orElseThrow());
        assertEquals(List.of("A", "B", "E"), IterativeDeepeningSearch.search(g, "A", "E", 5).orElseThrow());
    }

    @Test
    void iddfsTerminatesWhenGoalIsUnreachable() {
        Graph<String> cyclic = new Graph<String>().addEdge("A", "B").addEdge("B", "A").addVertex("Z");
        assertFalse(IterativeDeepeningSearch.search(cyclic, "A", "Z", 10).isPresent());
    }

    @Test
    void aStarFindsOptimalPathAroundWall() {
        boolean[][] blocked = new boolean[5][8];
        blocked[1][7] = true;
        for (int c = 3; c <= 7; c++) {
            blocked[2][c] = true;
        }
        List<Cell> path = new GridAStar(blocked).findPath(new Cell(3, 3), new Cell(1, 6)).orElseThrow();
        assertEquals(new Cell(3, 3), path.get(0));
        assertEquals(new Cell(1, 6), path.get(path.size() - 1));
        assertEquals(7, path.size() - 1, "optimal route goes round the left end of the wall");
        for (int k = 1; k < path.size(); k++) {
            Cell a = path.get(k - 1);
            Cell b = path.get(k);
            assertEquals(1, GridAStar.heuristic(a, b), "moves must be to adjacent cells");
            assertFalse(blocked[b.row()][b.col()], "path must avoid obstacles");
        }
    }

    @Test
    void aStarReportsNoPathWhenGoalIsWalledOff() {
        boolean[][] blocked = new boolean[3][3];
        blocked[0][1] = true;
        blocked[1][0] = true;
        blocked[1][1] = true;
        assertTrue(new GridAStar(blocked).findPath(new Cell(2, 2), new Cell(0, 0)).isEmpty());
    }
}
