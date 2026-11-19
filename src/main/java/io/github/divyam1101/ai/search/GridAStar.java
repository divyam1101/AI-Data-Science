package io.github.divyam1101.ai.search;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.PriorityQueue;
import java.util.Set;

/**
 * A* shortest-path search on a 2D grid with obstacles and 4-directional movement.
 *
 * <p>Uses the Manhattan distance, which is admissible and consistent for 4-neighbour moves,
 * so the returned path is guaranteed to be optimal.
 */
public final class GridAStar {

    /** A grid coordinate. */
    public record Cell(int row, int col) {
    }

    private record Entry(Cell cell, int f, int g) {
    }

    private static final int[][] MOVES = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};

    private final boolean[][] blocked;
    private final int rows;
    private final int cols;

    /** @param blocked {@code blocked[r][c] == true} marks an obstacle */
    public GridAStar(boolean[][] blocked) {
        this.rows = blocked.length;
        this.cols = rows == 0 ? 0 : blocked[0].length;
        this.blocked = new boolean[rows][];
        for (int r = 0; r < rows; r++) {
            this.blocked[r] = blocked[r].clone();
        }
    }

    /** @return the optimal path start-first, or empty if the goal cannot be reached */
    public Optional<List<Cell>> findPath(Cell start, Cell goal) {
        if (!isFree(start) || !isFree(goal)) {
            return Optional.empty();
        }
        Map<Cell, Integer> bestG = new HashMap<>();
        Map<Cell, Cell> parent = new HashMap<>();
        Set<Cell> closed = new HashSet<>();
        PriorityQueue<Entry> open = new PriorityQueue<>(
                (a, b) -> a.f() != b.f() ? Integer.compare(a.f(), b.f()) : Integer.compare(b.g(), a.g()));

        bestG.put(start, 0);
        open.add(new Entry(start, heuristic(start, goal), 0));

        while (!open.isEmpty()) {
            Entry entry = open.poll();
            Cell current = entry.cell();
            if (!closed.add(current)) {
                continue; // stale queue entry (lazy deletion)
            }
            if (current.equals(goal)) {
                return Optional.of(Paths.reconstruct(parent, goal));
            }
            for (Cell next : neighbours(current)) {
                if (closed.contains(next)) {
                    continue;
                }
                int tentativeG = entry.g() + 1;
                if (tentativeG < bestG.getOrDefault(next, Integer.MAX_VALUE)) {
                    bestG.put(next, tentativeG);
                    parent.put(next, current);
                    open.add(new Entry(next, tentativeG + heuristic(next, goal), tentativeG));
                }
            }
        }
        return Optional.empty();
    }

    private List<Cell> neighbours(Cell cell) {
        List<Cell> result = new ArrayList<>(4);
        for (int[] move : MOVES) {
            Cell next = new Cell(cell.row() + move[0], cell.col() + move[1]);
            if (isFree(next)) {
                result.add(next);
            }
        }
        return result;
    }

    private boolean isFree(Cell cell) {
        return cell.row() >= 0 && cell.row() < rows && cell.col() >= 0 && cell.col() < cols
                && !blocked[cell.row()][cell.col()];
    }

    static int heuristic(Cell a, Cell b) {
        return Math.abs(a.row() - b.row()) + Math.abs(a.col() - b.col());
    }
}
