"""Uninformed and informed graph search: BFS, DFS, iterative deepening and grid A*."""

from __future__ import annotations

import heapq
from collections import deque
from collections.abc import Hashable, Iterable
from typing import Generic, NamedTuple, TypeVar

T = TypeVar("T", bound=Hashable)


class Graph(Generic[T]):
    """A directed graph backed by an adjacency list. Insertion order is preserved."""

    def __init__(self) -> None:
        self._adjacency: dict[T, list[T]] = {}

    def add_vertex(self, vertex: T) -> Graph[T]:
        self._adjacency.setdefault(vertex, [])
        return self

    def add_edge(self, source: T, target: T) -> Graph[T]:
        self.add_vertex(source).add_vertex(target)
        self._adjacency[source].append(target)
        return self

    def add_undirected_edge(self, a: T, b: T) -> Graph[T]:
        return self.add_edge(a, b).add_edge(b, a)

    def neighbours(self, vertex: T) -> tuple[T, ...]:
        return tuple(self._adjacency.get(vertex, ()))

    def vertices(self) -> tuple[T, ...]:
        return tuple(self._adjacency)

    def __contains__(self, vertex: object) -> bool:
        return vertex in self._adjacency


def _reconstruct(parent: dict[T, T], goal: T) -> list[T]:
    path = [goal]
    while path[-1] in parent:
        path.append(parent[path[-1]])
    return path[::-1]


def bfs(graph: Graph[T], start: T) -> list[T]:
    """Vertices reachable from ``start`` in breadth-first order."""
    order, seen, queue = [], {start}, deque([start])
    while queue:
        current = queue.popleft()
        order.append(current)
        for nxt in graph.neighbours(current):
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return order


def shortest_path(graph: Graph[T], start: T, goal: T) -> list[T] | None:
    """Fewest-edges path (BFS is optimal on unweighted graphs), or ``None`` if unreachable."""
    parent: dict[T, T] = {}
    seen, queue = {start}, deque([start])
    while queue:
        current = queue.popleft()
        if current == goal:
            return _reconstruct(parent, goal)
        for nxt in graph.neighbours(current):
            if nxt not in seen:
                seen.add(nxt)
                parent[nxt] = current
                queue.append(nxt)
    return None


def dfs(graph: Graph[T], start: T) -> list[T]:
    """Depth-first pre-order using an explicit stack (no recursion limit)."""
    order: list[T] = []
    seen: set[T] = set()
    stack = [start]
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        order.append(current)
        stack.extend(n for n in reversed(graph.neighbours(current)) if n not in seen)
    return order


def dfs_all(graph: Graph[T]) -> list[T]:
    """DFS over every vertex, starting a new tree from each unvisited one."""
    order: list[T] = []
    seen: set[T] = set()
    for vertex in graph.vertices():
        if vertex not in seen:
            for v in dfs(graph, vertex):
                if v not in seen:
                    seen.add(v)
                    order.append(v)
    return order


def iterative_deepening(graph: Graph[T], start: T, goal: T, max_depth: int) -> list[T] | None:
    """Iterative deepening DFS: shallowest path with memory linear in the depth.

    Returns ``None`` if ``goal`` is not reachable within ``max_depth`` edges.
    """

    def limited(node: T, remaining: int, path: list[T]) -> bool:
        path.append(node)
        if node == goal:
            return True
        if remaining > 0:
            for nxt in graph.neighbours(node):
                if nxt not in path and limited(nxt, remaining - 1, path):
                    return True
        path.pop()
        return False

    for limit in range(max_depth + 1):
        path: list[T] = []
        if limited(start, limit, path):
            return path
    return None


class Cell(NamedTuple):
    row: int
    col: int


def manhattan(a: Cell, b: Cell) -> int:
    return abs(a.row - b.row) + abs(a.col - b.col)


class GridAStar:
    """A* on a 2D grid with obstacles and 4-directional moves.

    The Manhattan heuristic is admissible and consistent here, so returned paths are optimal.
    """

    _MOVES = ((-1, 0), (1, 0), (0, -1), (0, 1))

    def __init__(self, blocked: Iterable[Iterable[bool]]) -> None:
        self._blocked = [list(row) for row in blocked]
        self.rows = len(self._blocked)
        self.cols = len(self._blocked[0]) if self.rows else 0

    def is_free(self, cell: Cell) -> bool:
        return 0 <= cell.row < self.rows and 0 <= cell.col < self.cols and not self._blocked[cell.row][cell.col]

    def neighbours(self, cell: Cell) -> list[Cell]:
        candidates = (Cell(cell.row + dr, cell.col + dc) for dr, dc in self._MOVES)
        return [c for c in candidates if self.is_free(c)]

    def find_path(self, start: Cell, goal: Cell) -> list[Cell] | None:
        start, goal = Cell(*start), Cell(*goal)
        if not (self.is_free(start) and self.is_free(goal)):
            return None
        best_g = {start: 0}
        parent: dict[Cell, Cell] = {}
        closed: set[Cell] = set()
        # (f, -g, cell): ties broken towards deeper nodes, which reach the goal sooner
        open_heap = [(manhattan(start, goal), 0, start)]
        while open_heap:
            _, neg_g, current = heapq.heappop(open_heap)
            if current in closed:
                continue  # stale entry (lazy deletion)
            closed.add(current)
            if current == goal:
                return _reconstruct(parent, goal)
            g = -neg_g + 1
            for nxt in self.neighbours(current):
                if nxt not in closed and g < best_g.get(nxt, float("inf")):
                    best_g[nxt] = g
                    parent[nxt] = current
                    heapq.heappush(open_heap, (g + manhattan(nxt, goal), -g, nxt))
        return None
