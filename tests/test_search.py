from ai_algorithms.search import (
    Cell,
    Graph,
    GridAStar,
    bfs,
    dfs,
    dfs_all,
    iterative_deepening,
    manhattan,
    shortest_path,
)


def binary_tree() -> Graph:
    g = Graph()
    for a, b in [("A", "B"), ("A", "C"), ("B", "D"), ("B", "E"), ("C", "F"), ("C", "G")]:
        g.add_edge(a, b)
    return g


def test_bfs_visits_level_by_level():
    assert bfs(binary_tree(), "A") == list("ABCDEFG")


def test_bfs_shortest_path_and_unreachable():
    g = Graph()
    for a, b in [(1, 2), (2, 3), (3, 4), (1, 5), (5, 4)]:
        g.add_undirected_edge(a, b)
    assert shortest_path(g, 1, 4) == [1, 5, 4]
    assert shortest_path(g.add_vertex(9), 1, 9) is None


def test_dfs_goes_deep_first():
    assert dfs(binary_tree(), "A") == list("ABDECFG")


def test_dfs_all_covers_disconnected_components():
    g = Graph().add_edge(1, 2).add_edge(3, 4)
    assert dfs_all(g) == [1, 2, 3, 4]


def test_iddfs_finds_shallowest_path():
    g = binary_tree().add_edge("A", "G")
    assert iterative_deepening(g, "A", "G", 5) == ["A", "G"]
    assert iterative_deepening(g, "A", "E", 5) == ["A", "B", "E"]


def test_iddfs_terminates_when_unreachable():
    g = Graph().add_edge("A", "B").add_edge("B", "A").add_vertex("Z")
    assert iterative_deepening(g, "A", "Z", 10) is None


def test_astar_finds_optimal_path_around_wall():
    blocked = [[False] * 8 for _ in range(5)]
    blocked[1][7] = True
    for c in range(3, 8):
        blocked[2][c] = True
    path = GridAStar(blocked).find_path(Cell(3, 3), Cell(1, 6))
    assert path[0] == (3, 3) and path[-1] == (1, 6)
    assert len(path) - 1 == 7  # round the left end of the wall
    for a, b in zip(path, path[1:], strict=False):
        assert manhattan(a, b) == 1
        assert not blocked[b.row][b.col]


def test_astar_reports_no_path_when_walled_off():
    blocked = [[False, True, False], [True, True, False], [False, False, False]]
    assert GridAStar(blocked).find_path(Cell(2, 2), Cell(0, 0)) is None
