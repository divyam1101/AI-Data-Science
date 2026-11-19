"""Tic-tac-toe with an unbeatable negamax + alpha-beta AI. Run this module to play."""

from __future__ import annotations

from functools import cache

EMPTY = "-"
LINES = ((0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6))


def opponent(player: str) -> str:
    return "O" if player == "X" else "X"


class Board:
    def __init__(self, layout: str = EMPTY * 9) -> None:
        if len(layout) != 9 or set(layout) - {"X", "O", EMPTY}:
            raise ValueError("layout must be 9 characters of 'X', 'O' or '-'")
        self.cells = list(layout)

    @property
    def layout(self) -> str:
        return "".join(self.cells)

    def empty_cells(self) -> list[int]:
        return [i for i, c in enumerate(self.cells) if c == EMPTY]

    def play(self, cell: int, player: str) -> None:
        if not 0 <= cell <= 8 or self.cells[cell] != EMPTY:
            raise ValueError(f"cell {cell} is not available")
        self.cells[cell] = player

    def winner(self) -> str:
        return _winner(self.layout)

    def is_over(self) -> bool:
        return self.winner() != EMPTY or EMPTY not in self.cells

    def best_move(self, player: str) -> int:
        """Optimal move: wins as fast as possible, otherwise delays losing as long as possible."""
        if self.is_over():
            raise ValueError("game is over")
        layout = self.layout

        def score(cell: int) -> int:
            after = layout[:cell] + player + layout[cell + 1:]
            return -_negamax(after, opponent(player), 1, -100, 100)

        return max(self.empty_cells(), key=score)

    def __str__(self) -> str:
        return "\n".join(" ".join(self.cells[r * 3: r * 3 + 3]) for r in range(3))


@cache
def _winner(layout: str) -> str:
    for a, b, c in LINES:
        if layout[a] != EMPTY and layout[a] == layout[b] == layout[c]:
            return layout[a]
    return EMPTY


@cache
def _negamax(layout: str, to_move: str, depth: int, alpha: int, beta: int) -> int:
    """Score from ``to_move``'s point of view; faster wins and slower losses score better."""
    if _winner(layout) != EMPTY:
        return -(10 - depth)  # the previous player has just won
    if EMPTY not in layout:
        return 0
    best = -100
    for cell, c in enumerate(layout):
        if c != EMPTY:
            continue
        score = -_negamax(layout[:cell] + to_move + layout[cell + 1:], opponent(to_move), depth + 1, -beta, -alpha)
        best = max(best, score)
        alpha = max(alpha, score)
        if alpha >= beta:
            break
    return best


def _read_int(prompt: str, lo: int, hi: int) -> int:
    while True:
        raw = input(prompt).strip()
        if raw.isdigit() and lo <= int(raw) <= hi:
            return int(raw)
        print(f"Enter a number from {lo} to {hi}.")


def play() -> None:
    board = Board()
    print("Cells are numbered 0-8, left to right, top to bottom. You are O.")
    player = "X" if _read_int("Who starts? 1 = computer, 2 = you: ", 1, 2) == 1 else "O"
    while not board.is_over():
        if player == "X":
            cell = board.best_move("X")
            print(f"Computer plays {cell}")
        else:
            print(board)
            cell = _read_int("Your move (0-8): ", 0, 8)
            while cell not in board.empty_cells():
                cell = _read_int("That cell is taken, try again: ", 0, 8)
        board.play(cell, player)
        player = opponent(player)
    print(board)
    print({"X": "Computer wins.", "O": "You win!"}.get(board.winner(), "It's a draw."))


if __name__ == "__main__":
    play()
