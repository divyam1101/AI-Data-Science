package io.github.divyam1101.ai.games;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Tic-tac-toe board with an unbeatable minimax player using alpha-beta pruning.
 *
 * <p>Scores are depth-adjusted (a faster win scores higher and a slower loss scores less badly),
 * so the AI wins as quickly as possible and delays losing as long as possible.
 */
public final class TicTacToe {

    public static final char EMPTY = '-';
    private static final int[][] LINES = {
        {0, 1, 2}, {3, 4, 5}, {6, 7, 8}, // rows
        {0, 3, 6}, {1, 4, 7}, {2, 5, 8}, // columns
        {0, 4, 8}, {2, 4, 6}             // diagonals
    };

    private final char[] cells = new char[9];

    public TicTacToe() {
        Arrays.fill(cells, EMPTY);
    }

    /** Creates a board from a 9-character string, e.g. {@code "XO-X-----"}. */
    public static TicTacToe of(String layout) {
        if (layout.length() != 9) {
            throw new IllegalArgumentException("layout must have 9 characters");
        }
        TicTacToe board = new TicTacToe();
        for (int i = 0; i < 9; i++) {
            char c = layout.charAt(i);
            if (c != 'X' && c != 'O' && c != EMPTY) {
                throw new IllegalArgumentException("invalid cell '" + c + "'");
            }
            board.cells[i] = c;
        }
        return board;
    }

    public void play(int cell, char player) {
        if (cell < 0 || cell > 8 || cells[cell] != EMPTY) {
            throw new IllegalArgumentException("cell " + cell + " is not available");
        }
        cells[cell] = player;
    }

    public List<Integer> emptyCells() {
        List<Integer> empty = new ArrayList<>();
        for (int i = 0; i < 9; i++) {
            if (cells[i] == EMPTY) {
                empty.add(i);
            }
        }
        return empty;
    }

    /** @return 'X' or 'O' if that player has three in a row, otherwise {@link #EMPTY} */
    public char winner() {
        for (int[] line : LINES) {
            char c = cells[line[0]];
            if (c != EMPTY && c == cells[line[1]] && c == cells[line[2]]) {
                return c;
            }
        }
        return EMPTY;
    }

    public boolean isOver() {
        return winner() != EMPTY || emptyCells().isEmpty();
    }

    /** Returns the optimal cell for {@code player} to play next. */
    public int bestMove(char player) {
        if (isOver()) {
            throw new IllegalStateException("game is over");
        }
        int bestCell = -1;
        int bestScore = Integer.MIN_VALUE;
        for (int cell : emptyCells()) {
            cells[cell] = player;
            int score = -negamax(opponent(player), 1, Integer.MIN_VALUE + 1, Integer.MAX_VALUE);
            cells[cell] = EMPTY;
            if (score > bestScore) {
                bestScore = score;
                bestCell = cell;
            }
        }
        return bestCell;
    }

    /** Negamax formulation of minimax with alpha-beta pruning, scored from {@code toMove}'s view. */
    private int negamax(char toMove, int depth, int alpha, int beta) {
        char w = winner();
        if (w != EMPTY) {
            // the previous player just won: bad for the player to move
            return -(10 - depth);
        }
        List<Integer> moves = emptyCells();
        if (moves.isEmpty()) {
            return 0;
        }
        int best = Integer.MIN_VALUE + 1;
        for (int cell : moves) {
            cells[cell] = toMove;
            int score = -negamax(opponent(toMove), depth + 1, -beta, -alpha);
            cells[cell] = EMPTY;
            best = Math.max(best, score);
            alpha = Math.max(alpha, score);
            if (alpha >= beta) {
                break;
            }
        }
        return best;
    }

    public static char opponent(char player) {
        return player == 'X' ? 'O' : 'X';
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        for (int r = 0; r < 3; r++) {
            sb.append(cells[3 * r]).append(' ').append(cells[3 * r + 1]).append(' ').append(cells[3 * r + 2]);
            sb.append(System.lineSeparator());
        }
        return sb.toString();
    }
}
