package io.github.divyam1101.ai.games;

import java.util.Scanner;

/** Play tic-tac-toe against the minimax AI in the terminal. You are 'O', the computer is 'X'. */
public final class TicTacToeConsole {

    private TicTacToeConsole() {
    }

    public static void main(String[] args) {
        Scanner in = new Scanner(System.in);
        TicTacToe board = new TicTacToe();
        System.out.println("Cells are numbered 0-8, left to right, top to bottom.");
        System.out.print("Who starts? 1 = computer, 2 = you: ");
        char toMove = readInt(in, 1, 2) == 1 ? 'X' : 'O';

        while (!board.isOver()) {
            if (toMove == 'X') {
                int cell = board.bestMove('X');
                board.play(cell, 'X');
                System.out.println("Computer plays " + cell);
            } else {
                System.out.print(board + "Your move (0-8): ");
                int cell = readInt(in, 0, 8);
                while (!board.emptyCells().contains(cell)) {
                    System.out.print("That cell is taken, try again: ");
                    cell = readInt(in, 0, 8);
                }
                board.play(cell, 'O');
            }
            toMove = TicTacToe.opponent(toMove);
        }

        System.out.print(board);
        char winner = board.winner();
        System.out.println(winner == 'X' ? "Computer wins." : winner == 'O' ? "You win!" : "It's a draw.");
    }

    private static int readInt(Scanner in, int min, int max) {
        while (true) {
            if (in.hasNextInt()) {
                int value = in.nextInt();
                if (value >= min && value <= max) {
                    return value;
                }
            } else if (in.hasNext()) {
                in.next();
            } else {
                throw new IllegalStateException("no more input");
            }
            System.out.print("Enter a number from " + min + " to " + max + ": ");
        }
    }
}
