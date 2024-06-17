"""Get chess diagrams from real games."""

from typing import Iterable

import chess
import chess.pgn

from .dirs import DATA_DIR

MAX_DIAGRAMS: int = 28929  # The number of positions in sample_games.pgn.


def gen_diagrams(n: int = MAX_DIAGRAMS) -> Iterable[chess.Board]:
    assert n <= MAX_DIAGRAMS, f"Can at most generate {MAX_DIAGRAMS} diagrams"
    i: int = 0
    with open(DATA_DIR / "sample_games.pgn") as f:
        while i < n and (game := chess.pgn.read_game(f)) is not None:
            board = game.board()
            yield board.copy()
            i += 1
            for move in game.mainline_moves():
                if i >= n:
                    break
                i += 1
                board.push(move)
                yield board.copy()


def main() -> None:
    import argparse

    argparser = argparse.ArgumentParser(description="Generate a list of fens, one FEN per line.")
    argparser.add_argument("n", type=int, default=MAX_DIAGRAMS, help="Number of FENs to generate.")
    args = argparser.parse_args()
    for board in gen_diagrams(args.n):
        print(board.fen())


if __name__ == "__main__":
    main()
