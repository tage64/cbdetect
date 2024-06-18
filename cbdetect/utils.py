import chess


def piece_str(piece: chess.Piece) -> str:
    """Convert a chess piece to a two letter identifier."""
    return ("w" if piece.color == chess.WHITE else "b") + chess.piece_symbol(piece.piece_type)


def piece2id(piece: chess.Piece | None) -> int:
    """Convert the content of a chess square to an id."""
    if piece is None:
        return 0
    return 6 * int(piece.color) + piece.piece_type
