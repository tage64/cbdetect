import chess


def piece_str(piece: chess.Piece) -> str:
    """Convert a chess piece to a two letter identifier."""
    return ("w" if piece.color == chess.WHITE else "b") + chess.piece_symbol(piece.piece_type)
