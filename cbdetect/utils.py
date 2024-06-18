import chess


def piece2str(piece: chess.Piece | None) -> str:
    """Convert a chess piece to a two letter identifier, or None to "empty"."""
    if piece is None:
        return "empty"
    return ("w" if piece.color == chess.WHITE else "b") + chess.piece_symbol(piece.piece_type)


def piece2id(piece: chess.Piece | None) -> int:
    """Convert the content of a chess square to an id."""
    if piece is None:
        return 0
    return 6 * int(piece.color) + piece.piece_type

def id2piece(id: int) -> chess.Piece | None:
    """Inverse of piece2id()."""
    if id == 0:
        return None
    return chess.Piece(piece_type=((id - 1) % 6) + 1, color=bool((id-1) // 6))
# Make sure id2piece() and piece2id() are inverses.
assert all((piece2id(id2piece(id)) == id for id in range(13)))

id2label: dict[int, str] = {id: piece2str(id2piece(id)) for id in range(13)}
