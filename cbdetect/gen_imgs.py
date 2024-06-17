import argparse
import math
import random
import shutil
from io import BytesIO
from pathlib import Path
from typing import IO, Tuple

import chess
import numpy as np
import progressbar
from cairosvg import svg2png
from PIL import Image, ImageDraw
from PIL.Image import Image as ImageTy

from .dirs import BOARD_IMGS_DIR, DATA_DIR, SQUARE_IMGS_DIR
from .gen_diagrams import gen_diagrams
from .utils import piece_str

BOARD_STYLES: list[Path] = [
    b for b in (DATA_DIR / "board_styles").iterdir() if b.is_file() and b.suffix == ".svg"
]
PIECE_STYLES: list[Path] = [p for p in (DATA_DIR / "piece_styles").iterdir() if p.is_dir()]

BOARD_IMG_SIZE: int = 400  # The size of the image of the board.
# There will be randomly many, but at most this number of circles on a board:
MAX_CIRCLES_PER_BOARD: int = 16
# There will be randomly many, but at most this number of arrows on a board:
MAX_ARROWS_PER_BOARD: int = 4


def main() -> None:
    argparser = argparse.ArgumentParser(
        description="Generate a whole bunch of chess squares with different types "
        "of pieces or empty.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    argparser.add_argument("--seed", type=int, default=42, help="Random initial seed.")
    argparser.add_argument(
        "no_boards",
        type=int,
        default=4096,
        help="Number of boards to produce, note that the number of squares may be 64 times higher.",
    )
    args = argparser.parse_args()
    random.seed(args.seed)

    print(f"{len(PIECE_STYLES)} piece styles and {len(BOARD_STYLES)} board styles.")

    no_boards: int = args.no_boards

    shutil.rmtree(SQUARE_IMGS_DIR, ignore_errors=True)
    shutil.rmtree(BOARD_IMGS_DIR, ignore_errors=True)
    SQUARE_IMGS_DIR.mkdir(parents=True)
    BOARD_IMGS_DIR.mkdir(parents=True)

    with progressbar.ProgressBar(
        min_value=0,
        max_value=no_boards,
        left_justify=False,
        widgets=[
            progressbar.SimpleProgress(format="%(value_s)s/%(max_value_s)s boards "),
            progressbar.Bar(marker="=", left="[", right="]"),
        ],
    ) as pro_bar:
        for i, board in enumerate(gen_diagrams(no_boards), start=1):
            # Select a random piece and board style:
            piece_style = random.choice(PIECE_STYLES)
            board_style = random.choice(BOARD_STYLES)
            generate_imgs(board, board_style, piece_style, i)
            pro_bar.increment()
    print(f"{no_boards} board images in {BOARD_IMGS_DIR}/")
    print(f"{no_boards * 64} square images in {SQUARE_IMGS_DIR}/")


def generate_imgs(board: chess.Board, board_style: Path, piece_style: Path, i: int) -> None:
    board_img = Image.open(load_svg(board_style, BOARD_IMG_SIZE, BOARD_IMG_SIZE))

    piece_width, piece_height = BOARD_IMG_SIZE // 8, BOARD_IMG_SIZE // 8
    for sq in chess.SQUARES:
        if (piece := board.piece_at(sq)) is None:
            continue
        piece_file = piece_style / f"{piece_str(piece)}.png"
        piece_img = Image.open(piece_file).convert("RGBA").resize((piece_width, piece_height))
        board_img.paste(
            piece_img,
            (chess.square_file(sq) * piece_height, chess.square_rank(sq) * piece_width),
            piece_img,
        )
    board_img = add_noise(board_img)
    piece_imgs = split_squares(board_img, board)
    for class_, square, img in piece_imgs:
        dir = SQUARE_IMGS_DIR / class_
        dir.mkdir(exist_ok=True)
        img.save(dir / f"{i}_{chess.SQUARE_NAMES[square]}.png")
    board_img.save(BOARD_IMGS_DIR / f"{i}.png")


def load_svg(svg_file: Path, width: int, height: int) -> IO[bytes]:
    png = svg2png(url=str(svg_file), output_width=width, output_height=height)
    assert isinstance(png, bytes)
    return BytesIO(png)


def add_noise(image: ImageTy) -> ImageTy:
    """Add noise like circles and arrows to a chess board image."""
    width, height = image.size

    # Create a new transparent image
    new_image: ImageTy = Image.new("RGBA", image.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(new_image)

    # Add random circles with transparency within entire image
    for _ in range(random.randint(0, MAX_CIRCLES_PER_BOARD)):
        circle_color = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(100, 200),
        )
        circle_radius = random.randint(BOARD_IMG_SIZE // 96, BOARD_IMG_SIZE // 32)
        circle_position = (
            random.randint(0 + circle_radius, width - circle_radius),
            random.randint(0 + circle_radius, height - circle_radius),
        )
        draw.ellipse(
            (
                circle_position[0] - circle_radius,
                circle_position[1] - circle_radius,
                circle_position[0] + circle_radius,
                circle_position[1] + circle_radius,
            ),
            fill=circle_color,
        )

    # Add random arrows with transparency within entire image
    for _ in range(random.randint(0, MAX_ARROWS_PER_BOARD)):
        arrow_color = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(100, 200),
        )
        arrow_start = (int(random.randint(0, width)), int(random.randint(0, height)))
        arrow_end = (int(random.randint(0, width)), int(random.randint(0, height)))
        draw.line((arrow_start, arrow_end), fill=arrow_color, width=4)
        arrowed_head(new_image, arrow_start, arrow_end, 3, arrow_color)

    # Composite the new image onto the original image with transparency
    final_image = Image.alpha_composite(image.convert("RGBA"), new_image)
    return final_image


def arrowed_head(im, ptA, ptB, width, color):
    """Draw line from ptA to ptB with arrowhead at ptB"""
    # Get drawing context
    draw = ImageDraw.Draw(im)
    # Ensure coordinates are integers
    ptA = (int(ptA[0]), int(ptA[1]))
    ptB = (int(ptB[0]), int(ptB[1]))

    # Draw the line without arrows
    draw.line((ptA[0], ptB[0]), color, width)

    # Now work out the arrowhead
    # = it will be a triangle with one vertex at ptB
    # - it will start at 95% of the length of the line
    # - it will extend 8 pixels either side of the line
    x0, y0 = ptA
    x1, y1 = ptB
    # Now we can work out the x,y coordinates of the bottom of the arrowhead triangle
    xb = 0.95 * (x1 - x0) + x0
    yb = 0.95 * (y1 - y0) + y0

    # Work out the other two vertices of the triangle
    # Check if line is vertical
    if x0 == x1:
        vtx0 = (xb - 5, yb)
        vtx1 = (xb + 5, yb)
    # Check if line is horizontal
    elif y0 == y1:
        vtx0 = (xb, yb + 5)
        vtx1 = (xb, yb - 5)
    else:
        alpha = math.atan2(y1 - y0, x1 - x0) - 90 * math.pi / 180
        a = 8 * math.cos(alpha)
        b = 8 * math.sin(alpha)
        vtx0 = (xb + a, yb + b)
        vtx1 = (xb - a, yb - b)

    draw.polygon([vtx0, vtx1, ptB], fill=color)
    return im


def split_squares(
    board_img: ImageTy, board: chess.Board
) -> list[Tuple[str, chess.Square, ImageTy]]:
    """Take an image of a chess board and return a list of (class, square, img) tuples,
    where class is a piece like "wn" for white knight or "empty" for an empty square,
    square is the square of the piece, and img is an image of a square.
    """
    img_array = np.asarray(board_img)
    h, w, _ = img_array.shape
    assert h == w, "The picture of the chess board must be quadratic."
    square_imgs: dict[chess.Square, ImageTy] = {}
    for file in range(8):
        for rank in range(8):
            square = chess.square(file, rank)
            square_img: ImageTy = Image.fromarray(
                img_array[
                    round(h / 8 * (7 - rank)) : round(h / 8 * (8 - rank)),
                    round(w / 8 * file) : round(w / 8 * (file + 1)),
                ]
            )
            square_imgs[square] = square_img

    res: list[Tuple[str, chess.Square, ImageTy]] = []
    for square, img in square_imgs.items():
        piece = board.piece_at(square)
        class_: str = ""
        if piece is None:
            class_ = "empty"
        else:
            # Some bug somewhere flips the colors of pieces.
            class_ = "b" if piece.color == chess.WHITE else "w"
            class_ += chess.piece_symbol(piece.piece_type)
        res.append((class_, square, img))
    return res


if __name__ == "__main__":
    main()
