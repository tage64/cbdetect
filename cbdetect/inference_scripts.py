"""This file contains various scripts for chess board detection and piece classification."""

import argparse
import concurrent.futures
import io
import json
import logging
import os
import subprocess
import tempfile
import time
from contextlib import suppress
from pathlib import Path
from typing import Literal

import chess
import chess.pgn
import cv2 as cv
import fitz
import markdown
import numpy as np
import pymupdf4llm
import torch
from PIL import Image
from PIL.Image import Image as ImageTy
from skimage.metrics import structural_similarity
from transformers import ViTForImageClassification, ViTImageProcessor

from . import chessboard_detect

LOGGER: logging.Logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

MODEL_PATH: str = "Nothasan/Chessboard"
PREDICTION_THRESHOLD: float = 0.9998
MIN_BOARD_SIDE_LEN_PIXELS: int = 32

# The times it takes to classify one square.
classification_times: list[float] = []
n_workers: int = os.cpu_count() or 2
# Squares with a mean structural similarity index >= this threshold will be
# considered similar and not reclassified by the neural network.
SIMILARITY_THRESHOLD: float = 0.95
PIECE_BY_ID: dict[str, chess.Piece | None] = {
    "empty": None,
    **{
        f"{c_char}{chess.piece_symbol(p)}": chess.Piece(p, c)
        for (c_char, c) in (("w", chess.WHITE), ("b", chess.BLACK))
        for p in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
    },
}
ID_TO_LABEL: dict[int, str] = {
    0: "bb",
    1: "bk",
    2: "bn",
    3: "bp",
    4: "bq",
    5: "br",
    6: "empty",
    7: "wb",
    8: "wk",
    9: "wn",
    10: "wp",
    11: "wq",
    12: "wr",
}


class SquareClassifier:
    """Classifier to classify pieces on squares."""

    device: torch.device  # Pytorch device
    metadata: dict
    id_to_label: dict[str, str]
    model: ViTForImageClassification
    image_processor: ViTImageProcessor

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        LOGGER.info(f"Using {self.device} as Pytorch device.")

        # Load the model checkpoint
        model = ViTForImageClassification.from_pretrained(MODEL_PATH, device_map=self.device)
        assert isinstance(model, ViTForImageClassification)
        self.model = model
        image_processor = ViTImageProcessor.from_pretrained(MODEL_PATH, device_map=self.device)
        assert isinstance(image_processor, ViTImageProcessor)
        self.image_processor = image_processor

    def classify(self, img: ImageTy, square: chess.Square | None = None) -> str:
        img = img.convert("RGB")
        assert img.mode == "RGB", f"Image mode is {img.mode}"
        inputs = self.image_processor(img, return_tensors="pt")
        with torch.no_grad():
            logits = self.model(**inputs).logits

        probs = torch.softmax(logits, dim=-1)
        # Uncomment to save an image of the square.
        #if square is not None:
        #    img.save(f"imgs/{chess.square_name(square)}.png")
        predicted = torch.max(probs, dim=-1)
        if predicted.values.item() < PREDICTION_THRESHOLD:
            return "empty"
        predicted_piece = ID_TO_LABEL[predicted.indices.item()]  # type: ignore
        return predicted_piece


class ChessBoardDetection:
    """Class which can detect a chess board from an image and remember classifications in a video."""

    square_classifier: SquareClassifier
    # The image (RGB array) for a square on the previous image.
    prev_square_imgs: dict[chess.Square, np.ndarray]
    # The piece on a square from the previous image.
    prev_pieces: dict[chess.Square, chess.Piece | None]
    ## Statistics
    # The total number of squares seen.
    total_squares: int = 0
    # The number of classified squares:
    classified_squares: int = 0

    def __init__(self):
        self.square_classifier = SquareClassifier()
        self.prev_square_imgs = {}
        self.prev_pieces = {}

    def detect_squares(self, img: np.ndarray) -> dict[chess.Square, np.ndarray] | None:
        """Take an image (RGB) with a chessboard, locate the chess board and split
        into the 64 squares. Returns None if no board was detected.
        """
        gray: np.ndarray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        with_border = cv.copyMakeBorder(
            src=gray, top=1, bottom=1, left=1, right=1, borderType=cv.BORDER_CONSTANT, value=[0]
        )
        (x, y, dimx, dimy), _ = chessboard_detect.locate(with_border)
        assert isinstance(x, int)
        assert isinstance(y, int)
        if x == -1:
            return None
        x, y = max(x - 1, 0), max(y - 1, 0)
        if min(dimx, dimy) * 1.2 < max(dimx, dimy):
            return None
        # Uncomment to print dimensions:
        # LOGGER.info(f"{x}, {y}, {dimx}, {dimy}")
        if min(dimx, dimy) < MIN_BOARD_SIDE_LEN_PIXELS:
            return None
        cropped: np.ndarray = img[y : y + dimy, x : x + dimx]
        square_imgs: dict[chess.Square, np.ndarray] = {}
        for sq in chess.SQUARES:
            f = chess.square_file(sq)
            r = chess.square_rank(sq)
            square_imgs[sq] = cropped[
                int((7 - r + 0.1) * dimy / 8) : int((7 - r + 0.9) * dimy / 8),
                int((f + 0.1) * dimx / 8) : int((f + 0.9) * dimx / 8),
            ]
            if 0 in square_imgs[sq].shape:
                breakpoint()
        return square_imgs

    def classify_squares(self, square_imgs: dict[chess.Square, np.ndarray]) -> chess.Board:
        """Given a dictionary of all squares and corresponding images, classify
        each square and collect the result into a board.
        """

        def classify_square(square_img: np.ndarray, square: chess.Square) -> str:
            square_img_img: ImageTy = Image.fromarray(square_img)
            start_time = time.perf_counter()
            class_ = self.square_classifier.classify(square_img_img, square)
            classification_times.append(time.perf_counter() - start_time)
            return class_

        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
            future_to_square: dict[concurrent.futures.Future, chess.Square] = {
                executor.submit(classify_square, square_imgs[sq], sq): sq for sq in chess.SQUARES
            }

            pieces: dict[chess.Square, chess.Piece | None] = {}
            for future in concurrent.futures.as_completed(future_to_square):
                sq: chess.Square = future_to_square[future]
                res: str = future.result()
                pieces[sq] = PIECE_BY_ID[res]
        board: chess.Board = chess.Board(None)
        board.set_piece_map({sq: p for (sq, p) in pieces.items() if p is not None})
        return board

    def get_board(self, img: ImageTy) -> chess.Board | Literal["No board found"]:
        if img.mode != "RGB":
            img = img.convert("RGB")
        img_array: np.ndarray = np.asarray(img)
        square_imgs = self.detect_squares(img_array)
        if square_imgs is None:
            return "No board found"

        def classify_square(
            square_img: np.ndarray, prev_square_img: np.ndarray | None
        ) -> str | None:
            """Check if square_img is equal to prev_square_img and if so return None.
            Otherwise classify square_img and return the resulting class as a str.
            """
            if prev_square_img is not None and square_img.shape == prev_square_img.shape:
                ssim = structural_similarity(square_img, prev_square_img, channel_axis=2)
                if ssim >= SIMILARITY_THRESHOLD:
                    return None
            square_img_img: ImageTy = Image.fromarray(square_img)
            start_time = time.perf_counter()
            class_ = self.square_classifier.classify(square_img_img)
            classification_times.append(time.perf_counter() - start_time)
            return class_

        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
            future_to_square: dict[concurrent.futures.Future, chess.Square] = {
                executor.submit(
                    classify_square,
                    square_imgs[sq],
                    self.prev_square_imgs[sq] if self.classified_squares > 0 else None,
                ): sq
                for sq in chess.SQUARES
            }

            pieces: dict[chess.Square, chess.Piece | None] = {}
            classified_squares: int = 0
            for future in concurrent.futures.as_completed(future_to_square):
                sq: chess.Square = future_to_square[future]
                res: str | None = future.result()
                if res is None:
                    # The square was unchanged.
                    pieces[sq] = self.prev_pieces[sq]
                else:
                    pieces[sq] = PIECE_BY_ID[res]
                    classified_squares += 1
            self.classified_squares += classified_squares
            # LOGGER.info(f"Classified {classified_squares}/64 squares.")
            self.total_squares += 64
        self.prev_square_imgs = square_imgs
        self.prev_pieces = pieces
        board: chess.Board = chess.Board(None)
        board.set_piece_map({sq: p for (sq, p) in pieces.items() if p is not None})
        return board

    def get_board_2(self, img: ImageTy) -> chess.Board | Literal["No board found"]:
        """Given an image of a chess board, split it into squares, but don't cache the result.
        Useful for single images (not in a video).

        This function uses my own algorithm for locating the chess board in detect_board.py.
        """
        img_array = np.asarray(img)
        # square_contours = find_board(cv.cvtColor(img_array, cv.COLOR_RGB2GRAY))
        # if square_contours is None:
        #    return "No board found"
        # square_imgs = {}
        # for sq in chess.SQUARES:
        #    lower_left, upper_right = square_contours.predict_square_corners(sq)
        #    square_imgs[sq] = img_array[
        #        lower_left[0] : upper_right[0] + 1, lower_left[1] : upper_right[1] + 1
        #    ]
        square_imgs = self.detect_squares(img_array)
        if square_imgs is None:
            return "No board found"
        return self.classify_squares(square_imgs)


class GameRecorder:
    """Detect chess boards and try to record a game."""

    cb_detector: ChessBoardDetection
    # The current game node.
    game_node: chess.pgn.GameNode | None = None
    # If we got lost in the game, we will start a new and push the old to this list.
    old_games: list[chess.pgn.Game]

    def __init__(self):
        self.cb_detector = ChessBoardDetection()
        self.old_games = []

    def push_img(
        self, img: ImageTy
    ) -> chess.Move | chess.Board | Literal["Board unchanged", "No board found"]:
        """Push an image and figure out the move played.
        Returns the move played or a board if no move applied from the previous position.
        """

        def _mk_move(
            seen_board: chess.Board,
        ) -> chess.Move | chess.Board | Literal["Board unchanged"]:
            if self.game_node is None:
                self.game_node = chess.pgn.Game()
                self.game_node.setup(seen_board)
                # logging.info(f"Castling rights: {seen_board.castling_rights}")
                return seen_board
            else:

                def find_move(prev_board: chess.Board) -> chess.Move | None:
                    """Given prev_board, try to find a move on prev_board which
                    leads to seen_board.
                    """
                    assert self.game_node is not None
                    for move in prev_board.legal_moves:
                        possible_board = prev_board.copy()
                        possible_board.push(move)
                        if possible_board.piece_map() == seen_board.piece_map():
                            self.game_node = self.game_node.add_variation(move)
                            return move

                prev_board: chess.Board = self.game_node.board()
                if prev_board.piece_map() == seen_board.piece_map():
                    return "Board unchanged"

                if (move := find_move(prev_board)) is not None:
                    return move
                # No move was found which led to seen_board.
                # If self.game only has a root, we want to try to change turn:
                if isinstance(self.game_node, chess.pgn.Game):
                    assert self.game_node.parent is None
                    prev_board.turn = not prev_board.turn
                    self.game_node.setup(prev_board)
                    if (move := find_move(prev_board)) is not None:
                        return move
                self.old_games.append(self.game_node.game())
                self.game_node = None
                return _mk_move(seen_board)

        match self.cb_detector.get_board(img):
            case "No board found" as x:
                return x
            case board:
                return _mk_move(board)


def single_square_script() -> None:
    argparser = argparse.ArgumentParser()
    argparser.add_argument("image", type=Path, help="Path to an image of a chess square.")
    args = argparser.parse_args()
    sq_classifier = SquareClassifier()
    img: ImageTy = Image.open(args.image)
    class_: str = sq_classifier.classify(img)
    print(class_)


def single_board_script() -> None:
    argparser = argparse.ArgumentParser()
    argparser.add_argument("image", type=Path, help="Path to an image containing a chess board.")
    argparser.add_argument(
        "-n",
        "--threads",
        type=int,
        default=os.cpu_count() or 2,
        help="Number of threads to use when classifying squares.",
    )
    args = argparser.parse_args()
    cb_detector = ChessBoardDetection()
    img: ImageTy = Image.open(args.image)
    res = cb_detector.get_board(img)
    if isinstance(res, chess.Board):
        print(res)
    else:
        print(res)


def video_script() -> None:
    argparser = argparse.ArgumentParser()
    argparser.add_argument("video", type=Path, help="Path to a chess video.")
    argparser.add_argument(
        "--fps", type=int, default=5, help="FPS to check for image differences, defaults to 5."
    )
    argparser.add_argument(
        "-a", "--all-boards", action="store_true", help="Print all boards, even after a valid move."
    )
    argparser.add_argument(
        "-n",
        "--threads",
        type=int,
        default=os.cpu_count() or 2,  # pyright: ignore
        help="Number of threads to use when classifying squares.",
    )
    args = argparser.parse_args()
    game_recorder = GameRecorder()
    with tempfile.TemporaryDirectory() as image_dir, suppress(KeyboardInterrupt):
        image_dir = Path(image_dir)
        # Create ffmpeg command to extract frames
        ffmpeg_command = [
            "ffmpeg",
            "-hide_banner",
            "-i",
            args.video,
            "-vf",
            f"fps={args.fps}",
            image_dir / "frame_%04d.png",
        ]

        # Run the ffmpeg with the command
        subprocess.check_call(ffmpeg_command)

        # Loop through all images
        for img_file in sorted(image_dir.iterdir()):
            img = Image.open(img_file)
            print()
            res = game_recorder.push_img(img)
            print(res)
            if args.all_boards and isinstance(res, chess.Move):
                assert game_recorder.game_node is not None
                print(game_recorder.game_node.board())
    total_squares = game_recorder.cb_detector.total_squares
    if total_squares > 0:
        equal_squares: int = total_squares - game_recorder.cb_detector.classified_squares
        print(
            f"{equal_squares}/{total_squares} ({round(100 * equal_squares / total_squares, 2)} %) "
            "of seen squares were equal to a previous square "
            "and therefore not evaluated by the neural net."
        )
        print(
            f"classification_times: mean: {sum(classification_times) / len(classification_times)}, "
            f"max: {max(classification_times)}, min: {min(classification_times)}"
        )


def pdf_script() -> None:
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "pdf",
        type=Path,
        nargs="?",
        help="Path to a PDF file.  If omitted, a open dialog will be shown.",
    )
    argparser.add_argument("--page", type=int, help="Only read this page from the PDF.")
    args = argparser.parse_args()
    pdf_file_name: Path
    if args.pdf is not None:
        pdf_file_name = args.pdf
    else:
        try:
            import tkinter
            import tkinter.filedialog
        except ImportError:
            print(
                f"No pdf file provided and tkinter couldn't be imported so no file dialog will be shown."
            )
            return
        fname: str = tkinter.filedialog.askopenfilename(
            title="PDF File",
            filetypes=[("PDF Files", "*.pdf"), ("All Files", "*")],
            defaultextension=".pdf",
        )
        if not fname or not isinstance(fname, str):
            print("No file selected.")
            return
        print(f"Loading {fname}...")
        pdf_file_name = Path(fname)

    pdf: fitz.Document = fitz.open(pdf_file_name)
    LOGGER.info(f"Successfully opened {pdf_file_name}")
    markdown_document: str = ""
    cb_detector = ChessBoardDetection()
    for page in pdf:
        # page.get_pixmap().save(f"imgs/page_{page.number}.png")
        if args.page is not None and not page.number + 1 == args.page:
            continue
        LOGGER.info(f"Analyzing page {page.number+1}")
        images = page.get_images(full=True)
        boards: list[chess.Board] = []
        for img in images:
            xref = img[0]  # Get the XREF of the image.
            base_img = pdf.extract_image(xref)
            pil_image: ImageTy = Image.open(
                io.BytesIO(base_img["image"]), formats=(base_img["ext"],)
            ).convert("RGB")
            w, h = pil_image.size
            if min(h, w) < MIN_BOARD_SIDE_LEN_PIXELS:
                continue
            board = cb_detector.get_board_2(pil_image)
            if isinstance(board, chess.Board):
                boards.append(board)
        markdown_page: str = pymupdf4llm.to_markdown(pdf, pages=[page.number], show_progress=False)
        if boards:
            LOGGER.info(f"Found {len(boards)} chess boards at page {page.number + 1}.")
            boards_description: str = "--------\n"
            boards_description += (
                f"It seems to be {len(boards)} chess boards displayed on this image:\n\n"
            )
            for board in boards:
                boards_description += "* " + show_pieces_on_board(board).replace("\n", "  \n")
            boards_description += "--------\n\n"
            markdown_page = boards_description + markdown_page
        markdown_document += f"# Page {page.number + 1}\n\n" + markdown_page + "\n\n"  # type: ignore
    html = markdown.markdown(markdown_document)
    out_file_name: Path = pdf_file_name.with_suffix(".html")
    with open(out_file_name, "w") as f:
        f.write(html)
        LOGGER.info(f"Successfully wrote {out_file_name}")


def show_pieces_on_board(board: chess.Board) -> str:
    """List all pieces on the board on two separate lines."""
    text: str = ""
    for color in [chess.WHITE, chess.BLACK]:
        text += "White" if color == chess.WHITE else "Black"
        text += ": "
        for piece_type in [
            chess.KING,
            chess.QUEEN,
            chess.ROOK,
            chess.BISHOP,
            chess.KNIGHT,
            chess.PAWN,
        ]:
            piece = chess.Piece(piece_type, color)
            squares = board.pieces(piece_type, color)
            if squares:
                text += piece.symbol()
                text += ",".join(chess.SQUARE_NAMES[sq] for sq in squares)
                text += " "
        text += "\n"
    return text


def parse_ascii_board(ascii_board: str) -> chess.Board:
    """Parse an ascii board as returned by the __str__ method of chess.Board."""
    board = chess.Board.empty()
    rows: list[str] = [line.strip() for line in ascii_board.splitlines() if line.strip()]
    assert len(rows) == 8, f"Should be 8 rows, is {len(rows)}"
    for i, row in enumerate(rows):
        for j, content in enumerate(row.split()):
            if content == ".":
                continue
            piece = chess.Piece.from_symbol(content)
            square = chess.square(j, 7 - i)
            board.set_piece_at(square, piece)
    return board
