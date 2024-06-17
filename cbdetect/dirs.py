"""Some useful directories in the repo."""

from pathlib import Path

# Directory for static data:
DATA_DIR: Path = Path(__file__).parent.parent / "data"
# Directory for generated data:
TARGET_DIR: Path = Path(__file__).parent.parent / "target"
# Directory for generated board images:
BOARD_IMGS_DIR: Path = TARGET_DIR / "boarde_imgs"
# Directory for generated square images:
SQUARE_IMGS_DIR: Path = TARGET_DIR / "square_imgs"
