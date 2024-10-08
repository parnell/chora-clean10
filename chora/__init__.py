from pathlib import Path

# Resolve the root directory where this script resides
ROOT_DIR = Path(__file__).resolve().parent
PROJ_DIR = ROOT_DIR.parent
DATA_DIR = PROJ_DIR / "data"
TEST_DIR = PROJ_DIR / "tests"
