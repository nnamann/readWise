from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
OUTPUT_DIR = ROOT / "outputs"
FEEDBACK_DB_PATH = DATA_DIR / "feedback.db"
RATING_THRESHOLD = 8
RANDOM_SEED = 3143
