from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
OUTPUT_DIR = ROOT / "outputs"
LEGACY_FEEDBACK_DB_PATH = DATA_DIR / "feedback.db"
LOCAL_APP_DATA_DIR = Path.home() / "Documents" / "Codex" / "ReadWise"
FEEDBACK_DB_PATH = Path(os.environ.get("READWISE_FEEDBACK_DB_PATH", LOCAL_APP_DATA_DIR / "feedback.db"))
RATING_THRESHOLD = 8
RANDOM_SEED = 3143
