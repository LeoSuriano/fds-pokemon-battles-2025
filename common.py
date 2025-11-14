from pathlib import Path
import json

# Percorsi dei file nella competition Kaggle
DATA_DIR = Path("/kaggle/input/fds-pokemon-battles-prediction-2025")
train_file_path = DATA_DIR / "train.jsonl"
test_file_path  = DATA_DIR / "test.jsonl"

def read_data(path: Path):
    """
    Read a .jsonl file (one JSON per line) and return
    a list of Python dicts (one dict per battle).
    """
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
