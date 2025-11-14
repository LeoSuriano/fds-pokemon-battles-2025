from pathlib import Path
import json
import pandas as pd

DATA_DIR = Path("/kaggle/input/fds-pokemon-battles-prediction-2025")
train_file_path = DATA_DIR / "train.jsonl"
test_file_path  = DATA_DIR / "test.jsonl"


def read_data(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def get_battle_ids(battles):
    return [b.get("battle_id") for b in battles]


def gen_submission(get_model_fn, feature_fn):
    train_battles = read_data(train_file_path)
    test_battles  = read_data(test_file_path)
    X_train, y_train, X_test = feature_fn(train_battles, test_battles)
    model = get_model_fn()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test).astype(int)
    battle_ids = get_battle_ids(test_battles)
    return pd.DataFrame({
        "battle_id": battle_ids,
        "player_won": y_pred,
    })

