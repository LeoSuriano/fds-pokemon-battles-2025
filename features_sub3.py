# This code has been produced by the group "team_pk" group, composed of 
# Leonardo Suriano, Riccardo Pugliese and Mariana Dos Campos.
#
# AI assistance disclaimer
# Parts of this code (in particular some comments, the iterative feature search,
# and minor implementation details) may have been drafted or refined with the help
# of AI-based tools. The use of AI was strictly limited to these aspects. 
# All core ideas, modeling choices, and logical structures implemented in the code 
# and in the models were entirely conceived and designed by the members of the group,
# without external intellectual contribution, relying solely on online documentation,
# our own knowledge and the insights provided by the course lectures.

from tqdm.notebook import tqdm
import pandas as pd
import numpy as np
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

# ======================================================================
# Status severity mapping
# ======================================================================
MAP_STATUS = {
    "nostatus": 0,
    "par": 1,
    "brn": 1,
    "psn": 1,
    "tox": 2,
    "frz": 2,
    "slp": 3,
    "fnt": 0,
}

# ======================================================================
# Feature generator for the "5-feature" model
# ======================================================================
def create_features1(data: list[dict]) -> pd.DataFrame:
    """
    Generate the 5 features used in the simple model:
      - hp_edge_final
      - used_count_diff
      - status_severity_gap_final
      - revealed_count_diff
      - p1_status_mean_final

    The returned DataFrame always includes:
      - battle_id
      - player_won (if present in the input battles)
      - the 5 engineered features above
    """

    rows = []

    for battle in tqdm(data, desc="Extracting 5 simple features"):
        feats: dict = {}

        timeline = battle.get("battle_timeline") or []
        n_turns = len(timeline)

        # Battle identifier + target label (if present)
        feats["battle_id"] = battle.get("battle_id")
        if "player_won" in battle:
            feats["player_won"] = int(battle["player_won"])

        # Active Pokémon per turn
        p1_active = [
            str(t["p1_pokemon_state"]["name"]).lower()
            for t in timeline
        ]
        p2_active = [
            str(t["p2_pokemon_state"]["name"]).lower()
            for t in timeline
        ]

        # Raw status sequences (only used to get the final status per species)
        p1_status_raw = [
            t["p1_pokemon_state"].get("status", "nostatus")
            for t in timeline
        ]
        p2_status_raw = [
            t["p2_pokemon_state"].get("status", "nostatus")
            for t in timeline
        ]

        # --------------------------------------------------
        # 1) revealed_count_diff
        # --------------------------------------------------
        p1_seen = set(p1_active)
        p2_seen = set(p2_active)
        feats["revealed_count_diff"] = int(len(p1_seen) - len(p2_seen))

        # --------------------------------------------------
        # 2) used_count_diff
        # --------------------------------------------------
        c1 = Counter(p1_active)
        c2 = Counter(p2_active)
        used_count_p1 = len(c1)
        used_count_p2 = len(c2)
        feats["used_count_diff"] = int(used_count_p1 - used_count_p2)

        # --------------------------------------------------
        # 3) hp_edge_final
        # --------------------------------------------------
        last_hp_p1, last_hp_p2 = {}, {}
        last_status_p1, last_status_p2 = {}, {}

        for t in timeline:
            n1 = str(t["p1_pokemon_state"]["name"]).lower()
            n2 = str(t["p2_pokemon_state"]["name"]).lower()

            last_hp_p1[n1] = float(t["p1_pokemon_state"]["hp_pct"])
            last_hp_p2[n2] = float(t["p2_pokemon_state"]["hp_pct"])

            last_status_p1[n1] = t["p1_pokemon_state"].get("status", "nostatus")
            last_status_p2[n2] = t["p2_pokemon_state"].get("status", "nostatus")

        mean_hp_p1 = float(np.mean(list(last_hp_p1.values()))) if last_hp_p1 else 0.0
        mean_hp_p2 = float(np.mean(list(last_hp_p2.values()))) if last_hp_p2 else 0.0
        feats["hp_edge_final"] = float(mean_hp_p2 - mean_hp_p1)

        # --------------------------------------------------
        # 4) p1_status_mean_final
        # 5) status_severity_gap_final
        # --------------------------------------------------
        p1_status_vals = [MAP_STATUS.get(s, 0) for s in last_status_p1.values()]
        p2_status_vals = [MAP_STATUS.get(s, 0) for s in last_status_p2.values()]

        p1_status_mean_final = float(np.mean(p1_status_vals)) if p1_status_vals else 0.0
        p2_status_mean_final = float(np.mean(p2_status_vals)) if p2_status_vals else 0.0

        feats["p1_status_mean_final"] = p1_status_mean_final
        feats["status_severity_gap_final"] = float(p2_status_mean_final - p1_status_mean_final)

        rows.append(feats)

    return pd.DataFrame(rows).fillna(0)


# ======================================================================
# Feature list and helper wrapper for the main notebook
# ======================================================================

FEATURE_COLS_SUB3 = [
    "hp_edge_final",
    "used_count_diff",
    "status_severity_gap_final",
    "revealed_count_diff",
    "p1_status_mean_final",
]

def create_features_sub3(train_data, test_data):
    """
    Wrapper used in the main Kaggle notebook.

    Input:
      - train_data: list of battles (train.jsonl)
      - test_data : list of battles (test.jsonl)

    Output:
      - X_train: feature matrix for training (only FEATURE_COLS_SUB3)
      - y_train: target vector (player_won)
      - X_test : feature matrix for test (only FEATURE_COLS_SUB3)
    """
    train_df = create_features1(train_data)
    test_df  = create_features1(test_data)

    X_train = train_df[FEATURE_COLS_SUB3].copy()
    y_train = train_df["player_won"].astype(int).copy()
    X_test  = test_df[FEATURE_COLS_SUB3].copy()

    return X_train, y_train, X_test
