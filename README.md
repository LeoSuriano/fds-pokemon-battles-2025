> **“FDS: Pokemon Battles prediction 2025”**  
> course: *Fundamentals of Data Science*, Sapienza University of Rome.

The project was developed by the group **“team_pk”** and focuses on building
predictive models for Pokémon battles using hand–crafted features extracted
from the raw JSONL battle logs.

We provide three separate notebooks, each corresponding to one of our
final Kaggle submissions. All three notebooks share the same overall
pipeline structure:

1. **Feature generation** from raw `train.jsonl` and `test.jsonl`.
2. **Construction of the matrices** `X`, `y`, `X_test`.
3. **Model training + hyperparameter search + submission CSV.**

Each notebook, however, uses a **different feature set and a different model**.

---

## Repository contents

### 1. `Submission_1_LR.ipynb` – Full feature Logistic Regression

This notebook implements our first submission, based on a rich,
high–dimensional feature set and a regularized Logistic Regression model.

**Main components:**

- **Feature engineering**
  - Uses a function `create_features(...)` that converts each battle
    (from the JSONL logs) into a row of a feature table.
  - The resulting `train_df` contains dozens of features describing:
    - HP gaps and damage patterns (early, mid, final phases),
    - type match–ups and type edges between the two teams,
    - status effects and their durations (paralysis, burn, poison, etc.),
    - number of switches, attacks, revealed Pokémon, KOs,
    - late–game board state indicators.
  - `train_df` includes both the features and the target `player_won`,
    while `test_df` includes the same features plus `battle_id`.

- **Data matrices**
  - Builds `X`, `y`, `X_test` as NumPy arrays:
    - `X`  = feature matrix for the training battles,
    - `y`  = binary target (1 if player 1 won, 0 otherwise),
    - `X_test` = feature matrix for the test battles.

- **Model**
  - A **scikit-learn pipeline**: `StandardScaler` + `LogisticRegression`.
  - Hyperparameters are tuned with **GridSearchCV** and
    **5-fold Stratified cross-validation**:
    - elastic-net penalty (`penalty='elasticnet'` with `solver='saga'`)
      and/or L2 (`penalty='l2'` with `solver='lbfgs'`),
    - search over `C`, `l1_ratio`, `tol`, and `class_weight`.
  - The best model is re-trained on the full training set and used to
    generate a submission file for Kaggle.

---

### 2. `Submission_2_ENS.ipynb` – Ensemble of LR, AdaBoost and XGBoost

This notebook implements our **ensemble submission**, combining three
different models: Logistic Regression, AdaBoost, and XGBoost.

**Main components:**

- **Feature engineering**
  - Uses a function `build_features(...)` to create `train_df` and `test_df`
    from the raw JSONL logs.
  - The feature table summarizes, for each battle, multiple aspects of the
    match (type edges, HP dynamics, status information, switching behaviour,
    offensive/defensive patterns, etc.).
  - As before:
    - `train_df` contains all features + `player_won`,
    - `test_df` contains the same features + `battle_id`.

- **Data matrices**
  - Builds `X`, `y`, `X_test` in the same way as in the first notebook:
    - common columns between `train_df` and `test_df`,
    - removal of identifiers/target (`battle_id`, `player_won`).

- **Models**
  - **Base models**:
    1. Logistic Regression (with preprocessing pipeline),
    2. AdaBoost with shallow decision trees as weak learners,
    3. XGBoost (gradient boosting classifier).
  - **Hyperparameter search**:
    - Logistic Regression: **GridSearchCV** (as suggested by the course staff),
    - AdaBoost: **GridSearchCV**,
    - XGBoost: **RandomizedSearchCV** over a reasonable parameter space.
  - **Ensemble**:
    - Builds a **soft-voting ensemble** of the three tuned models.
    - The voting weights are derived from the cross-validation accuracy
      of each base model.
    - The final ensemble is trained on the full training set and used
      to produce a submission CSV.

This notebook corresponds to our **most complex model** in terms of
combining different learners and tuning their hyperparameters.

---

### 3. `Submission_3_LR.ipynb` – Interpretable 5-feature Logistic Regression

The third notebook focuses on **interpretability and simplicity**.  
Instead of using a large feature set, it builds a very compact table with
only **five hand-crafted features**, and trains a Logistic Regression model.

**Feature engineering**

The function `create_features1(...)` extracts, for each battle, the
following features:

1. **`hp_edge_final`**  
   Final HP edge: average HP% of player 2 minus average HP% of player 1
   at the end of the battle (over all revealed Pokémon).

2. **`used_count_diff`**  
   Difference in the number of distinct Pokémon actually used by the two
   players (player 2 minus player 1).

3. **`revealed_count_diff`**  
   Difference in the number of distinct Pokémon revealed by the two
   players (player 2 minus player 1).

4. **`p1_status_mean_final`**  
   Mean final status severity for player 1, using a mapping of status
   conditions to numeric severity (e.g. normal, paralyzed, burned, etc.).

5. **`status_severity_gap_final`**  
   Final status severity gap: mean status severity of player 2 minus that
   of player 1.

The resulting `train_df` therefore has these five features, plus
`battle_id` and `player_won`. The test table has the same five features
and `battle_id`, but no target.

**Model**

- Builds `X`, `y`, `X_test` from these five features.
- Trains a **Logistic Regression** model with a small yet meaningful
  hyperparameter grid:
  - two configurations:
    - elastic-net (`solver='saga', penalty='elasticnet'`) with a few values
      of `C` and `l1_ratio`,
    - standard L2 (`solver='lbfgs', penalty='l2'`) with its own `C` range.
  - **GridSearchCV** with 5-fold Stratified CV to select the best setting.
- The chosen model is then re-trained on the full training set and used to
  generate a submission file.

This notebook is designed to be **easy to read and to interpret**, since
all features have a clear intuitive meaning.
