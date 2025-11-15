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

In the notebooks we have provided some comments, that aims to help the reader moving around the code and get what the code is doing.
In case you would like to deep dive into the code a bit more, please check the notebooks themselvelf.

**AI assistance disclaimer**

Parts of the code contained in our project may have been drafted or refined with the help of AI-based tools. The use of AI was strictly limited to aspects like comments, iterative feature search and minor implementation details. All core ideas, modeling choices, and logical structures implemented in the code and in the models were entirely conceived and designed by the members of the group, without external intellectual contribution, relying solely on online documentation, our own knowledge and the insights provided by the course lectures.

---

## Repository contents

### Code 1. `Submission_1_LR.ipynb` – Full feature Logistic Regression

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

## Notebook overview – `fin_reg_log_commented.ipynb`

**Here is a small cell-by-cell dive**

---

- **Cell 1 – Data loading & 5-feature engineering**
  - Input files & paths:
    - Defines `DATA_DIR` and JSONL files: `train.jsonl` (with labels) and `test.jsonl` (without labels).
  - JSONL loader:
    - `load_jsonl(path)` reads each non-empty line as a JSON object (one battle per line).
  - Status mapping:
    - `MAP_STATUS` dictionary converts textual statuses (PAR, BRN, PSN, TOX, FRZ, SLP, etc.) into numeric severity scores.
  - Target & identifiers:
    - `battle_id`: unique ID of each battle.
    - `player_won`: binary target (1 if player 1 wins, 0 otherwise), present only in the training data.
  - Feature creation (one row per battle) via `create_features1(data)`:
    - Parses `battle_timeline` and tracks last-seen HP and status for each Pokémon.
    - Builds the following simple, interpretable features:
      - `hp_edge_final`  
        – Average final HP% of player 2’s team minus player 1’s team.
      - `used_count_diff`  
        – (# distinct species used by player 1) − (# distinct species used by player 2).
      - `status_severity_gap_final`  
        – Mean final status severity of player 2’s team minus player 1’s team (using `MAP_STATUS`).
      - `revealed_count_diff`  
        – (# distinct species that appeared on field for player 1) − (# for player 2).
      - `p1_status_mean_final`  
        – Mean final status severity of player 1’s team.
  - Output of this cell:
    - `train_df`: feature table for training battles (includes `battle_id`, `player_won`, and the 5 engineered features).
    - `test_df`: feature table for test battles (includes `battle_id` and the 5 engineered features).

---

- **Cell 2 – Train/test matrices**
  - Column alignment:
    - Intersects columns of `train_df` and `test_df` and removes non-feature fields (`battle_id`, `player_won`).
  - Feature matrices:
    - `X`: NumPy array with all feature columns from `train_df`.
    - `y`: target vector (`player_won` as integers).
    - `X_test`: NumPy array with the same feature columns from `test_df`.
  - Sanity checks:
    - Prints shapes of `X`, `y`, and `X_test` to verify consistency before model training.

---

- **Cell 3 – Logistic Regression model, tuning & submission**
  - Pipeline structure:
    - `Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(...))])`
    - Standardizes all features, then fits a Logistic Regression classifier.
  - Hyperparameter search (GridSearchCV) with 3 LR families:
    - **Elastic Net (solver `"saga"`):**
      - Tunes `C`, `l1_ratio`, `tol`, `class_weight`, with `penalty="elasticnet"`.
    - **L2 regularization with `"lbfgs"`:**
      - Tunes `C`, `tol`, `class_weight`, with `penalty="l2"`.
    - **L1/L2 regularization with `"liblinear"`:**
      - Tunes `C`, `tol`, `class_weight`, with `penalty` in `["l1", "l2"]`.
  - Cross Validation setup:
    - Uses `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`.
    - `GridSearchCV` optimizes accuracy with `n_jobs=-1` and `refit=True`.
  - Final model & coefficients:
    - Extracts `best_model = grid.best_estimator_`.
    - Retrieves logistic coefficients for the 5 features and builds a small importance table (feature name, coefficient, absolute coefficient).
    - Prints cross-validated accuracy (mean and per-fold scores) and training-set accuracy as a sanity check.
  - Predictions and submission file:
    - Refits `best_model` on the full training set (`X`, `y`).
    - Predicts `player_won` on `X_test`.
    - Builds the submission DataFrame with columns:
      - `battle_id`
      - `player_won`
    - Saves the final file as `submission_lr2.csv`, ready for Kaggle submission.


### Code 2. `Submission_2_ENS.ipynb` – Ensemble of LR, AdaBoost and XGBoost

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

## Notebook overview – `fin_01ensamble_commented.ipynb`

**Here is a small cell-by-cell dive**

- **Cell 1 – Data loading & feature engineering**
  - Input files & paths (`train.jsonl`, `test.jsonl`)
  - JSONL loader:
    - Simple helper that reads each line as a JSON object (one battle per line).
  - Game knowledge & mappings:
    - Gen-1 base stats for each Pokémon (`species` dict).
    - Type mapping for each Pokémon (`types_map`).
    - Status mapping (`MAP_STATUS`) with severity scores for conditions like PAR, BRN, TOX, SLP, FRZ.
  - Target & identifiers:
    - `battle_id`: unique ID of the battle.
    - `player_won`: binary target, 1 if player 1 wins, 0 otherwise (only in `train`).
  - Feature creation (one row per battle):
    - Final board state:
      - `hp_edge_final` – average HP% of p2 team minus p1 team at the end.
      - `p1_alive_final` – number of surviving Pokémon on p1’s side.
      - `p1_status_mean_final` – mean status severity on p1’s team at the end.
      - `status_severity_gap_final` – final status severity of p2 minus p1.
      - `revealed_count_diff` – (# mons p1 revealed) − (# mons p2 revealed).
    - HP dynamics & tempo:
      - `hp_gap_peak`, `hp_gap_peak_turn_share`, `hp_gap_var`, `hp_gap_autocorr`.
      - `hp_gap_sign_flips`, `hp_gap_slope_jump`, `p2_late_damage`.
    - Status & control:
      - `status_turns_advantage`, `tox_ratio_diff`, `status_diversity_p1`.
      - `p1_sleep_streak_max`, `sleep_streak_max_diff`.
      - `p1_turns_par`, `p2_turns_brn`.
    - Speed, switching & initiative:
      - `used_mean_spe_diff`, `p2_used_count`.
      - `eff_speed_adv_share_p2`, `eff_speed_edge_avg`.
      - `initiative_early_diff`, `initiative_late_diff`.
      - `p1_pingpong_switches`, `pingpong_switches_diff`.
      - `both_switched_share`, `p1_switch_late_share`, `forced_switch_share_diff`.
      - `comeback_time_share_diff`.
    - Move usage & power:
      - `attacks_rate_diff` – difference in attack frequency between players.
      - `bp_mean_p2` – mean base power of moves used by p2.
      - `move_diversity_p1`, `counter_count_diff`, `boom_count_diff`.
      - `confusion_exp_dmg_ratio_diff`, `substitute_break_rate_diff`.
      - `heal_efficiency_diff`, `heal_mid_diff`, `heal_late_diff`.
    - Type matchups & immunities:
      - `lead_type_edge`, `lead_def_edge` – type/defense edge at turn 1.
      - `types_last_round` – type edge on the last turn.
      - `type_seen_count_diff`, `p2_seen_type_count`.
      - `rs_hit_share_diff`, `p1_immune_count`, `immune_count_diff`.
    - Diversity & entropy:
      - `active_entropy_diff` – difference in entropy of active-mon usage.
  - Output of this cell:
    - `train_df`: feature table for training battles (includes `battle_id` and `player_won`).
    - `test_df`: feature table for test battles (includes `battle_id`).

---

- **Cell 2 – Train/test matrices**
  - Column alignment:
    - Intersects train/test columns and drops non-feature fields (`battle_id`, `player_won`).
  - Feature matrices:
    - `X`: NumPy array of features from `train_df` (all engineered columns).
    - `y`: target vector (`player_won` as int).
    - `X_test`: NumPy array of features from `test_df`.
  - Sanity checks:
    - Prints shapes of `X`, `y`, `X_test` to ensure consistency.

---

- **Cell 3 – Models, tuning, ensemble & submission**
  - Pipeline structure:
    - **Logistic Regression**:
      - `Pipeline([("scale", StandardScaler(with_mean=False)), ("clf", LogisticRegression(...))])`
      - Linear baseline with L1 regularization (to be tuned).
    - **AdaBoost**:
      - `Pipeline([("ada", AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1 or 2), ...))])`
      - Boosted shallow decision trees to capture non-linear effects.
    - **XGBoost**:
      - `Pipeline([("xgb", XGBClassifier(objective="binary:logistic", eval_metric="logloss", ...))])`
      - Gradient-boosted trees tailored for binary classification.
  - Hyperparameter search:
    - **Logistic Regression (GridSearchCV)**:
      - Tunes solver/penalty/regularization (`C`, `tol`, `class_weight`, `max_iter`) around L1-regularized `liblinear`.
    - **AdaBoost (GridSearchCV)**:
      - Tunes number of estimators, learning rate, and tree depth (1–2).
    - **XGBoost (RandomizedSearchCV)**:
      - Tunes `n_estimators`, `max_depth`, `learning_rate`, `subsample`,
        `colsample_bytree`, `min_child_weight`, `reg_alpha`, `reg_lambda`, `gamma`.
      - Samples up to 200 parameter combinations.
    - All searches use:
      - `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`.
      - `scoring="accuracy"`, `n_jobs=-1`.
  - Cross-validation & model comparison:
    - Computes mean CV accuracy for each tuned base model (`lr_best`, `ada_best`, `xgb_best`).
    - Uses these accuracies to derive ensemble weights.
  - Final ensemble model:
    - **Soft voting classifier** with probability averaging:
      - Estimators: tuned Logistic Regression, AdaBoost, XGBoost.
      - Weights proportional to squared CV accuracy of each component.
    - Evaluates ensemble CV accuracy and selects it as `best_model`.
  - Training, predictions & submission:
    - Fits `best_model` on the full training set (`X`, `y`).
    - Computes training accuracy for sanity check.
    - Predicts probabilities on `X_test`, thresholds at 0.5 to get `player_won`.
    - Builds submission DataFrame:
      - Columns: `battle_id`, `player_won`.
    - Saves the final file as `submission_ens.csv` ready for Kaggle submission.


---

### Code 3. `Submission_3_LR.ipynb` – Interpretable 5-feature Logistic Regression

The third notebook focuses on **interpretability and simplicity**.  
Instead of using a large feature set, it builds a very compact table with
only **five hand-crafted features**, and trains a Logistic Regression model.

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

## Notebook overview – `fin_terzo_modello_commented.ipynb`

**Here is a small cell-by-cell dive**

- **Cell 1 – Data loading & feature engineering**
  - Input files & paths:
    - Absolute paths to `train.jsonl` and `test.jsonl`.
    - Helper `load_jsonl(path)` that reads one JSON battle per line.
  - Game knowledge & mappings:
    - Gen-1 base stats for each Pokémon (`species` dict).
    - Type mapping for each Pokémon (`types` dict).
    - Type effectiveness chart (`effectiveness`) and `type_match(...)` helper to turn it into a numeric edge.
    - Status mapping (`MAP_STATUS`) with severity scores for conditions like PAR, BRN, TOX, SLP, FRZ.
  - Target & identifiers:
    - `battle_id`: unique ID of the battle.
    - `player_won`: binary target, 1 if player 1 wins, 0 otherwise (only in `train`).
  - Feature creation (one row per battle):
    - Final board & status:
      - `hp_edge_final` – final HP edge between p2 and p1 teams.
      - `alive_diff_final` – difference in # of surviving Pokémon.
      - `p1_status_mean_final` – mean final status severity on p1’s team.
      - `status_severity_gap_final` – final status severity of p2 minus p1.
      - `severe_status_share`, `severe_status_early_share`, `severe2_turns_diff`.
      - `revealed_count_diff` – (# mons p1 revealed) − (# mons p2 revealed).
    - HP dynamics & damage / healing:
      - `p2_early_damage`, `p2_late_damage`, `p1_mid_damage`.
      - `hp_gap_early`, `hp_gap_mid`, `hp_gap_var`, `hp_gap_sign_flips`.
      - `heal_mid_diff`, `heal_late_diff`, `mean_remaining_hp_p2`.
    - Status control:
      - `p1_turns_par`, `p1_turns_brn`, `p1_turns_psn_tox`.
      - `p2_turns_psn_tox`, `p2_turns_brn`, `p2_turns_slp`.
      - `par_turns_diff`, `status_diversity_diff`.
    - Speed, initiative & switching:
      - `used_mean_spe_diff`, `p1_used_mean_spe`.
      - `initiative_early_diff`, `initiative_late_diff`.
      - `p2_switch_early_share`, `run_len_mean_diff`, `p2_run_len_mean`.
      - `p1_switch_late_share`.
    - Type matchups & immunities:
      - `lead_type_edge`, `lead_def_edge`, `lead_type_fb_agree`, `lead_speed_fb_agree`.
      - `types_last_round` – final type situation.
      - `type_edge_avg_diff`, `p2_to_p1_type_edge_avg`.
      - `type_seen_count_diff`, `p2_seen_type_count`.
      - `resist_count_diff`, `p1_immune_count`, `immune_count_diff`.
    - Move usage, power & accuracy:
      - `bp_mean_p2`, `bp_std_p1`, `bp_std_diff`.
      - `low_acc_share_diff`, `acc_mean_diff`, `acc_mean_p2`.
      - `atk_edge_used`.
    - KOs & boosts:
      - `first_blood_happened`, `first_blood_side`.
      - `ko_rate_total`.
      - `p2_max_boost_sum`, `boost_turns_diff`.
    - Diversity & entropy:
      - `active_entropy_diff` – difference in entropy of active-mon usage.
      - `p2_used_count`.
  - Output of this cell:
    - `train_df`: feature table for training battles (includes `battle_id` and `player_won`).
    - `test_df`: feature table for test battles (includes `battle_id`).

---

- **Cell 2 – Train/test matrices**
  - Column alignment:
    - Intersects train/test columns and drops non-feature fields (`battle_id`, `player_won`).
  - Feature matrices:
    - `X`: NumPy array of features from `train_df` (all engineered columns).
    - `y`: target vector (`player_won` as int).
    - `X_test`: NumPy array of features from `test_df` with identical columns/order.
  - Sanity checks:
    - Prints shapes of `X`, `y`, `X_test` to make sure everything aligns.

---

- **Cell 3 – Logistic Regression model & submission**
  - Pipeline structure:
    - `Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(...))])`
    - Step 1: `StandardScaler` to standardize all features.
    - Step 2: `LogisticRegression` with elastic net regularization.
  - Hyperparameter search (GridSearchCV):
    - Tunes around a narrow region of good values using:
      - `solver="saga"` with `penalty="elasticnet"`.
      - `C` values near 0.01 (regularization strength).
      - `l1_ratio` values around 0.07–0.08 (L1/L2 mix).
      - Small variations of `tol` for convergence precision.
      - `class_weight=None`, `n_jobs=-1` to use all cores.
    - Uses `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` and `scoring="accuracy"`.
  - Cross-validation & best model:
    - Runs 5-fold CV over the grid and reports:
      - `best_score_`: mean CV accuracy of the best setting.
      - `best_params_`: selected hyperparameters.
    - Extracts `best_model = grid.best_estimator_` (pipeline with scaler + tuned logistic regression).
  - Training, predictions & submission:
    - Refits `best_model` on the full training set (`X`, `y`).
    - Predicts on `X_test` to obtain `player_won` for each test battle.
    - Builds the submission DataFrame with:
      - `battle_id` from `test_df`.
      - Predicted `player_won` (0/1) from the model.
    - Saves the final CSV as `submission1.csv`, ready for Kaggle submission.
