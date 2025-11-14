from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier 

#Submission 1 model tuned
def get_model_sub1():
    """
    Logistic Regression (Submission 1) with fixed best hyperparameters.
    """
    model = Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(
            solver="saga",
            penalty="elasticnet",
            C=0.0092,
            l1_ratio=0.077,
            tol=8e-05,
            class_weight=None,
            n_jobs=-1,
            max_iter=12000,
            random_state=42,
        )),
    ])
    return model


def get_model_sub2():
    """
    Ensemble (Submission 2): LR + AdaBoost + XGBoost with fixed best hyperparameters.
    """
    # Logistic Regression (best params)
    lr = Pipeline([
        ("scale", StandardScaler(with_mean=False)),
        ("clf", LogisticRegression(
            solver="liblinear",
            penalty="l1",
            C=2.5,
            tol=1e-06,
            class_weight=None,
            max_iter=6000,
            n_jobs=-1,
            random_state=42,
        )),
    ])

    # AdaBoost (best params)
    ada = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1, random_state=42),
        algorithm="SAMME.R",
        n_estimators=330,
        learning_rate=0.26,
        random_state=42,
    )

    # XGBoost (best params)
    xgb = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=1000,
        max_depth=4,
        learning_rate=0.015,
        subsample=0.6,
        colsample_bytree=0.65,
        min_child_weight=3,
        reg_alpha=0.05,
        reg_lambda=1.0,
        gamma=0.1,
        n_jobs=-1,
        random_state=42,
    )

    ensemble = VotingClassifier(
        estimators=[
            ("lr", lr),
            ("ada", ada),
            ("xgb", xgb),
        ],
        voting="soft",
        weights=[0.7332496900000001, 0.72199009, 0.7247116899999999],
        n_jobs=-1,
    )

    return ensemble



def get_model_sub3():
    """
    Simple Logistic Regression (Submission 3) with fixed best hyperparameters.
    """
    model = Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(
            solver="lbfgs",
            penalty="l2",
            C=0.03,
            tol=0.001,
            class_weight=None,
            n_jobs=-1,
            max_iter=20000,
            random_state=42,
        )),
    ])
    return model

