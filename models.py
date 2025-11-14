from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier

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

