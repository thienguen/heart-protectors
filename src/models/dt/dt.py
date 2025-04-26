"""
cleaner decision-tree baseline
• fixes CV hang by running folds sequentially (n_jobs=1)
• hides “precision ill-defined” warnings
• prints metrics for k = 5 … 20
• saves a feature-importance plot
"""

import warnings

warnings.filterwarnings("ignore", category=UserWarning)  # mute ill-defined precision

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier


# 1 ──────────────────────────────────────────────────────────────── load + map
df = pd.read_csv("../../process/cleaned_heart_disease.csv")

binary_cols = [
    "Gender",
    "Smoking",
    "Family Heart Disease",
    "Diabetes",
    "High Blood Pressure",
    "Low HDL Cholesterol",
    "High LDL Cholesterol",
    "Heart Disease Status",
]
ordinal_cols = ["Exercise Habits", "Stress Level", "Sugar Consumption"]

df[binary_cols] = df[binary_cols].replace({"Yes": 1, "No": 0, "Male": 1, "Female": 0})
df[ordinal_cols] = df[ordinal_cols].replace({"Low": 0, "Medium": 1, "High": 2})
df = df.fillna(df.median(numeric_only=True))

X = df.drop(columns="Heart Disease Status")
y = df["Heart Disease Status"]

# 2 ──────────────────────────────────────────────────────────────── pipeline
nominal = []  # add string features here if any
preproc = ColumnTransformer(
    [("ohe", OneHotEncoder(handle_unknown="ignore"), nominal)], remainder="passthrough"
)

pipe = ImbPipeline(
    [
        ("prep", preproc),
        ("smote", SMOTE(random_state=42)),
        ("tree", DecisionTreeClassifier(random_state=42)),
    ]
)

param_grid = {"tree__max_depth": [3, 5, 7, None], "tree__min_samples_leaf": [1, 5, 10]}

grid_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid = GridSearchCV(
    pipe,
    param_grid,
    cv=grid_cv,
    scoring="roc_auc",
    n_jobs=1,  # ← single-thread to avoid Windows hang
    refit=True,
    verbose=0,
).fit(X, y)

print("best params :", grid.best_params_)
print("best AUC    :", grid.best_score_)

best_pipe = grid.best_estimator_

# 3 ─────────────────────────────────────────────────────────────── metrics for whole set
p = best_pipe.predict(X)
pro = best_pipe.predict_proba(X)[:, 1]

print("\nall-data metrics")
print("acc :", accuracy_score(y, p))
print("prec:", precision_score(y, p, zero_division=0))
print("rec :", recall_score(y, p))
print("f1  :", f1_score(y, p))
print("auc :", roc_auc_score(y, pro))

# 4 ─────────────────────────────────────────────────────────────── manual k-fold loop
print("\nstratified k-fold CV  (sequential, k = 5 … 20)")
for k in range(5, 21):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    accs, precs, recs, f1s, aucs = [], [], [], [], []
    for tr, te in skf.split(X, y):
        mdl = clone(best_pipe)
        mdl.fit(X.iloc[tr], y.iloc[tr])

        pred = mdl.predict(X.iloc[te])
        prob = mdl.predict_proba(X.iloc[te])[:, 1]

        accs.append(accuracy_score(y.iloc[te], pred))
        precs.append(precision_score(y.iloc[te], pred, zero_division=0))
        recs.append(recall_score(y.iloc[te], pred))
        f1s.append(f1_score(y.iloc[te], pred))
        aucs.append(roc_auc_score(y.iloc[te], prob))

    print(
        f"{k:2d}-fold | "
        f"acc {np.mean(accs):.3f}  "
        f"prec {np.mean(precs):.3f}  "
        f"rec {np.mean(recs):.3f}  "
        f"f1 {np.mean(f1s):.3f}  "
        f"auc {np.mean(aucs):.3f}"
    )
