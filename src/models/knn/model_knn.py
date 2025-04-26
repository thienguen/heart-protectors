import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


# ------------------------- helper ---------------------------------
def knn_metrics(X_train, y_train, X_test, y_test, k=5):
    """Train KNN(k) and return dict of evaluation metrics on X_test/y_test."""
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)
    y_proba = knn.predict_proba(X_test)[:, 1]

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "auc": roc_auc_score(y_test, y_proba),
    }


# ------------------------- load & encode --------------------------
data = pd.read_csv("../../process/cleaned_heart_disease.csv")

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
ordinal_map = {"Low": 0, "Medium": 1, "High": 2}

for col in binary_cols:
    if data[col].dtype == "object":
        cats = data[col].unique()
        data[col] = data[col].map({cats[0]: 0, cats[1]: 1})

for col in ordinal_cols:
    data[col] = data[col].map(ordinal_map)

X = data.drop(columns="Heart Disease Status")
y = data["Heart Disease Status"]

# ------------------------- hold-out test --------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=0, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

K_CHOSEN = 5  # ← adjust if you prefer another k
metrics = knn_metrics(X_train_s, y_train, X_test_s, y_test, k=K_CHOSEN)
print(f"\n=== Hold-out evaluation (k={K_CHOSEN}) ===")
for m, v in metrics.items():
    print(f"{m:9s}: {v:.3f}  ({v * 100:.1f}%)")

# ------------------------- coarse grid (k = 1 … 10) ---------------
print("\nAccuracy for k = 1 … 10 (hold-out split):")
for k in range(1, 11):
    acc = knn_metrics(X_train_s, y_train, X_test_s, y_test, k)["accuracy"]
    print(f" k={k:2d}: {acc:.3f}  ({acc * 100:.1f}%)")

mean_acc = np.mean(
    [
        knn_metrics(X_train_s, y_train, X_test_s, y_test, k)["accuracy"]
        for k in range(1, 11)
    ]
)
print(f"\nMean accuracy (k=1–10): {mean_acc:.3f}  ({mean_acc * 100:.1f}%)")

# ------------------------- k-fold cross-val (accuracy only) -------
print("\nK-Fold Cross-Validation Accuracies (k folds = 5 … 20, kNN k=5):")
for folds in range(5, 21):
    kf = KFold(n_splits=folds, shuffle=True, random_state=0)
    fold_acc = []

    for train_idx, test_idx in kf.split(X):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr)
        X_te_s = sc.transform(X_te)

        fold_acc.append(knn_metrics(X_tr_s, y_tr, X_te_s, y_te, k=K_CHOSEN)["accuracy"])

    print(
        f" {folds:2d}-fold: {np.mean(fold_acc):.3f}  ({np.mean(fold_acc) * 100:.1f}%)"
    )
