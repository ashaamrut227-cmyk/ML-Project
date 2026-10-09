"""Compare ways of handling class imbalance: none, class weights, SMOTE, SMOTENC.

Run from the repo root:   python src/smote.py

SMOTE only ever touches the TRAINING data. It sits inside an imblearn Pipeline, so the
test set is never resampled and keeps its real 9.6% acceptor rate.
"""
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from imblearn.over_sampling import SMOTE, SMOTENC
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.neural_network import MLPClassifier

from evaluate import RESULTS_DIR, compute_metrics, get_scores
from preprocess import RANDOM_STATE, prepare_data


def make_classifier(kind, class_weight=None):
    if kind == "Logistic Regression":
        return LogisticRegression(max_iter=1000, class_weight=class_weight)
    if kind == "Random Forest":
        return RandomForestClassifier(n_estimators=300, class_weight=class_weight,
                                      random_state=RANDOM_STATE, n_jobs=-1)
    if kind == "MLP":  # MLPClassifier has no class_weight option
        return MLPClassifier(hidden_layer_sizes=(50,), alpha=0.01, max_iter=500,
                             random_state=RANDOM_STATE)
    raise ValueError(kind)


def binary_columns(X):
    """Indices of 0/1 columns (account flags etc.), treated as categorical by SMOTENC."""
    return [i for i, c in enumerate(X.columns) if set(X[c].unique()) <= {0.0, 1.0}]


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    X_train, X_test, y_train, y_test = prepare_data()
    cat_idx = binary_columns(X_train)

    # Show what SMOTE does to the training set (the test set is never resampled).
    _, y_res = SMOTE(random_state=RANDOM_STATE).fit_resample(X_train, y_train)
    print(f"\nTraining class counts before SMOTE: {dict(Counter(y_train))}")
    print(f"Training class counts after  SMOTE: {dict(Counter(y_res))}")
    print(f"Test class counts (unchanged):      {dict(Counter(y_test))}\n")

    strategies = ["No balancing", "Class weights", "SMOTE", "SMOTENC"]
    kinds = ["Logistic Regression", "Random Forest", "MLP"]

    models = {}
    for kind in kinds:
        for strat in strategies:
            if strat == "Class weights":
                if kind == "MLP":
                    continue  # not supported by MLPClassifier
                clf = make_classifier(kind, class_weight="balanced")
                model = clf
            elif strat == "SMOTE":
                model = Pipeline([("smote", SMOTE(random_state=RANDOM_STATE)),
                                  ("clf", make_classifier(kind))])
            elif strat == "SMOTENC":
                model = Pipeline([("smote", SMOTENC(categorical_features=cat_idx,
                                                    random_state=RANDOM_STATE)),
                                  ("clf", make_classifier(kind))])
            else:
                model = make_classifier(kind)
            models[(kind, strat)] = model

    for (kind, strat), model in models.items():
        model.fit(X_train, y_train)
        print(f"trained {kind} | {strat}")

    rows = []
    for (kind, strat), model in models.items():
        row = compute_metrics(f"{kind} | {strat}", model, X_test, y_test)
        row["Strategy"] = strat
        rows.append(row)
    results = pd.DataFrame(rows).round(4)
    results.to_csv(RESULTS_DIR / "smote_metrics.csv", index=False)
    print("\n=== Class imbalance strategies (test set) ===")
    print(results.drop(columns="Strategy").to_string(index=False))

    # ROC curves: one panel per model type, one curve per strategy.
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, kind in zip(axes, kinds):
        for (k, strat), model in models.items():
            if k != kind:
                continue
            fpr, tpr, _ = roc_curve(y_test, get_scores(model, X_test))
            auc = results.loc[results["Model"] == f"{k} | {strat}", "ROC-AUC"].iloc[0]
            ax.plot(fpr, tpr, label=f"{strat} (AUC={auc:.3f})")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
        ax.set_title(kind)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "roc_smote.png", dpi=150)
    plt.close()

    # Recall vs precision bars: shows the trade-off each strategy makes.
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for ax, kind in zip(axes, kinds):
        sub = results[results["Model"].str.startswith(kind)]
        x = range(len(sub))
        ax.bar([i - 0.2 for i in x], sub["Recall"], width=0.4, label="Recall")
        ax.bar([i + 0.2 for i in x], sub["Precision"], width=0.4, label="Precision")
        ax.set_xticks(list(x))
        ax.set_xticklabels(sub["Strategy"], rotation=20, fontsize=8)
        ax.set_title(kind)
        ax.set_ylim(0, 1.05)
    axes[0].legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "smote_recall_precision.png", dpi=150)
    plt.close()

    print(f"\nSaved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()