"""Combine Logistic Regression, Random Forest and MLP into ensembles.

Run from the repo root:   python src/ensemble.py
"""
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier

from evaluate import RESULTS_DIR, compute_metrics, plot_confusion_matrices, plot_roc
from preprocess import RANDOM_STATE, prepare_data


def base_models():
    """The three member models (fresh, unfitted copies each call)."""
    return [
        ("lr", LogisticRegression(class_weight="balanced", max_iter=1000)),
        ("rf", RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                      random_state=RANDOM_STATE, n_jobs=-1)),
        ("mlp", MLPClassifier(hidden_layer_sizes=(50,), alpha=0.01, max_iter=500,
                              random_state=RANDOM_STATE)),
    ]


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    X_train, X_test, y_train, y_test = prepare_data()

    members = {
        "Logistic Regression (balanced)": clone(base_models()[0][1]),
        "Random Forest (balanced)": clone(base_models()[1][1]),
        "MLP": clone(base_models()[2][1]),
    }

    ensembles = {
        # Soft voting: average the predicted probabilities of the three models.
        "Voting (soft, equal weights)": VotingClassifier(base_models(), voting="soft"),
        # Weighted voting: trust RF and MLP more than Logistic Regression.
        "Voting (soft, weighted 1:2:2)": VotingClassifier(base_models(), voting="soft",
                                                          weights=[1, 2, 2]),
        # Stacking: a Logistic Regression learns how best to combine the 3 models.
        "Stacking (meta = LogReg)": StackingClassifier(
            base_models(), final_estimator=LogisticRegression(max_iter=1000),
            cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)),
    }

    all_models = {**members, **ensembles}
    for name, model in all_models.items():
        model.fit(X_train, y_train)
        print(f"trained {name}")

    results = pd.DataFrame(
        [compute_metrics(n, m, X_test, y_test) for n, m in all_models.items()]).round(4)
    results.to_csv(RESULTS_DIR / "ensemble_metrics.csv", index=False)
    print("\n=== Ensemble results (test set) ===")
    print(results.to_string(index=False))

    plot_roc(all_models, X_test, y_test, "ROC curves: individual models vs ensembles",
             "roc_ensemble.png")
    plot_confusion_matrices(all_models, X_test, y_test, "confusion_matrices_ensemble.png")
    print(f"\nSaved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()