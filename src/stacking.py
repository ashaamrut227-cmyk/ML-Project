"""Stacking ensemble for personal loan prediction.

Level 0 (base models): Logistic Regression, Random Forest, SVM (RBF), MLP
Level 1 (meta model):  Logistic Regression that learns how to combine their probabilities

Run from the repo root:   python src/stacking.py
"""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from evaluate import RESULTS_DIR, compute_metrics, plot_confusion_matrices, plot_roc
from preprocess import RANDOM_STATE, prepare_data


def base_models():
    """Level-0 models (fresh, unfitted copies each call)."""
    return [
        ("lr", LogisticRegression(class_weight="balanced", max_iter=1000)),
        ("rf", RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                      random_state=RANDOM_STATE, n_jobs=-1)),
        ("svm", SVC(C=10, kernel="rbf", class_weight="balanced", probability=True,
                    random_state=RANDOM_STATE)),
        ("mlp", MLPClassifier(hidden_layer_sizes=(50,), alpha=0.01, max_iter=500,
                              random_state=RANDOM_STATE)),
    ]


def build_stack(meta_class_weight=None):
    """Stacking: base models are cross-validated (5-fold) so the meta model only ever
    sees predictions made on data the base models had NOT trained on."""
    return StackingClassifier(
        estimators=base_models(),
        final_estimator=LogisticRegression(max_iter=1000, class_weight=meta_class_weight),
        cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE),
        stack_method="predict_proba",
        n_jobs=-1,
    )


class ThresholdedModel:
    """Wraps a fitted model and predicts 1 when P(accept) >= threshold (default 0.5).
    Probabilities are unchanged, so ROC-AUC is the same as the wrapped model."""

    def __init__(self, model, threshold):
        self.model = model
        self.threshold = threshold

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= self.threshold).astype(int)


def best_f1_threshold(y_true, proba):
    """Threshold that maximises F1 on the given (out-of-fold) probabilities."""
    prec, rec, thr = precision_recall_curve(y_true, proba)
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
    return float(thr[np.argmax(f1)])


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    X_train, X_test, y_train, y_test = prepare_data()

    # 1) Individual base models, for comparison.
    models = {
        "Logistic Regression (balanced)": clone(base_models()[0][1]),
        "Random Forest (balanced)": clone(base_models()[1][1]),
        "SVM RBF (balanced)": clone(base_models()[2][1]),
        "MLP": clone(base_models()[3][1]),
    }
    for name, m in models.items():
        m.fit(X_train, y_train)
        print(f"trained {name}")

    # 2) Stacking ensemble (default 0.5 threshold).
    stack = build_stack().fit(X_train, y_train)
    models["Stacking (threshold 0.5)"] = stack
    print("trained Stacking ensemble")

    # What does the meta model trust? Larger coefficient = more weight on that model.
    coefs = stack.final_estimator_.coef_.ravel()
    print("\nMeta-model weights (how much stacking trusts each base model):")
    for (name, _), c in zip(base_models(), coefs):
        print(f"  {name:>4}: {c:+.3f}")

    # 3) Stacking with a tuned threshold. The threshold is chosen from OUT-OF-FOLD
    #    predictions on the training set only, so the test set is never used for tuning.
    oof = cross_val_predict(
        build_stack(), X_train, y_train, method="predict_proba",
        cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE))[:, 1]
    threshold = best_f1_threshold(y_train, oof)
    print(f"\nF1-optimal threshold from training out-of-fold predictions: {threshold:.3f}")
    models[f"Stacking (threshold {threshold:.2f})"] = ThresholdedModel(stack, threshold)

    # 4) Evaluate everything on the untouched test set.
    results = pd.DataFrame(
        [compute_metrics(n, m, X_test, y_test) for n, m in models.items()]).round(4)
    results.to_csv(RESULTS_DIR / "stacking_metrics.csv", index=False)
    print("\n=== Stacking results (test set) ===")
    print(results.to_string(index=False))

    plot_roc(models, X_test, y_test, "ROC curves: base models vs stacking",
             "roc_stacking.png")
    plot_confusion_matrices(models, X_test, y_test, "confusion_matrices_stacking.png")
    print(f"\nSaved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()