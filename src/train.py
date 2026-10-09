
"""Train and compare models for personal loan acceptance.
 
Run from the repo root:   python src/train.py
"""
import time
 
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
 
from evaluate import RESULTS_DIR, compute_metrics, plot_confusion_matrices, plot_roc
from models import KMeansClassifier
from preprocess import RANDOM_STATE, prepare_data
 
 
def fit_all(models, X_train, y_train):
    for name, model in models.items():
        t = time.time()
        model.fit(X_train, y_train)
        print(f"  trained {name} ({time.time() - t:.1f}s)")
    return models
 
 
def tune(estimator, grid, X_train, y_train, name):
    """Grid search with 5-fold stratified CV, optimising ROC-AUC."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    gs = GridSearchCV(estimator, grid, scoring="roc_auc", cv=cv, n_jobs=-1)
    gs.fit(X_train, y_train)
    print(f"  {name} best params: {gs.best_params_} (CV AUC = {gs.best_score_:.3f})")
    return gs.best_estimator_
 
 
def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    X_train, X_test, y_train, y_test = prepare_data()
    print(f"Train: {X_train.shape}, Test: {X_test.shape}, "
          f"positives in test: {int(y_test.sum())}\n")
 
    # ---- Part 1: baseline (the five algorithms from the problem statement) ----
    print("Part 1: baseline models (scaled features, no class weighting)")
    baseline = {
        "Perceptron": Perceptron(random_state=RANDOM_STATE),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "K-Means": KMeansClassifier(random_state=RANDOM_STATE),
        "SVM (linear)": SVC(kernel="linear", random_state=RANDOM_STATE),
        "MLP": MLPClassifier(hidden_layer_sizes=(100,), activation="relu", solver="adam",
                             alpha=0.0001, max_iter=500, random_state=RANDOM_STATE),
    }
    fit_all(baseline, X_train, y_train)
 
    # ---- Part 2: improvements (class weights, tuning, Random Forest) ----
    print("\nPart 2: improved models")
    improved = {
        "Perceptron (balanced)": Perceptron(class_weight="balanced", random_state=RANDOM_STATE),
        "LogReg (balanced)": LogisticRegression(class_weight="balanced", max_iter=1000),
        "SVM linear (balanced)": SVC(kernel="linear", class_weight="balanced",
                                     random_state=RANDOM_STATE),
        "Random Forest (balanced)": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1),
    }
    fit_all(improved, X_train, y_train)
 
    print("  Tuning SVM and MLP with GridSearchCV ...")
    improved["SVM (tuned)"] = tune(
        SVC(class_weight="balanced", random_state=RANDOM_STATE),
        {"C": [0.1, 1, 10, 100], "kernel": ["linear", "rbf"]},
        X_train, y_train, "SVM")
    improved["MLP (tuned)"] = tune(
        MLPClassifier(activation="relu", solver="adam", max_iter=500, random_state=RANDOM_STATE),
        {"hidden_layer_sizes": [(50,), (100,), (64, 32)], "alpha": [0.0001, 0.001, 0.01]},
        X_train, y_train, "MLP")
 
    # ---- Evaluate on the held-out test set ----
    all_models = {**baseline, **improved}
    rows = [compute_metrics(n, m, X_test, y_test) for n, m in all_models.items()]
    results = pd.DataFrame(rows).round(4)
    results.to_csv(RESULTS_DIR / "metrics.csv", index=False)
 
    print("\n=== Test-set results ===")
    print(results.to_string(index=False))
 
    # ---- Plots ----
    plot_roc(baseline, X_test, y_test, "ROC curves: baseline models", "roc_baseline.png")
    best_names = ["LogReg (balanced)", "SVM (tuned)", "MLP (tuned)", "Random Forest (balanced)"]
    best = {n: all_models[n] for n in best_names}
    plot_roc({**{"K-Means": baseline["K-Means"]}, **best}, X_test, y_test,
             "ROC curves: improved models", "roc_improved.png")
    plot_confusion_matrices(all_models, X_test, y_test, "confusion_matrices.png")
 
    winner = results.sort_values("ROC-AUC", ascending=False).iloc[0]
    print(f"\nBest by ROC-AUC: {winner['Model']} (AUC = {winner['ROC-AUC']})")
    print(f"Results saved to {RESULTS_DIR}")
 
 
if __name__ == "__main__":
    main()
 
