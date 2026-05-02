"""
train.py — Training pipeline: MLP classifier + baselines.

Orchestrates the entire experiment:
1. Load and preprocess data
2. Train MLP classifier
3. Train baseline models (Logistic Regression, Random Forest)
4. Evaluate all models and produce comparison table
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import copy
import warnings
warnings.filterwarnings("ignore")

from data import load_data, run_eda, preprocess, split_data, scale_data, compute_class_weights, add_noise
from classifier import VanillaMLP
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from evaluate import (evaluate_model, evaluate_sklearn_model,
                      plot_roc_curves, plot_pr_curves, plot_confusion_matrices,
                      print_comparison_table, run_cross_validation,
                      run_sklearn_cross_validation, plot_shap_values,
                      make_poster_figures)

FIGURES_DIR = "figures"
CHECKPOINTS_DIR = "checkpoints"
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

SEED = 42
NOISE_FLIP_PROB = 0.15  # Per-bit flip probability on A1-A10 to break the deterministic rule
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(SEED)
np.random.seed(SEED)


def train_classifier(model: nn.Module, X_train: np.ndarray, y_train: np.ndarray,
                     X_val: np.ndarray, y_val: np.ndarray,
                     class_weights: dict, epochs: int = 50,
                     lr: float = 1e-3, batch_size: int = 32,
                     patience: int = 10, model_name: str = "model") -> nn.Module:
    """
    Train a PyTorch binary classifier with:
    - Weighted BCE loss (handles class imbalance)
    - Early stopping on validation recall (we want to catch ASD cases)

    Early stopping on recall (not loss) because in medical screening,
    missing a positive case (false negative) is far worse than a false alarm.
    """
    model = model.to(DEVICE)

    sample_weights = torch.FloatTensor([class_weights[int(y)] for y in y_train]).to(DEVICE)

    X_t = torch.FloatTensor(X_train).to(DEVICE)
    y_t = torch.FloatTensor(y_train).to(DEVICE)
    X_v = torch.FloatTensor(X_val).to(DEVICE)
    y_v = torch.FloatTensor(y_val).to(DEVICE)

    dataset = TensorDataset(X_t, y_t, sample_weights)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.BCELoss(reduction='none')

    best_recall = -1.0
    best_val_loss = float('inf')
    best_state = None
    wait = 0

    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y, batch_w in loader:
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = (criterion(preds, batch_y) * batch_w).mean()
            loss.backward()
            optimizer.step()

        # Evaluate on validation set — track recall for early stopping.
        # When recall is tied, use val loss as tiebreaker so the model
        # keeps improving even after recall plateaus (e.g. at 1.0).
        model.eval()
        with torch.no_grad():
            val_probs = model(X_v)
            val_preds = (val_probs >= 0.5).float()
            tp = ((val_preds == 1) & (y_v == 1)).sum().item()
            fn = ((val_preds == 0) & (y_v == 1)).sum().item()
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

            val_loss = criterion(val_probs, y_v).mean().item()

        improved = (recall > best_recall) or (recall == best_recall and val_loss < best_val_loss)
        if improved:
            best_recall = recall
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"  [{model_name}] Early stopping at epoch {epoch+1} (best val recall: {best_recall:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"  [{model_name}] Training complete — best val recall: {best_recall:.4f}")
    return model


def main():
    # --- Data ---
    print("Loading and preprocessing data...")
    df = load_data()
    run_eda(df)
    X, y, feature_names = preprocess(df)
    input_dim = X.shape[1]
    X_train_raw, X_test_raw, y_train, y_test = split_data(X, y)

    # Inject noise INDEPENDENTLY into train and test (different RNG seeds).
    # Q-CHAT-10 is deterministic (sum >= 3 => ASD), so unmodified data
    # gives ~100% accuracy. Noise simulates realistic questionnaire errors.
    print(f"\nInjecting {NOISE_FLIP_PROB*100:.0f}% bit-flip noise into A1-A10...")
    X_train_raw = add_noise(X_train_raw, feature_names, flip_prob=NOISE_FLIP_PROB, seed=SEED)
    X_test_raw = add_noise(X_test_raw, feature_names, flip_prob=NOISE_FLIP_PROB, seed=SEED + 1)

    # Fit scaler on training data ONLY — prevents leakage of test statistics
    X_train, X_test, scaler = scale_data(X_train_raw, X_test_raw)

    class_weights = compute_class_weights(y_train)

    # Validation split from training for early stopping
    X_tr, X_val, y_tr, y_val = split_data(X_train, y_train, test_size=0.15, seed=SEED)

    # --- MLP ---
    print("\n" + "=" * 60)
    print("MLP CLASSIFIER")
    print("=" * 60)
    mlp = VanillaMLP(input_dim)
    mlp = train_classifier(
        mlp, X_tr, y_tr, X_val, y_val,
        class_weights, epochs=100, lr=1e-3, patience=15,
        model_name="MLP"
    )
    torch.save(mlp.state_dict(), os.path.join(CHECKPOINTS_DIR, "best_mlp.pth"))

    # --- Logistic Regression ---
    print("\n" + "=" * 60)
    print("LOGISTIC REGRESSION")
    print("=" * 60)
    lr_model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=SEED)
    lr_model.fit(X_tr, y_tr)
    print("  Logistic Regression trained.")

    # --- Random Forest ---
    print("\n" + "=" * 60)
    print("RANDOM FOREST")
    print("=" * 60)
    rf_model = RandomForestClassifier(
        n_estimators=200, class_weight='balanced', random_state=SEED, n_jobs=-1
    )
    rf_model.fit(X_tr, y_tr)
    print("  Random Forest trained.")

    # ============================================================
    # Evaluation on held-out test set
    # ============================================================
    print("\n" + "=" * 60)
    print("EVALUATION ON TEST SET")
    print("=" * 60)

    results = {}

    results["MLP"] = evaluate_model(mlp, X_test, y_test, DEVICE, "MLP")

    for name, model in [("Logistic Regression", lr_model),
                        ("Random Forest", rf_model)]:
        results[name] = evaluate_sklearn_model(model, X_test, y_test, name)

    print_comparison_table(results)

    plot_roc_curves(results)
    plot_pr_curves(results)
    plot_confusion_matrices(results)

    # ============================================================
    # 5-Fold Cross-Validation
    # ============================================================
    print("\n" + "=" * 60)
    print("5-FOLD STRATIFIED CROSS-VALIDATION")
    print("=" * 60)

    cv_results = {}

    # Pass UNSCALED training data — each CV fold fits its own scaler
    cv_results["MLP"] = run_cross_validation(
        X_train_raw, y_train, input_dim, device=DEVICE
    )

    cv_results["Logistic Regression"] = run_sklearn_cross_validation(
        X_train_raw, y_train, model_type="lr"
    )
    cv_results["Random Forest"] = run_sklearn_cross_validation(
        X_train_raw, y_train, model_type="rf"
    )

    # Print CV summary
    print("\n" + "-" * 80)
    print(f"{'Model':<28} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'AUC-ROC':>10}")
    print("-" * 80)
    for name, cv in cv_results.items():
        print(f"{name:<28} "
              f"{cv['accuracy_mean']:.4f}\u00b1{cv['accuracy_std']:.3f} "
              f"{cv['precision_mean']:.4f}\u00b1{cv['precision_std']:.3f} "
              f"{cv['recall_mean']:.4f}\u00b1{cv['recall_std']:.3f} "
              f"{cv['f1_mean']:.4f}\u00b1{cv['f1_std']:.3f} "
              f"{cv['auc_roc_mean']:.4f}\u00b1{cv['auc_roc_std']:.3f}")
    print("-" * 80)

    # ============================================================
    # Poster Figures
    # ============================================================
    print("\n" + "=" * 60)
    print("POSTER FIGURES")
    print("=" * 60)
    make_poster_figures(results, cv_results)

    # ============================================================
    # SHAP Explainability
    # ============================================================
    print("\n" + "=" * 60)
    print("SHAP EXPLAINABILITY ANALYSIS")
    print("=" * 60)
    plot_shap_values(mlp, X_test, feature_names, DEVICE)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE — all plots saved to figures/")
    print("=" * 60)


if __name__ == "__main__":
    main()
