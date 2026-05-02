"""
evaluate.py — Evaluation metrics, plots, cross-validation, and SHAP explainability.

All models are evaluated on the same metrics:
- Accuracy, Precision, Recall, F1, AUC-ROC
- Confusion matrix, ROC curve, Precision-Recall curve
- 5-fold stratified CV with per-fold + mean±std reporting

Priority metric is RECALL — in ASD screening, missing a positive case
(false negative) has far worse consequences than a false alarm.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             roc_curve, precision_recall_curve, auc)
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings("ignore")

FIGURES_DIR = "figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

SEED = 42


def get_predictions(model, X: np.ndarray, device: torch.device):
    """Get probability predictions from a PyTorch model."""
    model.eval()
    with torch.no_grad():
        X_t = torch.FloatTensor(X).to(device)
        probs = model(X_t).cpu().numpy()
    return probs


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    """Compute all classification metrics from probabilities."""
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc_roc": roc_auc_score(y_true, y_prob),
        "y_prob": y_prob,
        "y_pred": y_pred,
    }


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray,
                   device: torch.device, name: str) -> dict:
    """Evaluate a PyTorch model and print results."""
    y_prob = get_predictions(model, X_test, device)
    metrics = compute_metrics(y_test, y_prob)
    metrics["y_true"] = y_test
    _print_metrics(name, metrics)
    return metrics


def evaluate_sklearn_model(model, X_test: np.ndarray, y_test: np.ndarray, name: str) -> dict:
    """Evaluate a sklearn model and print results."""
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test, y_prob)
    metrics["y_true"] = y_test
    _print_metrics(name, metrics)
    return metrics


def _print_metrics(name: str, metrics: dict):
    """Pretty-print metrics for a single model."""
    print(f"\n  {name}:")
    print(f"    Accuracy:  {metrics['accuracy']:.4f}")
    print(f"    Precision: {metrics['precision']:.4f}")
    print(f"    Recall:    {metrics['recall']:.4f}  {'✓ ≥0.90' if metrics['recall'] >= 0.90 else '✗ <0.90'}")
    print(f"    F1:        {metrics['f1']:.4f}")
    print(f"    AUC-ROC:   {metrics['auc_roc']:.4f}")


def print_comparison_table(results: dict):
    """Print a side-by-side comparison of all models."""
    print("\n" + "=" * 80)
    print("FINAL MODEL COMPARISON (Test Set)")
    print("=" * 80)
    print(f"{'Model':<28} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'AUC-ROC':>10}")
    print("-" * 80)
    for name, m in results.items():
        recall_flag = " *" if m['recall'] >= 0.90 else ""
        print(f"{name:<28} {m['accuracy']:>10.4f} {m['precision']:>10.4f} "
              f"{m['recall']:>10.4f}{recall_flag} {m['f1']:>10.4f} {m['auc_roc']:>10.4f}")
    print("-" * 80)
    print("  * = recall ≥ 0.90 target met")


# ============================================================
# Plotting Functions
# ============================================================

def plot_roc_curves(results: dict):
    """Plot ROC curves for all models on the same axes."""
    plt.figure(figsize=(8, 6))
    colors = ["#2196F3", "#F44336", "#4CAF50", "#FF9800"]
    for (name, m), color in zip(results.items(), colors):
        y_true = m["y_true"]
        fpr, tpr, _ = roc_curve(y_true, m["y_prob"])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, linewidth=2, label=f"{name} (AUC={roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], 'k--', alpha=0.3, label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves — All Models")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "roc_curves.png"), dpi=150)
    plt.close()
    print(f"  ROC curves saved to {FIGURES_DIR}/roc_curves.png")


def plot_pr_curves(results: dict):
    """Plot precision-recall curves for all models."""
    plt.figure(figsize=(8, 6))
    colors = ["#2196F3", "#F44336", "#4CAF50", "#FF9800"]
    for (name, m), color in zip(results.items(), colors):
        y_true = m["y_true"]
        prec, rec, _ = precision_recall_curve(y_true, m["y_prob"])
        pr_auc = auc(rec, prec)
        plt.plot(rec, prec, color=color, linewidth=2, label=f"{name} (AUC={pr_auc:.3f})")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves — All Models")
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "pr_curves.png"), dpi=150)
    plt.close()
    print(f"  PR curves saved to {FIGURES_DIR}/pr_curves.png")


def plot_confusion_matrices(results: dict):
    """Plot confusion matrices for all models in a grid."""
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (name, m) in zip(axes, results.items()):
        y_true = m["y_true"]
        cm = confusion_matrix(y_true, m["y_pred"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Non-ASD", "ASD"], yticklabels=["Non-ASD", "ASD"])
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "confusion_matrices.png"), dpi=150)
    plt.close()
    print(f"  Confusion matrices saved to {FIGURES_DIR}/confusion_matrices.png")


# ============================================================
# Poster-Quality Charts
# ============================================================

POSTER_COLORS = {
    "MLP": "#1f77b4",
    "Logistic Regression": "#ff7f0e",
    "Random Forest": "#2ca02c",
}


def plot_metrics_grouped_bars(results: dict, save_name: str = "poster_metrics_test.png",
                               title: str = "Test Set Performance"):
    """Grouped bar chart comparing all metrics across models."""
    metrics = ["accuracy", "precision", "recall", "f1", "auc_roc"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1", "AUC-ROC"]
    models = list(results.keys())

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(metrics))
    width = 0.8 / len(models)

    for i, model in enumerate(models):
        values = [results[model][m] for m in metrics]
        offset = (i - (len(models) - 1) / 2) * width
        bars = ax.bar(x + offset, values, width, label=model,
                      color=POSTER_COLORS.get(model, "#888"), edgecolor="black", linewidth=0.5)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.axhline(y=0.90, color="red", linestyle="--", linewidth=1.5, alpha=0.7,
               label="Recall target (0.90)")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=12, fontweight="bold")
    ax.set_ylabel("Score", fontsize=13, fontweight="bold")
    ax.set_title(title, fontsize=15, fontweight="bold", pad=15)
    ax.set_ylim(0.7, 1.05)
    ax.legend(loc="lower right", fontsize=11, framealpha=0.95)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, save_name), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Poster metrics chart saved to {FIGURES_DIR}/{save_name}")


def plot_cv_with_error_bars(cv_results: dict, save_name: str = "poster_cv_metrics.png"):
    """Grouped bar chart with error bars showing 5-fold CV mean +/- std."""
    metrics = ["accuracy", "precision", "recall", "f1", "auc_roc"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1", "AUC-ROC"]
    models = list(cv_results.keys())

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(metrics))
    width = 0.8 / len(models)

    for i, model in enumerate(models):
        means = [cv_results[model][f"{m}_mean"] for m in metrics]
        stds = [cv_results[model][f"{m}_std"] for m in metrics]
        offset = (i - (len(models) - 1) / 2) * width
        ax.bar(x + offset, means, width, yerr=stds, label=model,
               color=POSTER_COLORS.get(model, "#888"), edgecolor="black", linewidth=0.5,
               capsize=5, error_kw={"linewidth": 1.5, "ecolor": "black"})

    ax.axhline(y=0.90, color="red", linestyle="--", linewidth=1.5, alpha=0.7,
               label="Recall target (0.90)")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=12, fontweight="bold")
    ax.set_ylabel("Score", fontsize=13, fontweight="bold")
    ax.set_title("5-Fold Stratified Cross-Validation (mean ± std)",
                 fontsize=15, fontweight="bold", pad=15)
    ax.set_ylim(0.7, 1.05)
    ax.legend(loc="lower right", fontsize=11, framealpha=0.95)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, save_name), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Poster CV chart saved to {FIGURES_DIR}/{save_name}")


def plot_recall_headline(results: dict, save_name: str = "poster_recall_headline.png"):
    """Big, prominent recall comparison — the priority metric for the poster."""
    models = list(results.keys())
    recalls = [results[m]["recall"] for m in models]

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(models, recalls,
                   color=[POSTER_COLORS.get(m, "#888") for m in models],
                   edgecolor="black", linewidth=1.5, height=0.6)

    for bar, v in zip(bars, recalls):
        ax.text(v + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{v:.3f}", va="center", ha="left",
                fontsize=16, fontweight="bold")

    ax.axvline(x=0.90, color="red", linestyle="--", linewidth=2, alpha=0.7)
    ax.text(0.90, len(models) - 0.4, "Target: 0.90",
            color="red", fontsize=11, fontweight="bold", ha="center")

    ax.set_xlabel("Recall (Sensitivity)", fontsize=14, fontweight="bold")
    ax.set_title("Model Recall on Test Set\n(Priority Metric: Catching ASD-Positive Cases)",
                 fontsize=15, fontweight="bold", pad=15)
    ax.set_xlim(0.7, 1.0)
    ax.tick_params(axis="y", labelsize=13)
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, save_name), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Recall headline chart saved to {FIGURES_DIR}/{save_name}")


def plot_combined_curves(results: dict, save_name: str = "poster_curves_combined.png"):
    """ROC and PR curves side-by-side, poster-styled."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for name, m in results.items():
        color = POSTER_COLORS.get(name, "#888")
        y_true = m["y_true"]

        fpr, tpr, _ = roc_curve(y_true, m["y_prob"])
        roc_auc = auc(fpr, tpr)
        axes[0].plot(fpr, tpr, color=color, linewidth=2.5,
                     label=f"{name} (AUC={roc_auc:.3f})")

        prec, rec, _ = precision_recall_curve(y_true, m["y_prob"])
        pr_auc = auc(rec, prec)
        axes[1].plot(rec, prec, color=color, linewidth=2.5,
                     label=f"{name} (AUC={pr_auc:.3f})")

    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random")
    axes[0].set_xlabel("False Positive Rate", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("True Positive Rate", fontsize=12, fontweight="bold")
    axes[0].set_title("ROC Curves", fontsize=14, fontweight="bold")
    axes[0].legend(loc="lower right", fontsize=10)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Recall", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Precision", fontsize=12, fontweight="bold")
    axes[1].set_title("Precision-Recall Curves", fontsize=14, fontweight="bold")
    axes[1].legend(loc="lower left", fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, save_name), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Combined curves saved to {FIGURES_DIR}/{save_name}")


def plot_pipeline_diagram(save_name: str = "poster_pipeline.png"):
    """Visual pipeline overview for the poster."""
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.axis("off")

    stages = [
        ("Raw Data\n1054 toddlers\nQ-CHAT-10", "#FFE0B2"),
        ("Preprocess\nEncode + Drop\nleak features", "#FFCC80"),
        ("80/20 Split\nStratified", "#FFB74D"),
        ("15% Noise\nA1–A10\nbit-flips", "#FF7043"),
        ("Scale\n(train-only fit)", "#FF8A65"),
        ("Train\nMLP / LR / RF\n+ class weights", "#4FC3F7"),
        ("Evaluate\nTest + 5-fold CV\n+ SHAP", "#81C784"),
    ]

    n = len(stages)
    box_w = 1.0 / n * 0.85
    gap = 1.0 / n * 0.15
    y_center = 0.5
    box_h = 0.55

    for i, (text, color) in enumerate(stages):
        x = i * (box_w + gap) + gap / 2
        rect = plt.Rectangle((x, y_center - box_h / 2), box_w, box_h,
                             facecolor=color, edgecolor="black", linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + box_w / 2, y_center, text,
                ha="center", va="center", fontsize=10, fontweight="bold")

        if i < n - 1:
            arrow_x = x + box_w
            ax.annotate("", xy=(arrow_x + gap, y_center), xytext=(arrow_x, y_center),
                        arrowprops=dict(arrowstyle="->", lw=2, color="black"))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Pipeline Overview", fontsize=16, fontweight="bold", pad=15)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, save_name), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Pipeline diagram saved to {FIGURES_DIR}/{save_name}")


def make_poster_figures(results: dict, cv_results: dict):
    """Generate all poster-quality figures in one call."""
    print("\nGenerating poster figures...")
    plot_pipeline_diagram()
    plot_recall_headline(results)
    plot_metrics_grouped_bars(results)
    plot_cv_with_error_bars(cv_results)
    plot_combined_curves(results)


# ============================================================
# Cross-Validation
# ============================================================

def run_cross_validation(X: np.ndarray, y: np.ndarray, input_dim: int,
                         n_folds: int = 5, device: torch.device = torch.device("cpu")) -> dict:
    """
    5-fold stratified CV for the MLP classifier.
    Each fold fits its own scaler on fold-training data to prevent leakage.
    """
    from classifier import VanillaMLP
    from data import compute_class_weights
    from sklearn.preprocessing import StandardScaler

    kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    fold_metrics = {k: [] for k in ["accuracy", "precision", "recall", "f1", "auc_roc"]}

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
        X_tr, X_va = X[train_idx], X[val_idx]
        y_tr, y_va = y[train_idx], y[val_idx]

        # Per-fold scaling — fit on fold-train only
        fold_scaler = StandardScaler()
        X_tr = fold_scaler.fit_transform(X_tr)
        X_va = fold_scaler.transform(X_va)

        fold_weights = compute_class_weights(y_tr)

        torch.manual_seed(SEED + fold)
        np.random.seed(SEED + fold)
        model = VanillaMLP(input_dim).to(device)
        model = _train_fold_model(model, X_tr, y_tr, fold_weights, device, epochs=100, lr=1e-3)

        y_prob = get_predictions(model, X_va, device)
        m = compute_metrics(y_va, y_prob)
        for k in fold_metrics:
            fold_metrics[k].append(m[k])

        print(f"  Fold {fold+1}: Acc={m['accuracy']:.4f} Prec={m['precision']:.4f} "
              f"Rec={m['recall']:.4f} F1={m['f1']:.4f} AUC={m['auc_roc']:.4f}")

    summary = {}
    for k in fold_metrics:
        summary[f"{k}_mean"] = np.mean(fold_metrics[k])
        summary[f"{k}_std"] = np.std(fold_metrics[k])
    return summary


def run_sklearn_cross_validation(X: np.ndarray, y: np.ndarray,
                                  model_type: str = "lr", n_folds: int = 5) -> dict:
    """5-fold stratified CV for sklearn models. Per-fold scaling to prevent leakage."""
    from sklearn.preprocessing import StandardScaler

    kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    fold_metrics = {k: [] for k in ["accuracy", "precision", "recall", "f1", "auc_roc"]}

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
        X_tr, X_va = X[train_idx], X[val_idx]
        y_tr, y_va = y[train_idx], y[val_idx]

        # Per-fold scaling — fit on fold-train only
        fold_scaler = StandardScaler()
        X_tr = fold_scaler.fit_transform(X_tr)
        X_va = fold_scaler.transform(X_va)

        if model_type == "lr":
            model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=SEED)
        else:
            model = RandomForestClassifier(n_estimators=200, class_weight='balanced',
                                           random_state=SEED, n_jobs=-1)
        model.fit(X_tr, y_tr)
        y_prob = model.predict_proba(X_va)[:, 1]
        m = compute_metrics(y_va, y_prob)
        for k in fold_metrics:
            fold_metrics[k].append(m[k])

        print(f"  Fold {fold+1}: Acc={m['accuracy']:.4f} Prec={m['precision']:.4f} "
              f"Rec={m['recall']:.4f} F1={m['f1']:.4f} AUC={m['auc_roc']:.4f}")

    summary = {}
    for k in fold_metrics:
        summary[f"{k}_mean"] = np.mean(fold_metrics[k])
        summary[f"{k}_std"] = np.std(fold_metrics[k])
    return summary


def _train_fold_model(model, X_tr, y_tr, class_weights, device, epochs=50, lr=1e-3, batch_size=32):
    """Mini-batch training loop for CV folds."""
    sample_weights = torch.FloatTensor([class_weights[int(y)] for y in y_tr]).to(device)
    X_t = torch.FloatTensor(X_tr).to(device)
    y_t = torch.FloatTensor(y_tr).to(device)

    dataset = torch.utils.data.TensorDataset(X_t, y_t, sample_weights)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss(reduction='none')

    for _ in range(epochs):
        model.train()
        for batch_x, batch_y, batch_w in loader:
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = (criterion(preds, batch_y) * batch_w).mean()
            loss.backward()
            optimizer.step()
    return model


# ============================================================
# SHAP Explainability
# ============================================================

def plot_shap_values(model, X_test: np.ndarray, feature_names: list, device: torch.device):
    """
    SHAP analysis on the classifier.
    Shows which Q-CHAT-10 questions contribute most to ASD classification.
    """
    import shap

    model.eval()

    def predict_fn(x):
        with torch.no_grad():
            t = torch.FloatTensor(x).to(device)
            return model(t).cpu().numpy()

    background = X_test[np.random.choice(len(X_test), min(50, len(X_test)), replace=False)]
    explainer = shap.KernelExplainer(predict_fn, background)
    shap_values = explainer.shap_values(X_test, nsamples=100)

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, feature_names=feature_names,
                      plot_type="bar", show=False)
    plt.title("SHAP Feature Importance — Q-CHAT-10 ASD Classifier")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "shap_importance.png"), dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
    plt.title("SHAP Beeswarm — Feature Effects on ASD Prediction")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "shap_beeswarm.png"), dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  SHAP plots saved to {FIGURES_DIR}/shap_importance.png and shap_beeswarm.png")
