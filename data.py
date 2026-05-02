"""
data.py — Data loading, EDA, preprocessing, and splitting for Q-CHAT-10 toddler dataset.

The Q-CHAT-10 consists of 10 binary screening questions (A1–A10) plus demographic
features. The target is ASD trait classification (Yes/No).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import warnings
warnings.filterwarnings("ignore")

# Paths
DATA_PATH = "data/Toddler Autism dataset July 2018.csv"
FIGURES_DIR = "figures"
os.makedirs(FIGURES_DIR, exist_ok=True)


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load raw CSV and clean column names."""
    df = pd.read_csv(path)
    # Strip whitespace from column names (the target column has trailing space)
    df.columns = df.columns.str.strip()
    return df


def run_eda(df: pd.DataFrame) -> None:
    """Exploratory data analysis: class distribution, missing values, correlations."""
    print("=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    # --- Basic info ---
    print(f"\nShape: {df.shape}")
    print(f"\nColumn types:\n{df.dtypes}")

    # --- Missing values ---
    missing = df.isnull().sum()
    print(f"\nMissing values:\n{missing[missing > 0] if missing.sum() > 0 else 'None'}")

    # --- Class distribution ---
    target_col = "Class/ASD Traits"
    class_counts = df[target_col].value_counts()
    print(f"\nClass distribution:\n{class_counts}")
    print(f"Imbalance ratio: {class_counts.max() / class_counts.min():.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Bar plot of class distribution
    class_counts.plot(kind="bar", ax=axes[0], color=["#4CAF50", "#F44336"], edgecolor="black")
    axes[0].set_title("Class Distribution (ASD Traits)")
    axes[0].set_ylabel("Count")
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=0)

    # Correlation heatmap of Q-CHAT-10 questions (A1–A10)
    q_cols = [f"A{i}" for i in range(1, 11)]
    corr = df[q_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=axes[1],
                vmin=-1, vmax=1, square=True, cbar_kws={"shrink": 0.8})
    axes[1].set_title("Q-CHAT-10 Feature Correlations")

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "eda_overview.png"), dpi=150)
    plt.close()
    print(f"\nEDA plots saved to {FIGURES_DIR}/eda_overview.png")


def preprocess(df: pd.DataFrame) -> tuple:
    """
    Encode categoricals, return UNSCALED features + labels.
    Scaling is deferred to after train/test split to prevent data leakage
    (scaler must be fit on training data only).

    Encoding strategy:
    - Binary categoricals (Sex, Jaundice, Family_mem_with_ASD): label encode
    - Multi-class categoricals (Ethnicity, Who completed the test): one-hot encode
    - Target: binary encode (Yes=1, No=0)
    - Drop Case_No (ID) and Qchat-10-Score (leaks the answer — it's the sum of A1–A10)
    """
    df = df.copy()
    target_col = "Class/ASD Traits"

    # Encode target: Yes -> 1 (ASD traits present), No -> 0
    df[target_col] = df[target_col].map({"Yes": 1, "No": 0})

    # Drop ID column and Q-CHAT score (direct leak — it's literally sum(A1..A10))
    df = df.drop(columns=["Case_No", "Qchat-10-Score"])

    # Binary categorical encoding
    binary_cols = {"Sex": {"m": 0, "f": 1},
                   "Jaundice": {"no": 0, "yes": 1},
                   "Family_mem_with_ASD": {"no": 0, "yes": 1}}
    for col, mapping in binary_cols.items():
        df[col] = df[col].map(mapping)

    # One-hot encode multi-class categoricals
    multi_cat_cols = ["Ethnicity", "Who completed the test"]
    df = pd.get_dummies(df, columns=multi_cat_cols, drop_first=True, dtype=int)

    # Separate features and target
    y = df[target_col].values
    X = df.drop(columns=[target_col])
    feature_names = X.columns.tolist()

    return X.values, y, feature_names


def add_noise(X: np.ndarray, feature_names: list,
              flip_prob: float = 0.15, seed: int = 42) -> np.ndarray:
    """
    Inject realistic noise into the screening answers (A1-A10).
    Each binary answer is flipped with probability `flip_prob` independently.

    Why: Q-CHAT-10 has a deterministic scoring rule (score >= 3 => ASD),
    so models trivially recover the rule and hit ~100% accuracy. Bit-flipping
    simulates real-world questionnaire noise: parents misremembering, ambiguous
    interpretation, or borderline behaviors. This makes the task non-trivial.

    Only A1-A10 are noised — demographics (age, sex, ethnicity) are left clean
    since they're factual and unlikely to be misreported.
    """
    rng = np.random.RandomState(seed)
    X_noisy = X.copy().astype(float)

    q_indices = [i for i, name in enumerate(feature_names) if name.startswith("A") and name[1:].isdigit()]

    for idx in q_indices:
        flip_mask = rng.rand(X_noisy.shape[0]) < flip_prob
        X_noisy[flip_mask, idx] = 1 - X_noisy[flip_mask, idx]

    n_flips = int((X_noisy[:, q_indices] != X[:, q_indices]).sum())
    total = len(q_indices) * X.shape[0]
    print(f"  Noise injected: {n_flips}/{total} bits flipped ({100*n_flips/total:.1f}%) "
          f"across A1-A10 (target rate: {flip_prob*100:.0f}%)")
    return X_noisy


def scale_data(X_train: np.ndarray, X_test: np.ndarray) -> tuple:
    """
    Fit scaler on training data ONLY, then transform both sets.
    This prevents data leakage — test set statistics never influence scaling.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def split_data(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, seed: int = 42):
    """
    Stratified 80/20 train/test split.
    Stratification preserves class proportions in both sets — critical
    when the dataset is imbalanced so the test set is representative.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    print(f"\nTrain: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")
    print(f"Train class dist: {np.bincount(y_train)} | Test class dist: {np.bincount(y_test)}")
    return X_train, X_test, y_train, y_test


def compute_class_weights(y: np.ndarray) -> dict:
    """
    Compute inverse-frequency class weights.
    We use class weighting instead of SMOTE to avoid synthetic samples
    contaminating validation folds — SMOTE before CV leaks information.
    """
    counts = np.bincount(y)
    total = len(y)
    weights = {i: total / (len(counts) * c) for i, c in enumerate(counts)}
    print(f"Class weights: {weights}")
    return weights


if __name__ == "__main__":
    df = load_data()
    run_eda(df)
    X, y, feature_names = preprocess(df)
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Features: {feature_names}")
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train, X_test, scaler = scale_data(X_train, X_test)
    weights = compute_class_weights(y_train)
