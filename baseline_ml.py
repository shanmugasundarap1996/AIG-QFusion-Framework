#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Classical ML baselines for genomic variant classification.

Runs multiple classical baselines on the balanced ClinVar/dbNSFP dataset
using a selected PCA setting (default: PCA-12), evaluates across seeds,
and exports raw, mean, std, mean±std, and styled summary tables.

Example:
python baseline_ml.py \
  --data-root ./data/dataset_saved_500_300_pca4to12 \
  --dataset balanced \
  --pca 12 \
  --out-dir ./outputs/classical_baselines
"""

from __future__ import annotations

import argparse
import os
import time
import warnings
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

warnings.filterwarnings("ignore")

HAS_XGBOOST = True
HAS_CATBOOST = True
try:
    from xgboost import XGBClassifier
except Exception:
    HAS_XGBOOST = False

try:
    from catboost import CatBoostClassifier
except Exception:
    HAS_CATBOOST = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, required=True, help="Root dataset directory")
    parser.add_argument("--dataset", type=str, default="balanced", help="Dataset subfolder name")
    parser.add_argument("--label-col", type=str, default="CLNSIG", help="Label column")
    parser.add_argument("--pca", type=int, default=12, help="PCA dimension to evaluate")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4], help="Random seeds")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold")
    parser.add_argument("--out-dir", type=str, required=True, help="Output directory")
    return parser.parse_args()


def load_train_test(data_root: str, dataset: str, pca_k: int, label_col: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pdir = Path(data_root) / dataset / f"pca_{pca_k}"
    tr_path = pdir / "train.csv"
    te_path = pdir / "test.csv"

    if not tr_path.exists() or not te_path.exists():
        raise FileNotFoundError(f"Missing train/test CSVs under: {pdir}")

    tr = pd.read_csv(tr_path)
    te = pd.read_csv(te_path)

    if label_col not in tr.columns or label_col not in te.columns:
        raise ValueError(f"Label column '{label_col}' not found in train/test CSVs")

    Xtr = tr.drop(columns=[label_col]).values.astype(float)
    ytr = tr[label_col].values.astype(int)
    Xte = te.drop(columns=[label_col]).values.astype(float)
    yte = te[label_col].values.astype(int)
    return Xtr, ytr, Xte, yte


def ece_score(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10, threshold: float = 0.5) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i < n_bins - 1:
            mask = (y_prob >= lo) & (y_prob < hi)
        else:
            mask = (y_prob >= lo) & (y_prob <= hi)

        if np.any(mask):
            acc = np.mean(y_pred[mask] == y_true[mask])
            conf = np.mean(y_prob[mask])
            ece += (np.sum(mask) / len(y_true)) * abs(acc - conf)
    return float(ece)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp + 1e-12)
    sensitivity = recall_score(y_true, y_pred, zero_division=0)
    bal_acc = 0.5 * (specificity + sensitivity)

    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "MCC": float(matthews_corrcoef(y_true, y_pred)),
        "Specificity": float(specificity),
        "BalancedAcc": float(bal_acc),
        "ROC_AUC": float(roc_auc_score(y_true, y_prob)),
        "PR_AUC": float(average_precision_score(y_true, y_prob)),
        "Brier": float(brier_score_loss(y_true, y_prob)),
        "ECE": float(ece_score(y_true, y_prob, threshold=threshold)),
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
    }


def make_models(seed: int) -> Dict[str, object]:
    models: Dict[str, object] = {}

    models["DecisionTree"] = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", DecisionTreeClassifier(random_state=seed))
    ])

    models["RandomForest"] = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1))
    ])

    models["GradientBoost"] = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=seed))
    ])

    models["KNN"] = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier(n_neighbors=7, weights="distance"))
    ])

    models["SVM"] = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", SVC(C=2.0, kernel="rbf", probability=True, random_state=seed))
    ])

    models["MLP"] = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            learning_rate_init=1e-3,
            max_iter=1000,
            early_stopping=True,
            random_state=seed,
        ))
    ])

    if HAS_XGBOOST:
        models["XGBoost"] = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", XGBClassifier(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=seed,
                n_jobs=-1,
            ))
        ])

    if HAS_CATBOOST:
        models["CatBoost"] = CatBoostClassifier(
            iterations=300,
            depth=6,
            learning_rate=0.05,
            loss_function="Logloss",
            eval_metric="AUC",
            verbose=False,
            random_seed=seed,
        )

    return models


def fit_predict_prob_with_time(model: object, Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray, algo_name: str) -> Tuple[np.ndarray, float, float]:
    if algo_name == "CatBoost":
        imputer = SimpleImputer(strategy="median")
        t0 = time.time()
        Xtr_imp = imputer.fit_transform(Xtr)
        Xte_imp = imputer.transform(Xte)
        model.fit(Xtr_imp, ytr)
        t1 = time.time()

        t2 = time.time()
        y_prob = model.predict_proba(Xte_imp)[:, 1]
        t3 = time.time()
        return y_prob, float(t1 - t0), float(t3 - t2)

    t0 = time.time()
    model.fit(Xtr, ytr)
    t1 = time.time()

    t2 = time.time()
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(Xte)[:, 1]
    else:
        score = model.decision_function(Xte)
        y_prob = 1.0 / (1.0 + np.exp(-score))
    t3 = time.time()

    return y_prob, float(t1 - t0), float(t3 - t2)


def save_summary_tables(raw_df: pd.DataFrame, out_dir: Path) -> None:
    group_cols = ["Dataset", "PCA", "Algorithm"]
    ignore_cols = group_cols + ["Seed", "TrainN", "TestN", "TP", "TN", "FP", "FN"]
    metric_cols = [c for c in raw_df.columns if c not in ignore_cols]

    mean_df = raw_df.groupby(group_cols)[metric_cols].mean().reset_index()
    std_df = raw_df.groupby(group_cols)[metric_cols].std().reset_index()

    mean_df.to_csv(out_dir / "mean_ml_baselines_balanced_pca12.csv", index=False)
    std_df.to_csv(out_dir / "std_ml_baselines_balanced_pca12.csv", index=False)

    pm_df = mean_df.copy()
    for c in metric_cols:
        pm_df[c] = mean_df[c].map(lambda x: f"{x:.4f}") + " ± " + std_df[c].map(lambda x: f"{x:.4f}")
    pm_df.to_csv(out_dir / "meanpmstd_ml_baselines_balanced_pca12.csv", index=False)

    cols_show = [
        "Dataset", "PCA", "Algorithm",
        "Accuracy", "Precision", "Recall", "F1", "MCC",
        "ROC_AUC", "PR_AUC", "Brier", "ECE",
        "TrainTime_s", "TestTime_s", "TotalTime_s",
    ]
    disp_pm_df = pm_df[cols_show].copy()
    disp_mean_df = mean_df[cols_show].copy()

    higher_better = ["Accuracy", "Precision", "Recall", "F1", "MCC", "ROC_AUC", "PR_AUC"]
    lower_better = ["Brier", "ECE", "TrainTime_s", "TestTime_s", "TotalTime_s"]
    best_color = "background-color: #d9ead3;"
    border = "1px solid black"

    def highlight_best_cells_within_pca(_):
        styles = pd.DataFrame("", index=disp_pm_df.index, columns=disp_pm_df.columns)
        for pca_val in disp_mean_df["PCA"].unique():
            mask = disp_mean_df["PCA"] == pca_val
            sub = disp_mean_df.loc[mask]
            for c in higher_better:
                best = sub[c].max()
                styles.loc[mask & np.isclose(disp_mean_df[c], best), c] = best_color
            for c in lower_better:
                best = sub[c].min()
                styles.loc[mask & np.isclose(disp_mean_df[c], best), c] = best_color
        return styles

    sty = (
        disp_pm_df.style
        .apply(highlight_best_cells_within_pca, axis=None)
        .set_properties(**{
            "text-align": "center",
            "vertical-align": "middle",
            "border": border,
            "font-size": "11pt",
            "font-family": "Times New Roman",
        })
        .set_table_styles([
            {"selector": "table", "props": [("border-collapse", "collapse"), ("border", border), ("width", "100%")]},
            {"selector": "th", "props": [("border", border), ("text-align", "center"), ("vertical-align", "middle"), ("background-color", "#f2f2f2"), ("font-weight", "bold"), ("padding", "6px"), ("font-family", "Times New Roman"), ("font-size", "11pt")]},
            {"selector": "td", "props": [("border", border), ("text-align", "center"), ("vertical-align", "middle"), ("padding", "6px"), ("font-family", "Times New Roman"), ("font-size", "11pt")]},
        ])
    )

    try:
        sty.to_excel(out_dir / "highlighted_bestcells_meanpmstd_ml_baselines_balanced_pca12.xlsx", engine="openpyxl", index=False)
    except Exception as exc:
        print(f"Excel export skipped: {exc}")

    try:
        with open(out_dir / "highlighted_bestcells_meanpmstd_ml_baselines_balanced_pca12.html", "w", encoding="utf-8") as f:
            f.write(sty.to_html())
    except Exception as exc:
        print(f"HTML export skipped: {exc}")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not HAS_XGBOOST:
        print("XGBoost not installed -> skipped")
    if not HAS_CATBOOST:
        print("CatBoost not installed -> skipped")

    Xtr, ytr, Xte, yte = load_train_test(args.data_root, args.dataset, args.pca, args.label_col)

    raw_rows = []
    for seed in args.seeds:
        print("\n" + "=" * 100)
        print(f"RUNNING CLASSICAL BASELINES | {args.dataset.upper()} | PCA-{args.pca} | seed={seed}")
        print("=" * 100)

        models = make_models(seed)
        for algo_name, model in models.items():
            y_prob, train_t, test_t = fit_predict_prob_with_time(model, Xtr, ytr, Xte, algo_name)
            metrics = compute_metrics(yte, y_prob, threshold=args.threshold)
            metrics.update({
                "Dataset": args.dataset,
                "PCA": args.pca,
                "Seed": seed,
                "Algorithm": algo_name,
                "TrainN": int(len(Xtr)),
                "TestN": int(len(Xte)),
                "TrainTime_s": train_t,
                "TestTime_s": test_t,
                "TotalTime_s": train_t + test_t,
            })
            raw_rows.append(metrics)

            print(
                f"{algo_name:>13} | seed={seed} | "
                f"Acc={metrics['Accuracy']:.4f} | F1={metrics['F1']:.4f} | "
                f"MCC={metrics['MCC']:.4f} | ROC_AUC={metrics['ROC_AUC']:.4f} | "
                f"Train={train_t:.4f}s | Test={test_t:.4f}s | Total={train_t + test_t:.4f}s"
            )

    raw_df = pd.DataFrame(raw_rows)
    raw_path = out_dir / "raw_all_seeds_ml_baselines_balanced_pca12.csv"
    raw_df.to_csv(raw_path, index=False)
    save_summary_tables(raw_df, out_dir)

    print("\nSaved raw results:", raw_path)
    print("Done.")


if __name__ == "__main__":
    main()
