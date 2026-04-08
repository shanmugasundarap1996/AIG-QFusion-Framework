#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AIG-QFusion proposed system across PCA-4 to PCA-12.

This is a cleaned GitHub-ready version of the proposed method:
- balanced dataset
- PCA 4..12 evaluation
- adaptive interaction graph
- Z / ZZ / interaction-aware quantum branches
- adaptive kernel fusion
- logistic regression meta-learner
- threshold optimization

Example:
python proposed_pca4to12.py \
  --data-root ./data/dataset_saved_500_300_pca4to12 \
  --dataset balanced \
  --out-dir ./outputs/aigqfusion_pca4to12
"""

from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
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

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector

warnings.filterwarnings("ignore")

_SV_CACHE: Dict[Tuple, np.ndarray] = {}


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


def scale_to_pi(Xtr: np.ndarray, Xte: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    scaler = MinMaxScaler(feature_range=(0.0, np.pi))
    return scaler.fit_transform(Xtr), scaler.transform(Xte)


def build_interaction_graph(Xtr: np.ndarray, top_k: int = 8, use_skip_edges: bool = True) -> List[Tuple[int, int, float]]:
    d = Xtr.shape[1]
    corr = np.corrcoef(Xtr, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)
    vars_ = np.var(Xtr, axis=0)

    pair_scores: List[Tuple[float, int, int]] = []
    for i in range(d):
        for j in range(i + 1, d):
            if (not use_skip_edges) and (j != i + 1):
                continue
            score = abs(corr[i, j]) * (vars_[i] + vars_[j] + 1e-12)
            pair_scores.append((float(score), i, j))

    pair_scores.sort(reverse=True, key=lambda t: t[0])
    selected = pair_scores[:top_k]
    return [(i, j, score) for score, i, j in selected]


def feature_map_Z(d: int, reps: int) -> Tuple[QuantumCircuit, Sequence]:
    x = ParameterVector("x", d)
    qc = QuantumCircuit(d, name="ZMap")
    for _ in range(reps):
        for i in range(d):
            qc.h(i)
            qc.rz(x[i], i)
        for i in range(d - 1):
            qc.cx(i, i + 1)
    return qc, list(x)


def feature_map_ZZ(d: int, reps: int) -> Tuple[QuantumCircuit, Sequence]:
    x = ParameterVector("x", d)
    qc = QuantumCircuit(d, name="ZZMap")
    for _ in range(reps):
        for i in range(d):
            qc.h(i)
            qc.rz(x[i], i)
        for i in range(d - 1):
            qc.rzz(x[i] * x[i + 1], i, i + 1)
    return qc, list(x)


def feature_map_INT(d: int, interaction_edges: Sequence[Tuple[int, int, float]], reps: int, interaction_scale: float) -> Tuple[QuantumCircuit, Sequence]:
    x = ParameterVector("x", d)
    qc = QuantumCircuit(d, name="INTMap")
    max_edge_score = max([s for _, _, s in interaction_edges], default=1.0)
    max_edge_score = max(max_edge_score, 1e-12)

    for _ in range(reps):
        for i in range(d):
            qc.h(i)
            qc.rx(0.50 * x[i], i)
            qc.rz(x[i], i)
            qc.ry(0.25 * x[i], i)

        for i in range(d - 1):
            qc.cx(i, i + 1)

        for i, j, s in interaction_edges:
            alpha = interaction_scale * (s / max_edge_score)
            qc.rzz(alpha * x[i] * x[j], i, j)
            qc.rx(0.10 * alpha * (x[i] + x[j]), i)
            qc.ry(0.10 * alpha * (x[i] - x[j]), j)

    return qc, list(x)


def _hash_X(X: np.ndarray) -> int:
    b = X.tobytes()
    head = b[:4096]
    tail = b[-4096:] if len(b) > 4096 else b
    return hash((X.shape, X.dtype.str, head, tail))


def statevectors(X_: np.ndarray, map_name: str, reps: int, interaction_edges: Sequence[Tuple[int, int, float]] | None = None, interaction_scale: float = 1.25) -> np.ndarray:
    key = (map_name, reps, tuple(interaction_edges) if interaction_edges is not None else None, _hash_X(X_))
    if key in _SV_CACHE:
        return _SV_CACHE[key]

    d = X_.shape[1]
    if map_name == "Z":
        qc, params = feature_map_Z(d, reps)
    elif map_name == "ZZ":
        qc, params = feature_map_ZZ(d, reps)
    elif map_name == "INT":
        qc, params = feature_map_INT(d, interaction_edges or [], reps, interaction_scale)
    else:
        raise ValueError(f"Unknown map_name: {map_name}")

    svs = []
    for row in X_:
        bind = {p: float(v) for p, v in zip(params, row)}
        sv = Statevector.from_instruction(qc.assign_parameters(bind, inplace=False))
        svs.append(np.asarray(sv.data))
    S = np.vstack(svs)
    _SV_CACHE[key] = S
    return S


def kernel_train_test(XA: np.ndarray, XB: np.ndarray, map_name: str, reps: int, interaction_edges: Sequence[Tuple[int, int, float]] | None = None, interaction_scale: float = 1.25) -> Tuple[np.ndarray, np.ndarray]:
    SA = statevectors(XA, map_name, reps, interaction_edges, interaction_scale)
    SB = statevectors(XB, map_name, reps, interaction_edges, interaction_scale)
    K_AA = np.abs(SA @ np.conjugate(SA).T) ** 2
    K_BA = np.abs(SB @ np.conjugate(SA).T) ** 2
    return K_AA, K_BA


def normalize_positive_weights(w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=float)
    w = np.maximum(w, 1e-10)
    return w / np.sum(w)


def metric_from_probs(y_true: np.ndarray, y_prob: np.ndarray, metric_name: str = "MCC", threshold: float = 0.5) -> float:
    y_pred = (y_prob >= threshold).astype(int)
    if metric_name.upper() == "MCC":
        return float(matthews_corrcoef(y_true, y_pred))
    if metric_name.upper() == "F1":
        return float(f1_score(y_true, y_pred, zero_division=0))
    raise ValueError("metric_name must be 'MCC' or 'F1'")


def optimize_threshold(y_true: np.ndarray, y_prob: np.ndarray, metric_name: str, grid: np.ndarray) -> Tuple[float, float]:
    best_t = 0.5
    best_score = -1e18
    for t in grid:
        score = metric_from_probs(y_true, y_prob, metric_name=metric_name, threshold=float(t))
        if score > best_score:
            best_score = score
            best_t = float(t)
    return best_t, float(best_score)


def random_simplex_weights(n_models: int, n_samples: int, seed: int) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    weights: List[np.ndarray] = [np.ones(n_models) / n_models]
    for i in range(n_models):
        w = np.zeros(n_models)
        w[i] = 1.0
        weights.append(w)
    for _ in range(n_samples):
        weights.append(rng.dirichlet(alpha=np.ones(n_models)))
    return [normalize_positive_weights(w) for w in weights]


def optimize_kernel_weights_from_oof(ytr: np.ndarray, oof_probs_by_branch: np.ndarray, metric_name: str, n_weight_samples: int, seed: int, thresh_grid: np.ndarray) -> Tuple[np.ndarray, float, float]:
    candidates = random_simplex_weights(oof_probs_by_branch.shape[1], n_weight_samples, seed)
    best_w = None
    best_metric = -1e18
    best_threshold = 0.5

    for w in candidates:
        p = oof_probs_by_branch @ w
        t, _ = optimize_threshold(ytr, p, metric_name=metric_name, grid=thresh_grid)
        score = metric_from_probs(ytr, p, metric_name=metric_name, threshold=t)
        if score > best_metric:
            best_metric = score
            best_w = w.copy()
            best_threshold = t

    assert best_w is not None
    return best_w, float(best_metric), float(best_threshold)


def train_branch(K_tr: np.ndarray, y_tr: np.ndarray, K_te: np.ndarray, seed: int, c_svc: float) -> Tuple[np.ndarray, np.ndarray]:
    clf = SVC(kernel="precomputed", C=c_svc, probability=True, random_state=seed)
    clf.fit(K_tr, y_tr)
    prob = clf.predict_proba(K_te)[:, 1]
    dec = clf.decision_function(K_te)
    return prob, dec


def aig_qfusion_predict_seed(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xte: np.ndarray,
    seed: int,
    interaction_edges: Sequence[Tuple[int, int, float]],
    n_splits: int,
    reps_z: int,
    reps_zz: int,
    reps_int: int,
    c_svc: float,
    meta_maxit: int,
    n_weight_samples: int,
    metric_for_weight_search: str,
    optimize_threshold_for: str,
    thresh_grid: np.ndarray,
    interaction_scale: float,
) -> Tuple[np.ndarray, Dict[str, float]]:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_indices = list(skf.split(Xtr, ytr))

    base_names = ["Z", "ZZ", "INT"]
    reps_by_name = {"Z": reps_z, "ZZ": reps_zz, "INT": reps_int}
    M = len(base_names)

    oof_prob = np.zeros((len(Xtr), M), dtype=float)
    oof_dec = np.zeros((len(Xtr), M), dtype=float)

    for idx_tr, idx_va in fold_indices:
        X_fold_tr, y_fold_tr = Xtr[idx_tr], ytr[idx_tr]
        X_fold_va = Xtr[idx_va]

        for m, name in enumerate(base_names):
            reps = reps_by_name[name]
            K_tr, K_va = kernel_train_test(
                X_fold_tr,
                X_fold_va,
                map_name=name,
                reps=reps,
                interaction_edges=interaction_edges if name == "INT" else None,
                interaction_scale=interaction_scale,
            )
            p, d = train_branch(K_tr, y_fold_tr, K_va, seed, c_svc)
            oof_prob[idx_va, m] = p
            oof_dec[idx_va, m] = d

    kernel_weights, kernel_weight_score, kernel_weight_thresh = optimize_kernel_weights_from_oof(
        ytr=ytr,
        oof_probs_by_branch=oof_prob,
        metric_name=metric_for_weight_search,
        n_weight_samples=n_weight_samples,
        seed=seed,
        thresh_grid=thresh_grid,
    )

    oof_fused_prob = np.zeros((len(Xtr), 1), dtype=float)
    oof_fused_dec = np.zeros((len(Xtr), 1), dtype=float)

    for idx_tr, idx_va in fold_indices:
        X_fold_tr, y_fold_tr = Xtr[idx_tr], ytr[idx_tr]
        X_fold_va = Xtr[idx_va]

        K_tr_fused = None
        K_va_fused = None

        for w, name in zip(kernel_weights, base_names):
            reps = reps_by_name[name]
            K_tr, K_va = kernel_train_test(
                X_fold_tr,
                X_fold_va,
                map_name=name,
                reps=reps,
                interaction_edges=interaction_edges if name == "INT" else None,
                interaction_scale=interaction_scale,
            )
            K_tr_fused = w * K_tr if K_tr_fused is None else K_tr_fused + w * K_tr
            K_va_fused = w * K_va if K_va_fused is None else K_va_fused + w * K_va

        p, d = train_branch(K_tr_fused, y_fold_tr, K_va_fused, seed, c_svc)
        oof_fused_prob[idx_va, 0] = p
        oof_fused_dec[idx_va, 0] = d

    Z_train_meta = np.hstack([oof_prob, oof_dec, oof_fused_prob, oof_fused_dec])
    meta = LogisticRegression(max_iter=meta_maxit, random_state=seed)
    meta.fit(Z_train_meta, ytr)
    oof_meta_prob = meta.predict_proba(Z_train_meta)[:, 1]

    best_meta_threshold, best_meta_score = optimize_threshold(
        ytr, oof_meta_prob, metric_name=optimize_threshold_for, grid=thresh_grid
    )

    Z_test_prob = np.zeros((len(Xte), M), dtype=float)
    Z_test_dec = np.zeros((len(Xte), M), dtype=float)

    for m, name in enumerate(base_names):
        reps = reps_by_name[name]
        K_tr_full, K_te = kernel_train_test(
            Xtr, Xte, map_name=name, reps=reps,
            interaction_edges=interaction_edges if name == "INT" else None,
            interaction_scale=interaction_scale,
        )
        p, d = train_branch(K_tr_full, ytr, K_te, seed, c_svc)
        Z_test_prob[:, m] = p
        Z_test_dec[:, m] = d

    K_tr_full_fused = None
    K_te_fused = None
    for w, name in zip(kernel_weights, base_names):
        reps = reps_by_name[name]
        K_tr_full, K_te = kernel_train_test(
            Xtr, Xte, map_name=name, reps=reps,
            interaction_edges=interaction_edges if name == "INT" else None,
            interaction_scale=interaction_scale,
        )
        K_tr_full_fused = w * K_tr_full if K_tr_full_fused is None else K_tr_full_fused + w * K_tr_full
        K_te_fused = w * K_te if K_te_fused is None else K_te_fused + w * K_te

    fused_p, fused_d = train_branch(K_tr_full_fused, ytr, K_te_fused, seed, c_svc)
    Z_test_meta = np.hstack([Z_test_prob, Z_test_dec, fused_p.reshape(-1, 1), fused_d.reshape(-1, 1)])
    y_prob = meta.predict_proba(Z_test_meta)[:, 1]

    details = {
        "kernel_weights": kernel_weights,
        "kernel_weight_score": float(kernel_weight_score),
        "kernel_weight_thresh": float(kernel_weight_thresh),
        "meta_threshold": float(best_meta_threshold),
        "meta_threshold_score": float(best_meta_score),
    }
    return y_prob, details


def highlight_and_export(mean_df: pd.DataFrame, std_df: pd.DataFrame, out_dir: Path, prefix: str) -> None:
    cols_show = [
        "Dataset", "PCA",
        "Accuracy", "Precision", "Recall", "F1", "MCC",
        "ROC_AUC", "PR_AUC", "Brier", "ECE",
        "TotalTime_s", "Best_reps", "Best_Csvc", "Best_Cmeta",
    ]

    pm_df = mean_df.copy()
    metric_cols = [c for c in mean_df.columns if c not in ["Dataset", "PCA"]]
    for c in metric_cols:
        pm_df[c] = mean_df[c].map(lambda x: f"{x:.4f}") + " ± " + std_df[c].map(lambda x: f"{x:.4f}")

    pm_df.to_csv(out_dir / f"meanpmstd_{prefix}.csv", index=False)

    disp_pm_df = pm_df[cols_show].copy()
    disp_mean_df = mean_df[cols_show].copy()

    higher_better = ["Accuracy", "Precision", "Recall", "F1", "MCC", "ROC_AUC", "PR_AUC"]
    lower_better = ["Brier", "ECE", "TotalTime_s"]
    hilite = "background-color: #d9ead3"

    def highlight_best_cells(_):
        styles = pd.DataFrame("", index=disp_pm_df.index, columns=disp_pm_df.columns)
        for c in higher_better:
            best = disp_mean_df[c].max()
            styles.loc[np.isclose(disp_mean_df[c], best), c] = hilite
        for c in lower_better:
            best = disp_mean_df[c].min()
            styles.loc[np.isclose(disp_mean_df[c], best), c] = hilite
        return styles

    sty = (
        disp_pm_df.style
        .apply(highlight_best_cells, axis=None)
        .set_properties(**{
            "text-align": "center",
            "vertical-align": "middle",
            "border": "1px solid black",
            "font-size": "11pt",
            "font-family": "Times New Roman",
        })
        .set_table_styles([
            {"selector": "table", "props": [("border-collapse", "collapse"), ("border", "1px solid black"), ("width", "100%")]},
            {"selector": "th", "props": [("border", "1px solid black"), ("text-align", "center"), ("vertical-align", "middle"), ("background-color", "#f2f2f2"), ("font-weight", "bold"), ("padding", "6px"), ("font-family", "Times New Roman"), ("font-size", "11pt")]},
            {"selector": "td", "props": [("border", "1px solid black"), ("text-align", "center"), ("vertical-align", "middle"), ("padding", "6px"), ("font-family", "Times New Roman"), ("font-size", "11pt")]},
        ])
    )

    try:
        sty.to_excel(out_dir / f"highlighted_bestcells_meanpmstd_{prefix}.xlsx", engine="openpyxl", index=False)
    except Exception as exc:
        print(f"Excel export skipped: {exc}")
    try:
        with open(out_dir / f"highlighted_bestcells_meanpmstd_{prefix}.html", "w", encoding="utf-8") as f:
            f.write(sty.to_html())
    except Exception as exc:
        print(f"HTML export skipped: {exc}")



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="balanced")
    parser.add_argument("--label-col", type=str, default="CLNSIG")
    parser.add_argument("--pca-list", type=int, nargs="+", default=list(range(4, 13)))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--reps-z", type=int, default=2)
    parser.add_argument("--reps-zz", type=int, default=2)
    parser.add_argument("--reps-int", type=int, default=3)
    parser.add_argument("--c-svc", type=float, default=2.0)
    parser.add_argument("--meta-maxit", type=int, default=4000)
    parser.add_argument("--top-k-interactions", type=int, default=8)
    parser.add_argument("--use-skip-edges", action="store_true")
    parser.add_argument("--interaction-scale", type=float, default=1.25)
    parser.add_argument("--n-weight-samples", type=int, default=160)
    parser.add_argument("--metric-for-weight-search", type=str, default="MCC")
    parser.add_argument("--optimize-threshold-for", type=str, default="MCC")
    parser.add_argument("--thresh-start", type=float, default=0.30)
    parser.add_argument("--thresh-end", type=float, default=0.70)
    parser.add_argument("--thresh-steps", type=int, default=81)
    parser.add_argument("--out-dir", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    thresh_grid = np.linspace(args.thresh_start, args.thresh_end, args.thresh_steps)
    raw_rows = []

    for k in args.pca_list:
        _SV_CACHE.clear()
        print("\n" + "=" * 100)
        print(f"RUNNING AIG-QFUSION | {args.dataset.upper()} | PCA-{k}")
        print("=" * 100)

        Xtr_raw, ytr, Xte_raw, yte = load_train_test(args.data_root, args.dataset, k, args.label_col)
        Xtr, Xte = scale_to_pi(Xtr_raw, Xte_raw)

        interaction_edges = build_interaction_graph(
            Xtr,
            top_k=args.top_k_interactions,
            use_skip_edges=args.use_skip_edges,
        )

        for seed in args.seeds:
            t0 = time.time()
            y_prob, details = aig_qfusion_predict_seed(
                Xtr=Xtr,
                ytr=ytr,
                Xte=Xte,
                seed=seed,
                interaction_edges=interaction_edges,
                n_splits=args.n_splits,
                reps_z=args.reps_z,
                reps_zz=args.reps_zz,
                reps_int=args.reps_int,
                c_svc=args.c_svc,
                meta_maxit=args.meta_maxit,
                n_weight_samples=args.n_weight_samples,
                metric_for_weight_search=args.metric_for_weight_search,
                optimize_threshold_for=args.optimize_threshold_for,
                thresh_grid=thresh_grid,
                interaction_scale=args.interaction_scale,
            )
            t1 = time.time()

            threshold = details["meta_threshold"]
            m = compute_metrics(yte, y_prob, threshold=threshold)
            m.update({
                "Dataset": args.dataset,
                "PCA": k,
                "Seed": seed,
                "TrainN": int(len(Xtr)),
                "TestN": int(len(Xte)),
                "MetaThreshold": float(threshold),
                "W_Z": float(details["kernel_weights"][0]),
                "W_ZZ": float(details["kernel_weights"][1]),
                "W_INT": float(details["kernel_weights"][2]),
                "KernelWeightScore": float(details["kernel_weight_score"]),
                "KernelWeightThresh": float(details["kernel_weight_thresh"]),
                "Best_reps": float(args.reps_int),
                "Best_Csvc": float(args.c_svc),
                "Best_Cmeta": float(args.meta_maxit),
                "TotalTime_s": float(t1 - t0),
            })
            raw_rows.append(m)

            print(
                f"seed={seed} | Acc={m['Accuracy']:.4f} | F1={m['F1']:.4f} | "
                f"MCC={m['MCC']:.4f} | ROC_AUC={m['ROC_AUC']:.4f} | Time={m['TotalTime_s']:.2f}s"
            )

    raw_df = pd.DataFrame(raw_rows)
    raw_df.to_csv(out_dir / "raw_all_seeds_aigqfusion_balanced_pca4to12.csv", index=False)

    group_cols = ["Dataset", "PCA"]
    ignore_cols = group_cols + ["Seed", "TrainN", "TestN", "TP", "TN", "FP", "FN"]
    metric_cols = [c for c in raw_df.columns if c not in ignore_cols]
    mean_df = raw_df.groupby(group_cols)[metric_cols].mean().reset_index()
    std_df = raw_df.groupby(group_cols)[metric_cols].std().reset_index()

    mean_df.to_csv(out_dir / "mean_aigqfusion_balanced_pca4to12.csv", index=False)
    std_df.to_csv(out_dir / "std_aigqfusion_balanced_pca4to12.csv", index=False)
    highlight_and_export(mean_df, std_df, out_dir, "aigqfusion_balanced_pca4to12")

    print("\nSaved outputs in:", out_dir)
    print("Done.")


if __name__ == "__main__":
    main()
