#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Quantum ML baselines for genomic variant classification.

Supports QSVM with Z, ZZ, and Pauli feature maps by default.
Optional VQC and QNN execution can be enabled if qiskit-machine-learning
is installed.

Example:
python baseline_qml.py \
  --data-root ./data/dataset_saved_500_300_pca4to12 \
  --dataset balanced \
  --pca 12 \
  --out-dir ./outputs/qml_baselines \
  --train-subset 500 \
  --test-subset 300
"""

from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from sklearn.svm import SVC
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
from qiskit.circuit.library import ZFeatureMap, ZZFeatureMap, PauliFeatureMap, RealAmplitudes
from qiskit.quantum_info import Statevector, SparsePauliOp

warnings.filterwarnings("ignore")

HAS_QISKIT_ML = True
try:
    from qiskit_machine_learning.algorithms.classifiers import VQC, NeuralNetworkClassifier
    from qiskit_machine_learning.neural_networks import EstimatorQNN
    from qiskit_machine_learning.optimizers import COBYLA
except Exception:
    HAS_QISKIT_ML = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="balanced")
    parser.add_argument("--label-col", type=str, default="CLNSIG")
    parser.add_argument("--pca", type=int, default=12)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--train-subset", type=int, default=500)
    parser.add_argument("--test-subset", type=int, default=300)
    parser.add_argument("--reps-z", type=int, default=2)
    parser.add_argument("--reps-zz", type=int, default=2)
    parser.add_argument("--reps-pauli", type=int, default=2)
    parser.add_argument("--include-vqc", action="store_true")
    parser.add_argument("--include-qnn", action="store_true")
    parser.add_argument("--maxiter-vqc", type=int, default=80)
    parser.add_argument("--maxiter-qnn", type=int, default=80)
    parser.add_argument("--out-dir", type=str, required=True)
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


def make_reduced_subset(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xte: np.ndarray,
    yte: np.ndarray,
    train_n: int,
    test_n: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)

    tr_idx_0 = np.where(ytr == 0)[0]
    tr_idx_1 = np.where(ytr == 1)[0]
    te_idx_0 = np.where(yte == 0)[0]
    te_idx_1 = np.where(yte == 1)[0]

    ntr0 = train_n // 2
    ntr1 = train_n - ntr0
    nte0 = test_n // 2
    nte1 = test_n - nte0

    if len(tr_idx_0) < ntr0 or len(tr_idx_1) < ntr1:
        raise ValueError("Not enough train samples for balanced reduced subset")
    if len(te_idx_0) < nte0 or len(te_idx_1) < nte1:
        raise ValueError("Not enough test samples for balanced reduced subset")

    pick_tr = np.concatenate([
        rng.choice(tr_idx_0, size=ntr0, replace=False),
        rng.choice(tr_idx_1, size=ntr1, replace=False),
    ])
    pick_te = np.concatenate([
        rng.choice(te_idx_0, size=nte0, replace=False),
        rng.choice(te_idx_1, size=nte1, replace=False),
    ])

    rng.shuffle(pick_tr)
    rng.shuffle(pick_te)
    return Xtr[pick_tr], ytr[pick_tr], Xte[pick_te], yte[pick_te]


def scale_to_pi(Xtr: np.ndarray, Xte: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    scaler = MinMaxScaler(feature_range=(0.0, np.pi))
    return scaler.fit_transform(Xtr), scaler.transform(Xte)


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


def statevectors_from_featuremap(feature_map: QuantumCircuit, X: np.ndarray) -> np.ndarray:
    params = list(feature_map.parameters)
    svs = []
    for row in X:
        bind = {p: float(v) for p, v in zip(params, row)}
        sv = Statevector.from_instruction(feature_map.assign_parameters(bind, inplace=False))
        svs.append(np.asarray(sv.data))
    return np.vstack(svs)


def kernel_from_statevectors(SA: np.ndarray, SB: np.ndarray) -> np.ndarray:
    return np.abs(SA @ np.conjugate(SB).T) ** 2


def run_qsvm(feature_map: QuantumCircuit, Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray, seed: int) -> Tuple[np.ndarray, float, float]:
    t0 = time.time()
    S_tr = statevectors_from_featuremap(feature_map, Xtr)
    K_tr = kernel_from_statevectors(S_tr, S_tr)
    clf = SVC(kernel="precomputed", C=2.0, probability=True, random_state=seed)
    clf.fit(K_tr, ytr)
    t1 = time.time()

    t2 = time.time()
    S_te = statevectors_from_featuremap(feature_map, Xte)
    K_te = kernel_from_statevectors(S_te, S_tr)
    y_prob = clf.predict_proba(K_te)[:, 1]
    t3 = time.time()
    return y_prob, float(t1 - t0), float(t3 - t2)


def run_vqc(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray, reps: int, maxiter: int) -> Tuple[np.ndarray, float, float]:
    if not HAS_QISKIT_ML:
        raise RuntimeError("qiskit-machine-learning not installed")
    n_features = Xtr.shape[1]
    feature_map = ZZFeatureMap(feature_dimension=n_features, reps=reps)
    ansatz = RealAmplitudes(num_qubits=n_features, reps=reps)
    clf = VQC(feature_map=feature_map, ansatz=ansatz, optimizer=COBYLA(maxiter=maxiter))

    t0 = time.time()
    clf.fit(Xtr, ytr)
    t1 = time.time()

    t2 = time.time()
    y_prob = clf.predict_proba(Xte)[:, 1]
    t3 = time.time()
    return y_prob, float(t1 - t0), float(t3 - t2)


def run_qnn(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray, reps: int, maxiter: int) -> Tuple[np.ndarray, float, float]:
    if not HAS_QISKIT_ML:
        raise RuntimeError("qiskit-machine-learning not installed")
    n_features = Xtr.shape[1]
    feature_map = ZZFeatureMap(feature_dimension=n_features, reps=reps)
    ansatz = RealAmplitudes(num_qubits=n_features, reps=reps)

    qc = QuantumCircuit(n_features)
    qc.compose(feature_map, inplace=True)
    qc.compose(ansatz, inplace=True)

    observable = SparsePauliOp.from_list([("Z" + "I" * (n_features - 1), 1.0)])
    qnn = EstimatorQNN(
        circuit=qc,
        observables=observable,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
    )

    clf = NeuralNetworkClassifier(neural_network=qnn, optimizer=COBYLA(maxiter=maxiter), one_hot=False)

    t0 = time.time()
    clf.fit(Xtr, ytr)
    t1 = time.time()

    t2 = time.time()
    raw = np.asarray(clf.predict(Xte)).reshape(-1)
    y_prob = 1.0 / (1.0 + np.exp(-raw))
    t3 = time.time()
    return y_prob, float(t1 - t0), float(t3 - t2)


def save_summary_tables(raw_df: pd.DataFrame, out_dir: Path, train_subset: int, test_subset: int) -> None:
    group_cols = ["Dataset", "PCA", "Algorithm"]
    ignore_cols = group_cols + ["Seed", "TrainN", "TestN", "TP", "TN", "FP", "FN"]
    metric_cols = [c for c in raw_df.columns if c not in ignore_cols]

    mean_df = raw_df.groupby(group_cols)[metric_cols].mean().reset_index()
    std_df = raw_df.groupby(group_cols)[metric_cols].std().reset_index()

    suffix = f"reduced{train_subset}_{test_subset}"
    mean_df.to_csv(out_dir / f"mean_qml_baselines_balanced_pca12_{suffix}.csv", index=False)
    std_df.to_csv(out_dir / f"std_qml_baselines_balanced_pca12_{suffix}.csv", index=False)

    pm_df = mean_df.copy()
    for c in metric_cols:
        pm_df[c] = mean_df[c].map(lambda x: f"{x:.4f}") + " ± " + std_df[c].map(lambda x: f"{x:.4f}")
    pm_df.to_csv(out_dir / f"meanpmstd_qml_baselines_balanced_pca12_{suffix}.csv", index=False)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if (args.include_vqc or args.include_qnn) and not HAS_QISKIT_ML:
        print("qiskit-machine-learning not installed -> VQC/QNN will be skipped")

    Xtr_full, ytr_full, Xte_full, yte_full = load_train_test(args.data_root, args.dataset, args.pca, args.label_col)
    raw_rows = []

    for seed in args.seeds:
        print("\n" + "=" * 100)
        print(f"RUNNING QML BASELINES | {args.dataset.upper()} | PCA-{args.pca} | seed={seed}")
        print("=" * 100)

        Xtr_sub, ytr_sub, Xte_sub, yte_sub = make_reduced_subset(
            Xtr_full, ytr_full, Xte_full, yte_full,
            train_n=args.train_subset, test_n=args.test_subset, seed=seed,
        )
        Xtr, Xte = scale_to_pi(Xtr_sub, Xte_sub)
        d = Xtr.shape[1]

        qml_models = {
            "QSVM_Z": ZFeatureMap(feature_dimension=d, reps=args.reps_z),
            "QSVM_ZZ": ZZFeatureMap(feature_dimension=d, reps=args.reps_zz),
            "QSVM_Pauli": PauliFeatureMap(feature_dimension=d, reps=args.reps_pauli, paulis=["X", "Y", "Z"]),
        }

        for algo_name, fmap in qml_models.items():
            try:
                y_prob, train_t, test_t = run_qsvm(fmap, Xtr, ytr_sub, Xte, seed)
                m = compute_metrics(yte_sub, y_prob, threshold=args.threshold)
                m.update({
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
                raw_rows.append(m)
                print(
                    f"{algo_name:>12} | seed={seed} | "
                    f"Acc={m['Accuracy']:.4f} | F1={m['F1']:.4f} | "
                    f"MCC={m['MCC']:.4f} | ROC_AUC={m['ROC_AUC']:.4f} | "
                    f"Train={train_t:.4f}s | Test={test_t:.4f}s | Total={train_t + test_t:.4f}s"
                )
            except Exception as exc:
                print(f"{algo_name} failed on seed={seed}: {exc}")

        if args.include_vqc and HAS_QISKIT_ML:
            try:
                y_prob, train_t, test_t = run_vqc(Xtr, ytr_sub, Xte, reps=2, maxiter=args.maxiter_vqc)
                m = compute_metrics(yte_sub, y_prob, threshold=args.threshold)
                m.update({
                    "Dataset": args.dataset, "PCA": args.pca, "Seed": seed, "Algorithm": "VQC",
                    "TrainN": int(len(Xtr)), "TestN": int(len(Xte)),
                    "TrainTime_s": train_t, "TestTime_s": test_t, "TotalTime_s": train_t + test_t,
                })
                raw_rows.append(m)
            except Exception as exc:
                print(f"VQC failed on seed={seed}: {exc}")

        if args.include_qnn and HAS_QISKIT_ML:
            try:
                y_prob, train_t, test_t = run_qnn(Xtr, ytr_sub, Xte, reps=2, maxiter=args.maxiter_qnn)
                m = compute_metrics(yte_sub, y_prob, threshold=args.threshold)
                m.update({
                    "Dataset": args.dataset, "PCA": args.pca, "Seed": seed, "Algorithm": "QNN",
                    "TrainN": int(len(Xtr)), "TestN": int(len(Xte)),
                    "TrainTime_s": train_t, "TestTime_s": test_t, "TotalTime_s": train_t + test_t,
                })
                raw_rows.append(m)
            except Exception as exc:
                print(f"QNN failed on seed={seed}: {exc}")

    raw_df = pd.DataFrame(raw_rows)
    suffix = f"reduced{args.train_subset}_{args.test_subset}"
    raw_path = out_dir / f"raw_all_seeds_qml_baselines_balanced_pca12_{suffix}.csv"
    raw_df.to_csv(raw_path, index=False)
    save_summary_tables(raw_df, out_dir, args.train_subset, args.test_subset)

    print("\nSaved raw results:", raw_path)
    print("Done.")


if __name__ == "__main__":
    main()
