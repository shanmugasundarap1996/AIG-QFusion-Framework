#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    brier_score_loss,
)

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector

warnings.filterwarnings("ignore")

# ============================================================
# USER CONFIG
# ============================================================
DATA_ROOT = r"C:\Users\shanmugam\shan work 1\data\dataset_saved_500_300_pca4to12"
DS = "balanced"
LABEL_COL = "CLNSIG"

PCA_LIST = [12]
SEEDS = [0, 1, 2, 3, 4]
N_SPLITS = 3

# Quantum/model config
REPS_Z = 2
REPS_ZZ = 2
REPS_INT = 3
C_SVC = 2.0

# Meta configs
META_MAXIT_P5 = 1500
META_C_P5 = 0.8

META_MAXIT_P6 = 4000
META_C_P6 = 1.0

# Adaptive graph
TOP_K_INTERACTIONS = 8
USE_SKIP_EDGES = True
INTERACTION_SCALE = 1.25

# Weight optimization
# P4 uses coarse weighting, P6 uses fine weighting
N_WEIGHT_SAMPLES_P4 = 30
N_WEIGHT_SAMPLES_P5 = 80
N_WEIGHT_SAMPLES_P6 = 160
METRIC_FOR_WEIGHT_SEARCH = "MCC"

# Threshold optimization
THRESH_GRID_P6 = np.linspace(0.30, 0.70, 81)
OPTIMIZE_THRESHOLD_FOR = "MCC"

OUT_DIR = r"C:\Users\shanmugam\shan work 1\New folder\progressive_ablation_adjusted_output"
os.makedirs(OUT_DIR, exist_ok=True)

print("Adjusted progressive ablation outputs will be saved in:")
print(OUT_DIR)

# ============================================================
# PROGRESSIVE VARIANTS
# ============================================================
PROGRESSIVE_CONFIGS = {
    "P1_BaseFusion": {
        "branches": ["Z", "ZZ"],
        "graph_type": "none",
        "use_weight_optimization": False,
        "weight_samples": 0,
        "use_meta": False,
        "meta_use_decisions": False,
        "meta_maxit": None,
        "meta_C": None,
        "use_threshold_optimization": False,
    },
    "P2_AddINT": {
        "branches": ["Z", "ZZ", "INT"],
        "graph_type": "adjacent",
        "use_weight_optimization": False,
        "weight_samples": 0,
        "use_meta": False,
        "meta_use_decisions": False,
        "meta_maxit": None,
        "meta_C": None,
        "use_threshold_optimization": False,
    },
    "P3_GraphINT": {
        "branches": ["Z", "ZZ", "INT"],
        "graph_type": "adaptive",
        "use_weight_optimization": False,
        "weight_samples": 0,
        "use_meta": False,
        "meta_use_decisions": False,
        "meta_maxit": None,
        "meta_C": None,
        "use_threshold_optimization": False,
    },
    "P4_AdaptiveFusion": {
        "branches": ["Z", "ZZ", "INT"],
        "graph_type": "adaptive",
        "use_weight_optimization": True,
        "weight_samples": N_WEIGHT_SAMPLES_P4,   # coarse
        "use_meta": False,
        "meta_use_decisions": False,
        "meta_maxit": None,
        "meta_C": None,
        "use_threshold_optimization": False,
    },
    "P5_MetaFusion": {
        "branches": ["Z", "ZZ", "INT"],
        "graph_type": "adaptive",
        "use_weight_optimization": True,
        "weight_samples": N_WEIGHT_SAMPLES_P5,   # medium
        "use_meta": True,
        "meta_use_decisions": False,             # probabilities only
        "meta_maxit": META_MAXIT_P5,
        "meta_C": META_C_P5,
        "use_threshold_optimization": False,
    },
    "P6_AIGQFusion": {
        "branches": ["Z", "ZZ", "INT"],
        "graph_type": "adaptive",
        "use_weight_optimization": True,
        "weight_samples": N_WEIGHT_SAMPLES_P6,   # fine
        "use_meta": True,
        "meta_use_decisions": True,              # probabilities + decisions
        "meta_maxit": META_MAXIT_P6,
        "meta_C": META_C_P6,
        "use_threshold_optimization": True,
    },
}

VARIANT_ORDER = list(PROGRESSIVE_CONFIGS.keys())

# ============================================================
# LOAD DATA
# ============================================================
def load_train_test(root, ds, k):
    pdir = os.path.join(root, ds, f"pca_{k}")
    tr_path = os.path.join(pdir, "train.csv")
    te_path = os.path.join(pdir, "test.csv")

    if not (os.path.exists(tr_path) and os.path.exists(te_path)):
        raise FileNotFoundError(f"Missing: {tr_path} or {te_path}")

    tr = pd.read_csv(tr_path)
    te = pd.read_csv(te_path)

    if LABEL_COL not in tr.columns or LABEL_COL not in te.columns:
        raise ValueError(f"'{LABEL_COL}' missing in {ds}/pca_{k}")

    Xtr = tr.drop(columns=[LABEL_COL]).values.astype(float)
    ytr = tr[LABEL_COL].values.astype(int)
    Xte = te.drop(columns=[LABEL_COL]).values.astype(float)
    yte = te[LABEL_COL].values.astype(int)
    return Xtr, ytr, Xte, yte

def scale_to_pi(Xtr, Xte):
    scaler = MinMaxScaler(feature_range=(0.0, np.pi))
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)
    return Xtr_s, Xte_s

# ============================================================
# INTERACTION GRAPH
# ============================================================
def build_interaction_graph(Xtr, top_k=8, use_skip_edges=True):
    d = Xtr.shape[1]
    corr = np.corrcoef(Xtr, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)
    vars_ = np.var(Xtr, axis=0)

    pair_scores = []
    for i in range(d):
        for j in range(i + 1, d):
            if (not use_skip_edges) and (j != i + 1):
                continue
            score = abs(corr[i, j]) * (vars_[i] + vars_[j] + 1e-12)
            pair_scores.append((score, i, j))

    pair_scores.sort(reverse=True, key=lambda t: t[0])
    selected = pair_scores[:top_k]
    edge_list = [(i, j, float(score)) for score, i, j in selected]
    return edge_list

def build_adjacent_graph(Xtr):
    d = Xtr.shape[1]
    vars_ = np.var(Xtr, axis=0)
    edge_list = []
    for i in range(d - 1):
        score = float(vars_[i] + vars_[i + 1] + 1e-12)
        edge_list.append((i, i + 1, score))
    return edge_list

# ============================================================
# QUANTUM FEATURE MAPS
# ============================================================
def feature_map_Z(d: int, reps: int = 2):
    x = ParameterVector("x", d)
    qc = QuantumCircuit(d, name="ZMap")
    for _ in range(reps):
        for i in range(d):
            qc.h(i)
            qc.rz(x[i], i)
        for i in range(d - 1):
            qc.cx(i, i + 1)
    return qc, list(x)

def feature_map_ZZ(d: int, reps: int = 2):
    x = ParameterVector("x", d)
    qc = QuantumCircuit(d, name="ZZMap")
    for _ in range(reps):
        for i in range(d):
            qc.h(i)
            qc.rz(x[i], i)
        for i in range(d - 1):
            qc.rzz(x[i] * x[i + 1], i, i + 1)
    return qc, list(x)

def feature_map_INT(d: int, interaction_edges, reps: int = 3, interaction_scale: float = 1.25):
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

# ============================================================
# STATEVECTOR + KERNEL
# ============================================================
def statevectors(X_, map_name, reps, interaction_edges=None):
    d = X_.shape[1]
    if map_name == "Z":
        qc, params = feature_map_Z(d, reps)
    elif map_name == "ZZ":
        qc, params = feature_map_ZZ(d, reps)
    elif map_name == "INT":
        qc, params = feature_map_INT(d, interaction_edges, reps, INTERACTION_SCALE)
    else:
        raise ValueError(f"Unknown map_name: {map_name}")

    svs = []
    for row in X_:
        bind = {p: float(v) for p, v in zip(params, row)}
        sv = Statevector.from_instruction(qc.assign_parameters(bind, inplace=False))
        svs.append(np.asarray(sv.data))
    return np.vstack(svs)

def precompute_branch_kernel(Xtr, Xte, map_name, reps, interaction_edges=None):
    t0 = time.time()
    S_tr = statevectors(Xtr, map_name, reps, interaction_edges)
    S_te = statevectors(Xte, map_name, reps, interaction_edges)

    K_trtr = np.abs(S_tr @ np.conjugate(S_tr).T) ** 2
    K_tetr = np.abs(S_te @ np.conjugate(S_tr).T) ** 2
    t1 = time.time()

    return {
        "K_trtr": K_trtr,
        "K_tetr": K_tetr,
        "kernel_build_time": float(t1 - t0),
    }

# ============================================================
# METRICS
# ============================================================
def ece_score(y_true, y_prob, n_bins=10, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    bins = np.linspace(0, 1, n_bins + 1)
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

def compute_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
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

# ============================================================
# HELPERS
# ============================================================
def metric_from_probs(y_true, y_prob, metric_name="MCC", threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    if metric_name.upper() == "MCC":
        return matthews_corrcoef(y_true, y_pred)
    elif metric_name.upper() == "F1":
        return f1_score(y_true, y_pred, zero_division=0)
    else:
        raise ValueError("metric_name must be 'MCC' or 'F1'")

def optimize_threshold(y_true, y_prob, metric_name="MCC", grid=None):
    if grid is None:
        grid = np.linspace(0.30, 0.70, 81)

    best_t = 0.5
    best_score = -1e18
    for t in grid:
        score = metric_from_probs(y_true, y_prob, metric_name=metric_name, threshold=t)
        if score > best_score:
            best_score = score
            best_t = float(t)
    return best_t, best_score

def normalize_positive_weights(w):
    w = np.asarray(w, dtype=float)
    w = np.maximum(w, 1e-10)
    return w / np.sum(w)

def random_simplex_weights(n_models, n_samples=128, seed=0):
    rng = np.random.default_rng(seed)
    weights = [np.ones(n_models) / n_models]
    for i in range(n_models):
        w = np.zeros(n_models)
        w[i] = 1.0
        weights.append(w)
    for _ in range(n_samples):
        w = rng.dirichlet(alpha=np.ones(n_models))
        weights.append(w)
    return [normalize_positive_weights(w) for w in weights]

def optimize_kernel_weights_from_oof(ytr, oof_probs_by_branch, metric_name="MCC", seed=0, n_weight_samples=80):
    candidates = random_simplex_weights(
        n_models=oof_probs_by_branch.shape[1],
        n_samples=n_weight_samples,
        seed=seed
    )

    best_w = None
    best_metric = -1e18
    best_threshold = 0.5

    for w in candidates:
        p = oof_probs_by_branch @ w
        t, _ = optimize_threshold(ytr, p, metric_name=metric_name, grid=np.linspace(0.35, 0.65, 31))
        score = metric_from_probs(ytr, p, metric_name=metric_name, threshold=t)
        if score > best_metric:
            best_metric = score
            best_w = w.copy()
            best_threshold = t

    return best_w, best_metric, best_threshold

def train_branch_from_kernel(K_tr, y_tr, K_eval, seed):
    t0 = time.time()
    clf = SVC(kernel="precomputed", C=C_SVC, probability=True, random_state=seed)
    clf.fit(K_tr, y_tr)
    t1 = time.time()

    t2 = time.time()
    prob = clf.predict_proba(K_eval)[:, 1]
    dec = clf.decision_function(K_eval)
    t3 = time.time()

    return prob, dec, (t1 - t0), (t3 - t2)

# ============================================================
# PRECOMPUTE KERNEL BANKS
# ============================================================
def build_kernel_banks(Xtr, Xte):
    print("\nPrecomputing kernel banks ...")

    adaptive_edges = build_interaction_graph(
        Xtr,
        top_k=TOP_K_INTERACTIONS,
        use_skip_edges=USE_SKIP_EDGES
    )
    adjacent_edges = build_adjacent_graph(Xtr)

    bank_adaptive = {}
    bank_adjacent = {}

    bank_adaptive["Z"] = precompute_branch_kernel(Xtr, Xte, "Z", REPS_Z, None)
    bank_adjacent["Z"] = bank_adaptive["Z"]

    bank_adaptive["ZZ"] = precompute_branch_kernel(Xtr, Xte, "ZZ", REPS_ZZ, None)
    bank_adjacent["ZZ"] = bank_adaptive["ZZ"]

    bank_adaptive["INT"] = precompute_branch_kernel(Xtr, Xte, "INT", REPS_INT, adaptive_edges)
    bank_adjacent["INT"] = precompute_branch_kernel(Xtr, Xte, "INT", REPS_INT, adjacent_edges)

    print("Kernel banks ready.")
    return bank_adaptive, bank_adjacent

# ============================================================
# FAST PROGRESSIVE ONE-SEED RUN
# ============================================================
def run_progressive_seed_fast(Xtr, ytr, Xte, seed, cfg, kernel_bank):
    total_train_time = 0.0
    total_test_time = 0.0

    active_branches = cfg["branches"]
    M = len(active_branches)

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    fold_indices = list(skf.split(Xtr, ytr))

    # ------------------------------------------------------------
    # P1 / P2 / P3: equal-weight fused kernel directly
    # ------------------------------------------------------------
    if (not cfg["use_weight_optimization"]) and (not cfg["use_meta"]) and (not cfg["use_threshold_optimization"]):
        kernel_weights = np.ones(M, dtype=float) / M

        K_tr_full_fused = None
        K_te_fused = None

        for w, name in zip(kernel_weights, active_branches):
            K_tr_full = kernel_bank[name]["K_trtr"]
            K_te = kernel_bank[name]["K_tetr"]

            if K_tr_full_fused is None:
                K_tr_full_fused = w * K_tr_full
                K_te_fused = w * K_te
            else:
                K_tr_full_fused += w * K_tr_full
                K_te_fused += w * K_te

        fused_p, fused_d, fit_t, pred_t = train_branch_from_kernel(
            K_tr_full_fused, ytr, K_te_fused, seed
        )
        total_train_time += fit_t
        total_test_time += pred_t

        details = {
            "kernel_weights": kernel_weights,
            "kernel_weight_score": np.nan,
            "kernel_weight_thresh": np.nan,
            "threshold": 0.5,
            "threshold_score": np.nan,
            "TrainTime_s": float(total_train_time),
            "TestTime_s": float(total_test_time),
            "TotalTime_s": float(total_train_time + total_test_time),
            "n_active_branches": int(M),
        }
        return fused_p, details

    # ------------------------------------------------------------
    # P4 / P5 / P6
    # ------------------------------------------------------------
    oof_prob = np.zeros((len(Xtr), M), dtype=float)
    oof_dec = np.zeros((len(Xtr), M), dtype=float)

    # Stage 1: OOF branch models
    for idx_tr, idx_va in fold_indices:
        y_fold_tr = ytr[idx_tr]

        for m, name in enumerate(active_branches):
            K_full = kernel_bank[name]["K_trtr"]
            K_tr = K_full[np.ix_(idx_tr, idx_tr)]
            K_va = K_full[np.ix_(idx_va, idx_tr)]

            p, d, fit_t, pred_t = train_branch_from_kernel(K_tr, y_fold_tr, K_va, seed)
            total_train_time += fit_t
            total_test_time += pred_t

            oof_prob[idx_va, m] = p
            oof_dec[idx_va, m] = d

    # Stage 2: adaptive weights
    if cfg["use_weight_optimization"]:
        t0 = time.time()
        kernel_weights, kernel_weight_score, kernel_weight_thresh = optimize_kernel_weights_from_oof(
            ytr=ytr,
            oof_probs_by_branch=oof_prob,
            metric_name=METRIC_FOR_WEIGHT_SEARCH,
            seed=seed,
            n_weight_samples=cfg["weight_samples"]
        )
        t1 = time.time()
        total_train_time += (t1 - t0)
    else:
        kernel_weights = np.ones(M, dtype=float) / M
        kernel_weight_score = np.nan
        kernel_weight_thresh = np.nan

    # Stage 3: OOF fused kernel
    oof_fused_prob = np.zeros((len(Xtr), 1), dtype=float)
    oof_fused_dec = np.zeros((len(Xtr), 1), dtype=float)

    for idx_tr, idx_va in fold_indices:
        y_fold_tr = ytr[idx_tr]

        K_tr_fused = None
        K_va_fused = None

        for w, name in zip(kernel_weights, active_branches):
            K_full = kernel_bank[name]["K_trtr"]
            K_tr = K_full[np.ix_(idx_tr, idx_tr)]
            K_va = K_full[np.ix_(idx_va, idx_tr)]

            if K_tr_fused is None:
                K_tr_fused = w * K_tr
                K_va_fused = w * K_va
            else:
                K_tr_fused += w * K_tr
                K_va_fused += w * K_va

        p, d, fit_t, pred_t = train_branch_from_kernel(K_tr_fused, y_fold_tr, K_va_fused, seed)
        total_train_time += fit_t
        total_test_time += pred_t

        oof_fused_prob[idx_va, 0] = p
        oof_fused_dec[idx_va, 0] = d

    # Stage 4: meta
    if cfg["use_meta"]:
        if cfg["meta_use_decisions"]:
            Z_train_meta = np.hstack([oof_prob, oof_dec, oof_fused_prob, oof_fused_dec])
        else:
            Z_train_meta = np.hstack([oof_prob, oof_fused_prob])

        t0 = time.time()
        meta = LogisticRegression(
            max_iter=cfg["meta_maxit"],
            C=cfg["meta_C"],
            random_state=seed
        )
        meta.fit(Z_train_meta, ytr)
        t1 = time.time()
        total_train_time += (t1 - t0)

        t2 = time.time()
        oof_final_prob = meta.predict_proba(Z_train_meta)[:, 1]
        t3 = time.time()
        total_test_time += (t3 - t2)
    else:
        meta = None
        oof_final_prob = oof_fused_prob[:, 0]

    # Stage 5: threshold
    if cfg["use_threshold_optimization"]:
        t4 = time.time()
        threshold, threshold_score = optimize_threshold(
            ytr, oof_final_prob,
            metric_name=OPTIMIZE_THRESHOLD_FOR,
            grid=THRESH_GRID_P6
        )
        t5 = time.time()
        total_train_time += (t5 - t4)
    else:
        threshold = 0.5
        threshold_score = np.nan

    # Stage 6: full train -> test branch models
    Z_test_prob = np.zeros((len(Xte), M), dtype=float)
    Z_test_dec = np.zeros((len(Xte), M), dtype=float)

    for m, name in enumerate(active_branches):
        K_tr_full = kernel_bank[name]["K_trtr"]
        K_te = kernel_bank[name]["K_tetr"]

        p, d, fit_t, pred_t = train_branch_from_kernel(K_tr_full, ytr, K_te, seed)
        total_train_time += fit_t
        total_test_time += pred_t

        Z_test_prob[:, m] = p
        Z_test_dec[:, m] = d

    # Stage 7: full fused kernel -> test
    K_tr_full_fused = None
    K_te_fused = None

    for w, name in zip(kernel_weights, active_branches):
        K_tr_full = kernel_bank[name]["K_trtr"]
        K_te = kernel_bank[name]["K_tetr"]

        if K_tr_full_fused is None:
            K_tr_full_fused = w * K_tr_full
            K_te_fused = w * K_te
        else:
            K_tr_full_fused += w * K_tr_full
            K_te_fused += w * K_te

    fused_p, fused_d, fit_t, pred_t = train_branch_from_kernel(
        K_tr_full_fused, ytr, K_te_fused, seed
    )
    total_train_time += fit_t
    total_test_time += pred_t

    # Stage 8: final output
    if cfg["use_meta"]:
        if cfg["meta_use_decisions"]:
            Z_test_meta = np.hstack([
                Z_test_prob, Z_test_dec,
                fused_p.reshape(-1, 1),
                fused_d.reshape(-1, 1)
            ])
        else:
            Z_test_meta = np.hstack([
                Z_test_prob,
                fused_p.reshape(-1, 1)
            ])

        t0 = time.time()
        y_prob = meta.predict_proba(Z_test_meta)[:, 1]
        t1 = time.time()
        total_test_time += (t1 - t0)
    else:
        y_prob = fused_p

    details = {
        "kernel_weights": kernel_weights,
        "kernel_weight_score": kernel_weight_score,
        "kernel_weight_thresh": kernel_weight_thresh,
        "threshold": threshold,
        "threshold_score": threshold_score,
        "TrainTime_s": float(total_train_time),
        "TestTime_s": float(total_test_time),
        "TotalTime_s": float(total_train_time + total_test_time),
        "n_active_branches": int(M),
    }
    return y_prob, details

# ============================================================
# PRECOMPUTE KERNEL BANKS
# ============================================================
def build_kernel_banks(Xtr, Xte):
    print("\nPrecomputing kernel banks ...")

    adaptive_edges = build_interaction_graph(
        Xtr,
        top_k=TOP_K_INTERACTIONS,
        use_skip_edges=USE_SKIP_EDGES
    )
    adjacent_edges = build_adjacent_graph(Xtr)

    bank_adaptive = {}
    bank_adjacent = {}

    bank_adaptive["Z"] = precompute_branch_kernel(Xtr, Xte, "Z", REPS_Z, None)
    bank_adjacent["Z"] = bank_adaptive["Z"]

    bank_adaptive["ZZ"] = precompute_branch_kernel(Xtr, Xte, "ZZ", REPS_ZZ, None)
    bank_adjacent["ZZ"] = bank_adaptive["ZZ"]

    bank_adaptive["INT"] = precompute_branch_kernel(Xtr, Xte, "INT", REPS_INT, adaptive_edges)
    bank_adjacent["INT"] = precompute_branch_kernel(Xtr, Xte, "INT", REPS_INT, adjacent_edges)

    print("Kernel banks ready.")
    return bank_adaptive, bank_adjacent

# ============================================================
# RUN PROGRESSIVE ABLATION
# ============================================================
raw_rows = []

for k in PCA_LIST:
    print("\n" + "=" * 100)
    print(f"ADJUSTED PROGRESSIVE ABLATION | DATASET={DS.upper()} | PCA-{k}")
    print("=" * 100)

    Xtr_raw, ytr, Xte_raw, yte = load_train_test(DATA_ROOT, DS, k)
    Xtr, Xte = scale_to_pi(Xtr_raw, Xte_raw)

    bank_adaptive, bank_adjacent = build_kernel_banks(Xtr, Xte)

    for variant_name, cfg in PROGRESSIVE_CONFIGS.items():
        print("\n" + "-" * 100)
        print(f"RUNNING VARIANT: {variant_name}")
        print("-" * 100)

        if cfg["graph_type"] == "adaptive":
            kernel_bank = bank_adaptive
        elif cfg["graph_type"] == "adjacent":
            kernel_bank = bank_adjacent
        else:
            kernel_bank = bank_adaptive

        print("Active branches:", cfg["branches"])
        print("Graph type:", cfg["graph_type"])
        print("Use weight optimization:", cfg["use_weight_optimization"])
        print("Meta uses decisions:", cfg["meta_use_decisions"])
        print("Use meta:", cfg["use_meta"])
        print("Use threshold optimization:", cfg["use_threshold_optimization"])

        for seed in SEEDS:
            t0 = time.time()
            y_prob, details = run_progressive_seed_fast(
                Xtr, ytr, Xte, seed=seed, cfg=cfg, kernel_bank=kernel_bank
            )
            t1 = time.time()

            threshold = details["threshold"]
            m = compute_metrics(yte, y_prob, threshold=threshold)

            w = details["kernel_weights"]
            w_z = np.nan
            w_zz = np.nan
            w_int = np.nan
            for ww, name in zip(w, cfg["branches"]):
                if name == "Z":
                    w_z = float(ww)
                elif name == "ZZ":
                    w_zz = float(ww)
                elif name == "INT":
                    w_int = float(ww)

            m.update({
                "Dataset": DS,
                "PCA": k,
                "Variant": variant_name,
                "Seed": seed,
                "TrainN": int(len(Xtr)),
                "TestN": int(len(Xte)),
                "Threshold": float(threshold),
                "ThresholdScore": float(details["threshold_score"]) if not np.isnan(details["threshold_score"]) else np.nan,
                "W_Z": w_z,
                "W_ZZ": w_zz,
                "W_INT": w_int,
                "KernelWeightScore": float(details["kernel_weight_score"]) if not np.isnan(details["kernel_weight_score"]) else np.nan,
                "KernelWeightThresh": float(details["kernel_weight_thresh"]) if not np.isnan(details["kernel_weight_thresh"]) else np.nan,
                "N_ActiveBranches": int(details["n_active_branches"]),
                "TrainTime_s": float(details["TrainTime_s"]),
                "TestTime_s": float(details["TestTime_s"]),
                "TotalTime_s": float(details["TotalTime_s"]),
                "WallClock_s": float(t1 - t0),
            })
            raw_rows.append(m)

            print(
                f"{variant_name:>20} | seed={seed} | "
                f"Acc={m['Accuracy']:.4f} | F1={m['F1']:.4f} | MCC={m['MCC']:.4f} | "
                f"ROC_AUC={m['ROC_AUC']:.4f} | Total={m['TotalTime_s']:.2f}s | "
                f"Wall={m['WallClock_s']:.2f}s"
            )

raw_df = pd.DataFrame(raw_rows)
raw_csv = os.path.join(OUT_DIR, "adjusted_progressive_raw_results_aigqfusion.csv")
raw_df.to_csv(raw_csv, index=False)

# ============================================================
# MEAN / STD / MEAN±STD
# ============================================================
group_cols = ["Dataset", "PCA", "Variant"]
ignore_cols = group_cols + ["Seed", "TrainN", "TestN", "TP", "TN", "FP", "FN"]
metric_cols = [c for c in raw_df.columns if c not in ignore_cols]

mean_df = raw_df.groupby(group_cols)[metric_cols].mean().reset_index()
std_df = raw_df.groupby(group_cols)[metric_cols].std().reset_index()

mean_df["Variant"] = pd.Categorical(mean_df["Variant"], categories=VARIANT_ORDER, ordered=True)
std_df["Variant"] = pd.Categorical(std_df["Variant"], categories=VARIANT_ORDER, ordered=True)

mean_df = mean_df.sort_values(["Dataset", "PCA", "Variant"]).reset_index(drop=True)
std_df = std_df.sort_values(["Dataset", "PCA", "Variant"]).reset_index(drop=True)

mean_csv = os.path.join(OUT_DIR, "adjusted_progressive_mean_aigqfusion.csv")
std_csv = os.path.join(OUT_DIR, "adjusted_progressive_std_aigqfusion.csv")
mean_df.to_csv(mean_csv, index=False)
std_df.to_csv(std_csv, index=False)

pm_df = mean_df.copy()
for c in metric_cols:
    pm_df[c] = mean_df[c].map(lambda x: f"{x:.4f}") + " ± " + std_df[c].map(lambda x: f"{x:.4f}")

pm_csv = os.path.join(OUT_DIR, "adjusted_progressive_meanpmstd_aigqfusion.csv")
pm_df.to_csv(pm_csv, index=False)

# ============================================================
# DISPLAY TABLE + HIGHLIGHT
# ============================================================
cols_show = [
    "Dataset", "PCA", "Variant",
    "Accuracy", "Precision", "Recall", "F1", "MCC",
    "ROC_AUC", "PR_AUC", "Brier", "ECE",
    "TrainTime_s", "TestTime_s", "TotalTime_s"
]

disp_pm_df = pm_df[cols_show].copy()
disp_mean_df = mean_df[cols_show].copy()

higher_better = ["Accuracy", "Precision", "Recall", "F1", "MCC", "ROC_AUC", "PR_AUC"]
lower_better = ["Brier", "ECE", "TrainTime_s", "TestTime_s", "TotalTime_s"]

BEST_COLOR = "background-color: #d9ead3;"
BORDER = "1px solid black"

def highlight_best_cells_within_pca(_):
    styles = pd.DataFrame("", index=disp_pm_df.index, columns=disp_pm_df.columns)

    for pca_val in disp_mean_df["PCA"].unique():
        mask = disp_mean_df["PCA"] == pca_val
        sub = disp_mean_df.loc[mask]

        for c in higher_better:
            best = sub[c].max()
            styles.loc[mask & np.isclose(disp_mean_df[c], best), c] = BEST_COLOR

        for c in lower_better:
            best = sub[c].min()
            styles.loc[mask & np.isclose(disp_mean_df[c], best), c] = BEST_COLOR

    return styles

sty = (
    disp_pm_df.style
    .apply(highlight_best_cells_within_pca, axis=None)
    .hide(axis="index")
    .set_properties(**{
        "text-align": "center",
        "vertical-align": "middle",
        "border": BORDER,
        "font-size": "11pt",
        "font-family": "Times New Roman"
    })
    .set_table_styles([
        {
            "selector": "table",
            "props": [
                ("border-collapse", "collapse"),
                ("border", BORDER),
                ("width", "100%"),
            ]
        },
        {
            "selector": "th",
            "props": [
                ("border", BORDER),
                ("text-align", "center"),
                ("vertical-align", "middle"),
                ("background-color", "#f2f2f2"),
                ("font-weight", "bold"),
                ("padding", "6px"),
                ("font-family", "Times New Roman"),
                ("font-size", "11pt")
            ]
        },
        {
            "selector": "td",
            "props": [
                ("border", BORDER),
                ("text-align", "center"),
                ("vertical-align", "middle"),
                ("padding", "6px"),
                ("font-family", "Times New Roman"),
                ("font-size", "11pt")
            ]
        }
    ])
)

print("\n=== ADJUSTED PROGRESSIVE ABLATION MEAN ± STD TABLE: AIG-QFusion ===\n")
print("The progressive study shows how interaction-aware fusion, adaptive weighting, meta-learning, and threshold refinement change the performance of the multi-branch quantum system.\n")

try:
    from IPython.display import display
    display(sty)
except Exception:
    print(disp_pm_df.to_string(index=False))

# Save styled Excel
try:
    import openpyxl
    out_xlsx = os.path.join(OUT_DIR, "adjusted_progressive_highlighted_meanpmstd_aigqfusion.xlsx")
    sty.to_excel(out_xlsx, engine="openpyxl", index=False)
    print("\nHighlighted Excel saved:", out_xlsx)
except Exception as e:
    print("\nExcel export skipped:", e)

# Save styled HTML
try:
    html_path = os.path.join(OUT_DIR, "adjusted_progressive_highlighted_meanpmstd_aigqfusion.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(sty.to_html())
    print("Styled HTML saved:", html_path)
except Exception as e:
    print("HTML export skipped:", e)

# ============================================================
# COMPACT PAPER TABLE
# ============================================================
paper_cols = [
    "Dataset", "PCA", "Variant",
    "Accuracy", "F1", "MCC", "ROC_AUC", "PR_AUC", "Brier", "TotalTime_s"
]
paper_df = pm_df[paper_cols].copy()
paper_csv = os.path.join(OUT_DIR, "adjusted_progressive_paper_table_aigqfusion.csv")
paper_df.to_csv(paper_csv, index=False)

# ============================================================
# PLOTS
# ============================================================
plot_df = mean_df.copy()

for pca_val in plot_df["PCA"].unique():
    sub = plot_df[plot_df["PCA"] == pca_val].copy()
    sub["Variant"] = pd.Categorical(sub["Variant"], categories=VARIANT_ORDER, ordered=True)
    sub = sub.sort_values("Variant")

    plt.figure(figsize=(10, 4.5))
    plt.plot(sub["Variant"], sub["Accuracy"], marker="o")
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Accuracy")
    plt.title(f"Adjusted Progressive Ablation — Accuracy (PCA-{pca_val})")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"adjusted_progressive_Accuracy_pca{pca_val}.png"), dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 4.5))
    plt.plot(sub["Variant"], sub["F1"], marker="o")
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("F1")
    plt.title(f"Adjusted Progressive Ablation — F1 (PCA-{pca_val})")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"adjusted_progressive_F1_pca{pca_val}.png"), dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 4.5))
    plt.plot(sub["Variant"], sub["MCC"], marker="o")
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("MCC")
    plt.title(f"Adjusted Progressive Ablation — MCC (PCA-{pca_val})")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"adjusted_progressive_MCC_pca{pca_val}.png"), dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 4.5))
    plt.bar(sub["Variant"], sub["TotalTime_s"])
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("TotalTime_s")
    plt.title(f"Adjusted Progressive Ablation — Total Runtime (PCA-{pca_val})")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"adjusted_progressive_TotalTime_pca{pca_val}.png"), dpi=300, bbox_inches="tight")
    plt.close()

print("\nSaved files:")
print(raw_csv)
print(mean_csv)
print(std_csv)
print(pm_csv)
print(paper_csv)
print("\nDone.")


# In[ ]:




