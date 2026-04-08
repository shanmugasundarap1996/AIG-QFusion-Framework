#!/usr/bin/env python
# coding: utf-8

# In[1]:


# ============================================================
# AIG-QFusion (Quantum) — BALANCED | PCA 4..12
# Compact evaluation version for Section 4.2
# Final output:
#   1) raw all-seed metrics
#   2) mean table
#   3) std table
#   4) mean ± std table
#   5) highlighted Excel with best cells
# ============================================================

import os
import time
import warnings
import numpy as np
import pandas as pd

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

PCA_LIST = list(range(4, 13))   # 4..12
SEEDS = [0, 1, 2, 3, 4]
N_SPLITS = 3

# Quantum / model config
REPS_Z = 2
REPS_ZZ = 2
REPS_INT = 3
C_SVC = 2.0
META_MAXIT = 4000

# Adaptive graph
TOP_K_INTERACTIONS = 8
USE_SKIP_EDGES = True
INTERACTION_SCALE = 1.25

# Weight optimization
N_WEIGHT_SAMPLES = 160
METRIC_FOR_WEIGHT_SEARCH = "MCC"

# Threshold optimization
THRESH_GRID = np.linspace(0.30, 0.70, 81)
OPTIMIZE_THRESHOLD_FOR = "MCC"

OUT_DIR = os.path.join(DATA_ROOT, "AIGQFUSION_BALANCED_PCA4to12_outputs")
os.makedirs(OUT_DIR, exist_ok=True)

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

# ============================================================
# SCALE TO [0, pi]
# ============================================================
def scale_to_pi(Xtr, Xte):
    scaler = MinMaxScaler(feature_range=(0.0, np.pi))
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)
    return Xtr_s, Xte_s

# ============================================================
# ADAPTIVE INTERACTION GRAPH
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
# STATEVECTOR KERNELS
# ============================================================
_SV_CACHE = {}

def _hash_X(X: np.ndarray) -> int:
    b = X.tobytes()
    head = b[:4096]
    tail = b[-4096:] if len(b) > 4096 else b
    return hash((X.shape, X.dtype.str, head, tail))

def statevectors(X_, map_name, reps, interaction_edges=None):
    key = (map_name, reps, tuple(interaction_edges) if interaction_edges is not None else None, _hash_X(X_))
    if key in _SV_CACHE:
        return _SV_CACHE[key]

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

    S = np.vstack(svs)
    _SV_CACHE[key] = S
    return S

def kernel_train_test(XA, XB, map_name, reps, interaction_edges=None):
    SA = statevectors(XA, map_name, reps, interaction_edges)
    SB = statevectors(XB, map_name, reps, interaction_edges)
    K_AA = np.abs(SA @ np.conjugate(SA).T) ** 2
    K_BA = np.abs(SB @ np.conjugate(SA).T) ** 2
    return K_AA, K_BA

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
# OPTIMIZATION HELPERS
# ============================================================
def normalize_positive_weights(w):
    w = np.asarray(w, dtype=float)
    w = np.maximum(w, 1e-10)
    return w / np.sum(w)

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
        grid = np.linspace(0.3, 0.7, 81)

    best_t = 0.5
    best_score = -1e18
    for t in grid:
        score = metric_from_probs(y_true, y_prob, metric_name=metric_name, threshold=t)
        if score > best_score:
            best_score = score
            best_t = float(t)
    return best_t, best_score

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

def optimize_kernel_weights_from_oof(ytr, oof_probs_by_branch, metric_name="MCC", seed=0):
    candidates = random_simplex_weights(
        n_models=oof_probs_by_branch.shape[1],
        n_samples=N_WEIGHT_SAMPLES,
        seed=seed
    )

    best_w = None
    best_metric = -1e18
    best_threshold = 0.5

    for w in candidates:
        p = oof_probs_by_branch @ w
        t, _ = optimize_threshold(ytr, p, metric_name=metric_name, grid=THRESH_GRID)
        score = metric_from_probs(ytr, p, metric_name=metric_name, threshold=t)
        if score > best_metric:
            best_metric = score
            best_w = w.copy()
            best_threshold = t

    return best_w, best_metric, best_threshold

# ============================================================
# BRANCH TRAINING
# ============================================================
def train_branch(K_tr, y_tr, K_te, seed):
    clf = SVC(kernel="precomputed", C=C_SVC, probability=True, random_state=seed)
    clf.fit(K_tr, y_tr)
    prob = clf.predict_proba(K_te)[:, 1]
    dec = clf.decision_function(K_te)
    return prob, dec

# ============================================================
# ONE-SEED AIG-QFUSION
# ============================================================
def aig_qfusion_predict_seed(Xtr, ytr, Xte, seed, interaction_edges):
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    fold_indices = list(skf.split(Xtr, ytr))

    base_names = ["Z", "ZZ", "INT"]
    reps_by_name = {"Z": REPS_Z, "ZZ": REPS_ZZ, "INT": REPS_INT}
    M = len(base_names)

    oof_prob = np.zeros((len(Xtr), M), dtype=float)
    oof_dec = np.zeros((len(Xtr), M), dtype=float)

    for idx_tr, idx_va in fold_indices:
        X_fold_tr, y_fold_tr = Xtr[idx_tr], ytr[idx_tr]
        X_fold_va = Xtr[idx_va]

        for m, name in enumerate(base_names):
            reps = reps_by_name[name]
            K_tr, K_va = kernel_train_test(
                X_fold_tr, X_fold_va,
                map_name=name,
                reps=reps,
                interaction_edges=interaction_edges if name == "INT" else None
            )
            p, d = train_branch(K_tr, y_fold_tr, K_va, seed)
            oof_prob[idx_va, m] = p
            oof_dec[idx_va, m] = d

    kernel_weights, kernel_weight_score, kernel_weight_thresh = optimize_kernel_weights_from_oof(
        ytr=ytr,
        oof_probs_by_branch=oof_prob,
        metric_name=METRIC_FOR_WEIGHT_SEARCH,
        seed=seed
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
                X_fold_tr, X_fold_va,
                map_name=name,
                reps=reps,
                interaction_edges=interaction_edges if name == "INT" else None
            )
            if K_tr_fused is None:
                K_tr_fused = w * K_tr
                K_va_fused = w * K_va
            else:
                K_tr_fused += w * K_tr
                K_va_fused += w * K_va

        p, d = train_branch(K_tr_fused, y_fold_tr, K_va_fused, seed)
        oof_fused_prob[idx_va, 0] = p
        oof_fused_dec[idx_va, 0] = d

    Z_train_meta = np.hstack([oof_prob, oof_dec, oof_fused_prob, oof_fused_dec])

    meta = LogisticRegression(max_iter=META_MAXIT, random_state=seed)
    meta.fit(Z_train_meta, ytr)
    oof_meta_prob = meta.predict_proba(Z_train_meta)[:, 1]

    best_meta_threshold, best_meta_score = optimize_threshold(
        ytr, oof_meta_prob,
        metric_name=OPTIMIZE_THRESHOLD_FOR,
        grid=THRESH_GRID
    )

    Z_test_prob = np.zeros((len(Xte), M), dtype=float)
    Z_test_dec = np.zeros((len(Xte), M), dtype=float)

    for m, name in enumerate(base_names):
        reps = reps_by_name[name]
        K_tr_full, K_te = kernel_train_test(
            Xtr, Xte,
            map_name=name,
            reps=reps,
            interaction_edges=interaction_edges if name == "INT" else None
        )
        p, d = train_branch(K_tr_full, ytr, K_te, seed)
        Z_test_prob[:, m] = p
        Z_test_dec[:, m] = d

    K_tr_full_fused = None
    K_te_fused = None
    for w, name in zip(kernel_weights, base_names):
        reps = reps_by_name[name]
        K_tr_full, K_te = kernel_train_test(
            Xtr, Xte,
            map_name=name,
            reps=reps,
            interaction_edges=interaction_edges if name == "INT" else None
        )
        if K_tr_full_fused is None:
            K_tr_full_fused = w * K_tr_full
            K_te_fused = w * K_te
        else:
            K_tr_full_fused += w * K_tr_full
            K_te_fused += w * K_te

    fused_p, fused_d = train_branch(K_tr_full_fused, ytr, K_te_fused, seed)

    Z_test_meta = np.hstack([
        Z_test_prob,
        Z_test_dec,
        fused_p.reshape(-1, 1),
        fused_d.reshape(-1, 1)
    ])

    y_prob = meta.predict_proba(Z_test_meta)[:, 1]

    details = {
        "kernel_weights": kernel_weights,
        "kernel_weight_score": kernel_weight_score,
        "kernel_weight_thresh": kernel_weight_thresh,
        "meta_threshold": best_meta_threshold,
        "meta_threshold_score": best_meta_score,
    }

    return y_prob, details

# ============================================================
# RUN ALL PCA
# ============================================================
raw_rows = []

for k in PCA_LIST:
    _SV_CACHE.clear()

    print("\n" + "=" * 100)
    print(f"RUNNING AIG-QFUSION | {DS.upper()} | PCA-{k}")
    print("=" * 100)

    Xtr_raw, ytr, Xte_raw, yte = load_train_test(DATA_ROOT, DS, k)
    Xtr, Xte = scale_to_pi(Xtr_raw, Xte_raw)

    interaction_edges = build_interaction_graph(
        Xtr,
        top_k=TOP_K_INTERACTIONS,
        use_skip_edges=USE_SKIP_EDGES
    )

    for seed in SEEDS:
        t0 = time.time()
        y_prob, details = aig_qfusion_predict_seed(Xtr, ytr, Xte, seed=seed, interaction_edges=interaction_edges)
        t1 = time.time()

        threshold = details["meta_threshold"]
        m = compute_metrics(yte, y_prob, threshold=threshold)

        m.update({
            "Dataset": DS,
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
            "Best_reps": float(REPS_INT),   # if needed, storing representative value
            "Best_Csvc": float(C_SVC),
            "Best_Cmeta": float(META_MAXIT),
            "TotalTime_s": float(t1 - t0),
        })
        raw_rows.append(m)

        print(
            f"seed={seed} | "
            f"Acc={m['Accuracy']:.4f} | "
            f"F1={m['F1']:.4f} | "
            f"MCC={m['MCC']:.4f} | "
            f"ROC_AUC={m['ROC_AUC']:.4f} | "
            f"Time={m['TotalTime_s']:.2f}s"
        )

raw_df = pd.DataFrame(raw_rows)

raw_csv = os.path.join(OUT_DIR, "raw_all_seeds_aigqfusion_balanced_pca4to12.csv")
raw_df.to_csv(raw_csv, index=False)

# ============================================================
# MEAN / STD / MEAN±STD
# ============================================================
group_cols = ["Dataset", "PCA"]
ignore_cols = group_cols + ["Seed", "TrainN", "TestN", "TP", "TN", "FP", "FN"]
metric_cols = [c for c in raw_df.columns if c not in ignore_cols]

mean_df = raw_df.groupby(group_cols)[metric_cols].mean().reset_index()
std_df = raw_df.groupby(group_cols)[metric_cols].std().reset_index()

mean_csv = os.path.join(OUT_DIR, "mean_aigqfusion_balanced_pca4to12.csv")
std_csv = os.path.join(OUT_DIR, "std_aigqfusion_balanced_pca4to12.csv")

mean_df.to_csv(mean_csv, index=False)
std_df.to_csv(std_csv, index=False)

pm_df = mean_df.copy()
for c in metric_cols:
    pm_df[c] = mean_df[c].map(lambda x: f"{x:.4f}") + " ± " + std_df[c].map(lambda x: f"{x:.4f}")

pm_csv = os.path.join(OUT_DIR, "meanpmstd_aigqfusion_balanced_pca4to12.csv")
pm_df.to_csv(pm_csv, index=False)

# ============================================================
# DISPLAY TABLE + HIGHLIGHT BEST CELLS ON MEAN VALUES
# ============================================================
cols_show = [
    "Dataset", "PCA",
    "Accuracy", "Precision", "Recall", "F1", "MCC",
    "ROC_AUC", "PR_AUC", "Brier", "ECE",
    "TotalTime_s", "Best_reps", "Best_Csvc", "Best_Cmeta"
]

disp_pm_df = pm_df[cols_show].copy()      # formatted strings: mean ± std
disp_mean_df = mean_df[cols_show].copy()  # numeric values for highlighting

higher_better = [
    "Accuracy", "Precision", "Recall", "F1", "MCC", "ROC_AUC", "PR_AUC"
]

lower_better = [
    "Brier", "ECE", "TotalTime_s"
]

# if you want to highlight these hyperparameter columns too, uncomment below:
# lower_better += ["Best_reps", "Best_Csvc", "Best_Cmeta"]

HILITE = "background-color: #d9ead3"

def highlight_best_cells(data):
    styles = pd.DataFrame("", index=data.index, columns=data.columns)

    for c in higher_better:
        best = disp_mean_df[c].max()
        styles.loc[np.isclose(disp_mean_df[c], best), c] = HILITE

    for c in lower_better:
        best = disp_mean_df[c].min()
        styles.loc[np.isclose(disp_mean_df[c], best), c] = HILITE

    return styles

sty = (
    disp_pm_df.style
    .set_properties(**{
        "text-align": "center",
        "border": "1px solid black",
        "font-size": "11pt",
        "font-family": "Times New Roman"
    })
    .set_table_styles([
        {"selector": "th", "props": [
            ("text-align", "center"),
            ("font-weight", "bold"),
            ("border", "1px solid black"),
            ("background-color", "#f2f2f2")
        ]},
        {"selector": "td", "props": [
            ("text-align", "center"),
            ("border", "1px solid black")
        ]},
        {"selector": "table", "props": [
            ("border-collapse", "collapse"),
            ("width", "100%")
        ]}
    ])
    .apply(highlight_best_cells, axis=None)
)

print("\n=== MEAN ± STD TABLE: AIG-QFusion | Balanced | PCA 4..12 ===\n")
print(disp_pm_df.to_string(index=False))

try:
    import openpyxl
    out_xlsx = os.path.join(OUT_DIR, "highlighted_bestcells_meanpmstd_aigqfusion_balanced_pca4to12.xlsx")
    sty.to_excel(out_xlsx, engine="openpyxl", index=False)
    print("\nHighlighted Excel saved:", out_xlsx)
except Exception as e:
    print("\nExcel export skipped:", e)

print("\nSaved files:")
print(raw_csv)
print(mean_csv)
print(std_csv)
print(pm_csv)
print("\nDone.")


# In[2]:


# ============================================================
# DISPLAY COLORED TABLE AS MEAN ± STD + HIGHLIGHT BEST CELLS
# ============================================================
from IPython.display import display

cols_show = [
    "Dataset", "PCA",
    "Accuracy", "Precision", "Recall", "F1", "MCC",
    "ROC_AUC", "PR_AUC", "Brier", "ECE",
    "TotalTime_s", "Best_reps", "Best_Csvc", "Best_Cmeta"
]

disp_pm_df = pm_df[cols_show].copy()
disp_mean_df = mean_df[cols_show].copy()

higher_better = [
    "Accuracy", "Precision", "Recall", "F1", "MCC",
    "ROC_AUC", "PR_AUC"
]

lower_better = [
    "Brier", "ECE", "TotalTime_s"
]

# uncomment if you want these also highlighted
# lower_better += ["Best_reps", "Best_Csvc", "Best_Cmeta"]

BEST_COLOR = "background-color: #d9ead3;"
BORDER = "1px solid black"

def highlight_best_cells(_):
    styles = pd.DataFrame("", index=disp_pm_df.index, columns=disp_pm_df.columns)

    for c in higher_better:
        best = disp_mean_df[c].max()
        styles.loc[np.isclose(disp_mean_df[c], best), c] = BEST_COLOR

    for c in lower_better:
        best = disp_mean_df[c].min()
        styles.loc[np.isclose(disp_mean_df[c], best), c] = BEST_COLOR

    return styles

sty = (
    disp_pm_df.style
    .apply(highlight_best_cells, axis=None)
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
        },
        {
            "selector": ".row_heading",
            "props": [
                ("border", BORDER),
                ("text-align", "center"),
                ("background-color", "#f2f2f2"),
                ("font-weight", "bold")
            ]
        },
        {
            "selector": ".blank",
            "props": [
                ("border", BORDER),
                ("background-color", "#f2f2f2")
            ]
        }
    ])
)

print("\n=== MEAN ± STD TABLE: AIG-QFusion | Balanced | PCA 4..12 ===\n")

try:
    display(sty)
except Exception:
    print(disp_pm_df.to_string(index=False))

try:
    import openpyxl
    out_xlsx = os.path.join(
        OUT_DIR,
        "highlighted_bestcells_meanpmstd_aigqfusion_balanced_pca4to12.xlsx"
    )
    sty.to_excel(out_xlsx, engine="openpyxl", index=True)
    print("\nHighlighted Excel saved:", out_xlsx)
except Exception as e:
    print("\nExcel export skipped:", e)

try:
    html_path = os.path.join(
        OUT_DIR,
        "highlighted_bestcells_meanpmstd_aigqfusion_balanced_pca4to12.html"
    )
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(sty.to_html())
    print("Styled HTML saved:", html_path)
except Exception as e:
    print("HTML export skipped:", e)


# In[ ]:





# In[ ]:


# FIniding time 


# In[3]:


# ============================================================
# AIG-QFusion (Quantum) — BALANCED | PCA 4..12
# Compact evaluation version for Section 4.2
# Final output:
#   1) raw all-seed metrics
#   2) mean table
#   3) std table
#   4) mean ± std table
#   5) highlighted Excel with best cells
#   6) styled HTML table
#   7) TrainTime_s, TestTime_s, TotalTime_s
# ============================================================

import os
import time
import warnings
import numpy as np
import pandas as pd

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

PCA_LIST = list(range(4, 13))   # 4..12
SEEDS = [0, 1, 2, 3, 4]
N_SPLITS = 3

# Quantum / model config
REPS_Z = 2
REPS_ZZ = 2
REPS_INT = 3
C_SVC = 2.0
META_MAXIT = 4000

# Adaptive graph
TOP_K_INTERACTIONS = 8
USE_SKIP_EDGES = True
INTERACTION_SCALE = 1.25

# Weight optimization
N_WEIGHT_SAMPLES = 160
METRIC_FOR_WEIGHT_SEARCH = "MCC"

# Threshold optimization
THRESH_GRID = np.linspace(0.30, 0.70, 81)
OPTIMIZE_THRESHOLD_FOR = "MCC"

OUT_DIR = os.path.join(DATA_ROOT, "AIGQFUSION_BALANCED_PCA4to12_outputs")
os.makedirs(OUT_DIR, exist_ok=True)

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

# ============================================================
# SCALE TO [0, pi]
# ============================================================
def scale_to_pi(Xtr, Xte):
    scaler = MinMaxScaler(feature_range=(0.0, np.pi))
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)
    return Xtr_s, Xte_s

# ============================================================
# ADAPTIVE INTERACTION GRAPH
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
# STATEVECTOR KERNELS
# ============================================================
_SV_CACHE = {}

def _hash_X(X: np.ndarray) -> int:
    b = X.tobytes()
    head = b[:4096]
    tail = b[-4096:] if len(b) > 4096 else b
    return hash((X.shape, X.dtype.str, head, tail))

def statevectors(X_, map_name, reps, interaction_edges=None):
    key = (map_name, reps, tuple(interaction_edges) if interaction_edges is not None else None, _hash_X(X_))
    if key in _SV_CACHE:
        return _SV_CACHE[key]

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

    S = np.vstack(svs)
    _SV_CACHE[key] = S
    return S

def kernel_train_test(XA, XB, map_name, reps, interaction_edges=None):
    SA = statevectors(XA, map_name, reps, interaction_edges)
    SB = statevectors(XB, map_name, reps, interaction_edges)
    K_AA = np.abs(SA @ np.conjugate(SA).T) ** 2
    K_BA = np.abs(SB @ np.conjugate(SA).T) ** 2
    return K_AA, K_BA

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
# OPTIMIZATION HELPERS
# ============================================================
def normalize_positive_weights(w):
    w = np.asarray(w, dtype=float)
    w = np.maximum(w, 1e-10)
    return w / np.sum(w)

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
        grid = np.linspace(0.3, 0.7, 81)

    best_t = 0.5
    best_score = -1e18
    for t in grid:
        score = metric_from_probs(y_true, y_prob, metric_name=metric_name, threshold=t)
        if score > best_score:
            best_score = score
            best_t = float(t)
    return best_t, best_score

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

def optimize_kernel_weights_from_oof(ytr, oof_probs_by_branch, metric_name="MCC", seed=0):
    candidates = random_simplex_weights(
        n_models=oof_probs_by_branch.shape[1],
        n_samples=N_WEIGHT_SAMPLES,
        seed=seed
    )

    best_w = None
    best_metric = -1e18
    best_threshold = 0.5

    for w in candidates:
        p = oof_probs_by_branch @ w
        t, _ = optimize_threshold(ytr, p, metric_name=metric_name, grid=THRESH_GRID)
        score = metric_from_probs(ytr, p, metric_name=metric_name, threshold=t)
        if score > best_metric:
            best_metric = score
            best_w = w.copy()
            best_threshold = t

    return best_w, best_metric, best_threshold

# ============================================================
# BRANCH TRAINING
# ============================================================
def train_branch(K_tr, y_tr, K_te, seed, return_time=False):
    t_fit0 = time.time()
    clf = SVC(kernel="precomputed", C=C_SVC, probability=True, random_state=seed)
    clf.fit(K_tr, y_tr)
    t_fit1 = time.time()

    t_pred0 = time.time()
    prob = clf.predict_proba(K_te)[:, 1]
    dec = clf.decision_function(K_te)
    t_pred1 = time.time()

    if return_time:
        return prob, dec, (t_fit1 - t_fit0), (t_pred1 - t_pred0)
    return prob, dec

# ============================================================
# ONE-SEED AIG-QFUSION
# ============================================================
def aig_qfusion_predict_seed(Xtr, ytr, Xte, seed, interaction_edges):
    total_train_time = 0.0
    total_test_time = 0.0

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    fold_indices = list(skf.split(Xtr, ytr))

    base_names = ["Z", "ZZ", "INT"]
    reps_by_name = {"Z": REPS_Z, "ZZ": REPS_ZZ, "INT": REPS_INT}
    M = len(base_names)

    oof_prob = np.zeros((len(Xtr), M), dtype=float)
    oof_dec = np.zeros((len(Xtr), M), dtype=float)

    # ------------------------------------------------------------
    # Stage 1: OOF branch models
    # ------------------------------------------------------------
    for idx_tr, idx_va in fold_indices:
        X_fold_tr, y_fold_tr = Xtr[idx_tr], ytr[idx_tr]
        X_fold_va = Xtr[idx_va]

        for m, name in enumerate(base_names):
            reps = reps_by_name[name]

            t_k0 = time.time()
            K_tr, K_va = kernel_train_test(
                X_fold_tr, X_fold_va,
                map_name=name,
                reps=reps,
                interaction_edges=interaction_edges if name == "INT" else None
            )
            t_k1 = time.time()

            p, d, fit_t, pred_t = train_branch(K_tr, y_fold_tr, K_va, seed, return_time=True)

            total_train_time += (t_k1 - t_k0) + fit_t
            total_test_time += pred_t

            oof_prob[idx_va, m] = p
            oof_dec[idx_va, m] = d

    # ------------------------------------------------------------
    # Stage 2: optimize kernel weights
    # ------------------------------------------------------------
    t_w0 = time.time()
    kernel_weights, kernel_weight_score, kernel_weight_thresh = optimize_kernel_weights_from_oof(
        ytr=ytr,
        oof_probs_by_branch=oof_prob,
        metric_name=METRIC_FOR_WEIGHT_SEARCH,
        seed=seed
    )
    t_w1 = time.time()
    total_train_time += (t_w1 - t_w0)

    oof_fused_prob = np.zeros((len(Xtr), 1), dtype=float)
    oof_fused_dec = np.zeros((len(Xtr), 1), dtype=float)

    # ------------------------------------------------------------
    # Stage 3: OOF fused kernel
    # ------------------------------------------------------------
    for idx_tr, idx_va in fold_indices:
        X_fold_tr, y_fold_tr = Xtr[idx_tr], ytr[idx_tr]
        X_fold_va = Xtr[idx_va]

        K_tr_fused = None
        K_va_fused = None

        t_k0 = time.time()
        for w, name in zip(kernel_weights, base_names):
            reps = reps_by_name[name]
            K_tr, K_va = kernel_train_test(
                X_fold_tr, X_fold_va,
                map_name=name,
                reps=reps,
                interaction_edges=interaction_edges if name == "INT" else None
            )
            if K_tr_fused is None:
                K_tr_fused = w * K_tr
                K_va_fused = w * K_va
            else:
                K_tr_fused += w * K_tr
                K_va_fused += w * K_va
        t_k1 = time.time()

        p, d, fit_t, pred_t = train_branch(K_tr_fused, y_fold_tr, K_va_fused, seed, return_time=True)

        total_train_time += (t_k1 - t_k0) + fit_t
        total_test_time += pred_t

        oof_fused_prob[idx_va, 0] = p
        oof_fused_dec[idx_va, 0] = d

    # ------------------------------------------------------------
    # Stage 4: meta-model training
    # ------------------------------------------------------------
    Z_train_meta = np.hstack([oof_prob, oof_dec, oof_fused_prob, oof_fused_dec])

    t_meta0 = time.time()
    meta = LogisticRegression(max_iter=META_MAXIT, random_state=seed)
    meta.fit(Z_train_meta, ytr)
    t_meta1 = time.time()
    total_train_time += (t_meta1 - t_meta0)

    t_meta_pred0 = time.time()
    oof_meta_prob = meta.predict_proba(Z_train_meta)[:, 1]
    t_meta_pred1 = time.time()
    total_test_time += (t_meta_pred1 - t_meta_pred0)

    t_thr0 = time.time()
    best_meta_threshold, best_meta_score = optimize_threshold(
        ytr, oof_meta_prob,
        metric_name=OPTIMIZE_THRESHOLD_FOR,
        grid=THRESH_GRID
    )
    t_thr1 = time.time()
    total_train_time += (t_thr1 - t_thr0)

    # ------------------------------------------------------------
    # Stage 5: final branch models on full training set -> test set
    # ------------------------------------------------------------
    Z_test_prob = np.zeros((len(Xte), M), dtype=float)
    Z_test_dec = np.zeros((len(Xte), M), dtype=float)

    for m, name in enumerate(base_names):
        reps = reps_by_name[name]

        t_k0 = time.time()
        K_tr_full, K_te = kernel_train_test(
            Xtr, Xte,
            map_name=name,
            reps=reps,
            interaction_edges=interaction_edges if name == "INT" else None
        )
        t_k1 = time.time()

        p, d, fit_t, pred_t = train_branch(K_tr_full, ytr, K_te, seed, return_time=True)

        total_train_time += (t_k1 - t_k0) + fit_t
        total_test_time += pred_t

        Z_test_prob[:, m] = p
        Z_test_dec[:, m] = d

    # ------------------------------------------------------------
    # Stage 6: final fused kernel on full training set -> test set
    # ------------------------------------------------------------
    K_tr_full_fused = None
    K_te_fused = None

    t_k0 = time.time()
    for w, name in zip(kernel_weights, base_names):
        reps = reps_by_name[name]
        K_tr_full, K_te = kernel_train_test(
            Xtr, Xte,
            map_name=name,
            reps=reps,
            interaction_edges=interaction_edges if name == "INT" else None
        )
        if K_tr_full_fused is None:
            K_tr_full_fused = w * K_tr_full
            K_te_fused = w * K_te
        else:
            K_tr_full_fused += w * K_tr_full
            K_te_fused += w * K_te
    t_k1 = time.time()

    fused_p, fused_d, fit_t, pred_t = train_branch(
        K_tr_full_fused, ytr, K_te_fused, seed, return_time=True
    )

    total_train_time += (t_k1 - t_k0) + fit_t
    total_test_time += pred_t

    # ------------------------------------------------------------
    # Stage 7: final meta prediction on test set
    # ------------------------------------------------------------
    Z_test_meta = np.hstack([
        Z_test_prob,
        Z_test_dec,
        fused_p.reshape(-1, 1),
        fused_d.reshape(-1, 1)
    ])

    t_meta_pred0 = time.time()
    y_prob = meta.predict_proba(Z_test_meta)[:, 1]
    t_meta_pred1 = time.time()
    total_test_time += (t_meta_pred1 - t_meta_pred0)

    details = {
        "kernel_weights": kernel_weights,
        "kernel_weight_score": kernel_weight_score,
        "kernel_weight_thresh": kernel_weight_thresh,
        "meta_threshold": best_meta_threshold,
        "meta_threshold_score": best_meta_score,
        "TrainTime_s": float(total_train_time),
        "TestTime_s": float(total_test_time),
        "TotalTime_s": float(total_train_time + total_test_time),
    }

    return y_prob, details

# ============================================================
# RUN ALL PCA
# ============================================================
raw_rows = []

for k in PCA_LIST:
    _SV_CACHE.clear()

    print("\n" + "=" * 100)
    print(f"RUNNING AIG-QFUSION | {DS.upper()} | PCA-{k}")
    print("=" * 100)

    Xtr_raw, ytr, Xte_raw, yte = load_train_test(DATA_ROOT, DS, k)
    Xtr, Xte = scale_to_pi(Xtr_raw, Xte_raw)

    interaction_edges = build_interaction_graph(
        Xtr,
        top_k=TOP_K_INTERACTIONS,
        use_skip_edges=USE_SKIP_EDGES
    )

    for seed in SEEDS:
        y_prob, details = aig_qfusion_predict_seed(
            Xtr, ytr, Xte, seed=seed, interaction_edges=interaction_edges
        )

        threshold = details["meta_threshold"]
        m = compute_metrics(yte, y_prob, threshold=threshold)

        m.update({
            "Dataset": DS,
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
            "Best_reps": float(REPS_INT),
            "Best_Csvc": float(C_SVC),
            "Best_Cmeta": float(META_MAXIT),
            "TrainTime_s": float(details["TrainTime_s"]),
            "TestTime_s": float(details["TestTime_s"]),
            "TotalTime_s": float(details["TotalTime_s"]),
        })
        raw_rows.append(m)

        print(
            f"seed={seed} | "
            f"Acc={m['Accuracy']:.4f} | "
            f"F1={m['F1']:.4f} | "
            f"MCC={m['MCC']:.4f} | "
            f"ROC_AUC={m['ROC_AUC']:.4f} | "
            f"Train={m['TrainTime_s']:.2f}s | "
            f"Test={m['TestTime_s']:.2f}s | "
            f"Total={m['TotalTime_s']:.2f}s"
        )

raw_df = pd.DataFrame(raw_rows)

raw_csv = os.path.join(OUT_DIR, "raw_all_seeds_aigqfusion_balanced_pca4to12.csv")
raw_df.to_csv(raw_csv, index=False)

# ============================================================
# MEAN / STD / MEAN±STD
# ============================================================
group_cols = ["Dataset", "PCA"]
ignore_cols = group_cols + ["Seed", "TrainN", "TestN", "TP", "TN", "FP", "FN"]
metric_cols = [c for c in raw_df.columns if c not in ignore_cols]

mean_df = raw_df.groupby(group_cols)[metric_cols].mean().reset_index()
std_df = raw_df.groupby(group_cols)[metric_cols].std().reset_index()

mean_csv = os.path.join(OUT_DIR, "mean_aigqfusion_balanced_pca4to12.csv")
std_csv = os.path.join(OUT_DIR, "std_aigqfusion_balanced_pca4to12.csv")

mean_df.to_csv(mean_csv, index=False)
std_df.to_csv(std_csv, index=False)

pm_df = mean_df.copy()
for c in metric_cols:
    pm_df[c] = mean_df[c].map(lambda x: f"{x:.4f}") + " ± " + std_df[c].map(lambda x: f"{x:.4f}")

pm_csv = os.path.join(OUT_DIR, "meanpmstd_aigqfusion_balanced_pca4to12.csv")
pm_df.to_csv(pm_csv, index=False)

# ============================================================
# DISPLAY COLORED TABLE AS MEAN ± STD + HIGHLIGHT BEST CELLS
# ============================================================
cols_show = [
    "Dataset", "PCA",
    "Accuracy", "Precision", "Recall", "F1", "MCC",
    "ROC_AUC", "PR_AUC", "Brier", "ECE",
    "TrainTime_s", "TestTime_s", "TotalTime_s",
    "Best_reps", "Best_Csvc", "Best_Cmeta"
]

disp_pm_df = pm_df[cols_show].copy()
disp_mean_df = mean_df[cols_show].copy()

higher_better = [
    "Accuracy", "Precision", "Recall", "F1", "MCC",
    "ROC_AUC", "PR_AUC"
]

lower_better = [
    "Brier", "ECE",
    "TrainTime_s", "TestTime_s", "TotalTime_s"
]

# Uncomment below if you also want these hyperparameter columns highlighted
# lower_better += ["Best_reps", "Best_Csvc", "Best_Cmeta"]

BEST_COLOR = "background-color: #d9ead3;"
BORDER = "1px solid black"

def highlight_best_cells(_):
    styles = pd.DataFrame("", index=disp_pm_df.index, columns=disp_pm_df.columns)

    for c in higher_better:
        best = disp_mean_df[c].max()
        styles.loc[np.isclose(disp_mean_df[c], best), c] = BEST_COLOR

    for c in lower_better:
        best = disp_mean_df[c].min()
        styles.loc[np.isclose(disp_mean_df[c], best), c] = BEST_COLOR

    return styles

sty = (
    disp_pm_df.style
    .apply(highlight_best_cells, axis=None)
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
        },
        {
            "selector": ".row_heading",
            "props": [
                ("border", BORDER),
                ("text-align", "center"),
                ("background-color", "#f2f2f2"),
                ("font-weight", "bold")
            ]
        },
        {
            "selector": ".blank",
            "props": [
                ("border", BORDER),
                ("background-color", "#f2f2f2")
            ]
        }
    ])
)

print("\n=== MEAN ± STD TABLE: AIG-QFusion | Balanced | PCA 4..12 ===\n")

# Notebook/Jupyter display if available, else plain text
try:
    from IPython.display import display
    display(sty)
except Exception:
    print(disp_pm_df.to_string(index=False))

# Save styled Excel
try:
    import openpyxl
    out_xlsx = os.path.join(
        OUT_DIR,
        "highlighted_bestcells_meanpmstd_aigqfusion_balanced_pca4to12.xlsx"
    )
    sty.to_excel(out_xlsx, engine="openpyxl", index=True)
    print("\nHighlighted Excel saved:", out_xlsx)
except Exception as e:
    print("\nExcel export skipped:", e)

# Save styled HTML
try:
    html_path = os.path.join(
        OUT_DIR,
        "highlighted_bestcells_meanpmstd_aigqfusion_balanced_pca4to12.html"
    )
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(sty.to_html())
    print("Styled HTML saved:", html_path)
except Exception as e:
    print("HTML export skipped:", e)

print("\nSaved files:")
print(raw_csv)
print(mean_csv)
print(std_csv)
print(pm_csv)
print("\nDone.")


# In[4]:


OUT_DIR = os.path.join(DATA_ROOT, "AIGQFUSION_BALANCED_PCA4to12_outputs")
os.makedirs(OUT_DIR, exist_ok=True)

print("Saving all outputs to:")
print(OUT_DIR)


# In[5]:


# ============================================================
# OUTPUT DIRECTORY
# ============================================================
OUT_DIR = r"C:\Users\shanmugam\shan work 1\New folder\new try\output"
os.makedirs(OUT_DIR, exist_ok=True)

print("Results will be saved in:")
print(OUT_DIR)

