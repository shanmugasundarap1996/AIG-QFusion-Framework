#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import time
import warnings
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
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

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

warnings.filterwarnings("ignore")

# ============================================================
# OPTIONAL LIBRARIES
# ============================================================
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

# ============================================================
# USER CONFIG
# ============================================================
DATA_ROOT = r"C:\Users\shanmugam\shan work 1\data\dataset_saved_500_300_pca4to12"
DS = "balanced"
LABEL_COL = "CLNSIG"

PCA_LIST = [12]   # PCA-12 ONLY
SEEDS = [0, 1, 2, 3, 4]
THRESH = 0.5

OUT_DIR = r"C:\Users\shanmugam\shan work 1\New folder\new try\output"
os.makedirs(OUT_DIR, exist_ok=True)

print("Results will be saved in:")
print(OUT_DIR)

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
# MODEL FACTORY
# ============================================================
def make_models(seed):
    models = {}

    models["DecisionTree"] = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", DecisionTreeClassifier(
            max_depth=None,
            random_state=seed
        ))
    ])

    models["RandomForest"] = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(
            n_estimators=300,
            random_state=seed,
            n_jobs=-1
        ))
    ])

    models["GradientBoost"] = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=seed
        ))
    ])

    models["KNN"] = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier(
            n_neighbors=7,
            weights="distance"
        ))
    ])

    models["SVM"] = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", SVC(
            C=2.0,
            kernel="rbf",
            probability=True,
            random_state=seed
        ))
    ])

    models["MLP"] = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            batch_size="auto",
            learning_rate_init=1e-3,
            max_iter=1000,
            early_stopping=True,
            random_state=seed
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
                n_jobs=-1
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
            random_seed=seed
        )

    return models

# ============================================================
# TRAIN + PREDICT WITH TIME SPLIT
# ============================================================
def fit_predict_prob_with_time(model, Xtr, ytr, Xte, algo_name=None):
    # Special handling for CatBoost
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

        return y_prob, (t1 - t0), (t3 - t2)

    # Default sklearn / pipeline flow
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

    return y_prob, (t1 - t0), (t3 - t2)

# ============================================================
# RUN ALL
# ============================================================
raw_rows = []

available_note = []
if not HAS_XGBOOST:
    available_note.append("XGBoost not installed -> skipped")
if not HAS_CATBOOST:
    available_note.append("CatBoost not installed -> skipped")

if available_note:
    print("\n".join(available_note))

for k in PCA_LIST:
    print("\n" + "=" * 100)
    print(f"RUNNING CLASSICAL BASELINES | {DS.upper()} | PCA-{k}")
    print("=" * 100)

    Xtr, ytr, Xte, yte = load_train_test(DATA_ROOT, DS, k)

    for seed in SEEDS:
        models = make_models(seed)

        for algo_name, model in models.items():
            y_prob, train_t, test_t = fit_predict_prob_with_time(
                model, Xtr, ytr, Xte, algo_name=algo_name
            )

            m = compute_metrics(yte, y_prob, threshold=THRESH)
            m.update({
                "Dataset": DS,
                "PCA": k,
                "Seed": seed,
                "Algorithm": algo_name,
                "TrainN": int(len(Xtr)),
                "TestN": int(len(Xte)),
                "TrainTime_s": float(train_t),
                "TestTime_s": float(test_t),
                "TotalTime_s": float(train_t + test_t),
            })
            raw_rows.append(m)

            print(
                f"{algo_name:>13} | seed={seed} | "
                f"Acc={m['Accuracy']:.4f} | "
                f"F1={m['F1']:.4f} | "
                f"MCC={m['MCC']:.4f} | "
                f"ROC_AUC={m['ROC_AUC']:.4f} | "
                f"Train={m['TrainTime_s']:.4f}s | "
                f"Test={m['TestTime_s']:.4f}s | "
                f"Total={m['TotalTime_s']:.4f}s"
            )

raw_df = pd.DataFrame(raw_rows)

raw_csv = os.path.join(OUT_DIR, "raw_all_seeds_ml_baselines_balanced_pca12.csv")
raw_df.to_csv(raw_csv, index=False)

# ============================================================
# MEAN / STD / MEAN±STD
# ============================================================
group_cols = ["Dataset", "PCA", "Algorithm"]
ignore_cols = group_cols + ["Seed", "TrainN", "TestN", "TP", "TN", "FP", "FN"]
metric_cols = [c for c in raw_df.columns if c not in ignore_cols]

mean_df = raw_df.groupby(group_cols)[metric_cols].mean().reset_index()
std_df = raw_df.groupby(group_cols)[metric_cols].std().reset_index()

mean_csv = os.path.join(OUT_DIR, "mean_ml_baselines_balanced_pca12.csv")
std_csv  = os.path.join(OUT_DIR, "std_ml_baselines_balanced_pca12.csv")

mean_df.to_csv(mean_csv, index=False)
std_df.to_csv(std_csv, index=False)

pm_df = mean_df.copy()
for c in metric_cols:
    pm_df[c] = mean_df[c].map(lambda x: f"{x:.4f}") + " ± " + std_df[c].map(lambda x: f"{x:.4f}")

pm_csv = os.path.join(OUT_DIR, "meanpmstd_ml_baselines_balanced_pca12.csv")
pm_df.to_csv(pm_csv, index=False)

# ============================================================
# DISPLAY TABLE + HIGHLIGHT BEST CELLS
# ============================================================
cols_show = [
    "Dataset", "PCA", "Algorithm",
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

print("\n=== MEAN ± STD TABLE: Classical ML Baselines | Balanced | PCA-12 ===\n")

try:
    from IPython.display import display
    display(sty)
except Exception:
    print(disp_pm_df.to_string(index=False))

try:
    import openpyxl
    out_xlsx = os.path.join(OUT_DIR, "highlighted_bestcells_meanpmstd_ml_baselines_balanced_pca12.xlsx")
    sty.to_excel(out_xlsx, engine="openpyxl", index=True)
    print("\nHighlighted Excel saved:", out_xlsx)
except Exception as e:
    print("\nExcel export skipped:", e)

try:
    html_path = os.path.join(OUT_DIR, "highlighted_bestcells_meanpmstd_ml_baselines_balanced_pca12.html")
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

