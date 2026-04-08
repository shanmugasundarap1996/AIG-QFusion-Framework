# 🧬 AIG-QFusion: Adaptive Interaction Graph Quantum Fusion for Genomic Variant Classification

## 📌 Overview
This repository provides the official implementation of AIG-QFusion, a hybrid quantum–classical framework for genomic variant classification.

## 🎯 Key Contributions
- Adaptive interaction-aware quantum encoding
- Multi-branch quantum kernel fusion (Z, ZZ, INT)
- Meta-learning with logistic regression
- Threshold optimization (MCC-based)

## 📊 Dataset
- ClinVar + dbNSFP
- Train: 500, Test: 300 (balanced)
- PCA: 4–12 dimensions

## ▶️ Usage
### Classical ML
python code/baseline_ml.py --data-root ./data --dataset balanced --pca 12 --out-dir ./outputs/classical

### QML
python code/baseline_qml.py --data-root ./data --dataset balanced --pca 12 --out-dir ./outputs/qml

### Proposed
python code/proposed_pca12.py --data-root ./data --dataset balanced --out-dir ./outputs/aigqfusion

## 📈 Results
- Accuracy: 0.992
- MCC: 0.984
- ROC-AUC: 0.9998
- PR-AUC: 0.9998

## 📊 Statistical Analysis
Includes:
- Paired t-test
- Wilcoxon test
- Cohen’s d

## 📜 Citation
@article{aig_qfusion_2026,
  title={AIG-QFusion},
  author={Shanmugasundaram P and Saroja S},
  year={2026}
}
