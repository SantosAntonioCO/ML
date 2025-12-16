#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
tabnet_threshold_analysis.py

- Carrega tabnet_best_model.zip
- Avalia TRAIN / VAL / TEST
- Sweep de thresholds
- Salva métricas completas em CSV
- Salva classification_report em TXT

python tabnet_threshold_analysis.py ^
  --model outputs_tabnet_v8\final_20251209T145950Z\tabnet_best_model.zip ^
  --train under_brain_insp\creditcard_train.csv ^
  --val under_brain_insp\creditcard_val.csv ^
  --test under_brain_insp\creditcard_test.csv ^
  --outdir outputs_tabnet_v8\threshold_analysis

python tabnet_threshold_analysis.py  --model outputs_tabnet_v8/final_20251209T145950Z/tabnet_best_model.zip  --train creditcard_train.csv --val creditcard_val.csv --test creditcard_test.csv --outdir outputs_tabnet_v8/threshold_analysis

"""

# ============================================================
# Imports
# ============================================================
import os
import json
import argparse
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    brier_score_loss,
    precision_recall_curve
)

from pytorch_tabnet.tab_model import TabNetClassifier

# ============================================================
# Utils
# ============================================================
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def detect_target_column(df):
    for c in ["class", "Class", "label", "Label", "y", "target"]:
        if c in df.columns:
            return c
    return df.columns[-1]

# ============================================================
# Full Metrics
# ============================================================
def compute_metrics_binary(y_true, y_prob, thr):
    y_pred = (y_prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    acc_class_0 = tn / (tn + fp) if (tn + fp) else 0.0
    acc_class_1 = tp / (tp + fn) if (tp + fn) else 0.0
    acc_macro   = (acc_class_0 + acc_class_1) / 2
    acc_micro   = (tp + tn) / (tp + tn + fp + fn)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0

    f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    f1_weight = f1

    roc_auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan
    avg_precision = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan

    pr_prec, pr_rec, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = np.trapz(pr_prec[::-1], pr_rec[::-1]) if len(pr_prec) > 1 else np.nan

    neg_brier = -brier_score_loss(y_true, y_prob)
    mcc = matthews_corrcoef(y_true, y_pred) if len(np.unique(y_true)) > 1 else np.nan

    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    gmean = np.sqrt(recall * specificity)
    frr = fn / (fn + tn) if (fn + tn) else np.nan

    return {
        "threshold": thr,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "acc_class_0": acc_class_0,
        "acc_class_1": acc_class_1,
        "acc_macro": acc_macro,
        "acc_micro": acc_micro,
        "balanced_acc": balanced_accuracy_score(y_true, y_pred),
        "f1": f1,
        "f1_weight": f1_weight,
        "precision": precision,
        "recall": recall,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "avg_precision": avg_precision,
        "neg_brier": neg_brier,
        "mcc": mcc,
        "gmean": gmean,
        "frr": frr
    }

# ============================================================
# Main
# ============================================================
def main(args):
    ensure_dir(args.outdir)

    # -------------------------
    # Load data
    # -------------------------
    df_tr = pd.read_csv(args.train)
    df_va = pd.read_csv(args.val)
    df_te = pd.read_csv(args.test)

    tcol = detect_target_column(df_tr)

    ytr = df_tr[tcol].values.astype(int)
    yva = df_va[tcol].values.astype(int)
    yte = df_te[tcol].values.astype(int)

    Xtr = df_tr.drop(columns=[tcol]).values
    Xva = df_va.drop(columns=[tcol]).values
    Xte = df_te.drop(columns=[tcol]).values

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(Xtr)
    Xva = scaler.transform(Xva)
    Xte = scaler.transform(Xte)

    # -------------------------
    # Load model
    # -------------------------
    clf = TabNetClassifier()
    clf.load_model(args.model)

    # -------------------------
    # Predict
    # -------------------------
    p_tr = clf.predict_proba(Xtr)[:, 1]
    p_va = clf.predict_proba(Xva)[:, 1]
    p_te = clf.predict_proba(Xte)[:, 1]

    # ========================================================
    # TRAIN — classification report (thr = 0.5)
    # ========================================================
    train_report = classification_report(
        ytr, (p_tr >= 0.5).astype(int),
        digits=4, zero_division=0
    )

    with open(os.path.join(args.outdir, "classification_report_train_thr_0.50.txt"), "w") as f:
        f.write(train_report)

    # ========================================================
    # TEST — threshold sweep
    # ========================================================
    rows = []
    reports_txt = []

    thresholds = np.linspace(args.thr_min, args.thr_max, args.thr_steps)

    for thr in thresholds:
        metrics = compute_metrics_binary(yte, p_te, thr)
        rows.append(metrics)

        rep = classification_report(
            yte, (p_te >= thr).astype(int),
            digits=4, zero_division=0
        )

        reports_txt.append(
            f"\n\n=== Threshold {thr:.3f} ===\n{rep}"
        )

    # CSV metrics
    df_metrics = pd.DataFrame(rows)
    df_metrics.to_csv(
        os.path.join(args.outdir, "metrics_by_threshold_test.csv"),
        index=False,
        float_format="%.6f"
    )

    # TXT classification reports
    with open(os.path.join(args.outdir, "classification_report_test_by_threshold.txt"), "w") as f:
        f.write("\n".join(reports_txt))

    print("✔ Análise finalizada")
    print("📁 Outputs em:", args.outdir)

# ============================================================
# Entry
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser("TabNet threshold analysis")
    parser.add_argument("--model", required=True, help="tabnet_best_model.zip")
    parser.add_argument("--train", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--outdir", default="tabnet_threshold_analysis")
    parser.add_argument("--thr_min", type=float, default=0.01)
    parser.add_argument("--thr_max", type=float, default=0.99)
    parser.add_argument("--thr_steps", type=int, default=99)
    args = parser.parse_args()
    main(args)
