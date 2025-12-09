#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
catboost_general_csv_benchmark_v1.py

Versão 1 — equivalente ao xgb_general_csv_benchmark_v3.py, porém adaptado para CatBoost.

Recursos:
- Hyperopt com attachments + CSV
- Reconstrução segura dos melhores parâmetros
- Treino final com métricas, plots, threshold-scan
- Compatível CatBoost >= 1.2
"""

import os
import time
import argparse
import json
from functools import partial
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, average_precision_score,
    precision_score, recall_score, f1_score,
    confusion_matrix, brier_score_loss, accuracy_score,
    balanced_accuracy_score, matthews_corrcoef
)
import matplotlib.pyplot as plt
import joblib

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from catboost import CatBoostClassifier

# =====================================================================
# Utils / métricas
# =====================================================================

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def compute_metrics_binary(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    acc_class_0 = tn / (tn + fp) if (tn + fp) else 0.0
    acc_class_1 = tp / (tp + fn) if (tp + fn) else 0.0
    acc_macro = (acc_class_0 + acc_class_1) / 2.0
    acc_micro = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)

    f1 = f1_score(y_true, y_pred, zero_division=0)
    f1_weight = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan
    avg_precision = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan

    neg_brier = -brier_score_loss(y_true, y_prob)
    mcc = matthews_corrcoef(y_true, y_pred) if len(np.unique(y_true)) > 1 else np.nan

    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    gmean = np.sqrt(recall * specificity) if (recall >= 0 and specificity >= 0) else np.nan
    frr = fn / (fn + tn) if (fn + tn) else np.nan

    return {
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "acc_class_0": acc_class_0, "acc_class_1": acc_class_1,
        "acc_macro": acc_macro, "acc_micro": acc_micro,
        "balanced_acc": balanced_acc,
        "f1": f1, "f1_weight": f1_weight,
        "precision": precision, "recall": recall,
        "roc_auc": roc_auc, "avg_precision": avg_precision,
        "neg_brier": neg_brier, "mcc": mcc,
        "gmean": gmean, "frr": frr
    }

def threshold_scan(y_true, y_prob, thresholds=None):
    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 101)
    rows = []
    for th in thresholds:
        m = compute_metrics_binary(y_true, y_prob, threshold=th)
        m['threshold'] = th
        rows.append(m)
    return pd.DataFrame(rows)

# =====================================================================
# Load & preprocess
# =====================================================================

def load_and_prep(train_path, val_path, test_path, target_col='Class', scale=True):
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)

    X_train = df_train.drop(columns=[target_col]).values
    y_train = df_train[target_col].values.astype(int)

    X_val = df_val.drop(columns=[target_col]).values
    y_val = df_val[target_col].values.astype(int)

    X_test = df_test.drop(columns=[target_col]).values
    y_test = df_test[target_col].values.astype(int)

    scaler = None
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

    return X_train, y_train, X_val, y_val, X_test, y_test, scaler

# =====================================================================
# Treino CatBoost
# =====================================================================

def train_catboost(
    X_train, y_train,
    X_valid=None, y_valid=None,
    params=None, seed=42
):
    start = time.time()

    cb_params = params.copy() if params else {}
    cb_params["random_seed"] = seed
    cb_params["eval_metric"] = "AUC"
    cb_params["loss_function"] = "Logloss"
    cb_params["verbose"] = False

    clf = CatBoostClassifier(**cb_params)

    fit_params = {}
    if X_valid is not None:
        fit_params["eval_set"] = (X_valid, y_valid)

    clf.fit(X_train, y_train, **fit_params)

    end = time.time()
    return clf, (end - start)

# =====================================================================
# Hyperopt search space
# =====================================================================

def build_search_space():
    return {
        "depth": hp.quniform("depth", 4, 10, 1),
        "learning_rate": hp.loguniform("learning_rate", np.log(0.01), np.log(0.5)),
        "l2_leaf_reg": hp.loguniform("l2_leaf_reg", np.log(1e-3), np.log(100)),
        "bagging_temperature": hp.uniform("bagging_temperature", 0.0, 1.0),
        "border_count": hp.quniform("border_count", 32, 255, 1),
        "iterations": hp.quniform("iterations", 200, 2000, 1),

        # scale pos weight
        "scale_pos_weight_choice": hp.choice("scale_pos_weight_choice", [
            {"type": "fixed", "value": 1.0},
            {"type": "fixed", "value": 500.0},
            {"type": "auto"}
        ])
    }

# =====================================================================
# Trial logger (CSV)
# =====================================================================

class TrialLogger:
    def __init__(self, out_csv_path):
        self.out_csv_path = out_csv_path
        self._records = []

    def append_record(self, rec):
        self._records.append(rec)
        self.flush()

    def flush(self):
        if not self._records:
            return
        df = pd.DataFrame(self._records)
        if not os.path.exists(self.out_csv_path):
            df.to_csv(self.out_csv_path, index=False)
        else:
            df.to_csv(self.out_csv_path, index=False, mode="a", header=False)
        self._records = []

# =====================================================================
# Hyperopt Objective
# =====================================================================

def objective(params, X_train, y_train, X_val, y_val, trial_logger=None, seed=42):
    p = params.copy()
    p["depth"] = int(p["depth"])
    p["border_count"] = int(p["border_count"])
    p["iterations"] = int(p["iterations"])

    p_raw = params.get("scale_pos_weight_choice")
    sp = p.pop("scale_pos_weight_choice", {"type": "fixed", "value": 1.0})

    if sp.get("type") == "auto":
        neg = (y_train == 0).sum()
        pos = (y_train == 1).sum()
        p["scale_pos_weight"] = float(neg / max(1, pos))
    else:
        p["scale_pos_weight"] = float(sp.get("value", 1.0))

    t0 = time.time()
    clf, train_time = train_catboost(
        X_train, y_train,
        X_valid=X_val, y_valid=y_val,
        params=p, seed=seed
    )
    t1 = time.time()

    y_train_prob = clf.predict_proba(X_train)[:, 1]
    y_val_prob = clf.predict_proba(X_val)[:, 1]

    train_auc = roc_auc_score(y_train, y_train_prob)
    val_auc = roc_auc_score(y_val, y_val_prob)
    mean_auc = np.mean([train_auc, val_auc])

    train_metrics = compute_metrics_binary(y_train, y_train_prob)
    val_metrics = compute_metrics_binary(y_val, y_val_prob)

    rec = {
        "time_utc": datetime.utcnow().isoformat(),
        "elapsed_sec": t1 - t0,
        "train_auc": train_auc,
        "val_auc": val_auc,
        "mean_auc": mean_auc,
        "params_resolved": p,
        "scale_pos_weight_choice_raw": p_raw
    }

    for k, v in train_metrics.items():
        rec[f"train_{k}"] = v
    for k, v in val_metrics.items():
        rec[f"val_{k}"] = v
    for k, v in p.items():
        rec[f"param_{k}"] = v

    if trial_logger:
        trial_logger.append_record(rec)

    return {"loss": -mean_auc, "status": STATUS_OK, "attachments": {"record": rec}}

# =====================================================================
# Plots
# =====================================================================

def plot_precision_recall(y_true, y_prob, out_path):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve (AP={ap:.4f})")
    plt.grid(True)
    plt.savefig(out_path)
    plt.close()

def plot_prob_histogram(y_prob, out_path):
    plt.figure()
    plt.hist(y_prob, bins=50)
    plt.xlabel("Predicted probability")
    plt.ylabel("Count")
    plt.title("Predicted probabilities")
    plt.savefig(out_path)
    plt.close()

# =====================================================================
# Treino final + avaliação
# =====================================================================

def train_final_and_evaluate(best_params, X_train_full, y_train_full, X_test, y_test, scaler, outdir, seed=42):
    ensure_dir(outdir)

    model, train_time = train_catboost(
        X_train_full, y_train_full,
        params=best_params, seed=seed
    )

    model_path = os.path.join(outdir, "catboost_best_model.cbm")
    model.save_model(model_path)

    y_test_prob = model.predict_proba(X_test)[:, 1]
    test_metrics = compute_metrics_binary(y_test, y_test_prob)

    plot_precision_recall(y_test, y_test_prob, os.path.join(outdir, "precision_recall_test.png"))
    plot_prob_histogram(y_test_prob, os.path.join(outdir, "prob_histogram_test.png"))

    thr_df = threshold_scan(y_test, y_test_prob)
    thr_df.to_csv(os.path.join(outdir, "threshold_scan_test.csv"), index=False)

    with open(os.path.join(outdir, "final_report.json"), "w") as f:
        json.dump({"train_time": train_time, "test_metrics": test_metrics}, f, indent=2)

    return {"test_metrics": test_metrics, "train_time": train_time}

# =====================================================================
# Reconstrução de parâmetros
# =====================================================================

def reconstruct_best_params(best_dict, trials_obj, X_train_example_y=None):
    try:
        best_trial = min(trials_obj.results, key=lambda x: x.get("loss", float("inf")))
        att = best_trial.get("attachments", {})
        if "record" in att:
            resolved = att["record"]["params_resolved"]
            resolved["depth"] = int(resolved["depth"])
            resolved["border_count"] = int(resolved["border_count"])
            resolved["iterations"] = int(resolved["iterations"])
            return resolved
    except:
        pass

    # fallback
    bp = {}
    for k, v in best_dict.items():
        if k in ["depth", "border_count", "iterations"]:
            bp[k] = int(v)
        else:
            bp[k] = float(v)

    # scale_pos_weight_choice
    if "scale_pos_weight_choice" in best_dict:
        idx = int(best_dict["scale_pos_weight_choice"])
        options = [
            {"type": "fixed", "value": 1.0},
            {"type": "fixed", "value": 500.0},
            {"type": "auto"}
        ]
        sel = options[idx]
        if sel["type"] == "auto":
            if X_train_example_y:
                _, y_train = X_train_example_y
                neg = (y_train == 0).sum()
                pos = (y_train == 1).sum()
                bp["scale_pos_weight"] = float(neg / max(1, pos))
            else:
                bp["scale_pos_weight"] = 1.0
        else:
            bp["scale_pos_weight"] = float(sel["value"])
        bp.pop("scale_pos_weight_choice", None)

    return bp

# =====================================================================
# Main
# =====================================================================

def main(args):
    ensure_dir(args.output_dir)
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    trial_csv = os.path.join(args.output_dir, f"hyperopt_trials_{timestamp}.csv")

    print("Carregando dados...")
    X_train, y_train, X_val, y_val, X_test, y_test, scaler = load_and_prep(
        args.train, args.val, args.test,
        target_col=args.target_col,
        scale=args.scale
    )
    print("Shapes:", X_train.shape, X_val.shape, X_test.shape)

    logger = TrialLogger(trial_csv)
    space = build_search_space()

    obj = partial(
        objective,
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        trial_logger=logger,
        seed=args.seed
    )

    trials = Trials()
    print("Iniciando TPE...")
    t0 = time.time()
    best = fmin(
        fn=obj,
        space=space,
        algo=tpe.suggest,
        max_evals=args.max_evals,
        trials=trials,
        rstate=np.random.default_rng(args.seed)
    )
    t1 = time.time()

    print(f"Busca finalizada em {t1 - t0:.2f}s")

    best_params = reconstruct_best_params(best, trials, X_train_example_y=(None, y_train))
    print("\nMelhores parâmetros reconstruídos:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")

    X_train_full = np.vstack([X_train, X_val])
    y_train_full = np.concatenate([y_train, y_val])

    final_dir = os.path.join(args.output_dir, f"final_{timestamp}")
    ensure_dir(final_dir)

    print("\nTreinando modelo final...")
    final_report = train_final_and_evaluate(
        best_params,
        X_train_full, y_train_full,
        X_test, y_test,
        scaler, final_dir,
        seed=args.seed
    )

    print("\nMétricas finais (TEST):")
    for k, v in final_report["test_metrics"].items():
        print(f"  {k}: {v}")

    trials_path = os.path.join(args.output_dir, f"hyperopt_trials_obj_{timestamp}.pkl")
    joblib.dump(trials, trials_path)

    meta = {
        "timestamp": timestamp,
        "total_search_time_sec": float(t1 - t0),
        "max_evals": int(args.max_evals),
        "best_mean_auc_est": float(-min([r.get("loss", 1.0) for r in trials.results]))
    }
    with open(os.path.join(args.output_dir, f"search_meta_{timestamp}.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("\nSaída gerada em:", args.output_dir)
    print(" - CSV trials:", trial_csv)
    print(" - Final model dir:", final_dir)
    print(" - Trials object:", trials_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CatBoost + Hyperopt Pipeline")
    parser.add_argument("--train", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--target_col", default="Class")
    parser.add_argument("--output_dir", default="outputs_cat_v1")
    parser.add_argument("--max_evals", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scale", type=bool, default=True)
    args = parser.parse_args()

    main(args)
