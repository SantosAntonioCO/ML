#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tabnet_general_csv_benchmark_v7_ultrafast.py

Versão 7 — UltraFast, robusta e com:
 - detecção automática da coluna alvo
 - correção do hp.choice (índice -> valor)
 - attachments + CSV por trial
 - treino TabNet com logging (verbose)
 - otimizada para debug/execução rápida (20 ep, patience 5, batch_size 2048)
 (gymfraud_py311_v2) C:\Users\anton\Documents\programming\python\ML\XGB\Benchmark_08_dez_2025>python tabnet_general_csv_benchmark_v6_ultrafast.py --train creditcard_train.csv --val creditcard_val.csv --test creditcard_test.csv --max_eval 5
"""

import os
import time
import json
import argparse
from datetime import datetime
from functools import partial

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, average_precision_score,
    confusion_matrix, brier_score_loss, accuracy_score,
    balanced_accuracy_score, matthews_corrcoef
)
import matplotlib.pyplot as plt
import joblib

from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from pytorch_tabnet.tab_model import TabNetClassifier
import torch

# -------------------------
# Utils
# -------------------------
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def detect_target_column(df, fname="data"):
    candidates = ["class", "Class", "label", "Label", "y", "target"]
    for c in candidates:
        if c in df.columns:
            print(f"[INFO] ({fname}) target detectado: '{c}'")
            return c
    last = df.columns[-1]
    print(f"[WARN] ({fname}) Nenhuma coluna padrão encontrada. Usando última coluna: '{last}'")
    return last

def compute_metrics_binary(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    acc_class_0 = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    acc_class_1 = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    acc_macro = (acc_class_0 + acc_class_1) / 2.0
    acc_micro = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    f1 = (2*tp) / (2*tp + fp + fn) if (2*tp + fp + fn) > 0 else 0.0
    f1_weight = f1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    roc_auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan
    avg_precision = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan
    neg_brier = -brier_score_loss(y_true, y_prob)
    mcc = matthews_corrcoef(y_true, y_pred) if len(np.unique(y_true)) > 1 else np.nan
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    gmean = np.sqrt(recall * specificity) if (recall >= 0 and specificity >= 0) else np.nan
    frr = fn / (fn + tn) if (fn + tn) > 0 else np.nan

    return {
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "acc_class_0": acc_class_0, "acc_class_1": acc_class_1,
        "acc_macro": acc_macro, "acc_micro": acc_micro, "balanced_acc": balanced_acc,
        "f1": f1, "f1_weight": f1_weight, "precision": precision, "recall": recall,
        "roc_auc": roc_auc, "avg_precision": avg_precision, "neg_brier": neg_brier,
        "mcc": mcc, "gmean": gmean, "frr": frr
    }

def plot_precision_recall(y_true, y_prob, out_path):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan
    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve (AP={ap:.4f})")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_prob_histogram(y_prob, out_path):
    plt.figure()
    plt.hist(y_prob, bins=50)
    plt.xlabel("Predicted probability")
    plt.ylabel("Count")
    plt.title("Predicted probabilities")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

# -------------------------
# TrialLogger (CSV)
# -------------------------
class TrialLogger:
    def __init__(self, out_csv_path):
        self.out_csv_path = out_csv_path
        self._records = []

    def append_record(self, rec: dict):
        self._records.append(rec)
        self.flush()

    def flush(self):
        if not self._records:
            return
        df = pd.DataFrame(self._records)
        if not os.path.exists(self.out_csv_path):
            df.to_csv(self.out_csv_path, index=False)
        else:
            df.to_csv(self.out_csv_path, index=False, mode='a', header=False)
        self._records = []

# -------------------------
# TabNet train wrapper
# -------------------------
def train_tabnet(Xtr, ytr, Xval=None, yval=None, params=None, seed=42):
    """
    Train TabNet classifier (pytorch_tabnet). params is a dict of resolved numeric values.
    """
    # TabNet construction params (minimal)
    tabnet_kwargs = dict(
        n_d=int(params.get('n_d', 8)),
        n_a=int(params.get('n_a', 8)),
        n_steps=int(params.get('n_steps', 3)),
        gamma=float(params.get('gamma', 1.3)),
        lambda_sparse=float(params.get('lambda_sparse', 1e-5)),
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=float(params.get('lr', 1e-3))),
        mask_type='sparsemax',
        seed=seed,
        verbose=1
    )

    clf = TabNetClassifier(**tabnet_kwargs)

    # sample weights for imbalance (if provided)
    spw = float(params.get('scale_pos_weight', 1.0))
    sample_weights = np.ones(len(ytr), dtype=np.float32)
    sample_weights[ytr == 1] = spw

    fit_args = dict(
        max_epochs=int(params.get('max_epochs', 20)),
        patience=int(params.get('patience', 5)),
        batch_size=int(params.get('batch_size', 2048)),
        virtual_batch_size=int(params.get('virtual_batch_size', 64)),
        num_workers=int(params.get('num_workers', 0)),
        drop_last=False
    )

    start = time.time()
    clf.fit(
        Xtr, ytr,
        eval_set=[(Xval, yval)] if Xval is not None else None,
        eval_name=['valid'] if Xval is not None else None,
        eval_metric=['auc'],
        weights=sample_weights,
        **fit_args
    )
    end = time.time()
    return clf, end - start

# -------------------------
# Hyperopt space (uses indices for choices)
# -------------------------
def build_space():
    space = {
        # choices are indices; we'll map to real values later
        "n_d": hp.choice("n_d", [0, 1, 2]),           # -> [8,16,24]
        "n_a": hp.choice("n_a", [0, 1, 2]),           # -> [8,16,24]
        "n_steps": hp.choice("n_steps", [0, 1]),      # -> [3,4]
        "gamma": hp.uniform("gamma", 1.0, 2.0),
        "lambda_sparse": hp.loguniform("lambda_sparse", np.log(1e-6), np.log(1e-3)),
        "lr": hp.loguniform("lr", np.log(1e-5), np.log(5e-3)),
        "weight_decay": hp.loguniform("weight_decay", np.log(1e-9), np.log(1e-5)),
        "batch_size": hp.choice("batch_size", [0, 1]),      # -> [2048,4096]
        "virtual_batch_size": hp.choice("virtual_batch_size", [0, 1]), # -> [32,64]
        "max_epochs": hp.choice("max_epochs", [0]),        # -> [20]
        "patience": hp.choice("patience", [0]),            # -> [5]
        "num_workers": hp.choice("num_workers", [0, 1]),   # -> [0,2]
        "scale_pos_weight_choice": hp.choice("scale_pos_weight_choice", [0,1,2])  # map later
    }
    return space

# -------------------------
# choice maps (index -> real value)
# -------------------------
CHOICE_MAP = {
    'n_d': [8, 16, 24],
    'n_a': [8, 16, 24],
    'n_steps': [3, 4],
    'batch_size': [2048, 4096],
    'virtual_batch_size': [32, 64],
    'max_epochs': [20],
    'patience': [5],
    'num_workers': [0, 2],
}

SP_OPTIONS = [
    {'type': 'fixed', 'value': 1.0},
    {'type': 'fixed', 'value': 500.0},
    {'type': 'auto'}
]

# -------------------------
# Objective
# -------------------------
def objective(space_sample, Xtr, ytr, Xval, yval, trial_logger=None, seed=42):
    # reconstruct params
    params = {}
    for k, v in space_sample.items():
        if k == 'scale_pos_weight_choice':
            continue
        if k in CHOICE_MAP:
            params[k] = CHOICE_MAP[k][int(v)]
        else:
            params[k] = float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v

    # handle scale_pos_weight
    sp_sel = SP_OPTIONS[int(space_sample['scale_pos_weight_choice'])]
    if sp_sel['type'] == 'auto':
        neg = (ytr == 0).sum()
        pos = (ytr == 1).sum()
        params['scale_pos_weight'] = float(neg / max(1, pos))
    else:
        params['scale_pos_weight'] = float(sp_sel['value'])

    # ensure integer casting for TabNet-specific fields
    for k in ['n_d','n_a','n_steps','batch_size','virtual_batch_size','max_epochs','patience','num_workers']:
        if k in params:
            params[k] = int(params[k])

    # Train
    t0 = time.time()
    clf, train_time = train_tabnet(Xtr, ytr, Xval, yval, params=params, seed=seed)
    t1 = time.time()

    # Evaluate on validation
    yval_prob = clf.predict_proba(Xval)[:, 1]
    val_auc = roc_auc_score(yval, yval_prob) if len(np.unique(yval)) > 1 else np.nan

    # Gather metrics at threshold 0.5
    val_metrics = compute_metrics_binary(yval, yval_prob, threshold=0.5)

    record = {
        "time_utc": datetime.utcnow().isoformat(),
        "elapsed_sec": t1 - t0,
        "val_auc": float(val_auc),
        "params_resolved": params,
        "scale_pos_weight_choice_raw": SP_OPTIONS[int(space_sample['scale_pos_weight_choice'])]
    }
    # add val metrics
    for k,v in val_metrics.items():
        record[f"val_{k}"] = v
    # flattened params
    for k,v in params.items():
        record[f"param_{k}"] = v

    # log to CSV
    if trial_logger is not None:
        trial_logger.append_record(record)

    # attach record to trial
    return {"loss": -float(val_auc), "status": STATUS_OK, "attachments": {"record": record}}

# -------------------------
# Final training & evaluation
# -------------------------
def final_train_and_eval(best_params, Xtr, ytr, Xval, yval, Xte, yte, outdir, seed=42):
    ensure_dir(outdir)
    # Train final on train+val
    Xfull = np.vstack([Xtr, Xval])
    yfull = np.concatenate([ytr, yval])
    clf, train_time = train_tabnet(Xfull, yfull, Xval, yval, params=best_params, seed=seed)

    # Save model
    model_path = os.path.join(outdir, "tabnet_best_model.zip")
    clf.save_model(model_path)

    # Predict test
    y_test_prob = clf.predict_proba(Xte)[:, 1]
    test_metrics = compute_metrics_binary(yte, y_test_prob, threshold=0.5)

    # plots
    plot_precision_recall(yte, y_test_prob, os.path.join(outdir, "precision_recall_test.png"))
    plot_prob_histogram(y_test_prob, os.path.join(outdir, "prob_histogram_test.png"))
    thr_df = pd.DataFrame()  # keep simple, user can call threshold_scan if wants
    # save final report
    with open(os.path.join(outdir, "final_report.json"), "w") as f:
        json.dump({"train_time_sec": train_time, "test_metrics": test_metrics}, f, indent=2, default=str)

    return {"model_path": model_path, "train_time_sec": train_time, "test_metrics": test_metrics}

# -------------------------
# Reconstruct best params robustly
# -------------------------
def reconstruct_best_params(best_dict, trials_obj, Xtr_y=None):
    # try to use attachments first
    try:
        best_trial = min(trials_obj.results, key=lambda x: x.get('loss', float('inf')))
        att = best_trial.get('attachments', {})
        if att and 'record' in att and 'params_resolved' in att['record']:
            resolved = att['record']['params_resolved']
            # ensure int casts
            for k in ['n_d','n_a','n_steps','batch_size','virtual_batch_size','max_epochs','patience','num_workers']:
                if k in resolved:
                    resolved[k] = int(resolved[k])
            return resolved
    except Exception:
        pass

    # fallback: best_dict from fmin (contains indices) -> map using CHOICE_MAP
    bp = {}
    for k,v in best_dict.items():
        if k == 'scale_pos_weight_choice':
            continue
        if k in CHOICE_MAP:
            bp[k] = CHOICE_MAP[k][int(v)]
        else:
            bp[k] = float(v) if isinstance(v, (int,float,np.integer,np.floating)) else v

    if 'scale_pos_weight_choice' in best_dict:
        sp_sel = SP_OPTIONS[int(best_dict['scale_pos_weight_choice'])]
        if sp_sel['type'] == 'auto':
            if Xtr_y is not None:
                _, ytr = Xtr_y
                neg = (ytr == 0).sum()
                pos = (ytr == 1).sum()
                bp['scale_pos_weight'] = float(neg / max(1, pos))
            else:
                bp['scale_pos_weight'] = 1.0
        else:
            bp['scale_pos_weight'] = float(sp_sel['value'])

    # ensure int casts
    for k in ['n_d','n_a','n_steps','batch_size','virtual_batch_size','max_epochs','patience','num_workers']:
        if k in bp:
            bp[k] = int(bp[k])

    return bp

# -------------------------
# Main
# -------------------------
def main(args):
    ensure_dir(args.output_dir)
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    trial_csv = os.path.join(args.output_dir, f"tabnet_hyperopt_trials_{timestamp}.csv")

    print("Carregando dados...")
    df_tr = pd.read_csv(args.train)
    df_va = pd.read_csv(args.val)
    df_te = pd.read_csv(args.test)

    # detect target col
    tcol_tr = detect_target_column(df_tr, "train")
    tcol_va = detect_target_column(df_va, "val")
    tcol_te = detect_target_column(df_te, "test")

    # extract X/y
    ytr = df_tr[tcol_tr].values.astype(int)
    yva = df_va[tcol_va].values.astype(int)
    global yte  # used by final_train_and_eval scope above
    yte = df_te[tcol_te].values.astype(int)

    Xtr_raw = df_tr.drop(columns=[tcol_tr]).values
    Xva_raw = df_va.drop(columns=[tcol_va]).values
    Xte_raw = df_te.drop(columns=[tcol_te]).values

    # scale
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(Xtr_raw)
    Xva = scaler.transform(Xva_raw)
    Xte = scaler.transform(Xte_raw)

    print("Shapes:", Xtr.shape, Xva.shape, Xte.shape)

    logger = TrialLogger(trial_csv)
    space = build_space()

    obj = partial(objective, Xtr=Xtr, ytr=ytr, Xval=Xva, yval=yva, trial_logger=logger, seed=args.seed)
    trials = Trials()

    print("Iniciando TPE (ultrafast)...")
    t0 = time.time()
    # use max_eval (note arg name)
    best = fmin(fn=lambda s: objective(s, Xtr, ytr, Xva, yva, trial_logger=logger, seed=args.seed),
                space=space,
                algo=tpe.suggest,
                max_evals=args.max_eval,
                trials=trials,
                rstate=np.random.default_rng(args.seed))
    t1 = time.time()
    print(f"Busca finalizada em {t1 - t0:.2f} s")

    # reconstruct best params
    best_params = reconstruct_best_params(best, trials, Xtr_y=(None, ytr))

    print("\nMelhores parâmetros (resolvidos):")
    for k,v in best_params.items():
        print(f"  {k}: {v}")

    # final training + evaluation
    final_dir = os.path.join(args.output_dir, f"final_{timestamp}")
    ensure_dir(final_dir)
    final_report = final_train_and_eval(best_params, Xtr, ytr, Xva, yva, Xte, yte, final_dir, seed=args.seed)

    print("\nMétricas finais (TEST):")
    for k,v in final_report['test_metrics'].items():
        print(f"  {k}: {v}")

    # save objects
    trials_path = os.path.join(args.output_dir, f"tabnet_trials_obj_{timestamp}.pkl")
    joblib.dump(trials, trials_path)

    meta = {
        "timestamp": timestamp,
        "total_search_time_sec": float(t1 - t0),
        "max_eval": int(args.max_eval),
        "best_estimated_val_auc": float(-min([r.get('loss', 0) for r in trials.results])) if trials.results else None
    }
    with open(os.path.join(args.output_dir, f"search_meta_{timestamp}.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("\nOutputs salvos em:", args.output_dir)
    print(" - trial CSV:", trial_csv)
    print(" - final artifacts:", final_dir)
    print(" - trials object:", trials_path)

# -------------------------
# Entry
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TabNet hyperopt benchmark V7 - Ultrafast")
    parser.add_argument("--train", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--output_dir", default="outputs_tabnet_v7")
    parser.add_argument("--max_eval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)
