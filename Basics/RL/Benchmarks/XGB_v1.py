#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
xgb_general_csv_benchmark_v3.py

Versão 3 — correções e logging completo (attachments + CSV).
- Grava record em attachments (para auditar)
- Reconstrói best_params preferencialmente via attachments
- Compatível com XGBoost 1.x / 2.x (eval_metric no construtor, early_stopping_rounds no fit)
- Uso: python xgb_general_csv_benchmark_v3.py --train creditcard_train.csv --val creditcard_val.csv --test creditcard_test.csv --max_evals 80
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
import xgboost as xgb
from xgboost import XGBClassifier

# ---------------------------
# Utils / métricas
# ---------------------------
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

# ---------------------------
# Carregamento e preprocessamento
# ---------------------------
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

# ---------------------------
# Funções de treino / avaliação
# ---------------------------
def train_xgb(X_train, y_train, X_valid=None, y_valid=None, params=None, seed=42, early_rounds=100):
    """
    Compatível com XGBoost 1.x e 2.x:
    - eval_metric passado no construtor
    - usa early_stopping_rounds no fit (funciona em builds sem callbacks)
    """
    start = time.time()

    # Guarantee numeric types for required params that XGBClassifier expects
    xgb_params = params.copy() if params is not None else {}
    # Ensure integer casts where logical
    if 'n_estimators' in xgb_params:
        xgb_params['n_estimators'] = int(xgb_params['n_estimators'])
    if 'max_depth' in xgb_params:
        xgb_params['max_depth'] = int(xgb_params['max_depth'])

    # Build classifier with eval_metric in constructor (safe for XGBoost 2.x)
    clf = XGBClassifier(
        **xgb_params,
        random_state=seed,
        tree_method="hist",
        eval_metric="auc"
    )

    fit_params = {}
    if X_valid is not None and y_valid is not None:
        fit_params['eval_set'] = [(X_valid, y_valid)]
        # Universal early stopping approach (no callbacks)
        fit_params['verbose'] = False

    # Fit (works across versions)
    clf.fit(X_train, y_train, **fit_params)

    end = time.time()
    return clf, end - start

# ---------------------------
# Espaço de busca Hyperopt      'n_estimators', 50, 2000, 1
# ---------------------------
def build_search_space():
    space = {
        'n_estimators': hp.quniform('n_estimators', 50, 100, 1),
        'max_depth': hp.quniform('max_depth', 7, 15, 1),
        'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.5)),
        'subsample': hp.uniform('subsample', 0.1, 1.0),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.1, 1.0),
        'colsample_bylevel': hp.uniform('colsample_bylevel', 0.1, 1.0),
        'min_child_weight': hp.uniform('min_child_weight', 0.0, 20.0),
        'gamma': hp.uniform('gamma', 0.0, 10.0),
        'max_delta_step': hp.uniform('max_delta_step', 0.0, 10.0),
        'reg_alpha': hp.loguniform('reg_alpha', np.log(1e-8), np.log(10.0)),
        'reg_lambda': hp.loguniform('reg_lambda', np.log(1e-8), np.log(10.0)),
        # choice preserved exactly in attachments (type + value)
        'scale_pos_weight_choice': hp.choice('scale_pos_weight_choice', [
            {'type': 'fixed', 'value': 1.0},
            {'type': 'fixed', 'value': 1000.0},
            {'type': 'auto'}
        ])
    }
    return space

# ---------------------------
# Objetivo Hyperopt (grava attachment + CSV)
# ---------------------------
def objective(params, X_train, y_train, X_val, y_val, trial_logger=None, seed=42):
    """
    Treina e retorna loss = -mean(train_auc, val_auc).
    Salva um 'record' detalhado: métricas train/val, params (incl. scale_pos_weight_choice).
    Esse 'record' é retornado como attachment para o trials object.
    """
    p = params.copy()
    # cast ints
    p['n_estimators'] = int(p['n_estimators'])
    p['max_depth'] = int(p['max_depth'])

    # preserve the raw 'scale_pos_weight_choice' in the p_raw for logging (so attachments contain it)
    p_raw = params.get('scale_pos_weight_choice')

    # resolve scale_pos_weight for training
    spc = p.pop('scale_pos_weight_choice', {'type': 'fixed', 'value': 1.0})
    if spc.get('type') == 'auto':
        neg = (y_train == 0).sum()
        pos = (y_train == 1).sum()
        p['scale_pos_weight'] = float(neg / max(1, pos))
    else:
        p['scale_pos_weight'] = float(spc.get('value', 1.0))

    t0 = time.time()
    clf, train_time = train_xgb(X_train, y_train, X_valid=X_val, y_valid=y_val, params=p, seed=seed)
    t1 = time.time()

    y_train_prob = clf.predict_proba(X_train)[:, 1]
    y_val_prob = clf.predict_proba(X_val)[:, 1]

    train_auc = roc_auc_score(y_train, y_train_prob) if len(np.unique(y_train)) > 1 else np.nan
    val_auc = roc_auc_score(y_val, y_val_prob) if len(np.unique(y_val)) > 1 else np.nan
    mean_auc = np.nanmean([train_auc, val_auc])

    train_metrics = compute_metrics_binary(y_train, y_train_prob, threshold=0.5)
    val_metrics = compute_metrics_binary(y_val, y_val_prob, threshold=0.5)

    rec = {
        "time_utc": datetime.utcnow().isoformat(),
        "elapsed_sec": t1 - t0,
        "train_auc": train_auc,
        "val_auc": val_auc,
        "mean_auc": mean_auc,
        # store both p (resolved) and p_raw (original choice) for auditing
        "params_resolved": p,
        "scale_pos_weight_choice_raw": p_raw
    }

    # add metrics
    for k, v in train_metrics.items():
        rec[f"train_{k}"] = v
    for k, v in val_metrics.items():
        rec[f"val_{k}"] = v

    # add flattened resolved params for CSV convenience
    for k, v in p.items():
        rec[f"param_{k}"] = v

    # log to CSV
    if trial_logger is not None:
        trial_logger.append_record(rec)

    # return attachments so Trials will have the record too
    return {"loss": -mean_auc, "status": STATUS_OK, "attachments": {"record": rec}}

# ---------------------------
# Trial logger (CSV)
# ---------------------------
class TrialLogger:
    def __init__(self, out_csv_path):
        self.out_csv_path = out_csv_path
        self._records = []

    def append_record(self, rec: dict):
        """Append and flush to CSV immediately to keep persistence."""
        self._records.append(rec)
        self.flush()

    def flush(self):
        if not self._records:
            return
        df = pd.DataFrame(self._records)
        # ensure deterministic column order: time, elapsed, a bunch, then params...
        if not os.path.exists(self.out_csv_path):
            df.to_csv(self.out_csv_path, index=False)
        else:
            df.to_csv(self.out_csv_path, index=False, mode='a', header=False)
        self._records = []

# ---------------------------
# Plot helpers
# ---------------------------
def plot_precision_recall(y_true, y_prob, out_path, title="Precision-Recall Curve"):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan
    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{title} (AP={ap:.4f})")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_prob_histogram(y_prob, out_path, bins=50, title="Histogram of predicted probabilities"):
    plt.figure()
    plt.hist(y_prob, bins=bins)
    plt.xlabel("Predicted probability")
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

# ---------------------------
# Treino final e avaliação
# ---------------------------
def train_final_and_evaluate(best_params, X_train_full, y_train_full, X_test, y_test, scaler, outdir, seed=42):
    ensure_dir(outdir)

    # best_params assumed resolved (contains scale_pos_weight numeric if auto)
    model, train_time = train_xgb(X_train_full, y_train_full, X_valid=None, y_valid=None, params=best_params, seed=seed)

    model_path = os.path.join(outdir, "xgb_best_model.joblib")
    joblib.dump(model, model_path)

    y_test_prob = model.predict_proba(X_test)[:, 1]
    test_metrics = compute_metrics_binary(y_test, y_test_prob, threshold=0.5)

    # plots + threshold scan
    plot_precision_recall(y_test, y_test_prob, os.path.join(outdir, "precision_recall_test.png"))
    plot_prob_histogram(y_test_prob, os.path.join(outdir, "prob_histogram_test.png"))
    thr_df = threshold_scan(y_test, y_test_prob)
    thr_df.to_csv(os.path.join(outdir, "threshold_scan_test.csv"), index=False)

    # final report
    with open(os.path.join(outdir, "final_report.json"), "w") as f:
        json.dump({
            "train_time_sec": train_time,
            "test_metrics": test_metrics
        }, f, indent=2, default=str)

    return {
        "model_path": model_path,
        "train_time_sec": train_time,
        "test_metrics": test_metrics
    }

# ---------------------------
# Reconstruct best params robustly (prefer attachments)
# ---------------------------
def reconstruct_best_params(best_dict, trials_obj, X_train_example_y=None):
    """
    best_dict : dict returned by hyperopt.fmin (mapping varname -> value or index)
    trials_obj : Trials instance (to inspect attachments if available)
    X_train_example_y : tuple (X_train, y_train) used to compute auto scale_pos_weight if needed
    """
    # Try retrieve best trial with smallest loss and attachment record
    try:
        best_trial = min(trials_obj.results, key=lambda x: x.get('loss', float('inf')))
        att = best_trial.get('attachments', {})
        if 'record' in att and 'params_resolved' in att['record']:
            # prefer the stored resolved params (already numeric)
            resolved = att['record']['params_resolved']
            # ensure integer casts
            if 'n_estimators' in resolved:
                resolved['n_estimators'] = int(resolved['n_estimators'])
            if 'max_depth' in resolved:
                resolved['max_depth'] = int(resolved['max_depth'])
            return resolved
    except Exception:
        pass

    # fallback: rebuild from best_dict returned by fmin
    bp = {}
    for k, v in best_dict.items():
        if k in ['n_estimators', 'max_depth']:
            bp[k] = int(v)
        else:
            bp[k] = float(v) if isinstance(v, (float, int)) else v

    # handle scale_pos_weight_choice
    if 'scale_pos_weight_choice' in best_dict:
        idx = int(best_dict['scale_pos_weight_choice'])
        options = [
            {'type': 'fixed', 'value': 1.0},
            {'type': 'fixed', 'value': 1000.0},
            {'type': 'auto'}
        ]
        sel = options[idx]
        if sel['type'] == 'auto':
            if X_train_example_y is not None:
                _, y_train = X_train_example_y
                neg = (y_train == 0).sum()
                pos = (y_train == 1).sum()
                bp['scale_pos_weight'] = float(neg / max(1, pos))
            else:
                bp['scale_pos_weight'] = 1.0
        else:
            bp['scale_pos_weight'] = float(sel['value'])
        bp.pop('scale_pos_weight_choice', None)

    return bp

# ---------------------------
# Main flow
# ---------------------------
def main(args):
    ensure_dir(args.output_dir)
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    trial_csv = os.path.join(args.output_dir, f"hyperopt_trials_{timestamp}.csv")

    print("Carregando dados...")
    X_train, y_train, X_val, y_val, X_test, y_test, scaler = load_and_prep(
        args.train, args.val, args.test, target_col=args.target_col, scale=args.scale
    )
    print("Shapes:", X_train.shape, X_val.shape, X_test.shape)

    logger = TrialLogger(trial_csv)
    space = build_search_space()

    obj = partial(objective,
                  X_train=X_train, y_train=y_train,
                  X_val=X_val, y_val=y_val,
                  trial_logger=logger, seed=args.seed)

    trials = Trials()
    print("Iniciando busca (TPE)...")
    t0 = time.time()
    best = fmin(fn=obj, space=space, algo=tpe.suggest, max_evals=args.max_evals, trials=trials, rstate=np.random.default_rng(args.seed))
    t1 = time.time()
    print(f"Busca finalizada em {t1 - t0:.2f}s")

    # Reconstruct best params preferring attachments
    best_params = reconstruct_best_params(best, trials, X_train_example_y=(None, y_train))

    print("\nMelhores parâmetros reconstruídos:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")

    # Train final model on train+val
    print("\nTreinando modelo final (train+val) e avaliando em test...")
    X_train_full = np.vstack([X_train, X_val])
    y_train_full = np.concatenate([y_train, y_val])

    final_dir = os.path.join(args.output_dir, f"final_{timestamp}")
    ensure_dir(final_dir)

    final_report = train_final_and_evaluate(best_params, X_train_full, y_train_full, X_test, y_test, scaler, final_dir, seed=args.seed)

    print("\nMétricas finais (TEST):")
    for k, v in final_report['test_metrics'].items():
        print(f"  {k}: {v}")

    # Save trials object and meta
    trials_path = os.path.join(args.output_dir, f"hyperopt_trials_obj_{timestamp}.pkl")
    joblib.dump(trials, trials_path)

    meta = {
        "timestamp": timestamp,
        "total_search_time_sec": float(t1 - t0),
        "max_evals": int(args.max_evals),
        "best_mean_auc_est": float(-min([r.get('loss', 1.0) for r in trials.results])) if trials.results else None
    }
    with open(os.path.join(args.output_dir, f"search_meta_{timestamp}.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("\nSaída gerada em:", args.output_dir)
    print(" - trial CSV:", trial_csv)
    print(" - final artifacts:", final_dir)
    print(" - trials object:", trials_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="XGBoost + Hyperopt pipeline (v3)")
    parser.add_argument("--train", required=True, help="CSV de treino com coluna target (default 'Class')")
    parser.add_argument("--val", required=True, help="CSV de validação")
    parser.add_argument("--test", required=True, help="CSV de teste")
    parser.add_argument("--target_col", default="Class", help="nome da coluna target (padrão: Class)")
    parser.add_argument("--output_dir", default="outputs_xgb_v3", help="pasta onde salvar resultados")
    parser.add_argument("--max_evals", type=int, default=80, help="número máximo de trials do hyperopt")
    parser.add_argument("--seed", type=int, default=42, help="seed")
    parser.add_argument("--scale", type=bool, default=True, help="aplicar StandardScaler nas features?")
    args = parser.parse_args()

    main(args)
