#!/usr/bin/env python3
# xgb_baseline_pipeline.py
"""
Baseline pipeline: XGBoost per-target with scaffold K-fold CV.
This version will attempt to load Optuna-best parameters for each target from:
 - results/optuna_{target}_best.json  OR
 - results/optuna_{target}.json
If found, those params override the default xgb_params for that target.
"""

import os
import sys
import math
import json
import random
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr, spearmanr

import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# ---------------------------
# Utility functions
# ---------------------------
def read_smiles(path):
    """Read molecule.smi; tolerant to one-column or two-column formats.
    Returns DataFrame with columns ['molecule_id','smiles'].
    """
    df = None
    # try two-column tab/space
    try:
        df = pd.read_csv(path, sep=r'\s+|\t+', header=None, engine='python', names=['col1','col2'])
        # if second column looks like smiles (contains letters and digits and parentheses), take it
        if 'col2' in df.columns and df['col2'].apply(lambda x: isinstance(x, str) and any(c.isalpha() for c in x)).sum() > 0:
            df = df.rename(columns={'col1':'molecule_id','col2':'smiles'})
            # ensure molecule_id is string
            df['molecule_id'] = df['molecule_id'].astype(str)
            return df[['molecule_id','smiles']]
    except Exception:
        pass
    # fallback: try comma-separated two columns
    try:
        df = pd.read_csv(path, header=None, names=['molecule_id','smiles'], sep=',', engine='python', dtype=str, keep_default_na=False)
        if df.shape[1] >= 2:
            df['molecule_id'] = df['molecule_id'].astype(str)
            df['smiles'] = df['smiles'].astype(str)
            return df[['molecule_id','smiles']]
    except Exception:
        pass
    # fallback: read single-column smiles
    try:
        df = pd.read_csv(path, header=None, names=['smiles'])
        df = df.reset_index().rename(columns={'index':'molecule_id'})
        df['molecule_id'] = df['molecule_id'].astype(str)
        return df[['molecule_id','smiles']]
    except Exception as e:
        raise RuntimeError(f"Failed to read smiles from {path}: {e}")

def mol_from_smiles_safe(smi):
    try:
        m = Chem.MolFromSmiles(smi)
        return m
    except Exception:
        return None

# Modified compute_morgan_fp to handle RDKit versions
def compute_morgan_fp(mol, nBits=1024, radius=2):
    """
    Compute Morgan fingerprint robustly. Prefer new rdFingerprintGenerator if available.
    Returns (numpy_array, bitinfo_dict)
    """
    if mol is None:
        return np.zeros((nBits,), dtype=np.float32), {}
    try:
        # Try newer generator
        from rdkit.Chem import rdFingerprintGenerator
        MorganGenerator = rdFingerprintGenerator.GetMorganGenerator
        gen = MorganGenerator(radius=radius, fpSize=nBits)
        fp = gen.GetFingerprint(mol)
        arr = np.zeros((nBits,), dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr.astype(np.float32), {}
    except Exception:
        bitinfo = {}
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits, bitInfo=bitinfo)
        arr = np.zeros((nBits,), dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr.astype(np.float32), bitinfo

def compute_physchem(mol):
    # returns 6 physchem descriptors as float
    try:
        return np.array([
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol),
            Descriptors.TPSA(mol),
            Descriptors.NumRotatableBonds(mol)
        ], dtype=np.float32)
    except Exception:
        return np.zeros((6,), dtype=np.float32)

def safe_get_scaffold(smi):
    m = mol_from_smiles_safe(smi)
    if m is None:
        return None
    try:
        scaf = MurckoScaffold.GetScaffoldForMol(m)
        return Chem.MolToSmiles(scaf)
    except Exception:
        return None

def safe_rmse(y_true, y_pred):
    # robust RMSE across sklearn versions
    try:
        return mean_squared_error(y_true, y_pred, squared=False)
    except TypeError:
        try:
            # newer sklearn might have root_mean_squared_error (unlikely here)
            from sklearn.metrics import root_mean_squared_error
            return root_mean_squared_error(y_true, y_pred)
        except Exception:
            return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def evaluate_vector(y_true, y_pred):
    mask = ~np.isnan(y_true)
    if mask.sum() < 2:
        return {'MAE':np.nan,'RMSE':np.nan,'Pearson':np.nan,'Spearman':np.nan}
    yt = y_true[mask]
    yp = y_pred[mask]
    mae = mean_absolute_error(yt, yp)
    rmse = safe_rmse(yt, yp)
    try:
        pearson = pearsonr(yt, yp)[0]
    except Exception:
        pearson = np.nan
    try:
        spearman = spearmanr(yt, yp)[0]
    except Exception:
        spearman = np.nan
    return {'MAE':float(mae),'RMSE':float(rmse),'Pearson':float(pearson),'Spearman':float(spearman)}

# ---------------------------
# Optuna params loader
# ---------------------------
def load_optuna_best_params(results_dir, target_col):
    """
    Try to load an Optuna JSON file for this target.
    Looks for:
      results/optuna_{target_col}_best.json
      results/optuna_{target_col}.json
    Returns dict(params) or None if not found.
    """
    candidates = [
        os.path.join(results_dir, f"optuna_{target_col}_best.json"),
        os.path.join(results_dir, f"optuna_{target_col}.json"),
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                with open(p,'r') as f:
                    obj = json.load(f)
                # obj may be {'value':..., 'params':{...}} or may directly be params dict
                if isinstance(obj, dict) and 'params' in obj:
                    return obj['params']
                # sometimes saved as {"params": {...}} or directly params
                if isinstance(obj, dict):
                    # Heuristics: if keys look like param names (max_depth, eta...), return obj
                    known_keys = {'max_depth','eta','subsample','colsample_bytree','lambda','alpha','min_child_weight'}
                    if any(k in obj for k in known_keys):
                        return obj
                # else, not recognized
            except Exception as e:
                print(f"[load_optuna_best_params] Failed to read {p}: {e}")
    return None

def prepare_xgb_params_from_optuna(base_params, optuna_params):
    """
    Merge base_params with optuna_params (which may contain keys as strings).
    Convert types as needed for XGBoost.
    """
    if optuna_params is None:
        return base_params.copy()
    p = base_params.copy()
    # allowed keys to override
    allowed = {'max_depth','eta','subsample','colsample_bytree','lambda','alpha','min_child_weight'}
    for k,v in optuna_params.items():
        if k not in allowed:
            # ignore unknown keys silently
            continue
        try:
            if k in {'max_depth','min_child_weight'}:
                p[k] = int(v)
            elif k in {'eta','subsample','colsample_bytree','lambda','alpha'}:
                p[k] = float(v)
            else:
                p[k] = v
        except Exception:
            # fallback to raw assign
            p[k] = v
    return p

# ---------------------------
# Main pipeline
# ---------------------------
def main():
    # paths
    data_dir = "data"
    results_dir = "results"
    models_dir = os.path.join(results_dir, "models")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    smiles_path = os.path.join(data_dir, "molecule.smi")
    activity_path = os.path.join(data_dir, "activity_train.csv")
    prop_path = os.path.join(data_dir, "property.csv")  # optional

    print("Reading smiles...")
    smis_df = read_smiles(smiles_path)
    print(f"Read {len(smis_df)} smiles.")

    print("Reading activity...")
    act = pd.read_csv(activity_path)
    # Expect columns: molecule_id,target_id,pIC50
    required_cols = {'molecule_id','target_id','pIC50'}
    if not required_cols.issubset(set(act.columns)):
        raise RuntimeError(f"activity_train.csv must contain columns: {required_cols}. Found: {list(act.columns)}")

    # pivot to wide: one row per molecule_id, columns per target
    act_wide = act.pivot(index='molecule_id', columns='target_id', values='pIC50').reset_index()
    # rename columns to safe names (T_<id>)
    target_ids = [str(c) for c in act_wide.columns.tolist() if c != 'molecule_id']
    target_cols = []
    for c in target_ids:
        target_cols.append(f"T_{c}")
    # construct mapping
    rename_map = {}
    for orig, new in zip([c for c in act_wide.columns if c!='molecule_id'], target_cols):
        rename_map[orig] = new
    act_wide = act_wide.rename(columns=rename_map)

    # merge smiles + activity
    df = smis_df.merge(act_wide, on='molecule_id', how='left')
    print(f"Molecules after merge: {len(df)}")

    # optional merge properties
    if os.path.exists(prop_path):
        prop = pd.read_csv(prop_path)
        df = df.merge(prop, on='molecule_id', how='left')
        print("Merged property.csv")

    # parse molecules and filter invalid
    print("Parsing SMILES and computing scaffolds...")
    df['rdkit_mol'] = df['smiles'].apply(mol_from_smiles_safe)
    invalid_count = df['rdkit_mol'].isna().sum()
    if invalid_count > 0:
        print(f"Warning: {invalid_count} SMILES failed to parse. These will be dropped.")
        df = df[~df['rdkit_mol'].isna()].reset_index(drop=True)

    # compute scaffold
    df['scaffold'] = df['smiles'].apply(safe_get_scaffold)
    missing_scaffold = df['scaffold'].isna().sum()
    if missing_scaffold > 0:
        print(f"Warning: {missing_scaffold} scaffolds missing (kept as unique keys).")

    # FEATURIZATION
    print("Featurizing molecules (Morgan 1024 + physchem)...")
    N = len(df)
    nBits = 1024
    fp_list = []
    bitinfo_list = []
    phys_list = []
    for i, m in enumerate(df['rdkit_mol'].values):
        arr, bitinfo = compute_morgan_fp(m, nBits=nBits, radius=2)
        phys = compute_physchem(m)
        fp_list.append(arr)
        phys_list.append(phys)
        bitinfo_list.append(bitinfo)
        if (i+1) % 200 == 0 or (i+1)==N:
            print(f"  featurized {i+1}/{N}")
    X_fp = np.vstack(fp_list)  # N x 1024
    X_phys = np.vstack(phys_list)  # N x 6
    X = np.hstack([X_fp, X_phys])  # N x (1024+6)
    feature_names = [f'FP_{i}' for i in range(nBits)] + ['MolWt','LogP','HBD','HBA','TPSA','RotBonds']

    # Save bitinfo for downstream interpretability
    joblib.dump(bitinfo_list, os.path.join(results_dir, "bitinfo_list.pkl"))
    print("Saved bitinfo_list.pkl")

    # Build Y matrix and mask
    target_cols_ordered = target_cols  # e.g., ["T_1","T_2",...]
    T = len(target_cols_ordered)
    Y = np.zeros((N, T), dtype=np.float32)
    mask = np.zeros((N, T), dtype=np.uint8)
    for j, col in enumerate(target_cols_ordered):
        if col in df.columns:
            vals = df[col].values
            nanmask = pd.isna(vals)
            Y[:, j] = np.where(nanmask, np.nan, vals)
            mask[:, j] = (~nanmask).astype(np.uint8)
        else:
            Y[:, j] = np.nan
            mask[:, j] = 0

    # scaffold K-fold split (group by scaffold)
    print("Creating scaffold folds...")
    n_folds = 5
    groups = df.groupby('scaffold').indices  # scaffold -> array of indices
    scaffold_items = list(groups.items())
    scaffold_items.sort(key=lambda x: len(x[1]), reverse=True)
    folds = [[] for _ in range(n_folds)]
    fold_sizes = [0]*n_folds
    for scaffold, idxs in scaffold_items:
        min_fold = int(np.argmin(fold_sizes))
        folds[min_fold].extend(list(idxs))
        fold_sizes[min_fold] += len(idxs)
    for k in range(n_folds):
        print(f"  fold {k}: {len(folds[k])} samples")

    # Save folds for reproducibility
    with open(os.path.join(results_dir, "scaffold_folds.json"), "w") as f:
        json.dump({f"fold_{i}": [int(idx) for idx in folds[i]] for i in range(n_folds)}, f)

    # StandardScaler for physchem features (apply only to phys columns)
    phys_scaler = StandardScaler()
    X_phys_scaled = phys_scaler.fit_transform(X_phys)
    X_scaled = np.hstack([X_fp, X_phys_scaled])
    joblib.dump(phys_scaler, os.path.join(results_dir, "phys_scaler.joblib"))

    # Prepare OOF containers
    oof_preds = np.full_like(Y, np.nan, dtype=np.float32)
    models_per_target_per_fold = dict()
    metrics_records = []

    # Default XGBoost parameters (starter)
    xgb_params_default = {
        'objective': 'reg:squarederror',
        'eta': 0.008,
        'max_depth': 10,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'seed': SEED,
        'verbosity': 0,
        'nthread': 4,
    }

    num_boost_round = 20000
    early_stopping_rounds = 50

    # For each fold: train models per target using training indices (all folds except k)
    print("Begin K-fold scaffold CV training (per-target XGBoost)...")
    for k in range(n_folds):
        val_idx = np.array(folds[k], dtype=int)
        train_idx = np.array([i for f in range(n_folds) if f!=k for i in folds[f]], dtype=int)
        X_tr = X_scaled[train_idx]
        X_va = X_scaled[val_idx]
        print(f" Fold {k}: train {len(train_idx)} val {len(val_idx)}")
        # for each target train a separate xgboost model
        for t_idx, tcol in enumerate(target_cols_ordered):
            # Prepare per-target params: try to load Optuna-best and merge
            optuna_params = load_optuna_best_params(results_dir, tcol)
            if optuna_params is not None:
                xgb_params = prepare_xgb_params_from_optuna(xgb_params_default, optuna_params)
                print(f"  [INFO] Loaded Optuna params for {tcol}: {optuna_params}")
            else:
                xgb_params = xgb_params_default.copy()

            y_tr = Y[train_idx, t_idx]
            y_va = Y[val_idx, t_idx]
            mask_tr = ~np.isnan(y_tr)
            if mask_tr.sum() < 5:
                # too few training examples for this target in this split
                print(f"  [fold {k}] target {tcol}: too few train samples ({mask_tr.sum()}), skip training this fold.")
                continue
            # create DMatrix: only include rows with labels
            dtrain = xgb.DMatrix(X_tr[mask_tr], label=y_tr[mask_tr])
            # validation set uses full val set but we will evaluate only where label exists
            val_mask = ~np.isnan(y_va)
            if val_mask.sum() >= 5:
                dval = xgb.DMatrix(X_va[val_mask], label=y_va[val_mask])
                evals = [(dtrain, 'train'), (dval, 'val')]
                try:
                    bst = xgb.train(xgb_params, dtrain, num_boost_round=num_boost_round, evals=evals,
                                    early_stopping_rounds=early_stopping_rounds, verbose_eval=False)
                except Exception as e:
                    print(f"  [WARN] xgb.train failed with merged params for {tcol} on fold {k}: {e}")
                    # fallback to default params if failed
                    bst = xgb.train(xgb_params_default, dtrain, num_boost_round=500, verbose_eval=False)
            else:
                # no valid val labels -> train without early stopping using smaller rounds
                bst = xgb.train(xgb_params, dtrain, num_boost_round=200, verbose_eval=False)
            # save model
            model_path = os.path.join(models_dir, f"{tcol}_fold{k}.model")
            bst.save_model(model_path)
            models_per_target_per_fold[(tcol, k)] = model_path
            # predict on val (we predict all val rows but only fill where label exists)
            pred_val_all = bst.predict(xgb.DMatrix(X_va))
            # write to oof container (will be NaN where true label missing)
            oof_preds[val_idx, t_idx] = np.where(np.isnan(Y[val_idx, t_idx]), np.nan, pred_val_all)
            # compute metric for this fold and target (only where y exists)
            metric = evaluate_vector(Y[val_idx, t_idx], pred_val_all)
            metric_record = {
                'fold': k,
                'target': tcol,
                'n_train_labeled': int(mask_tr.sum()),
                'n_val_labeled': int((~np.isnan(y_va)).sum()),
                'MAE': metric['MAE'],
                'RMSE': metric['RMSE'],
                'Pearson': metric['Pearson'],
                'Spearman': metric['Spearman'],
                'model_path': model_path
            }
            metrics_records.append(metric_record)
            print(f"  [fold {k}] {tcol}: n_tr={mask_tr.sum()}, n_val={metric_record['n_val_labeled']}, MAE={metric['MAE']:.4f}, RMSE={metric['RMSE']:.4f}, P={metric['Pearson']:.3f}, S={metric['Spearman']:.3f}")

    # Save OOF predictions (align with df)
    oof_df = df[['molecule_id','smiles']].copy().reset_index(drop=True)
    for j, col in enumerate(target_cols_ordered):
        oof_df[f'oof_{col}'] = oof_preds[:, j]
    oof_df.to_csv(os.path.join(results_dir, "oof_predictions.csv"), index=False)
    print(f"Saved OOF predictions to {os.path.join(results_dir, 'oof_predictions.csv')}")

    # Save metrics per fold
    metrics_df = pd.DataFrame(metrics_records)
    metrics_df.to_csv(os.path.join(results_dir, "metrics_per_fold.csv"), index=False)
    print(f"Saved metrics per fold to {os.path.join(results_dir, 'metrics_per_fold.csv')}")

    # Summarize per-target mean ± std across folds
    summary_rows = []
    for tcol in target_cols_ordered:
        dd = metrics_df[metrics_df['target']==tcol]
        if len(dd)==0:
            continue
        row = {
            'target': tcol,
            'MAE_mean': dd['MAE'].mean(),
            'MAE_std': dd['MAE'].std(),
            'RMSE_mean': dd['RMSE'].mean(),
            'RMSE_std': dd['RMSE'].std(),
            'Pearson_mean': dd['Pearson'].mean(),
            'Pearson_std': dd['Pearson'].std(),
            'Spearman_mean': dd['Spearman'].mean(),
            'Spearman_std': dd['Spearman'].std(),
            'n_folds': len(dd)
        }
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(results_dir, "metrics_summary.csv"), index=False)
    print(f"Saved metrics summary to {os.path.join(results_dir, 'metrics_summary.csv')}")
    print("Baseline XGBoost per-target training complete.")

    # Optional: quick diagnostic plots for one target
    try:
        for tcol in target_cols_ordered:
            col_oof = f'oof_{tcol}'
            if col_oof not in oof_df.columns:
                continue
            mask_valid = ~oof_df[col_oof].isna() & ~pd.isna(df.get(tcol))
            if mask_valid.sum() < 5:
                continue
            plt.figure(figsize=(5,5))
            plt.scatter(df.loc[mask_valid, tcol], oof_df.loc[mask_valid, col_oof], alpha=0.6, s=20)
            mn = min(df.loc[mask_valid, tcol].min(), oof_df.loc[mask_valid, col_oof].min())
            mx = max(df.loc[mask_valid, tcol].max(), oof_df.loc[mask_valid, col_oof].max())
            plt.plot([mn,mx],[mn,mx], color='red', lw=1)
            plt.xlabel('True pIC50'); plt.ylabel('OOF Pred'); plt.title(tcol)
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, f"diag_{tcol}.png"))
            plt.close()
    except Exception as e:
        print("Plotting diagnostics failed:", e)

if __name__ == "__main__":
    main()
