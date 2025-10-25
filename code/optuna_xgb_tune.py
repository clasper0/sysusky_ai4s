#!/usr/bin/env python3
# optuna_xgb_tune.py
"""
Optuna-based XGBoost hyperparameter tuning on scaffold CV folds.

Usage:
    python optuna_xgb_tune.py --target T_1 --n_trials 500 --n_jobs 1

Requires:
    pip install optuna xgboost joblib numpy pandas scikit-learn
    RDKit not required at tune-time if features X.npy already computed.

Expectations:
    - results/scaffold_folds.json exists from previous step OR script will compute folds (requires data/molecule.smi).
    - features (X) and labels (Y) are loaded from data or results produced earlier. For simplicity this script recomputes features if needed.
Outputs:
    - results/optuna_{target}.json  (best params)
    - results/optuna_{target}_trials.csv (trial history)
"""

import os, sys, json, argparse, time
import numpy as np, pandas as pd
import joblib
from functools import partial

# ML libs
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error
from scipy.stats import pearsonr
import optuna

# optional RDKit features recompute if necessary
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

SEED = 42
np.random.seed(SEED)

# ---------- utility functions (simplified versions) -----------
from rdkit import Chem
import pandas as pd
import os
import re
from robust_read_smiles import read_smiles
df = read_smiles("data/molecule.smi")
print(df.head())


def _is_valid_smiles(smi):
    """Return True if RDKit can parse the SMILES string (non-empty)."""
    if smi is None:

        return False
    smi = str(smi).strip()
    if smi == "":
        return False
    try:
        m = Chem.MolFromSmiles(smi)
        return m is not None
    except Exception:
        return False


def mol_from_smiles_safe(smi):
    try:
        return Chem.MolFromSmiles(smi)
    except:
        return None

def compute_morgan_fp(mol, nBits=1024, radius=2):
    bitinfo = {}
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits, bitInfo=bitinfo)
    arr = np.zeros((nBits,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr.astype(np.float32)

def compute_physchem(mol):
    return np.array([
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.TPSA(mol),
        Descriptors.NumRotatableBonds(mol)
    ], dtype=np.float32)

def safe_get_scaffold(smi):
    m = mol_from_smiles_safe(smi)
    if m is None: return None
    scaf = MurckoScaffold.GetScaffoldForMol(m)
    return Chem.MolToSmiles(scaf, isomericSmiles=False)

# ---------- main tuning logic -----------
def load_or_build_features(data_dir="data", results_dir="results", nBits=1024):
    # If results/X.npy exists, load; else compute from data/molecule.smi
    X_path = os.path.join(results_dir, "X.npy")
    Y_path = os.path.join(results_dir, "Y.npy")
    mask_path = os.path.join(results_dir, "mask.npy")
    df_path = os.path.join(results_dir, "df.pkl")

    if os.path.exists(X_path) and os.path.exists(Y_path) and os.path.exists(mask_path) and os.path.exists(df_path):
        X = np.load(X_path)
        Y = np.load(Y_path)
        mask = np.load(mask_path)
        df = pd.read_pickle(df_path)
        print("Loaded precomputed features from results/")
        return df, X, Y, mask

    # else compute (similar to baseline script)
    smiles_path = os.path.join(data_dir, "molecule.smi")
    act_path = os.path.join(data_dir, "activity_train.csv")
    if not os.path.exists(smiles_path) or not os.path.exists(act_path):
        raise RuntimeError("Need data/molecule.smi and data/activity_train.csv to build features")

    print("Computing features from raw data (this may take a while)...")
    smis = read_smiles(smiles_path)
    act = pd.read_csv(act_path)
    act_wide = act.pivot(index='molecule_id', columns='target_id', values='pIC50').reset_index()
    # rename columns T_<id>
    orig = [c for c in act_wide.columns if c!='molecule_id']
    target_cols = [f"T_{str(c)}" for c in orig]
    rm = {o:n for o,n in zip(orig, target_cols)}
    act_wide = act_wide.rename(columns=rm)
    df = smis.merge(act_wide, on='molecule_id', how='left')
    # compute molecules
    df['mol'] = df['smiles'].apply(mol_from_smiles_safe)
    # compute features
    fps = []
    phys = []
    for m in df['mol']:
        fps.append(compute_morgan_fp(m, nBits=nBits))
        phys.append(compute_physchem(m))
    X_fp = np.vstack(fps)
    X_phys = np.vstack(phys)
    X = np.hstack([X_fp, X_phys])
    # Y and mask
    T = len(target_cols)
    Y = np.full((len(df), T), np.nan, dtype=np.float32)
    mask = np.zeros((len(df), T), dtype=np.uint8)
    for j,c in enumerate(target_cols):
        if c in df.columns:
            arr = df[c].values
            nanm = pd.isna(arr)
            Y[:, j] = np.where(nanm, np.nan, arr)
            mask[:, j] = (~nanm).astype(np.uint8)
    # save
    os.makedirs(results_dir, exist_ok=True)
    np.save(os.path.join(results_dir, "X.npy"), X)
    np.save(os.path.join(results_dir, "Y.npy"), Y)
    np.save(os.path.join(results_dir, "mask.npy"), mask)
    df.to_pickle(os.path.join(results_dir, "df.pkl"))
    print("Saved features to results/")
    return df, X, Y, mask

def build_scaffold_folds(df, results_dir="results", n_folds=5):
    folds_path = os.path.join(results_dir, "scaffold_folds.json")


    if os.path.exists(folds_path):
        with open(folds_path,'r') as f:
            folds = json.load(f)
        # convert to list of lists
        folds_list = [folds[f"fold_{i}"] for i in range(len(folds))]
        print("Loaded scaffold folds from results/")
        return folds_list
    

    # else compute
    scafs = df['smiles'].apply(safe_get_scaffold)
    groups = df.groupby(scafs).indices
    items = list(groups.items())
    items.sort(key=lambda x: len(x[1]), reverse=True)
    folds = [[] for _ in range(n_folds)]
    sizes = [0]*n_folds
    for scaf, idxs in items:
        k = int(np.argmin(sizes))
        folds[k].extend(idxs.tolist())
        sizes[k] += len(idxs)
    # save
    d = {f"fold_{i}": folds[i] for i in range(n_folds)}
    with open(os.path.join(results_dir, "scaffold_folds.json"), "w") as f:
        json.dump(d, f)
    print("Saved scaffold_folds.json")
    return folds

def objective_xgb(trial, X, Y, mask, folds, target_index, results_dir, num_boost_round=500, early_stopping_rounds=30):
    """Trial objective: return mean RMSE across folds for the given target index."""
    # suggest hyperparams
    param = {
        'objective': 'reg:squarederror',
        'tree_method': 'hist',
        'eval_metric': 'rmse',
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'eta': trial.suggest_loguniform('eta', 1e-3, 0.2),
        'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
        'lambda': trial.suggest_loguniform('lambda', 1e-6, 10.0),
        'alpha': trial.suggest_loguniform('alpha', 1e-6, 10.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
        'seed': SEED,
        'verbosity': 0,
    }
    rmses = []
    # For speed: use StandardScaler on phys columns if any (assume last 6 columns are phys)
    n_features = X.shape[1]
    n_phy = 6
    n_fp = n_features - n_phy
    # prepare scaled X once outside fold loop? But keep simple
    for k in range(len(folds)):
        val_idx = np.array(folds[k], dtype=int)
        train_idx = np.array([i for kk in range(len(folds)) if kk!=k for i in folds[kk]], dtype=int)
        X_tr = X[train_idx]
        X_va = X[val_idx]
        y_tr = Y[train_idx, target_index]
        y_va = Y[val_idx, target_index]
        # only use labeled rows
        mask_tr = ~np.isnan(y_tr)
        mask_va = ~np.isnan(y_va)
        if mask_tr.sum() < 5 or mask_va.sum() < 1:
            # penalize trials that cannot be evaluated
            return float('inf')
        dtrain = xgb.DMatrix(X_tr[mask_tr], label=y_tr[mask_tr])
        if mask_va.sum() >= 5:
            dval = xgb.DMatrix(X_va[mask_va], label=y_va[mask_va])
            bst = xgb.train(param, dtrain, num_boost_round=num_boost_round, evals=[(dval,'val')], early_stopping_rounds=early_stopping_rounds, verbose_eval=False)
            pred = bst.predict(xgb.DMatrix(X_va[mask_va]))
            rmse = root_mean_squared_error(y_va[mask_va], pred)
            rmses.append(rmse)
        else:
            # small val, use train OOF-like evaluation
            bst = xgb.train(param, dtrain, num_boost_round=100, verbose_eval=False)
            pred = bst.predict(xgb.DMatrix(X_va))
            rmse = root_mean_squared_error(y_va[~np.isnan(y_va)], pred[~np.isnan(y_va)])
            rmses.append(rmse)
    # return average rmse
    avg_rmse = float(np.mean(rmses))
    # log intermediate result in trial
    trial.set_user_attr('avg_rmse', avg_rmse)
    return avg_rmse

def tune_target_with_optuna(target_col, df, X, Y, mask, folds, results_dir, n_trials=200, n_jobs=1):
    print(f"Start tuning for {target_col} with {n_trials} trials (n_jobs={n_jobs})")
    target_index = list(df.columns).index(target_col) if target_col in df.columns else int(target_col.split('_')[-1]) - 1
    # but simpler: target index mapping from saved df object: we expect df.pkl to have columns names saved earlier
    # For safety: user can pass integer target index via CLI; we'll check types
    # build study
    study_name = f"optuna_xgb_{target_col}"
    storage = None
    study = optuna.create_study(direction="minimize", study_name=study_name, sampler=optuna.samplers.TPESampler(seed=SEED))
    obj = partial(objective_xgb, X=X, Y=Y, mask=mask, folds=folds, target_index=target_index, results_dir=results_dir)
    # run optimization
    study.optimize(obj, n_trials=n_trials, n_jobs=n_jobs)
    print("Best trial:", study.best_trial.params, "value:", study.best_trial.value)
    # save best params
    best = study.best_trial.params
    with open(os.path.join(results_dir, f"optuna_{target_col}_best.json"), 'w') as f:
        json.dump({'params': best, 'value': study.best_value}, f, indent=2)
    # export trials to CSV
    
    # Safe export of trials to DataFrame (compatible across Optuna versions)
    import pandas as pd
    records = []
    for t in study.trials:
        # t is a FrozenTrial
        rec = {
            'number': t.number,
            'value': float(t.value) if t.value is not None else None,
            'state': str(t.state),
            'datetime_start': getattr(t, 'datetime_start', None),
            'datetime_complete': getattr(t, 'datetime_complete', None),
            'params': t.params,
            'user_attrs': t.user_attrs,
        }
        # flatten params into columns (optional)
        for k,v in t.params.items():
            rec[f"param_{k}"] = v
        # optionally include error if present
        if 'error' in t.user_attrs:
            rec['error'] = t.user_attrs.get('error')
        records.append(rec)

    trials_df = pd.DataFrame.from_records(records)
    trials_csv_path = os.path.join(results_dir, f"optuna_{target_col}_trials.csv")
    trials_df.to_csv(trials_csv_path, index=False)
    print(f"[INFO] saved trials to {trials_csv_path}")

    return study

# ------------------- CLI entry --------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, required=True, help="Target column name (e.g., T_1) or index")
    parser.add_argument("--n_trials", type=int, default=200, help="Number of trials")
    parser.add_argument("--n_jobs", type=int, default=1, help="Parallel jobs for optuna")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--results_dir", type=str, default="results")
    args = parser.parse_args()

    df, X, Y, mask = load_or_build_features(data_dir=args.data_dir, results_dir=args.results_dir)
    # determine target index mapping: assume Y columns correspond to df pivoted earlier: we saved df.pkl with columns
    # attempt to get target index by matching name in df.pkl columns
    # load fold info




    folds = build_scaffold_folds(df, results_dir=args.results_dir, n_folds=5)
    # If target given as 'T_1' find its index in saved df
    # For safety, reconstruct target_cols from results_df columns in df
    # Find target cols in df:
    target_cols = [c for c in df.columns if str(c).startswith('T_')]
    if isinstance(args.target, str) and args.target in target_cols:
        target_col = args.target
        target_index = target_cols.index(target_col)
    else:
        # try if numeric index
        try:
            target_index = int(args.target)
            target_col = target_cols[target_index]
        except Exception:
            # fallback first target
            target_index = 0
            target_col = target_cols[0]
            print(f"Warning: target {args.target} not found; defaulting to {target_col}")
    # run tuning
    study = tune_target_with_optuna(target_col=target_col, df=df, X=X, Y=Y, mask=mask, folds=folds, results_dir=args.results_dir, n_trials=args.n_trials, n_jobs=args.n_jobs)


    print("Tuning finished. Best params saved to results/")

import json, os
import pandas as pd

def save_optuna_results(study, results_dir, target_col):
    os.makedirs(results_dir, exist_ok=True)
    # 1) 保存 best trial 参数和 value
    best = study.best_trial
    best_out = {'value': float(best.value) if best.value is not None else None, 'params': best.params}
    with open(os.path.join(results_dir, f"optuna_{target_col}_best.json"), 'w') as f:
        json.dump(best_out, f, indent=2)
    print(f"[INFO] Saved best params to results/optuna_{target_col}_best.json")

    # 2) 保存所有 trials（简洁表格）
    records = []
    for t in study.trials:
        rec = {
            'number': t.number,
            'value': float(t.value) if t.value is not None else None,
            'state': str(t.state),
            'params': t.params,
            'user_attrs': t.user_attrs,
        }
        # flatten params
        for k, v in t.params.items():
            rec[f'param_{k}'] = v
        records.append(rec)
    trials_df = pd.DataFrame.from_records(records)
    trials_df.to_csv(os.path.join(results_dir, f"optuna_{target_col}_trials.csv"), index=False)
    print(f"[INFO] Saved trials CSV to results/optuna_{target_col}_trials.csv")
