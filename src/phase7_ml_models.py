"""
PHASE 7: ML MODELS (XGBoost + Random Forest)
==============================================
Trains and evaluates ML models for 3 target dimensions:
- disclosure_type
- framing
- risk_tier

Variants:
1. Text-Only: TF-IDF + Empath
2. KG-Augmented: TF-IDF + Empath + phi_G

Outputs: Models and JSON results for ablation table.
"""
import os
import json
import pickle
import warnings
import numpy as np
import pandas as pd
from scipy.sparse import load_npz, hstack, csr_matrix
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.feature_selection import SelectKBest, chi2
warnings.filterwarnings('ignore')

os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

print("=" * 65)
print("PHASE 7 — ML MODELS (XGBoost & RandomForest)")
print("=" * 65)

# ──────────────────────────────────────────────────────────────────────
# STEP 1 — Load all features and labels
# ──────────────────────────────────────────────────────────────────────
print("\n[STEP 1] Loading features and labels...")
X_train_tfidf = load_npz('data/features/tfidf_train.npz')
X_val_tfidf   = load_npz('data/features/tfidf_val.npz')
X_test_tfidf  = load_npz('data/features/tfidf_test.npz')

empath_train = np.load('data/features/empath_train.npy')
empath_val   = np.load('data/features/empath_val.npy')
empath_test  = np.load('data/features/empath_test.npy')

phi_g_train = np.load('data/features/phi_g_train.npy')
phi_g_val   = np.load('data/features/phi_g_val.npy')
phi_g_test  = np.load('data/features/phi_g_test.npy')

# Combine features (sparse format)
X_train_text = hstack([X_train_tfidf, csr_matrix(empath_train)])
X_val_text   = hstack([X_val_tfidf,   csr_matrix(empath_val)])
X_test_text  = hstack([X_test_tfidf,  csr_matrix(empath_test)])

X_train_kg = hstack([X_train_text, csr_matrix(phi_g_train)])
X_val_kg   = hstack([X_val_text,   csr_matrix(phi_g_val)])
X_test_kg  = hstack([X_test_text,  csr_matrix(phi_g_test)])

labels = {}
for dim in ['type', 'framing', 'risk']:
    for split in ['train', 'val', 'test']:
        labels[f'y_{split}_{dim}'] = np.load(f'data/features/y_{split}_{dim}.npy')

print(f"X_train_text shape: {X_train_text.shape}")
print(f"X_train_kg shape:   {X_train_kg.shape}")

# ──────────────────────────────────────────────────────────────────────
# STEP 2 — Function for Class Weights
# ──────────────────────────────────────────────────────────────────────
def get_class_weights(y):
    # Filter out NaNs if any, though there shouldn't be
    y = np.array(y, dtype=int)
    classes = np.unique(y)
    weights = compute_class_weight('balanced', classes=classes, y=y)
    return dict(zip(classes, weights))

results = {}

# ──────────────────────────────────────────────────────────────────────
# STEP 3 — Train and evaluate XGBoost
# ──────────────────────────────────────────────────────────────────────
print("\n[STEP 3] Training XGBoost...")
for feat_name, X_tr, X_va, X_te in [
    ('text_only', X_train_text, X_val_text, X_test_text),
    ('kg_augmented', X_train_kg, X_val_kg, X_test_kg)
]:
    results[f'xgboost_{feat_name}'] = {}
    for dim in ['type', 'framing', 'risk']:
        y_tr = labels[f'y_train_{dim}']
        y_va = labels[f'y_val_{dim}']
        y_te = labels[f'y_test_{dim}']
        
        cw = get_class_weights(y_tr)
        sample_weights = np.array([cw[y] for y in y_tr])
        
        model = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric='mlogloss',
            early_stopping_rounds=20,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(
            X_tr, y_tr,
            sample_weight=sample_weights,
            eval_set=[(X_va, y_va)],
            verbose=False
        )
        
        y_pred = model.predict(X_te)
        macro_f1 = f1_score(y_te, y_pred, average='macro', zero_division=0)
        report = classification_report(y_te, y_pred, output_dict=True, zero_division=0)
        
        results[f'xgboost_{feat_name}'][dim] = {
            'macro_f1': macro_f1,
            'best_iteration': int(model.best_iteration) if hasattr(model, 'best_iteration') else 0,
            'report': report
        }
        print(f"XGBoost {feat_name:12s} | {dim:7s}: macro-F1 = {macro_f1:.4f}")
        
        with open(f'models/xgb_{feat_name}_{dim}.pkl', 'wb') as f:
            pickle.dump(model, f)

# ──────────────────────────────────────────────────────────────────────
# STEP 4 — Train and evaluate Random Forest
# ──────────────────────────────────────────────────────────────────────
print("\n[STEP 4] Training Random Forest (with dimensionality reduction to avoid OOM)...")
# Reduce dimensions for RF to top 2000 features using Chi-Square, keeping only positive features (TF-IDF/Empath/centrality are positive)
# Make features strictly positive
X_train_text_abs = X_train_text.copy()
X_train_text_abs.data = np.abs(X_train_text_abs.data)
X_train_kg_abs = X_train_kg.copy()
X_train_kg_abs.data = np.abs(X_train_kg_abs.data)

selector_text = SelectKBest(chi2, k=min(2000, X_train_text.shape[1]))
X_tr_text_red = selector_text.fit_transform(X_train_text_abs, labels['y_train_type'])
X_te_text_red = selector_text.transform(np.abs(X_test_text.data) if hasattr(X_test_text, 'data') else np.abs(X_test_text)) # Need to handle sparse properly
# Easier:
X_te_text_red = selector_text.transform(X_test_text.copy().abs() if hasattr(X_test_text, 'abs') else np.abs(X_test_text))

selector_kg = SelectKBest(chi2, k=min(2000, X_train_kg.shape[1]))
X_tr_kg_red = selector_kg.fit_transform(X_train_kg_abs, labels['y_train_type'])
X_te_kg_red = selector_kg.transform(X_test_kg.copy().abs() if hasattr(X_test_kg, 'abs') else np.abs(X_test_kg))

for feat_name, X_tr_sparse, X_te_sparse in [
    ('text_only', X_tr_text_red, X_te_text_red),
    ('kg_augmented', X_tr_kg_red, X_te_kg_red)
]:
    # Convert to dense for Random Forest
    X_tr_dense = X_tr_sparse.toarray() if hasattr(X_tr_sparse, 'toarray') else X_tr_sparse
    X_te_dense = X_te_sparse.toarray() if hasattr(X_te_sparse, 'toarray') else X_te_sparse
    
    results[f'random_forest_{feat_name}'] = {}
    for dim in ['type', 'framing', 'risk']:
        y_tr = labels[f'y_train_{dim}']
        y_te = labels[f'y_test_{dim}']
        
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_tr_dense, y_tr)
        
        y_pred = model.predict(X_te_dense)
        macro_f1 = f1_score(y_te, y_pred, average='macro', zero_division=0)
        report = classification_report(y_te, y_pred, output_dict=True, zero_division=0)
        
        results[f'random_forest_{feat_name}'][dim] = {
            'macro_f1': macro_f1,
            'report': report
        }
        print(f"Random Forest {feat_name:12s} | {dim:7s}: macro-F1 = {macro_f1:.4f}")
        
        with open(f'models/rf_{feat_name}_{dim}.pkl', 'wb') as f:
            pickle.dump(model, f)

# ──────────────────────────────────────────────────────────────────────
# STEP 5 — Ablation Analysis
# ──────────────────────────────────────────────────────────────────────
print("\n=== ABLATION ANALYSIS (Text-Only vs KG-Augmented) ===")
for model_name in ['xgboost', 'random_forest']:
    for dim in ['type', 'framing', 'risk']:
        text_f1 = results[f'{model_name}_text_only'][dim]['macro_f1']
        kg_f1   = results[f'{model_name}_kg_augmented'][dim]['macro_f1']
        delta   = kg_f1 - text_f1
        signal = "KG HELPS" if delta > 0.01 else "MARGINAL" if delta >= -0.01 else "KG HURTS"
        print(f"{model_name:13s} | {dim:7s}: Δ = {delta:+.4f} ({signal})")

# ──────────────────────────────────────────────────────────────────────
# STEP 6 — Save Results
# ──────────────────────────────────────────────────────────────────────
with open('results/ml_results.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)
    
print("\nPHASE 7 COMPLETE — Models and results saved to ML pipeline directory")
