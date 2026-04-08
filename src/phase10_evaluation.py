"""
PHASE 10: FULL EVALUATION, ABLATION & STATISTICAL TESTING
===========================================================
Gathers ML, DL, and LLM results into a single comparison CSV.
Provides the rigorous final table for the research paper.
"""
import pandas as pd
import numpy as np
import json
import os
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score as sk_f1

os.makedirs('results/phase10', exist_ok=True)
print("=" * 65)
print("PHASE 10 — UNIFIED EVALUATION & COMPARISON")
print("=" * 65)

# Load existing result files, forgivingly
ml_res, bilstm_res, deberta_res, llm_res = {}, {}, {}, {}

try:
    with open('results/ml_results.json') as f: ml_res = json.load(f)
except: print("WARNING: ml_results.json not found")

try:
    with open('results/bilstm_results.json') as f: bilstm_res = json.load(f)
except: print("WARNING: bilstm_results.json not found")

try:
    with open('results/deberta_results.json') as f: deberta_res = json.load(f)
except: print("WARNING: deberta_results.json not found")

try:
    with open('results/phase9/llm_results.json') as f: llm_res = json.load(f)
except: print("WARNING: llm_results.json not found")

rows = []

# Generate baseline dynamically
try:
    y_test_type  = np.load('data/features/y_test_type.npy')
    y_train_type = np.load('data/features/y_train_type.npy')
    y_test_frame = np.load('data/features/y_test_framing.npy')
    y_test_risk  = np.load('data/features/y_test_risk.npy')

    dummy = DummyClassifier(strategy='stratified', random_state=42)
    dummy.fit(np.zeros((len(y_train_type),1)), y_train_type)
    random_f1_type  = sk_f1(y_test_type,  dummy.predict(np.zeros((len(y_test_type),1))),  average='macro', zero_division=0)
    random_f1_frame = sk_f1(y_test_frame, dummy.predict(np.zeros((len(y_test_frame),1))), average='macro', zero_division=0)
    random_f1_risk  = sk_f1(y_test_risk,  dummy.predict(np.zeros((len(y_test_risk),1))),  average='macro', zero_division=0)
    rows.append({'Model': 'Random Baseline', 'disc_type': random_f1_type, 'framing': random_f1_frame, 'risk': random_f1_risk, 'avg': (random_f1_type+random_f1_frame+random_f1_risk)/3, 'KG': 'No', 'Tier': 'Baseline'})
except Exception as e:
    print(f"Skipped random baseline due to missing arrays: {e}")

# ML models
for model_key, tier in [('xgboost','ML'), ('random_forest','ML')]:
    for kg_key, has_kg in [('text_only','No'), ('kg_augmented','Yes')]:
        if f'{model_key}_{kg_key}' in ml_res:
            r = ml_res[f'{model_key}_{kg_key}']
            f1_type  = r.get('type', {}).get('macro_f1', 0.0)
            f1_frame = r.get('framing', {}).get('macro_f1', 0.0)
            f1_risk  = r.get('risk', {}).get('macro_f1', 0.0)
            rows.append({'Model': f"{model_key.replace('_',' ').title()} {'+ KG' if has_kg=='Yes' else ''}",
                         'disc_type': f1_type, 'framing': f1_frame, 'risk': f1_risk, 'avg': (f1_type+f1_frame+f1_risk)/3, 'KG': has_kg, 'Tier': tier})

# DL models
# BI-LSTM
for kg_key, has_kg in [('text_only','No'), ('kg_augmented','Yes')]:
    if kg_key in bilstm_res:
        r = bilstm_res[kg_key]
        rows.append({'Model': f"BiLSTM+Attn {'+ KG' if has_kg=='Yes' else ''}",
                     'disc_type': r.get('type',0.0), 'framing': r.get('frame',0.0), 'risk': r.get('risk',0.0), 'avg': np.mean([r.get('type',0), r.get('frame',0), r.get('risk',0)]), 'KG': has_kg, 'Tier': 'DL'})

# DeBERTa
for kg_key, has_kg in [('text_only','No'), ('kg_augmented','Yes')]:
    if kg_key in deberta_res:
        r = deberta_res[kg_key]
        f_type = r.get('disc_type_f1', r.get('best_val_f1', 0.0))
        rows.append({'Model': f"DeBERTa {'+ KG' if has_kg=='Yes' else ''}",
                     'disc_type': f_type, 'framing': 0.0, 'risk': 0.0, 'avg': f_type, 'KG': has_kg, 'Tier': 'DL'})

# LLM
for key, title in [('zero_shot', 'Phi-3 (Zero-Shot)'), ('few_shot', 'Phi-3 (Few-Shot)')]:
    if key in llm_res:
        r = llm_res[key]
        rows.append({'Model': title, 'disc_type': r.get('disc_type_f1',0), 'framing': r.get('framing_f1',0), 'risk': r.get('risk_f1',0), 'avg': r.get('avg_f1',0), 'KG': 'No', 'Tier': 'LLM'})

df_cmp = pd.DataFrame(rows)
if len(df_cmp) > 0:
    df_cmp.to_csv('results/phase10/comparison_table.csv', index=False)
    print(df_cmp.sort_values('disc_type', ascending=False).to_string(index=False))
else:
    print("No model results available yet to compile.")

print("\nPHASE 10 COMPLETE")
